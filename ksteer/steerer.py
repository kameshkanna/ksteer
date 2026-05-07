"""
Multi-layer activation steering with shaped K ramps.

Injection at layer i:
    K_i = f_scale × K_optimal × shape_weights[i]

K_optimal is the residual stream scale, derived one of two ways:
    - "mid":    mean_norm at the single middle layer / sqrt(d)  [zip approach]
    - "window": mean K_l over the 40-80% steering window        [ksteer approach]

shape_weights[i] in [0, 1], normalised so peak = 1. Five shapes:
    linear      — linspace(f_start_frac, 1.0, n), front-to-back ramp
    cosine      — smooth S-curve, same endpoints as linear
    bell        — sin(πt), peaks at the centre of the window
    exponential — slow start, fast finish
    constant    — uniform injection at all layers

f_scale is swept externally (find_f_scale_max) to find the maximum coherent scale.
The formula prediction: f_scale_max ≈ 0.48 for all shapes and models.
"""

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer

from ksteer.profiler import NormProfile, _is_coherent
from ksteer.utils.model_utils import get_layer_by_index

logger = logging.getLogger(__name__)


class RampShape(str, Enum):
    LINEAR      = "linear"
    COSINE      = "cosine"
    BELL        = "bell"
    EXPONENTIAL = "exponential"
    CONSTANT    = "constant"

    @classmethod
    def all(cls) -> List["RampShape"]:
        return list(cls)


def make_ramp(shape: RampShape, n: int, f_start_frac: float = 0.27) -> np.ndarray:
    """
    Return shape weights in [0, 1] with peak normalised to 1.

    f_start_frac: ratio of start to peak value, used by linear and cosine
                  (default 0.27 ≈ 0.13 / 0.48, the empirical baseline ratio).
    """
    t = np.linspace(0.0, 1.0, n)

    if shape == RampShape.LINEAR:
        weights = f_start_frac + (1.0 - f_start_frac) * t

    elif shape == RampShape.COSINE:
        # S-curve: same start/end as linear but smooth acceleration
        weights = f_start_frac + (1.0 - f_start_frac) * (1.0 - np.cos(math.pi * t)) / 2.0

    elif shape == RampShape.BELL:
        # Symmetric peak at window midpoint, tapers to ~0 at edges
        weights = np.sin(math.pi * t)
        weights = weights / weights.max()

    elif shape == RampShape.EXPONENTIAL:
        # Slow start, accelerates toward end
        raw = np.exp(3.0 * t) - 1.0
        weights = raw / raw.max()

    elif shape == RampShape.CONSTANT:
        weights = np.ones(n)

    else:
        raise ValueError(f"Unknown ramp shape: {shape!r}")

    return weights.astype(np.float32)


@dataclass
class RampProbeResult:
    shape: str
    f_scale: float
    k_optimal: float
    k_optimal_source: str    # "mid" or "window"
    output_text: str
    is_coherent: bool

    def to_dict(self) -> dict:
        return {
            "shape": self.shape,
            "f_scale": round(self.f_scale, 4),
            "k_optimal": round(self.k_optimal, 4),
            "k_optimal_source": self.k_optimal_source,
            "k_peak_raw": round(self.f_scale * self.k_optimal, 6),
            "is_coherent": self.is_coherent,
            "output_text": self.output_text,
        }


class MultiLayerSteerer:
    """
    Injects per-layer behavioral vectors across the steering window using a
    shaped K ramp. Supports two K_optimal definitions and five ramp shapes.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        profile: NormProfile,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._profile = profile
        self._device = next(model.parameters()).device
        self._dtype = next(model.parameters()).dtype

        # K_optimal — two definitions for comparison
        mid = profile.num_layers // 2
        self._k_optimal_mid = profile.k_values[mid]

        start, end = profile.steering_window
        self._k_optimal_window = float(np.mean(profile.k_values[start:end]))

        logger.info(
            "MultiLayerSteerer  K_optimal_mid=%.4f  K_optimal_window=%.4f",
            self._k_optimal_mid, self._k_optimal_window,
        )

    def k_optimal(self, source: str = "window") -> float:
        """source: 'mid' (zip approach) or 'window' (ksteer approach)."""
        if source == "mid":
            return self._k_optimal_mid
        return self._k_optimal_window

    def generate_steered(
        self,
        prompt: str,
        behavioral_vectors: Dict[int, torch.Tensor],
        layer_indices: List[int],
        shape: RampShape,
        f_scale: float,
        k_optimal_source: str = "window",
        f_start_frac: float = 0.27,
        max_new_tokens: int = 60,
    ) -> str:
        """
        Run one steered generation.
        K_i = f_scale × K_optimal × shape_weights[i]
        """
        k_opt = self.k_optimal(k_optimal_source)
        weights = make_ramp(shape, len(layer_indices), f_start_frac)
        k_values = f_scale * k_opt * weights

        handles = []
        for i, layer_idx in enumerate(layer_indices):
            v = behavioral_vectors.get(layer_idx)
            if v is None:
                continue
            norm = v.norm()
            if norm == 0:
                continue
            v_unit = (v / norm).to(self._device, dtype=self._dtype)
            v_scaled = v_unit * float(k_values[i])

            def _make_hook(vs: torch.Tensor):
                def hook(module: nn.Module, input: tuple, output) -> tuple:
                    hs = output[0] if isinstance(output, tuple) else output
                    hs = hs + vs.unsqueeze(0).unsqueeze(0)
                    return (hs,) + output[1:] if isinstance(output, tuple) else hs
                return hook

            layer = get_layer_by_index(self._model, layer_idx)
            handles.append(layer.register_forward_hook(_make_hook(v_scaled)))

        try:
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
            with torch.no_grad():
                ids = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            new_ids = ids[0, inputs["input_ids"].shape[1]:]
            return self._tokenizer.decode(new_ids, skip_special_tokens=True)
        finally:
            for h in handles:
                h.remove()
            if self._device.type == "cuda":
                torch.cuda.empty_cache()

    def find_f_scale_max(
        self,
        prompt: str,
        behavioral_vectors: Dict[int, torch.Tensor],
        layer_indices: List[int],
        shape: RampShape,
        f_scale_values: List[float],
        k_optimal_source: str = "window",
        f_start_frac: float = 0.27,
        max_new_tokens: int = 60,
    ) -> Tuple[Optional[float], List[RampProbeResult]]:
        """
        Sweep f_scale upward to find the maximum coherent scale for a given shape.

        Returns (f_scale_max, probes).
        Stops at the first incoherent probe (monotonicity assumption).
        """
        k_opt = self.k_optimal(k_optimal_source)
        probes: List[RampProbeResult] = []
        f_scale_max: Optional[float] = None

        for f_scale in sorted(f_scale_values):
            text = self.generate_steered(
                prompt, behavioral_vectors, layer_indices,
                shape, f_scale, k_optimal_source, f_start_frac, max_new_tokens,
            )
            coherent = _is_coherent(text)
            probes.append(RampProbeResult(
                shape=shape.value,
                f_scale=f_scale,
                k_optimal=k_opt,
                k_optimal_source=k_optimal_source,
                output_text=text,
                is_coherent=coherent,
            ))
            logger.info(
                "  [%s|%s]  f=%.2f  K_peak=%.4f  coherent=%-5s | %r",
                shape.value, k_optimal_source, f_scale,
                f_scale * k_opt, coherent, text[:80],
            )
            if coherent:
                f_scale_max = f_scale
            else:
                logger.info("  → collapse  f_scale_max=%.2f", f_scale_max or 0.0)
                break

        return f_scale_max, probes
