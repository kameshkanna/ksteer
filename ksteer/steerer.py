"""
Multi-layer activation steering with ramped K injection.

Injects a behavioral direction vector across all layers in the steering window
using a linear magnitude ramp:

    K_ramp = linspace(f_start, f_end, n_layers) × K_optimal

K_optimal = mean K_l over the 40-80% window = mean(mean_norm_l / sqrt(d)).
f_start and f_end are dimensionless fractions of K_optimal.

Each layer receives its own per-layer behavioral unit vector (from Exp 02),
scaled by K_ramp[i]. The per-layer vector captures the behavioral direction
as it manifests at that specific depth, giving the strongest signal where
the representation is sharpest.

Empirical baseline (Qwen2.5-3B-Instruct, anti-safety steering):
    f_start = 0.13, f_end = 0.48 → effective behavioral steering, no gibberish.
    f_end > 0.48 → coherence collapse.

Formula claim: f_end_max ≈ constant across model families.
This is validated by exp03_ramp_validation.py.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer

from ksteer.profiler import NormProfile, _is_coherent
from ksteer.utils.model_utils import get_layer_by_index

logger = logging.getLogger(__name__)


@dataclass
class RampProbeResult:
    f_end: float
    k_end_raw: float       # f_end × K_optimal — actual injection magnitude at last layer
    k_start_raw: float     # f_start × K_optimal
    output_text: str
    is_coherent: bool

    def to_dict(self) -> dict:
        return {
            "f_end": round(self.f_end, 4),
            "k_end_raw": round(self.k_end_raw, 6),
            "k_start_raw": round(self.k_start_raw, 6),
            "is_coherent": self.is_coherent,
            "output_text": self.output_text,
        }


class MultiLayerSteerer:
    """
    Injects per-layer behavioral vectors with a linear K ramp across the steering window.

    Matches the deployment mechanism validated empirically: each layer i in the
    steering window receives its own unit behavioral vector scaled by
    K_ramp[i] = linspace(f_start, f_end, n) × K_optimal.
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
        self._k_optimal = profile.window_k_mean
        logger.info("MultiLayerSteerer: K_optimal=%.4f  (window mean K_l)", self._k_optimal)

    @property
    def k_optimal(self) -> float:
        return self._k_optimal

    def generate_steered(
        self,
        prompt: str,
        behavioral_vectors: Dict[int, torch.Tensor],
        layer_indices: List[int],
        f_start: float,
        f_end: float,
        max_new_tokens: int = 60,
    ) -> str:
        """
        Run one steered generation with K_ramp = linspace(f_start, f_end, n) × K_optimal.
        behavioral_vectors: {layer_idx: unit_vector_or_raw_vector} — normalized inside.
        """
        n = len(layer_indices)
        k_values = np.linspace(f_start * self._k_optimal, f_end * self._k_optimal, n)

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

    def find_f_max(
        self,
        prompt: str,
        behavioral_vectors: Dict[int, torch.Tensor],
        layer_indices: List[int],
        f_start: float = 0.13,
        f_values: Optional[List[float]] = None,
        max_new_tokens: int = 60,
    ) -> Tuple[Optional[float], List[RampProbeResult]]:
        """
        Sweep f_end upward to find the maximum ramp fraction before coherence collapse.

        Returns (f_max, probes):
            f_max — largest f_end with coherent output.
                    None if already incoherent at the smallest f_values entry.
            probes — all RampProbeResult instances, ordered by f_end.

        Assumes monotonicity: once incoherent at f_end, all higher values are too.
        """
        if f_values is None:
            f_values = [round(x, 2) for x in np.arange(0.10, 1.01, 0.05).tolist()]

        probes: List[RampProbeResult] = []
        f_max: Optional[float] = None

        for f_end in sorted(f_values):
            text = self.generate_steered(
                prompt, behavioral_vectors, layer_indices, f_start, f_end, max_new_tokens
            )
            coherent = _is_coherent(text)
            k_end_raw = f_end * self._k_optimal
            k_start_raw = f_start * self._k_optimal

            probes.append(RampProbeResult(
                f_end=f_end,
                k_end_raw=k_end_raw,
                k_start_raw=k_start_raw,
                output_text=text,
                is_coherent=coherent,
            ))
            logger.info(
                "  f_end=%.2f  K_raw=%.4f  coherent=%-5s | %r",
                f_end, k_end_raw, coherent, text[:80],
            )

            if coherent:
                f_max = f_end
            else:
                logger.info("  → collapse at f_end=%.2f  K_raw=%.4f", f_end, k_end_raw)
                break

        if f_max is not None:
            logger.info(
                "  f_max=%.2f  K_max_raw=%.4f  (K_optimal=%.4f)",
                f_max, f_max * self._k_optimal, self._k_optimal,
            )
        else:
            logger.info("  Incoherent at all tested f_end values.")

        return f_max, probes
