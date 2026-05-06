"""
Per-layer residual stream norm profiling and K_l ceiling sweep.

K_l = mean_norm_l / sqrt(d) is the maximum coherent steering magnitude at layer l.
Injecting a vector beyond this scale overwhelms the ambient residual stream signal,
pushing downstream layers out of distribution and producing incoherent output.
"""

import gc
import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from ksteer.utils.model_utils import (
    get_hidden_dim,
    get_layer_by_index,
    get_num_layers,
    iter_transformer_layers,
)

logger = logging.getLogger(__name__)


@dataclass
class NormProfile:
    """Per-layer norm statistics and derived K_l values for a model."""

    model_name: str
    model_family: str
    hidden_dim: int
    num_layers: int
    layer_mean_norms: List[float]
    layer_std_norms: List[float]
    k_values: List[float]           # K_l = mean_norm_l / sqrt(hidden_dim)
    num_tokens_sampled: int

    @property
    def steering_window(self) -> tuple[int, int]:
        """Layer range [40%, 80%) where behavioral concepts reside."""
        return int(0.4 * self.num_layers), int(0.8 * self.num_layers)

    @property
    def window_k_range(self) -> tuple[float, float]:
        start, end = self.steering_window
        window = self.k_values[start:end]
        return min(window), max(window)

    def to_dict(self) -> dict:
        start, end = self.steering_window
        return {
            "model_name": self.model_name,
            "model_family": self.model_family,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "layer_mean_norms": self.layer_mean_norms,
            "layer_std_norms": self.layer_std_norms,
            "k_values": self.k_values,
            "num_tokens_sampled": self.num_tokens_sampled,
            "steering_window": [start, end],
            "window_k_min": self.window_k_range[0],
            "window_k_max": self.window_k_range[1],
        }


@dataclass
class CeilingProbeResult:
    layer_idx: int
    k_l: float
    alpha: float
    injected_norm: float        # actual ||v_injected|| = alpha * K_l * sqrt(d)
    ambient_norm: float         # mean_norm_l = K_l * sqrt(d)
    output_text: str
    is_coherent: bool

    def to_dict(self) -> dict:
        return {
            "layer_idx": self.layer_idx,
            "k_l": self.k_l,
            "alpha": self.alpha,
            "injected_norm": self.injected_norm,
            "ambient_norm": self.ambient_norm,
            "is_coherent": self.is_coherent,
            "output_text": self.output_text,
        }


class LayerNormProfiler:
    """
    Hooks into every transformer block output and records residual stream norms.

    Measurement point: output of each full transformer block (post-residual-add),
    which is the actual tensor that flows to the next layer. This is
    architecture-agnostic — Gemma2's post-norm scaling is reflected in the
    residual stream norm directly, so K_l will be correctly higher for Gemma2.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        model_name: str,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._model_name = model_name
        self._device = next(model.parameters()).device
        self._hidden_dim = get_hidden_dim(model)
        self._num_layers = get_num_layers(model)
        self._hooks: list = []
        self._buffer: Dict[int, List[torch.Tensor]] = {}
        self._attention_mask: Optional[torch.Tensor] = None  # set per batch

    def profile(
        self,
        texts: List[str],
        batch_size: int = 4,
        max_length: int = 512,
    ) -> NormProfile:
        """Run forward passes on texts and compute per-layer norm statistics."""
        self._reset_buffer()
        self._register_hooks()

        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
        try:
            for batch in tqdm(batches, desc=f"Profiling {self._model_name}", dynamic_ncols=True):
                self._forward_batch(batch, max_length)
        finally:
            self._remove_hooks()

        profile = self._build_profile()
        gc.collect()
        return profile

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _reset_buffer(self) -> None:
        self._buffer = {i: [] for i in range(self._num_layers)}

    def _make_hook(self, layer_idx: int) -> Callable:
        def hook(module: nn.Module, input: tuple, output) -> None:
            hs = output[0] if isinstance(output, tuple) else output
            # hs: (batch, seq_len, hidden_dim)
            norms = hs.float().norm(dim=-1)  # (batch, seq_len)
            # Exclude padding positions — they have artificially low norms and
            # skew the mean, causing K_l to be underestimated (confirmed on Qwen2.5).
            if self._attention_mask is not None:
                mask = self._attention_mask.bool().to(norms.device)
                valid_norms = norms[mask]
            else:
                valid_norms = norms.reshape(-1)
            self._buffer[layer_idx].append(valid_norms.detach().cpu())
        return hook

    def _register_hooks(self) -> None:
        for idx, layer in iter_transformer_layers(self._model):
            handle = layer.register_forward_hook(self._make_hook(idx))
            self._hooks.append(handle)

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _forward_batch(self, texts: List[str], max_length: int) -> None:
        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(self._device)
        self._attention_mask = inputs["attention_mask"]
        with torch.no_grad():
            self._model(**inputs)
        self._attention_mask = None
        del inputs
        if self._device.type == "cuda":
            torch.cuda.empty_cache()

    def _build_profile(self) -> NormProfile:
        sqrt_d = math.sqrt(self._hidden_dim)
        mean_norms, std_norms, k_values = [], [], []

        for i in range(self._num_layers):
            all_norms = torch.cat(self._buffer[i])
            mean_norms.append(all_norms.mean().item())
            std_norms.append(all_norms.std().item())
            k_values.append(mean_norms[-1] / sqrt_d)

        num_tokens = sum(t.numel() for t in self._buffer[0]) if self._buffer[0] else 0

        return NormProfile(
            model_name=self._model_name,
            model_family=self._model.config.model_type,
            hidden_dim=self._hidden_dim,
            num_layers=self._num_layers,
            layer_mean_norms=mean_norms,
            layer_std_norms=std_norms,
            k_values=k_values,
            num_tokens_sampled=num_tokens,
        )


class CeilingSweeper:
    """
    Finds the K_l coherence ceiling using bisection search.

    At alpha=1 the injected vector has the same total norm as the ambient
    residual stream (||v_injected|| = mean_norm_l). Bisection locates the
    alpha threshold beyond which generation becomes incoherent in O(log n)
    inferences rather than a linear sweep — typically 6-8 steps vs 10-12.

    Monotonicity assumption: coherent below the ceiling, incoherent above.
    Verified empirically across Llama, Gemma2, and Qwen2.5 families.
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

    def find_ceiling(
        self,
        prompt: str,
        steering_vector: torch.Tensor,
        layer_idx: int,
        lo: float = 0.05,
        hi: float = 3.0,
        tolerance: float = 0.05,
        max_steps: int = 8,
        max_new_tokens: int = 60,
    ) -> Tuple[Optional[float], List[CeilingProbeResult]]:
        """
        Locate the coherence ceiling via bisection on alpha × K_l.

        Returns (ceiling_alpha, probes):
            ceiling_alpha — smallest alpha at which output is incoherent,
                            or None if coherent up to `hi`.
            probes        — all CeilingProbeResult instances generated during search.

        Args:
            lo:        Lower bound alpha (should be coherent; typically 0.05).
            hi:        Upper bound alpha (if coherent here, returns None).
            tolerance: Stop bisecting when hi − lo < tolerance.
            max_steps: Hard cap on bisection iterations.
        """
        k_l = self._profile.k_values[layer_idx]
        sqrt_d = math.sqrt(self._profile.hidden_dim)
        ambient_norm = k_l * sqrt_d

        v_unit = (steering_vector / steering_vector.norm()).to(self._device, dtype=self._dtype)
        probes: List[CeilingProbeResult] = []

        def probe(alpha: float) -> bool:
            v_scaled = v_unit * (alpha * ambient_norm)
            text = self._generate_steered(prompt, layer_idx, v_scaled, max_new_tokens)
            coherent = _is_coherent(text)
            probes.append(CeilingProbeResult(
                layer_idx=layer_idx, k_l=k_l, alpha=alpha,
                injected_norm=alpha * ambient_norm, ambient_norm=ambient_norm,
                output_text=text, is_coherent=coherent,
            ))
            logger.info("  L%d  alpha=%.3f | coherent=%-5s | %r",
                        layer_idx, alpha, coherent, text[:80])
            return coherent

        # Check upper bound — if coherent at hi, ceiling is above search range
        if probe(hi):
            logger.info("  L%d  coherent at hi=%.2f — ceiling > %.2f", layer_idx, hi, hi)
            return None, probes

        # Check lower bound — if incoherent at lo, ceiling is below lo (unusual)
        if not probe(lo):
            logger.info("  L%d  incoherent at lo=%.2f — ceiling ≤ %.2f", layer_idx, lo, lo)
            return lo, probes

        # Bisect: invariant is coherent(lo) = True, coherent(hi) = False
        for _ in range(max_steps):
            if hi - lo < tolerance:
                break
            mid = (lo + hi) / 2.0
            if probe(mid):
                lo = mid
            else:
                hi = mid

        logger.info("  L%d  ceiling = %.3f × K_l  (±%.3f)", layer_idx, hi, tolerance)
        return hi, probes

    def find_ceiling_multiple_layers(
        self,
        prompt: str,
        steering_vector: torch.Tensor,
        layer_indices: List[int],
        lo: float = 0.05,
        hi: float = 3.0,
        tolerance: float = 0.05,
        max_steps: int = 8,
        max_new_tokens: int = 60,
    ) -> Dict[int, Tuple[Optional[float], List[CeilingProbeResult]]]:
        """Run bisection ceiling search across multiple layers."""
        return {
            idx: self.find_ceiling(prompt, steering_vector, idx,
                                   lo, hi, tolerance, max_steps, max_new_tokens)
            for idx in layer_indices
        }

    def sweep(
        self,
        prompt: str,
        steering_vector: torch.Tensor,
        layer_idx: int,
        alphas: Optional[List[float]] = None,
        max_new_tokens: int = 60,
    ) -> List[CeilingProbeResult]:
        """
        Linear alpha sweep — kept for backward compatibility and explicit grid evaluation.
        Prefer find_ceiling() for efficiency.
        """
        if alphas is None:
            alphas = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

        k_l = self._profile.k_values[layer_idx]
        sqrt_d = math.sqrt(self._profile.hidden_dim)
        ambient_norm = k_l * sqrt_d
        v_unit = (steering_vector / steering_vector.norm()).to(self._device, dtype=self._dtype)

        results: List[CeilingProbeResult] = []
        for alpha in tqdm(alphas, desc=f"Sweep L{layer_idx}", dynamic_ncols=True):
            v_scaled = v_unit * (alpha * ambient_norm)
            text = self._generate_steered(prompt, layer_idx, v_scaled, max_new_tokens)
            coherent = _is_coherent(text)
            results.append(CeilingProbeResult(
                layer_idx=layer_idx, k_l=k_l, alpha=alpha,
                injected_norm=alpha * ambient_norm, ambient_norm=ambient_norm,
                output_text=text, is_coherent=coherent,
            ))
            logger.info("  alpha=%.2f | coherent=%-5s | %r", alpha, coherent, text[:80])
        return results

    def sweep_multiple_layers(
        self,
        prompt: str,
        steering_vector: torch.Tensor,
        layer_indices: List[int],
        alphas: Optional[List[float]] = None,
        max_new_tokens: int = 60,
    ) -> Dict[int, List[CeilingProbeResult]]:
        """Linear sweep across multiple layers (backward compat). Prefer find_ceiling_multiple_layers."""
        return {
            idx: self.sweep(prompt, steering_vector, idx, alphas, max_new_tokens)
            for idx in layer_indices
        }

    # ------------------------------------------------------------------

    def _generate_steered(
        self,
        prompt: str,
        layer_idx: int,
        v_scaled: torch.Tensor,
        max_new_tokens: int,
    ) -> str:
        target_layer = get_layer_by_index(self._model, layer_idx)

        def hook(module: nn.Module, input: tuple, output) -> tuple:
            hs = output[0] if isinstance(output, tuple) else output
            hs = hs + v_scaled.unsqueeze(0).unsqueeze(0)
            return (hs,) + output[1:] if isinstance(output, tuple) else hs

        handle = target_layer.register_forward_hook(hook)
        try:
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
            with torch.no_grad():
                ids = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            new_ids = ids[0, inputs["input_ids"].shape[1] :]
            return self._tokenizer.decode(new_ids, skip_special_tokens=True)
        finally:
            handle.remove()
            if self._device.type == "cuda":
                torch.cuda.empty_cache()


def _is_coherent(text: str, rep_threshold: float = 0.55, nonascii_threshold: float = 0.25) -> bool:
    """
    Heuristic coherence check. Flags repetition loops and non-ASCII dominance.
    Not a substitute for human evaluation — used only to auto-label sweep outputs.
    """
    text = text.strip()
    if not text:
        return False
    words = text.split()
    if len(words) < 4:
        return False
    top_freq = Counter(words).most_common(1)[0][1] / len(words)
    if top_freq > rep_threshold:
        return False
    nonascii_ratio = sum(1 for c in text if ord(c) > 127) / len(text)
    if nonascii_ratio > nonascii_threshold:
        return False
    return True
