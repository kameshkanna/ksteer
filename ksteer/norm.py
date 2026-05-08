"""
Per-layer norm profiling.

K_l = mean_norm_l / sqrt(d)

K_l is the per-dimension RMS of the residual stream at layer l.
It is the coherence ceiling — injecting beyond K_l at layer l
overwhelms the ambient signal and produces incoherent output.
"""

import gc
import logging
import math
from typing import List

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from ksteer.utils import hidden_dim, iter_layers, num_layers

logger = logging.getLogger(__name__)

_PROFILE_TEXTS = [
    "The capital of France is Paris.",
    "Water boils at 100 degrees Celsius at sea level.",
    "The speed of light is approximately 299,792 kilometres per second.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
    "The human genome contains approximately 3 billion base pairs.",
    "Newton's first law states that an object at rest stays at rest.",
    "The Earth orbits the Sun once every 365.25 days.",
    "DNA is a double helix made of nucleotide base pairs.",
    "The mitochondria is the powerhouse of the cell.",
    "Gravity accelerates objects at 9.8 metres per second squared near Earth.",
]


class LayerNormProfiler:
    """
    Measures per-layer residual stream norms and derives K_l.

    Hooks into every transformer block output, records L2 norms of all
    valid (non-padding) token positions, computes mean norm per layer,
    then divides by sqrt(d) to get the per-dimension scale K_l.
    """

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device = next(model.parameters()).device
        self._d = hidden_dim(model)
        self._n = num_layers(model)
        self._sqrt_d = math.sqrt(self._d)

    def profile(
        self,
        texts: List[str] | None = None,
        batch_size: int = 4,
        max_length: int = 256,
    ) -> List[float]:
        """
        Run forward passes on texts and return K_l for every layer.

        Args:
            texts: Sentences to profile on. Defaults to neutral factual set.
            batch_size: Tokenisation batch size.
            max_length: Truncation length.

        Returns:
            List of K_l floats, length = num_layers.
        """
        if texts is None:
            texts = _PROFILE_TEXTS

        # buffer[layer_idx] accumulates valid-token norm tensors
        buf: dict[int, list[torch.Tensor]] = {i: [] for i in range(self._n)}
        cur_mask: list[torch.Tensor] = []

        def _make_hook(idx: int):
            def hook(module: nn.Module, inp: tuple, out) -> None:
                hs = out[0] if isinstance(out, tuple) else out
                norms = hs.float().norm(dim=-1)          # (B, T)
                mask = cur_mask[0].bool().to(norms.device)
                buf[idx].append(norms[mask].detach().cpu())
            return hook

        handles = [
            layer.register_forward_hook(_make_hook(i))
            for i, layer in iter_layers(self._model)
        ]

        batches = [texts[i: i + batch_size] for i in range(0, len(texts), batch_size)]
        try:
            for batch in tqdm(batches, desc="Norm profiling", dynamic_ncols=True):
                enc = self._tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                ).to(self._device)
                cur_mask[:] = [enc["attention_mask"]]
                with torch.no_grad():
                    self._model(**enc)
                cur_mask.clear()
                del enc
                if self._device.type == "cuda":
                    torch.cuda.empty_cache()
        finally:
            for h in handles:
                h.remove()

        k_l = [
            float(torch.cat(buf[i]).mean().item()) / self._sqrt_d
            for i in range(self._n)
        ]
        gc.collect()
        logger.info("K_l  min=%.3f  max=%.3f  (layers=%d  d=%d)",
                    min(k_l), max(k_l), self._n, self._d)
        return k_l
