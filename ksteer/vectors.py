"""
Per-layer behavioral vector extraction from contrastive pairs.

For each layer l in the target window:
    mean_diff_l = mean( h_pos[:, -1, :] - h_neg[:, -1, :] )  over all pairs
    S_l         = ||mean_diff_l||          ← behavioral contrast norm
    v_l         = mean_diff_l / S_l        ← unit direction

S_l is the natural injection scale at layer l for this model.
It encodes how strongly the model differentiates the two behavioral poles.
Instruct-tuned models have larger S_l than base models on the same pairs.
"""

import gc
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from ksteer.utils import get_layer, num_layers

logger = logging.getLogger(__name__)


class IronWallExtractor:
    """
    Extract per-layer (v_l, S_l) from Dictator Pairs using last-token activations.

    Last token is used because it has attended to the full sequence and carries
    the model's compressed decision state for that position.
    """

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device = next(model.parameters()).device

    def extract(
        self,
        pairs: List[Tuple[str, str]],
        target_indices: List[int],
    ) -> Dict[int, Dict]:
        """
        Extract unit vector and behavioral scale for each target layer.

        Args:
            pairs: List of (positive, negative) text pairs.
            target_indices: Layer indices to extract from.

        Returns:
            {layer_idx: {"v": unit_tensor (float32, CPU), "S": float}}
        """
        logger.info("Extracting vectors: %d pairs, %d layers",
                    len(pairs), len(target_indices))

        result: Dict[int, Dict] = {}
        for idx in tqdm(target_indices, desc="Extracting", dynamic_ncols=True):
            v, S = self._extract_layer(pairs, idx)
            result[idx] = {"v": v, "S": S}
            logger.debug("  layer %d  S_l=%.4f", idx, S)

        s_values = [result[i]["S"] for i in target_indices]
        logger.info("S_l  min=%.4f  max=%.4f  mean=%.4f",
                    min(s_values), max(s_values),
                    sum(s_values) / len(s_values))
        return result

    # ------------------------------------------------------------------

    def _extract_layer(
        self,
        pairs: List[Tuple[str, str]],
        layer_idx: int,
    ) -> Tuple[torch.Tensor, float]:
        layer = get_layer(self._model, layer_idx)
        buf: List[torch.Tensor] = []

        def _hook(module: nn.Module, inp: tuple, out) -> None:
            hs = out[0] if isinstance(out, tuple) else out
            buf.append(hs[:, -1, :].detach().cpu())

        handle = layer.register_forward_hook(_hook)
        diffs: List[torch.Tensor] = []

        try:
            for pos_text, neg_text in pairs:
                diffs.append(
                    self._forward(pos_text, buf) - self._forward(neg_text, buf)
                )
        finally:
            handle.remove()

        mean_diff = torch.stack(diffs).mean(dim=0).float()   # (d,)
        S = float(mean_diff.norm().item())
        if S < 1e-8:
            logger.warning("Layer %d: near-zero S_l (%.2e) — returning zero vector", layer_idx, S)
            return torch.zeros_like(mean_diff), 0.0
        return (mean_diff / S), S

    def _forward(self, text: str, buf: List[torch.Tensor]) -> torch.Tensor:
        buf.clear()
        enc = self._tokenizer(text, return_tensors="pt").to(self._device)
        with torch.no_grad():
            self._model(**enc)
        act = buf[0].squeeze(0).float()
        buf.clear()
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
        return act
