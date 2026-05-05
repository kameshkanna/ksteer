"""
Per-layer behavioral direction extraction via mean activation difference.

For each contrastive pair (positive_text, negative_text):
    diff_l = mean_pool(residual_pos_l) - mean_pool(residual_neg_l)

The final behavioral vector at layer l is the unit-normalized mean diff across
all pairs. Layer consistency = mean cosine similarity of individual pair diffs
to the aggregate direction — measures how well-defined the behavioral axis is.
"""

import gc
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from ksteer.utils.model_utils import (
    get_hidden_dim,
    get_num_layers,
    iter_transformer_layers,
)

logger = logging.getLogger(__name__)


@dataclass
class BehavioralVector:
    """Per-layer unit behavioral direction extracted from contrastive pairs."""

    behavior: str
    model_name: str
    num_pairs: int
    hidden_dim: int
    num_layers: int
    # (num_layers, hidden_dim) float32 unit vectors
    layer_vectors: np.ndarray
    # Mean cosine similarity of per-pair diffs to the aggregate direction
    layer_consistency: List[float]
    # Raw (unnormalized) mean diff norms before unit normalization
    layer_raw_norms: List[float]

    def get_vector(self, layer_idx: int) -> torch.Tensor:
        """Return the unit vector for a specific layer as a float32 CPU tensor."""
        return torch.from_numpy(self.layer_vectors[layer_idx]).float()

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_dir / "vectors.npz", layer_vectors=self.layer_vectors)
        with open(output_dir / "vectors_meta.json", "w") as f:
            json.dump(
                {
                    "behavior": self.behavior,
                    "model_name": self.model_name,
                    "num_pairs": self.num_pairs,
                    "hidden_dim": self.hidden_dim,
                    "num_layers": self.num_layers,
                    "layer_consistency": self.layer_consistency,
                    "layer_raw_norms": self.layer_raw_norms,
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, output_dir: Path) -> "BehavioralVector":
        with open(output_dir / "vectors_meta.json") as f:
            meta = json.load(f)
        data = np.load(output_dir / "vectors.npz")
        return cls(
            behavior=meta["behavior"],
            model_name=meta["model_name"],
            num_pairs=meta["num_pairs"],
            hidden_dim=meta["hidden_dim"],
            num_layers=meta["num_layers"],
            layer_vectors=data["layer_vectors"],
            layer_consistency=meta["layer_consistency"],
            layer_raw_norms=meta["layer_raw_norms"],
        )


def load_behavior_pairs(jsonl_path: Path) -> List[Tuple[str, str]]:
    """Load (positive, negative) text pairs from a .jsonl behavior file."""
    pairs: List[Tuple[str, str]] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pos = obj.get("positive", "")
            neg = obj.get("negative", "")
            if pos and neg:
                pairs.append((pos, neg))
    if not pairs:
        raise ValueError(f"No valid pairs found in {jsonl_path}")
    return pairs


class ContrastiveExtractor:
    """
    Extracts per-layer behavioral directions from contrastive (positive, negative) pairs.

    For each pair, a single forward pass encodes both texts and records the
    mean-pooled residual stream at every transformer block output. The
    aggregate direction is the mean diff across all pairs, unit-normalized.

    Consistency ∈ [−1, 1]: mean cosine alignment of individual pair diffs
    to the aggregate. Values near 1 indicate a clean, consistent behavioral axis.
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

    def extract(
        self,
        pairs: List[Tuple[str, str]],
        behavior: str,
        max_length: int = 256,
    ) -> BehavioralVector:
        """
        Extract per-layer behavioral direction from contrastive pairs.

        Args:
            pairs: List of (positive_text, negative_text) tuples.
            behavior: Behavior label used in output metadata.
            max_length: Tokenization truncation length.

        Returns:
            BehavioralVector with unit directions and consistency scores.
        """
        if not pairs:
            raise ValueError("pairs must be non-empty")

        # Accumulate per-pair diffs: (num_pairs, num_layers, hidden_dim)
        pair_diffs = torch.zeros(len(pairs), self._num_layers, self._hidden_dim)

        for i, (pos_text, neg_text) in enumerate(
            tqdm(pairs, desc=f"Extracting [{behavior}]", dynamic_ncols=True)
        ):
            pos_mean = self._mean_pool(pos_text, max_length)  # (num_layers, hidden_dim)
            neg_mean = self._mean_pool(neg_text, max_length)  # (num_layers, hidden_dim)
            pair_diffs[i] = pos_mean - neg_mean

        # mean_diff: (num_layers, hidden_dim)
        mean_diff = pair_diffs.mean(dim=0)

        norms = mean_diff.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # (num_layers, 1)
        unit_vecs = mean_diff / norms                                   # (num_layers, hidden_dim)

        # Per-pair unit diffs: (num_pairs, num_layers, hidden_dim)
        pair_norms = pair_diffs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        pair_units = pair_diffs / pair_norms

        # Consistency: (num_pairs, num_layers) → mean over pairs → (num_layers,)
        cos_sims = (pair_units * unit_vecs.unsqueeze(0)).sum(dim=-1)
        layer_consistency: List[float] = cos_sims.mean(dim=0).tolist()
        layer_raw_norms: List[float] = norms.squeeze(-1).tolist()

        gc.collect()

        return BehavioralVector(
            behavior=behavior,
            model_name=self._model_name,
            num_pairs=len(pairs),
            hidden_dim=self._hidden_dim,
            num_layers=self._num_layers,
            layer_vectors=unit_vecs.numpy().astype(np.float32),
            layer_consistency=layer_consistency,
            layer_raw_norms=layer_raw_norms,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _mean_pool(self, text: str, max_length: int) -> torch.Tensor:
        """
        Single forward pass returning the mean-pooled residual stream at every layer.

        Returns:
            Tensor (num_layers, hidden_dim) on CPU in float32.
        """
        self._reset_buffer()
        self._register_hooks()
        try:
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(self._device)
            mask = inputs["attention_mask"]  # (1, seq_len)
            with torch.no_grad():
                self._model(**inputs)
        finally:
            self._remove_hooks()

        result = torch.zeros(self._num_layers, self._hidden_dim)
        for l_idx in range(self._num_layers):
            hs = self._buffer[l_idx][0]          # (1, seq_len, hidden_dim)
            m = mask.bool().to(hs.device)         # (1, seq_len)
            valid = hs[m].float()                 # (num_valid_tokens, hidden_dim)
            result[l_idx] = valid.mean(dim=0).cpu()

        del inputs
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
        return result

    def _reset_buffer(self) -> None:
        self._buffer = {i: [] for i in range(self._num_layers)}

    def _make_hook(self, layer_idx: int) -> Callable:
        def hook(module: nn.Module, input: tuple, output) -> None:
            hs = output[0] if isinstance(output, tuple) else output
            self._buffer[layer_idx].append(hs.detach().cpu())
        return hook

    def _register_hooks(self) -> None:
        for idx, layer in iter_transformer_layers(self._model):
            handle = layer.register_forward_hook(self._make_hook(idx))
            self._hooks.append(handle)

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
