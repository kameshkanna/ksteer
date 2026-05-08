"""
Activation injection with whole-number K interface.

Formula:
    K_inject_l = K × (S_l / S_max)

    K      — user-specified integer (1 … K_max)
    S_l    — behavioral contrast norm at layer l (from IronWallExtractor)
    S_max  — peak S_l across target layers (normalises the shape)

The peak layer always receives exactly K. All other layers scale down
proportionally via their S_l. This distributes injection in proportion
to how strongly each layer differentiates the behavioral poles.

Coherence ceiling:
    K_max = floor( min_l( K_l × S_max / S_l ) )

    K_l   — per-dimension residual stream scale at layer l (from LayerNormProfiler)

Injecting at K > K_max would exceed K_l at the tightest layer and
risk incoherent output. Users stay within [1, K_max].
"""

import gc
import logging
import math
from typing import Dict, List

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from ksteer.utils import get_layer

logger = logging.getLogger(__name__)


def compute_k_max(
    k_l: List[float],
    s_l: List[float],
) -> int:
    """
    Derive the maximum safe K for a given model from its norm profile.

    Args:
        k_l: K_l values for the target layers (from LayerNormProfiler).
        s_l: S_l values for the target layers (from IronWallExtractor).

    Returns:
        K_max as an integer — the largest whole-number K that keeps
        K_inject_l ≤ K_l at every target layer.
    """
    s_max = max(s_l)
    # At each layer: K × (S_l / S_max) ≤ K_l  →  K ≤ K_l × S_max / S_l
    ceilings = [kl * s_max / sl for kl, sl in zip(k_l, s_l) if sl > 1e-8]
    if not ceilings:
        raise ValueError("All S_l values are near zero — extraction may have failed.")
    k_max = math.floor(min(ceilings))
    logger.info("K_max = %d  (S_max=%.4f  tightest ceiling=%.4f)",
                k_max, s_max, min(ceilings))
    return k_max


def inject(
    model: PreTrainedModel,
    vectors: Dict[int, Dict],
    target_indices: List[int],
    K: int,
    anti: bool = False,
) -> List:
    """
    Inject Iron Wall vectors into the model at the specified K level.

    K_inject_l = K × (S_l / S_max), broadcast over all token positions.
    Equivalent to the baked resid_bias approach in the zip.

    Args:
        model: Model to inject into.
        vectors: {layer_idx: {"v": tensor, "S": float}} from IronWallExtractor.
        target_indices: Ordered list of layer indices.
        K: Injection strength (integer, 1 … K_max).
        anti: If True, negate — push away from the refusal direction.

    Returns:
        List of hook handles. Pass to remove_hooks() when done.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    sign = -1.0 if anti else 1.0

    s_values = [vectors[i]["S"] for i in target_indices if i in vectors]
    s_max = max(s_values)

    handles = []
    for layer_idx in target_indices:
        entry = vectors.get(layer_idx)
        if entry is None or entry["S"] < 1e-8:
            continue

        k_inject = sign * K * (entry["S"] / s_max)
        bias = (entry["v"] * k_inject).to(device=device, dtype=dtype)

        def _hook(module: nn.Module, inp: tuple, out, b: torch.Tensor = bias):
            hs = out[0] if isinstance(out, tuple) else out
            hs = hs + b.view(1, 1, -1)
            return (hs,) + out[1:] if isinstance(out, tuple) else hs

        handles.append(get_layer(model, layer_idx).register_forward_hook(_hook))

    logger.debug("Injected K=%d  sign=%+.0f  layers=%d", K, sign, len(handles))
    return handles


def remove_hooks(handles: List) -> None:
    """Remove all injection hooks and free resources."""
    for h in handles:
        h.remove()
    handles.clear()
    gc.collect()
