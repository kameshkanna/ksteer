"""
ksteer — Calibrated activation steering for behavioral alignment.

Public API::

    from ksteer import load_model, LayerNormProfiler, IronWallExtractor
    from ksteer import PAIRS, compute_k_max, inject, remove_hooks
"""

from ksteer.inject import compute_k_max, inject, remove_hooks
from ksteer.norm import LayerNormProfiler
from ksteer.pairs import PAIRS
from ksteer.utils import load_model
from ksteer.vectors import IronWallExtractor

__all__ = [
    "load_model",
    "LayerNormProfiler",
    "IronWallExtractor",
    "PAIRS",
    "compute_k_max",
    "inject",
    "remove_hooks",
]
