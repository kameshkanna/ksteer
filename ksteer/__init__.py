"""ksteer — Calibrated activation steering for behavioral alignment."""

from ksteer.contrastive import BehavioralVector, ContrastiveExtractor, load_behavior_pairs
from ksteer.profiler import CeilingSweeper, LayerNormProfiler, NormProfile

__version__ = "0.1.0"
__all__ = [
    "LayerNormProfiler",
    "NormProfile",
    "CeilingSweeper",
    "ContrastiveExtractor",
    "BehavioralVector",
    "load_behavior_pairs",
]
