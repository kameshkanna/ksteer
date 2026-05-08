"""
Experiment 01 — Norm profile and vector extraction.

Runs LayerNormProfiler and IronWallExtractor on a model, computes K_max,
and saves a profile checkpoint that exp02 loads for evaluation.

Saved checkpoint keys:
    model_id       str
    num_layers     int
    hidden_dim     int
    k_l            List[float]   — K_l for every layer
    target_indices List[int]     — layers in [start_frac, end_frac] window
    vectors        Dict[int, {"v": Tensor, "S": float}]
    k_max          int
    s_max          float         — peak S_l across target layers

Usage::

    python experiments/exp01_profile.py \\
        --model-id Qwen/Qwen2.5-3B-Instruct \\
        --output-dir results/exp01

    # Custom layer window (default: 40 %–90 % of depth)
    python experiments/exp01_profile.py \\
        --model-id meta-llama/Llama-3.1-8B-Instruct \\
        --start-frac 0.35 --end-frac 0.90 \\
        --output-dir results/exp01
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ksteer import (
    IronWallExtractor,
    LayerNormProfiler,
    PAIRS,
    compute_k_max,
    load_model,
)
from ksteer.utils import hidden_dim, num_layers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ksteer exp01: norm profile + vector extraction")
    p.add_argument("--model-id", required=True, help="HuggingFace model ID or local path")
    p.add_argument("--device", default=None, help="Force device (cpu / cuda / cuda:N)")
    p.add_argument(
        "--start-frac", type=float, default=0.40,
        help="Layer window start as fraction of total depth (default: 0.40)",
    )
    p.add_argument(
        "--end-frac", type=float, default=0.90,
        help="Layer window end as fraction of total depth (default: 0.90)",
    )
    p.add_argument(
        "--output-dir", default="results/exp01",
        help="Directory to write the profile checkpoint (default: results/exp01)",
    )
    return p.parse_args()


def _target_indices(n: int, start_frac: float, end_frac: float) -> list[int]:
    lo = max(0, int(n * start_frac))
    hi = min(n - 1, int(n * end_frac))
    return list(range(lo, hi + 1))


def _model_slug(model_id: str) -> str:
    return model_id.replace("/", "_").replace("\\", "_")


def main() -> None:
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ──────────────────────────────────────────────────────────
    logger.info("Loading model: %s", args.model_id)
    model, tokenizer = load_model(args.model_id, device=args.device)

    n = num_layers(model)
    d = hidden_dim(model)
    logger.info("Depth=%d  d=%d", n, d)

    target = _target_indices(n, args.start_frac, args.end_frac)
    logger.info(
        "Target window: layers %d–%d  (%d layers, %.0f%%–%.0f%% depth)",
        target[0], target[-1], len(target),
        args.start_frac * 100, args.end_frac * 100,
    )

    # ── Norm profile → K_l ─────────────────────────────────────────────────
    logger.info("Running norm profile…")
    profiler = LayerNormProfiler(model, tokenizer)
    k_l = profiler.profile()

    # ── Vector extraction → v_l, S_l ───────────────────────────────────────
    logger.info("Extracting Iron Wall vectors…")
    extractor = IronWallExtractor(model, tokenizer)
    vectors = extractor.extract(PAIRS, target)

    # ── Compute K_max ───────────────────────────────────────────────────────
    k_l_target = [k_l[i] for i in target]
    s_l_target = [vectors[i]["S"] for i in target]
    k_max = compute_k_max(k_l_target, s_l_target)
    s_max = max(s_l_target)

    logger.info("K_max = %d", k_max)

    # ── Per-layer summary table ─────────────────────────────────────────────
    header = f"{'Layer':>6}  {'K_l':>8}  {'S_l':>8}  {'K_inject@1':>12}"
    logger.info("\n%s\n%s", header, "-" * len(header))
    for i in target:
        k_inject_1 = 1.0 * (vectors[i]["S"] / s_max)
        logger.info("  %4d    %8.3f  %8.3f  %12.4f", i, k_l[i], vectors[i]["S"], k_inject_1)

    # ── Save checkpoint ─────────────────────────────────────────────────────
    slug = _model_slug(args.model_id)
    out_path = out_dir / f"{slug}_profile.pt"

    checkpoint = {
        "model_id": args.model_id,
        "num_layers": n,
        "hidden_dim": d,
        "k_l": k_l,
        "target_indices": target,
        "vectors": vectors,
        "k_max": k_max,
        "s_max": s_max,
    }
    torch.save(checkpoint, out_path)
    logger.info("Saved profile → %s", out_path)
    logger.info("Done.  K_max=%d  Run exp02 with --profile-path %s", k_max, out_path)


if __name__ == "__main__":
    main()
