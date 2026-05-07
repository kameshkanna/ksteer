"""
Experiment 03 — Multi-layer K ramp validation across shapes and K_optimal sources.

Empirically proves: for any transformer model, multi-layer activation steering
with K_ramp = f_scale × K_optimal × shape_weights remains coherent for
f_scale ≤ f_scale_max ≈ 0.48, regardless of ramp shape or K_optimal definition.

Two K_optimal definitions compared:
    "mid"    — K_l at the single middle layer   (zip approach)
    "window" — mean K_l over the 40-80% window  (ksteer approach)

Five ramp shapes tested:
    linear, cosine, bell, exponential, constant

If K_optimal correctly captures residual stream scale, then f_scale_max ≈ 0.48
should hold across all shapes, K_optimal sources, and model families.

Outputs:
    results/exp03/{model}/ramp_validation.json   — full probe logs per (shape, source)
    results/exp03/{model}/ramp_summary.json       — f_scale_max table per (shape, source)
    results/exp03/cross_model_ramp.json           — all models aggregated
    results/exp03/cross_model_ramp.md             — human-readable proof table

Usage:
    python experiments/exp03_ramp_validation.py \\
        --model meta-llama/Llama-3.2-1B --model-name llama-3.2-1b

    # All small models via batch runner
    python experiments/run_all.py --tiers small --skip-existing
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ksteer.contrastive import BehavioralVector
from ksteer.profiler import NormProfile
from ksteer.steerer import MultiLayerSteerer, RampShape
from ksteer.utils.model_utils import load_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BASELINE_F_START = 0.13
BASELINE_F_MAX = 0.48
K_OPTIMAL_SOURCES = ["mid", "window"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp 03: multi-layer K ramp validation")
    p.add_argument("--model", required=True, type=str)
    p.add_argument("--model-name", default=None, type=str)
    p.add_argument("--device", default=None, type=str)
    p.add_argument("--exp01-dir", default="results/exp01", type=str)
    p.add_argument("--exp02-dir", default="results/exp02", type=str)
    p.add_argument("--output-dir", default="results/exp03", type=str)
    p.add_argument("--behaviors", nargs="+", default=None)
    p.add_argument(
        "--f-start", type=float, default=BASELINE_F_START,
        help=f"Ramp start as fraction of K_optimal peak (default: {BASELINE_F_START})",
    )
    p.add_argument(
        "--f-values", nargs="+", type=float, default=None,
        help="f_scale values to sweep (default: 0.10 to 1.0 in 0.05 steps)",
    )
    p.add_argument(
        "--shapes", nargs="+", default=None,
        choices=[s.value for s in RampShape],
        help="Ramp shapes to test (default: all five)",
    )
    p.add_argument(
        "--k-sources", nargs="+", default=None,
        choices=K_OPTIMAL_SOURCES,
        help="K_optimal sources to compare (default: both mid and window)",
    )
    p.add_argument("--window-min", type=float, default=0.4)
    p.add_argument("--window-max", type=float, default=0.8)
    p.add_argument(
        "--n-layers", type=int, default=9,
        help="Number of evenly-spaced injection layers in the steering window (default: 9)",
    )
    p.add_argument(
        "--sweep-prompt",
        default="Tell me something interesting about the history of science.",
        type=str,
    )
    p.add_argument("--sweep-max-tokens", default=60, type=int)
    p.add_argument("--seed", default=42, type=int)
    return p.parse_args()


def load_norm_profile(exp01_dir: Path, model_name: str) -> NormProfile:
    path = exp01_dir / model_name / "norm_profile.json"
    if not path.exists():
        raise FileNotFoundError(f"Norm profile not found: {path}. Run Exp 01 first.")
    with open(path) as f:
        d = json.load(f)
    return NormProfile(
        model_name=d["model_name"],
        model_family=d["model_family"],
        hidden_dim=d["hidden_dim"],
        num_layers=d["num_layers"],
        layer_mean_norms=d["layer_mean_norms"],
        layer_std_norms=d["layer_std_norms"],
        k_values=d["k_values"],
        num_tokens_sampled=d["num_tokens_sampled"],
    )


def find_behaviors(exp02_model_dir: Path, requested: Optional[List[str]]) -> List[str]:
    if requested:
        return requested
    return sorted(
        p.name for p in exp02_model_dir.iterdir()
        if p.is_dir() and (p / "vectors.npz").exists()
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_name = args.model_name or args.model.split("/")[-1]
    exp01_dir = Path(args.exp01_dir)
    exp02_dir = Path(args.exp02_dir)
    exp02_model_dir = exp02_dir / model_name
    out_dir = Path(args.output_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    profile = load_norm_profile(exp01_dir, model_name)
    logger.info(
        "Profile: family=%s  layers=%d  hidden=%d  K_opt_mid=%.4f  K_opt_window=%.4f",
        profile.model_family, profile.num_layers, profile.hidden_dim,
        profile.k_values[profile.num_layers // 2], profile.window_k_mean,
    )

    behaviors = find_behaviors(exp02_model_dir, args.behaviors)
    if not behaviors:
        raise FileNotFoundError(f"No behaviors in {exp02_model_dir}. Run Exp 02 first.")
    logger.info("Behaviors: %s", behaviors)

    shapes = [RampShape(s) for s in args.shapes] if args.shapes else RampShape.all()
    k_sources = args.k_sources or K_OPTIMAL_SOURCES
    logger.info("Shapes: %s", [s.value for s in shapes])
    logger.info("K_optimal sources: %s", k_sources)

    n = profile.num_layers
    layer_indices = sorted({
        max(0, min(n - 1, round(pct * (n - 1))))
        for pct in np.linspace(args.window_min, args.window_max, args.n_layers)
    })
    logger.info("Injection layers (%d): %s", len(layer_indices), layer_indices)

    f_scale_values = args.f_values or [round(x, 2) for x in np.arange(0.10, 1.01, 0.05).tolist()]
    logger.info("f_scale sweep: %s", f_scale_values)

    model, tokenizer = load_model(args.model, device=args.device)
    steerer = MultiLayerSteerer(model, tokenizer, profile)

    # f_start_frac: the ratio of the ramp start value to its peak (= 1.0 × f_scale)
    # e.g. f_start=0.13, f_scale_peak≈0.48  →  f_start_frac ≈ 0.27
    f_start_frac = round(args.f_start / BASELINE_F_MAX, 4)
    logger.info("f_start=%.3f  f_start_frac=%.4f (ratio used by ramp shapes)", args.f_start, f_start_frac)

    # Structure: behavior → shape → k_source → {f_scale_max, probes, ...}
    behavior_results: Dict[str, Dict[str, Dict[str, dict]]] = {}

    for behavior in behaviors:
        bvec = BehavioralVector.load(exp02_model_dir / behavior)
        vec_dict = {li: bvec.get_vector(li) for li in layer_indices}
        behavior_results[behavior] = {}
        logger.info("=== behavior: %s ===", behavior)

        for shape in shapes:
            behavior_results[behavior][shape.value] = {}
            for k_source in k_sources:
                k_opt = steerer.k_optimal(k_source)
                f_scale_max, probes = steerer.find_f_scale_max(
                    prompt=args.sweep_prompt,
                    behavioral_vectors=vec_dict,
                    layer_indices=layer_indices,
                    shape=shape,
                    f_scale_values=f_scale_values,
                    k_optimal_source=k_source,
                    f_start_frac=f_start_frac,
                    max_new_tokens=args.sweep_max_tokens,
                )

                k_peak_raw = round(f_scale_max * k_opt, 6) if f_scale_max is not None else None
                deviation = (
                    round((f_scale_max - BASELINE_F_MAX) / BASELINE_F_MAX, 4)
                    if f_scale_max is not None else None
                )
                behavior_results[behavior][shape.value][k_source] = {
                    "k_optimal": round(k_opt, 4),
                    "f_start": args.f_start,
                    "f_start_frac": f_start_frac,
                    "f_scale_max": f_scale_max,
                    "k_peak_raw": k_peak_raw,
                    "baseline_f_max": BASELINE_F_MAX,
                    "deviation_from_baseline": deviation,
                    "probes": [p.to_dict() for p in probes],
                }
                logger.info(
                    "  %-18s  %-11s  %-6s  f_max=%-5s  K_peak=%-8s  dev=%s",
                    behavior, shape.value, k_source,
                    f"{f_scale_max:.2f}" if f_scale_max is not None else "None",
                    f"{k_peak_raw:.4f}" if k_peak_raw is not None else "None",
                    f"{deviation:+.3f}" if deviation is not None else "—",
                )

    # ── Per-model ramp summary ────────────────────────────────────────────────
    # Aggregate f_scale_max per (shape, k_source) across behaviors
    shape_source_stats: Dict[str, Dict[str, dict]] = {}
    for shape in shapes:
        shape_source_stats[shape.value] = {}
        for k_source in k_sources:
            k_opt = steerer.k_optimal(k_source)
            measured = [
                behavior_results[b][shape.value][k_source]["f_scale_max"]
                for b in behaviors
                if behavior_results[b][shape.value][k_source]["f_scale_max"] is not None
            ]
            if measured:
                mean_f = float(np.mean(measured))
                std_f = float(np.std(measured))
                consistent = all(abs(f - BASELINE_F_MAX) / BASELINE_F_MAX < 0.20 for f in measured)
                shape_source_stats[shape.value][k_source] = {
                    "k_optimal": round(k_opt, 4),
                    "mean_f_scale_max": round(mean_f, 4),
                    "std_f_scale_max": round(std_f, 4),
                    "mean_k_peak_raw": round(mean_f * k_opt, 6),
                    "formula_holds": consistent,
                }
            else:
                shape_source_stats[shape.value][k_source] = {
                    "k_optimal": round(k_opt, 4),
                    "mean_f_scale_max": None,
                    "std_f_scale_max": None,
                    "mean_k_peak_raw": None,
                    "formula_holds": False,
                }

    summary = {
        "model": model_name,
        "family": profile.model_family,
        "baseline_f_max": BASELINE_F_MAX,
        "f_start": args.f_start,
        "f_start_frac": f_start_frac,
        "k_optimal_mid": round(steerer.k_optimal("mid"), 4),
        "k_optimal_window": round(steerer.k_optimal("window"), 4),
        "shape_source_stats": shape_source_stats,
    }

    summary_path = out_dir / "ramp_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    raw_path = out_dir / "ramp_validation.json"
    with open(raw_path, "w") as f:
        json.dump({
            "model": model_name,
            "f_start": args.f_start,
            "f_start_frac": f_start_frac,
            "layer_indices": layer_indices,
            "behaviors": behavior_results,
        }, f, indent=2)

    logger.info("Summary → %s", summary_path)
    logger.info("Raw data → %s", raw_path)

    # Print per-model table
    _log_model_table(model_name, shape_source_stats, shapes, k_sources)

    # ── Cross-model aggregation ───────────────────────────────────────────────
    cross_path = Path(args.output_dir) / "cross_model_ramp.json"
    cross: Dict[str, dict] = {}
    if cross_path.exists():
        with open(cross_path) as f:
            cross = json.load(f)
    cross[model_name] = summary
    with open(cross_path, "w") as f:
        json.dump(cross, f, indent=2)

    _write_markdown(cross, Path(args.output_dir) / "cross_model_ramp.md")
    logger.info("Cross-model → %s", cross_path)


def _log_model_table(
    model_name: str,
    stats: Dict[str, Dict[str, dict]],
    shapes: List[RampShape],
    k_sources: List[str],
) -> None:
    logger.info("\n  Results for %s", model_name)
    header = f"  {'shape':<12}  {'k_source':<8}  {'K_opt':>7}  {'f_max':>6}  {'K_peak':>8}  {'std':>6}  holds"
    logger.info(header)
    logger.info("  " + "─" * 60)
    for shape in shapes:
        for k_source in k_sources:
            s = stats.get(shape.value, {}).get(k_source, {})
            mf = s.get("mean_f_scale_max")
            km = s.get("mean_k_peak_raw")
            std = s.get("std_f_scale_max")
            holds = "✓" if s.get("formula_holds") else "✗"
            logger.info(
                "  %-12s  %-8s  %7.4f  %6s  %8s  %6s  %s",
                shape.value, k_source,
                s.get("k_optimal", 0.0),
                f"{mf:.3f}" if mf is not None else "—",
                f"{km:.4f}" if km is not None else "—",
                f"{std:.3f}" if std is not None else "—",
                holds,
            )


def _write_markdown(cross: Dict[str, dict], path: Path) -> None:
    lines = [
        "# Exp 03 — Multi-layer K Ramp Validation",
        "",
        "**Claim**: `f_scale_max ≈ 0.48` across all model families, ramp shapes,",
        "and K_optimal definitions, where `K_i = f_scale × K_optimal × shape_weights[i]`.",
        "",
        f"Baseline (Qwen2.5-3B-Instruct empirical): f_scale_max = {BASELINE_F_MAX}",
        "Consistent = |f_scale_max − 0.48| / 0.48 < 20%",
        "",
    ]

    for model_name, ms in sorted(cross.items()):
        lines += [
            f"## {model_name}  (family={ms.get('family', '?')})",
            "",
            f"K_optimal_mid = {ms.get('k_optimal_mid', '?'):.4f}  "
            f"| K_optimal_window = {ms.get('k_optimal_window', '?'):.4f}",
            "",
            "| Shape | K_source | K_optimal | mean f_max | K_peak_raw | std | ±baseline | Holds? |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for shape_name, src_stats in ms.get("shape_source_stats", {}).items():
            for k_source, s in src_stats.items():
                mf = s.get("mean_f_scale_max")
                km = s.get("mean_k_peak_raw")
                std = s.get("std_f_scale_max")
                dev = round((mf - BASELINE_F_MAX) / BASELINE_F_MAX, 3) if mf is not None else None
                holds = "✓" if s.get("formula_holds") else "✗" if "formula_holds" in s else "—"
                lines.append(
                    f"| {shape_name} | {k_source} "
                    f"| {s.get('k_optimal', 0.0):.4f} "
                    f"| {f'{mf:.3f}' if mf is not None else '—'} "
                    f"| {f'{km:.4f}' if km is not None else '—'} "
                    f"| {f'{std:.3f}' if std is not None else '—'} "
                    f"| {f'{dev:+.3f}' if dev is not None else '—'} "
                    f"| {holds} |"
                )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
