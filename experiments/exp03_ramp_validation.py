"""
Experiment 03 — Multi-layer K ramp validation.

Empirically proves: for any transformer model, multi-layer activation steering
with K_ramp = linspace(f_start, f_end, n) × K_optimal steers behavior without
coherence collapse for f_end ≤ f_max, where f_max ≈ 0.48 (baseline from
Qwen2.5-3B-Instruct empirical work).

If K_optimal = mean_norm_window / sqrt(d) correctly captures each model's
residual stream scale, then f_max should be ≈ constant across Llama, Qwen,
and Gemma families, proving K_optimal as a universal steering budget.

Method:
  - Load K_optimal from Exp 01 norm profile.
  - Load per-layer behavioral vectors from Exp 02.
  - Run multi-layer ramped injection across the 40-80% steering window.
  - Sweep f_end from 0.10 to 1.0 in 0.05 steps.
  - Find f_max = largest f_end with coherent output.
  - Report: f_max, K_max_raw = f_max × K_optimal.

Outputs:
    results/exp03/{model}/ramp_validation.json   — per-behavior f_max, probes
    results/exp03/{model}/ramp_summary.json       — f_max table vs K_optimal
    results/exp03/cross_model_ramp.json           — f_max across all models
    results/exp03/cross_model_ramp.md             — human-readable proof table

Usage:
    python experiments/exp03_ramp_validation.py \\
        --model meta-llama/Llama-3.2-1B --model-name llama-3.2-1b

    # All small models
    python experiments/run_all.py --tiers small --run-ramp-validation
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from ksteer.contrastive import BehavioralVector
from ksteer.profiler import NormProfile
from ksteer.steerer import MultiLayerSteerer
from ksteer.utils.model_utils import load_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Empirical baseline from Qwen2.5-3B-Instruct deployment work.
# f_start = 0.13 (13% of K_optimal), f_end_max = 0.48 (48%).
BASELINE_F_START = 0.13
BASELINE_F_MAX = 0.48


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
        help=f"Ramp start fraction of K_optimal (default: {BASELINE_F_START}, empirical baseline)",
    )
    p.add_argument(
        "--f-values", nargs="+", type=float, default=None,
        help="f_end fractions to sweep (default: 0.10 to 1.0 in 0.05 steps)",
    )
    p.add_argument("--window-min", type=float, default=0.4)
    p.add_argument("--window-max", type=float, default=0.8)
    p.add_argument(
        "--n-layers", type=int, default=9,
        help="Number of evenly-spaced layers to sample in the window (default: 9)",
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
        "Profile: family=%s  layers=%d  hidden=%d  K_optimal=%.4f",
        profile.model_family, profile.num_layers, profile.hidden_dim, profile.window_k_mean,
    )

    behaviors = find_behaviors(exp02_model_dir, args.behaviors)
    if not behaviors:
        raise FileNotFoundError(f"No behaviors in {exp02_model_dir}. Run Exp 02 first.")
    logger.info("Behaviors: %s", behaviors)

    # Evenly-spaced layer indices across the steering window
    n = profile.num_layers
    layer_indices = sorted({
        max(0, min(n - 1, round(pct * (n - 1))))
        for pct in np.linspace(args.window_min, args.window_max, args.n_layers)
    })
    logger.info("Injection layers (%d): %s", len(layer_indices), layer_indices)

    f_values = args.f_values or [round(x, 2) for x in np.arange(0.10, 1.01, 0.05).tolist()]
    logger.info("f_end sweep: %s", f_values)

    model, tokenizer = load_model(args.model, device=args.device)
    steerer = MultiLayerSteerer(model, tokenizer, profile)

    behavior_results: Dict[str, dict] = {}

    for behavior in behaviors:
        bvec = BehavioralVector.load(exp02_model_dir / behavior)
        logger.info("=== behavior: %s ===", behavior)

        vec_dict = {li: bvec.get_vector(li) for li in layer_indices}

        f_max, probes = steerer.find_f_max(
            prompt=args.sweep_prompt,
            behavioral_vectors=vec_dict,
            layer_indices=layer_indices,
            f_start=args.f_start,
            f_values=f_values,
            max_new_tokens=args.sweep_max_tokens,
        )

        k_max_raw = round(f_max * steerer.k_optimal, 6) if f_max is not None else None
        deviation = round((f_max - BASELINE_F_MAX) / BASELINE_F_MAX, 4) if f_max else None

        behavior_results[behavior] = {
            "k_optimal": round(steerer.k_optimal, 4),
            "f_start": args.f_start,
            "f_max": f_max,
            "k_max_raw": k_max_raw,
            "baseline_f_max": BASELINE_F_MAX,
            "deviation_from_baseline": deviation,
            "probes": [p.to_dict() for p in probes],
        }

        logger.info(
            "  %-18s  f_max=%s  K_max=%s  deviation=%s",
            behavior,
            f"{f_max:.2f}" if f_max else "None",
            f"{k_max_raw:.4f}" if k_max_raw else "None",
            f"{deviation:+.3f}" if deviation is not None else "—",
        )

    # Per-model ramp summary
    summary = {
        "model": model_name,
        "family": profile.model_family,
        "k_optimal": round(steerer.k_optimal, 4),
        "baseline_f_max": BASELINE_F_MAX,
        "behaviors": {
            b: {
                "f_max": r["f_max"],
                "k_max_raw": r["k_max_raw"],
                "deviation_from_baseline": r["deviation_from_baseline"],
            }
            for b, r in behavior_results.items()
        },
    }
    measured = [r["f_max"] for r in behavior_results.values() if r["f_max"] is not None]
    if measured:
        summary["mean_f_max"] = round(sum(measured) / len(measured), 4)
        summary["std_f_max"] = round(float(np.std(measured)), 4)
        consistent = all(abs(f - BASELINE_F_MAX) / BASELINE_F_MAX < 0.20 for f in measured)
        summary["formula_holds"] = consistent
        logger.info(
            "  mean_f_max=%.3f  std=%.3f  formula_holds=%s",
            summary["mean_f_max"], summary["std_f_max"], consistent,
        )

    summary_path = out_dir / "ramp_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    raw_path = out_dir / "ramp_validation.json"
    with open(raw_path, "w") as f:
        json.dump({
            "model": model_name,
            "k_optimal": round(steerer.k_optimal, 4),
            "f_start": args.f_start,
            "layer_indices": layer_indices,
            "behaviors": behavior_results,
        }, f, indent=2)

    logger.info("Summary → %s", summary_path)
    logger.info("Raw data → %s", raw_path)

    # Cross-model aggregation (append to shared file)
    cross_path = Path(args.output_dir) / "cross_model_ramp.json"
    cross: Dict[str, dict] = {}
    if cross_path.exists():
        with open(cross_path) as f:
            cross = json.load(f)
    cross[model_name] = summary
    with open(cross_path, "w") as f:
        json.dump(cross, f, indent=2)

    # Human-readable markdown proof table
    _write_markdown(cross, Path(args.output_dir) / "cross_model_ramp.md")
    logger.info("Cross-model → %s", cross_path)

    logger.info("\n%-22s  %8s  %8s  %8s  %10s", "model", "K_opt", "f_max", "K_max", "consistent")
    logger.info("─" * 65)
    for mn, ms in cross.items():
        mf = ms.get("mean_f_max")
        km = round(mf * ms["k_optimal"], 4) if mf else None
        logger.info(
            "%-22s  %8.4f  %8s  %8s  %10s",
            mn,
            ms["k_optimal"],
            f"{mf:.3f}" if mf else "—",
            f"{km:.4f}" if km else "—",
            "✓" if ms.get("formula_holds") else "✗" if "formula_holds" in ms else "—",
        )


def _write_markdown(cross: Dict[str, dict], path: Path) -> None:
    lines = [
        "# Exp 03 — Multi-layer K Ramp Validation",
        "",
        "**Claim**: `f_max ≈ 0.48` across all model families, where",
        "`K_ramp = linspace(f_start, f_end, n) × K_optimal` and `f_max` is the",
        "largest `f_end` that produces coherent output.",
        "",
        "| Model | Family | K_optimal | mean f_max | K_max_raw | ±baseline | Holds? |",
        "|---|---|---|---|---|---|---|",
    ]
    for mn, ms in sorted(cross.items()):
        mf = ms.get("mean_f_max")
        km = round(mf * ms["k_optimal"], 4) if mf else None
        dev = round((mf - BASELINE_F_MAX) / BASELINE_F_MAX, 3) if mf else None
        holds = "✓" if ms.get("formula_holds") else "✗" if "formula_holds" in ms else "—"
        lines.append(
            f"| {mn} | {ms.get('family', '?')} "
            f"| {ms['k_optimal']:.4f} "
            f"| {mf:.3f if mf else '—'} "
            f"| {km:.4f if km else '—'} "
            f"| {f'{dev:+.3f}' if dev is not None else '—'} "
            f"| {holds} |"
        )
    lines += [
        "",
        f"Baseline (Qwen2.5-3B-Instruct empirical): f_max = {BASELINE_F_MAX}",
        "Consistent = |f_max − 0.48| / 0.48 < 20%",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
