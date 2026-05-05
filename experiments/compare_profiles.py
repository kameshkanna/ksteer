"""
Load all completed norm_profile.json files and generate cross-model comparison plots.

Produces:
  - results/exp01/comparison_norm_profiles.png  — K_l curves for all models
  - results/exp01/comparison_k_table.json       — K_l at 40/60/80% depth per model
  - results/exp01/comparison_family_summary.json — mean K in steering window per family

Usage:
    python experiments/compare_profiles.py
    python experiments/compare_profiles.py --results-dir results/exp01
    python experiments/compare_profiles.py --families llama gemma2
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

from ksteer.profiler import NormProfile
from ksteer.utils.plot_utils import plot_norm_profiles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-model K_l comparison")
    p.add_argument("--results-dir", default="results/exp01", type=str)
    p.add_argument("--families", nargs="+", default=None,
                   help="Filter by family name")
    p.add_argument("--show", action="store_true", help="Display plots interactively")
    return p.parse_args()


def load_profiles(results_dir: Path, families: Optional[list[str]] = None) -> list[NormProfile]:
    profiles = []
    for profile_path in sorted(results_dir.glob("*/norm_profile.json")):
        with open(profile_path) as f:
            data = json.load(f)

        if families and data.get("model_family") not in families:
            continue

        profiles.append(NormProfile(
            model_name=data["model_name"],
            model_family=data["model_family"],
            hidden_dim=data["hidden_dim"],
            num_layers=data["num_layers"],
            layer_mean_norms=data["layer_mean_norms"],
            layer_std_norms=data["layer_std_norms"],
            k_values=data["k_values"],
            num_tokens_sampled=data.get("num_tokens_sampled", 0),
        ))
        logger.info("Loaded: %s  (%s, %d layers)", data["model_name"], data["model_family"], data["num_layers"])

    return profiles


def build_k_table(profiles: list[NormProfile]) -> list[dict]:
    """K_l values at 40%, 60%, 80% depth and window summary per model."""
    rows = []
    for p in profiles:
        start, end = p.steering_window
        k_min, k_max = p.window_k_range
        rows.append({
            "model": p.model_name,
            "family": p.model_family,
            "num_layers": p.num_layers,
            "hidden_dim": p.hidden_dim,
            "k_at_40pct": round(p.k_values[int(0.4 * p.num_layers)], 5),
            "k_at_60pct": round(p.k_values[int(0.6 * p.num_layers)], 5),
            "k_at_80pct": round(p.k_values[int(0.8 * p.num_layers)], 5),
            "window_k_min": round(k_min, 5),
            "window_k_max": round(k_max, 5),
            "window_k_mean": round(
                sum(p.k_values[start:end]) / max(end - start, 1), 5
            ),
        })
    return rows


def build_family_summary(profiles: list[NormProfile]) -> dict[str, dict]:
    """Aggregate window K stats by model family."""
    from collections import defaultdict
    family_data: dict[str, list[float]] = defaultdict(list)

    for p in profiles:
        start, end = p.steering_window
        window_mean = sum(p.k_values[start:end]) / max(end - start, 1)
        family_data[p.model_family].append(window_mean)

    return {
        family: {
            "num_models": len(vals),
            "mean_window_k": round(sum(vals) / len(vals), 5),
            "min_window_k": round(min(vals), 5),
            "max_window_k": round(max(vals), 5),
        }
        for family, vals in sorted(family_data.items())
    }


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)

    profiles = load_profiles(results_dir, args.families)
    if not profiles:
        logger.error("No completed profiles found in %s", results_dir)
        return

    logger.info("Loaded %d profile(s)", len(profiles))

    # ── Comparison plot ─────────────────────────────────────────────────
    plot_norm_profiles(
        profiles,
        output_path=results_dir / "comparison_norm_profiles.png",
        show=args.show,
    )

    # ── K table ─────────────────────────────────────────────────────────
    k_table = build_k_table(profiles)
    k_table_path = results_dir / "comparison_k_table.json"
    with open(k_table_path, "w") as f:
        json.dump(k_table, f, indent=2)

    # Print as a readable table
    header = f"{'Model':<25} {'Family':<10} {'K@40%':>8} {'K@60%':>8} {'K@80%':>8} {'Win.Mean':>10}"
    logger.info("\n%s\n%s", header, "─" * len(header))
    for row in k_table:
        logger.info(
            "%-25s %-10s %8.4f %8.4f %8.4f %10.4f",
            row["model"], row["family"],
            row["k_at_40pct"], row["k_at_60pct"], row["k_at_80pct"], row["window_k_mean"],
        )

    # ── Family summary ──────────────────────────────────────────────────
    family_summary = build_family_summary(profiles)
    family_path = results_dir / "comparison_family_summary.json"
    with open(family_path, "w") as f:
        json.dump(family_summary, f, indent=2)

    logger.info("\nFamily summary:")
    for fam, stats in family_summary.items():
        logger.info("  %-10s  mean_window_K=%.4f  (n=%d models)", fam, stats["mean_window_k"], stats["num_models"])

    logger.info("Comparison plot → %s", results_dir / "comparison_norm_profiles.png")
    logger.info("K table         → %s", k_table_path)
    logger.info("Family summary  → %s", family_path)


if __name__ == "__main__":
    main()
