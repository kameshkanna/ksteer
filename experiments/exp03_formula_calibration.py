"""
Experiment 03 — Formula calibration: does K_l^b = K_l / ρ_l universally predict
the behavioral coherence ceiling?

K_l is the random-vector ceiling (mean_norm_l / sqrt(d)).  Real behavioral
vectors break coherence at alpha_eff × K_l where alpha_eff << 1.  The
behavioral SNR ρ_l = ||mean_diff_l|| / mean_norm_l should explain alpha_eff:

    K_l^b = K_l / ρ_l  =  mean_norm_l² / (sqrt(d) × ||mean_diff_l||)

If this holds, the ratio  α_eff × K_l / K_l^b ≈ constant  across all layers
and behaviors.  A flat ratio means K_l^b is a behavior-calibrated ceiling;
any residual layer-depth drift quantifies the remaining correction needed.

Pure post-hoc analysis — no GPU or model loading required.

Reads (already produced by Exp 01, 02, and 02b):
    results/exp01/{model}/norm_profile.json
    results/exp02/{model}/{behavior}/vectors_meta.json
    results/exp02/{model}/formula_validation/formula_summary.json

Writes:
    results/exp03/{model}/{behavior}_calibration.json
    results/exp03/{model}/calibration_summary.json
    results/exp03/{model}/calibration_plot.png
    results/exp03/cross_model_summary.json
    results/exp03/cross_model_summary.md

Usage:
    python experiments/exp03_formula_calibration.py
    python experiments/exp03_formula_calibration.py --models llama-3.2-1b qwen2.5-1.5b
    python experiments/exp03_formula_calibration.py --exp01-dir results/exp01 --exp02-dir results/exp02
"""

import argparse
import json
import logging
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class LayerCalibration:
    """Calibration data for a single (behavior, layer) point."""
    layer_idx: int
    layer_pct: float
    k_l: float          # K_l = mean_norm_l / sqrt(d) from Exp 01
    mean_norm_l: float  # mean residual stream norm at this layer
    rho_l: float        # behavioral SNR = ||mean_diff_l|| / mean_norm_l
    k_l_b: float        # generalized ceiling = K_l / rho_l
    alpha_empirical: float   # measured ceiling alpha (in units of K_l)
    abs_ceiling: float       # empirical absolute ceiling = alpha_empirical × K_l
    ratio: float             # abs_ceiling / k_l_b  (≈ 1.0 if formula is calibrated)


@dataclass
class BehaviorCalibration:
    behavior: str
    model_name: str
    layers: List[LayerCalibration]
    mean_ratio: float
    std_ratio: float
    cv_ratio: float
    # R² of linear fit: abs_ceiling ≈ ratio_mean × k_l_b
    r_squared: Optional[float]
    calibration_quality: str   # "excellent" <0.10, "good" <0.20, "partial" <0.40, "poor"


# ── Loaders ────────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_norm_profile(exp01_dir: Path, model_name: str) -> Optional[dict]:
    return _load_json(exp01_dir / model_name / "norm_profile.json")


def load_bvec_meta(exp02_dir: Path, model_name: str, behavior: str) -> Optional[dict]:
    return _load_json(exp02_dir / model_name / behavior / "vectors_meta.json")


def load_formula_summary(exp02_dir: Path, model_name: str) -> Optional[dict]:
    return _load_json(exp02_dir / model_name / "formula_validation" / "formula_summary.json")


# ── Core computation ───────────────────────────────────────────────────────────

def compute_layer_calibration(
    layer_idx: int,
    num_layers: int,
    k_l: float,
    mean_norm_l: float,
    raw_norm_l: float,     # ||mean_diff_l^b|| from BehavioralVector.layer_raw_norms
    alpha_empirical: float,
) -> LayerCalibration:
    rho_l = raw_norm_l / mean_norm_l if mean_norm_l > 0 else float("inf")
    k_l_b = k_l / rho_l if rho_l > 0 else float("inf")
    abs_ceiling = alpha_empirical * k_l
    ratio = abs_ceiling / k_l_b if k_l_b > 0 else float("nan")
    return LayerCalibration(
        layer_idx=layer_idx,
        layer_pct=layer_idx / num_layers,
        k_l=k_l,
        mean_norm_l=mean_norm_l,
        rho_l=rho_l,
        k_l_b=k_l_b,
        alpha_empirical=alpha_empirical,
        abs_ceiling=abs_ceiling,
        ratio=ratio,
    )


def _r_squared(xs: List[float], ys: List[float], slope: float) -> Optional[float]:
    """R² for the model y ≈ slope × x (no intercept)."""
    if len(xs) < 2:
        return None
    predicted = [slope * x for x in xs]
    ss_res = sum((y - p) ** 2 for y, p in zip(ys, predicted))
    y_mean = sum(ys) / len(ys)
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else None


def calibrate_behavior(
    behavior: str,
    model_name: str,
    profile: dict,
    bvec_meta: dict,
    formula_summary: dict,
) -> Optional[BehaviorCalibration]:
    behavior_data = formula_summary.get(behavior)
    if behavior_data is None:
        logger.warning("No formula_summary entry for behavior=%s", behavior)
        return None

    ceiling_by_layer: Dict[str, Optional[float]] = behavior_data.get("ceiling_by_layer", {})
    if not ceiling_by_layer:
        return None

    num_layers: int = profile["num_layers"]
    k_values: List[float] = profile["k_values"]
    mean_norms: List[float] = profile["layer_mean_norms"]
    raw_norms: List[float] = bvec_meta["layer_raw_norms"]

    layers: List[LayerCalibration] = []
    for layer_str, alpha in ceiling_by_layer.items():
        if alpha is None:
            continue   # uncapped — cannot compute empirical ceiling
        layer_idx = int(layer_str)
        if layer_idx >= num_layers:
            continue
        cal = compute_layer_calibration(
            layer_idx=layer_idx,
            num_layers=num_layers,
            k_l=k_values[layer_idx],
            mean_norm_l=mean_norms[layer_idx],
            raw_norm_l=raw_norms[layer_idx],
            alpha_empirical=alpha,
        )
        layers.append(cal)

    if len(layers) < 2:
        logger.warning("Too few calibrated layers for behavior=%s (got %d)", behavior, len(layers))
        return None

    ratios = [c.ratio for c in layers if math.isfinite(c.ratio)]
    if not ratios:
        return None

    mean_r = sum(ratios) / len(ratios)
    std_r = (sum((r - mean_r) ** 2 for r in ratios) / len(ratios)) ** 0.5
    cv_r = std_r / mean_r if mean_r > 0 else float("inf")

    xs = [c.k_l_b for c in layers if math.isfinite(c.ratio)]
    ys = [c.abs_ceiling for c in layers if math.isfinite(c.ratio)]
    r2 = _r_squared(xs, ys, slope=mean_r)

    quality = (
        "excellent" if cv_r < 0.10 else
        "good" if cv_r < 0.20 else
        "partial" if cv_r < 0.40 else
        "poor"
    )

    return BehaviorCalibration(
        behavior=behavior,
        model_name=model_name,
        layers=layers,
        mean_ratio=mean_r,
        std_ratio=std_r,
        cv_ratio=cv_r,
        r_squared=r2,
        calibration_quality=quality,
    )


# ── Plotting ───────────────────────────────────────────────────────────────────

def _plot_calibration(
    calibrations: Dict[str, BehaviorCalibration],
    model_name: str,
    output_path: Path,
) -> None:
    """
    Two-panel figure:
      Left:  K_l^b (x) vs empirical absolute ceiling (y), colored per behavior.
             y = x line is perfect calibration; slope ≈ mean_ratio.
      Right: ratio distribution per behavior (box + scatter), with reference at 1.0.
    """
    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    all_ratios: List[float] = []
    behavior_labels: List[str] = []
    behavior_ratio_lists: List[List[float]] = []

    for i, (beh, cal) in enumerate(calibrations.items()):
        color = colors[i % len(colors)]
        k_l_bs = [c.k_l_b for c in cal.layers]
        abs_ceilings = [c.abs_ceiling for c in cal.layers]
        ratios = [c.ratio for c in cal.layers]

        axes[0].scatter(k_l_bs, abs_ceilings, color=color, label=beh, s=40, alpha=0.8, zorder=3)
        all_ratios.extend(ratios)
        behavior_labels.append(beh)
        behavior_ratio_lists.append(ratios)

    # y = mean_ratio × x reference line
    if all_ratios:
        grand_mean = sum(all_ratios) / len(all_ratios)
        x_vals = np.array(sorted(
            c.k_l_b for cal in calibrations.values() for c in cal.layers
        ))
        axes[0].plot(x_vals, grand_mean * x_vals, "--", color="black", linewidth=1.2,
                     alpha=0.7, label=f"fit: y = {grand_mean:.3f}×K_l^b", zorder=2)
        axes[0].plot(x_vals, x_vals, ":", color="gray", linewidth=0.9,
                     alpha=0.5, label="y = K_l^b (perfect calibration)", zorder=1)

    axes[0].set_xlabel("K_l^b (predicted ceiling)", fontsize=11)
    axes[0].set_ylabel("Empirical absolute ceiling (α_eff × K_l)", fontsize=11)
    axes[0].set_title(f"K_l^b vs Empirical Ceiling\n{model_name}", fontsize=11)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.25)

    # Right panel: ratio distributions per behavior
    positions = list(range(len(behavior_labels)))
    for i, (label, ratios) in enumerate(zip(behavior_labels, behavior_ratio_lists)):
        color = colors[i % len(colors)]
        axes[1].scatter([i] * len(ratios), ratios, color=color, alpha=0.7, s=40, zorder=3)
        if len(ratios) > 1:
            axes[1].plot([i - 0.2, i + 0.2],
                         [np.mean(ratios)] * 2, color=color, linewidth=2.5, zorder=4)

    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6, label="ratio = 1.0")
    if all_ratios:
        axes[1].axhline(grand_mean, color="red", linestyle=":", linewidth=1.0, alpha=0.6,
                        label=f"mean ratio = {grand_mean:.3f}")

    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(behavior_labels, rotation=20, ha="right", fontsize=9)
    axes[1].set_ylabel("ratio = empirical_ceiling / K_l^b", fontsize=11)
    axes[1].set_title(f"Calibration Ratio Distribution\n{model_name}", fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.25)

    fig.suptitle(
        f"Exp 03 — K_l^b = K_l / ρ_l as universal behavioral ceiling\n"
        f"{model_name}  |  ratio ≈ 1 means K_l^b perfectly predicts the empirical ceiling",
        fontsize=10,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info("Plot saved → %s", output_path)
    plt.close(fig)


def _plot_cross_model(
    cross_model: Dict[str, Dict[str, float]],
    output_path: Path,
) -> None:
    """
    Bar chart of mean_ratio and CV per model (all behaviors aggregated).
    Ideal: all bars near 1.0 with low CV.
    """
    models = list(cross_model.keys())
    mean_ratios = [cross_model[m]["mean_ratio"] for m in models]
    cv_ratios = [cross_model[m]["cv_ratio"] for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(len(models))
    axes[0].bar(x, mean_ratios, width=0.6, color="steelblue", alpha=0.8)
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=25, ha="right", fontsize=9)
    axes[0].set_ylabel("Mean ratio (empirical / K_l^b)", fontsize=11)
    axes[0].set_title("Cross-model: mean calibration ratio", fontsize=11)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, cv_ratios, width=0.6, color="darkorange", alpha=0.8)
    axes[1].axhline(0.20, color="green", linestyle="--", linewidth=0.9, alpha=0.7, label="CV=0.20 (good)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=25, ha="right", fontsize=9)
    axes[1].set_ylabel("CV of ratio (lower = better calibration)", fontsize=11)
    axes[1].set_title("Cross-model: ratio coefficient of variation", fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("Exp 03 — K_l^b calibration across all models", fontsize=11)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info("Plot saved → %s", output_path)
    plt.close(fig)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp 03: K_l^b formula calibration (post-hoc analysis)")
    p.add_argument("--exp01-dir", default="results/exp01")
    p.add_argument("--exp02-dir", default="results/exp02")
    p.add_argument("--output-dir", default="results/exp03")
    p.add_argument("--models", nargs="+", default=None,
                   help="Specific model keys to analyse (default: all found in exp02-dir)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    exp01_dir = Path(args.exp01_dir)
    exp02_dir = Path(args.exp02_dir)
    out_dir = Path(args.output_dir)

    # Discover models
    if args.models:
        model_names = args.models
    else:
        model_names = sorted(
            d.name for d in exp02_dir.iterdir()
            if d.is_dir() and (d / "formula_validation" / "formula_summary.json").exists()
        )

    if not model_names:
        logger.error(
            "No models found in %s with formula_validation results. "
            "Run Exp 01, Exp 02, and Exp 02b first.",
            exp02_dir,
        )
        return

    logger.info("Models to calibrate: %s", model_names)

    # cross-model summary: model → aggregated stats
    cross_model: Dict[str, dict] = {}

    for model_name in model_names:
        logger.info("━━━ %s ━━━", model_name)

        profile = load_norm_profile(exp01_dir, model_name)
        if profile is None:
            logger.warning("  Norm profile not found — skipping %s", model_name)
            continue

        formula_summary = load_formula_summary(exp02_dir, model_name)
        if formula_summary is None:
            logger.warning("  Formula summary not found — skipping %s", model_name)
            continue

        behaviors = list(formula_summary.keys())
        model_out = out_dir / model_name
        model_out.mkdir(parents=True, exist_ok=True)

        model_calibrations: Dict[str, BehaviorCalibration] = {}

        for behavior in behaviors:
            bvec_meta = load_bvec_meta(exp02_dir, model_name, behavior)
            if bvec_meta is None:
                logger.warning("  vectors_meta.json not found for behavior=%s — skipping", behavior)
                continue

            cal = calibrate_behavior(behavior, model_name, profile, bvec_meta, formula_summary)
            if cal is None:
                continue

            model_calibrations[behavior] = cal

            per_layer = [asdict(c) for c in cal.layers]
            cal_path = model_out / f"{behavior}_calibration.json"
            with open(cal_path, "w") as f:
                json.dump({
                    "behavior": cal.behavior,
                    "model_name": cal.model_name,
                    "mean_ratio": round(cal.mean_ratio, 4),
                    "std_ratio": round(cal.std_ratio, 4),
                    "cv_ratio": round(cal.cv_ratio, 4),
                    "r_squared": round(cal.r_squared, 4) if cal.r_squared is not None else None,
                    "calibration_quality": cal.calibration_quality,
                    "layers": per_layer,
                }, f, indent=2)

            logger.info(
                "  %-20s  ratio=%.3f±%.3f  CV=%.3f  R²=%s  → %s",
                behavior,
                cal.mean_ratio,
                cal.std_ratio,
                cal.cv_ratio,
                f"{cal.r_squared:.3f}" if cal.r_squared is not None else "—",
                cal.calibration_quality,
            )

        if not model_calibrations:
            continue

        # Per-model summary
        all_ratios = [c.ratio for cal in model_calibrations.values() for c in cal.layers if math.isfinite(c.ratio)]
        grand_mean = sum(all_ratios) / len(all_ratios)
        grand_std = (sum((r - grand_mean) ** 2 for r in all_ratios) / len(all_ratios)) ** 0.5
        grand_cv = grand_std / grand_mean if grand_mean > 0 else float("inf")

        model_summary = {
            "model_name": model_name,
            "num_behaviors": len(model_calibrations),
            "num_calibrated_layers": len(all_ratios),
            "grand_mean_ratio": round(grand_mean, 4),
            "grand_std_ratio": round(grand_std, 4),
            "grand_cv_ratio": round(grand_cv, 4),
            "per_behavior": {
                beh: {
                    "mean_ratio": round(cal.mean_ratio, 4),
                    "cv_ratio": round(cal.cv_ratio, 4),
                    "r_squared": round(cal.r_squared, 4) if cal.r_squared is not None else None,
                    "calibration_quality": cal.calibration_quality,
                }
                for beh, cal in model_calibrations.items()
            },
        }
        summary_path = model_out / "calibration_summary.json"
        with open(summary_path, "w") as f:
            json.dump(model_summary, f, indent=2)
        logger.info("  Summary → %s", summary_path)

        _plot_calibration(model_calibrations, model_name, model_out / "calibration_plot.png")

        cross_model[model_name] = {
            "mean_ratio": round(grand_mean, 4),
            "std_ratio": round(grand_std, 4),
            "cv_ratio": round(grand_cv, 4),
            "num_points": len(all_ratios),
        }

    if not cross_model:
        logger.error("No models calibrated — check that Exp 01, 02, and 02b have been run.")
        return

    # Cross-model JSON
    cross_path = out_dir / "cross_model_summary.json"
    with open(cross_path, "w") as f:
        json.dump(cross_model, f, indent=2)
    logger.info("Cross-model summary → %s", cross_path)

    _plot_cross_model(cross_model, out_dir / "cross_model_calibration.png")

    # Cross-model markdown
    lines = [
        "# Exp 03 — K_l^b = K_l / ρ_l Formula Calibration\n",
        "## Interpretation\n",
        "- **ratio = empirical absolute ceiling / K_l^b** — should ≈ 1.0 if K_l^b is a universal ceiling.",
        "- **CV < 0.10**: excellent calibration (formula holds across all layers).",
        "- **CV 0.10–0.20**: good (small layer-depth correction may be needed).",
        "- **CV 0.20–0.40**: partial (layer-depth trend remains after normalization).",
        "- **CV > 0.40**: poor (ρ_l alone insufficient; higher-order correction required).\n",
        "## Per-model calibration\n",
        "| Model | Mean ratio | CV | n points | Assessment |",
        "|---|---|---|---|---|",
    ]
    for model, stats in cross_model.items():
        cv = stats["cv_ratio"]
        quality = (
            "excellent" if cv < 0.10 else
            "good" if cv < 0.20 else
            "partial" if cv < 0.40 else
            "poor"
        )
        lines.append(
            f"| {model} | {stats['mean_ratio']:.3f} | {cv:.3f} | {stats['num_points']} | {quality} |"
        )

    lines += [
        "\n## Key interpretation\n",
        f"- Grand mean ratio across models: {sum(s['mean_ratio'] for s in cross_model.values()) / len(cross_model):.3f}",
        "- If mean ratio ≈ 1.0: K_l^b = K_l / ρ_l is directly calibrated.",
        "- If mean ratio ≈ c ≠ 1.0: the generalized formula needs a family-level constant  K_l^b × c.",
        "- Low CV (< 0.20) with mean ratio near 1 confirms K_l^b as the practical steering budget.",
    ]

    md_path = out_dir / "cross_model_summary.md"
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Markdown → %s", md_path)

    # Final console table
    logger.info("\n%-25s  %10s  %8s  %8s  %s", "model", "mean_ratio", "CV", "n", "quality")
    logger.info("─" * 68)
    for model, stats in cross_model.items():
        cv = stats["cv_ratio"]
        q = "excellent" if cv < 0.10 else "good" if cv < 0.20 else "partial" if cv < 0.40 else "poor"
        logger.info("%-25s  %10.3f  %8.3f  %8d  %s",
                    model, stats["mean_ratio"], cv, stats["num_points"], q)


if __name__ == "__main__":
    main()
