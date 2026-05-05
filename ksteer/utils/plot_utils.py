"""Plotting utilities for norm profiles, ceiling sweeps, and formula validation."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from ksteer.profiler import CeilingProbeResult, NormProfile

if TYPE_CHECKING:
    from ksteer.contrastive import BehavioralVector

logger = logging.getLogger(__name__)


def plot_norm_profiles(
    profiles: List[NormProfile],
    output_path: Optional[Path] = None,
    show: bool = False,
) -> None:
    """
    Two-panel figure: mean residual stream norm and K_l per layer depth,
    across one or more models. Shades the 40–80% steering window.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10.colors

    for i, profile in enumerate(profiles):
        color = colors[i % len(colors)]
        layer_pcts = [l / profile.num_layers for l in range(profile.num_layers)]

        axes[0].plot(
            layer_pcts, profile.layer_mean_norms,
            label=profile.model_name, color=color, linewidth=1.8,
        )
        axes[0].fill_between(
            layer_pcts,
            [m - s for m, s in zip(profile.layer_mean_norms, profile.layer_std_norms)],
            [m + s for m, s in zip(profile.layer_mean_norms, profile.layer_std_norms)],
            color=color, alpha=0.12,
        )

        axes[1].plot(
            layer_pcts, profile.k_values,
            label=profile.model_name, color=color, linewidth=1.8,
        )

    for ax in axes:
        ax.axvspan(0.4, 0.8, alpha=0.08, color="green")
        ax.axvline(0.4, color="green", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.axvline(0.8, color="green", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xlabel("Layer depth (fraction of total layers)", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    axes[0].set_title("Mean Residual Stream Norm  ‖h_l‖", fontsize=12)
    axes[0].set_ylabel("mean ‖h_l‖", fontsize=11)

    axes[1].set_title("Per-layer K_l = mean_norm_l / √d", fontsize=12)
    axes[1].set_ylabel("K_l", fontsize=11)

    fig.suptitle("Layer-wise Norm Profiles  (green band = 40–80% steering window)", fontsize=11)
    plt.tight_layout()
    _save_or_show(fig, output_path, show)


def plot_ceiling_sweep(
    results: List[CeilingProbeResult],
    model_name: str,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> None:
    """
    Bar chart of coherence across alpha values for a single layer sweep.
    Marks the empirical ceiling (first incoherent alpha).
    """
    alphas = [r.alpha for r in results]
    coherent = [1 if r.is_coherent else 0 for r in results]
    colors = ["steelblue" if c else "tomato" for c in coherent]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar([str(a) for a in alphas], coherent, color=colors, edgecolor="white", width=0.6)

    incoherent = [r.alpha for r in results if not r.is_coherent]
    if incoherent:
        ceiling = min(incoherent)
        ax.axvline(
            x=[str(a) for a in alphas].index(str(ceiling)) - 0.5,
            color="crimson", linestyle="--", linewidth=1.5, label=f"ceiling ≈ {ceiling}×K_l",
        )
        ax.legend(fontsize=9)

    k_l = results[0].k_l if results else 0
    ax.set_title(
        f"Ceiling Sweep — {model_name}  |  layer {results[0].layer_idx if results else '?'}"
        f"  |  K_l = {k_l:.4f}",
        fontsize=11,
    )
    ax.set_xlabel("Steering magnitude (α × K_l)", fontsize=11)
    ax.set_ylabel("Coherent (1) / Gibberish (0)", fontsize=11)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["gibberish", "coherent"])
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save_or_show(fig, output_path, show)


def plot_multi_layer_ceiling(
    sweep_results: Dict[int, List[CeilingProbeResult]],
    model_name: str,
    num_layers: int,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> None:
    """
    Heatmap: layers (y-axis) × alpha values (x-axis), colored by coherence.
    Shows at a glance which layers tolerate higher steering magnitudes.
    """
    layer_indices = sorted(sweep_results.keys())
    if not layer_indices:
        return

    alphas = [r.alpha for r in next(iter(sweep_results.values()))]
    matrix = np.array([
        [1 if r.is_coherent else 0 for r in sweep_results[l]]
        for l in layer_indices
    ], dtype=float)

    fig, ax = plt.subplots(figsize=(10, max(4, len(layer_indices) * 0.4)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, origin="lower")

    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"{a}×" for a in alphas], fontsize=9)
    ax.set_yticks(range(len(layer_indices)))
    ax.set_yticklabels(
        [f"L{l} ({l/num_layers:.0%})" for l in layer_indices], fontsize=8
    )
    ax.set_xlabel("Steering magnitude (α × K_l)", fontsize=11)
    ax.set_ylabel("Layer", fontsize=11)
    ax.set_title(f"Coherence Map — {model_name}  (green=coherent, red=gibberish)", fontsize=11)

    plt.colorbar(im, ax=ax, shrink=0.6)
    plt.tight_layout()
    _save_or_show(fig, output_path, show)


def plot_behavioral_consistency(
    bvec: "BehavioralVector",
    output_path: Optional[Path] = None,
    show: bool = False,
) -> None:
    """
    Two-panel figure: per-layer consistency score and raw diff norm.

    Consistency = mean cosine alignment of individual pair diffs to the
    aggregate direction. High consistency (>0.6) indicates a clean behavioral axis.
    """
    layer_pcts = [l / bvec.num_layers for l in range(bvec.num_layers)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].plot(layer_pcts, bvec.layer_consistency, color="steelblue", linewidth=1.8)
    axes[0].axhline(0.6, color="green", linestyle="--", linewidth=0.9, alpha=0.7, label="0.6 threshold")
    axes[0].axhline(0.0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    axes[0].axvspan(0.4, 0.8, alpha=0.08, color="green")
    axes[0].set_title(f"Behavioral Consistency — {bvec.behavior}", fontsize=12)
    axes[0].set_ylabel("Mean cosine alignment (pairs → aggregate)", fontsize=10)
    axes[0].set_ylim(-0.1, 1.05)
    axes[0].legend(fontsize=9)

    axes[1].plot(layer_pcts, bvec.layer_raw_norms, color="darkorange", linewidth=1.8)
    axes[1].axvspan(0.4, 0.8, alpha=0.08, color="green")
    axes[1].set_title("Raw Mean Diff Norm per Layer", fontsize=12)
    axes[1].set_ylabel("||mean_diff_l||", fontsize=10)

    for ax in axes:
        ax.set_xlabel("Layer depth", fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    fig.suptitle(
        f"{bvec.model_name}  |  {bvec.behavior}  |  {bvec.num_pairs} pairs"
        "  (green band = 40–80% steering window)",
        fontsize=10,
    )
    plt.tight_layout()
    _save_or_show(fig, output_path, show)


def plot_formula_validation(
    all_ceilings: Dict[str, Dict[int, Optional[float]]],
    profile: "NormProfile",
    model_name: str,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> None:
    """
    Formula validation plot: empirical ceiling alpha vs layer depth, per behavior.

    A flat line at α ≈ 1 means K_l perfectly normalizes norm growth across depth.
    Systematic slope reveals where the formula over- or under-estimates the ceiling.

    Second panel: K_l profile overlaid to show the raw norm variation being normalized.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10.colors
    num_layers = profile.num_layers

    for i, (behavior, ceilings) in enumerate(all_ceilings.items()):
        color = colors[i % len(colors)]
        measured_layers = [l for l, v in ceilings.items() if v is not None]
        measured_alphas = [ceilings[l] for l in measured_layers]
        uncapped_layers = [l for l, v in ceilings.items() if v is None]

        if measured_layers:
            xs = [l / num_layers for l in measured_layers]
            axes[0].plot(xs, measured_alphas, "o-", color=color, label=behavior,
                         linewidth=1.6, markersize=5)

        if uncapped_layers:
            xs_unc = [l / num_layers for l in uncapped_layers]
            axes[0].plot(xs_unc, [max(ceilings.values() or [0]) + 0.5] * len(xs_unc),
                         "^", color=color, alpha=0.5, markersize=6,
                         label=f"{behavior} (>max)" if i == 0 else "")

    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1.0,
                    alpha=0.6, label="α = 1 (formula prediction)")
    axes[0].axvspan(0.4, 0.8, alpha=0.06, color="green")
    axes[0].set_title(f"Empirical Ceiling Alpha vs Layer Depth\n{model_name}", fontsize=11)
    axes[0].set_xlabel("Layer depth", fontsize=11)
    axes[0].set_ylabel("Empirical ceiling (× K_l)", fontsize=11)
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].grid(True, alpha=0.25)
    axes[0].xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    axes[0].annotate(
        "flat line = formula holds universally",
        xy=(0.5, 0.04), xycoords="axes fraction",
        fontsize=8, ha="center", color="gray",
    )

    # Panel 2: K_l profile to show what's being normalized
    layer_pcts = [l / num_layers for l in range(num_layers)]
    axes[1].plot(layer_pcts, profile.k_values, color="steelblue", linewidth=1.8,
                 label=f"K_l = mean_norm / √d")
    axes[1].fill_between(
        layer_pcts,
        [max(0, k - profile.layer_std_norms[l] / (profile.hidden_dim ** 0.5))
         for l, k in enumerate(profile.k_values)],
        [k + profile.layer_std_norms[l] / (profile.hidden_dim ** 0.5)
         for l, k in enumerate(profile.k_values)],
        alpha=0.15, color="steelblue",
    )
    axes[1].axvspan(0.4, 0.8, alpha=0.06, color="green")
    axes[1].set_title(f"K_l Profile (what formula normalizes)\n{model_name}", fontsize=11)
    axes[1].set_xlabel("Layer depth", fontsize=11)
    axes[1].set_ylabel("K_l", fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.25)
    axes[1].xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    fig.suptitle(
        "Formula Validation: K_l = mean_norm_l / √d as universal coherence ceiling"
        "  (green band = 40–80% steering window)",
        fontsize=10,
    )
    plt.tight_layout()
    _save_or_show(fig, output_path, show)


def _save_or_show(fig: plt.Figure, output_path: Optional[Path], show: bool) -> None:
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Plot saved to %s", output_path)
    if show:
        plt.show()
    plt.close(fig)
