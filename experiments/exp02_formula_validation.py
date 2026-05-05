"""
Experiment 02b — Formula validation: does K_l = mean_norm_l / sqrt(d) hold across layer depths?

Uses real contrastive behavioral vectors (from Exp 02) to sweep alpha × K_l at
multiple layer depths. If the formula is a universal ceiling, the empirical
break-point alpha should be approximately constant across the full layer range.

Systematic drift reveals:
  - Decreasing alpha_break at higher layers → late layers are MORE sensitive
    than K_l predicts (formula overestimates the ceiling there).
  - Increasing alpha_break at higher layers → late layers are less sensitive
    (formula underestimates, or a correction factor is needed at low-norm layers).
  - ~Flat curve → K_l correctly normalizes the depth-dependent norm growth.

Output layout:
    results/exp02/{model_name}/formula_validation/
        {behavior}_validation.json   — per-layer alpha sweep results
        formula_validation.png       — summary plot across all behaviors
        formula_summary.json         — empirical ceiling alphas + formula accuracy score

Usage:
    python experiments/exp02_formula_validation.py \\
        --model meta-llama/Llama-3.2-1B \\
        --model-name llama-3.2-1b \\
        --exp01-dir results/exp01 \\
        --exp02-dir results/exp02

    # Fewer sweep layers, faster run
    python experiments/exp02_formula_validation.py \\
        --model meta-llama/Llama-3.2-1B --model-name llama-3.2-1b \\
        --sweep-layer-pcts 0.2 0.4 0.6 0.8 \\
        --behaviors sycophancy refusal
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List

import torch

from ksteer.contrastive import BehavioralVector
from ksteer.profiler import CeilingSweeper, NormProfile
from ksteer.utils.model_utils import load_model
from ksteer.utils.plot_utils import plot_formula_validation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp 02b: formula validation across layer depths")
    p.add_argument("--model", required=True, type=str)
    p.add_argument("--model-name", default=None, type=str)
    p.add_argument("--device", default=None, type=str)
    p.add_argument("--exp01-dir", default="results/exp01", type=str,
                   help="Directory containing Exp 01 norm profile results")
    p.add_argument("--exp02-dir", default="results/exp02", type=str,
                   help="Directory containing Exp 02 contrastive vector results")
    p.add_argument("--output-dir", default=None, type=str,
                   help="Output dir (default: {exp02-dir}/{model-name}/formula_validation)")
    p.add_argument(
        "--sweep-layer-pcts", nargs="+", type=float,
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        help="Layer depths as fractions to test the formula at",
    )
    p.add_argument(
        "--alphas", nargs="+", type=float,
        default=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0],
        help="Alpha multipliers of K_l to sweep",
    )
    p.add_argument(
        "--behaviors", nargs="+", default=None,
        help="Specific behaviors to validate (default: all found in exp02-dir)",
    )
    p.add_argument("--sweep-prompt", default="Tell me something interesting about the history of science.", type=str)
    p.add_argument("--sweep-max-tokens", default=60, type=int)
    p.add_argument("--seed", default=42, type=int)
    return p.parse_args()


def load_norm_profile(exp01_dir: Path, model_name: str) -> NormProfile:
    profile_path = exp01_dir / model_name / "norm_profile.json"
    if not profile_path.exists():
        raise FileNotFoundError(
            f"Norm profile not found at {profile_path}. Run Exp 01 first."
        )
    with open(profile_path) as f:
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


def find_behaviors(exp02_model_dir: Path, requested: List[str] | None) -> List[str]:
    if requested:
        return requested
    return sorted(
        p.name for p in exp02_model_dir.iterdir()
        if p.is_dir() and (p / "vectors.npz").exists()
    )


def empirical_ceiling_alpha(results) -> float | None:
    """First alpha where coherence breaks, or None if coherent at all tested alphas."""
    for r in results:
        if not r.is_coherent:
            return r.alpha
    return None


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_name = args.model_name or args.model.split("/")[-1]
    exp01_dir = Path(args.exp01_dir)
    exp02_dir = Path(args.exp02_dir)
    exp02_model_dir = exp02_dir / model_name

    out_dir = Path(args.output_dir) if args.output_dir else exp02_model_dir / "formula_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load prerequisites
    profile = load_norm_profile(exp01_dir, model_name)
    logger.info("Loaded norm profile: %d layers, hidden_dim=%d", profile.num_layers, profile.hidden_dim)

    behaviors = find_behaviors(exp02_model_dir, args.behaviors)
    if not behaviors:
        raise FileNotFoundError(f"No extracted behaviors found in {exp02_model_dir}. Run Exp 02 first.")
    logger.info("Behaviors to validate: %s", behaviors)

    # Resolve sweep layers
    sweep_layers = sorted({
        max(0, min(profile.num_layers - 1, int(pct * profile.num_layers)))
        for pct in args.sweep_layer_pcts
    })
    logger.info("Sweep layers: %s", sweep_layers)

    model, tokenizer = load_model(args.model, device=args.device)
    sweeper = CeilingSweeper(model, tokenizer, profile)

    # behavior → layer_idx → empirical_ceiling_alpha
    all_ceilings: Dict[str, Dict[int, float | None]] = {}
    all_raw_results: Dict[str, Dict[str, list]] = {}

    for behavior in behaviors:
        bvec = BehavioralVector.load(exp02_model_dir / behavior)
        logger.info("=== Validating formula: behavior=%s ===", behavior)

        behavior_ceilings: Dict[int, float | None] = {}
        behavior_raw: Dict[str, list] = {}

        for layer_idx in sweep_layers:
            v = bvec.get_vector(layer_idx)
            results = sweeper.sweep(
                prompt=args.sweep_prompt,
                steering_vector=v,
                layer_idx=layer_idx,
                alphas=args.alphas,
                max_new_tokens=args.sweep_max_tokens,
            )
            ceiling = empirical_ceiling_alpha(results)
            behavior_ceilings[layer_idx] = ceiling

            layer_pct = layer_idx / profile.num_layers
            k_l = profile.k_values[layer_idx]
            if ceiling is not None:
                logger.info(
                    "  L%d (%.0f%%)  K_l=%.4f  ceiling=%.2f×K_l  raw_ceiling_norm=%.4f",
                    layer_idx, 100 * layer_pct, k_l, ceiling, ceiling * k_l,
                )
            else:
                logger.info(
                    "  L%d (%.0f%%)  K_l=%.4f  ceiling>%.2f×K_l (all coherent)",
                    layer_idx, 100 * layer_pct, k_l, max(args.alphas),
                )

            behavior_raw[str(layer_idx)] = [r.to_dict() for r in results]

        all_ceilings[behavior] = behavior_ceilings

        # Save per-behavior raw sweep data
        raw_path = out_dir / f"{behavior}_validation.json"
        with open(raw_path, "w") as f:
            json.dump(behavior_raw, f, indent=2)
        logger.info("  Raw sweep → %s", raw_path)

    # Compute formula accuracy metrics
    # For each behavior: std of ceiling alpha across layers (lower = more uniform = better formula)
    formula_summary: Dict[str, dict] = {}
    for behavior, ceilings in all_ceilings.items():
        measured = {l: v for l, v in ceilings.items() if v is not None}
        if len(measured) < 2:
            formula_summary[behavior] = {
                "ceiling_by_layer": {
                    str(l): v for l, v in ceilings.items()
                },
                "formula_accuracy": None,
                "note": "Too few measured ceilings for accuracy computation",
            }
            continue

        alpha_values = list(measured.values())
        mean_alpha = sum(alpha_values) / len(alpha_values)
        variance = sum((a - mean_alpha) ** 2 for a in alpha_values) / len(alpha_values)
        std_alpha = variance ** 0.5
        cv = std_alpha / mean_alpha if mean_alpha > 0 else float("inf")

        formula_summary[behavior] = {
            "ceiling_by_layer": {str(l): v for l, v in ceilings.items()},
            "mean_ceiling_alpha": mean_alpha,
            "std_ceiling_alpha": std_alpha,
            "coefficient_of_variation": cv,
            # CV < 0.15 → formula holds; 0.15–0.3 → partial; > 0.3 → formula needs correction
            "formula_accuracy": "strong" if cv < 0.15 else "partial" if cv < 0.30 else "weak",
        }

        layer_pcts = [l / profile.num_layers for l in ceilings.keys()]
        logger.info(
            "  [%s] mean_ceiling=%.2f  std=%.2f  CV=%.2f  → %s",
            behavior,
            mean_alpha,
            std_alpha,
            cv,
            formula_summary[behavior]["formula_accuracy"],
        )

    summary_path = out_dir / "formula_summary.json"
    with open(summary_path, "w") as f:
        json.dump(formula_summary, f, indent=2)
    logger.info("Formula summary → %s", summary_path)

    # Summary table
    logger.info("\n%-20s  %10s  %8s  %8s  %12s",
                "behavior", "mean_alpha", "std", "CV", "formula")
    logger.info("─" * 62)
    for beh, stats in formula_summary.items():
        if stats.get("formula_accuracy") is None:
            logger.info("%-20s  (insufficient data)", beh)
        else:
            logger.info(
                "%-20s  %10.3f  %8.3f  %8.3f  %12s",
                beh,
                stats["mean_ceiling_alpha"],
                stats["std_ceiling_alpha"],
                stats["coefficient_of_variation"],
                stats["formula_accuracy"],
            )

    plot_formula_validation(
        all_ceilings=all_ceilings,
        profile=profile,
        model_name=model_name,
        output_path=out_dir / "formula_validation.png",
    )


if __name__ == "__main__":
    main()
