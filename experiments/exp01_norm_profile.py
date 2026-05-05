"""
Experiment 01 — Per-layer residual stream norm profiling.

For each model:
  1. Profile mean_norm_l and K_l = mean_norm_l / sqrt(d) across all layers.
  2. Optionally sweep alpha × K_l at representative layers to find the
     empirical coherence ceiling, validating the K_l interpretation.

Usage:
    # Profile only
    python experiments/exp01_norm_profile.py --model meta-llama/Llama-3.2-1B --model-name llama-3.2-1b

    # Profile + ceiling sweep at 60% depth
    python experiments/exp01_norm_profile.py \\
        --model meta-llama/Llama-3.2-1B --model-name llama-3.2-1b \\
        --run-ceiling-sweep

    # Multi-layer ceiling sweep at 30%, 50%, 70%, 90% depth
    python experiments/exp01_norm_profile.py \\
        --model google/gemma-2-2b --model-name gemma-2-2b \\
        --run-ceiling-sweep --sweep-layer-pcts 0.3 0.5 0.7 0.9
"""

import argparse
import json
import logging
import random
from pathlib import Path

import torch

from ksteer.profiler import CeilingSweeper, LayerNormProfiler
from ksteer.utils.model_utils import load_model
from ksteer.utils.plot_utils import plot_ceiling_sweep, plot_multi_layer_ceiling, plot_norm_profiles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Diverse prompts spanning domains to get representative activation statistics
PROFILE_PROMPTS: list[str] = [
    "The capital of France is",
    "In quantum mechanics, the uncertainty principle states",
    "To make a delicious pasta carbonara, you need",
    "The history of the Roman Empire spans many centuries and",
    "Machine learning models learn from data by",
    "Climate change is primarily caused by",
    "The human brain contains approximately 86 billion neurons and",
    "Shakespeare's most famous play, Hamlet, begins with",
    "The process of photosynthesis converts sunlight into",
    "To solve a quadratic equation, one can use the formula",
    "The largest ocean on Earth, the Pacific, covers",
    "In programming, a recursive function calls itself to",
    "The fundamental theorem of calculus connects",
    "Ancient Egyptian civilization was known for its pyramids and",
    "The speed of light in a vacuum is approximately",
    "Modern cryptography relies on mathematical problems that are",
    "Darwin's theory of evolution by natural selection proposes that",
    "To write a persuasive essay, you should begin by",
    "The immune system protects the body from pathogens by",
    "Newton's second law states that force equals",
    "The Amazon rainforest is home to",
    "In music theory, a chord consists of",
    "The French Revolution began in 1789 when",
    "Mitochondria are often called the powerhouse of the cell because",
    "The Pythagorean theorem states that in a right triangle",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp 01: Per-layer norm profiling + ceiling sweep")
    p.add_argument("--model", required=True, type=str, help="HuggingFace model ID or local path")
    p.add_argument("--model-name", default=None, type=str, help="Short tag for output files")
    p.add_argument("--device", default=None, type=str, help="Force device: cuda / cpu / mps")
    p.add_argument("--quantize", default=None, choices=["4bit", "8bit"],
                   help="bitsandbytes quantization for large models that exceed single-GPU VRAM")
    p.add_argument("--batch-size", default=4, type=int)
    p.add_argument("--max-length", default=256, type=int)
    p.add_argument("--output-dir", default="results/exp01", type=str)
    p.add_argument("--run-ceiling-sweep", action="store_true",
                   help="Run alpha × K_l sweep to empirically confirm the coherence ceiling")
    p.add_argument("--sweep-layer-pcts", nargs="+", type=float, default=[0.6],
                   help="Layer depths (as fractions) at which to run the ceiling sweep. Default: 0.6")
    p.add_argument("--sweep-prompt", default="Tell me something interesting about the history of science.",
                   type=str)
    p.add_argument("--sweep-max-tokens", default=80, type=int)
    p.add_argument("--seed", default=42, type=int)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_name = args.model_name or args.model.split("/")[-1]
    out = Path(args.output_dir) / model_name
    out.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(args.model, device=args.device, quantize=args.quantize)

    # ── Step 1a: profile all layers ──────────────────────────────────────
    logger.info("=== Step 1a: Layer norm profiling ===")
    profiler = LayerNormProfiler(model, tokenizer, model_name)
    profile = profiler.profile(
        texts=PROFILE_PROMPTS,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    profile_path = out / "norm_profile.json"
    with open(profile_path, "w") as f:
        json.dump(profile.to_dict(), f, indent=2)
    logger.info("Norm profile → %s", profile_path)

    start, end = profile.steering_window
    k_min, k_max = profile.window_k_range
    logger.info(
        "Summary: layers=%d  hidden=%d  "
        "steering_window=[%d, %d] (%.0f%%–%.0f%%)  "
        "K_l in window=[%.4f, %.4f]",
        profile.num_layers, profile.hidden_dim,
        start, end,
        100 * start / profile.num_layers, 100 * end / profile.num_layers,
        k_min, k_max,
    )

    plot_norm_profiles([profile], output_path=out / "norm_profile.png")

    # ── Step 1b: ceiling sweep (optional) ───────────────────────────────
    if not args.run_ceiling_sweep:
        logger.info("Skipping ceiling sweep (pass --run-ceiling-sweep to enable).")
        return

    logger.info("=== Step 1b: Ceiling sweep ===")
    sweep_layers = sorted({
        max(0, min(profile.num_layers - 1, int(pct * profile.num_layers)))
        for pct in args.sweep_layer_pcts
    })
    logger.info("Sweep layers: %s", sweep_layers)

    # Random unit vector as placeholder — real behavioral vectors come in Exp 02.
    # Purpose here is purely to confirm the K_l ceiling, not to test directionality.
    torch.manual_seed(args.seed)
    v_random = torch.randn(profile.hidden_dim)

    sweeper = CeilingSweeper(model, tokenizer, profile)
    all_sweep_results = sweeper.sweep_multiple_layers(
        prompt=args.sweep_prompt,
        steering_vector=v_random,
        layer_indices=sweep_layers,
        max_new_tokens=args.sweep_max_tokens,
    )

    # Save raw results
    sweep_data = {
        str(layer_idx): [r.to_dict() for r in results]
        for layer_idx, results in all_sweep_results.items()
    }
    sweep_path = out / "ceiling_sweep.json"
    with open(sweep_path, "w") as f:
        json.dump(sweep_data, f, indent=2)
    logger.info("Ceiling sweep → %s", sweep_path)

    # Per-layer plots
    for layer_idx, results in all_sweep_results.items():
        coherent_alphas = [r.alpha for r in results if r.is_coherent]
        incoherent_alphas = [r.alpha for r in results if not r.is_coherent]
        if incoherent_alphas:
            logger.info(
                "Layer %d (%.0f%%): ceiling confirmed at α ≈ %.2f × K_l",
                layer_idx, 100 * layer_idx / profile.num_layers, min(incoherent_alphas),
            )
        else:
            logger.info(
                "Layer %d (%.0f%%): coherent at all tested alphas (max=%.2f) — ceiling may be higher",
                layer_idx, 100 * layer_idx / profile.num_layers,
                max(r.alpha for r in results),
            )
        plot_ceiling_sweep(results, model_name, output_path=out / f"ceiling_sweep_L{layer_idx}.png")

    # Multi-layer heatmap if more than one layer was swept
    if len(sweep_layers) > 1:
        plot_multi_layer_ceiling(
            all_sweep_results, model_name, profile.num_layers,
            output_path=out / "ceiling_heatmap.png",
        )


if __name__ == "__main__":
    main()
