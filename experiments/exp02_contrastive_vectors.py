"""
Experiment 02 — Per-layer behavioral direction extraction.

Extracts contrastive steering vectors for each behavior in data/behaviors/ by
running both halves of each (positive, negative) pair through the model and
computing mean-pooled residual stream differences at every layer.

Output layout:
    results/exp02/{model_name}/{behavior}/
        vectors.npz          — (num_layers, hidden_dim) unit direction per layer
        vectors_meta.json    — consistency scores, raw norms, metadata

Usage:
    # Extract all behaviors for one model
    python experiments/exp02_contrastive_vectors.py \\
        --model meta-llama/Llama-3.2-1B --model-name llama-3.2-1b

    # Single behavior, custom data dir
    python experiments/exp02_contrastive_vectors.py \\
        --model meta-llama/Llama-3.2-1B --model-name llama-3.2-1b \\
        --behaviors sycophancy refusal

    # Skip behaviors already extracted
    python experiments/exp02_contrastive_vectors.py \\
        --model Qwen/Qwen2.5-7B --model-name qwen2.5-7b \\
        --skip-existing
"""

import argparse
import json
import logging
import random
from pathlib import Path

import torch

from ksteer.contrastive import ContrastiveExtractor, load_behavior_pairs
from ksteer.utils.model_utils import load_model
from ksteer.utils.plot_utils import plot_behavioral_consistency

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp 02: contrastive behavioral vector extraction")
    p.add_argument("--model", required=True, type=str, help="HuggingFace model ID or local path")
    p.add_argument("--model-name", default=None, type=str)
    p.add_argument("--device", default=None, type=str)
    p.add_argument("--max-length", default=256, type=int)
    p.add_argument("--data-dir", default="data/behaviors", type=str)
    p.add_argument("--output-dir", default="results/exp02", type=str)
    p.add_argument(
        "--behaviors", nargs="+", default=None,
        help="Behavior names to extract (default: all .jsonl files in data-dir)"
    )
    p.add_argument(
        "--skip-existing", action="store_true",
        help="Skip behaviors whose vectors.npz already exists"
    )
    p.add_argument("--seed", default=42, type=int)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_name = args.model_name or args.model.split("/")[-1]
    data_dir = Path(args.data_dir)
    out_root = Path(args.output_dir) / model_name

    # Resolve behaviors from data dir
    if args.behaviors:
        behavior_files = [data_dir / f"{b}.jsonl" for b in args.behaviors]
    else:
        behavior_files = sorted(data_dir.glob("*.jsonl"))

    if not behavior_files:
        raise FileNotFoundError(f"No behavior .jsonl files found in {data_dir}")

    # Filter to behaviors that still need extraction
    to_run = []
    for bf in behavior_files:
        behavior = bf.stem
        vec_path = out_root / behavior / "vectors.npz"
        if args.skip_existing and vec_path.exists():
            logger.info("Skipping %s — vectors already exist.", behavior)
        else:
            to_run.append(bf)

    if not to_run:
        logger.info("All behaviors already extracted. Done.")
        return

    model, tokenizer = load_model(args.model, device=args.device)
    extractor = ContrastiveExtractor(model, tokenizer, model_name)

    extracted = []
    for bf in to_run:
        behavior = bf.stem
        logger.info("=== Behavior: %s ===", behavior)

        pairs = load_behavior_pairs(bf)
        logger.info("  Loaded %d pairs from %s", len(pairs), bf)

        bvec = extractor.extract(pairs, behavior=behavior, max_length=args.max_length)

        out_dir = out_root / behavior
        bvec.save(out_dir)
        logger.info("  Saved vectors → %s", out_dir)

        # Steering window summary (40–80% depth)
        start = int(0.4 * bvec.num_layers)
        end = int(0.8 * bvec.num_layers)
        window_cons = bvec.layer_consistency[start:end]
        mean_cons = sum(window_cons) / len(window_cons)
        logger.info(
            "  Steering window [L%d–L%d] consistency: mean=%.3f  min=%.3f  max=%.3f",
            start, end, mean_cons, min(window_cons), max(window_cons),
        )

        plot_behavioral_consistency(
            bvec,
            output_path=out_dir / "consistency.png",
        )
        extracted.append(bvec)

    # Cross-behavior summary
    if extracted:
        summary = {
            bv.behavior: {
                "num_pairs": bv.num_pairs,
                "window_consistency_mean": float(
                    sum(bv.layer_consistency[int(0.4 * bv.num_layers): int(0.8 * bv.num_layers)])
                    / max(1, int(0.8 * bv.num_layers) - int(0.4 * bv.num_layers))
                ),
                "peak_consistency_layer": int(
                    max(range(bv.num_layers), key=lambda l: bv.layer_consistency[l])
                ),
                "peak_consistency": float(max(bv.layer_consistency)),
            }
            for bv in extracted
        }
        summary_path = out_root / "extraction_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Extraction summary → %s", summary_path)

        logger.info("\n%-20s  %5s  %12s  %12s  %12s",
                    "behavior", "pairs", "win_cons", "peak_cons", "peak_layer")
        logger.info("─" * 68)
        for bv, stats in zip(extracted, summary.values()):
            logger.info("%-20s  %5d  %12.3f  %12.3f  %12d",
                        bv.behavior, bv.num_pairs,
                        stats["window_consistency_mean"],
                        stats["peak_consistency"],
                        stats["peak_consistency_layer"])


if __name__ == "__main__":
    main()
