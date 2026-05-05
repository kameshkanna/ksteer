"""
Batch runner for Exp 01 and/or Exp 02 across all models in configs/models.yaml.

Each model runs in its own subprocess with aggressive GPU cleanup between runs.

Usage:
    # Exp 01 only — profile + ceiling sweep, small + medium
    python experiments/run_all.py --tiers small medium --run-ceiling-sweep --sweep-layer-pcts 0.3 0.5 0.7 0.9 --skip-existing

    # Exp 02 only — contrastive vector extraction, small + medium
    python experiments/run_all.py --tiers small medium --run-exp02 --skip-existing

    # Both experiments in sequence for each model
    python experiments/run_all.py --tiers small medium --run-ceiling-sweep --sweep-layer-pcts 0.3 0.5 0.7 0.9 --run-exp02 --skip-existing

    # Dry-run to see what would execute
    python experiments/run_all.py --tiers small medium --run-exp02 --dry-run
"""

import argparse
import gc
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "models.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch runner for Exp 01 and/or Exp 02")
    p.add_argument("--config", default=str(CONFIG_PATH), type=str)
    p.add_argument("--models", nargs="+", default=None,
                   help="Specific model keys to run. Runs all if omitted.")
    p.add_argument("--families", nargs="+", default=None,
                   help="Filter by family: llama mistral mixtral qwen2 gemma gemma2")
    p.add_argument("--tiers", nargs="+", default=None,
                   help="Filter by tier: small medium large")
    p.add_argument("--device", default=None, type=str)
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip models whose outputs already exist")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would run without executing")

    # Exp 01 flags
    p.add_argument("--batch-size", default=4, type=int)
    p.add_argument("--max-length", default=256, type=int)
    p.add_argument("--exp01-output-dir", default="results/exp01", type=str)
    p.add_argument("--run-ceiling-sweep", action="store_true")
    p.add_argument("--sweep-layer-pcts", nargs="+", type=float, default=[0.3, 0.5, 0.7, 0.9])

    # Exp 02 flags
    p.add_argument("--run-exp02", action="store_true",
                   help="Run Exp 02 (contrastive vector extraction) for each model")
    p.add_argument("--exp02-output-dir", default="results/exp02", type=str)
    p.add_argument("--exp02-data-dir", default="data/behaviors", type=str)
    p.add_argument("--behaviors", nargs="+", default=None,
                   help="Behaviors to extract in Exp 02 (default: all in data-dir)")

    return p.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def select_models(config: dict, args: argparse.Namespace) -> list[tuple[str, dict]]:
    models = list(config["models"].items())
    if args.models:
        models = [(k, v) for k, v in models if k in args.models]
    if args.families:
        models = [(k, v) for k, v in models if v.get("family") in args.families]
    if args.tiers:
        models = [(k, v) for k, v in models if v.get("tier") in args.tiers]
    return models


def exp01_done(model_key: str, args: argparse.Namespace) -> bool:
    base = Path(args.exp01_output_dir) / model_key
    if not (base / "norm_profile.json").exists():
        return False
    if args.run_ceiling_sweep:
        return (base / "ceiling_sweep.json").exists()
    return True


def exp02_done(model_key: str, args: argparse.Namespace) -> bool:
    """All requested behaviors must have vectors.npz for the model to count as done."""
    base = Path(args.exp02_output_dir) / model_key
    if not base.exists():
        return False
    if args.behaviors:
        behaviors = args.behaviors
    else:
        behaviors = [p.stem for p in Path(args.exp02_data_dir).glob("*.jsonl")]
    return all((base / b / "vectors.npz").exists() for b in behaviors)


def build_exp01_cmd(model_key: str, model_cfg: dict, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "experiments" / "exp01_norm_profile.py"),
        "--model", model_cfg["model_id"],
        "--model-name", model_key,
        "--batch-size", str(args.batch_size),
        "--max-length", str(args.max_length),
        "--output-dir", args.exp01_output_dir,
    ]
    if args.device:
        cmd += ["--device", args.device]
    if args.run_ceiling_sweep:
        cmd += ["--run-ceiling-sweep", "--sweep-layer-pcts"] + [str(p) for p in args.sweep_layer_pcts]
    return cmd


def build_exp02_cmd(model_key: str, model_cfg: dict, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "experiments" / "exp02_contrastive_vectors.py"),
        "--model", model_cfg["model_id"],
        "--model-name", model_key,
        "--output-dir", args.exp02_output_dir,
        "--data-dir", args.exp02_data_dir,
    ]
    if args.device:
        cmd += ["--device", args.device]
    if args.behaviors:
        cmd += ["--behaviors"] + args.behaviors
    if args.skip_existing:
        cmd += ["--skip-existing"]
    return cmd


def purge_gpu_memory() -> None:
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        mem_free, mem_total = torch.cuda.mem_get_info()
        logger.info(
            "GPU memory after cleanup: %.1f GB free / %.1f GB total",
            mem_free / 1e9, mem_total / 1e9,
        )


def run_subprocess(cmd: list[str], label: str) -> bool:
    logger.info("CMD [%s]: %s", label, " ".join(cmd))
    start = time.time()
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    elapsed = time.time() - start
    if result.returncode == 0:
        logger.info("✓ %s completed in %.1fs", label, elapsed)
    else:
        logger.error("✗ %s failed (exit %d) after %.1fs", label, result.returncode, elapsed)
    purge_gpu_memory()
    return result.returncode == 0


def main() -> None:
    args = parse_args()

    if not args.run_ceiling_sweep and not args.run_exp02:
        # Default: run exp01 profile only
        pass

    config = load_config(args.config)
    selected = select_models(config, args)

    if not selected:
        logger.error("No models matched the given filters.")
        sys.exit(1)

    run_exp01 = True  # always profile unless everything is done
    run_exp02 = args.run_exp02

    logger.info("Selected %d model(s):", len(selected))
    for key, cfg in selected:
        e1 = "SKIP" if (args.skip_existing and exp01_done(key, args)) else "exp01"
        e2 = ("SKIP" if (args.skip_existing and exp02_done(key, args)) else "exp02") if run_exp02 else "—"
        logger.info("  %-25s  exp01=%-6s  exp02=%s", key, e1, e2)

    if args.dry_run:
        logger.info("Dry run — no models executed.")
        return

    results: dict[str, dict[str, bool]] = {}
    total_start = time.time()

    for model_key, model_cfg in selected:
        results[model_key] = {}
        family = model_cfg.get("family", "?")
        tier = model_cfg.get("tier", "?")
        logger.info("━━━ %s  [family=%s  tier=%s]", model_key, family, tier)

        # ── Exp 01 ──────────────────────────────────────────────────────────
        if args.skip_existing and exp01_done(model_key, args):
            logger.info("  exp01: skipping %s — results exist.", model_key)
            results[model_key]["exp01"] = True
        else:
            cmd = build_exp01_cmd(model_key, model_cfg, args)
            ok = run_subprocess(cmd, f"exp01/{model_key}")
            results[model_key]["exp01"] = ok
            if not ok:
                logger.warning("  exp01 failed for %s — skipping exp02 for this model.", model_key)
                results[model_key]["exp02"] = False
                continue

        # ── Exp 02 ──────────────────────────────────────────────────────────
        if run_exp02:
            if args.skip_existing and exp02_done(model_key, args):
                logger.info("  exp02: skipping %s — vectors exist.", model_key)
                results[model_key]["exp02"] = True
            else:
                cmd = build_exp02_cmd(model_key, model_cfg, args)
                results[model_key]["exp02"] = run_subprocess(cmd, f"exp02/{model_key}")

    # ── Summary ─────────────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("Run complete in %.1fs", total_elapsed)
    logger.info("%-25s  %6s  %6s", "model", "exp01", "exp02")
    logger.info("─" * 44)
    for key, res in results.items():
        e1 = "✓" if res.get("exp01") else "✗"
        e2 = "✓" if res.get("exp02") else ("—" if not run_exp02 else "✗")
        logger.info("  %-25s  %6s  %6s", key, e1, e2)

    failed = [k for k, r in results.items() if not all(r.values())]
    if failed:
        logger.error("Failed: %s", ", ".join(failed))

    summary_path = Path(args.exp01_output_dir) / "run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({"results": results, "total_elapsed_s": round(total_elapsed, 1)}, f, indent=2)
    logger.info("Summary → %s", summary_path)

    # Cross-model comparison plot for exp01
    exp01_passed = [k for k, r in results.items() if r.get("exp01")]
    if len(exp01_passed) >= 2:
        subprocess.run([
            sys.executable,
            str(REPO_ROOT / "experiments" / "compare_profiles.py"),
            "--results-dir", args.exp01_output_dir,
        ], cwd=str(REPO_ROOT))

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
