"""
Run Exp 01 (norm profiling + optional ceiling sweep) across all models in configs/models.yaml.

Usage:
    # Run all models sequentially (profile only)
    python experiments/run_all.py

    # Run only small-tier models
    python experiments/run_all.py --tiers small

    # Run specific families
    python experiments/run_all.py --families llama gemma2

    # Run with ceiling sweep, skip already-completed models
    python experiments/run_all.py --run-ceiling-sweep --skip-existing

    # Run only specific model keys
    python experiments/run_all.py --models llama-3.2-1b qwen2.5-7b gemma-2-2b mistral-7b

    # Dry-run to see what would execute
    python experiments/run_all.py --tiers small medium --dry-run
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

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
    p = argparse.ArgumentParser(description="Run Exp 01 across model configs")
    p.add_argument("--config", default=str(CONFIG_PATH), type=str)
    p.add_argument("--models", nargs="+", default=None,
                   help="Specific model keys to run (from config). Runs all if omitted.")
    p.add_argument("--families", nargs="+", default=None,
                   help="Filter by family: llama mistral mixtral qwen2 gemma gemma2")
    p.add_argument("--tiers", nargs="+", default=None,
                   help="Filter by tier: small medium large")
    p.add_argument("--device", default=None, type=str)
    p.add_argument("--batch-size", default=4, type=int)
    p.add_argument("--max-length", default=256, type=int)
    p.add_argument("--output-dir", default="results/exp01", type=str)
    p.add_argument("--run-ceiling-sweep", action="store_true")
    p.add_argument("--sweep-layer-pcts", nargs="+", type=float, default=[0.3, 0.5, 0.7, 0.9])
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip models whose norm_profile.json already exists")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would run without executing")
    return p.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def select_models(config: dict, args: argparse.Namespace) -> list[tuple[str, dict]]:
    """Return list of (model_key, model_cfg) to run, after applying all filters."""
    models = list(config["models"].items())

    if args.models:
        models = [(k, v) for k, v in models if k in args.models]
    if args.families:
        models = [(k, v) for k, v in models if v.get("family") in args.families]
    if args.tiers:
        models = [(k, v) for k, v in models if v.get("tier") in args.tiers]

    return models


def already_done(model_key: str, output_dir: str, need_ceiling: bool = False) -> bool:
    base = Path(output_dir) / model_key
    profile_exists = (base / "norm_profile.json").exists()
    if not profile_exists:
        return False
    if need_ceiling:
        # Skip only if ceiling sweep results also exist
        return (base / "ceiling_sweep.json").exists()
    return True


def build_command(model_key: str, model_cfg: dict, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "experiments" / "exp01_norm_profile.py"),
        "--model", model_cfg["model_id"],
        "--model-name", model_key,
        "--batch-size", str(args.batch_size),
        "--max-length", str(args.max_length),
        "--output-dir", args.output_dir,
    ]
    if args.device:
        cmd += ["--device", args.device]
    if args.run_ceiling_sweep:
        cmd += ["--run-ceiling-sweep"]
        cmd += ["--sweep-layer-pcts"] + [str(p) for p in args.sweep_layer_pcts]
    return cmd


def run_model(model_key: str, model_cfg: dict, args: argparse.Namespace) -> bool:
    """Run exp01 for one model. Returns True on success."""
    cmd = build_command(model_key, model_cfg, args)
    family = model_cfg.get("family", "?")
    tier = model_cfg.get("tier", "?")
    notes = model_cfg.get("notes", "")

    logger.info("━━━ %s  [family=%s  tier=%s]%s", model_key, family, tier,
                f"  — {notes}" if notes else "")
    logger.info("CMD: %s", " ".join(cmd))

    start = time.time()
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    elapsed = time.time() - start

    if result.returncode == 0:
        logger.info("✓ %s completed in %.1fs", model_key, elapsed)
        return True
    else:
        logger.error("✗ %s failed (exit %d) after %.1fs", model_key, result.returncode, elapsed)
        return False


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    selected = select_models(config, args)

    if not selected:
        logger.error("No models matched the given filters.")
        sys.exit(1)

    logger.info("Selected %d model(s):", len(selected))
    for key, cfg in selected:
        skip = args.skip_existing and already_done(key, args.output_dir, need_ceiling=args.run_ceiling_sweep)
        status = "SKIP (exists)" if skip else f"tier={cfg.get('tier','?')}  family={cfg.get('family','?')}"
        logger.info("  %-25s %s", key, status)

    if args.dry_run:
        logger.info("Dry run — no models executed.")
        return

    results: dict[str, bool] = {}
    total_start = time.time()

    for model_key, model_cfg in selected:
        if args.skip_existing and already_done(model_key, args.output_dir, need_ceiling=args.run_ceiling_sweep):
            logger.info("Skipping %s — results already exist.", model_key)
            results[model_key] = True
            continue

        success = run_model(model_key, model_cfg, args)
        results[model_key] = success

        if not success:
            logger.warning("Continuing to next model despite failure.")

    # ── Summary ─────────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    passed = [k for k, ok in results.items() if ok]
    failed = [k for k, ok in results.items() if not ok]

    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("Run complete in %.1fs  |  ✓ %d passed  ✗ %d failed",
                total_elapsed, len(passed), len(failed))
    if failed:
        logger.error("Failed: %s", ", ".join(failed))
    if passed:
        logger.info("Passed: %s", ", ".join(passed))

    # Write run summary
    summary_path = Path(args.output_dir) / "run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({
            "passed": passed,
            "failed": failed,
            "total_elapsed_s": round(total_elapsed, 1),
            "args": vars(args),
        }, f, indent=2)
    logger.info("Summary → %s", summary_path)

    # Trigger cross-model comparison plot if at least 2 models completed
    if len(passed) >= 2:
        logger.info("Running cross-model comparison plot...")
        compare_cmd = [
            sys.executable,
            str(REPO_ROOT / "experiments" / "compare_profiles.py"),
            "--results-dir", args.output_dir,
        ]
        subprocess.run(compare_cmd, cwd=str(REPO_ROOT))

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
