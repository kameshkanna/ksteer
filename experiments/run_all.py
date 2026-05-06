"""
Batch runner — orchestrates Exp 01, Exp 02, and formula validation across all
models in configs/models.yaml. Each model runs in its own subprocess with
aggressive GPU cleanup between runs.

By default (no experiment flags) only Exp 01 profiling runs.
Add flags to include additional experiments for each model.

Single-model equivalents:
    python experiments/exp01_norm_profile.py --model <id> --model-name <name>
    python experiments/exp02_contrastive_vectors.py --model <id> --model-name <name>
    python experiments/exp02_formula_validation.py --model <id> --model-name <name>

Batch usage:
    python experiments/run_all.py --tiers small medium --skip-existing
    python experiments/run_all.py --tiers small medium --run-ceiling-sweep --sweep-layer-pcts 0.4 0.5 0.6 0.7 0.8 --skip-existing
    python experiments/run_all.py --tiers small medium --run-exp02 --skip-existing
    python experiments/run_all.py --tiers small medium --run-exp02 --run-formula-validation --skip-existing
    python experiments/run_all.py --tiers small medium --run-ceiling-sweep --run-exp02 --run-formula-validation --skip-existing
    python experiments/run_all.py --tiers small medium --dry-run
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
    p = argparse.ArgumentParser(description="Batch runner for ksteer experiments")

    # ── Model selection ──────────────────────────────────────────────────────
    p.add_argument("--config", default=str(CONFIG_PATH))
    p.add_argument("--models", nargs="+", default=None,
                   help="Specific model keys from config. Runs all if omitted.")
    p.add_argument("--families", nargs="+", default=None,
                   help="Filter by family: llama qwen2 gemma2")
    p.add_argument("--tiers", nargs="+", default=None,
                   help="Filter by tier: small medium large")

    # ── Output ───────────────────────────────────────────────────────────────
    p.add_argument("--output-dir", default="results",
                   help="Base output directory. Exp 01 → {output-dir}/exp01, Exp 02 → {output-dir}/exp02")
    p.add_argument("--device", default=None)

    # ── Exp 01 flags ─────────────────────────────────────────────────────────
    p.add_argument("--batch-size", default=4, type=int)
    p.add_argument("--max-length", default=256, type=int)
    p.add_argument("--run-ceiling-sweep", action="store_true",
                   help="Run alpha×K_l ceiling sweep during Exp 01")
    p.add_argument("--sweep-layer-pcts", nargs="+", type=float,
                   default=[0.4, 0.5, 0.6, 0.7, 0.8],
                   help="Layer depths for Exp 01 ceiling sweep (default: 40-80%% steering window)")

    # ── Exp 02 flags ─────────────────────────────────────────────────────────
    p.add_argument("--run-exp02", action="store_true",
                   help="Run Exp 02 contrastive vector extraction after Exp 01")
    p.add_argument("--data-dir", default="data/behaviors",
                   help="Behavior JSONL files directory (passed to Exp 02)")
    p.add_argument("--behaviors", nargs="+", default=None,
                   help="Specific behaviors to extract (default: all in data-dir)")

    # ── Formula validation flags ──────────────────────────────────────────────
    p.add_argument("--run-formula-validation", action="store_true",
                   help="Run Exp 02b formula validation after Exp 01 + Exp 02")
    p.add_argument("--val-bisect-lo", type=float, default=0.05,
                   help="Bisection lower bound alpha for formula validation (default: 0.05)")
    p.add_argument("--val-bisect-hi", type=float, default=3.0,
                   help="Bisection upper bound alpha for formula validation (default: 3.0)")
    p.add_argument("--val-bisect-tol", type=float, default=0.05,
                   help="Bisection stopping tolerance for formula validation (default: 0.05)")
    p.add_argument("--val-sweep-layer-pcts", nargs="+", type=float,
                   default=[0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
                   help="Layer depth fractions for formula validation sweep (default: 40-80%% steering window)")

    # ── Run control ──────────────────────────────────────────────────────────
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip models whose outputs already exist")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would run without executing")

    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
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


def exp01_done(model_key: str, exp01_dir: Path, need_ceiling: bool) -> bool:
    base = exp01_dir / model_key
    if not (base / "norm_profile.json").exists():
        return False
    if need_ceiling:
        return (base / "ceiling_sweep.json").exists()
    return True


def exp02_done(model_key: str, exp02_dir: Path, data_dir: Path, behaviors: list[str] | None) -> bool:
    base = exp02_dir / model_key
    if not base.exists():
        return False
    targets = behaviors if behaviors else [p.stem for p in data_dir.glob("*.jsonl")]
    return all((base / b / "vectors.npz").exists() for b in targets)


def formula_val_done(model_key: str, exp02_dir: Path) -> bool:
    return (exp02_dir / model_key / "formula_validation" / "formula_summary.json").exists()


def build_exp01_cmd(model_key: str, model_cfg: dict, args: argparse.Namespace, exp01_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "experiments" / "exp01_norm_profile.py"),
        "--model", model_cfg["model_id"],
        "--model-name", model_key,
        "--batch-size", str(args.batch_size),
        "--max-length", str(args.max_length),
        "--output-dir", str(exp01_dir),
    ]
    if args.device:
        cmd += ["--device", args.device]
    if args.run_ceiling_sweep:
        cmd += ["--run-ceiling-sweep", "--sweep-layer-pcts"] + [str(p) for p in args.sweep_layer_pcts]
    return cmd


def build_exp02_cmd(model_key: str, model_cfg: dict, args: argparse.Namespace, exp02_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "experiments" / "exp02_contrastive_vectors.py"),
        "--model", model_cfg["model_id"],
        "--model-name", model_key,
        "--output-dir", str(exp02_dir),
        "--data-dir", args.data_dir,
        "--max-length", str(args.max_length),
    ]
    if args.device:
        cmd += ["--device", args.device]
    if args.behaviors:
        cmd += ["--behaviors"] + args.behaviors
    if args.skip_existing:
        cmd += ["--skip-existing"]
    return cmd


def build_formula_val_cmd(model_key: str, model_cfg: dict, args: argparse.Namespace, exp01_dir: Path, exp02_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "experiments" / "exp02_formula_validation.py"),
        "--model", model_cfg["model_id"],
        "--model-name", model_key,
        "--exp01-dir", str(exp01_dir),
        "--exp02-dir", str(exp02_dir),
        "--bisect-lo", str(args.val_bisect_lo),
        "--bisect-hi", str(args.val_bisect_hi),
        "--bisect-tol", str(args.val_bisect_tol),
        "--sweep-layer-pcts",
    ] + [str(p) for p in args.val_sweep_layer_pcts]
    if args.device:
        cmd += ["--device", args.device]
    if args.behaviors:
        cmd += ["--behaviors"] + args.behaviors
    return cmd


def purge_gpu_memory() -> None:
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        free, total = torch.cuda.mem_get_info()
        logger.info("GPU: %.1f GB free / %.1f GB total", free / 1e9, total / 1e9)


def run_subprocess(cmd: list[str], label: str) -> bool:
    logger.info("RUN [%s]: %s", label, " ".join(cmd))
    start = time.time()
    ok = subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode == 0
    logger.info("%s %s (%.1fs)", "✓" if ok else "✗", label, time.time() - start)
    purge_gpu_memory()
    return ok


def main() -> None:
    args = parse_args()

    base_dir = Path(args.output_dir)
    exp01_dir = base_dir / "exp01"
    exp02_dir = base_dir / "exp02"

    config = load_config(args.config)
    selected = select_models(config, args)
    if not selected:
        logger.error("No models matched the given filters.")
        sys.exit(1)

    data_dir = Path(args.data_dir)
    do_exp02 = args.run_exp02 or args.run_formula_validation
    do_val = args.run_formula_validation

    # ── Print plan ───────────────────────────────────────────────────────────
    logger.info("Selected %d model(s):", len(selected))
    logger.info("  %-25s  %6s  %6s  %10s", "model", "exp01", "exp02", "val")
    logger.info("  " + "─" * 52)
    for key, cfg in selected:
        e1 = "SKIP" if (args.skip_existing and exp01_done(key, exp01_dir, args.run_ceiling_sweep)) else "run"
        e2 = ("SKIP" if (args.skip_existing and exp02_done(key, exp02_dir, data_dir, args.behaviors)) else "run") if do_exp02 else "—"
        ev = ("SKIP" if (args.skip_existing and formula_val_done(key, exp02_dir)) else "run") if do_val else "—"
        logger.info("  %-25s  %6s  %6s  %10s", key, e1, e2, ev)

    if args.dry_run:
        logger.info("Dry run — nothing executed.")
        return

    # ── Run ──────────────────────────────────────────────────────────────────
    results: dict[str, dict[str, bool]] = {}
    total_start = time.time()

    for model_key, model_cfg in selected:
        results[model_key] = {}
        logger.info("━━━ %s  [family=%s  tier=%s]",
                    model_key, model_cfg.get("family", "?"), model_cfg.get("tier", "?"))

        # Exp 01
        if args.skip_existing and exp01_done(model_key, exp01_dir, args.run_ceiling_sweep):
            logger.info("  exp01: exists, skipping.")
            results[model_key]["exp01"] = True
        else:
            ok = run_subprocess(build_exp01_cmd(model_key, model_cfg, args, exp01_dir), f"exp01/{model_key}")
            results[model_key]["exp01"] = ok
            if not ok:
                logger.warning("  exp01 failed — skipping remaining experiments for %s.", model_key)
                if do_exp02:
                    results[model_key]["exp02"] = False
                if do_val:
                    results[model_key]["val"] = False
                continue

        # Exp 02
        if do_exp02:
            if args.skip_existing and exp02_done(model_key, exp02_dir, data_dir, args.behaviors):
                logger.info("  exp02: exists, skipping.")
                results[model_key]["exp02"] = True
            else:
                ok = run_subprocess(build_exp02_cmd(model_key, model_cfg, args, exp02_dir), f"exp02/{model_key}")
                results[model_key]["exp02"] = ok
                if not ok and do_val:
                    logger.warning("  exp02 failed — skipping validation for %s.", model_key)
                    results[model_key]["val"] = False
                    continue

        # Formula validation
        if do_val:
            if args.skip_existing and formula_val_done(model_key, exp02_dir):
                logger.info("  val: exists, skipping.")
                results[model_key]["val"] = True
            else:
                ok = run_subprocess(
                    build_formula_val_cmd(model_key, model_cfg, args, exp01_dir, exp02_dir),
                    f"val/{model_key}",
                )
                results[model_key]["val"] = ok

    # ── Summary ──────────────────────────────────────────────────────────────
    elapsed = time.time() - total_start
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("Done in %.1fs", elapsed)
    logger.info("  %-25s  %6s  %6s  %10s", "model", "exp01", "exp02", "val")
    logger.info("  " + "─" * 52)
    for key, res in results.items():
        fmt = lambda k: ("✓" if res.get(k) else ("—" if k not in res else "✗"))
        logger.info("  %-25s  %6s  %6s  %10s", key, fmt("exp01"), fmt("exp02"), fmt("val"))

    failed = [k for k, r in results.items() if not all(r.values())]
    if failed:
        logger.error("Failed: %s", ", ".join(failed))

    summary_path = exp01_dir / "run_summary.json"
    exp01_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({"results": results, "total_elapsed_s": round(elapsed, 1)}, f, indent=2)
    logger.info("Summary → %s", summary_path)

    # Cross-model comparison plot
    exp01_passed = [k for k, r in results.items() if r.get("exp01")]
    if len(exp01_passed) >= 2:
        subprocess.run([
            sys.executable,
            str(REPO_ROOT / "experiments" / "compare_profiles.py"),
            "--results-dir", str(exp01_dir),
        ], cwd=str(REPO_ROOT))

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
