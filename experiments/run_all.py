"""
Batch runner — Exp 01 → Exp 02 → Exp 03 across all models in configs/models.yaml.
Each model runs in its own subprocess with GPU cleanup between runs.

Pipeline per model:
    Exp 01  Norm profiling → K_optimal
    Exp 02  Behavioral vector extraction
    Exp 03  Multi-layer K ramp validation → f_max / K_optimal

Usage:
    # Full pipeline, small models
    python experiments/run_all.py --tiers small --skip-existing

    # All models, skip what's done
    python experiments/run_all.py --tiers small medium --skip-existing

    # Dry run to preview
    python experiments/run_all.py --tiers small --dry-run
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
    p = argparse.ArgumentParser(description="Batch runner: Exp 01 → 02 → 03")

    # ── Model selection ──────────────────────────────────────────────────────
    p.add_argument("--config", default=str(CONFIG_PATH))
    p.add_argument("--models", nargs="+", default=None,
                   help="Specific model keys (default: all in config)")
    p.add_argument("--families", nargs="+", default=None,
                   help="Filter by family: llama qwen2 gemma2")
    p.add_argument("--tiers", nargs="+", default=None,
                   help="Filter by tier: small medium")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--device", default=None)

    # ── Exp 01 flags ─────────────────────────────────────────────────────────
    p.add_argument("--batch-size", default=4, type=int)
    p.add_argument("--max-length", default=256, type=int)

    # ── Exp 02 flags ─────────────────────────────────────────────────────────
    p.add_argument("--data-dir", default="data/behaviors")
    p.add_argument("--behaviors", nargs="+", default=None)

    # ── Exp 03 flags ─────────────────────────────────────────────────────────
    p.add_argument("--ramp-f-start", type=float, default=0.13,
                   help="Ramp start fraction of K_optimal (default: 0.13)")
    p.add_argument("--ramp-f-values", nargs="+", type=float, default=None,
                   help="f_end values to sweep (default: 0.10 to 1.0 in 0.05 steps)")

    # ── Exp 04 flags ─────────────────────────────────────────────────────────
    p.add_argument("--run-harmbench", action="store_true",
                   help="Run Exp 04 HarmBench steering evaluation (Llama family only)")
    p.add_argument("--judge-model", default="Qwen/Qwen2.5-7B-Instruct", type=str)
    p.add_argument("--harmbench-f-scale", type=float, default=0.35)
    p.add_argument("--harmbench-n", type=int, default=20)

    # ── Run control ──────────────────────────────────────────────────────────
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")

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


def exp01_done(model_key: str, exp01_dir: Path) -> bool:
    return (exp01_dir / model_key / "norm_profile.json").exists()


def exp02_done(model_key: str, exp02_dir: Path, data_dir: Path, behaviors: list[str] | None) -> bool:
    base = exp02_dir / model_key
    if not base.exists():
        return False
    targets = behaviors if behaviors else [p.stem for p in data_dir.glob("*.jsonl")]
    return all((base / b / "vectors.npz").exists() for b in targets)


def exp03_done(model_key: str, exp03_dir: Path) -> bool:
    return (exp03_dir / model_key / "ramp_summary.json").exists()


def exp04_done(model_key: str, exp04_dir: Path) -> bool:
    return (exp04_dir / model_key / "harmbench_eval.json").exists()


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


def build_exp04_cmd(model_key: str, model_cfg: dict, args: argparse.Namespace,
                    exp01_dir: Path, exp02_dir: Path, exp04_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "experiments" / "exp04_steer_harmbench.py"),
        "--model", model_cfg["model_id"],
        "--model-name", model_key,
        "--exp01-dir", str(exp01_dir),
        "--exp02-dir", str(exp02_dir),
        "--output-dir", str(exp04_dir),
        "--judge-model", args.judge_model,
        "--f-scale", str(args.harmbench_f_scale),
        "--n-behaviors", str(args.harmbench_n),
    ]
    if args.device:
        cmd += ["--device", args.device]
    return cmd


def build_exp03_cmd(model_key: str, model_cfg: dict, args: argparse.Namespace,
                    exp01_dir: Path, exp02_dir: Path, exp03_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "experiments" / "exp03_ramp_validation.py"),
        "--model", model_cfg["model_id"],
        "--model-name", model_key,
        "--exp01-dir", str(exp01_dir),
        "--exp02-dir", str(exp02_dir),
        "--output-dir", str(exp03_dir),
        "--f-start", str(args.ramp_f_start),
    ]
    if args.device:
        cmd += ["--device", args.device]
    if args.behaviors:
        cmd += ["--behaviors"] + args.behaviors
    if args.ramp_f_values:
        cmd += ["--f-values"] + [str(v) for v in args.ramp_f_values]
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
    exp03_dir = base_dir / "exp03"
    exp04_dir = base_dir / "exp04"

    config = load_config(args.config)
    selected = select_models(config, args)
    if not selected:
        logger.error("No models matched the given filters.")
        sys.exit(1)

    data_dir = Path(args.data_dir)

    # ── Print plan ───────────────────────────────────────────────────────────
    logger.info("Selected %d model(s):", len(selected))
    logger.info("  %-25s  %6s  %6s  %6s  %6s", "model", "exp01", "exp02", "exp03", "exp04")
    logger.info("  " + "─" * 56)
    for key, cfg in selected:
        e1 = "SKIP" if (args.skip_existing and exp01_done(key, exp01_dir)) else "run"
        e2 = "SKIP" if (args.skip_existing and exp02_done(key, exp02_dir, data_dir, args.behaviors)) else "run"
        e3 = "SKIP" if (args.skip_existing and exp03_done(key, exp03_dir)) else "run"
        e4 = ("SKIP" if (args.skip_existing and exp04_done(key, exp04_dir)) else "run") \
             if (args.run_harmbench and cfg.get("family") == "llama") else "—"
        logger.info("  %-25s  %6s  %6s  %6s  %6s", key, e1, e2, e3, e4)

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
        if args.skip_existing and exp01_done(model_key, exp01_dir):
            logger.info("  exp01: exists, skipping.")
            results[model_key]["exp01"] = True
        else:
            ok = run_subprocess(build_exp01_cmd(model_key, model_cfg, args, exp01_dir), f"exp01/{model_key}")
            results[model_key]["exp01"] = ok
            if not ok:
                logger.warning("  exp01 failed — skipping exp02 and exp03 for %s.", model_key)
                results[model_key]["exp02"] = False
                results[model_key]["exp03"] = False
                continue

        # Exp 02
        if args.skip_existing and exp02_done(model_key, exp02_dir, data_dir, args.behaviors):
            logger.info("  exp02: exists, skipping.")
            results[model_key]["exp02"] = True
        else:
            ok = run_subprocess(build_exp02_cmd(model_key, model_cfg, args, exp02_dir), f"exp02/{model_key}")
            results[model_key]["exp02"] = ok
            if not ok:
                logger.warning("  exp02 failed — skipping exp03 for %s.", model_key)
                results[model_key]["exp03"] = False
                continue

        # Exp 03
        if args.skip_existing and exp03_done(model_key, exp03_dir):
            logger.info("  exp03: exists, skipping.")
            results[model_key]["exp03"] = True
        else:
            ok = run_subprocess(
                build_exp03_cmd(model_key, model_cfg, args, exp01_dir, exp02_dir, exp03_dir),
                f"exp03/{model_key}",
            )
            results[model_key]["exp03"] = ok

        # Exp 04 — only for Llama family when --run-harmbench is set
        if args.run_harmbench and model_cfg.get("family") == "llama":
            if args.skip_existing and exp04_done(model_key, exp04_dir):
                logger.info("  exp04: exists, skipping.")
                results[model_key]["exp04"] = True
            else:
                ok = run_subprocess(
                    build_exp04_cmd(model_key, model_cfg, args, exp01_dir, exp02_dir, exp04_dir),
                    f"exp04/{model_key}",
                )
                results[model_key]["exp04"] = ok

    # ── Summary ──────────────────────────────────────────────────────────────
    elapsed = time.time() - total_start
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("Done in %.1fs", elapsed)
    logger.info("  %-25s  %6s  %6s  %6s  %6s", "model", "exp01", "exp02", "exp03", "exp04")
    logger.info("  " + "─" * 56)
    for key, res in results.items():
        fmt = lambda k: ("✓" if res.get(k) else ("—" if k not in res else "✗"))
        logger.info("  %-25s  %6s  %6s  %6s  %6s",
                    key, fmt("exp01"), fmt("exp02"), fmt("exp03"), fmt("exp04"))

    failed = [k for k, r in results.items() if not all(r.values())]
    if failed:
        logger.error("Failed: %s", ", ".join(failed))

    summary_path = exp01_dir / "run_summary.json"
    exp01_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({"results": results, "total_elapsed_s": round(elapsed, 1)}, f, indent=2)
    logger.info("Summary → %s", summary_path)

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
