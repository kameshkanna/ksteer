# ksteer — Calibrated Activation Steering

Activation steering without guessing the magnitude. **K_l = mean_norm_l / √d** is the per-layer maximum coherent steering magnitude. Inject beyond it and the residual stream is overwhelmed, producing gibberish. Stay within it and behavior changes predictably.

This repo measures K_l across architectures, extracts real behavioral direction vectors from contrastive pairs, and validates whether the formula holds as a universal ceiling across layer depths — positioning calibrated steering as a principled, interpretable alternative to behavioral fine-tuning.

---

## Core Claim

For a transformer with hidden dimension `d`, the residual stream at layer `l` has mean token norm `mean_norm_l`. The maximum coherent steering magnitude at that layer is:

```
K_l = mean_norm_l / sqrt(d)
```

Injecting a vector with `||v|| = alpha * K_l * sqrt(d)` is incoherent when `alpha > 1`. This is architecture-dependent: K_l varies **64×** across families (Mistral-7B: 0.52 → Qwen2.5-1.5B: 33.42). Crucially, K_l is stable within a family regardless of model size — Gemma-2-2B and Gemma-2-9B differ by 0.4%.

---

## Results (9 models, 4 families)

| Model | Family | K@40% | K@60% | K@80% | Win.Mean |
|---|---|---|---|---|---|
| Mistral-7B | mistral | 0.4271 | 0.5378 | 0.6672 | 0.5212 |
| Llama-3.1-8B | llama | 0.9028 | 1.0150 | 1.1934 | 1.0080 |
| Llama-3.2-1B | llama | 1.0376 | 1.0899 | 1.1905 | 1.0827 |
| Llama-3.2-3B | llama | 1.2480 | 1.3499 | 1.5607 | 1.3613 |
| Mistral-Nemo-12B | mistral | 1.4902 | 1.9754 | 2.6356 | 1.9538 |
| Gemma-2-2B | gemma2 | 6.6409 | 9.8668 | 12.7469 | 9.3481 |
| Gemma-2-9B | gemma2 | 7.8141 | 9.5762 | 12.1864 | 9.3892 |
| Qwen2.5-7B | qwen2 | — | — | — | ~28.55 |
| Qwen2.5-1.5B | qwen2 | — | — | — | ~33.42 |

Win.Mean = mean K_l over the 40–80% steering window.

---

## Repository Structure

```
ksteer/
├── configs/
│   └── models.yaml               model registry (id, family, tier)
├── data/
│   └── behaviors/
│       ├── sycophancy.jsonl      contrastive pairs: positive / negative
│       ├── refusal.jsonl
│       ├── formality.jsonl
│       └── verbosity.jsonl
├── experiments/
│   ├── exp01_norm_profile.py     Exp 01: norm profiling + ceiling sweep
│   ├── exp02_contrastive_vectors.py   Exp 02: behavioral direction extraction
│   ├── exp02_formula_validation.py    Exp 02b: does K_l hold across layer depths?
│   ├── compare_profiles.py       cross-model K_l comparison plots
│   ├── audit_architecture.py     pre-run sanity check for a model
│   └── run_all.py                batch runner for Exp 01 across models
├── ksteer/
│   ├── profiler.py               LayerNormProfiler, CeilingSweeper, NormProfile
│   ├── contrastive.py            ContrastiveExtractor, BehavioralVector
│   └── utils/
│       ├── model_utils.py        device-agnostic model loading, layer access
│       └── plot_utils.py         all figure generation
├── results/
│   ├── exp01/{model}/            norm_profile.json, ceiling_sweep.json, plots
│   └── exp02/{model}/{behavior}/ vectors.npz, vectors_meta.json, plots
├── requirements.txt
├── setup.py
└── setup.sh
```

---

## Setup

```bash
source setup.sh
```

This creates `.venv/`, installs PyTorch (CUDA 12.1 by default), installs all dependencies, and runs a smoke test. For CPU-only or a different CUDA version:

```bash
KSTEER_TORCH_INDEX=https://download.pytorch.org/whl/cpu source setup.sh
```

```bash
KSTEER_TORCH_INDEX=https://download.pytorch.org/whl/cu118 source setup.sh
```

To use a specific Python interpreter:

```bash
KSTEER_PYTHON=/usr/bin/python3.11 source setup.sh
```

To activate the environment in a later shell:

```bash
source .venv/bin/activate
```

---

## Experiment 01 — Norm Profiling and Ceiling Sweep

Profiles the residual stream norm at every layer and optionally sweeps `alpha × K_l` to empirically confirm the coherence ceiling.

### Step 1a — Profile one model

```bash
python experiments/exp01_norm_profile.py --model meta-llama/Llama-3.2-1B --model-name llama-3.2-1b
```

Output: `results/exp01/llama-3.2-1b/norm_profile.json` and `norm_profile.png`.

### Step 1b — Profile + ceiling sweep at one model

```bash
python experiments/exp01_norm_profile.py --model meta-llama/Llama-3.2-1B --model-name llama-3.2-1b --run-ceiling-sweep
```

Sweeps at 60% depth by default.

### Step 1c — Multi-layer ceiling sweep (recommended)

```bash
python experiments/exp01_norm_profile.py --model meta-llama/Llama-3.2-1B --model-name llama-3.2-1b --run-ceiling-sweep --sweep-layer-pcts 0.3 0.5 0.7 0.9
```

Output: `ceiling_sweep.json`, `ceiling_sweep_L{idx}.png`, `ceiling_heatmap.png`.

### Step 1d — Profile all small models in one go

```bash
python experiments/run_all.py --tiers small --run-ceiling-sweep --sweep-layer-pcts 0.3 0.5 0.7 0.9
```

### Step 1e — Profile small + medium, skip already-done models

```bash
python experiments/run_all.py --tiers small medium --run-ceiling-sweep --sweep-layer-pcts 0.3 0.5 0.7 0.9 --skip-existing
```

### Step 1f — Profile specific models by key

```bash
python experiments/run_all.py --models gemma-2-2b qwen2.5-1.5b --run-ceiling-sweep --sweep-layer-pcts 0.3 0.5 0.7 0.9
```

### Step 1g — Dry-run to see what would execute without running it

```bash
python experiments/run_all.py --tiers small medium --dry-run
```

### Step 1h — Cross-model comparison plot

Run after profiling at least two models:

```bash
python experiments/compare_profiles.py --results-dir results/exp01
```

Output: `comparison_norm_profiles.png`, `comparison_k_table.json`, `comparison_family_summary.json`.

---

## Architecture Audit (optional, run before profiling a new model)

Verifies hook shapes, padding, and steering injection are correct for a model before committing to a full run.

```bash
python experiments/audit_architecture.py --model meta-llama/Llama-3.2-1B
```

```bash
python experiments/audit_architecture.py --model google/gemma-2-2b
```

```bash
python experiments/audit_architecture.py --model Qwen/Qwen2.5-1.5B
```

---

## Experiment 02 — Contrastive Behavioral Vector Extraction

Extracts a per-layer unit steering direction for each behavior by computing mean-pooled residual stream differences between positive and negative text pairs.

### Step 2a — Extract all behaviors for one model

```bash
python experiments/exp02_contrastive_vectors.py --model meta-llama/Llama-3.2-1B --model-name llama-3.2-1b
```

Output: `results/exp02/llama-3.2-1b/{behavior}/vectors.npz` and `vectors_meta.json` for each of `sycophancy`, `refusal`, `formality`, `verbosity`.

### Step 2b — Extract a specific subset of behaviors

```bash
python experiments/exp02_contrastive_vectors.py --model meta-llama/Llama-3.2-1B --model-name llama-3.2-1b --behaviors sycophancy refusal
```

### Step 2c — Skip behaviors already extracted

```bash
python experiments/exp02_contrastive_vectors.py --model meta-llama/Llama-3.2-1B --model-name llama-3.2-1b --skip-existing
```

### Step 2d — Extract for a different model

```bash
python experiments/exp02_contrastive_vectors.py --model google/gemma-2-2b --model-name gemma-2-2b
```

```bash
python experiments/exp02_contrastive_vectors.py --model Qwen/Qwen2.5-1.5B --model-name qwen2.5-1.5b
```

```bash
python experiments/exp02_contrastive_vectors.py --model mistralai/Mistral-7B-v0.1 --model-name mistral-7b
```

Key output to read: `extraction_summary.json` shows `window_consistency_mean` per behavior. Values above 0.6 indicate a well-defined behavioral axis in the 40–80% steering window. Values below 0.3 suggest the behavior is diffuse and may not steer cleanly.

---

## Experiment 02b — Formula Validation

Tests whether `K_l = mean_norm_l / sqrt(d)` is a universal ceiling by sweeping `alpha × K_l` using real contrastive vectors at multiple layer depths. A flat empirical ceiling across depth means the formula holds universally. Systematic drift reveals where it breaks.

Requires: Exp 01 norm profile and Exp 02 vectors both completed for the model.

### Step 2e — Run formula validation on one model

```bash
python experiments/exp02_formula_validation.py --model meta-llama/Llama-3.2-1B --model-name llama-3.2-1b --exp01-dir results/exp01 --exp02-dir results/exp02
```

### Step 2f — Validate fewer layer depths for a faster run

```bash
python experiments/exp02_formula_validation.py --model meta-llama/Llama-3.2-1B --model-name llama-3.2-1b --exp01-dir results/exp01 --exp02-dir results/exp02 --sweep-layer-pcts 0.2 0.4 0.6 0.8
```

### Step 2g — Validate a specific behavior only

```bash
python experiments/exp02_formula_validation.py --model meta-llama/Llama-3.2-1B --model-name llama-3.2-1b --exp01-dir results/exp01 --exp02-dir results/exp02 --behaviors sycophancy
```

### Step 2h — Validate on architecturally distinct models

These are the most informative because their K_l ranges are very different:

```bash
python experiments/exp02_formula_validation.py --model google/gemma-2-2b --model-name gemma-2-2b --exp01-dir results/exp01 --exp02-dir results/exp02
```

```bash
python experiments/exp02_formula_validation.py --model Qwen/Qwen2.5-1.5B --model-name qwen2.5-1.5b --exp01-dir results/exp01 --exp02-dir results/exp02
```

```bash
python experiments/exp02_formula_validation.py --model mistralai/Mistral-7B-v0.1 --model-name mistral-7b --exp01-dir results/exp01 --exp02-dir results/exp02
```

Key output: `formula_summary.json` reports a **coefficient of variation (CV)** of the empirical ceiling alpha across layers per behavior.

| CV | Grade | Meaning |
|---|---|---|
| < 0.15 | `strong` | K_l correctly normalizes depth — universally valid formula |
| 0.15–0.30 | `partial` | Mild drift — early or late layers deviate slightly |
| > 0.30 | `weak` | Formula breaks down — layer-range correction needed |

Output: `formula_validation.png` shows empirical ceiling alpha vs layer depth for each behavior. A flat line at `alpha ≈ 1` is the ideal result.

---

## Recommended Run Order for a New Model

1. Audit: `python experiments/audit_architecture.py --model <model_id>`
2. Profile: `python experiments/exp01_norm_profile.py --model <model_id> --model-name <name> --run-ceiling-sweep --sweep-layer-pcts 0.3 0.5 0.7 0.9`
3. Extract: `python experiments/exp02_contrastive_vectors.py --model <model_id> --model-name <name>`
4. Validate: `python experiments/exp02_formula_validation.py --model <model_id> --model-name <name> --exp01-dir results/exp01 --exp02-dir results/exp02`
5. Compare: `python experiments/compare_profiles.py --results-dir results/exp01`

---

## Model Registry

Models are defined in `configs/models.yaml`. Each entry has:
- `model_id`: HuggingFace model ID
- `family`: architecture family (`llama`, `mistral`, `mixtral`, `qwen2`, `gemma2`)
- `tier`: compute tier (`small` = <4B, `medium` = 4–15B, `large` = >15B)

To filter runs by tier or family:

```bash
python experiments/run_all.py --families llama
```

```bash
python experiments/run_all.py --tiers small
```

```bash
python experiments/run_all.py --families gemma2 --run-ceiling-sweep --sweep-layer-pcts 0.3 0.5 0.7 0.9
```

---

## Architecture Notes

**Llama / Mistral**: Pre-norm (RMSNorm before each sub-layer). Clean residual stream. K_l is low (~1.0 for Llama-3, ~0.5 for Mistral-7B).

**Gemma-2**: Pre-norm + post-norm on each sub-layer output before residual add. This inflates the residual stream norm. K_l is ~18× higher than Llama at the same depth. This is a real architectural property, not a measurement artifact.

**Qwen2.5**: Pre-norm like Llama but trained with a large norm scale. K_l is ~60× higher than Mistral-7B. Steering at K_l works but random vectors collapse at even `alpha < 0.25`. Real contrastive vectors are more stable.

**Mixtral**: MoE — residual stream norms behave like a dense 7B per expert. Treat as Mistral for K_l purposes.

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU recommended (CPU works for small models)
- HuggingFace access token for gated models (Llama, Gemma)

Set your token before running:

```bash
huggingface-cli login
```
