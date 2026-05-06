# ksteer — Calibrated Activation Steering

Activation steering without guessing the magnitude. **K_l = mean_norm_l / √d** is the per-layer maximum coherent steering magnitude. Inject beyond it and the residual stream is overwhelmed, producing gibberish. Stay within it and behavior changes predictably.

This repo:
1. Measures K_l across architectures and model sizes (Exp 01)
2. Extracts real behavioral direction vectors from contrastive pairs (Exp 02)
3. Validates whether K_l is a universal ceiling across layer depths (Exp 02b)
4. Tests the generalized behavior-calibrated formula K_l^b = K_l / ρ_l (Exp 03)

---

## Core Claim

For a transformer with hidden dimension `d`, the residual stream at layer `l` has mean token norm `mean_norm_l`. The maximum coherent steering magnitude at that layer is:

```
K_l = mean_norm_l / sqrt(d)
```

Injecting a unit vector scaled to `alpha × K_l × sqrt(d)` causes gibberish when `alpha > 1`. K_l is architecture-dependent — it varies **55×** across families (Mistral-7B: 0.52 → Qwen2.5-7B: 28.55) — and is **scale-invariant only in families with consistent architecture**: Gemma-2 (0.4% gap across 2B→9B) and Llama-3 (~35% spread across 1B→70B). Mistral and Qwen2.5 are NOT scale-invariant — each size must be profiled independently.

The generalized formula accounts for behavioral signal strength:

```
K_l^b = K_l / rho_l    where rho_l = ||mean_diff_l|| / mean_norm_l
```

`rho_l` is the behavioral SNR at layer `l`. High-SNR behaviors (sycophancy, refusal) have lower effective ceilings — they break coherence at `alpha_eff ≈ 0.2–0.35 × K_l` rather than `1.0 × K_l`.

---

## Results (9 models, 4 families)

| Model | Family | K@40% | K@60% | K@80% | Win.Mean K |
|---|---|---|---|---|---|
| Mistral-7B | mistral | 0.4271 | 0.5378 | 0.6672 | 0.5212 |
| Llama-3.1-8B | llama | 0.9028 | 1.0150 | 1.1934 | 1.0080 |
| Llama-3.2-1B | llama | 1.0376 | 1.0899 | 1.1905 | 1.0827 |
| Llama-3.2-3B | llama | 1.2480 | 1.3499 | 1.5607 | 1.3613 |
| Mistral-Nemo-12B | mistral | 1.4902 | 1.9754 | 2.6356 | 1.9538 |
| Gemma-2-2B | gemma2 | 6.6409 | 9.8669 | 12.7469 | 9.3481 |
| Gemma-2-9B | gemma2 | 7.8141 | 9.5762 | 12.1864 | 9.3892 |
| Qwen2.5-3B | qwen2 | 10.9222 | 11.4727 | 12.3698 | 11.4311 |
| Qwen2.5-7B | qwen2 | 27.7858 | 28.4754 | 29.9449 | 28.5528 |

Win.Mean = mean K_l over the 40–80% steering window. Practical steering range: `alpha = 0.2–0.3 × K_l`.

**Family constants** (Win.Mean averaged across models in family):

| Family | Win.Mean K | Scale-invariant? | Note |
|---|---|---|---|
| gemma2 | 9.37 | ✓ Yes (0.4% gap) | Double-norm architecture |
| llama | 1.15 | ✓ Approximately | Pre-norm, consistent pattern |
| mistral | 1.24 | ✗ No | 7B ≠ Nemo-12B architecture |
| qwen2 | — | ✗ No | 3B ≠ 7B (non-uniform scaling) |

---

## Repository Structure

```
ksteer/
├── configs/
│   └── models.yaml                     model registry (id, family, tier)
├── data/
│   └── behaviors/
│       ├── sycophancy.jsonl            contrastive pairs: {"positive": "...", "negative": "..."}
│       ├── refusal.jsonl
│       ├── formality.jsonl
│       └── verbosity.jsonl
├── experiments/
│   ├── exp01_norm_profile.py           Exp 01: norm profiling + optional ceiling sweep
│   ├── exp02_contrastive_vectors.py    Exp 02: behavioral direction extraction
│   ├── exp02_formula_validation.py     Exp 02b: K_l universality test across layer depths
│   ├── exp03_formula_calibration.py    Exp 03: K_l^b = K_l/ρ_l calibration (no GPU needed)
│   ├── exp04_instruct_vs_base.py       Exp 04: attractor amplification gamma_l = alpha_eff^IT / alpha_eff^base
│   ├── run_all.py                      batch runner — orchestrates Exp 01/02/02b across models
│   ├── aggregate_results.py            consolidate all results into cross-family summary
│   ├── compare_profiles.py             cross-model K_l comparison plots
│   └── audit_architecture.py          sanity check hooks + shapes for a new model
├── ksteer/
│   ├── profiler.py                     LayerNormProfiler, CeilingSweeper, NormProfile
│   ├── contrastive.py                  ContrastiveExtractor, BehavioralVector
│   └── utils/
│       ├── model_utils.py              device-agnostic model loading, layer iteration
│       └── plot_utils.py               all figure generation
└── results/                            created at runtime
    ├── exp01/{model}/
    ├── exp02/{model}/{behavior}/
    └── exp03/{model}/
```

---

## Setup

```bash
source setup.sh
```

Creates `.venv/`, installs PyTorch (CUDA 12.1 by default), installs all dependencies, runs a smoke test. Override options:

```bash
KSTEER_TORCH_INDEX=https://download.pytorch.org/whl/cpu source setup.sh
```

```bash
KSTEER_TORCH_INDEX=https://download.pytorch.org/whl/cu118 source setup.sh
```

```bash
KSTEER_PYTHON=/usr/bin/python3.11 source setup.sh
```

Activate in a later shell:

```bash
source .venv/bin/activate
```

Authenticate HuggingFace (required for Llama and Gemma):

```bash
huggingface-cli login
```

---

## Quick Start — Run Everything in One Shot

Run all four experiments for all small models (≤3B), skipping anything already done:

```bash
python experiments/run_all.py --tiers small --run-ceiling-sweep --run-exp02 --run-formula-validation --skip-existing
```

Small + medium models (up to 12B):

```bash
python experiments/run_all.py --tiers small medium --run-ceiling-sweep --run-exp02 --run-formula-validation --skip-existing
```

After the batch run finishes, run the no-GPU calibration and aggregate:

```bash
python experiments/exp03_formula_calibration.py
python experiments/aggregate_results.py
```

Dry-run first to preview the plan without running anything:

```bash
python experiments/run_all.py --tiers small medium --run-ceiling-sweep --run-exp02 --run-formula-validation --dry-run
```

---

## Single Model — End-to-End

Full pipeline for one model in five commands. Replace `<model_id>` and `<name>` from the [model registry](#model-registry) below.

```bash
python experiments/audit_architecture.py --model <model_id>
```

```bash
python experiments/exp01_norm_profile.py --model <model_id> --model-name <name> --run-ceiling-sweep --sweep-layer-pcts 0.4 0.5 0.6 0.7 0.8
```

```bash
python experiments/exp02_contrastive_vectors.py --model <model_id> --model-name <name>
```

```bash
python experiments/exp02_formula_validation.py --model <model_id> --model-name <name> --exp01-dir results/exp01 --exp02-dir results/exp02
```

```bash
python experiments/exp03_formula_calibration.py --models <name>
```

Example — Llama-3.2-1B:

```bash
python experiments/audit_architecture.py --model meta-llama/Llama-3.2-1B
python experiments/exp01_norm_profile.py --model meta-llama/Llama-3.2-1B --model-name llama-3.2-1b --run-ceiling-sweep --sweep-layer-pcts 0.4 0.5 0.6 0.7 0.8
python experiments/exp02_contrastive_vectors.py --model meta-llama/Llama-3.2-1B --model-name llama-3.2-1b
python experiments/exp02_formula_validation.py --model meta-llama/Llama-3.2-1B --model-name llama-3.2-1b --exp01-dir results/exp01 --exp02-dir results/exp02
python experiments/exp03_formula_calibration.py --models llama-3.2-1b
```

---

## Experiment 01 — Norm Profiling and Ceiling Sweep

**What it does:** Runs a forward pass on a sample corpus, records the mean and std of the residual stream L2 norm at every transformer layer, and computes `K_l = mean_norm_l / sqrt(d)`. Optionally runs a live coherence sweep to empirically confirm the ceiling.

**Requires:** Nothing (first experiment in the chain).

**Key outputs:**

| File | Contents |
|---|---|
| `results/exp01/{model}/norm_profile.json` | `layer_mean_norms`, `layer_std_norms`, `k_values`, `hidden_dim`, `num_layers` |
| `results/exp01/{model}/norm_profile.png` | Mean norm and K_l per layer depth, 40–80% window shaded |
| `results/exp01/{model}/ceiling_sweep.json` | Per-layer coherence results for the alpha sweep |
| `results/exp01/{model}/ceiling_heatmap.png` | Coherence heatmap: layers × alpha values |

### Run — single model, profile only

```bash
python experiments/exp01_norm_profile.py --model meta-llama/Llama-3.2-1B --model-name llama-3.2-1b
```

### Run — single model, profile + ceiling sweep at multiple depths (recommended)

```bash
python experiments/exp01_norm_profile.py --model meta-llama/Llama-3.2-1B --model-name llama-3.2-1b --run-ceiling-sweep --sweep-layer-pcts 0.4 0.5 0.6 0.7 0.8
```

### Run — all small models in batch

```bash
python experiments/run_all.py --tiers small --run-ceiling-sweep --sweep-layer-pcts 0.4 0.5 0.6 0.7 0.8 --skip-existing
```

### Run — specific models by key

```bash
python experiments/run_all.py --models gemma-2-2b qwen2.5-1.5b --run-ceiling-sweep --sweep-layer-pcts 0.4 0.5 0.6 0.7 0.8
```

### Run — cross-model comparison plot (after profiling ≥2 models)

```bash
python experiments/compare_profiles.py --results-dir results/exp01
```

Outputs `comparison_norm_profiles.png`, `comparison_k_table.json`, and `comparison_family_summary.json`.

### Key flags

| Flag | Default | Description |
|---|---|---|
| `--model` | required | HuggingFace model ID |
| `--model-name` | last part of model ID | Key used for output directory name |
| `--batch-size` | 4 | Batch size for profiling forward passes |
| `--max-length` | 256 | Tokenization truncation length |
| `--run-ceiling-sweep` | off | Enable the alpha×K_l coherence sweep |
| `--sweep-layer-pcts` | `0.4 0.5 0.6 0.7 0.8` | Layer depths for the ceiling sweep (default: 40–80% window) |
| `--output-dir` | `results/exp01` | Where to write results |
| `--device` | auto | `cuda`, `cpu`, or leave blank for auto |

---

## Architecture Audit (run before any new model)

Verifies hook attachment, tensor shapes, padding, and steering injection before committing to a full run. Takes under a minute.

```bash
python experiments/audit_architecture.py --model meta-llama/Llama-3.2-1B
```

```bash
python experiments/audit_architecture.py --model google/gemma-2-2b
```

```bash
python experiments/audit_architecture.py --model Qwen/Qwen2.5-1.5B
```

```bash
python experiments/audit_architecture.py --model mistralai/Mistral-7B-v0.1
```

---

## Experiment 02 — Contrastive Behavioral Vector Extraction

**What it does:** For each behavior JSONL file, runs forward passes on every (positive, negative) text pair, mean-pools the residual stream at every layer, takes the mean difference across pairs, and unit-normalizes it to produce a per-layer behavioral direction vector. Reports a layer consistency score (mean cosine alignment of individual pair diffs to the aggregate direction).

**Requires:** Nothing (independent of Exp 01, but Exp 01 must be done before Exp 02b).

**Behavior files** live in `data/behaviors/`. Each line is `{"positive": "...", "negative": "..."}`.

**Key outputs:**

| File | Contents |
|---|---|
| `results/exp02/{model}/{behavior}/vectors.npz` | `layer_vectors`: float32 array, shape `(num_layers, hidden_dim)` — unit behavioral directions |
| `results/exp02/{model}/{behavior}/vectors_meta.json` | `layer_consistency`, `layer_raw_norms`, `num_pairs`, `hidden_dim`, `num_layers` |
| `results/exp02/{model}/{behavior}/consistency.png` | Per-layer consistency score and raw diff norm |
| `results/exp02/{model}/extraction_summary.json` | `window_consistency_mean` per behavior (key metric) |

**Interpreting `window_consistency_mean`** (40–80% depth):

| Value | Meaning |
|---|---|
| > 0.6 | Clean behavioral axis — vector will steer reliably |
| 0.3–0.6 | Moderate — behavior is somewhat diffuse across pairs |
| < 0.3 | Diffuse — contrastive pairs disagree; expect unreliable steering |

### Run — single model, all behaviors

```bash
python experiments/exp02_contrastive_vectors.py --model meta-llama/Llama-3.2-1B --model-name llama-3.2-1b
```

### Run — specific behaviors only

```bash
python experiments/exp02_contrastive_vectors.py --model meta-llama/Llama-3.2-1B --model-name llama-3.2-1b --behaviors sycophancy refusal
```

### Run — skip already extracted behaviors

```bash
python experiments/exp02_contrastive_vectors.py --model meta-llama/Llama-3.2-1B --model-name llama-3.2-1b --skip-existing
```

### Run — other models

```bash
python experiments/exp02_contrastive_vectors.py --model google/gemma-2-2b --model-name gemma-2-2b
```

```bash
python experiments/exp02_contrastive_vectors.py --model Qwen/Qwen2.5-1.5B --model-name qwen2.5-1.5b
```

```bash
python experiments/exp02_contrastive_vectors.py --model mistralai/Mistral-7B-v0.1 --model-name mistral-7b
```

### Run — all small + medium models in batch

```bash
python experiments/run_all.py --tiers small medium --run-exp02 --skip-existing
```

### Key flags

| Flag | Default | Description |
|---|---|---|
| `--model` | required | HuggingFace model ID |
| `--model-name` | last part of model ID | Output directory key |
| `--data-dir` | `data/behaviors` | Directory containing behavior JSONL files |
| `--behaviors` | all in data-dir | Specific behavior names to extract |
| `--max-length` | 256 | Tokenization truncation length |
| `--skip-existing` | off | Skip behaviors whose vectors.npz already exists |
| `--output-dir` | `results/exp02` | Where to write results |

---

## Experiment 02b — Formula Validation

**What it does:** Uses real contrastive vectors (from Exp 02) to sweep `alpha × K_l` at multiple layer depths and measures the empirical coherence ceiling at each depth. If K_l is a universal formula, the ceiling alpha should be approximately constant across all layer depths. Systematic drift reveals where the formula over- or under-estimates.

**Requires:** Exp 01 (norm profile) and Exp 02 (behavioral vectors) for the model.

**Key outputs:**

| File | Contents |
|---|---|
| `results/exp02/{model}/formula_validation/formula_summary.json` | `ceiling_by_layer`, `mean_ceiling_alpha`, `std_ceiling_alpha`, `coefficient_of_variation`, `formula_accuracy` per behavior |
| `results/exp02/{model}/formula_validation/{behavior}_validation.json` | Raw per-layer alpha sweep results |
| `results/exp02/{model}/formula_validation/formula_validation.png` | Empirical ceiling alpha vs layer depth, per behavior |

**Interpreting `formula_accuracy` (CV of ceiling alpha across layers):**

| CV | Grade | Meaning |
|---|---|---|
| < 0.15 | `strong` | K_l normalizes depth correctly — formula is universal |
| 0.15–0.30 | `partial` | Mild depth drift — early or late layers deviate slightly |
| > 0.30 | `weak` | Systematic depth trend — correction factor needed |

Note: `weak` is not a failure — it quantifies **where** K_l deviates and motivates the generalized K_l^b formula (Exp 03).

### Run — single model, full depth sweep

```bash
python experiments/exp02_formula_validation.py --model meta-llama/Llama-3.2-1B --model-name llama-3.2-1b --exp01-dir results/exp01 --exp02-dir results/exp02
```

### Run — fewer depths for a faster result

```bash
python experiments/exp02_formula_validation.py --model meta-llama/Llama-3.2-1B --model-name llama-3.2-1b --exp01-dir results/exp01 --exp02-dir results/exp02 --sweep-layer-pcts 0.2 0.4 0.6 0.8
```

### Run — every single layer (slowest, most complete)

```bash
python experiments/exp02_formula_validation.py --model meta-llama/Llama-3.2-1B --model-name llama-3.2-1b --exp01-dir results/exp01 --exp02-dir results/exp02 --sweep-all-layers
```

### Run — specific behavior only

```bash
python experiments/exp02_formula_validation.py --model meta-llama/Llama-3.2-1B --model-name llama-3.2-1b --exp01-dir results/exp01 --exp02-dir results/exp02 --behaviors sycophancy
```

### Run — other models

```bash
python experiments/exp02_formula_validation.py --model google/gemma-2-2b --model-name gemma-2-2b --exp01-dir results/exp01 --exp02-dir results/exp02
```

```bash
python experiments/exp02_formula_validation.py --model Qwen/Qwen2.5-1.5B --model-name qwen2.5-1.5b --exp01-dir results/exp01 --exp02-dir results/exp02
```

```bash
python experiments/exp02_formula_validation.py --model mistralai/Mistral-7B-v0.1 --model-name mistral-7b --exp01-dir results/exp01 --exp02-dir results/exp02
```

### Run — all small + medium models in batch

```bash
python experiments/run_all.py --tiers small medium --run-exp02 --run-formula-validation --skip-existing
```

### Key flags

| Flag | Default | Description |
|---|---|---|
| `--model` | required | HuggingFace model ID |
| `--model-name` | last part of model ID | Must match the name used in Exp 01 and 02 |
| `--exp01-dir` | `results/exp01` | Directory containing Exp 01 norm profiles |
| `--exp02-dir` | `results/exp02` | Directory containing Exp 02 behavioral vectors |
| `--sweep-layer-pcts` | `0.4 0.45 … 0.8` | Layer depths as fractions — default is the 40–80% steering window |
| `--sweep-all-layers` | off | Test every single layer (overrides --sweep-layer-pcts) |
| `--alphas` | `0.25 0.5 … 3.0` | Alpha multipliers of K_l to sweep |
| `--behaviors` | all found in exp02-dir | Specific behaviors to validate |

---

## Experiment 03 — Formula Calibration (no GPU required)

**What it does:** Post-hoc analysis of the generalized formula. Computes the behavioral SNR `rho_l = ||mean_diff_l|| / mean_norm_l` from the already-extracted vectors and derives `K_l^b = K_l / rho_l`. Then checks whether the ratio `(alpha_eff × K_l) / K_l^b` is approximately constant across all layers and behaviors. A constant ratio means K_l^b is a universal, behavior-calibrated steering budget.

**Requires:** Exp 01, Exp 02, and Exp 02b all completed for the target models. **No GPU needed.**

**Key outputs:**

| File | Contents |
|---|---|
| `results/exp03/{model}/calibration_summary.json` | Per-behavior mean ratio, std, CV, R² |
| `results/exp03/{model}/{behavior}_calibration.json` | Per-layer: k_l, rho_l, k_l_b, alpha_empirical, abs_ceiling, ratio |
| `results/exp03/{model}/calibration_plot.png` | Scatter of K_l^b vs empirical ceiling; ratio distribution per behavior |
| `results/exp03/cross_model_summary.json` | Aggregated ratio stats across all models |
| `results/exp03/cross_model_summary.md` | Human-readable table + interpretation |

**Interpreting the calibration ratio CV:**

| CV | Grade | Meaning |
|---|---|---|
| < 0.10 | `excellent` | K_l^b is a near-perfect universal ceiling |
| 0.10–0.20 | `good` | Small residual layer-depth correction needed |
| 0.20–0.40 | `partial` | ρ_l partially explains the gap |
| > 0.40 | `poor` | ρ_l alone insufficient; higher-order term needed |

### Run — all models with formula validation results

```bash
python experiments/exp03_formula_calibration.py
```

### Run — specific models

```bash
python experiments/exp03_formula_calibration.py --models llama-3.2-1b qwen2.5-1.5b gemma-2-2b mistral-7b
```

### Key flags

| Flag | Default | Description |
|---|---|---|
| `--exp01-dir` | `results/exp01` | Norm profile directory |
| `--exp02-dir` | `results/exp02` | Vectors + formula validation directory |
| `--output-dir` | `results/exp03` | Output directory |
| `--models` | all found | Model name keys to process |
| `--window-min` | `0.4` | Minimum layer depth fraction (default: 40%) |
| `--window-max` | `0.8` | Maximum layer depth fraction (default: 80%) |

---

## Experiment 04 — Attractor Amplification: Base vs Instruct

**What it does:** Tests whether instruction-tuned models have a higher effective steering ceiling than their base counterparts, and whether the effect is direction-dependent. Computes the attractor amplification factor `gamma_l = alpha_eff^IT / alpha_eff^base` using the same behavioral vector on both variants. Separates two effects: norm inflation (K_l changes post-RLHF) and directional resistance (K_l same, but model resists perturbations in specific directions).

**Critical prediction:** If RLHF creates directionally biased hardening:
- `gamma_l(unsafe direction)` >> 1 — RLHF resists steering toward harmful outputs
- `gamma_l(safe direction)` ≈ 1 or < 1 — RLHF does not resist steering toward safe outputs
- `gamma_l(neutral)` ≈ 1 — RLHF indifferent to non-safety directions

The **asymmetry index** = `mean_gamma(unsafe) / mean_gamma(safe)` is a quantitative measure of how directionally biased the safety training is.

**Requires:** Exp 01 (base model norm profile) and Exp 02 (behavioral vectors) for each pair. Instruct model is profiled live during this experiment.

**Pairs config:** `configs/instruct_pairs.yaml` — defines base ↔ instruct pairs for each model.

### Run — specific pairs

```bash
python experiments/exp04_instruct_vs_base.py --pairs llama-3.2-1b qwen2.5-3b
```

### Run — all small models

```bash
python experiments/exp04_instruct_vs_base.py --tiers small
```

### Run — all Llama pairs

```bash
python experiments/exp04_instruct_vs_base.py --families llama
```

### Key outputs

| File | Contents |
|---|---|
| `results/exp04/{pair}/norm_comparison.json` | K_l^base vs K_l^IT per layer (Effect 1 measurement) |
| `results/exp04/{pair}/{behavior}_gamma.json` | gamma_l per layer, raw and norm-corrected |
| `results/exp04/{pair}/gamma_summary.json` | mean_gamma per behavior, asymmetry_index |
| `results/exp04/{pair}/gamma_plot.png` | gamma_l vs layer depth, colored by safety class |
| `results/exp04/cross_pair_summary.json` | asymmetry index across all pairs |

### Key flags

| Flag | Default | Description |
|---|---|---|
| `--pairs` | all in config | Specific pair keys to run |
| `--families` / `--tiers` | all | Filter pairs |
| `--alphas` | `0.1 0.2 … 3.0` | Alpha sweep — fine resolution below 1.0 needed to catch base model ceilings |
| `--window-min` / `--window-max` | `0.4` / `0.8` | Layer depth window |
| `--skip-existing` | off | Skip pairs with existing gamma_summary.json |

---

## Aggregation — Cross-Family Summary

Consolidates Exp 01 K_l values and Exp 03 calibration results into a single cross-family table. Run after all models are processed.

```bash
python experiments/aggregate_results.py
```

Outputs `results/cross_family_summary.json` and `results/cross_family_summary.md`.

### Key flags

| Flag | Default | Description |
|---|---|---|
| `--exp01-dir` | `results/exp01` | Norm profile directory |
| `--exp02-dir` | `results/exp02` | Formula validation directory |
| `--output-dir` | `results` | Where to write the summary |

---

## Batch Runner Reference

`run_all.py` orchestrates Exp 01 → 02 → 02b in sequence per model, with subprocess isolation and GPU cleanup between runs. Exp 03 and aggregation are run separately (they are fast, no-GPU steps).

### Model selection flags

| Flag | Description |
|---|---|
| `--models <keys>` | Run specific model keys (e.g. `llama-3.2-1b gemma-2-2b`) |
| `--families <names>` | Filter by family: `llama mistral qwen2 gemma2` |
| `--tiers <names>` | Filter by tier: `small medium large` |

### Experiment flags

| Flag | Description |
|---|---|
| `--run-ceiling-sweep` | Include alpha×K_l sweep in Exp 01 |
| `--sweep-layer-pcts` | Layer depths for the Exp 01 ceiling sweep (default: 40–80% window) |
| `--run-exp02` | Run Exp 02 contrastive extraction after Exp 01 |
| `--run-formula-validation` | Run Exp 02b formula validation after Exp 02 |
| `--val-alphas` | Alpha values for the validation sweep |
| `--val-sweep-layer-pcts` | Layer depths for the validation sweep (default: 40–80% window) |

### Control flags

| Flag | Description |
|---|---|
| `--skip-existing` | Skip models whose output files already exist |
| `--dry-run` | Print what would run, execute nothing |
| `--output-dir` | Base output directory (default: `results`). Exp 01 → `{output-dir}/exp01`, Exp 02 → `{output-dir}/exp02` |

---

## Model Registry

Models are defined in `configs/models.yaml`. Add new entries there to include them in batch runs.

| Key | Model ID | Family | Tier |
|---|---|---|---|
| `llama-3.2-1b` | `meta-llama/Llama-3.2-1B` | llama | small |
| `llama-3.2-3b` | `meta-llama/Llama-3.2-3B` | llama | small |
| `llama-3.1-8b` | `meta-llama/Llama-3.1-8B` | llama | medium |
| `llama-3.1-70b` | `meta-llama/Llama-3.1-70B` | llama | large |
| `mistral-7b` | `mistralai/Mistral-7B-v0.1` | mistral | medium |
| `mistral-nemo-12b` | `mistralai/Mistral-Nemo-Base-2407` | mistral | medium |
| `qwen2.5-3b` | `Qwen/Qwen2.5-3B` | qwen2 | small |
| `qwen2.5-7b` | `Qwen/Qwen2.5-7B` | qwen2 | medium |
| `gemma-2-2b` | `google/gemma-2-2b` | gemma2 | small |
| `gemma-2-9b` | `google/gemma-2-9b` | gemma2 | medium |

Tier definitions: `small` = <4B params, `medium` = 4–15B, `large` = >15B.

---

## Architecture Notes

### Llama 3 (1B / 3B / 8B / 70B)
Pre-norm only: `h_{l+1} = h_l + SubLayer(RMSNorm(h_l))`. The same architectural pattern — GQA, SwiGLU, RoPE, RMSNorm — is preserved consistently across all Llama 3 sizes. This structural consistency is why K_l is approximately scale-invariant (Win.Mean 1.008–1.361 from 1B to 8B, confirmed ~1.08 at 70B). Profile any one Llama-3 size and the K_l budget applies across the family within ~35%.

### Gemma 2 (2B / 9B)
Double-norm: `h_{l+1} = h_l + PostNorm(SubLayer(PreNorm(h_l)))`. RMSNorm is applied both before and after every sub-layer (attention and MLP). The post-norm clamps each sub-layer's output to unit scale before the residual add, causing residual norms to accumulate much faster than in pre-norm-only architectures. K_l ≈ 9.37 — stable to 0.4% between 2B and 9B because both use the identical double-norm scheme with the same epsilon (1e-6). Do not share K_l budgets with Llama or Mistral.

### Qwen2.5 (3B / 7B)
**Not scale-invariant.** Qwen2.5 does not use a consistent architectural scaling law across sizes:

| Size | Layers | Hidden | Intermediate | K_l (Win.Mean) |
|---|---|---|---|---|
| 3B | 36 | 3072 | 12288 | 11.43 |
| 7B | 28 | 3584 | 18944 | 28.55 |

The 3B has more layers but a smaller hidden dim than the 7B — non-uniform scaling that makes K_l vary 2.5× within the family. Profile each Qwen2.5 size independently; do not share K_l values across sizes.

### Mistral-7B v0.1 vs Mistral-Nemo-12B
**Not scale-invariant — fundamentally different architectures.** Mistral-7B (32 layers, hidden 4096, sliding window attention 4096 tokens, K_l = 0.52) vs Mistral-Nemo-12B (40 layers, hidden 5120, 128K context window, co-developed with NVIDIA, K_l = 1.95). These are distinct architectures that happen to share the Mistral name. Treat each as its own family for K_l purposes.

### Instruction-Tuned vs Base Models
Empirically, instruction-tuned variants (SFT + RLHF) require **significantly higher steering magnitude to break coherence** than their base counterparts. The mechanism: RLHF reinforces specific output attractors in the residual stream, increasing the behavioral SNR (ρ_l) for safety-relevant directions. In the K_l^b = K_l / ρ_l framework, higher ρ_l means a tighter behavior-calibrated ceiling — more magnitude is needed to escape the safe basin. Always profile the exact variant you intend to steer (base vs instruct); K_l does not transfer between them.

---

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA
- ~24GB VRAM for 7–12B models (`device_map="auto"`)
- ~80GB VRAM + 200GB RAM for 70B models (A100 + `device_map="auto"` distributes across GPU and CPU RAM)
- HuggingFace access token for Llama and Gemma gated repos

CPU is supported for small models but will be slow. All experiments use `device_map="auto"` and purge GPU memory between model runs.
