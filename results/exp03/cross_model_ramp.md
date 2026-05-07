# Exp 03 — Multi-layer K Ramp Validation

**Claim**: `f_scale_max ≈ 0.48` across all model families, ramp shapes,
and K_optimal definitions, where `K_i = f_scale × K_optimal × shape_weights[i]`.

Baseline (Qwen2.5-3B-Instruct empirical): f_scale_max = 0.48
Consistent = |f_scale_max − 0.48| / 0.48 < 20%

## gemma-2-2b  (family=gemma2)

K_optimal_mid = 8.5108  | K_optimal_window = 9.3481

| Shape | K_source | K_optimal | mean f_max | K_peak_raw | std | ±baseline | Holds? |
|---|---|---|---|---|---|---|---|
| linear | mid | 8.5108 | 1.000 | 8.5108 | 0.000 | +1.083 | ✗ |
| linear | window | 9.3481 | 0.975 | 9.1144 | 0.043 | +1.031 | ✗ |
| cosine | mid | 8.5108 | 1.000 | 8.5108 | 0.000 | +1.083 | ✗ |
| cosine | window | 9.3481 | 0.988 | 9.2313 | 0.022 | +1.057 | ✗ |
| bell | mid | 8.5108 | 1.000 | 8.5108 | 0.000 | +1.083 | ✗ |
| bell | window | 9.3481 | 0.975 | 9.1144 | 0.043 | +1.031 | ✗ |
| exponential | mid | 8.5108 | 1.000 | 8.5108 | 0.000 | +1.083 | ✗ |
| exponential | window | 9.3481 | 1.000 | 9.3481 | 0.000 | +1.083 | ✗ |
| constant | mid | 8.5108 | 0.887 | 7.5533 | 0.195 | +0.849 | ✗ |
| constant | window | 9.3481 | 1.000 | 9.3481 | 0.000 | +1.083 | ✗ |

## gemma-2-9b  (family=gemma2)

K_optimal_mid = 8.9510  | K_optimal_window = 9.3892

| Shape | K_source | K_optimal | mean f_max | K_peak_raw | std | ±baseline | Holds? |
|---|---|---|---|---|---|---|---|
| linear | mid | 8.9510 | 0.575 | 5.1468 | 0.425 | +0.198 | ✗ |
| linear | window | 9.3892 | 0.700 | 6.5724 | 0.424 | +0.458 | ✗ |
| cosine | mid | 8.9510 | 0.550 | 4.9231 | 0.450 | +0.146 | ✗ |
| cosine | window | 9.3892 | 0.550 | 5.1641 | 0.450 | +0.146 | ✗ |
| bell | mid | 8.9510 | 0.400 | 3.5804 | 0.424 | -0.167 | ✗ |
| bell | window | 9.3892 | 0.550 | 5.1641 | 0.450 | +0.146 | ✗ |
| exponential | mid | 8.9510 | 1.000 | 8.9510 | 0.000 | +1.083 | ✗ |
| exponential | window | 9.3892 | 1.000 | 9.3892 | 0.000 | +1.083 | ✗ |
| constant | mid | 8.9510 | 1.000 | 8.9510 | 0.000 | +1.083 | ✗ |
| constant | window | 9.3892 | 1.000 | 9.3892 | 0.000 | +1.083 | ✗ |

## llama-3.1-8b  (family=llama)

K_optimal_mid = 0.9542  | K_optimal_window = 1.0080

| Shape | K_source | K_optimal | mean f_max | K_peak_raw | std | ±baseline | Holds? |
|---|---|---|---|---|---|---|---|
| linear | mid | 0.9542 | 0.417 | 0.3976 | 0.413 | -0.132 | ✗ |
| linear | window | 1.0080 | 0.400 | 0.4032 | 0.424 | -0.167 | ✗ |
| cosine | mid | 0.9542 | 0.467 | 0.4453 | 0.379 | -0.028 | ✗ |
| cosine | window | 1.0080 | 0.433 | 0.4368 | 0.403 | -0.097 | ✗ |
| bell | mid | 0.9542 | 0.467 | 0.4453 | 0.386 | -0.028 | ✗ |
| bell | window | 1.0080 | 0.400 | 0.4032 | 0.424 | -0.167 | ✗ |
| exponential | mid | 0.9542 | 0.537 | 0.5129 | 0.370 | +0.120 | ✗ |
| exponential | window | 1.0080 | 0.500 | 0.5040 | 0.366 | +0.042 | ✗ |
| constant | mid | 0.9542 | 0.850 | 0.8110 | 0.000 | +0.771 | ✗ |
| constant | window | 1.0080 | 0.800 | 0.8064 | 0.000 | +0.667 | ✗ |

## llama-3.2-1b  (family=llama)

K_optimal_mid = 1.0622  | K_optimal_window = 1.0827

| Shape | K_source | K_optimal | mean f_max | K_peak_raw | std | ±baseline | Holds? |
|---|---|---|---|---|---|---|---|
| linear | mid | 1.0622 | 0.275 | 0.2921 | 0.175 | -0.427 | ✗ |
| linear | window | 1.0827 | 0.275 | 0.2977 | 0.175 | -0.427 | ✗ |
| cosine | mid | 1.0622 | 0.275 | 0.2921 | 0.175 | -0.427 | ✗ |
| cosine | window | 1.0827 | 0.275 | 0.2977 | 0.175 | -0.427 | ✗ |
| bell | mid | 1.0622 | 0.217 | 0.2301 | 0.165 | -0.549 | ✗ |
| bell | window | 1.0827 | 0.217 | 0.2346 | 0.165 | -0.549 | ✗ |
| exponential | mid | 1.0622 | 0.362 | 0.3851 | 0.297 | -0.245 | ✗ |
| exponential | window | 1.0827 | 0.375 | 0.4060 | 0.317 | -0.219 | ✗ |
| constant | mid | 1.0622 | 0.400 | 0.4249 | 0.000 | -0.167 | ✓ |
| constant | window | 1.0827 | 0.350 | 0.3789 | 0.000 | -0.271 | ✗ |

## llama-3.2-3b  (family=llama)

K_optimal_mid = 1.3019  | K_optimal_window = 1.3612

| Shape | K_source | K_optimal | mean f_max | K_peak_raw | std | ±baseline | Holds? |
|---|---|---|---|---|---|---|---|
| linear | mid | 1.3019 | 0.450 | 0.5858 | 0.000 | -0.062 | ✓ |
| linear | window | 1.3612 | 0.400 | 0.5445 | 0.000 | -0.167 | ✓ |
| cosine | mid | 1.3019 | 0.400 | 0.5207 | 0.000 | -0.167 | ✓ |
| cosine | window | 1.3612 | 0.400 | 0.5445 | 0.000 | -0.167 | ✓ |
| bell | mid | 1.3019 | 0.450 | 0.5858 | 0.000 | -0.062 | ✓ |
| bell | window | 1.3612 | 0.400 | 0.5445 | 0.000 | -0.167 | ✓ |
| exponential | mid | 1.3019 | 0.950 | 1.2368 | 0.000 | +0.979 | ✗ |
| exponential | window | 1.3612 | 0.900 | 1.2251 | 0.000 | +0.875 | ✗ |
| constant | mid | 1.3019 | 0.250 | 0.3255 | 0.000 | -0.479 | ✗ |
| constant | window | 1.3612 | 0.200 | 0.2722 | 0.000 | -0.583 | ✗ |

## qwen2.5-3b  (family=qwen2)

K_optimal_mid = 11.3183  | K_optimal_window = 11.4311

| Shape | K_source | K_optimal | mean f_max | K_peak_raw | std | ±baseline | Holds? |
|---|---|---|---|---|---|---|---|
| linear | mid | 11.3183 | 0.988 | 11.1768 | 0.022 | +1.057 | ✗ |
| linear | window | 11.4311 | 0.988 | 11.2882 | 0.022 | +1.057 | ✗ |
| cosine | mid | 11.3183 | 1.000 | 11.3183 | 0.000 | +1.083 | ✗ |
| cosine | window | 11.4311 | 1.000 | 11.4311 | 0.000 | +1.083 | ✗ |
| bell | mid | 11.3183 | 0.988 | 11.1768 | 0.022 | +1.057 | ✗ |
| bell | window | 11.4311 | 0.975 | 11.1453 | 0.043 | +1.031 | ✗ |
| exponential | mid | 11.3183 | 1.000 | 11.3183 | 0.000 | +1.083 | ✗ |
| exponential | window | 11.4311 | 1.000 | 11.4311 | 0.000 | +1.083 | ✗ |
| constant | mid | 11.3183 | 0.838 | 9.4790 | 0.227 | +0.745 | ✗ |
| constant | window | 11.4311 | 0.688 | 7.8589 | 0.198 | +0.432 | ✗ |

## qwen2.5-7b  (family=qwen2)

K_optimal_mid = 28.3019  | K_optimal_window = 28.5528

| Shape | K_source | K_optimal | mean f_max | K_peak_raw | std | ±baseline | Holds? |
|---|---|---|---|---|---|---|---|
| linear | mid | 28.3019 | 0.850 | 24.0566 | 0.106 | +0.771 | ✗ |
| linear | window | 28.5528 | 0.850 | 24.2699 | 0.106 | +0.771 | ✗ |
| cosine | mid | 28.3019 | 0.812 | 22.9953 | 0.108 | +0.693 | ✗ |
| cosine | window | 28.5528 | 0.812 | 23.1992 | 0.108 | +0.693 | ✗ |
| bell | mid | 28.3019 | 0.825 | 23.3490 | 0.103 | +0.719 | ✗ |
| bell | window | 28.5528 | 0.812 | 23.1992 | 0.108 | +0.693 | ✗ |
| exponential | mid | 28.3019 | 1.000 | 28.3019 | 0.000 | +1.083 | ✗ |
| exponential | window | 28.5528 | 1.000 | 28.5528 | 0.000 | +1.083 | ✗ |
| constant | mid | 28.3019 | 0.637 | 18.0424 | 0.213 | +0.328 | ✗ |
| constant | window | 28.5528 | 0.637 | 18.2024 | 0.213 | +0.328 | ✗ |
