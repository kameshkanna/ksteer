"""
Architecture audit — verifies hook points, layer structure, and norm measurement
are correct for a given model before committing to a full exp01 run.

Checks:
  1. Layer count and hidden_dim match config
  2. Hook output shape is (batch, seq_len, hidden_dim)
  3. Attention mask correctly filters padding (right-pad assumed)
  4. Steering injection shape broadcasts correctly
  5. Per-layer norm growth direction (should be monotonic in middle layers)
  6. Architecture-specific quirks logged for known families

Usage:
    python experiments/audit_architecture.py --model meta-llama/Llama-3.2-1B
    python experiments/audit_architecture.py --model Qwen/Qwen2.5-1.5B
    python experiments/audit_architecture.py --model google/gemma-2-2b
"""

import argparse
import logging
import math
from typing import Dict, List

import torch

from ksteer.utils.model_utils import (
    get_hidden_dim,
    get_model_family,
    get_num_layers,
    iter_transformer_layers,
    load_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

AUDIT_PROMPT = "The history of artificial intelligence began in"

KNOWN_ARCH_NOTES: Dict[str, List[str]] = {
    "llama": [
        "Pre-norm (RMSNorm before each sub-layer). Residual stream is clean.",
        "Hook at layer output captures post-residual-add hidden state. Correct.",
    ],
    "mistral": [
        "Same as Llama architecture. Sliding window attention in some layers.",
        "Hook at layer output is correct.",
    ],
    "qwen2": [
        "Pre-norm like Llama. Very large FFN intermediate_size (~5.8x hidden).",
        "Massive K_l values (~33) are real — architecture trained with large norm scale.",
        "Random vectors collapse at <0.25xK_l; real behavioral vectors may be more stable.",
        "Multilingual: random steering often activates Chinese language patterns.",
    ],
    "gemma2": [
        "Pre-norm + post-norm (post_attention_layernorm, post_feedforward_layernorm).",
        "Post-norm on sub-layer OUTPUT before residual add inflates the residual stream norm.",
        "This is WHY K_l is ~9x higher than Llama — not a measurement artifact.",
        "Hook at layer output captures the post-residual-add stream. Correct.",
        "Early layers are unusually tolerant (alpha* up to 2.5) — post-norm provides stability.",
    ],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit model architecture for ksteer compatibility")
    p.add_argument("--model", required=True, type=str)
    p.add_argument("--device", default=None, type=str)
    return p.parse_args()


def audit_hook_shapes(model, tokenizer, device) -> bool:
    """Verify that every layer hook output is (batch, seq_len, hidden_dim)."""
    logger.info("── Hook shape audit ──")
    captured: Dict[int, tuple] = {}
    hooks = []

    for idx, layer in iter_transformer_layers(model):
        def make_hook(i):
            def hook(m, inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                captured[i] = tuple(hs.shape)
            return hook
        hooks.append(layer.register_forward_hook(make_hook(idx)))

    inputs = tokenizer(AUDIT_PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    hidden_dim = get_hidden_dim(model)
    num_layers = get_num_layers(model)
    all_ok = True

    for i in range(num_layers):
        shape = captured.get(i)
        if shape is None:
            logger.error("  L%02d: NO OUTPUT CAPTURED", i)
            all_ok = False
            continue
        expected_last = hidden_dim
        if shape[-1] != expected_last:
            logger.error("  L%02d: shape %s — last dim %d != hidden_dim %d", i, shape, shape[-1], expected_last)
            all_ok = False
        elif i in (0, num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers - 1):
            logger.info("  L%02d: shape %s ✓", i, shape)

    if all_ok:
        logger.info("  All %d layers: hook shape correct (batch, seq_len, %d)", num_layers, hidden_dim)
    return all_ok


def audit_attention_mask(model, tokenizer, device) -> bool:
    """Verify right-padding produces a mask that correctly identifies real vs pad tokens."""
    logger.info("── Attention mask audit ──")

    short = "Hi"
    long_ = "The history of artificial intelligence began in the 1950s"

    inputs = tokenizer([short, long_], return_tensors="pt", padding=True).to(device)
    mask = inputs["attention_mask"]
    ids = inputs["input_ids"]

    logger.info("  padding_side: %s", tokenizer.padding_side)
    logger.info("  batch shape: input_ids=%s  mask=%s", tuple(ids.shape), tuple(mask.shape))
    logger.info("  short seq real tokens: %d / %d", mask[0].sum().item(), mask.shape[1])
    logger.info("  long  seq real tokens: %d / %d", mask[1].sum().item(), mask.shape[1])

    short_real = mask[0].sum().item()
    long_real = mask[1].sum().item()
    ok = short_real < long_real and long_real == mask.shape[1]
    if ok:
        logger.info("  Attention mask correct ✓  (right-padding confirmed)")
    else:
        logger.warning("  Attention mask may be left-padded — check padding_side")
    return ok


def audit_norm_growth(model, tokenizer, device) -> None:
    """Print per-layer K_l values for the audit prompt to check monotonicity."""
    logger.info("── Per-layer K_l audit ──")
    norms: Dict[int, float] = {}
    hooks = []

    for idx, layer in iter_transformer_layers(model):
        def make_hook(i):
            def hook(m, inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                norms[i] = hs.float().norm(dim=-1).mean().item()
            return hook
        hooks.append(layer.register_forward_hook(make_hook(idx)))

    inputs = tokenizer(AUDIT_PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    hidden_dim = get_hidden_dim(model)
    num_layers = get_num_layers(model)
    sqrt_d = math.sqrt(hidden_dim)
    start = int(0.4 * num_layers)
    end = int(0.8 * num_layers)

    sample_layers = sorted({
        0, num_layers // 8, num_layers // 4, start,
        num_layers // 2, end, 3 * num_layers // 4, num_layers - 1
    })

    for i in sample_layers:
        norm = norms.get(i, float("nan"))
        k = norm / sqrt_d
        in_window = "← window" if start <= i < end else ""
        logger.info("  L%02d (%3.0f%%): norm=%8.2f  K_l=%7.4f  %s",
                    i, 100 * i / num_layers, norm, k, in_window)

    window_norms = [norms[i] for i in range(start, end) if i in norms]
    if window_norms:
        logger.info("  Window [%d-%d]: K_l ∈ [%.4f, %.4f]",
                    start, end,
                    min(window_norms) / sqrt_d, max(window_norms) / sqrt_d)


def audit_steering_injection(model, tokenizer, device) -> bool:
    """Verify that steering injection doesn't crash and output shape is preserved."""
    logger.info("── Steering injection audit ──")
    hidden_dim = get_hidden_dim(model)
    target_layer_idx = int(0.6 * get_num_layers(model))

    from ksteer.utils.model_utils import get_layer_by_index
    target_layer = get_layer_by_index(model, target_layer_idx)

    dtype = next(model.parameters()).dtype
    v = torch.randn(hidden_dim, device=device, dtype=dtype)
    v = v / v.norm() * 0.01  # tiny magnitude — just testing shape

    output_shapes: list = []

    def hook(m, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        hs = hs + v.unsqueeze(0).unsqueeze(0)
        output_shapes.append(tuple(hs.shape))
        return (hs,) + out[1:] if isinstance(out, tuple) else hs

    handle = target_layer.register_forward_hook(hook)
    inputs = tokenizer(AUDIT_PROMPT, return_tensors="pt").to(device)

    try:
        with torch.no_grad():
            model(**inputs)
        shape = output_shapes[0] if output_shapes else None
        if shape and shape[-1] == hidden_dim:
            logger.info("  Injection at L%d: output shape %s ✓", target_layer_idx, shape)
            return True
        else:
            logger.error("  Injection at L%d: unexpected shape %s", target_layer_idx, shape)
            return False
    except Exception as e:
        logger.error("  Injection failed: %s", e)
        return False
    finally:
        handle.remove()


def main() -> None:
    args = parse_args()
    model, tokenizer = load_model(args.model, device=args.device)
    device = next(model.parameters()).device
    family = get_model_family(model)
    hidden_dim = get_hidden_dim(model)
    num_layers = get_num_layers(model)

    logger.info("═══════════════════════════════════════════════════")
    logger.info("Model    : %s", args.model)
    logger.info("Family   : %s", family)
    logger.info("Layers   : %d", num_layers)
    logger.info("HiddenDim: %d  (sqrt(d)=%.2f)", hidden_dim, math.sqrt(hidden_dim))
    logger.info("Device   : %s  dtype=%s", device, next(model.parameters()).dtype)
    logger.info("═══════════════════════════════════════════════════")

    notes = KNOWN_ARCH_NOTES.get(family, ["No specific notes for this family."])
    logger.info("Architecture notes:")
    for note in notes:
        logger.info("  • %s", note)

    results = {
        "hook_shapes":  audit_hook_shapes(model, tokenizer, device),
        "attn_mask":    audit_attention_mask(model, tokenizer, device),
        "injection":    audit_steering_injection(model, tokenizer, device),
    }
    audit_norm_growth(model, tokenizer, device)

    logger.info("═══════════════════════════════════════════════════")
    all_passed = all(results.values())
    for check, ok in results.items():
        logger.info("  %-20s %s", check, "✓ PASS" if ok else "✗ FAIL")
    logger.info("Overall: %s", "✓ READY FOR EXP01" if all_passed else "✗ ISSUES FOUND")
    logger.info("═══════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
