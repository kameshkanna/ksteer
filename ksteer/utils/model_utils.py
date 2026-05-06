"""Utilities for model loading and architecture-agnostic layer access."""

import logging
from typing import Iterator, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)

# Maps HuggingFace model_type → dotted path to the list of transformer blocks
LAYERS_ATTR_MAP: dict[str, str] = {
    "llama": "model.layers",
    "mistral": "model.layers",
    "mixtral": "model.layers",
    "qwen2": "model.layers",
    "qwen2_moe": "model.layers",
    "gemma": "model.layers",
    "gemma2": "model.layers",
    "gpt2": "transformer.h",
    "gpt_neox": "gpt_neox.layers",
    "falcon": "transformer.h",
    "phi": "model.layers",
    "phi3": "model.layers",
}


def resolve_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(
    model_name_or_path: str,
    device: Optional[str] = None,
    torch_dtype: torch.dtype = torch.float16,
    trust_remote_code: bool = False,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a causal LM and its tokenizer.

    Uses device_map="auto" for all GPU/MPS targets so accelerate distributes
    layers across available VRAM and spills the remainder to CPU RAM. This
    handles large models (70B, 72B) on a single A100 + host RAM without
    quantization, preserving full float16 activation fidelity for K_l measurement.
    """
    resolved_device = resolve_device(device)
    logger.info("Loading %s  target=%s", model_name_or_path, resolved_device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        clean_up_tokenization_spaces=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict = {
        "dtype": torch_dtype,
        "trust_remote_code": trust_remote_code,
    }
    if resolved_device.type == "cpu":
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)
        model = model.to(resolved_device)
    else:
        load_kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)

    # Force right-padding so attention_mask positions are consistent across batch
    # items regardless of model default (Qwen2 defaults to left-pad).
    tokenizer.padding_side = "right"

    model.eval()
    logger.info(
        "Loaded: family=%s  hidden_dim=%d  num_layers=%d  dtype=%s",
        model.config.model_type,
        get_hidden_dim(model),
        get_num_layers(model),
        next(model.parameters()).dtype,
    )
    return model, tokenizer


def get_model_family(model: PreTrainedModel) -> str:
    return model.config.model_type.lower()


def get_hidden_dim(model: PreTrainedModel) -> int:
    cfg = model.config
    for attr in ("hidden_size", "d_model", "n_embd"):
        if hasattr(cfg, attr):
            return int(getattr(cfg, attr))
    raise ValueError(f"Cannot infer hidden_dim for model_type={cfg.model_type!r}")


def get_num_layers(model: PreTrainedModel) -> int:
    cfg = model.config
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        if hasattr(cfg, attr):
            return int(getattr(cfg, attr))
    raise ValueError(f"Cannot infer num_layers for model_type={cfg.model_type!r}")


def iter_transformer_layers(model: PreTrainedModel) -> Iterator[Tuple[int, nn.Module]]:
    """Yield (layer_index, layer_module) for every transformer block in the model."""
    family = get_model_family(model)
    layers_attr = LAYERS_ATTR_MAP.get(family)

    if layers_attr is None:
        raise ValueError(
            f"Unknown model family {family!r}. "
            f"Add it to LAYERS_ATTR_MAP in ksteer/utils/model_utils.py."
        )

    obj = model
    for part in layers_attr.split("."):
        obj = getattr(obj, part)

    for idx, layer in enumerate(obj):
        yield idx, layer


def get_layer_by_index(model: PreTrainedModel, layer_idx: int) -> nn.Module:
    """Return the transformer block at the given index."""
    for idx, layer in iter_transformer_layers(model):
        if idx == layer_idx:
            return layer
    raise IndexError(f"Layer {layer_idx} not found in model with {get_num_layers(model)} layers.")
