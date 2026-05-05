"""Utilities for model loading and architecture-agnostic layer access."""

import logging
from typing import Iterator, Literal, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizer

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
    quantize: Optional[Literal["4bit", "8bit"]] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a causal LM and its tokenizer, routing to the best available device.

    Args:
        quantize: "4bit" or "8bit" for bitsandbytes quantization. Required for
                  models that exceed single-GPU VRAM (e.g. 70B on 80GB A100).
                  Activations and norms are measured in the dequantized dtype so
                  K_l values remain valid.
    """
    resolved_device = resolve_device(device)
    logger.info("Loading %s on %s  quantize=%s", model_name_or_path, resolved_device, quantize or "none")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict = {"trust_remote_code": trust_remote_code}

    if quantize == "4bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        # device_map="auto" is required for bitsandbytes and handles CPU offload automatically
        load_kwargs["device_map"] = "auto"
    elif quantize == "8bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["dtype"] = torch_dtype
        if resolved_device.type != "cpu":
            load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)

    if resolved_device.type == "cpu" and quantize is None:
        model = model.to(resolved_device)

    # Force right-padding for all models so attention_mask positions are consistent
    # across batch items regardless of model default (Qwen2 defaults to left-pad).
    tokenizer.padding_side = "right"

    model.eval()
    logger.info(
        "Loaded: family=%s  hidden_dim=%d  num_layers=%d  pad_side=%s  dtype=%s",
        model.config.model_type,
        get_hidden_dim(model),
        get_num_layers(model),
        tokenizer.padding_side,
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
