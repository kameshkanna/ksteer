"""Architecture-agnostic model utilities."""

import logging
from typing import Iterator, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

logger = logging.getLogger(__name__)


def load_model(
    model_id: str,
    device: str | None = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device or ("auto" if torch.cuda.is_available() else "cpu"),
        trust_remote_code=True,
    )
    model.eval()
    logger.info("Loaded %s  layers=%d  d=%d",
                model_id, model.config.num_hidden_layers, model.config.hidden_size)
    return model, tokenizer


def get_layers(model: PreTrainedModel) -> torch.nn.ModuleList:
    """Return the transformer block list, architecture-agnostic."""
    # Qwen2, Llama, Gemma2, Mistral
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # Falcon, GPT-NeoX
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    # GPT-2
    if hasattr(model, "transformer") and hasattr(model.transformer, "blocks"):
        return model.transformer.blocks
    raise RuntimeError(
        f"Cannot locate transformer blocks in {type(model).__name__}. "
        "Add the attribute path to ksteer/utils.py:get_layers()."
    )


def get_layer(model: PreTrainedModel, idx: int) -> torch.nn.Module:
    return get_layers(model)[idx]


def iter_layers(model: PreTrainedModel) -> Iterator[Tuple[int, torch.nn.Module]]:
    yield from enumerate(get_layers(model))


def num_layers(model: PreTrainedModel) -> int:
    return model.config.num_hidden_layers


def hidden_dim(model: PreTrainedModel) -> int:
    return model.config.hidden_size
