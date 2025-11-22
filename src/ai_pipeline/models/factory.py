"""Model factory utilities."""

from __future__ import annotations

import torch

from transformers import AutoModelForCausalLM
from typing import Optional

from ai_pipeline.config.schema import ModelConfig


DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def _resolve_dtype(name: str) -> Optional[torch.dtype]:
    """Resolve dtype string to torch dtype."""
    return DTYPE_MAP.get(name.lower())


def create_causal_lm(model_cfg: ModelConfig) -> AutoModelForCausalLM:
    """Create causal LM from model config."""
    torch_dtype = _resolve_dtype(model_cfg.torch_dtype)

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_name,
        torch_dtype=torch_dtype,
        device_map=None,
    )

    if model_cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if model_cfg.use_flash_attention:
        # TODO: integrate with model/hardware-specific flash attention.
        pass

    return model
