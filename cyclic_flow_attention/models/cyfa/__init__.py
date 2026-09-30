from __future__ import annotations

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_cyfa import CyclicFlowAttentionConfig
from .modeling_cyfa import (
    CyclicFlowAttentionForCausalLM,
    CyclicFlowAttentionModel,
    CyclicFlowAttentionPreTrainedModel,
)

__all__ = [
    "CyclicFlowAttentionConfig",
    "CyclicFlowAttentionForCausalLM",
    "CyclicFlowAttentionModel",
    "CyclicFlowAttentionPreTrainedModel",
]

AutoConfig.register(CyclicFlowAttentionConfig.model_type, CyclicFlowAttentionConfig, exist_ok=True)
AutoModel.register(CyclicFlowAttentionConfig, CyclicFlowAttentionModel, exist_ok=True)
AutoModelForCausalLM.register(
    CyclicFlowAttentionConfig,
    CyclicFlowAttentionForCausalLM,
    exist_ok=True,
)
