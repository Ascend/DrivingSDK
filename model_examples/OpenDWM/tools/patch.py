
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright (c) OpenMMLab. All rights reserved.
# Copyright 2021 The OpenAI Team Authors and The HuggingFace Team. All rights reserved.
import importlib
import os
import math
from types import ModuleType
from typing import Dict
from typing import Tuple
from typing import Optional
import torch
import torch_npu
import mx_driving
from mx_driving.patcher import PatcherBuilder, Patch


def RMSNorm_forward(normalization: ModuleType, options: Dict):
    def _RMSNorm_forward(self, hidden_states):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        return torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.eps)[0]

    if hasattr(normalization, "RMSNorm"):
        normalization.RMSNorm.forward = _RMSNorm_forward


def CLIPSdpaAttn_forward(attn: ModuleType, options: Dict):
    def _CLIPSdpaAttn_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if output_attentions:
            logger.warning_once(
                "CLIPModel is using CLIPSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not "
                "support `output_attentions=True`. Falling back to the manual attention implementation, but specifying "
                "the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can "
                'be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                causal_attention_mask=causal_attention_mask,
                output_attentions=output_attentions,
            )

        # CLIP text model uses both `causal_attention_mask` and `attention_mask`
        if attention_mask is not None and causal_attention_mask is not None:
            attn_mask = attention_mask + causal_attention_mask

        elif causal_attention_mask is not None:
            attn_mask = causal_attention_mask
        else:
            attn_mask = attention_mask
        
        bsz, tgt_len, embed_dim = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # NOTE:attn_mask取反
        dropout_p = self.dropout if self.training else 0.0
        attn_output = torch_npu.npu_fusion_attention(query_states, key_states, value_states, head_num=self.num_heads, pse=None, atten_mask=attn_mask.bool(),
                                        input_layout="BSH", scale=1.0 / math.sqrt(query_states.shape[-1]), sparse_mode=1,
                                        keep_prob=1 - dropout_p)[0]
        
        attn_output = attn_output.view(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)
        return attn_output, None

    if hasattr(attn, "CLIPSdpaAttention"):
        attn.CLIPSdpaAttention.forward = _CLIPSdpaAttn_forward


# get the patch for OpenDWM
def generate_patcher_builder():
    opendwm_patcher_builder = (
        PatcherBuilder()
        .add_module_patch("diffusers.models.normalization", Patch(RMSNorm_forward))
        .add_module_patch("transformers.models.clip.modeling_clip", Patch(CLIPSdpaAttn_forward))
    )
    return opendwm_patcher_builder
