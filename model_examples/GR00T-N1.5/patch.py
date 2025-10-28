# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import os
import warnings
from typing import Callable, Dict, List, Optional, Set

from types import ModuleType
from typing import Dict
from typing import Tuple
from typing import Optional
import torch
import torch_npu


def rmsnorm_patch(modeling_qwen3: ModuleType, options: Dict):
    def rmsnorm_forward(self, hidden_states):
        # change the code
        # using npu_rms_norm
        return torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.variance_epsilon)[0]

    if hasattr(modeling_qwen3, "Qwen3RMSNorm"):
        modeling_qwen3.Qwen3RMSNorm.forward = rmsnorm_forward
    else:
        warnings.warn(f"Failed to apply patch RMSNorm to module modeling_qwen3")


def rope_patch(modeling_qwen3: ModuleType, options: Dict):

    # pylint: disable=huawei-too-many-arguments
    def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        # change the code
        # using npu_rotary_mul
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
        k_embed = torch_npu.npu_rotary_mul(k, cos, sin)

        return q_embed, k_embed
        
    if hasattr(modeling_qwen3, "apply_rotary_pos_emb"):
        modeling_qwen3.apply_rotary_pos_emb = apply_rotary_pos_emb
    else:
        warnings.warn(f"Failed to apply patch apply_rotary_pos_emb to module modeling_qwen3")


ATTN_MASK_NPU_CACHE = {}

def get_attn_mask_npu(device):
    """Get or create attention mask for the specified device."""
    if device not in ATTN_MASK_NPU_CACHE:
        ATTN_MASK_NPU_CACHE[device] = torch.triu(torch.ones([2048, 2048], device=device), diagonal=1).bool()
    return ATTN_MASK_NPU_CACHE[device]

TOP_LEFT_ALIGNED_CAUSAL_MASK_MODE = 2
DOWN_RIGHT_ALIGNED_CAUSAL_MASK_MODE = 3

SPARSE_MODE = int(os.getenv("NPU_FA2_SPARSE_MODE", default=DOWN_RIGHT_ALIGNED_CAUSAL_MASK_MODE))


def flash_attn_func_patch(npu_flash_attention: ModuleType, options: Dict):

    # pylint: disable=huawei-too-many-arguments
    def npu_flash_attn_func(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        **kwargs,
    ):
        # change the code
        # using get_attn_mask_npu
        keep_prob = 1.0 - dropout_p

        if not causal:
            head_num = q.shape[2]
            output = torch_npu.npu_fusion_attention(q, k, v, head_num, "BSND", keep_prob=keep_prob, scale=softmax_scale)[0]
        else:
            attn_mask_npu = get_attn_mask_npu(q.device)
            head_num = q.shape[2]
            output = torch_npu.npu_fusion_attention(
                q,
                k,
                v,
                head_num,
                "BSND",
                keep_prob=keep_prob,
                scale=softmax_scale,
                atten_mask=attn_mask_npu,
                sparse_mode=SPARSE_MODE,
            )[0]

        return output

    if hasattr(npu_flash_attention, "flash_attn_func"):
        npu_flash_attention.flash_attn_func = npu_flash_attn_func
    elif hasattr(npu_flash_attention, "npu_flash_attn_func"):
        npu_flash_attention.npu_flash_attn_func = npu_flash_attn_func
    else:
        warnings.warn(f"Failed to apply patch flash_attn_func to module npu_flash_attention")


def flash_attn_varlen_func_patch(npu_flash_attention: ModuleType, options: Dict):

    # pylint: disable=huawei-too-many-arguments
    def npu_flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        **kwargs,
    ):
        # change the code
        # using get_attn_mask_npu
        keep_prob = 1.0 - dropout_p

        if not causal:
            head_num = q.shape[1]
            output = torch_npu.npu_fusion_attention(
                q,
                k,
                v,
                head_num,
                pse=None,
                atten_mask=None,
                scale=softmax_scale,
                keep_prob=keep_prob,
                input_layout="TND",
                actual_seq_qlen=tuple(cu_seqlens_q[1:].cpu().numpy().tolist()),
                actual_seq_kvlen=tuple(cu_seqlens_k[1:].cpu().numpy().tolist()),
            )[0]
        else:
            attn_mask_npu = get_attn_mask_npu(q.device)
            head_num = q.shape[1]
            output = torch_npu.npu_fusion_attention(
                q,
                k,
                v,
                head_num,
                pse=None,
                padding_mask=None,
                atten_mask=attn_mask_npu,
                scale=softmax_scale,
                keep_prob=keep_prob,
                input_layout="TND",
                actual_seq_qlen=tuple(cu_seqlens_q[1:].cpu().numpy().tolist()),
                actual_seq_kvlen=tuple(cu_seqlens_k[1:].cpu().numpy().tolist()),
                sparse_mode=SPARSE_MODE,
            )[0]

        return output

    if hasattr(npu_flash_attention, "flash_attn_varlen_func"):
        npu_flash_attention.flash_attn_varlen_func = npu_flash_attn_varlen_func
    elif hasattr(npu_flash_attention, "npu_flash_attn_varlen_func"):
        npu_flash_attention.npu_flash_attn_varlen_func = npu_flash_attn_varlen_func
    else:
        warnings.warn(f"Failed to apply patch flash_attn_varlen_func to module npu_flash_attention")


# get the patch for gr00t
def generate_patcher_builder():
    from transformers.integrations import npu_flash_attention
    from transformers import modeling_flash_attention_utils
    from transformers.models.qwen3 import modeling_qwen3

    flash_attn_func_patch(npu_flash_attention, {})
    flash_attn_varlen_func_patch(npu_flash_attention, {})
    flash_attn_func_patch(modeling_flash_attention_utils, {})
    flash_attn_varlen_func_patch(modeling_flash_attention_utils, {})
    rope_patch(modeling_qwen3, {})
    rmsnorm_patch(modeling_qwen3, {})