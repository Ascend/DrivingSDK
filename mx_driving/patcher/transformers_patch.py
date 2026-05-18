# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# pylint: disable=undefined-variable
"""Transformers model patches for NPU optimization.

Patches:
- Qwen3RMSNorm: torch_npu.npu_rms_norm (transformers >= 4.51.0, including v5.x)
- Qwen3RoPE: torch_npu.npu_rotary_mul (transformers >= 4.51.0, including v5.x)
- FlashAttention: mask caching via ATTN_MASK_NPU_CACHE (transformers 4.51.x, 4.52.x only)
- FlashAttentionVarlen: mask caching via ATTN_MASK_NPU_CACHE (transformers 4.51.x, 4.52.x only)

Version compatibility (source verified):
  Qwen3: 4.51.x+ and 5.x (v5 removed position_ids from apply_rotary_pos_emb, patch compatible)
  FlashAttention: 4.51.x/4.52.x need patch (no mask caching); 4.53.x+ built-in caching, no patch needed
"""

import os
import importlib
from typing import List, Optional

from mx_driving.patcher.patch import AtomicPatch, BasePatch, Patch, with_imports
from mx_driving.patcher.version import get_version


# =============================================================================
# Version Detection
# =============================================================================


def _get_transformers_version() -> Optional[str]:
    return get_version("transformers")


def _parse_version(version_str: str) -> List[int]:
    try:
        parts = version_str.split('.')
        return [int(p.split('+')[0].split('-')[0]) for p in parts if p]
    except (ValueError, AttributeError):
        return []


def _is_transformers_compatible_for_qwen3() -> bool:
    """Qwen3 was added in 4.51.0. v5 removed position_ids from apply_rotary_pos_emb
    but our patch doesn't use it, so it remains compatible.
    """
    version = _get_transformers_version()
    if not version:
        return False

    parts = _parse_version(version)
    if len(parts) < 2:
        return False

    major, minor = parts[0], parts[1]

    if major == 4 and minor >= 51:
        return True
    if major >= 5:
        return True

    return False


def _is_transformers_compatible_for_flash_attention() -> bool:
    """FlashAttention patch provides mask caching (ATTN_MASK_NPU_CACHE).
    4.51.x/4.52.x: no caching in upstream, patch needed.
    4.53.x+: caching built-in, patch not needed.
    """
    version = _get_transformers_version()
    if not version:
        return False

    parts = _parse_version(version)
    if len(parts) < 2:
        return False

    major, minor = parts[0], parts[1]

    if major == 4 and minor in (51, 52):
        return True

    return False


def _is_qwen3_available() -> bool:
    try:
        importlib.import_module("transformers.models.qwen3.modeling_qwen3")
        return True
    except ImportError:
        return False


def _is_flash_attention_available() -> bool:
    try:
        importlib.import_module("transformers.modeling_flash_attention_utils")
        importlib.import_module("transformers.integrations.npu_flash_attention")
        return True
    except ImportError:
        return False


# =============================================================================
# Global State for Flash Attention
# =============================================================================

ATTN_MASK_NPU_CACHE = {}

TOP_LEFT_ALIGNED_CAUSAL_MASK_MODE = 2
DOWN_RIGHT_ALIGNED_CAUSAL_MASK_MODE = 3

SPARSE_MODE = int(os.getenv("NPU_FA2_SPARSE_MODE", default=str(DOWN_RIGHT_ALIGNED_CAUSAL_MASK_MODE)))


def get_attn_mask_npu(device):
    import torch

    if device not in ATTN_MASK_NPU_CACHE:
        ATTN_MASK_NPU_CACHE[device] = torch.triu(torch.ones([2048, 2048], device=device), diagonal=1).bool()
    return ATTN_MASK_NPU_CACHE[device]


# =============================================================================
# Qwen3 RMSNorm Patch
# =============================================================================


class Qwen3RMSNorm(Patch):
    """Replace Qwen3RMSNorm.forward with torch_npu.npu_rms_norm.

    Requirements: transformers >= 4.51.0 (including v5.x), torch_npu
    """

    name = "qwen3_rmsnorm"
    legacy_name = "qwen3_rmsnorm"
    target_module = "transformers.models.qwen3.modeling_qwen3"

    @staticmethod
    def precheck() -> bool:
        return _is_transformers_compatible_for_qwen3() and _is_qwen3_available()

    @staticmethod
    @with_imports("torch_npu")
    def forward(self, hidden_states):  # pylint: disable=bad-staticmethod-argument
        return torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.variance_epsilon)[0]  # noqa: F821

    @classmethod
    def patches(cls, options=None) -> List[BasePatch]:
        return [
            AtomicPatch(
                "transformers.models.qwen3.modeling_qwen3.Qwen3RMSNorm.forward",
                cls.forward,
                precheck=cls.precheck,
            ),
        ]


# =============================================================================
# Qwen3 RoPE Patch
# =============================================================================


class Qwen3RoPE(Patch):
    """Replace apply_rotary_pos_emb with torch_npu.npu_rotary_mul.

    v5 removed position_ids param but our patch doesn't use it, so compatible.

    Requirements: transformers >= 4.51.0 (including v5.x), torch_npu
    """

    name = "qwen3_rope"
    legacy_name = "qwen3_rope"
    target_module = "transformers.models.qwen3.modeling_qwen3"

    @staticmethod
    def precheck() -> bool:
        return _is_transformers_compatible_for_qwen3() and _is_qwen3_available()

    @staticmethod
    @with_imports("torch_npu")
    def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = torch_npu.npu_rotary_mul(q, cos, sin)  # noqa: F821
        k_embed = torch_npu.npu_rotary_mul(k, cos, sin)  # noqa: F821
        return q_embed, k_embed

    @classmethod
    def patches(cls, options=None) -> List[BasePatch]:
        return [
            AtomicPatch(
                "transformers.models.qwen3.modeling_qwen3.apply_rotary_pos_emb",
                cls.apply_rotary_pos_emb,
                precheck=cls.precheck,
            ),
        ]


# =============================================================================
# Flash Attention Patches
# =============================================================================


class FlashAttention(Patch):
    """Replace flash_attn_func with torch_npu.npu_fusion_attention, adding mask caching.

    4.51.x/4.52.x: upstream creates mask on every call, patch adds caching.
    4.53.x+: upstream already caches mask, patch not needed.

    Requirements: transformers >= 4.51.0, < 4.53.0, torch_npu
    """

    name = "flash_attention"
    legacy_name = "flash_attention"
    target_module = "transformers.integrations.npu_flash_attention"

    @staticmethod
    def precheck() -> bool:
        return _is_transformers_compatible_for_flash_attention() and _is_flash_attention_available()

    @staticmethod
    @with_imports("torch_npu", "torch")
    def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **kwargs):
        keep_prob = 1.0 - dropout_p

        if not causal:
            head_num = q.shape[2]
            output = torch_npu.npu_fusion_attention(  # noqa: F821
                q, k, v, head_num, "BSND", keep_prob=keep_prob, scale=softmax_scale
            )[0]
        else:
            attn_mask_npu = get_attn_mask_npu(q.device)
            head_num = q.shape[2]
            output = torch_npu.npu_fusion_attention(  # noqa: F821
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

    @classmethod
    def patches(cls, options=None) -> List[BasePatch]:
        return [
            AtomicPatch(
                "transformers.modeling_flash_attention_utils.flash_attn_func",
                cls.flash_attn_func,
                precheck=cls.precheck,
                aliases=[
                    "transformers.integrations.npu_flash_attention.flash_attn_func",
                    "transformers.integrations.npu_flash_attention.npu_flash_attn_func",
                    "transformers.modeling_flash_attention_utils.npu_flash_attn_func",
                ],
            ),
        ]


class FlashAttentionVarlen(Patch):
    """Replace flash_attn_varlen_func with torch_npu.npu_fusion_attention, adding mask caching.

    4.51.x/4.52.x: upstream creates mask on every call, patch adds caching.
    4.53.x+: upstream already caches mask, patch not needed.

    Requirements: transformers >= 4.51.0, < 4.53.0, torch_npu
    """

    name = "flash_attention_varlen"
    legacy_name = "flash_attention_varlen"
    target_module = "transformers.integrations.npu_flash_attention"

    @staticmethod
    def precheck() -> bool:
        return _is_transformers_compatible_for_flash_attention() and _is_flash_attention_available()

    @staticmethod
    @with_imports("torch_npu", "torch")
    def flash_attn_varlen_func(
        q, k, v, cu_seqlens_q, cu_seqlens_k, dropout_p=0.0, softmax_scale=None, causal=False, **kwargs
    ):
        keep_prob = 1.0 - dropout_p

        if not causal:
            head_num = q.shape[1]
            output = torch_npu.npu_fusion_attention(  # noqa: F821
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
            output = torch_npu.npu_fusion_attention(  # noqa: F821
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

    @classmethod
    def patches(cls, options=None) -> List[BasePatch]:
        return [
            AtomicPatch(
                "transformers.modeling_flash_attention_utils.flash_attn_varlen_func",
                cls.flash_attn_varlen_func,
                precheck=cls.precheck,
                aliases=[
                    "transformers.integrations.npu_flash_attention.flash_attn_varlen_func",
                    "transformers.integrations.npu_flash_attention.npu_flash_attn_varlen_func",
                    "transformers.modeling_flash_attention_utils.npu_flash_attn_varlen_func",
                ],
            ),
        ]


# =============================================================================
# Composite Patch
# =============================================================================


class TransformersNPU(Patch):
    """Composite patch: Qwen3RMSNorm + Qwen3RoPE (>=4.51.0) + FlashAttention (4.51.x/4.52.x).

    FlashAttention patches only apply to 4.51.x/4.52.x (4.53.x+ has built-in mask caching).

    Requirements: transformers >= 4.51.0, torch_npu
    """

    name = "transformers_npu"
    legacy_name = "transformers_npu"
    target_module = "transformers"

    @classmethod
    def patches(cls, options=None) -> List[BasePatch]:
        all_patches = []
        if _is_qwen3_available():
            all_patches.extend(Qwen3RMSNorm.patches(options))
            all_patches.extend(Qwen3RoPE.patches(options))
        if _is_flash_attention_available():
            all_patches.extend(FlashAttention.patches(options))
            all_patches.extend(FlashAttentionVarlen.patches(options))
        return all_patches
