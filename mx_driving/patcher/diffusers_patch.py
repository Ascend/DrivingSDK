# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""Diffusers model patches for NPU optimization.

This module provides patches for diffusers library models, including:
- AttnProcessor2_0 attention mask shape fix

IMPORTANT: These patches are designed for diffusers == 0.35.1.
Using them with other versions may cause unexpected behavior.
"""

from typing import List

from mx_driving.patcher.patch import AtomicPatch, BasePatch, Patch
from mx_driving.patcher.version import get_version


# =============================================================================
# Version Detection for Diffusers
# =============================================================================


def _is_diffusers_0_35_1() -> bool:
    """Check if diffusers version is exactly 0.35.1."""
    version = get_version("diffusers")
    return version == "0.35.1"


# =============================================================================
# Attention Processor Patch
# =============================================================================


class AttnProcessor2_0(Patch):
    """AttnProcessor2_0 attention mask shape fix patch.

    Fixes attention_mask shape issue when size(-2) == 1 by expanding it
    to match the query length.

    Requirements:
        - diffusers == 0.35.1
    """

    name = "attn_processor_2_0"
    legacy_name = "attn_processor_2_0"
    target_module = "diffusers.models.attention_processor"

    @staticmethod
    def precheck() -> bool:
        """Check if diffusers version is compatible."""
        return _is_diffusers_0_35_1()

    @staticmethod
    def _wrap_call(original_call):
        """Wrap the original __call__ method with attention mask fix."""
        import torch.nn.functional as F

        original_sdpa = F.scaled_dot_product_attention

        def patched_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
            """
            Wrapper for scaled_dot_product_attention that fixes attention_mask shape.

            This is the ONLY fix needed: expand attention_mask when size(-2) == 1
            """
            if attn_mask is not None and attn_mask.size(-2) == 1:
                query_len = query.size(2)
                attn_mask = attn_mask.expand(-1, -1, query_len, -1)
            return original_sdpa(
                query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale
            )

        def patched_call(  # pylint: disable=keyword-arg-before-vararg
            self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs
        ):
            # Temporarily replace F.scaled_dot_product_attention
            F.scaled_dot_product_attention = patched_sdpa

            try:
                result = original_call(
                    self, attn, hidden_states, encoder_hidden_states, attention_mask, temb, *args, **kwargs
                )
            finally:
                # Always restore original function
                F.scaled_dot_product_attention = original_sdpa

            return result

        return patched_call

    @classmethod
    def patches(cls, options=None) -> List[BasePatch]:
        return [
            AtomicPatch(
                "diffusers.models.attention_processor.AttnProcessor2_0.__call__",
                target_wrapper=cls._wrap_call,
                precheck=cls.precheck,
            ),
        ]


# =============================================================================
# Composite Patch for Diffusers Models
# =============================================================================


class DiffusersNPU(Patch):
    """Composite patch for diffusers models with NPU optimization.

    This includes:
    - AttnProcessor2_0 attention mask fix

    Requirements:
        - diffusers == 0.35.1
    """

    name = "diffusers_npu"
    legacy_name = "diffusers_npu"
    target_module = "diffusers"

    @classmethod
    def patches(cls, options=None) -> List[BasePatch]:
        return AttnProcessor2_0.patches(options)
