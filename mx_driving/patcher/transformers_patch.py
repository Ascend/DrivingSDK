# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# pylint: disable=undefined-variable
"""Transformers model patches for NPU optimization.

Patches:
- Qwen3RMSNorm: torch_npu.npu_rms_norm (transformers >= 4.51.0, including v5.x)
- Qwen3RoPE: torch_npu.npu_rotary_mul (transformers >= 4.51.0, including v5.x)
- Qwen3VLTextRMSNorm: torch_npu.npu_rms_norm for Qwen3-VL text model (transformers >= 4.57.0, including v5.x)
- Qwen3VLRoPE: torch_npu.npu_rotary_mul for Qwen3-VL vision rotary (transformers >= 4.57.0, including v5.x)
- Qwen3VL_get_placeholder_mask: fixed get_placeholder_mask for Qwen3-VL (transformers >= 4.57.0, including v5.x)
- BaseImageProcessorFast_preprocess: force device to NPU in image preprocessor (transformers == 4.57.0)
- Qwen2VL_preprocess: optimized vision preprocessing for Qwen3-VL using Qwen2-VL processor (transformers == 4.57.0)
- FlashAttention: mask caching via ATTN_MASK_NPU_CACHE (transformers 4.51.x, 4.52.x only)
- FlashAttentionVarlen: mask caching via ATTN_MASK_NPU_CACHE (transformers 4.51.x, 4.52.x only)

Version compatibility (source verified):
  Qwen3: 4.51.x+ and 5.x (v5 removed position_ids from apply_rotary_pos_emb, patch compatible)
  Qwen3-VL: 4.57.x+ and 5.x (vision patches compatible with the same NPU ops)
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


def _is_transformers_compatible_for_preprocess() -> bool:
    """Check if transformers version supports preprocess patches (== 4.57.0)."""
    version = _get_transformers_version()
    if not version:
        return False

    parts = _parse_version(version)
    if len(parts) < 2:
        return False

    major, minor = parts[0], parts[1]

    if major == 4 and minor == 57:
        return True

    return False


def _is_transformers_compatible_for_qwen3_vl() -> bool:
    """Check if transformers version supports Qwen3-VL patches (>= 4.57.0 or 5.x)."""
    version = _get_transformers_version()
    if not version:
        return False

    parts = _parse_version(version)
    if len(parts) < 2:
        return False

    major, minor = parts[0], parts[1]

    if major == 4 and minor >= 57:
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


def _is_qwen3_vl_available() -> bool:
    try:
        importlib.import_module("transformers.models.qwen3_vl.modeling_qwen3_vl")
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
# Qwen3-VL Patches (activated when _is_qwen3_vl_available is True)
# =============================================================================


class Qwen3VLTextRMSNorm(Patch):
    """Replace Qwen3VLTextRMSNorm.forward with torch_npu.npu_rms_norm for Qwen3-VL text model.

    Requirements: transformers >= 4.57.0 (including v5.x), torch_npu
    """

    name = "qwen3vltext_rmsnorm"
    legacy_name = "qwen3vltext_rmsnorm"
    target_module = "transformers.models.qwen3_vl.modeling_qwen3_vl"

    @staticmethod
    def precheck() -> bool:
        return _is_transformers_compatible_for_qwen3_vl() and _is_qwen3_vl_available()

    @staticmethod
    @with_imports("torch_npu")
    def forward(self, hidden_states):  # pylint: disable=bad-staticmethod-argument
        return torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.variance_epsilon)[0]  # noqa: F821

    @classmethod
    def patches(cls, options=None) -> List[BasePatch]:
        return [
            AtomicPatch(
                "transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextRMSNorm.forward",
                cls.forward,
                precheck=cls.precheck,
            ),
        ]


class Qwen3VLRoPE(Patch):
    """Replace apply_rotary_pos_emb_vision with torch_npu.npu_rotary_mul for Qwen3-VL vision encoder.

    Requirements: transformers >= 4.57.0 (including v5.x), torch_npu
    """

    name = "qwen3_vl_rope_vision"
    legacy_name = "qwen3_vl_rope_vision"
    target_module = "transformers.models.qwen3_vl.modeling_qwen3_vl"

    @staticmethod
    def precheck() -> bool:
        return _is_transformers_compatible_for_qwen3_vl() and _is_qwen3_vl_available()

    @staticmethod
    @with_imports("torch_npu")
    def apply_rotary_pos_emb_vision(q, k, cos, sin):
        orig_q_dtype = q.dtype
        orig_k_dtype = k.dtype
        q, k = q.float(), k.float()
        cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
        q_embed = torch_npu.npu_rotary_mul(q, cos, sin)  # noqa: F821
        k_embed = torch_npu.npu_rotary_mul(k, cos, sin)  # noqa: F821
        q_embed = q_embed.to(orig_q_dtype)
        k_embed = k_embed.to(orig_k_dtype)
        return q_embed, k_embed

    @classmethod
    def patches(cls, options=None) -> List[BasePatch]:
        return [
            AtomicPatch(
                "transformers.models.qwen3_vl.modeling_qwen3_vl.apply_rotary_pos_emb_vision",
                cls.apply_rotary_pos_emb_vision,
                precheck=cls.precheck,
            ),
        ]


class Qwen3VL_get_placeholder_mask(Patch):
    """Replace Qwen3VLModel.get_placeholder_mask with an NPU-optimized version.

    Requirements: transformers >= 4.57.0 (including v5.x)
    """

    name = "get_placeholder_mask"
    legacy_name = "get_placeholder_mask"
    target_module = "transformers.models.qwen3_vl.modeling_qwen3_vl"

    @staticmethod
    def precheck() -> bool:
        return _is_transformers_compatible_for_qwen3_vl() and _is_qwen3_vl_available()

    @staticmethod
    @with_imports("torch")
    def get_placeholder_mask(  # pylint: disable=bad-staticmethod-argument
        self,
        input_ids,
        inputs_embeds,
        image_features=None,
        video_features=None,
    ):
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)  # noqa: F821
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)  # noqa: F821
            )
            special_video_mask = special_video_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and special_image_mask.sum().item() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None and special_video_mask.sum().item() != video_features.numel():
            raise ValueError(
                f"Videos features and video tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
            )

        return special_image_mask, special_video_mask

    @classmethod
    def patches(cls, options=None) -> List[BasePatch]:
        return [
            AtomicPatch(
                "transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLModel.get_placeholder_mask",
                cls.get_placeholder_mask,
                precheck=cls.precheck,
            ),
        ]


class BaseImageProcessorFast_preprocess(Patch):
    """Override BaseImageProcessorFast.preprocess to force device to NPU.

    Requirements: transformers == 4.57.0
    """

    name = "preprocess"
    legacy_name = "preprocess"
    target_module = "transformers.image_processing_utils_fast"

    @staticmethod
    def precheck() -> bool:
        return _is_transformers_compatible_for_preprocess() and _is_qwen3_vl_available()

    @staticmethod
    @with_imports("os", "transformers")
    def preprocess(self, images, *args, **kwargs):  # pylint: disable=bad-staticmethod-argument
        @transformers.utils.auto_docstring  # noqa: F821
        def preprocess_auto(self, images, *args, **kwargs):
            # args are not validated, but their order in the `preprocess` and `_preprocess` signatures must be the same
            transformers.image_utils.validate_kwargs(  # noqa: F821
                captured_kwargs=kwargs.keys(), valid_processor_keys=self._valid_kwargs_names
            )
            # Set default kwargs from self. This ensures that if a kwarg is not provided
            # by the user, it gets its default value from the instance, or is set to None.
            for kwarg_name in self._valid_kwargs_names:
                kwargs.setdefault(kwarg_name, getattr(self, kwarg_name, None))
            # Extract parameters that are only used for preparing the input images
            do_convert_rgb = kwargs.pop("do_convert_rgb")
            input_data_format = kwargs.pop("input_data_format")
            device = kwargs.pop("device")

            local_rank = int(os.environ["LOCAL_RANK"])  # noqa: F821
            device = f"npu:{local_rank}"

            # Update kwargs that need further processing before being validated
            kwargs = self._further_process_kwargs(**kwargs)

            # Validate kwargs
            self._validate_preprocess_kwargs(**kwargs)

            # Pop kwargs that are not needed in _preprocess
            kwargs.pop("data_format")

            return self._preprocess_image_like_inputs(
                images,
                *args,
                do_convert_rgb=do_convert_rgb,
                input_data_format=input_data_format,
                device=device,
                **kwargs,
            )

        return preprocess_auto(self, images, *args, **kwargs)

    @classmethod
    def patches(cls, options=None) -> List[BasePatch]:
        return [
            AtomicPatch(
                "transformers.image_processing_utils_fast.BaseImageProcessorFast.preprocess",
                cls.preprocess,
                precheck=cls.precheck,
            ),
        ]


class Qwen2VL_preprocess(Patch):
    """Replace Qwen2VLImageProcessorFast._preprocess with an NPU-optimized version.

    Requirements: transformers == 4.57.0
    """

    name = "_preprocess"
    legacy_name = "_preprocess"
    target_module = "transformers.models.qwen2_vl.image_processing_qwen2_vl_fast"

    @staticmethod
    def precheck() -> bool:
        return _is_transformers_compatible_for_preprocess() and _is_qwen3_vl_available()

    @staticmethod
    @with_imports("torch", "transformers")
    def _preprocess(  # pylint: disable=bad-staticmethod-argument
        self,
        images,
        do_resize,
        size,
        interpolation,
        do_rescale,
        rescale_factor,
        do_normalize,
        image_mean,
        image_std,
        patch_size,
        temporal_patch_size,
        merge_size,
        disable_grouping,
        return_tensors,
        **kwargs,
    ):
        # Group images by size for batched resizing
        grouped_images, grouped_images_index = transformers.image_processing_utils_fast.group_images_by_shape(  # noqa: F821
            images, disable_grouping=disable_grouping
        )
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            height, width = stacked_images.shape[-2:]
            if do_resize:
                resized_height, resized_width = transformers.models.qwen2_vl.image_processing_qwen2_vl.smart_resize(  # noqa: F821
                    height,
                    width,
                    factor=patch_size * merge_size,
                    min_pixels=size["shortest_edge"],
                    max_pixels=size["longest_edge"],
                )
                stacked_images = self.resize(
                    image=stacked_images,
                    size=transformers.image_utils.SizeDict(height=resized_height, width=resized_width),  # noqa: F821
                    interpolation=interpolation,
                )
            resized_images_grouped[shape] = stacked_images
        resized_images = transformers.image_processing_utils_fast.reorder_images(  # noqa: F821
            resized_images_grouped, grouped_images_index
        )

        # Group images by size for further processing
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_images, grouped_images_index = transformers.image_processing_utils_fast.group_images_by_shape(  # noqa: F821
            resized_images, disable_grouping=disable_grouping
        )
        processed_images_grouped = {}
        processed_grids = {}
        for shape, stacked_images in grouped_images.items():
            resized_height, resized_width = stacked_images.shape[-2:]
            # Fused rescale and normalize
            patches = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            if patches.ndim == 4:
                # add a temporal dimension if we have images
                patches = patches.unsqueeze(1)
            if patches.shape[1] % temporal_patch_size != 0:
                repeats = patches[:, -1:].repeat(1, temporal_patch_size - 1, 1, 1, 1)
                patches = torch.cat([patches, repeats], dim=1)  # noqa: F821
            batch_size, grid_t, channel = patches.shape[:3]
            grid_t = grid_t // temporal_patch_size
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

            tmp1 = patches.view(
                batch_size,
                grid_t,
                temporal_patch_size
                * channel
                * (grid_h // merge_size)
                * merge_size
                * patch_size
                * (grid_w // merge_size)
                * merge_size,
                patch_size,
            ).permute(0, 1, 3, 2)
            tmp2 = tmp1.reshape(
                batch_size * grid_t * patch_size,
                temporal_patch_size,
                channel,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
            ).permute(0, 3, 6, 4, 7, 2, 1, 5)
            tmp3 = tmp2.reshape(
                batch_size * grid_t * patch_size, grid_h * grid_w, channel * temporal_patch_size * patch_size
            )
            tmp4 = tmp3.view(
                batch_size, grid_t, patch_size, grid_h * grid_w, channel * temporal_patch_size * patch_size
            ).permute(0, 1, 3, 4, 2)
            flatten_patches = tmp4.reshape(
                batch_size,
                grid_t * grid_h * grid_w,
                channel * temporal_patch_size * patch_size * patch_size,
            )

            processed_images_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

        processed_images = transformers.image_processing_utils_fast.reorder_images(  # noqa: F821
            processed_images_grouped, grouped_images_index
        )
        processed_grids = transformers.image_processing_utils_fast.reorder_images(processed_grids, grouped_images_index)  # noqa: F821
        pixel_values = torch.cat(processed_images, dim=0)  # noqa: F821
        image_grid_thw = torch.tensor(processed_grids)  # noqa: F821

        return transformers.BatchFeature(  # noqa: F821
            data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}, tensor_type=return_tensors
        )

    @classmethod
    def patches(cls, options=None) -> List[BasePatch]:
        return [
            AtomicPatch(
                "transformers.models.qwen2_vl.image_processing_qwen2_vl_fast.Qwen2VLImageProcessorFast._preprocess",
                cls._preprocess,
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
    """Composite patch: Qwen3RMSNorm + Qwen3RoPE (>=4.51.0) + Qwen3-VL patches (>=4.57.0) + FlashAttention (4.51.x/4.52.x).

    FlashAttention patches only apply to 4.51.x/4.52.x (4.53.x+ has built-in mask caching).
    Qwen3-VL patches activate when the Qwen3-VL model is available and transformers >= 4.57.0 (or 5.x).

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
        if _is_qwen3_vl_available():
            all_patches.extend(Qwen3VLTextRMSNorm.patches(options))
            all_patches.extend(Qwen3VLRoPE.patches(options))
            all_patches.extend(Qwen3VL_get_placeholder_mask.patches(options))
            all_patches.extend(BaseImageProcessorFast_preprocess.patches(options))
            all_patches.extend(Qwen2VL_preprocess.patches(options))
        if _is_flash_attention_available():
            all_patches.extend(FlashAttention.patches(options))
            all_patches.extend(FlashAttentionVarlen.patches(options))
        return all_patches
