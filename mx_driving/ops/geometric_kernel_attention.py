"""
Copyright (c) OpenMMLab. All rights reserved.
Copyright (c) 2022 Hust Vision Lab. All rights reserved.
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
Licensed under the MIT License.
Licensed under the BSD 3-Clause License  (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://opensource.org/licenses/BSD-3-Clause

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Any, Optional, Tuple, Union

import torch
from torch.autograd import Function

import torch_npu
import mx_driving._C


class GeometricKernelAttentionFunc(Function):
    @staticmethod
    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def forward(ctx: Any, value: torch.Tensor, spatial_shapes: torch.Tensor, level_start_index: torch.Tensor,
                sampling_locations: torch.Tensor, attention_weights: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(value, spatial_shapes, level_start_index, sampling_locations, attention_weights)

        ctx.value_size = value.size()
        batch_size, num_keys, num_heads, dim = value.size()
        num_queries = attention_weights.size(1)

        output = value.new_zeros(batch_size, num_queries, num_heads * dim)

        mx_driving._C.geometric_kernel_attention_forward(
            value, spatial_shapes, level_start_index, sampling_locations, attention_weights, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        value, spatial_shapes, level_start_index, sampling_locations, attn_weights = ctx.saved_tensors
        grad_value, grad_attn_weights = mx_driving._C.geometric_kernel_attention_backward(
            value, spatial_shapes, level_start_index, sampling_locations, attn_weights, grad_output
        )
        return grad_value, None, None, None, grad_attn_weights


geometric_kernel_attention = GeometricKernelAttentionFunc.apply
