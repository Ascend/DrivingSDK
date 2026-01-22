// Copyright (c) 2024 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "csrc/OpApiCommon.h"
#include "csrc/functions.h"
#include <iostream>

void sigmoid_focal_loss(const at::Tensor& input, const at::Tensor& target, const at::Tensor& weight,
                                    const at::Tensor& output, double gamma, double alpha) {                                      
    int64_t n_class = input.size(1);
    at::Tensor target_y = at::ones_like(input);
    if (n_class == 1) {
        target_y = at::reshape(target, input.sizes());
        target_y = at::mul(target_y, -1.0);
        target_y = at::add(target_y, 1.0);
    } else {
        target_y = at::one_hot(target, n_class);
    }
    target_y = target_y.to(at::kInt);
    int64_t weight_size = weight.size(0);
    at::Tensor weight_y = at::ones_like(input);
    if (weight_size > 0) {
        at::Tensor weight_selected = weight.gather(0, target);
        weight_selected = weight_selected.unsqueeze(1);
        weight_y = weight_selected.expand_as(input);
    }
    string reduction = "none";
    EXEC_NPU_CMD(aclnnSigmoidFocalLoss, input, target_y, weight_y, gamma, alpha, output);
}

void sigmoid_focal_loss_backward(const at::Tensor& input, const at::Tensor& target, const at::Tensor& weight,
                                    const at::Tensor& grad_input, double gamma, double alpha) {
    int64_t n_class = input.size(1);
    at::Tensor target_y = at::ones_like(input);
    if (n_class == 1) {
        target_y = at::reshape(target, input.sizes());
        target_y = at::mul(target_y, -1.0);
        target_y = at::add(target_y, 1.0);
    } else {
        target_y = at::one_hot(target, n_class);
    }
    target_y = target_y.to(at::kInt);
    int64_t weight_size = weight.size(0);
    at::Tensor weight_y = at::ones_like(input);
    if (weight_size > 0) {
        at::Tensor weight_selected = weight.gather(0, target);
        weight_selected = weight_selected.unsqueeze(1);
        weight_y = weight_selected.expand_as(input);
    }
    string reduction = "none";
    EXEC_NPU_CMD(aclnnSigmoidFocalLossGrad, input, target_y, weight_y, gamma, alpha, grad_input);
}
