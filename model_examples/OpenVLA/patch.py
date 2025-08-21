# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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



def RMSNorm(modeling_llama: ModuleType, options: Dict):
    def RMSNorm_forward(self, hidden_states):
        # change the code
        # using npu_rms_norm
        return torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.variance_epsilon)[0]

    if hasattr(modeling_llama, "LlamaRMSNorm"):
        modeling_llama.LlamaRMSNorm.forward = RMSNorm_forward


# get the patch for openvla
def generate_patcher_builder():
    openvla_patcher_builder = (
        PatcherBuilder()
        .add_module_patch("transformers.models.llama.modeling_llama", Patch(RMSNorm))
    )
    return openvla_patcher_builder