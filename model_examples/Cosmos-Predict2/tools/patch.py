# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import warnings
from types import ModuleType
from typing import Dict


def snapshot_download_patch(huggingface_hub_module: ModuleType, options: Dict):        
    def snapshot_download(repo_id: str, *args, **kwargs) -> str:
        return repo_id
    
    if hasattr(huggingface_hub_module, "snapshot_download"):
        huggingface_hub_module.snapshot_download=snapshot_download
    else:
        warnings.warn(f"Failed to apply patch snapshot_download to module huggingface_hub")

def infer_device_patch(utils_module: ModuleType, options: Dict):        
    def infer_device() -> str:
        return "npu"
    
    if hasattr(utils_module, "infer_device"):
        utils_module.infer_device=infer_device
    else:
        warnings.warn(f"Failed to apply patch infer_device to module peft")


def generate_patcher_builder():
    import huggingface_hub
    from peft import peft_model
    snapshot_download_patch(huggingface_hub, {})
    infer_device_patch(peft_model, {})