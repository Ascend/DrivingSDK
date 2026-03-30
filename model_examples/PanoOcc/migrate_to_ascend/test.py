# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

# =============================================================================
# PATCHER SETUP - MUST BE FIRST (before any imports that need patching)
# =============================================================================
from mx_driving.patcher import default_patcher, ensure_mmcv_version
from migrate_to_ascend.patch_main import configure_patcher

ensure_mmcv_version("1.6.0")

configure_patcher(default_patcher)
default_patcher.apply()


# =============================================================================
# ORIGINAL CODE BELOW - UNCHANGED
# =============================================================================
# pylint: disable=huawei-wrong-import-position, wrong-import-order
import torch
import torch_npu


from tools.test import main


if __name__ == '__main__':
    main()
