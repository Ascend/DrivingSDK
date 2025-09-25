# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

# pylint: disable=huawei-wrong-import-position, wrong-import-order
import torch
import torch_npu
from mx_driving.patcher import patch_mmcv_version
patch_mmcv_version('1.6.0')
from migrate_to_ascend.patch_main import generate_patcher_builder


from tools.test import main


if __name__ == '__main__':
    
    patcher_builder = generate_patcher_builder()
    with patcher_builder.build():
        main()
