# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright (c) OpenMMLab. All rights reserved.
from types import ModuleType
from typing import List, Optional, Union, Dict


def stream(mmcvparallel: ModuleType, options: Dict):
    get_input_device = mmcvparallel._functions.get_input_device
    scatter = mmcvparallel._functions.scatter
    synchronize_stream = mmcvparallel._functions.synchronize_stream
    _get_stream = mmcvparallel._functions._get_stream
    Tensor = mmcvparallel._functions.Tensor
    

    @staticmethod
    def new_forward(target_gpus: List[int], input_: Union[List, Tensor]) -> tuple:
        input_device = get_input_device(input_)
        streams = None
        if input_device == -1 and target_gpus != [-1]:
            # Perform CPU to GPU copies in a background stream
            streams = [
                _get_stream(mmcvparallel._functions.torch.device("cuda", device))
                for device in target_gpus
            ]

        outputs = scatter(input_, target_gpus, streams)
        # Synchronize with the copy stream
        if streams is not None:
            synchronize_stream(outputs, target_gpus, streams)

        return tuple(outputs) if isinstance(outputs, list) else (outputs, )

    if hasattr(mmcvparallel._functions, "Scatter"):
        mmcvparallel._functions.Scatter.forward = new_forward
