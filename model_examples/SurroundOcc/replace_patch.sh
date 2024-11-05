#!/bin/bash
for para in $*
do
    if [[ $para == --packages_path* ]];then
        packages_path=`echo ${para#*=}`
    fi
done

cp -f patch/torch/conv.py ${packages_path}/torch/nn/modules/conv.py
cp -f patch/mmdet/resnet.py ${packages_path}/mmdet/models/backbones/resnet.py
cp -f patch/mmcv/optimizer.py mmcv/mmcv/runner/hooks/optimizer.py
cp -f patch/mmcv/epoch_based_runner.py mmcv/mmcv/runner/epoch_based_runner.py
cp -f patch/mmcv/distributed.py mmcv/mmcv/parallel/distributed.py
cp -f patch/mmcv/modulated_deform_conv.py mmcv/mmcv/ops/modulated_deform_conv.py