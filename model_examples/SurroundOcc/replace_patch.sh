#!/bin/bash
for para in $*
do
    if [[ $para == --packages_path* ]];then
        packages_path=`echo ${para#*=}`
    fi
done

cp -f patch/torch/conv.py ${packages_path}/torch/nn/modules/conv.py
cp -f patch/mmcv/optimizer.py mmcv/mmcv/runner/hooks/optimizer.py
cp -f patch/mmcv/epoch_based_runner.py mmcv/mmcv/runner/epoch_based_runner.py
