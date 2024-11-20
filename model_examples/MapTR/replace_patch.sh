#!/bin/bash
for para in $*
do
    if [[ $para == --packages_path* ]];then
        packages_path=`echo ${para#*=}`
    fi
done

cp -f patch/mmcv/distributed.py mmcv/mmcv/parallel/distributed.py
cp -f patch/mmcv/_functions.py mmcv/mmcv/parallel/_functions.py
cp -f patch/mmcv/deform_conv.py mmcv/mmcv/ops/deform_conv.py
cp -f patch/mmdet3d/__init__.py mmdetection3d/mmdet3d/__init__.py
cp -f patch/mmdet/__init__.py ${packages_path}/mmdet/__init__.py
cp -f patch/mmdet/resnet.py ${packages_path}/mmdet/models/backbones/resnet.py
cp -f patch/mmseg/__init__.py ${packages_path}/mmseg/__init__.py
cp -f patch/nuscenes/data_classes.py ${packages_path}/nuscenes/eval/detection/data_classes.py