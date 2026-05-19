#!/bin/bash

# 检测设备类型
device_name=$(python -c "import torch_npu; print(torch_npu.npu.get_device_name(0))" 2>/dev/null)
if echo "$device_name" | grep -q "950"; then
    IS_A5=1
else
    IS_A5=0
fi

src_path=$(pwd)

cd $src_path
##安装其他依赖
echo "模型依赖开始安装"
if [ "$IS_A5" -eq 1 ]; then
    pip install -r requirements_pytorch2.7.1_a5.txt
else
    pip install -r requirements.txt
fi


##安装mmcv
echo "mmcv开始安装"
git clone -b 1.x https://github.com/open-mmlab/mmcv.git
cp mmcv_config.patch mmcv
cd mmcv
git apply --reject mmcv_config.patch
pip install -r requirements/runtime.txt
MMCV_WITH_OPS=1 MAX_JOBS=255 FORCE_NPU=1 python setup.py build_ext
MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py develop
cd ..

##安装mmdet3d
echo "mmdet3d开始安装"
git clone -b v1.0.0rc4 https://github.com/open-mmlab/mmdetection3d.git
cp mmdet3d_config.patch mmdetection3d
cd mmdetection3d
git apply --reject mmdet3d_config.patch
pip install -e . --no-build-isolation
cd ..

##安装mmdet
echo "mmdet开始安装"
if [ "$IS_A5" -eq 1 ]; then
    git clone -b v2.24.0 https://github.com/open-mmlab/mmdetection.git
    cp mmdet_a5.patch mmdetection
    cd mmdetection
    git apply --reject mmdet_a5.patch
    pip install -e . --no-build-isolation
    cd ..
else
    git clone -b v2.24.0 https://github.com/open-mmlab/mmdetection.git
    cp mmdet_config.patch mmdetection
    cd mmdetection
    git apply --reject mmdet_config.patch
    pip install -e . --no-build-isolation
    cd ..
fi

##安装 detectron2
echo "detectron2开始安装"
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation

##模型代码更新
echo "模型代码开始更新"
if [ "$IS_A5" -eq 1 ]; then
    git clone https://github.com/fundamentalvision/BEVFormer.git
    cp bevformer_a5.patch BEVFormer
    cp patch_a5.py BEVFormer/tools
    mv BEVFormer/tools/patch_a5.py BEVFormer/tools/patch.py
    cd BEVFormer
    git checkout 66b65f3a1f58caf0507cb2a971b9c0e7f842376c
    git apply --reject --whitespace=fix bevformer_a5.patch
    cd ..
else
    git clone https://github.com/fundamentalvision/BEVFormer.git
    cp bev_former_config.patch BEVFormer
    cd BEVFormer
    git checkout 66b65f3a1f58caf0507cb2a971b9c0e7f842376c
    git apply --reject --whitespace=fix bev_former_config.patch
    cd ..
fi

echo "全部安装完成"
