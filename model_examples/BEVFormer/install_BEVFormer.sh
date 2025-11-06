src_path=$(pwd)

cd $src_path
##安装其他依赖
echo "模型依赖开始安装"
pip install -r requirements.txt

##安装mmcv
echo "mmcv开始安装"
git clone -b 1.x https://github.com/open-mmlab/mmcv.git
cp mmcv_config.patch mmcv
cd mmcv
git apply --reject mmcv_config.patch
pip install -r requirements/runtime.txt
MMCV_WITH_OPS=1 MAX_JOBS=8 FORCE_NPU=1 python setup.py build_ext
MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py develop
cd ..

##安装mmdet3d
echo "mmdet3d开始安装"
git clone -b v1.0.0rc4 https://github.com/open-mmlab/mmdetection3d.git
cp mmdet3d_config.patch mmdetection3d
cd mmdetection3d
git apply --reject mmdet3d_config.patch
pip install -e .
cd ..

##安装mmdet
echo "mmdet开始安装"
git clone -b v2.24.0 https://github.com/open-mmlab/mmdetection.git
cp mmdet_config.patch mmdetection
cd mmdetection
git apply --reject mmdet_config.patch
pip install -e .
cd ..

##安装 detectron2
echo "detectron2开始安装"
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

##模型代码更新
echo "模型代码开始更新"
git clone https://github.com/fundamentalvision/BEVFormer.git
cp bev_former_config.patch BEVFormer
cd BEVFormer
git checkout 66b65f3a1f58caf0507cb2a971b9c0e7f842376c
git apply --reject --whitespace=fix bev_former_config.patch
cd ..