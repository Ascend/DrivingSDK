src_path=$(pwd)

cd $src_path

##安装mmcv
echo "mmcv开始安装"
git clone -b main https://github.com/open-mmlab/mmcv.git
cd mmcv
pip install -r requirements/runtime.txt
pip install ninja
pip install "setuptools<=78.1.1"
MMCV_WITH_OPS=1 MAX_JOBS=8 FORCE_NPU=1 python setup.py build_ext
MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py develop
cd ../

##安装mmdetection3d v1.2.0
echo "mmdet3d开始安装"
git clone -b v1.2.0 https://github.com/open-mmlab/mmdetection3d.git
cp -f bevfusion.patch mmdetection3d/
cd mmdetection3d
git apply bevfusion.patch --reject
pip install mmengine==0.10.7 mmdet==3.1.0 numpy==1.23.5 yapf
pip install -e . --no-build-isolation
cd ../