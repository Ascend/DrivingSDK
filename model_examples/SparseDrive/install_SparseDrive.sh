src_path=$(pwd)

cd $src_path

##安装mmcv
echo "mmcv开始安装"
git clone -b 1.x https://github.com/open-mmlab/mmcv.git
cd mmcv
MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py install
cd ..

##模型代码更新
echo "模型代码开始更新"
git clone https://github.com/swc-17/SparseDrive.git
cp -rf ./test/ SparseDrive
cp -rf ./tools/ SparseDrive
cp -rf SparseDrive.patch SparseDrive
cd SparseDrive
git checkout 52c4c05b6d446b710c8a12eb9fb19d698b33cb2b
git apply --reject --whitespace=fix SparseDrive.patch

##安装基础依赖
echo "基础依赖开始安装"
pip install -r requirements.txt
cd ..