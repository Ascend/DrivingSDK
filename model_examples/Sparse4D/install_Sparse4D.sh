src_path=$(pwd)

SYSTEM=openeular
grep -q "openEuler" /etc/os-release && SYSTEM=openeular || SYSTEM=ubuntu
echo "系统类别: $SYSTEM"

cd $src_path
##安装依赖
echo "模型依赖开始安装"
pip install -r requirements.txt

##安装mmcv
echo "mmcv开始安装"
git clone -b 1.x https://github.com/open-mmlab/mmcv.git
cp mmcv.patch mmcv
cd mmcv
git apply mmcv.patch
MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py install
cd ..

##安装mmdet
echo "mmdet开始安装"
git clone -b v2.28.2 https://github.com/open-mmlab/mmdetection.git
cp mmdet.patch mmdetection
cd mmdetection
git apply mmdet.patch
pip install -e .
cd ..

##模型代码使用Patch
echo "模型代码开始更新"
git clone https://github.com/HorizonRobotics/Sparse4D.git
cp Sparse4D.patch Sparse4D
cp patch.py Sparse4D/tools
cd Sparse4D
git checkout c41df4bbf7bc82490f11ff55173abfcb3fb91425
git apply Sparse4D.patch
cp -rf ../test .
cd ..

##安装tcmalloc(openeular或者ubuntu)
echo "tcmalloc开始安装"
if [ "$SYSTEM"=="openeular" ]; then
    mkdir gperftools
    cd gperftools
    wget https://github.com/gperftools/gperftools/releases/download/gperftools-2.16/gperftools-2.16.tar.gz
    tar -zvxf gperftools-2.16.tar.gz
    cd gperftools-2.16
    ./configure --prefix=/usr/local/lib --with-tcmalloc-pagesize=64
    make
    make install
    echo '/usr/local/lib/lib/' >> /etc/ld.so.conf
    ldconfig
    export LD_LIBRARY_PATH=/usr/local/lib/lib/:$LD_LIBRARY_PATH
    export PATH=/usr/local/lib/bin:$PATH
    export LD_PRELOAD=/usr/local/lib/lib/libtcmalloc.so.4
else
    git clone https://github.com/libunwind/libunwind.git
    cd libunwind
    autoreconf -i
    ./configure --prefix=/usr/local
    make -j128
    make install
    cd ..
    wget https://github.com/gperftools/gperftools/releases/download/gperftools-2.16/gperftools-2.16.tar.gz
    tar -xf gperftools-2.16.tar.gz && cd gperftools-2.16
    ./configure --prefix=/usr/local/lib --with-tcmalloc-pagesize=64
    make -j128
    make install
    export LD_PRELOAD="$LD_PRELOAD:/usr/local/lib/lib/libtcmalloc.so"
fi

cd $src_path