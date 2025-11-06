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
git clone -b 1.x https://github.com/open-mmlab/mmcv
cp mmcv_config.patch mmcv/
cd mmcv
git apply --reject --whitespace=fix mmcv_config.patch
MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py install
cd ..

##安装mmdet和mmsegmentation
echo "mmdet和mmseg开始安装"
pip install mmdet==2.28.2
pip install mmsegmentation==0.30.0

##安装mmdet3d
echo "mmdet3d开始安装"
git clone -b v1.0.0rc4 https://github.com/open-mmlab/mmdetection3d.git
cp mmdet3d_config.patch mmdetection3d/
cd mmdetection3d
git apply --reject --whitespace=fix mmdet3d_config.patch
pip install -v -e .
cd ..

##安装 mindspeed
echo "mindspeed开始安装"
git clone https://gitcode.com/Ascend/MindSpeed.git
pip install -e MindSpeed

##模型代码更新
echo "模型代码开始更新"
git clone -b maptrv2 https://github.com/hustvl/MapTR.git MapTRv2
cp MapTRv2.patch MapTRv2/
cd MapTRv2
git checkout e03f097abef19e1ba3fed5f471a8d80fbfa0a064
git apply --reject --whitespace=fix MapTRv2.patch
cd ../

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