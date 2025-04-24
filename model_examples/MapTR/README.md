# MapTR for PyTorch

## 目录

- [MapTR for PyTorch](#maptr-for-pytorch)
  - [目录](#目录)
- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [支持任务列表](#支持任务列表)
  - [代码实现](#代码实现)
- [MapTR](#maptr)
  - [准备训练环境](#准备训练环境)
    - [安装环境](#安装环境)
    - [安装昇腾环境](#安装昇腾环境)
    - [准备数据集](#准备数据集)
    - [准备预训练权重](#准备预训练权重)
  - [快速开始](#快速开始)
    - [训练任务](#训练任务)
      - [开始训练](#开始训练)
      - [训练结果](#训练结果)
- [变更说明](#变更说明)
- [FAQ](#faq)

# 简介

## 模型介绍

MapTR是一种高效的端到端Transformer模型，用于在线构建矢量化高清地图（HD Map）。高清地图在自动驾驶系统中是规划的基础和关键组件，提供了丰富而精确的环境信息。MapTR提出了一种统一的置换等价建模方法，将地图元素表示为等价置换组的点集，这样不仅可以准确描述地图元素的形状，还能稳定学习过程。此外，MapTR设计了一个分层查询嵌入方案，以灵活地编码结构化地图信息，并执行分层二分匹配来学习地图元素。

## 支持任务列表

本仓已经支持以下模型任务类型

|    模型     | 任务列表 | 是否支持 |
| :---------: | :------: | :------: |
| MapTR |   训练   |    ✔     |

## 代码实现

- 参考实现：

  ```
  url=https://github.com/hustvl/MapTR
  commit_id=fa420a2e756c9e19b876bdf2f6d33a097d84be73
  ```

# MapTR

## 准备训练环境

### 安装环境

**表 1**  三方库版本支持表

| 三方库  | 支持版本 |
| :-----: | :------: |
| PyTorch |   2.1   |

### 安装昇腾环境

请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表2中软件版本。

**表 2**  昇腾软件版本支持表

|     软件类型      | 支持版本 |
| :---------------: | :------: |
| FrameworkPTAdaper | 7.0.0  |
|       CANN        | 8.1.RC1  |

1. 安装mmdet3d

  - 在模型根目录下，克隆mmdet3d仓，并进入mmdetection3d目录

    ```
    git clone -b v1.0.0rc4 https://github.com/open-mmlab/mmdetection3d.git
    cd mmdetection3d
    ```
  - 在mmdetection3d目录下，修改代码

    （1）删除requirements/runtime.txt中第3行 numba==0.53.0

    （2）修改mmdet3d/____init____.py中第22行 mmcv_maximum_version = '1.7.0'为mmcv_maximum_version = '1.7.2'
  - 安装包

    ```
    pip install -v -e .
    ```
2. 安装mmcv

  - 在模型根目录下，克隆mmcv仓，并进入mmcv目录安装

    ```
    git clone -b 1.x https://github.com/open-mmlab/mmcv
    cd mmcv
    MMCV_WITH_OPS=1 MAX_JOBS=8 FORCE_NPU=1 python setup.py build_ext
    MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py develop
    ```
3. 安装Driving SDK加速库，具体方法参考[原仓](https://gitee.com/ascend/DrivingSDK)。
4. 在模型根目录下执行以下命令，安装模型对应PyTorch版本需要的依赖。

  ```
  pip install -r requirement.txt
  ```
5. 在当前python环境下执行`pip show pip`，得到三方包安装路径Location，记作location_path，在模型根目录下执行以下命令来替换patch。

  ```
  bash replace_patch.sh --packages_path=location_path
  ```
6. 根据操作系统，安装tcmalloc动态库。

  - OpenEuler系统

  在当前python环境和路径下执行以下命令，安装并使用tcmalloc动态库。
  ```
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
  ```
  - Ubuntu系统

  参考[下载链接](http://mirrors.aliyun.com/ubuntu-ports/pool/main/g/google-perftools/?spm=a2c6h.25603864.0.0.731161f3db9Jrh)，下载三个文件。

    libgoogle-perftools4_2.7-1ubuntu2_arm64.deb

    libgoogle-perftools-dev_2.7-1ubuntu2_arm64.deb

    libtcmalloc-minimal4_2.7-1ubuntu2_arm64.deb

  安装三个文件：
  ```
  sudo dpkg -i libtcmalloc-minimal4_2.7-1ubuntu2_arm64.deb
  sudo dpkg -i libgoogle-perftools-dev_2.7-1ubuntu2_arm64.deb
  sudo dpkg -i libgoogle-perftools4_2.7-1ubuntu2_arm64.deb
  find /usr -name libtcmalloc.so*
  ```

  将find指令的输出路径记为libtomalloc_dir，执行下列文件使用tcmalloc动态库。
  ```
  export LD_PRELOAD="$LD_PRELOAD:/{libtcmalloc_root_dir}/libtcmalloc.so"
  ```
7. 编译优化

  编译优化是指通过毕昇编译器的LTO和PGO编译优化技术，源码构建编译Python、PyTorch、torch_npu（Ascend Extension for PyTorch）三个组件，有效提升程序性能。

  本节介绍Python、Torch和torch_npu LTO编译优化方式，采用编译优化后的性能见“训练结果”小节。

  - 安装毕昇编译器

  将CANN包安装目录记为cann_root_dir，执行下列命令安装毕昇编译器。
  ```
  wget https://kunpeng-repo.obs.cn-north-4.myhuaweicloud.com/BiSheng%20Enterprise/BiSheng%20Enterprise%20203.0.0/BiShengCompiler-4.1.0-aarch64-linux.tar.gz
  tar -xvf BiShengCompiler-4.1.0-aarch64-linux.tar.gz
  export PATH=$(pwd)/BiShengCompiler-4.1.0-aarch64-linux/bin:$PATH
  export LD_LIBRARY_PATH=$(pwd)/BiShengCompiler-4.1.0-aarch64-linux/lib:$LD_LIBRARY_PATH
  source {cann_root_dir}/set_env.sh
  ```

  - 安装依赖，将安装mpdecimal依赖包的目录记为mpdecimal_install_path。
  ```
  wget --no-check-certificate https://www.bytereef.org/software/mpdecimal/releases/mpdecimal-2.5.1.tar.gz
  tar -xvf mpdecimal-2.5.1.tar.gz
  cd mpdecimal-2.5.1
  bash ./configure --prefix=mpdecimal_install_path
  make -j
  make install
  ```

  - 获取Python源码并编译优化

  执行以下指令获取Python版本及安装目录，将Python安装路径记为python_path。
  ```
  python -V
  which python
  ```

  在[Python源码下载地址](https://www.python.org/downloads/source/)下载对应版本的Python源码并解压。

  以Python 3.8.17为例：
  ```
  tar -xvf Python-3.8.17.tgz
  cd Python-3.8.17
  export CC=clang
  export CXX=clang++
  ./configure --prefix=python_path > --with-lto --enable-optimizations
  make -j
  make install
  ```

  - Pytorch和torch_npu编译优化

  推荐Pytorch和torch_npu采用专有镜像编译。

  1） 创建编译优化基础镜像。以arm的镜像为例：

  - arm镜像地址：
      https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/docker_images/pytorcharm_compile.tar.gz

    将镜像的image_id记为image_id，将创建容器时的宿主机路径和容器路径分别记为path1和path2：
    ```
    docker load -i pytorcharm_compile.tar.gz
    docker images // 查看镜像的image_id
    docker run -it --network=host image_id -v path1:path2 bash // 创建容器
    ```

  2） 在编译优化基础镜像中配置环境，以python3.8为例，如果使用其他版本的python3.8需修改python3软链接：
  ```
  cd /usr/local/bin/
  ln -s /opt/_internal/cpython-3.7.17/bin/pip3.7 pip3.7
  ln -s /opt/_internal/cpython-3.8.17/bin/pip3.8 pip3.8
  ln -s /opt/internal/cpython-3.9.17/bin/pip3.9 pip3.9
  ln -s python3.8 python3
  ```

  3） 按照“安装毕昇编译器”一节在编译优化基础镜像中使能毕昇编译器。

  4） 下载Torch源码。以torch2.1.0为例：

  ```
  git clone -b v2.1.0 https://github.com/pytorch/pytorch.git pytorch-2.1.0
  cd pytorch-2.1.0
  git submodule sync
  git submodule update --init --recursive
  cd pytorch-2.1.0
  pip install -r requirements.txt
  ```

  打开CMakeLists.txt文件，注释第921行：
  ```
  append_cxx_flag_if_supported("-Werror=cast-function-type" CMAKE_FXX_FLAGS)
  ```
  屏蔽告警错误。

  5） 配置编译参数，设置环境变量，并进行LTO优化编译。
  ```
  export CMAKE_C_FLAGS="-flto=thin -fuse-ld=lld"
  export CMAKE_CXX_FLAGS="-flto=thin -fuse-ld=lld"
  export CC=clang
  export CXX=clang++
  export USE_XNNPACK=0
  export OMP_PROC_BIND=false
  git clean -dfx
  python3 setup.py bdist_wheel
  ```

  6） 安装Torch，并下载torch_npu源码，进行LTO优化编译。以torch2.1.0配套的torch_npu为例：

  ```
  pip3.8 install /dist/*.whl --force-reinstall --no-deps
  cd ../
  git clone -b v2.1.0 https://gitee.com/ascend/pytorch.git torch_npu
  cd torch_npu
  git clean -dfx
  bash ci/build.sh --python=3.8 --enable_lto
  ```

  7） 在模型所使用的Python环境下，安装LTO优化编译的Torch和torch_npu包。将LTO编译优化生成的torch包和torch_npu包路径分别记为torch_path和torch_npu_path：
  ```
  pip install torch_path/*.whl torch_npu_path/*.whl --force-reinstall --no-deps
  ```

8. 模型代码更新

  ```
  git clone https://github.com/hustvl/MapTR.git
  cp MapTR.patch MapTR
  cd MapTR
  git checkout 1b435fd9f0db9a14bb2a9baafb565200cc7028a2
  git apply --reject --whitespace=fix MapTR.patch
  cd ../
  ```

### 准备数据集

- 根据原仓**Prepare Dataset**章节准备数据集，数据集目录及结构如下：

```
MapTR
├── ckpts/
│   ├── resnet50-19c8e357.pth
├── data/
│   ├── can_bus/
│   ├── nuscenes/
│   │   ├── lidarseg/
│   │   ├── maps/
│   │   ├── panoptic/
│   │   ├── samples/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
|   |   ├── nuscenes_infos_temporal_test_mono3d.coco.json
|   |   ├── nuscenes_infos_temporal_train_mono3d.coco.json
|   |   ├── nuscenes_infos_temporal_val_mono3d.coco.json
|   |   ├── nuscenes_map_anns_val.json
|   |   ├── nuscenes_infos_temporal_test.pkl
|   |   ├── nuscenes_infos_temporal_train.pkl
|   |   ├── nuscenes_infos_temporal_val.pkl
├── patch/
├── projects/
├── test/
├── tools/
```

> **说明：**
> nuscenes数据集下的文件，通过运行以下指令生成：
```
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0 --canbus ./data
```

### 准备预训练权重

- 在模型根目录下，执行以下指令下载预训练权重：
```
mkdir ckpts
cd ckpts
wget https://download.pytorch.org/models/resnet50-19c8e357.pth
```

## 快速开始

### 训练任务

本任务主要提供单机的8卡训练脚本。

#### 开始训练

1. 在模型根目录下，运行训练脚本。

   该模型支持单机8卡、多机多卡训练。

   - 单机8卡精度训练

   ```
   bash test/train_8p.sh
   ```

   - 单机8卡性能训练

   ```
   bash test/train_8p_performance.sh
   ```

   - 多机多卡精度训练

   以双机举例，假设每台机器8卡，则总共有16卡。

   记主节点为master_addr，通信端口为port。

   主节点拉起训练的脚本为：

   ```
   bash test/nnodes_train_8p.sh 2 0 port master_addr
   ```

   副节点拉起训练的脚本为：
   ```
   bash test/nnodes_train_8p.sh 2 1 PORT MASTER_ADDR
   ```

   - 多机多卡性能训练

   主节点拉起训练的脚本为：

   ```
   bash test/nnodes_train_8p_performance.sh 2 0 port master_addr
   ```

   副节点拉起训练的脚本为：
   ```
   bash test/nnodes_train_8p_performance.sh 2 1 port master_addr
   ```

#### 训练结果

| 芯片          | 卡数 | global batch size | Precision | epoch |  mAP  | 性能-FPS |
| ------------- | :--: | :---------------: | :-------: | :---: | :----: | :-------------------: |
| 竞品A           |  8p  |         32         |   fp32    |  24   | 48.7 |         -          |
| Atlas 800T A2 |  8p  |         32         |   fp32    |  24   | 48.5 |         -          |
| 竞品A           |  8p  |         32         |   fp32    |  1   | - |         33.20          |
| Atlas 800T A2 |  8p  |         32         |   fp32    |  1   | - |         34.85          |


# 变更说明

2025.04.17：优化模型性能打屏格式，修改Torch2.1.0适配的依赖包版本。

2025.03.31：优化模型性能计算脚本，增加Python&Torch&torch_npu编译优化性能。

2025.03.10：增加Python编译优化性能。

2025.03.04：进一步优化模型性能，更新性能数据。增加多机多卡训练脚本。

2025.02.21：优化模型性能，更新性能数据。

2025.02.10: 增加1epoch性能训练配置文件，更新Readme说明。

2024.11.08：首次发布


# FAQ

无