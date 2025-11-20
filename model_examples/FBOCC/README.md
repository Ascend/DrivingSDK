# FBOCC

## 目录

- [FBOCC](#fbocc)
  - [目录](#目录)
- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [支持任务列表](#支持任务列表)
  - [代码实现](#代码实现)
- [准备训练环境](#准备训练环境)
  - [安装昇腾环境](#安装昇腾环境)
  - [安装模型环境](#安装模型环境)
- [准备训练数据集](#准备训练数据集)
- [准备预训练权重](#准备预训练权重)
- [快速开始](#快速开始)
    - [训练结果展示](#训练结果展示)
- [版本说明](#版本说明)
  - [变更](#变更)
  - [FAQ](#faq)

# 简介

## 模型介绍

FB-OCC基于FB-BEV，是英伟达和南京大学提出的前沿占用预测模型。FB-BEV是一种基于摄像头的鸟瞰图感知算法，采用前向-后向投影技术。在此基础上，FBOCC针对3D占用预测任务提出了一些新颖设计和优化方法，包括深度语义联合预训练、联合体素-鸟瞰图表示、模型扩展以及有效的后处理策略等。FBOCC在nuScenes数据集上获得了最先进的性能，也因此成为CVPR2023端到端自动驾驶研讨会和视觉中心自动驾驶研讨会联合举办的3D占用预测挑战赛中获胜的解决方案。


## 支持任务列表

本仓已经支持以下模型任务类型

|    模型     | 任务列表 | 是否支持 |
| :---------: | :------: | :------: |
| FBOCC |   训练   |    ✔     |

本仓已经支持以下模型

|    Backbone     | Method | 训练方式 |
| :---------: | :------: | :------: |
| resnet50 |   FBOCC  |    Mixed-precision Training     |
## 代码实现

- 参考实现：

  ```
  url=https://github.com/NVlabs/FB-BEV
  commit_id=6e25469256d98e7fcb52cc43efe812dc2fd2b446
  ```
- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitcode.com/Ascend/DrivingSDK.git.git
  code_path=model_examples/FBOCC
  ```

# 准备训练环境


## 安装昇腾环境

请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表1中软件版本。

**表 1**  昇腾软件版本支持表

|     软件类型      | 支持版本 |
| :---------------: | :------: |
| FrameworkPTAdapter | 7.2.0|
|       CANN        | 8.3.RC1  |
## 安装模型环境
当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

**表 2**  版本支持表

|      三方库       |  支持版本  |
|:--------------:|:------:|
|    Python      | 3.8 |
|    PyTorch     |  2.1.0   |
0. 激活 CANN 环境

1. 下载模型源码，并应用模型patch
    ```
    git clone https://github.com/NVlabs/FB-BEV
    cp fbocc.patch FB-BEV
    cp -rf ./test/ FB-BEV
    cp -rf ./tools/ FB-BEV
    cd FB-BEV
    git checkout  6e25469256d98e7fcb52cc43efe812dc2fd2b446
    git apply --reject --whitespace=fix fbocc.patch
    ```

2. 安装Driving SDK加速库，具体方法参考[原仓](https://gitcode.com/Ascend/DrivingSDK.git)。

3. 安装mmcv

  - 在模型源码目录下，克隆mmcv仓，并进入mmcv目录安装

    ```
    git clone -b 1.x https://github.com/open-mmlab/mmcv
    cd mmcv
    pip install opencv-python-headless
    pip install -r requirements/runtime.txt
    MMCV_WITH_OPS=1 MAX_JOBS=8 FORCE_NPU=1 python setup.py build_ext
    MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py develop
    cd ..
    ```

4. mmdet、torchvision等其他必要依赖安装
    ```
    pip install mmdet==2.28.0 torchvision==0.16.0 mmsegmentation==0.30.0 llvmlite==0.41.0 numba==0.58.0 IPython
    ```

5. 安装mmdet3d

  - 在模型源码目录下，克隆mmdet3d仓，并进入mmdetection3d目录,用原仓的mmdet3d代码进行替换

    ```
    git clone -b v1.0.0rc4 https://github.com/open-mmlab/mmdetection3d.git
    cd mmdetection3d
    rm -rf mmdet3d
    mv ../mmdet3d mmdet3d
    ```
  - 在mmdetection3d目录下，修改代码

    （1）修改requirements/runtime.txt中第3行 numba==0.58.0

  - 安装包

    ```
    pip install -v -e .
    cd ..
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

    在当前python环境和路径下执行以下命令，安装并使用tcmalloc动态库。在安装tcmalloc前，需确保环境中含有autoconf和libtool依赖包。

    安装libunwind依赖：
    ```
    git clone https://github.com/libunwind/libunwind.git
    cd libunwind
    autoreconf -i
    ./configure --prefix=/usr/local
    make -j128
    make install
    ```

    安装tcmalloc动态库：
    ```
    wget https://github.com/gperftools/gperftools/releases/download/gperftools-2.16/gperftools-2.16.tar.gz
    tar -xf gperftools-2.16.tar.gz && cd gperftools-2.16
    ./configure --prefix=/usr/local/lib --with-tcmalloc-pagesize=64
    make -j128
    make install
    export LD_PRELOAD="$LD_PRELOAD:/usr/local/lib/lib/libtcmalloc.so"
    ```

# 准备训练数据集

- 根据原仓**Prepare Dataset**章节准备数据集，数据集目录及结构如下：

```
FB-BEV
├── ckpts/
│   ├──0919ori_iter_80040.pth
│   ├──r50_256x705_depth_pretrain.pth
│   ├──resnet50-0676ba61.pth
├── data/
│   ├── nuscenes/
│   │   ├── gts/  # ln -s occupancy gts to this location
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
|   |   ├── bevdetv2-nuscenes_infos_val.pkl
|   |   ├── bevdetv2-nuscenes_infos_train.pkl
├── deployment/
├── docs/
├── figs/
├── kernel_meta/
├── mmdetection3d/
├── occupancy_configs/
├── requirements/
├── tools/
```

> **说明：**
  - 数据集主体为nuScenes V1.0 full数据集，对于体素预测任务，还需要下载额外的标注文件
    https://github.com/Tsinghua-MARS-Lab/Occ3D
  - 模型的标注文件与BEVDET的不同，通过执行如下命令生成：
    ```
    python tools/create_data_bevdet.py
    ```

# 准备预训练权重

- 在模型源码目录下，执行以下指令下载预训练权重：
```
mkdir ckpts
cd ckpts
wget https://github.com/zhiqi-li/storage/releases/download/v1.0/r50_256x705_depth_pretrain.pth
```

# 快速开始

本任务主要提供单机的8卡训练脚本，在模型源码目录下，运行训练脚本。
   - 单机8卡精度训练(20 epochs)
   ```
   bash test/train_8p_full.sh
   ```
   - 单机8卡性能训练(600 steps)
   ```
   bash test/train_8p_performance.sh
   ```

## 训练结果展示
**表 3** 训练结果展示表

| 芯片          | 卡数 | global batch size | epoch | mIoU | 单步耗时(ms) |FPS|
| ------------- | :--: | :---------------:  | :---: | :----: | :---------: |:--------:|
| 竞品A         |  8p  |         32     |20  | 39.80 |  952        |33.61|
| Atlas 800T A2 |  8p  |         32      |20 | 39.85 |   1538      |20.80|

**表 4** 结果字段说明
|     字段      | 说明 |
| :-----------: | :---------------: |
|     卡数     |    训练使用的昇腾芯片数量      |
| global batch size  |    每次训练迭代处理的样本总数      |
| epoch  |     训练的总轮次数      |
| mIoU |  mean Intersection over Union, 训练结束时的平均交并比指标      |
|单步耗时|模型完成单步迭代的平均耗时|
| FPS |     Frames Per Second，平均每秒处理的样本总数      |
# 版本说明

## 变更

2025.11.27：首次发布


## FAQ
1.为提升分布式训练性能，当前版本默认配置中同步批归一化（SyncBN）设置为关闭状态（'sync_bn = False'）。在8卡训练环境下，该配置精度误差较小，但大规模训练场景下的精度影响尚待进一步验证，可能会导致精度劣化。