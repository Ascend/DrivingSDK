# VAD

## 目录

- [VAD](#vad)
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
    - [训练任务](#训练任务)
  - [训练结果展示](#训练结果展示)
- [变更说明](#变更说明)
- [FAQ](#faq)

# 简介

## 模型介绍

VAD是一个向量化端到端的自动驾驶网络，将驾驶场景建模完全向量化表示，通过向量化的实例运动和地图元素作为显式的实例级规划约束，提升了规划安全性也提升了计算效率。

## 支持任务列表
本仓已经支持以下模型任务类型

|    模型     | 任务列表 | 是否支持 |
| :---------: | :------: | :------: |
| VAD |   训练   |    ✔     |

## 代码实现

- 参考实现：

  ```
  url=https://github.com/hustvl/VAD
  commit_id=70bb364aa3f33316960da06053c0d168628fb15f
  ```
- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitcode.com/Ascend/DrivingSDK.git
  code_path=model_examples/VAD
  ```

# 准备训练环境

## 安装昇腾环境

  请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/Pytorch/720/configandinstg/instg/insg_0001.html)》文档搭建昇腾环境，本仓已支持表1中软件版本。

  **表 1**  昇腾软件版本支持表

|     软件类型      | 首次支持版本 |
| :---------------: | :------: |
| FrameworkPTAdapter | 7.3.0  |
|       CANN        | 8.5.0 |

## 安装模型环境
当前模型支持的三方库版本如下表所示

**表 2**  三方库版本支持表

| 三方库  | 支持版本 |
| :-----: | :------: |
| PyTorch |   2.1   |
0. 激活 CANN 环境
1. 克隆代码仓到当前目录并使用patch文件

    ```
    git clone https://github.com/hustvl/VAD.git
    cd VAD
    git checkout 70bb364aa3f33316960da06053c0d168628fb15f
    cp -f ../VAD_npu.patch .
    cp -rf ../test .
    cp -rf ../tools .
    git apply --reject --whitespace=fix VAD_npu.patch
    ```

2. 安装Driving SDK加速库，具体方法参考[原仓](https://gitcode.com/Ascend/DrivingSDK)

3. 安装mmdet3d

  - 在应用过patch的模型根目录下，克隆mmdet3d仓，并进入mmdetection3d目录编译安装

    ```
    git clone -b v1.0.0rc4 https://github.com/open-mmlab/mmdetection3d.git
    cp -r ../mmdetection3d.patch mmdetection3d
    cd mmdetection3d
    git apply --reject mmdetection3d.patch
    pip install -v -e .
    ```

4. 安装mmcv

  - 在应用过patch的模型根目录下，克隆mmcv仓，并进入mmcv目录安装

    ```
    git clone -b 1.x https://github.com/open-mmlab/mmcv
    cd mmcv
    pip install -r requirements/runtime.txt
    MMCV_WITH_OPS=1 MAX_JOBS=8 FORCE_NPU=1 python setup.py build_ext
    MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py develop
    ```

5. 在应用过patch的模型根目录下，安装相关依赖。

    ```
    pip install -r requirements.txt
    ```
6. 根据操作系统，替换高性能内存库tcmalloc

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
VAD
├── projects/
├── tools/
├── configs/
├── ckpts/
│   ├── resnet50-19c8e357.pth
├── data/
│   ├── can_bus/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
|   |   ├── vad_nuscenes_infos_temporal_train.pkl
|   |   ├── vad_nuscenes_infos_temporal_val.pkl
```

> **说明：**  
> 该数据集的训练过程脚本只作为一种参考示例。      

# 准备预训练权重

- 在应用过patch的模型根目录下创建ckpts文件夹并下载预训练权重。
```
mkdir ckpts
cd ckpts 
wget https://download.pytorch.org/models/resnet50-19c8e357.pth
```

# 快速开始

### 训练任务

本任务主要提供**单机**的**8卡**训练脚本。

在应用过patch的模型根目录下，运行训练脚本。

    该模型支持单机8卡训练。

    - 单机8卡精度训练
    ```
    bash test/train_8p.sh
    ```

    - 单机8卡性能训练
    ```
    bash test/train_8p_performance.sh
    ```

## 训练结果展示
**表 3** 训练结果展示表
| 芯片          | 卡数 | global batch size | Precision | epoch |  loss   | FPS |
| ------------- | :--: | :---------------: | :-------: | :---: | :----: | :-------------------: |
| 竞品A           |  8p  |         8         |   fp32    |  8   | 10.6220 |     7.048         |
| Atlas 800T A2 |  8p  |         8         |   fp32    |  8   | 10.5733 |   4.121          |

**表 4** 结果字段说明
|     字段      | 说明 |
| :-----------: | :---------------: |
|     卡数     |    训练使用的昇腾芯片数量      |
| global batch size  |    每次训练迭代处理的样本总数      |
| Precision  |     训练数值精度      |
| epoch  |     训练的总轮次数      |
| loss |  模型训练的总损失函数值      |
| FPS |     Frames Per Second，平均每秒处理的样本总数      |

# 变更说明

2025.02.08：首次发布

2025.12.08：性能更新

# FAQ

1. tcmalloc的动态库文件位置可能因环境配置会有所不同，找不到文件时可以进行搜索，一般安装在`/usr/lib64`或者`/usr/local`目录下：
```
find /usr -name libtcmalloc.so*
```
找到对应路径下的动态库文件，`libtcmalloc.so`或者`libtcmalloc.so.版本号`都可以使用。