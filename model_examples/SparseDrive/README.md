# SparseDrive

# 目录
- [SparseDrive](#sparsedrive)
- [目录](#目录)
- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [支持任务列表](#支持任务列表)
  - [代码实现](#代码实现)
- [准备训练环境](#准备训练环境)
  - [安装昇腾环境](#安装昇腾环境)
  - [安装模型环境](#安装模型环境)
- [准备数据集](#准备数据集)
  - [预训练数据集](#预训练数据集)
  - [获取预训练权重](#获取预训练权重)
- [快速开始](#快速开始)
  - [训练模型](#训练模型)
  - [训练结果](#训练结果)
- [版本说明](#版本说明)
  - [变更](#变更)
  - [FAQ](#faq)


# 简介

## 模型介绍

SparseDrive是一种基于稀疏化表征的端到端自动驾驶模型，基于Sparse4D的整体思路，模型通过稀疏化进行检测与构图，大大提高了模型的训推速度，并通过并行的轨迹预测与规划模块，将自车的运动状态纳入到场景理解中，提高了模型轨迹预测与规划能力。

## 支持任务列表

本仓已经支持以下模型任务类型

|   模型   | 任务列表 | 是否支持 |
| :------: | :------: | :------: |
| SparseDrive |   训练   |    ✔     |

## 代码实现
- 参考实现：

  ```
  url=https://github.com/swc-17/SparseDrive
  commit_id=52c4c05b6d446b710c8a12eb9fb19d698b33cb2b
  ```
  
- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitcode.com/Ascend/DrivingSDK.git
  code_path=model_examples/SparseDrive
  ```

# 准备训练环境

## 安装昇腾环境

请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境。本仓已支持表1中软件版本。
  
  **表 1**  昇腾软件版本支持表

| 软件类型          | 支持版本     |
| ----------------- |----------|
| FrameworkPTAdapter | 7.0.0    |
| CANN              | 8.1.RC1    |

## 安装模型环境

 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 2**  版本支持表

  |      三方库       |  支持版本  |
  |:--------------:|:------:|
  |    PyTorch     |  2.1   |
  |    Driving SDK   | 7.1.0 |
  |      mmcv      |  1.x   |
  |     mmdet      | 2.28.2 |

- 安装Driving SDK

  请参考昇腾[Driving SDK](https://gitcode.com/Ascend/DrivingSDK)代码仓说明编译安装Driving SDK

- 安装基础依赖

  在模型源码包根目录下执行命令，安装模型需要的依赖。
  
  ```shell
  pip install -r requirements.txt
  ```

- 源码安装geos

  shapely模块依赖于geos安装包，需要本地编译并将so文件复制到环境中

  ```shell
  sparsedrive_env_path=`pip3 show mx_driving | grep "Location" | awk -F "Location: " '{print $2}' | awk -F "python" '{print $1}'`
  git clone https://github.com/libgeos/geos.git
  cd geos
  mkdir build
  cd build
  cmake ../
  make
  cp lib/libgeos* ${sparsedrive_env_path}
  ```

- 源码安装mmcv

  ```shell
    git clone -b 1.x https://github.com/open-mmlab/mmcv.git
    cd mmcv
    MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py install
  ```

- 模型代码Patch
  ```shell
  git clone https://github.com/swc-17/SparseDrive.git
  cp -rf ./test/ SparseDrive
  cp -rf ./tools/ SparseDrive
  cp -rf SparseDrive.patch SparseDrive
  cd SparseDrive
  git checkout 52c4c05b6d446b710c8a12eb9fb19d698b33cb2b
  git apply SparseDrive.patch
  ```

# 准备数据集

## 预训练数据集
用户自行获取*nuscenes*数据集，在源码目录创建软连接`data/nuscenes`指向解压后的nuscenes数据目录
  ```shell
  sparsedrive_path="path/to/sparsedrive"
  cd ${sparsedrive_path}
  mkdir data
  ln -s path/to/nuscenes ./data/nuscenes
  ```

运行数据预处理脚本生成SparseDrive模型训练需要的pkl文件与初始锚框
  ```shell
  sh test/preprocess.sh
  ```

## 获取预训练权重
下载backbone预训练权重
  ```shell
  mkdir ckpt
  wget https://download.pytorch.org/models/resnet50-19c8e357.pth -O ckpt/resnet50-19c8e357.pth
  ```

# 快速开始

## 训练模型

- 单机训练

    训练脚本支持命令行参数（支持默认值/关键字参数/位置参数）
    - `--batch_node_size_stage1` : stage1的global_batch_size，默认值为64（单卡batch_size=global_batch_size/NPU卡数）
    - `--batch_node_size_stage2` : stage2的global_batch_size，默认值为48
    - `--num_npu` : NPU卡数，默认值为8

- 单机多卡精度训练脚本
```shell
  #stage1默认训练100个epoch,stage2训练10个epoch
  bash test/train_8p_full.sh
  (option) bash test/train_8p_full.sh 64 48 8
  (option) bash test/train_8p_full.sh --batch_node_size_stage1=64 --batch_node_size_stage2=48 --num_npu=8
```
- 单机多卡性能测试脚本
```shell
  #stage1\stage2默认训练1000个step
  bash test/train_8p_performance.sh
  (option) bash test/train_8p_performance.sh 64 48 8
  (option) bash test/train_8p_performance.sh --batch_node_size_stage1=64 --batch_node_size_stage2=48 --num_npu=8

  # (option)若没有进行全量精度训练，可使用如下命令下载stage1的权重以测试stage2的性能
  wget https://github.com/swc-17/SparseDrive/releases/download/v1.0/sparsedrive_stage1.pth -O ckpt/sparsedrive_stage1.pth
```

## 训练结果

**表 3** 训练结果展示表

|     阶段   |      芯片       | 卡数 | global batch size  | FPS | 平均step耗时(s) | amota | L2 |
|:---------:|:---------------:|------|:------------------:|:----:|:--------------:|:---:|:---:|
|   stage1   |      竞品A      | 8p | 64 | 41.0 | 1.561 | 0.3764 | - |
|   stage1   | Atlas 800T A2   | 8p | 64 | 46.3 | 1.382 | 0.3864 | - |
|   stage2   |      竞品A      | 8p | 48 | 35.2 | 1.363 | - | 0.6280 |
|   stage2   | Atlas 800T A2   | 8p | 48 | 37.9 | 1.265 | - | 0.6069 |



# 版本说明
## 变更
2025.04.27：首次发布。

2025.07.07：瓶颈算子优化，刷新性能数据

2025.08.25：训练脚本优化
## FAQ
暂无。



