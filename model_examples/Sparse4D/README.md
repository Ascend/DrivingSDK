# Sparse4D

# 目录
- [Sparse4D](#sparse4d)
- [目录](#目录)
- [简介](#简介)
  - [模型介绍](#模型介绍)
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

近年来，基于鸟瞰图的方法在多视角三维检测任务中取得了很大进展。与基于BEV的方法相比，稀疏方法在性能上有所落后，但仍有许多不可忽视的优点。为了进一步推动稀疏3D检测，地平线提出一种名为Sparse4D的新方法，该方法通过稀疏采样和融合时空特征对锚框进行迭代细化。
- 稀疏四维采样:对于每个3D锚点，作者分配多个四维关键点，然后将其投影到多视图/尺度/时间戳图像特征上，以采样相应的特征;
- 层次特征融合:对不同视角/尺度、不同时间戳、不同关键点的采样特征进行层次融合，生成高质量的实例特征。  

这样，Sparse4D可以高效有效地实现3D检测，而不依赖于密集的视图变换和全局关注，并且对边缘设备的部署更加友好。

## 代码实现
- 参考实现：

  ```
  url=https://github.com/HorizonRobotics/Sparse4D
  commit_id=c41df4bbf7bc82490f11ff55173abfcb3fb91425
  ```
  
- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/DrivingSDK.git
  code_path=model_examples/Sparse4D
  ```

# 准备训练环境
## 安装昇腾环境
请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境。本仓已支持表1中软件版本。
  
  **表 1**  昇腾软件版本支持表

  |        软件类型        |   支持版本   |
  |:------------------:|:--------:|
  | FrameworkPTAdapter | 6.0.0  |
  |       CANN         | 8.0.0  |

## 安装模型环境

 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 2**  版本支持表

  |      三方库       |  支持版本  |
  |:--------------:|:------:|
  |    PyTorch     |  2.1   |
  |    Driving SDK   | 6.0.0 |
  |      mmcv      |  1.x   |
  |     mmdet      | 2.28.2 |

- 安装Driving SDK

  请参考昇腾[Driving SDK](https://gitee.com/ascend/DrivingSDK)代码仓说明编译安装Driving SDK

- 安装基础依赖

  在模型源码包根目录下执行命令，安装模型需要的依赖。
  
  ```shell
  pip install -r requirements.txt
  ```

- 源码安装mmcv

  ```shell
    git clone -b 1.x https://github.com/open-mmlab/mmcv.git
    cp mmcv.patch mmcv
    cd mmcv
    git apply mmcv.patch
    MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py install
  ```

- 源码安装mmdet

  ```shell
    git clone -b v2.28.2 https://github.com/open-mmlab/mmdetection.git
    cp mmdet.patch mmdetection
    cd mmdetection
    git apply mmdet.patch
    pip install -e .
  ```

- 模型代码Patch
  ```shell
  git clone https://github.com/HorizonRobotics/Sparse4D.git
  cp Sparse4D.patch Sparse4D
  cd Sparse4D
  git checkout c41df4bbf7bc82490f11ff55173abfcb3fb91425
  git apply Sparse4D.patch
  ```

# 准备数据集

## 预训练数据集
用户自行获取*nuscenes*数据集，在源码目录创建软连接`data/nuscenes`指向解压后的nuscenes数据目录
  ```shell
  sparse4d_path="path/to/sparse4d"
  cd ${sparse4d_path}
  mkdir data
  ln -s path/to/nuscenes ./data/nuscenes
  ```

运行数据预处理脚本生成Sparse4D模型训练需要的pkl文件
  ```shell
  pkl_path="data/nuscenes_anno_pkls"
  mkdir -p ${pkl_path}
  #对于mini数据集
  python3 tools/nuscenes_converter.py --version v1.0-mini --info_prefix ${pkl_path}/nuscenes-mini
  #对于完整数据集
  python3 tools/nuscenes_converter.py --version v1.0-trainval,v1.0-test --info_prefix ${pkl_path}/nuscenes
  ```

通过K-means生成初始锚框
  ```shell
  python3 tools/anchor_generator.py --ann_file ${pkl_path}/nuscenes_infos_train.pkl
  ```

## 获取预训练权重
下载backbone预训练权重
  ```shell
  mkdir ckpt
  wget https://download.pytorch.org/models/resnet50-19c8e357.pth -O ckpt/resnet50-19c8e357.pth
  ```

# 快速开始

## 训练模型

开始训练与验证
```shell
  # train
  # num代表训练所需卡数
  bash local_train.sh sparse4dv3_temporal_r50_1x8_bs6_256x704 num

  # test
  # num代表验证所需卡数
  bash local_test.sh sparse4dv3_temporal_r50_1x8_bs6_256x704 num path/to/checkpoint
  ```


## 训练结果

**表 3** 训练结果展示表

|      芯片       | 卡数 | mAP  | FPS | 平均step耗时(s) | Max epochs |
|:-------------:|----|:----:|:----:|:----------:|:----------:|
|      竞品A      | 1p | - | - | 0.793 |   1      |
|      竞品A      | 8p | 0.4534 | - | 0.777 |   100     |
| Atlas 800T A2   | 1p | - | - | 1.321 |  1      |
| Atlas 800T A2   | 8p | 0.4571 | - | 1.221 |     100     |




# 版本说明
## 变更
2025.1.23：首次发布。

## FAQ
暂无。



