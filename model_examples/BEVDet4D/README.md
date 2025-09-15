# BEVDet4D

# 目录
- [BEVDet4D](#bevdet)
- [目录](#目录)
- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [代码实现](#代码实现)
- [准备训练环境](#准备训练环境)
  - [安装昇腾环境](#安装昇腾环境)
  - [安装模型环境](#安装模型环境)
- [准备数据集](#准备数据集)
  - [获取训练数据集](#获取训练数据集)
  - [获取预训练权重](#获取预训练权重)
- [快速开始](#快速开始)
  - [模型训练](#模型训练)
  - [训练结果](#训练结果)
- [变更说明](#变更说明)
  - [FAQ](#faq)


# 简介

## 模型介绍

BEVDet4D 是一种将 BEVDet 从仅空间的 3D 扩展到时空 4D 工作空间的多相机三维目标检测范式。它通过融合前后帧特征，以极小的计算成本获取时间线索，将速度预测任务简化为位置偏移预测，在 nuScenes 基准测试中取得了优异的成绩。

## 代码实现
- 参考实现：

  ```
  url=https://github.com/HuangJunJie2017/BEVDet.git
  commit_id=58c2587a8f89a1927926f0bdb6cde2917c91a9a5
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitcode.com/Ascend/DrivingSDK.git
  code_path=model_examples/BEVDet4D
  ```

# 准备训练环境
## 安装昇腾环境
请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境。本仓已支持表1中软件版本。

  **表 1**  昇腾软件版本支持表

  |        软件类型        |   支持版本   |
  |:------------------:|:--------:|
  | FrameworkPTAdapter | 7.1.0  |
  |       CANN         | 8.2.RC1  |

## 安装模型环境

 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 2**  版本支持表

  |      三方库       |  支持版本  |
  |:--------------:|:------:|
  |    PyTorch     |  2.1   |
  |      mmcv      |  1.x   |
  |     mmdet      | 2.28.2 |
  | mmsegmentation | 0.30.0 |

- 安装Driving SDK

  请参考昇腾[Driving SDK](https://gitcode.com/Ascend/DrivingSDK)代码仓说明编译安装Driving SDK
  >【注意】请使用7.1.RC1及之后的Driving SDK

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
  cd ..
  ```

- 模型代码更新
  ```shell
  git clone https://github.com/HuangJunJie2017/BEVDet.git
  cp -r test BEVDet
  cp BEVDet.patch BEVDet
  cp bevdet4d_patch.py BEVDet/tools
  cd BEVDet
  git checkout 58c2587a8f89a1927926f0bdb6cde2917c91a9a5
  git apply BEVDet.patch
  ```

# 准备数据集

## 获取训练数据集
用户自行获取*nuscenes*数据集，在源码目录创建软连接`data/nuscenes`指向解压后的nuscenes数据目录

运行数据预处理脚本生成BEVDet模型训练需要的pkl文件
  ```shell
  python tools/create_data_bevdet.py
  ```

  整理好的数据集目录如下:

```
BEVDet/data
    nuscenes
        lidarseg
        maps
        samples
        sweeps
        v1.0-trainval
        bevdetv3-nuscenes_dbinfos_train.pkl
        bevdetv3-nuscenes_infos_train.pkl
        bevdetv3-nuscenes_infos_val.pkl
```
## 获取预训练权重
联网情况下，预训练权重会自动下载。无网络的情况参见 [FAQ](#faq)

# 快速开始

## 模型训练

1. 进入源码根目录

   ```shell
   cd /${模型文件夹名称}
   ```

2. 单机8卡训练

- 8卡性能
  ```shell
  # fp16 性能
  bash test/train_performance_8p.sh --py_config=configs/bevdet/bevdet-stbase-4d-stereo-512x1408-cbgs.py --fp16
  # fp32 性能
  bash test/train_performance_8p.sh --py_config=configs/bevdet/bevdet-stbase-4d-stereo-512x1408-cbgs.py
  ```
- 8卡精度
  ```shell
  # fp16 精度
  bash test/train_full_8p.sh --py_config=configs/bevdet/bevdet-stbase-4d-stereo-512x1408-cbgs.py --test=1 --fp16
  # fp32 精度
  bash test/train_full_8p.sh --py_config=configs/bevdet/bevdet-stbase-4d-stereo-512x1408-cbgs.py --test=1
  ```

  模型训练脚本参数说明如下。

   ```
   公共参数：
   --py_config                       //不同类型任务配置文件
   --test                            //--test=1固定随机性用于测试精度，默认不开启
   --fp16                            //使能fp16混合精度训练，默认精度为fp32
   --work_dir                        //输出路径包括日志和训练参数
   ```

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。



## 训练结果

**表 3** 训练结果展示表

|      芯片       | 精度 | 卡数 | Global Batchsize | Loss  | FPS | 平均step耗时 | Max epochs |
|:-------------:|:------:|:----:|:----:|:----:|:----:|:----------:|:----------:|
|      竞品A      |FP32 |8p | 64 |19.513 | 5.59  | 11.44秒 |   3    |
| Atlas 800T A2 |FP32 |8p | 64 | 19.402 | 7.04 | 9.09秒|     3    |
| Atlas 800T A2 |FP16 |8p | 64 | 19.371 | 8.82 | 7.26秒|     3    |

# 变更说明
- 2025.8.8：首次发布。

## FAQ
Q: 在无网络或设有防火墙的环境下如何下载预训练权重？

A: 无网络情况下，用户可以自行下载 *SwinTransformer* 预训练权重 [*swin_base_patch4_window12_384_22k.pth*](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth)。将下载好的权重拷贝至以下目录，其中 ${torch_hub} 替换为实际下载位置，默认为 ~/.cache/torch/hub
```
${torch_hub}/checkpoints/swin_base_patch4_window12_384_22k.pth
```




