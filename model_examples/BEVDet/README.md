# BEVDet

# 目录
- [BEVDet](#bevdet)
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
  - [模型训练](#模型训练)
  - [模型验证](#模型验证)
  - [训练结果](#训练结果)
- [变更说明](#变更说明)
  - [FAQ](#faq)


# 简介

## 模型介绍

*BEVDet*是一种用于3D目标检测的深度学习模型，可以从一个俯视图像中检测出三维空间中的物体，并预测他们的位置、大小和朝向。在自动驾驶、智能交通等领域中有广泛应用。其基于深度学习技术，使用卷积神经网络和残差网络，在训练过程中使用了大量的3D边界框数据，以优化模型的性能和准确性。

## 代码实现
- 参考实现：

  ```
  url=https://github.com/HuangJunJie2017/BEVDet.git
  commit_id=58c2587a8f89a1927926f0bdb6cde2917c91a9a5
  ```
  
- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/DrivingSDK.git
  code_path=model_examples/BEVDet
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
  | mmsegmentation | 0.30.0 |

- 安装Driving SDK

  请参考昇腾[Driving SDK](https://gitee.com/ascend/DrivingSDK)代码仓说明编译安装Driving SDK
  >【注意】请使用最新版本Driving SDK（包含bevpoolv3算子的版本）

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

- 模型代码更新
  ```shell
  git clone https://github.com/HuangJunJie2017/BEVDet.git
  cp -r test BEVDet
  cp BEVDet.patch BEVDet
  cd BEVDet
  git checkout 58c2587a8f89a1927926f0bdb6cde2917c91a9a5
  git apply BEVDet.patch
  ```
  
# 准备数据集

## 预训练数据集
用户自行获取*nuscenes*数据集，在源码目录创建软连接`data/nuscenes`指向解压后的nuscenes数据目录

运行数据预处理脚本生成BEVDet模型训练需要的pkl文件
  ```shell
  python tools/create_data_bevdet.py
  ```

  整理好的数据集目录如下:

```
BEVDet_for_PyTorch/data
    nuscenes
        lidarseg
        maps
        samples
        sweeps
        v1.0-trainval
        nuscenes_infos_train.pkl
        nuscenes_infos_val.pkl
        bevdetv3-nuscenes_infos_train.pkl
        bevdetv3-nuscenes_infos_val.pkl
```
## 获取预训练权重
1. 联网情况下，预训练权重会自动下载。
2. 无网络情况下，用户可以访问pytorch官网自行下载*resnet50*预训练[*resnet50-0676ba61.pth*](https://download.pytorch.org/models/resnet50-0676ba61.pth)。获取对应的预训练模型后，将预训练文件拷贝至对应目录。
```
${torch_hub}/checkpoints/resnet50-0676ba61.pth
```

# 快速开始

## 模型训练

1. 进入解压后的源码包根目录。

   ```shell
   cd /${模型文件夹名称} 
   ```

2. *lidar segmentation dim64*任务训练

- 单机单卡训练

     ```shell
     bash ./test/train_1p.sh --py_config=configs/bevdet/bevdet-r50.py # 单卡精度
     bash ./test/train_1p.sh --py_config=configs/bevdet/bevdet-r50.py --performance=1  # 单卡性能(只运行一个Epoch)
     ```
   
- 单机8卡训练

     ```shell
     bash ./test/train_8p.sh --py_config=configs/bevdet/bevdet-r50.py # 8卡精度
     bash ./test/train_8p.sh --py_config=configs/bevdet/bevdet-r50.py --performance=1 # 8卡性能(只运行一个Epoch)
     ```

  模型训练脚本参数说明如下。
   
   ```
   公共参数：
   --py_config                              //不同类型任务配置文件
   --performance                            //--performance=1开启性能测试，默认不开启
   --work_dir                               //输出路径包括日志和训练参数
   ```

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

## 模型验证

  ```shell
  py_config=configs/bevdet/bevdet-r50.py
  max_epochs=24
  eval_log_file=eval.log
  bash ./tools/dist_test.sh ${py_config} train_output_dir/epoch_${max_epochs}_ema.pth 8 --eval mAP | tee ${eval_log_file}
     
  ``` 


## 训练结果

**表 3** 训练结果展示表

|      芯片       | 卡数 | mAP  | FPS | 平均step耗时 | Max epochs |
|:-------------:|----|:----:|:----:|:----------:|:----------:|
|      竞品A      | 1p |  -   | - | -             |   1      |
|      竞品A      | 8p | 28.6 | 36.56  | 1.7489秒 |   24     |
| Atlas 800T A2 | 1p |  -   | | - | -               |  1      |
| Atlas 800T A2 | 8p | 28.3 | 51.87 | 1.2338秒|     24     |


# 变更说明
- 2024.11.25：首次发布。
- 2025.2.7： 修改为以patch模式发布模型形式。
- 2025.2.25：应用高斯融合算子提升模型性能

## FAQ
暂无。



