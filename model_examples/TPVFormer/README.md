# TPVFormer_for_PyTorch

# 目录
- [TPVFormer\_for\_PyTorch](#tpvformer_for_pytorch)
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
- [变更说明](#变更说明)
  - [FAQ](#faq)


# 简介

## 模型介绍

*TPVFormer*是一个经典的深度学习网络，可用于3D目标检测，3D语义分割，视频动作识别，自动驾驶等场景。它通过结合三维几何信息与时空Transformer来解决3D场景理解和视频分析中的复杂任务。其核心是将Transformer的注意力机制扩展到空间-时间域，用来捕捉帧间的动态信息和空间内的上下文关系。同时还融入3D几何信息，通过对输入数据预处理获取点云或3D网格的结构信息，然后与Transformer的特征表示融合，以增强模型的空间理解能力。

## 代码实现
- 参考实现：

  ```
  url=https://github.com/wzzheng/TPVFormer.git
  commit_id=459bc060901c9c4920f802252f04b290a449e4a1
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
  | Driving SDK | 6.0.0 |
  |      mmcv      |  1.x   |
  |     mmdet      | 2.28.2 |
  | mmsegmentation | 0.30.0 |

- 安装Mx_Driving-Accelerator

  请参考昇腾[Driving SDK](https://gitee.com/ascend/DrivingSDK)代码仓说明编译安装Driving SDK

- 克隆代码到当前目录并使用patch文件
```
git clone https://github.com/wzzheng/TPVFormer.git
cp -f TPVFormer.patch TPVFormer
cp -r test TPVFormer
cd TPVFormer
git checkout 459bc060901c9c4920f802252f04b290a449e4a1
git apply TPVFormer.patch
```

- 安装基础依赖

  在模型源码包根目录下执行命令，安装模型需要的依赖。
  
  ```
  pip install opencv-python==4.9.0.80

  pip install -r requirements.txt
  ```

- 源码安装 mmcv 1.x
  ```
  git clone -b 1.x https://github.com/open-mmlab/mmcv.git
  cp -f mmcv_need/distributed.py mmcv/mmcv/parallel/distributed.py
  cp -f mmcv_need/modulated_deform_conv.py mmcv/mmcv/ops/modulated_deform_conv.py
  cp -f mmcv_need/optimizer.py mmcv/mmcv/runner/hooks/optimizer.py
  cd mmcv
  MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py install
  ```

- 源码安装 mmdet 2.28.2
  ```
  git clone -b v2.28.2 https://github.com/open-mmlab/mmdetection.git
  cp -f mmdet_need/resnet.py mmdetection/mmdet/models/backbones/resnet.py
  cd mmdetection
  pip install -e .
  ```

- 安装mmsegmentation
  ```
  pip install mmsegmentation==0.30.0
  ```

# 准备数据集

## 预训练数据集
用户自行获取*nuscenes*数据集，在源码目录创建软连接`data/nuscenes`指向解压后的nuscenes数据目录

## 获取预训练权重
1. 下载预训练权重文件[r101_dcn_fcos3d_pretrain.pth](https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth)拷贝至目录`ckpts/`
2. 下载预生成的[nuscenes_infos_train.pkl](https://cloud.tsinghua.edu.cn/f/ede3023e01874b26bead/?dl=1)和[nuscenes_infos_val.pkl](https://cloud.tsinghua.edu.cn/f/61d839064a334630ac55/?dl=1) 拷贝至目录`data/`

  整理好的数据集目录如下:

```
TPVFormer_for_PyTorch/data
    nuscenes
        lidarseg
        maps
        samples
        sweeps
        v1.0-trainval
    nuscenes_infos_train.pkl
    nuscenes_infos_val.pkl
```

# 快速开始

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. *lidar segmentation dim64*任务训练

- 单机单卡训练

     ```
     bash ./test/train_1p.sh --py_config=config/tpv_lidarseg_dim64.py # 单卡精度
     
     bash ./test/train_1p.sh --py_config=config/tpv_lidarseg_dim64.py --performance=1  # 单卡性能
     ```
   
- 单机8卡训练

     ```
     bash ./test/train_8p.sh --py_config=config/tpv_lidarseg_dim64.py # 8卡精度

     bash ./test/train_8p.sh --py_config=config/tpv_lidarseg_dim64.py --performance=1 # 8卡性能 
     ```

  模型训练脚本参数说明如下。
   
   ```
   公共参数：
   --py_config                              //不同类型任务配置文件
   --performance                            //--performance=1开启性能测试，默认不开启
   --work_dir                               //输出路径包括日志和训练参数
   ```

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。


## 训练结果

**表 3**  *lidar segmentation dim64*训练结果展示表

|  芯片      | 卡数 |  Max epochs  | global_batch_size  | mIoU | 性能-单步迭代耗时(ms) | FPS |
|:--------:|----|:------:|:----:|:----:|:----------:|:----------:|
|   竞品A    | 8p | 24 | 8 | 67.984% | 775 | 10.32 |
| Atlas 800T A2 | 8p | 24 | 8 | 68.661% | 1195 | 6.69 |

# 变更说明
2024.05.13：首次发布。
2024.10.27：性能优化。

## FAQ
暂无。



