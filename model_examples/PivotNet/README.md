# PivotNet for PyTorch

## 目录

- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [支持任务列表](#支持任务列表)
  - [代码实现](#代码实现)
- [PivotNet（在研版本）](#PivotNet（在研版本）)
  - [准备训练环境](#准备训练环境)
    - [安装昇腾环境](#安装昇腾环境)
    - [安装模型环境](#安装模型环境)
    - [模型数据准备](#模型数据准备)
  - [快速开始](#快速开始)
    - [开始训练](#开始训练)
    - [训练结果](#训练结果)
- [公网地址说明](#公网地址说明)
- [变更说明](#变更说明)
- [FAQ](#FAQ)

# 简介

## 模型介绍

在自动驾驶研究领域，构建高分辨率地图（HD-map）的矢量化方法引起了广泛的关注。为了实现精确的地图元素学习，提出了一种名为PivotNet的简单且有效的架构，它采用统一的枢轴基于地图表示，并被形式化为直接集合预测范式。具体来说，PivotNet提出了一种名为Point-to-Line Mask的模块，以在网络中编码从属关系和几何的点线先验，以及Pivot Dynamic Matching的模块，以通过引入序列匹配概念来建模动态点序列的拓扑。为了监督矢量化点预测的位置和拓扑，PivotNet提出了一种动态矢量化序列损失。PivotNet由四个主要组件组成：摄像头特征提取器、平面视图特征解码器、线感知点解码器和枢轴点预测器。它以RGB图像作为输入，并生成灵活且紧凑的矢量化表示，无需进行任何后处理。

## 支持任务列表

本仓已经支持以下模型任务类型

|   模型   | 任务列表 | 是否支持 |
| :------: | :------: | :------: |
| PivotNet |   训练   |    ✔     |

## 代码实现

- 参考实现：

```
url=https://github.com/wenjie710/PivotNet
commit_id=3f334e499bae6c9e2f3ed06cf77ac6cbd22d0ba8
```

- 适配昇腾 AI 处理器的实现：

```
url=https://gitee.com/ascend/DrivingSDK.git
code_path=model_examples/PivotNet
```

# PivotNet

## 准备训练环境

### 安装昇腾环境

请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表1中软件版本。

**表 1** 昇腾软件版本支持表

|     软件类型      | 支持版本 |
| :---------------: | :------: |
| FrameworkPTAdaper | 7.0.0 |
|       CANN        | 8.1.RC1 |

### 安装模型环境

**表 2** 三方库版本支持表

| 三方库  | 支持版本 |
| :-----: | :------: |
| PyTorch |  2.1.0   |

0. 激活 CANN 环境

   将 CANN 包目录记作 cann_root_dir，执行以下命令以激活环境

   ```
   source {cann_root_dir}/set_env.sh
   ```

1. 参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》安装 2.1.0 版本的 PyTorch 框架和 torch_npu 插件。

2. 安装mmcv

   在模型根目录下，克隆mmcv仓，并进入mmcv目录安装。

   ```
   git clone -b 1.x https://github.com/open-mmlab/mmcv
   cd mmcv
   MMCV_WITH_OPS=1 python setup.py install
   ```

3. 安装 detectron2

   ```
   python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
   ```

4. 安装 Driving SDK 加速库

   安装方法参考[官方文档](https://gitee.com/ascend/DrivingSDK/wikis/DrivingSDK)。


5. 设置PivotNet
    ```
   git clone https://github.com/wenjie710/PivotNet.git
   cp -f pivotnet.patch PivotNet
   cd PivotNet
   git checkout 3f334e499bae6c9e2f3ed06cf77ac6cbd22d0ba8
   git apply --reject --whitespace=fix pivotnet.patch
   pip install -r requirement.txt
   ```

### 模型数据准备

进入[NuScenes](https://www.nuscenes.org/nuscenes#download)官网，下载 Nuscenes 数据集。将数据集上传到服务器任意路径下并解压，数据集结构排布成如下格式：

- 文件夹结构

```
  assets
    | -- weights (resnet, swin-t, efficient-b0, ...)
    | -- 
  mapmaster
  configs
  data
    | -- nuscenes
      | -- samples (CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, ...)
      | -- v1.0-trainval
      | -- ...
      | -- customer
        | -- pivot-bezier
          | -- *.npz
  tools
```

- 下载模型依赖的权重

```
cd /path/to/pivotnet
cd assets/weights
wget --no-check-certificate https://github.com/wenjie710/PivotNet/releases/download/v1.0/efficientnet-b0-355c32eb.pth .
wget --no-check-certificate https://github.com/wenjie710/PivotNet/releases/download/v1.0/resnet50-0676ba61.pth .
wget --no-check-certificate https://github.com/wenjie710/PivotNet/releases/download/v1.0/upernet_swin_tiny_patch4_window7_512x512.pth .
```

- 生成模型训练数据

```
cd /path/to/pivotnet
python3 tools/anno_converter/nuscenes/convert.py -d .data/nuscenes
```

## 快速开始

### 训练任务

本任务主要提供**单机**的**8卡**训练脚本。

### 开始训练

在模型根目录下，运行训练脚本。

```  
cd model_examples/PivotNet
```

- 单机8卡性能

  ```
  bash test/train_8p_performance.sh
  ```

- 单机8卡精度

  ```
  bash test/train_full_8p.sh
  ```

### 训练结果

|     芯片      | 卡数 | global batch size | epoch | mAP<sup>avg</sup>@EASY | mAP<sup>avg</sup>@HARD | 性能-单步迭代耗时(s) |
| :-----------: | :--: | :---------------: | :---: | :--------------------: | :--------------------: |--------------|
|     竞品A     |  8p  |         1         |  30   |         0.616          |         0.436          | 0.58         |
| Atlas 800T A2 |  8p  |         1         |  30   |         0.619          |         0.438          | 1.31         |

## 性能优化

参考指南：
[性能优化](https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/performance_tuning_0058.html)
1. 参考[编译优化](https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/performance_tuning_0061.html)完成Python\pytorch\torch_npu编译优化
2. 参考[OS性能优化](https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/performance_tuning_0067.html)完成高性能内存库替换

### 优化后训练结果

|     芯片      | 卡数 | global batch size | epoch | mAP<sup>avg</sup>@EASY | mAP<sup>avg</sup>@HARD | 性能-单步迭代耗时(s) |
| :-----------: | :--: | :---------------: | :---: |:----------------------:|:----------------------:| -------------------- |
|     竞品A     |  8p  |         1         |  30   |         0.616          |         0.436          | 0.58                 |
| Atlas 800T A2 |  8p  |         1         |  30   |         0.619          |         0.438          | 1.04                 |

# 变更说明

2024.12.20：首次发布

2025.03.17：性能优化

# FAQ

无