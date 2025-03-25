# DETR for PyTorch

## 目录

- [DETR for PyTorch](#detr-for-pytorch)
  - [目录](#目录)
- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [支持任务列表](#支持任务列表)
  - [代码实现](#代码实现)
- [DETR](#detr)
  - [准备训练环境](#准备训练环境)
    - [安装昇腾环境](#安装昇腾环境)
    - [安装模型环境](#安装模型环境)
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

DETR提出了一种将对象检测视为直接集合预测问题，能够一次性预测所有的目标，其训练采用一种集合损失函数以端到端方式进行，集合损失定义在预测结果与真实目标的二部图匹配结果上；该方法简化了检测管道，有效地消除了对许多手工设计组件的需求，例如非最大抑制程序或锚点生成，简化了检测流程；和存在的其他检测方法不一样，DETR不需要任何定制的层，因此能够便捷的在任何包含transformer和CNN的深度框架中进行复现。

## 支持任务列表

本仓已经支持以下模型任务类型

| 模型 |    任务列表     | 是否支持 |
| :--: | :-------------: | :------: |
| DETR | detection train |    ✔     |

## 代码实现

- 参考实现：

```
url=https://github.com/facebookresearch/detr
commit_id=29901c51d7fe8712168b8d0d64351170bc0f83e0
```

- 适配昇腾 AI 处理器的实现：

```
url=https://gitee.com/ascend/DrivingSDK.git
code_path=model_examples/DETR
```

# DETR

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

| Torch_Version | 三方库依赖版本 |
| :-----: | :------: |
| PyTorch 2.1 | torchvision==0.16.0 |

0. 激活 CANN 环境

   将 CANN 包目录记作 cann_root_dir，执行以下命令以激活环境

   ```
   source {cann_root_dir}/set_env.sh
   ```

1. 参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》安装 2.1.0 版本的 PyTorch 框架和 torch_npu 插件。

2. 设置 DETR 并安装相关依赖

   ```
   cd model_examples/DETR
   git clone https://github.com/facebookresearch/detr.git --depth=1
   
   cd detr/
   git fetch --unshallow
   git checkout 29901c51d7fe8712168b8d0d64351170bc0f83e0
   cp -f ../detr.patch ./
   cp -rf ../test ./
   git apply detr.patch
   
   pip install -r requirements.txt
   ```

### 准备数据集

进入 [COCO](http://cocodataset.org/#download) 官网，下载 COCO2017 数据集。将数据集上传到服务器任意路径下并解压，数据集结构排布成如下格式：

```
coco_path/
  annotations/  # annotation json files
  train2017/    # train images
  val2017/      # val images
```

### 准备预训练权重

1. 联网情况下，预训练权重会自动下载。

2. 无网络情况下，可以通过该链接自行下载 [resnet50-0676ba61.pth](https://download.pytorch.org/models/resnet50-0676ba61.pth)，并拷贝至对应目录下。默认存储目录为 PyTorch 缓存目录：

   ```
   ~/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth
   ```

## 快速开始

### 训练任务

本任务主要提供**单机**的**8卡**训练脚本。

#### 开始训练

- 在模型根目录下，运行训练脚本。

  ```
  cd model_examples/DETR/detr
  bash test/train_8p_full.sh --data_path='/coco_path' # 8p 精度训练，epochs 默认 300
  bash test/train_8p_performance.sh --data_path='/coco_path' # 8p 性能训练，epochs 默认 20
  ```

  训练脚本参数说明：

  ```
  --data_path    # 数据集路径，必填
  --epochs       # 重复训练次数，可选项
  ```

#### 训练结果

|     芯片      | 卡数 | epoch | mAP(IoU=0.50:0.95) | FPS | Torch_Version |
| :-----------: | :--: | :----: | :----------------: | :--: | :--: |
|     竞品A     |  8p   |  300   |       0.410        | 126 | PyTorch 2.1 |
| Atlas 800T A2 |  8p   |  300   |       0.410        | 122 | PyTorch 2.1 |

# 变更说明

2024.11.21：首次发布

2025.1.21: 在训练脚本中增加结果保存功能，更新资料中的torch版本描述和训练结果

# FAQ

无
