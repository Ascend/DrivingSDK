# Yolov8

# 概述
YOLOv8（You Only Look Once version 8）是YOLO系列算法中的第八个版本，是一种基于深度学习的目标检测模型。尽管其主要用途是目标检测（即同时预测目标的位置和类别），但YOLOv8也具备强大的图像分类能力，通过检测图像中的目标并确定其类别，间接实现了分类功能。


- 参考实现：

  ```
  url=https://github.com/ultralytics/ultralytics
  commit 25307552100e4c03c8fec7b0f7286b4244018e15
  ```

# 支持模型

| Modality  | Mode | 训练方式 |
|-----------|----- |------|
| coco2017  | images | FP32 |

# 准备训练环境

## 安装昇腾环境

  请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表1中软件版本。

  **表 1**  昇腾软件版本支持表

|     软件类型      | 支持版本 |
| :---------------: | :------: |
| FrameworkPTAdaper | 6.0.0  |
|       CANN        | 8.0.0 |

## 安装模型环境
- 当前模型支持的 PyTorch 版本如下表所示。

  **表 2**  版本支持表

  | Torch_Version |
  |:-------------:|
  |  PyTorch 2.1  |


- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》搭建 torch 环境。
- 安装依赖。
  首先进入代码目录：

  ```
  cd model_examples/Yolov8
  ```

  然后按步骤安装依赖：

  1. 设置Yolov8

  ```
  git clone https://github.com/ultralytics/ultralytics
  git checkout tags/v8.2.103
  git apply Yolov8.patch
  ```

  2. 安装Driving SDK加速库
  ```
  git clone https://gitee.com/ascend/DrivingSDK.git -b master
  cd mx_driving
  bash ci/build.sh --python=3.8
  cd dist
  pip3 install mx_driving-1.0.0+git{commit_id}-cp{python_version}-linux_{arch}.whl
  ```
  
  3. 对应的环境变量添加至 test/env_npu.sh 文件中。

## 准备数据集

1. 用户需自行下载 coco2017 数据集，放在 datasets 目录下，或者自行构建软连接，结构如下：

  ```
  datasets
  |-- annotations
  |-- images
      |-- train2017
      |-- val2017   
  |-- labels
      |-- train2017
      |-- val2017
  ```

  除了 datasets 目录，其他为原始仓库已有目录。

## 下载预训练权重
1. 在官网下载预训练权重yolov8n.pt，放在和模型脚本同级目录下


# 开始训练

回到最开始的模型目录：

```
cd model_examples/Yolov8
```

- 单机8卡训练

  ```shell
  bash test/train_full_8p_base_fp32.sh # 8卡训练，默认训练20个epochs
  ```

# 训练结果

| NAME             | Mode  | Epoch | mAP50|	mAP50-95|	FPS   |
|------------------|-----  |-------|------|-------|-------|
| 8p-Atlas 800T A2 | FP32  | 20    | 0.51 | 0.361 | 214.64|
| 8p-竞品A         | FP32  | 20    | 0.511 | 0.362 | 479.73 |

# 版本说明

## 变更

2024.12.30：首次发布。

## FAQ
