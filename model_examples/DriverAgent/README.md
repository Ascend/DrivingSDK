# DriverAgent

# 目录
- [DriverAgent](#driveragent)
- [目录](#目录)
- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [代码实现](#代码实现)
- [准备训练环境](#准备训练环境)
  - [安装昇腾环境](#安装昇腾环境)
  - [安装模型环境](#安装模型环境)
- [准备数据集](#准备数据集)
  - [获取数据集](#获取数据集)
- [快速开始](#快速开始)
  - [训练模型](#训练模型)
  - [训练结果](#训练结果)
- [变更说明](#变更说明)

# 简介

## 模型介绍
DriverAgent是一项针对自动驾驶车辆行为交互性与轨迹真实性的车辆驾驶行为代理模型，其核心目标是在多车交互场景下对周边车辆的驾驶行为与驾驶轨迹进行预测。该模型通过对海量广泛场景下真实驾驶行为数据进行训练，学习并模仿真实驾驶员的驾驶偏好，使得模型能够在广泛的场景下准确模拟司机的驾驶行为与驾驶轨迹。同时，为学习不同驾驶员行为的多样性，该模型调整后支持输出多模态轨迹与对应模态的概率。基于真实驾驶行为数据进行实验表明，该模型在轨迹预测准确性（MSE/RMSE/MAPE等指标）上超越了SOTA模型，即使是在高频交互场景下该模型预测也保持较高的准确性。使用该行为代理模型，在自动驾驶领域，可以用于对周边车辆的行为推断从而更好地支持自动驾驶决策。

## 代码实现
- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitcode.com/Ascend/DrivingSDK.git
  code_path=model_examples/DriverAgent
  ```

# 准备训练环境
## 安装昇腾环境
请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境。本仓已支持表1中软件版本。
  
  **表 1**  昇腾软件版本支持表

  |        软件类型        |   首次支持版本   |
  |:------------------:|:--------:|
  | FrameworkPTAdapter | 7.3.0  |
  |       CANN         | 8.5.1  |

## 安装模型环境

 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 2**  版本支持表

  |      三方库       |  支持版本  |
  |:--------------:|:------:|
  |    Python      | 3.10 |
  |    PyTorch     |  2.7.1   |


在模型根目录下执行命令，安装模型需要的依赖
  
  ```shell
  conda create -n driverAgent python=3.10
  conda activate driverAgent
  cd DrivingSDK/model_examples/DriverAgent
  pip install -r requirements.txt
  ```

# 准备数据集

## 获取数据集

通过该链接  [datasets_driver_agent](https://www.jianguoyun.com/p/Dc3zm24QqILRDBiv7JkGIAA) 获取训练和验证数据集

# 快速开始

## 训练模型

```shell
bash train.sh
```

## 训练结果

**表 1** 训练结果展示表

|      芯片       | 卡数 | global batchsize  | epochs  |MSE | FPS|
|:-------------:|:----:|:----:|:----------:|:----------:|:----:|
|      竞品A      | 1p | 64 |30| 3.129 |  149  |
| Atlas 800T A2   | 1p | 64 |30| 2.866  |   180 |

# 变更说明

2026.02.24：首次发布
