# MultiPath++

## 目录

- [模型介绍](#模型介绍)
- [支持任务列表](#支持任务列表)
- [代码实现](#代码实现)
- [准备训练环境](#准备训练环境)
- [快速开始](#快速开始)
    - [训练任务](#训练任务) 
    - [开始训练](#开始训练)
    - [训练结果](#训练结果)
-   [变更说明](#变更说明)
-   [FAQ](#faq)

## 模型介绍

Multipath++ 是自动驾驶轨迹预测模型，通过改进多模态概率建模和场景编码，采用 Transformer 架构融合高精地图与障碍物动态，优化轨迹生成。其利用隐变量模型提升预测多样性，结合课程学习策略，在准确性与实时性上显著提升，适用于复杂交通场景。

## 支持任务列表
本仓已经支持以下模型任务类型

|    模型     | 任务列表 | 是否支持 |
| :---------: | :------: | :------: |
| MultiPath++ |   训练   |    ✔     |

## 代码实现

- 参考实现：

  ```
  url=https://github.com/stepankonev/waymo-motion-prediction-challenge-2022-multipath-plus-plus
  commit_id=359670b954431d8d26b6807cbd4e5aa1ebbf98dd
  ```
- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/DrivingSDK.git
  code_path=model_examples/MultiPath++
  ```

## 准备训练环境

### 安装环境

  **表 1**  三方库版本支持表

| 三方库  | 支持版本 |
| :-----: | :------: |
| PyTorch |   2.1   |

### 安装昇腾环境

  请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表2中软件版本。

  **表 2**  昇腾软件版本支持表

|     软件类型      | 支持版本 |
| :---------------: | :------: |
| FrameworkPTAdaper | 7.0.0  |
|       CANN        | 8.1.RC1 |

- 克隆代码仓到当前目录：

    ```
    git clone https://gitee.com/ascend/DrivingSDK.git -b master
    cd DrivingSDK/model_examples/MultiPath++
    git clone https://github.com/stepankonev/waymo-motion-prediction-challenge-2022-multipath-plus-plus.git
    cd waymo-motion-prediction-challenge-2022-multipath-plus-plus
    git checkout 359670b954431d8d26b6807cbd4e5aa1ebbf98dd
    ```
    将模型根目录记作 `model-root-path`
    
- 使用 patch 文件：
    ```
    cp -f ../MultiPath++.patch .
    git apply --reject --whitespace=fix MultiPath++.patch
    cp -rf ../test ./code/
    ```


- 安装 Driving SDK 加速库，安装 master 分支，具体方法参考[原仓](https://gitee.com/ascend/DrivingSDK)。

- 在应用过patch的模型根目录下，安装相关依赖：

  ```
  pip install -r requirements.txt
  ```


### 准备数据集

- 根据原仓 **Code Usage** 章节准备数据集，处理好的数据集目录及结构如下：

```
prerendered/
├── training_sparse/
├── validation_sparse/
```   

### 修改config路径

- 将 `code/configs/final_RoP_Cov_Single.yaml` 文件中第6行、第28行替换为处理好的数据集文件夹 `training_sparse`、`validation_sparse` 在当前机器上的绝对路径。

## 快速开始

### 训练任务

本任务主要提供**单机**的**单卡**训练脚本。

### 开始训练
- 进入应用过patch的模型根目录`model-root-path`。

- 在`model-root-path`下创建保存模型checkpoints的文件夹。
  ```
  mkdir models
  ```

- 在`model-root-path`下的`code/`路径下，运行训练脚本。

     该模型支持单机单卡训练。

     - 单机单卡精度训练

     ```
     bash test/train_full_1p.sh
     ```

     - 单机单卡性能训练

     ```
     bash test/train_performance_1p.sh
     ```


### 训练结果
| 芯片          | 卡数 | global batch size | Precision | epoch |  loss   | 性能-单步迭代耗时(ms) |
| ------------- | :--: | :---------------: | :-------: | :---: | :----: | :-------------------: |
| 竞品A           |  1p  |         128         |   fp32    |  30   | 2.56 |     646         |
| Atlas 800T A2 |  1p  |         128         |   fp32    |  30   | 2.53 |   856         |


## 变更说明

2025.02.20：首次发布

## FAQ

1. 训练时偶发AssertionError导致训练中断（社区已知问题），重新拉起训练即可。
问题参考链接：[Assertion Error On Finiteness](https://github.com/stepankonev/waymo-motion-prediction-challenge-2022-multipath-plus-plus/issues/4)