# HPTR

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

HPTR 是一种用于自动驾驶实时运动预测的层次化Transformer模型，通过引入基于相对姿态编码的K近邻注意力机制和异步令牌更新，在预测多交通参与者时显著降低计算开销并提升扩展性，同时通过共享上下文实现高效推理，在Waymo和Argoverse-2数据集上达到先进的端到端性能。

## 支持任务列表
本仓已经支持以下模型任务类型

|    模型     | 任务列表 | 是否支持 |
| :---------: | :------: | :------: |
| HPTR |   训练   |    ✔     |

## 代码实现

- 参考实现：

  ```
  url=https://github.com/zhejz/HPTR
  commit_id=d2c1cb31ff5138ebf4b2490e2689c2f9da962120
  ```
- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/DrivingSDK.git
  code_path=model_examples/HPTR
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
    cd DrivingSDK/model_examples/HPTR
    git clone https://github.com/zhejz/HPTR.git
    cd HPTR
    git checkout d2c1cb31ff5138ebf4b2490e2689c2f9da962120
    ```
    将模型根目录记作 `model-root-path`
    
- 使用 patch 文件：
    ```
    cp -f ../HPTR.patch .
    git apply --reject --whitespace=fix HPTR.patch
    cp -rf ../test .
    ```


- 安装 Driving SDK 加速库，安装 master 分支，具体方法参考[原仓](https://gitee.com/ascend/DrivingSDK)。

- 在应用过patch的模型根目录下，安装相关依赖：

  ```
  python -m pip install pip==22.2.2
  pip install -r requirements.txt
  ```

- 安装 waymo-open-dataset 库：

    对于 x86 架构 Linux 系统：
    ```
    pip install waymo-open-dataset-tf-2-11-0==1.5.0
    ```
    对于 arm64 架构 Linux 系统，waymo 官方并没有提供预先编译好 whl 包。为了方便用户使用，我们提供 arm64 系统编译的 whl 包，可以直接在华为云 OBS 上进行下载：
    ```
    wget --no-check-certificate https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/DrivingSDK/packages/waymo_open_dataset_tf_2.11.0-1.5.0-py3-none-any.whl
    pip install waymo_open_dataset_tf_2.11.0-1.5.0-py3-none-any.whl
    ```

### 准备数据集

- 根据原仓 **Prepare Datasets** 章节准备数据集，处理好的数据集目录及结构如下：

```
HPTR
├──h5_womd_hptr/
   ├── training.h5
   ├── validation.h5
├──h5_av2_hptr/
   ├── training.h5
   ├── validation.h5
```   

### 修改训练数据集

- 修改训练脚本`train_full_8p.sh`和`train_performance_8p.sh` 文件中第6行:

    - `womd` 代表 Waymo-motion 数据集
    - `av2` 代表 Argoverse-2 数据集

## 快速开始

### 训练任务

本任务主要提供**单机**的**8卡**训练脚本。

### 开始训练
- 进入应用过patch的模型根目录`model-root-path`。

- 在`model-root-path`路径下，运行训练脚本。

     该模型支持单机8卡训练。

     - 单机8卡精度训练

     ```
     bash test/train_full_8p.sh
     ```

     - 单机8卡性能训练

     ```
     bash test/train_performance_8p.sh
     ```


### 训练结果

说明：实验结果使用了完整的 av2 数据集进行训练和验证；使用了 womd 数据集训练集的前1/10进行训练，测试集的前1/3进行验证。

| 芯片          | 卡数 | 数据集 | global batchsize |Precision | epoch |  minADE   |单步迭代耗时(ms) |
| --------- | :--: |:--: |:-----: | :-------: | :---: | :----: | :-----:|
| 竞品A           |  8p  |  av2   |  64  |   fp32    |  38   | 0.800 | 1774 |
| Atlas 800T A2 |  8p  |    av2    |  64   |   fp32    |  38   | 0.788 | 2547 |
| 竞品A           |  8p  |  womd   |  64  |   fp32    |  40  | 0.756 | 1816 |
| Atlas 800T A2 |  8p  |    womd    |  64   |   fp32    |  40   | 0.759 | 2726 |



## 变更说明

2025.03.12：首次发布

## FAQ

无