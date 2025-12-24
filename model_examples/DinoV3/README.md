# DinoV3 

## 目录

- [DinoV3](#dinov3)
  - [目录](#目录)
- [简介](#简介)
  - [概述](#概述)
  - [模型介绍](#模型介绍)
  - [支持任务列表](#支持任务列表)
  - [代码实现](#代码实现)
- [准备训练环境](#准备训练环境)
  - [安装昇腾环境](#安装昇腾环境)
  - [安装模型环境](#安装模型环境)
- [准备训练数据集](#准备训练数据集)
- [执行训练](#执行训练)
  - [训练结果展示](#训练结果展示)
- [版本说明](#版本说明)
  - [变更](#变更)
  - [FAQ](#faq)

# 简介

## 概述

本仓提供DinoV3模型`Pretrain`阶段在昇腾上的训练步骤，精度与竞品对齐，性能达到0.63倍竞品。

## 模型介绍

DINOv3是一项突破性的自监督视觉模型研究，其核心目标是无需人工标注数据即可构建通用的视觉基础模型。该研究通过规模化扩展数据集与模型，并引入创新技术（如解决特征退化的Gram锚定方法）和灵活的后处理策略，成功实现了这一愿景。实验表明，DINOv3生成的高质量视觉特征在多种任务上显著超越了此前需要大量标注数据的专用模型，展现了卓越的泛化能力。最终，研究团队开源了适配不同场景的模型系列，为视觉领域提供了强大且可扩展的解决方案，推动了自监督学习的发展。

## 支持任务列表

本仓已经支持以下模型任务类型

|    模型     | 任务列表 | 是否支持 |
| :---------: | :------: | :------: |
| DinoV3 |   pretrain   |    ✔     |

本仓已经支持以下模型

|    Backbone     | Method | 训练方式 |
| :---------: | :------: | :------: |
| ViT-L/16 |   DinoV3   |    Mixed-precision Training     |

## 代码实现

- 参考实现：
  
  ```
  url=https://github.com/facebookresearch/dinov3
  commit_id=cb054165e9ec6bd86dbae674416eb36c8349adcb
  ```

- 适配昇腾 AI 处理器的实现：
    ```
    url=https://gitcode.com/Ascend/DrivingSDK.git
    code_path=model_examples/DinoV3
    ```

# 准备训练环境

## 安装昇腾环境

请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/Pytorch/720/configandinstg/instg/insg_0001.html)》文档搭建昇腾环境，本仓已支持表1中软件版本。

**表 1**  昇腾软件版本支持表

|     软件类型      | 首次支持版本 |
| :---------------: | :------: |
| FrameworkPTAdapter | 7.2.0  |
|       CANN        | 8.3.RC1  |


## 安装模型环境

当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

**表 2**  版本支持表

|      三方库       |  支持版本  |
|:--------------:|:------:|
|    Python      | 3.11 |
|    PyTorch     |  2.8.0   |


0. 激活 CANN 环境

1. 准备模型源码

    在 DinoV3 根目录下，克隆原始仓，使用patch文件替换其中部分代码并安装
    ```sh
    git clone https://github.com/facebookresearch/dinov3.git
    cd dinov3
    git checkout cb054165e9ec6bd86dbae674416eb36c8349adcb
    cp -f ../dinov3.patch .
    git apply --reject --whitespace=fix dinov3.patch
    cp -rf ../test ./
    ```
    
2. 安装依赖

    ```sh
    pip install -r requirements.txt
    ```

3. 修改 `Reduce` 算子

目前昇腾不支持模型中使用的`PREMUL_SUM` reduce算子。然而在模型中，factor为1的情况下，`PREMUL_SUM` 与 `SUM`算子等价。因此，做如下替换：
    1. `pip show torch` 查看torch的安装路径，如xxx/lib/python3.11/site-packages/torch, 记做`$torch`
    2. 将`$torch/distributed/distributed_c10d.py` 文件中第4434行修改为：
    
    ```
    -   opts.reduceOp=op
    
    +   opts.reduceOp=torch.distributed.Reduceop.SUM
    ```


# 准备训练数据集

1. 前往[ImageNet官网](https://www.image-net.org/download.php)下载ImageNet-1K数据，将数据集的存放路径记做`$DATA_ROOT`。数据集结构如下：

    ```
    <DATA_ROOT>/test/ILSVRC2012_test_00000001.JPEG
    <DATA_ROOT>/test/[..]
    <DATA_ROOT>/test/ILSVRC2012_test_00100000.JPEG
    <DATA_ROOT>/train/n01440764/n01440764_10026.JPEG
    <DATA_ROOT>/train/[...]
    <DATA_ROOT>/train/n15075141/n15075141_9993.JPEG
    <DATA_ROOT>/val/n01440764/ILSVRC2012_val_00000293.JPEG
    <DATA_ROOT>/val/[...]
    <DATA_ROOT>/val/n15075141/ILSVRC2012_val_00049174.JPEG
    <DATA_ROOT>/labels.txt
    ```

2. 按照模型官网 `Data Preparation` 章节，进行数据预处理，将输出路径定为 `$EXTRA`。
    
    ```
    from dinov3.data.datasets import ImageNet

    for split in ImageNet.Split:
        dataset = ImageNet(split=split, root="<DATA_ROOT>", extra="<EXTRA>")
        dataset.dump_extra()
    
    ```
    生成的文件如下：
    ```
    <EXTRA>/class-ids-TRAIN.npy
    <EXTRA>/class-ids-VAL.npy
    <EXTRA>/class-names-TRAIN.npy
    <EXTRA>/class-names-VAL.npy
    <EXTRA>/entries-TEST.npy
    <EXTRA>/entries-TRAIN.npy
    <EXTRA>/entries-VAL.npy
    ```   

# 执行训练
以下提供本文中使用到的参数解释

**表 1** 训练参数
|     参数      | 可选/必选 | 说明 | 
| :-----------: | :--: | :---------------: | 
|     DATA_ROOT     |  必选  |    ImageNet-1k 数据集存放路径       |  
| EXTRA |  必选  |    数据预处理步骤中生成数据存放路径      |  
| OUTPUT_PATH |  必选  |     模型训练生成的权重保存路径      |  

cd至模型根目录（DrivingSDK/model_examples/DinoV3/dinov3）下执行以下命令，进行

* 单机8卡精度(1250 epochs)
    ```sh
    bash test/train_full.sh $DATA_ROOT $EXTRA $OUTPUT_PATH
    ```

* 单机8卡性能(1 epoch)
    ```sh
    bash test/train_performance.sh $DATA_ROOT $EXTRA $OUTPUT_PATH
    ```


## 训练结果展示

本次使用8卡环境对DinoV3模型进行训练，125000步后训练loss为9.9375，性能为 393.8 FPS

**表 2** 训练结果展示表

|     芯片      | 卡数 | global batch size | max steps | final loss | FPS|
| :-----------: | :--: | :---------------: | :---: | :--------------------: |:--------------------: |
|     竞品A     |  8p  |         512       |  125000 |  9.9345 |  616.8 |
| Atlas 800T A2 |  8p  |         512      |  125000 |   9.9374 | 393.8 |

**表 3** 结果字段说明
|     字段      | 说明 | 
| :-----------: | :---------------: | 
|     卡数     |    训练使用的昇腾芯片数量      |  
| global batch size  |    每次训练迭代处理的样本总数      |  
| max steps  |     训练的迭代数      |  
| final loss |     训练结束时的损失      |  
| FPS |     Frames Per Second，平均每秒处理的样本总数      |  

# 版本说明

## 变更

2025.11.11：首次发布


## FAQ
暂无