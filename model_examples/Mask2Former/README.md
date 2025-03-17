# Mask2Former-Swin_for_Pytorch

# 概述

## 简述

`Mask2Former` 是一种基于 `Transformer` 的语义分割模型，它可以在图像语义分割任务中实现像素级别的预测。相比于传统的卷积神经网络，`Transformer` 具有更好的并行性和全局感知能力，因此在语义分割任务中也表现出了很好的性能。`Mask2Former` 结合了 `Transformer` 的优势和语义分割的需求，它可以在保持高精度的同时，大大减少计算量和内存消耗。此外，`Mask2Former` 还可以通过自监督学习来进行预训练，从而进一步提高模型的性能。

## 代码实现

- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmsegmentation.git
  commit_id=c685fe6767c4cadf6b051983ca6208f1b9d1ccb8
  ```
  
- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/DrivingSDK.git
  code_path=model_examples/Mask2Former
  ```

# 准备训练环境

## 安装昇腾环境

请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境。本仓已支持表1中软件版本。

  **表 1**  昇腾软件版本支持表

|        软件类型       |   支持版本   |
|:--------:|:--------:|
| FrameworkPTAdapter | 7.0.0  |
|       CANN        | 8.1.RC1  |

## 安装模型环境

- 当前模型支持的 `PyTorch` 版本和已知三方库依赖如下表所示。

  **表 2**  版本支持表

  |      三方库      |  支持版本  |
  |:------:|:--------:|
  |    PyTorch    | 2.1.0  |
  |    mmengine   | 0.10.3 |
  |     mmdet     | 3.3.0  |

- 安装依赖

  首先进入模型目录：

  ```
  cd model_examples/Mask2Former
  ```
  
  1. 下载包含 `Mask2Former` 模型的原始仓库代码，并执行如下命令：

      ```
      git clone -b v1.2.2 https://github.com/open-mmlab/mmsegmentation.git
      cp -f mmsegmentation.patch mmsegmentation/
      cd mmsegmentation
      git apply mmsegmentation.patch
      pip install -r requirements.txt
      cd ../
      ```
  
  2. 源码编译安装 `mmcv rc4main` 分支：

      ```
      git clone -b rc4main https://github.com/momo609/mmcv.git
      cp -f mmcv.patch mmcv
      cd mmcv
      git checkout 10d07678aa14266637e23a97945ae39f00b6ce35
      git apply mmcv.patch
      pip install -r requirements/runtime.txt
      MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py build_ext
      MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py develop
      cd ../
      ```

  3. 源码编译安装 `mmdetection v3.3.0` 版本：

      ```
      git clone -b v3.3.0 https://github.com/open-mmlab/mmdetection.git
      cp -f mmdetection.patch mmdetection
      cd mmdetection
      git apply mmdetection.patch
      pip install -e .
      cd ../
      ```

  4. 请参考 [Driving SDK 使用说明](https://gitee.com/ascend/DrivingSDK/blob/master/README.md) 编译并安装 `Driving SDK` 包。



# 准备数据集

## 训练数据集

用户自行参考原始仓库说明获取 `cityscapes` 数据集，新建 `data` 目录，将数据解压或者软链接到 `mmsegmentation` 工程的 data 目录下，数据集样例结构如下：

```
data
├── cityscapes
│   ├── leftImg8bit
│   │   ├── train
│   │   ├── val
│   ├── gtFine
│   │   ├── train
└── └── └── val
```
如果放置目录不同，则需要根据需求调整 `mmsegmentation` 工程下 `configs/_base_/datasets/cityscapes.py` 配置文件中的 `data_root` 变量，其默认值如下：

```
data_root = 'data/cityscapes/'
```

在 `mmsegmentation` 目录下进行数据预处理：

```
# 安装相关依赖
pip install cityscapesscripts
# --nproc表示8个进程进行转换，也可以省略
python tools/dataset_converters/cityscapes.py data/cityscapes --nproc 8
```



## 获取预训练权重

1. 联网情况下，预训练权重会自动下载。
2. 无网络情况下，用户可以访问 [mmsegmentation官网](https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_base_patch4_window12_384_22k_20220317-e5c09f74.pth) 自行下载预训练模型，并将其拷贝至对应目录。目录样例如下：

    ```
    ${HOME}/.cache/torch/hub/checkpoints/swin_base_patch4_window12_384_22k_20220317-e5c09f74.pth
    ```

# 快速开始

## 训练模型

1. 回到最开始的模型目录：

    ```
    cd model_examples/Mask2Former
    ```

2. 运行训练脚本。

    启动 8 卡训练。
   
    ```
    bash ./test/train_full_8p.sh  # 8卡精度

    bash ./test/train_performance_8p.sh  # 8卡性能
    ```

    模型训练脚本参数说明如下。

    ```
    公共参数：
    --resume                                 //--resume=1 开启断点续训，默认不开启，train_performance_8p.sh脚本不支持该参数
    --batch_size                             //单卡batch_size，默认为2
    --num_workers                            //dataloader的workers数量，默认为2
    ```

    训练完成后，日志文件保存在 `test/output` 路径下，并输出模型训练精度和性能信息。


## 训练结果

**表 3**  训练结果展示表

|  芯片      | 卡数 | mIoU  | mAcc  | aAcc  |  FPS  | Max Iters |
|:--------:|----|:-----:|:------:|:------:|:-----:|:---------:|
|   竞品A    | 8p | 82.60 | 89.65 |   96.90   | 28.42 |   90000   |
| Atlas 900 A2 PODc | 8p | 83.26 | 89.93 |   96.92   | 26.03 |   90000   |

# 变更说明

2025.01.10：首次发布。
2025.02.15: Bug修复及Readme修改。

## FAQ

暂无。
