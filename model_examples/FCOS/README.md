# FCOS for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

FCOS是一个全卷积的one-stage目标检测模型，相比其他目标检测模型，FCOS没有锚框和提议，进而省去了相关的复杂计算，以及相关的超参，
这些超参通常对目标检测表现十分敏感。借助唯一的后处理NMS，结合ResNeXt-64X4d-101的FCOS在单模型和单尺度测试中取得了44.7%的AP，
因其简化性在现有one-stage目标检测模型中具有显著优势。

- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmdetection
  commit_id=cfd5d3a985b0249de009b67d04f37263e11cdf3d
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/DrivingSDK.git
  code_path=model_examples/FCOS
  ```


# 准备训练环境

## 准备环境

### 安装昇腾环境

请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境。

**表 1** 昇腾软件版本支持表

|     软件类型      | 支持版本 |
| :---------------: | :------: |
| FrameworkPTAdaper | 7.0.0 |
|       CANN        | 8.1.RC1 |

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 2**  版本支持表

  | Torch_Version | 三方库依赖版本 |
  | -------- | ------- |
  | PyTorch 2.1 | torchvision==0.16.0 |

### 安装模型环境

0. 激活 CANN 环境

  将 CANN 包目录记作 cann_root_dir，执行以下命令以激活环境
  ```
  source {cann_root_dir}/set_env.sh
  ```

1. 源码安装 mmcv

  在 FCOS 根目录下，克隆 mmcv 仓，并进入 mmcv 目录安装（安装完为 2.2.0 版本）

  ```
  git clone https://github.com/open-mmlab/mmcv
  cd mmcv/
  pip install -r requirements/runtime.txt
  MMCV_WITH_OPS=1 MAX_JOBS=8 FORCE_NPU=1 python setup.py build_ext
  MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py develop
  cd ../
  ```

2. 源码安装 mmengine

  在 FCOS 根目录下，克隆 mmengine 仓，并进入 mmengine 目录应用 patch 后安装
  
  ```
  yes | pip uninstall mmengine
  git clone -b v0.10.6 https://github.com/open-mmlab/mmengine.git
  cd mmengine/
  git checkout a8c74c346d2ef3e5501115529ba588accb5f2a03
  cp ../mmengine.patch ./
  git apply --reject mmengine.patch
  pip install -e .
  cd ../
  ```

3. 准备模型源码

  在 FCOS 根目录下，克隆 mmdetection 仓，替换其中部分代码

  ```
  git clone https://github.com/open-mmlab/mmdetection.git --depth=1
  cd mmdetection/
  git fetch --unshallow
  git checkout cfd5d3a985b0249de009b67d04f37263e11cdf3d
  cp -rf ../test/ ./
  cp ../mmdet.patch ./
  git apply --reject mmdet.patch
  ```

4. 安装其他依赖
  
  在 mmdetection 代码目录下，安装其他依赖

  ```
  pip install torchvision==0.16.0
  pip install -r requirements.txt
  ```

5. 安装 Driving SDK 加速库

  参考：https://gitee.com/ascend/DrivingSDK/blob/master/README.md

6. 替换高性能内存库

  参考[OS性能优化](https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/performance_tuning_0067.html)完成高性能内存库安装，并设置 tcmalloc 为优先加载。

## 准备数据集

1. 请用户自行准备好数据集，包含训练集、验证集和标签三部分，可选用的数据集有 COCO、PASCAL VOC 数据集等。
2. 上传数据集到服务器任意路径下解压，以 coco2017 为例，数据集在 `coco2017` 目录下分别存放于 train2017、val2017、annotations 文件夹下。
3. 当前提供的训练脚本中，是以 coco2017 数据集为例，在训练过程中进行数据预处理。 数据集目录结构参考如下：

   ```
   ├── coco2017
         ├──annotations
              ├── captions_train2017.json
              ├── captions_val2017.json
              ├── instances_train2017.json
              ├── instances_val2017.json
              ├── person_keypoints_train2017.json
              └── person_keypoints_val2017.json
             
         ├──train2017  
              ├── 000000000009.jpg
              ├── 000000000025.jpg
              ├── ...
         ├──val2017  
              ├── 000000000139.jpg
              ├── 000000000285.jpg
              ├── ...
   ```
   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。

## 准备预训练权重

1. 联网情况下，预训练权重会自动下载。

2. 无网络情况下，可以通过该链接自行下载 [resnet50_caffe-788b5fa3.pth](https://download.openmmlab.com/pretrain/third_party/resnet50_caffe-788b5fa3.pth)，并拷贝至对应目录下。默认存储目录为 PyTorch 缓存目录：

  ```
  ~/.cache/torch/hub/checkpoints/resnet50_caffe-788b5fa3.pth
  ```

# 开始训练

## 训练模型

该模型支持单机 8 卡训练，在 mmdetection 目录下运行训练脚本。

- 单机 8 卡训练

  ```
  cd model_examples/FCOS/mmdetection
  bash ./test/train_8p_full.sh --data_root=/home/datasets/coco2017 # 8p 精度训练，默认 12 epochs
  bash ./test/train_8p_full.sh --data_root=/home/datasets/coco2017 --max_epochs=2 # 8p 性能训练
  ```

  --data_path 参数填写数据集路径，需写到数据集的一级目录。

   模型训练脚本参数说明如下。

   ```
   --data_root                         // 数据集路径，必填参数
   --batch_size                        // 默认 4，训练批次大小，提高会降低ap值，可选参数
   --max_epochs                        // 默认 12，训练轮次，可选参数
   ```

   训练完成后，权重文件保存在当前路径的 `test/output` 目录下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

|  NAME  | cards | Epochs | FPS | mAP | Torch_Version |
|:------:|:--------:|:-----:|:-----:|:-----:|:-----:|
| 8p-竞品A | 8p | 12 | 196 | 0.353 | 2.1 |
| 8p-Atlas 800T A2 | 8p | 12 | 196 | 0.351 | 2.1 |

# 版本说明

## 变更

2024.11.8: 首次提交。

2025.1.23: 资料更新，mmengine 使用源码安装。

2025.2.26: 性能优化，增加训练结果保存到日志功能，删除单机单卡训练脚本，只保留单机 8 卡训练脚本。

## FAQ

无。
