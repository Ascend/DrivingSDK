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
  url=https://github.com/open-mmlab/mmdetection/tree/main
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/mxDriving/tree/master/model_examples/FCOS
  ```


# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 2.1 | torchvision==0.16.0 |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```
  pip install -r requirements.txt  # PyTorch2.1版本
  ```
  > **说明：** 
  >只需执行一条对应的PyTorch版本依赖安装命令。
  
- 安装mmcv环境。
  1. 安装mmcv，最好是2.2.0版本。
    ```
    git clone git://github.com/open-mmlab/mmcv.git
    cd mmcv
    MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py build_ext
    MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py develop
    pip3 show mmcv # 查看版本和路径
    ``` 
  2. 用mmengine_need里的文件替换mmengine中对应的文件。
  ```
    pip3 show mmengine # 查看版本和路径
    cp -f mmengine_need/loops.py ${mmengine_path}/mmengine/runner/loops.py
    ```
  

## 准备数据集

1. 请用户自行准备好数据集，包含训练集、验证集和标签三部分，可选用的数据集又COCO、PASCAL VOC数据集等。
2. 上传数据集到data文件夹，以coco2017为例，数据集在`data/coco`目录下分别存放于train2017、val2017、annotations文件夹下。
3. 当前提供的训练脚本中，是以coco2017数据集为例，在训练过程中进行数据预处理。 数据集目录结构参考如下：

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

# 开始训练

## 训练模型

1. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     ```
     bash ./test/train_1p.sh --data_root=/home/datasets/coco --batch_size=4 --max_epochs=1
     ```
     
   - 单机8卡训练

     ```
     bash ./test/train_8p.sh --data_root=/home/datasets/coco --batch_size=4 --max_epochs=1
     ```

  --data_path参数填写数据集路径，需写到数据集的一级目录。

   模型训练脚本参数说明如下。

   ```
   --data_root                         //数据集路径
   --batch_size                        //默认4，训练批次大小，提高会降低ap值
   --max_epochs                        //默认1，训练次数
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

|  NAME  | cards | FPS | Epochs | optim | mmdetection_Version | Torch_Version |
|:------:|:--------:|:-----:|:-----:|:-----:|:-----:|:-----:|
| 8p-竞品A | 8p | 197 | 12 | base | 2.9.0 | 1.11 |
| 8p-Atlas 800T A2 | 8p | 157 | 12 | base | 3.3.0 | 2.1 |
| 8p-Atlas 800T A2 | 8p | 214 | 12 | amp | 3.3.0 | 2.1 |

**表 3** 8p-竞品A (12 epochs, optim=base, mmdetection=2.9.0) 训练精度数据

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.354
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.551
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.376
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.206
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.389
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.452
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.527
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.527
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.527
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.341
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.575
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.672
```

**表 4** 8p-Atlas 800T A2 (12 epochs, optim=base, mmdetection=3.3.0) 训练精度数据

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.366
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.560
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.388
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.206
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.406
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.472
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.538
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.538
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.538
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.344
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.584
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.689
```

# 版本说明

## 变更

2024.11.8: 首次提交。

## FAQ

无。

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md
