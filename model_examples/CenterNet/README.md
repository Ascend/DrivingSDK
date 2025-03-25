# CenterNet for PyTorch

# 概述

## 简述

CenterNet使用关键点检测的方法去预测目标边框的中心点，然后回归出目标的其他属性，例如大小、3D位置、方向甚至是其姿态。而且这个方向相比之前的目标检测器，实现起来更加简单，推理速度更快，精度更高。

- 参考实现：

  ```
  url=https://github.com/xingyizhou/CenterNet.git 
  commit_id=4c50fd3a46bdf63dbf2082c5cbb3458d39579e6c
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/detection # ModelZoo-PyTorch
  ```
  ```
  url=https://gitee.com/ascend/DrivingSDK.git
  code_path=model_examples/CenterNet # Driving SDK
  ```

# 准备训练环境

## 安装昇腾环境
请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境。本仓已支持表1中软件版本。
  
  **表 1**  昇腾软件版本支持表

  |        软件类型        |   支持版本   |
  |:------------------:|:--------:|
  | FrameworkPTAdapter | 6.0.0  |
  |       CANN         | 8.0.0  |

## 安装模型环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 2**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 2.1 | torchvision==0.16.0 |
  | PyTorch 2.2 | torchvision==0.17.0 |
  | PyTorch 2.3 | torchvision==0.18.1 |
  | PyTorch 2.4 | torchvision==0.19.0 |
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。
  1. 首先下载Driving SDK仓，进入CenterNet模型代码目录：
  ```
  git clone https://gitee.com/ascend/DrivingSDK.git -b master
  cd DrivingSDK/model_examples/CenterNet
  ```

  2. 源码安装 CenterNet
  ```
  git clone https://github.com/xingyizhou/CenterNet.git
  cp -f CenterNet.patch CenterNet/
  cd CenterNet
  git apply CenterNet.patch --reject
  cp -f ../test ./
  ```

  3. 在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```
  pip install -r 2.1_requirements.txt  # PyTorch2.1版本

  pip install -r 2.2_requirements.txt  # PyTorch2.2版本

  pip install -r 2.3_requirements.txt  # PyTorch2.3版本

  pip install -r 2.4_requirements.txt  # PyTorch2.4版本
  ```
  > **说明：** 
  >只需执行一条对应的PyTorch版本依赖安装命令。

- 安装COCOAPI

  ```
  git clone https://github.com/cocodataset/cocoapi.git
  cd cocoapi/PythonAPI
  python setup.py install
  cd -
  ```

- 编译可变形卷积（来自DCNv2）

  ```
  cd ./src/lib/models/networks/DCNv2
  python setup.py build develop
  cd -
  ```

- 编译NMS

  ```
  cd ./src/lib/external
  python setup.py build_ext --inplace
  cd -
  ```

## 准备数据集

1. 获取数据集。

   用户可自行获取Pscal VOC数据集，将数据集上传到服务器任意路径下并解压；也可以通过下述脚本进行数据集的获取。

     - 运行脚本：

       ~~~
       cd ./src/tools/
       bash get_pascal_voc.sh
       ~~~

     - 上述脚本内容包含：

       - 从VOC网站下载、解压缩和移动Pascal VOC图像。
       - 下载COCO格式的Pascal VOC注释（从Detectron下载）。
       - 将train/val 2007/2012注释文件合并到单个json中。

   数据集目录结构参考如下所示。

   ```
   |-- data
   |-- |-- voc
       |-- |-- annotations
           |   |-- pascal_trainval0712.json
           |   |-- pascal_test2017.json
           |-- images
           |   |-- 000001.jpg
           |   ......
           |-- VOCdevkit        
   ```

   > **说明：** 
   > 该数据集的训练过程脚本只作为一种参考示例。 

备注：Vocdevkit需要用**faster rcnn**去运行评估脚本。

# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     启动单卡训练。

     ```
     bash ./test/train_full_1p.sh --data_path=数据集路径  # 单卡精度
     
     bash ./test/train_performance_1p.sh --data_path=数据集路径  # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=数据集路径  # 8卡精度
     
     bash ./test/train_performance_8p.sh --data_path=数据集路径  # 8卡性能
     ```

   - 单机8卡评测

     启动8卡评测。

     ```
     bash ./test/train_eval.sh --data_path=数据集路径  # 8卡精度评测
     ```

   - 多机多卡训练

     启动多机多卡训练。

     ```
     1. 安装环境
     2. 开始训练，每个机器请按下面提示进行配置
     bash ./test/train_performance_multinodes.sh --data_path=数据集路径 --batch_size=单卡batch_size --nnodes=机器总数量 --node_rank=当前机器rank(0,1,2..) --local_addr=当前机器IP(需要和master_addr处于同一网段) --master_addr=主节点IP
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data_path                         //数据集路径
   --num_workers                       //加载数据进程数
   --num_epochs                        //重复训练次数
   --batch_size                        //单卡训练批次大小
   --lr                           	    //初始学习率，默认：3.54e-4
   --device_list                       //训练指定训练用卡
   --world-size                        //分布式训练节点数量
   ```

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| NAME    | Acc@1 | FPS   | Epochs | AMP_Type | Torch_version |
| :-----: | :---: | :---: | :----: | :------: | :-----------: |
| 1p-竞品V | -    | 143  | 1     | -        | 1.5          |
| 8p-竞品V | 71.31 | 542   | 90     | -        | 1.5           |
| 1p-NPU-ARM  | -     | 164.08 | 1      | O1       | 1.8           |
| 8p-NPU-ARM  | 70.4 | 1257.444 | 90     | O1       | 1.8           |
| 1p-NPU-非ARM  |   | 169.69 | 1    | O1       | 1.8           |
| 8p-NPU-非ARM  |   | 1409.823 | 90     | O1       | 1.8           |

# 版本说明

## 变更

2024.12.11: 迁移模型至Driving SDK仓，优化DCNv2和NMS编译方式。

2023.02.14：更新readme，重新发布。

2021.10.09：首次发布。

## FAQ

1. 若出现无法找到datasets包的问题，本模型使用的是lib目录下的本地文件，请删除环境中同名三方库。
