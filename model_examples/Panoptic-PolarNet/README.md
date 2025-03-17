# Panoptic-PolarNet for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

PanopticPolarNet 是一种用于 3D 点云全景分割（Panoptic Segmentation）的深度学习模型。它结合了 全景分割（Panoptic Segmentation）和 点云处理（Point Cloud Processing）的技术，旨在同时对点云数据进行 实例分割（Instance Segmentation）和 语义分割（Semantic Segmentation）。PanopticPolarNet 的核心思想是将点云数据转换为极坐标表示，并利用深度学习模型进行处理。

- 参考实现：

  ```
  url=https://github.com/edwardzhou130/Panoptic-PolarNet/tree/main
  commit_id=3a72f2380a4e505e191b69da596f521a9d9f1a71
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/DrivingSDK.git
  code_path=model_examples/Panoptic-PolarNet
  ```


# 准备训练环境

## 准备环境

### 安装昇腾环境

请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表1中软件版本。

**表 1**  昇腾软件版本支持表

|     软件类型      | 支持版本 |
| :---------------: | :------: |
| FrameworkPTAdaper | 7.0.0  |
|       CANN        | 8.1.RC1  |

### 安装模型环境

**表 2**  三方库版本支持表

| Torch_Version      | 三方库依赖版本                                 |
| :--------: | :----------------------------------------------------------: |
| PyTorch 2.1 | torchvision==0.16.0 |

0. 激活 CANN 环境

  将 CANN 包目录记作 cann_root_dir，执行以下命令以激活环境
  ```
  source {cann_root_dir}/set_env.sh
  ```

3. 安装 mx_driving

  在 DrivingSDK 根目录下安装 mx_driving
  ```
  # 请先 cd 到 DrivingSDK 根目录
  pip install -r requirements.txt
  python setup.py develop --release
  ```

4. 准备模型源码，安装依赖项

  在 Panoptic-PolarNet 根目录下，克隆 Panoptic-PolarNet 仓，替换其中部分代码并安装依赖项

  ```
  git clone https://github.com/edwardzhou130/Panoptic-PolarNet/tree/main
  cd Panoptic-PolarNet/
  git checkout 3a72f2380a4e505e191b69da596f521a9d9f1a71
  cp -f ../Panoptic-PolarNet.patch ./
  git apply --reject Panoptic-PolarNet.patch
  cp -r ../test/ ./
  pip install -r requirements.txt
  pip install torchvision==0.16.0
  ```


## 准备数据集

1. 请用户自行准备好数据集，可选用 SemanticKITTI、NuScenes 数据集等。
2. 上传数据集到 data 文件夹，以 SemanticKITTI 为例，数据集在`data/`目录下。
3. 当前提供的训练脚本是以 SemanticKITTI 数据集为例，该数据集需要经过预处理方可用于训练。 数据集预处理前目录结构参考如下：

  ```
    ./
    ├── train.py
    ├── ...
    └── data/
        ├──sequences
            ├── 00/           
            │   ├── velodyne/	# Unzip from KITTI Odometry Benchmark Velodyne point clouds.
            |   |	├── 000000.bin
            |   |	├── 000001.bin
            |   |	└── ...
            │   └── labels/ 	# Unzip from SemanticKITTI label data.
            |       ├── 000000.label
            |       ├── 000001.label
            |       └── ...
            ├── ...
            └── 21/
            └── ...
  ```
  > **说明：** 
  >该数据集的训练过程脚本只作为一种参考示例。

4. 请在 Panoptic-PolarNet 模型根目录，使用如下命令预处理 SemanticKITTI 数据集。
  ```
  # data_root 请替换成 SemanticKITTI 数据集 sequences 目录所在路径
  data_root=/home/datasets/SemanticKITTI/data/sequences
  ln -nsf $data_root data/sequences
  python instance_preprocess.py
  ```
  处理后的文件夹应该如下。
  ```
    ./
    ├── train.py
    ├── ...
    └── data/
        ├──instance_path.pkl
        ├──sequences
            ├── 00/           
            │   ├── velodyne/	# Unzip from KITTI Odometry Benchmark Velodyne point clouds.
            |   |	├── 000000.bin
            |   |	├── 000001.bin
            |   |	└── ...
            │   ├── instance/
            |   |	├── 000000_1.bin
            |   |	├── 000000_2.bin
            |   |	└── ...
            │   └── labels/ 	# Unzip from SemanticKITTI label data.
            |       ├── 000000.label
            |       ├── 000001.label
            |       └── ...
            ├── ...
            └── 21/
            └── ...
  ```

# 开始训练

## 训练模型

1. 运行训练脚本。

   该模型仅支持单机单卡训练。

   - 单机单卡训练

     ```
     bash ./test/train.sh
     ```
   
  训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

|     芯片      | 卡数 | global batch size | epoch |     best PQ     |     FPS     |
| :-----------: | :--: | :---------------: | :---: | :------------: |--------------|
|     竞品A     |  1p  |         2         |  10   |     52.458     |      1.69   |
| Atlas 800T A2 |  1p  |         2         |  10   |     52.159     |      1.28    |

# 版本说明

## 变更

2025.03.05: 首次提交。

## FAQ

无。
