# FCOS3D for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

FCOS3D是一个全卷积、单阶段的三维目标检测模型，用于无任何2D检测或2D-3D对应先验的单目3D目标检测。在该框架中，首先将通常定义的7-DoF 3D目标转换到图像域，并将其解耦为2D和3D属性，以适应3D设置。在此基础上，考虑到目标的2D比例，将目标分配到不同的特征，并仅根据3D中心进一步分配。此外，中心度用基于3D中心的2D高斯分布重新定义，以与目标公式兼容。

- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmdetection3d
  commit_id=fe25f7a51d36e3702f961e198894580d83c4387b
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/DrivingSDK.git
  code_path=model_examples/FCOS3D
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

1. 安装 mmcv

  在 FCOS3D 根目录下，克隆 mmcv 仓，并进入 mmcv 目录安装

  ```
  git clone https://github.com/open-mmlab/mmcv
  cd mmcv
  MMCV_WITH_OPS=1 MAX_JOBS=8 FORCE_NPU=1 python setup.py build_ext
  MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py develop
  cd ../
  ```

2. 安装 mmdet
  
  ```
  git clone -b v3.3.0 https://github.com/open-mmlab/mmdetection.git
  cd mmdetection
  cp -f ../mmdet.patch ./
  git apply --reject --whitespace=fix mmdet.patch
  pip install -e .
  ```

3. 安装 mx_driving

  在 DrivingSDK 根目录下安装 mx_driving
  
  ```
  # 请先 cd 到 DrivingSDK 根目录
  pip install -r requirements.txt
  python setup.py develop --release
  ```

4. 准备模型源码，安装 mmdetection3d

  克隆 mmdetection3d 仓，替换其中部分代码并安装

  ```
  git clone https://github.com/open-mmlab/mmdetection3d.git
  cd mmdetection3d/
  git checkout fe25f7a51d36e3702f961e198894580d83c4387b
  cp -f ../mmdetection3d.patch ./
  git apply --reject mmdetection3d.patch
  cp -f ../dist_train_performance.sh tools/
  cp -f ../train_performance.py tools/
  pip install -e .
  cp -r ../test/ ./
  ```

5. 安装其他依赖
  
  在 mmdet 代码目录下，安装依赖

  ```
  pip install -r requirements.txt
  pip install torchvision==0.16.0
  ```

  在 mmdetection3d 代码目录下，安装依赖

  ```
  pip install -r requirements.txt
  ```


## 准备数据集

1. 请用户自行准备好数据集，包含训练集、验证集和标签三部分，可选用KITTI、NuScenes、Lyft、Waymo数据集等。
2. 上传数据集到data文件夹，以NuScenes为例，数据集在`data/nuscenes`目录下。
3. 当前提供的训练脚本是以NuScenes数据集为例，该数据集需要预处理成pkl格式数据方可用于训练。 数据集预处理前目录结构参考如下：

  ```
  mmdetection3d
  ├── mmdet3d
  ├── tools
  ├── configs
  ├── data
  │ ├── nuscenes
  │ │ ├── maps
  │ │ ├── samples
  │ │ ├── sweeps
  │ │ ├── v1.0-test
  | | ├── v1.0-trainval
  ```
  > **说明：** 
  >该数据集的训练过程脚本只作为一种参考示例。
4. 请在mmdetection3d根目录，使用如下命令预处理NuScenes数据集。
  ```
  # data_root 请替换成nuscenes数据集所在路径
  data_root=/home/datasets/nuscenes
  ln -nsf $data_root data/nuscenes
  python tools/create_data.py nuscenes --root-path ./data/nuscenes --version v1.0 --out-dir ./data/nuscenes --extra-tag nuscenes
  ```
  处理后的文件夹应该如下。
  ```
  mmdetection3d
  ├── mmdet3d
  ├── tools
  ├── configs
  ├── data
  │   ├── nuscenes
  │   │   ├── maps
  │   │   ├── samples
  │   │   ├── sweeps
  │   │   ├── v1.0-test
  |   |   ├── v1.0-trainval
  │   │   ├── nuscenes_gt_database
  │   │   ├── nuscenes_infos_train.pkl
  │   │   ├── nuscenes_infos_val.pkl
  │   │   ├── nuscenes_infos_test.pkl
  │   │   ├── nuscenes_dbinfos_train.pkl
  ```

# 开始训练

## 训练模型

1. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     ```
     bash ./test/train_1p.sh --data_root=/home/datasets/nuscenes --max_epochs=1
     ```
     
   - 单机8卡训练

     ```
     bash ./test/train_8p.sh --data_root=/home/datasets/nuscenes --max_epochs=1
     ```

   - 单机8卡训练性能

     ```
     bash ./test/train_8p_performance.sh --data_root=/home/datasets/nuscenes --max_epochs=1
     ```

  --data_path参数填写数据集路径，需写到数据集的一级目录。

  模型训练脚本参数说明如下。

  ```
  --data_root                         //数据集路径
  --batch_size                        //默认2，训练批次大小，提高会降低ap值
  --max_epochs                        //默认1，训练次数
  ```
   
  训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

|     芯片      | 卡数 | global batch size | epoch | mAP | NDS | FPS |
| :-----------: | :--: | :---------------: | :---: | :--------------------: | :--------------------: |--------------|
|     竞品A     |  8p  |         16        |  12   |         0.3049          |         0.3824          |       44.30      |
| Atlas 800T A2 |  8p  |         16         |  12   |         0.3007          |         0.3829          |       37.51    |

# 版本说明

## 变更

2025.01.20: 首次提交。
2025.02.05: 性能优化。
2025.02.18: 新增性能FPS计算，并添加到日志中。
2025.03.04: 新增性能测试脚本，大幅提升性能测试效率。
2025.03.25: 性能进一步优化。

## FAQ

无。
