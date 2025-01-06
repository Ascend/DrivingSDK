# MapTR for PyTorch

## 目录

- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [支持任务列表](#支持任务列表)
  - [代码实现](#代码实现)
- [MapTR](#MapTR)
  - [准备训练环境](#准备训练环境)
  - [快速开始](#快速开始)
    - [训练任务](#训练任务)
- [公网地址说明](#公网地址说明)
- [变更说明](#变更说明)
- [FAQ](#FAQ)

# 简介

## 模型介绍

MapTR是一种高效的端到端Transformer模型，用于在线构建矢量化高清地图（HD Map）。高清地图在自动驾驶系统中是规划的基础和关键组件，提供了丰富而精确的环境信息。MapTR提出了一种统一的置换等价建模方法，将地图元素表示为等价置换组的点集，这样不仅可以准确描述地图元素的形状，还能稳定学习过程。此外，MapTR设计了一个分层查询嵌入方案，以灵活地编码结构化地图信息，并执行分层二分匹配来学习地图元素。

## 支持任务列表

本仓已经支持以下模型任务类型

|    模型     | 任务列表 | 是否支持 |
| :---------: | :------: | :------: |
| MapTR |   训练   |    ✔     |

## 代码实现

- 参考实现：
  
  ```
  url=https://github.com/hustvl/MapTR
  commit_id=fa420a2e756c9e19b876bdf2f6d33a097d84be73
  ```

# MapTR

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
| FrameworkPTAdaper | 6.0.RC3  |
|       CANN        | 8.0.RC3  |
|    昇腾NPU固件    | 24.1.RC3 |
|    昇腾NPU驱动    | 24.1.RC3 |

1. 安装mmdet3d
  
  - 在模型根目录下，克隆mmdet3d仓，并进入mmdetection3d目录
    
    ```
    git clone -b v1.0.0rc4 https://github.com/open-mmlab/mmdetection3d.git
    cd mmdetection3d
    ```
  - 在mmdetection3d目录下，修改代码
    
    （1）删除requirements/runtime.txt中第3行 numba==0.53.0
    
    （2）修改mmdet3d/____init____.py中第22行 mmcv_maximum_version = '1.7.0'为mmcv_maximum_version = '1.7.2'
  - 安装包
    
    ```
    pip install -v -e .
    ```
2. 安装mmcv
  
  - 在模型根目录下，克隆mmcv仓，并进入mmcv目录安装
    
    ```
    git clone -b 1.x https://github.com/open-mmlab/mmcv
    cd mmcv
    MMCV_WITH_OPS=1 MAX_JOBS=8 FORCE_NPU=1 python setup.py build_ext
    MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py develop
    ```
3. 安装mxDriving加速库，具体方法参考[原仓](https://gitee.com/ascend/mxDriving)。
4. 在模型根目录下执行以下命令，安装模型对应PyTorch版本需要的依赖。
  
  ```
  pip install -r requirement.txt
  ```
5. 在当前python环境下执行`pip show pip`，得到三方包安装路径Location，记作location_path，在模型根目录下执行以下命令来替换patch。
  
  ```
  bash replace_patch.sh --packages_path=location_path
  ```

6. 模型代码更新
  ```
  git clone https://github.com/hustvl/MapTR.git
  cp MapTR.patch MapTR
  cd MapTR
  git checkout 1b435fd9f0db9a14bb2a9baafb565200cc7028a2
  git apply --reject --whitespace=fix MapTR.patch
  cd ../

### 准备数据集

- 根据原仓**Prepare Dataset**章节准备数据集，数据集目录及结构如下：

```
MapTR
├── ckpts/
│   ├── resnet50-19c8e357.pth
├── data/
│   ├── can_bus/
│   ├── nuscenes/
│   │   ├── lidarseg/
│   │   ├── maps/
│   │   ├── panoptic/
│   │   ├── samples/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
|   |   ├── nuscenes_infos_temporal_test_mono3d.coco.json
|   |   ├── nuscenes_infos_temporal_train_mono3d.coco.json
|   |   ├── nuscenes_infos_temporal_val_mono3d.coco.json
|   |   ├── nuscenes_map_anns_val.json
|   |   ├── nuscenes_infos_temporal_test.pkl
|   |   ├── nuscenes_infos_temporal_train.pkl
|   |   ├── nuscenes_infos_temporal_val.pkl
├── patch/
├── projects/
├── test/
├── tools/
```

> **说明：**
> nuscenes数据集下的文件，通过运行以下指令生成：
```
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0 --canbus ./data
```

### 准备预训练权重

- 在模型根目录下，执行以下指令下载预训练权重：
```
mkdir ckpts
cd ckpts 
wget https://download.pytorch.org/models/resnet50-19c8e357.pth
```

## 快速开始

### 训练任务

本任务主要提供单机的8卡训练脚本。

#### 开始训练

1. 在模型根目录下，运行训练脚本。
   
   该模型支持单机8卡训练。
   
   - 单机8卡精度训练
   
   ```
   bash test/train_8p.sh
   ```
   
   - 单机8卡性能训练
   
   ```
   bash test/train_8p_performance.sh
   ```

#### 训练结果

| 芯片          | 卡数 | global batch size | Precision | epoch |  mAP  | 性能-单步迭代耗时(ms) |
| ------------- | :--: | :---------------: | :-------: | :---: | :----: | :-------------------: |
| 竞品A           |  8p  |         32         |   fp32    |  24   | 48.7 |         710          |
| Atlas 800T A2 |  8p  |         32         |   fp32    |  24   | 48.5 |         1100          |


# 变更说明

2024.11.08：首次发布


# FAQ

无