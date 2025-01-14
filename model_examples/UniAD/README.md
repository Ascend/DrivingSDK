# UniAD for PyTorch

## 目录

-   [简介](#简介)
    - [模型介绍](#模型介绍)
    - [支持任务说明](#支持任务说明)
    - [代码实现](#代码实现)
-   [UniAD](#UniAD)
    - [准备训练环境](#准备训练环境)
    - [快速开始](#快速开始)
       - [训练任务](#训练任务) 
-   [公网地址说明](#公网地址说明)
-   [变更说明](#变更说明)
-   [FAQ](#FAQ)

# 简介

## 模型介绍

自动驾驶是一项高度复杂的技术，需要多个学科领域的知识和技能，包括传感器技术、机器学习、路径规划等方面。自动驾驶还需要适应不同的道路规则和交通文化，与其他车辆和行人进行良好的交互，以实现高度可靠和安全的自动驾驶系统。面对这种复杂的场景，大部分自动驾驶相关的工作都聚焦在具体的某个模块，关于框架性的研讨则相对匮乏。自动驾驶通用算法框架——Unified Autonomous Driving（UniAD）首次将检测、跟踪、建图、轨迹预测，占据栅格预测以及规划整合到一个基于Transformer的端到端网络框架下， 完美契合了”多任务”和“高性能”的特点，是自动驾驶中的重大技术突破。

## 支持任务说明
暂不支持，版本在研——当前存在精度问题，不推荐使用。

## 代码实现

- 参考实现：

  ```
  url=https://github.com/OpenDriveLab/UniAD
  commit_id=5927ba1ec10b71a45a57654dbb69139aeb893f50
  ```
- 适配昇腾 AI 处理器的实现：

    ```
    url=https://gitee.com/ascend/ModelZoo-PyTorch.git
    code_path=PyTorch/built-in/autonoumous_driving 
    ```
# UniAD

## 准备训练环境

### 安装环境

**表 1** 三方库版本支持表

| 三方库  | 支持版本 |
| ------- | -------- |
| PyTorch | 1.11     |

### 安装昇腾环境

请参考昇腾社区中《[Pytorch框架训练环境准备](https://gitee.com/link?target=https%3A%2F%2Fwww.hiascend.com%2Fdocument%2Fdetail%2Fzh%2FModelZoo%2Fpytorchframework%2Fptes)》文档搭建昇腾环境，本仓已支持表2中软件版本。

**表 2** 昇腾软件版本支持表

| 软件类型          | 支持版本     |
| ----------------- |----------|
| FrameworkPTAdaper | 在研       |
| CANN              | 在研  |
| 昇腾NPU固件       | 在研 |
| 昇腾NPU驱动       | 在研 |

- 源码安装mmcv

  - 在模型根目录下，克隆mmcv仓

    ```
    git clone -b 1.x https://github.com/open-mmlab/mmcv.git
    ```

  - 替换mmcv中的文件

    ```
    cp -f mmcv_need/modulated_deform_conv.py mmcv/mmcv/ops/modulated_deform_conv.py
    cp -f mmcv_need/multi_scale_deform_attn.py mmcv/mmcv/ops/multi_scale_deform_attn.py
    cp -f mmcv_need/optimizer.py mmcv/mmcv/runner/hooks/optimizer.py
    cp -f mmcv_need/epoch_based_runner.py mmcv/mmcv/runner/epoch_based_runner.py
    cp -f mmcv_need/runtime.txt mmcv/requirements/runtime.txt
    ```

  - 进入mmcv目录，安装依赖，执行编译安装命令

    ```
    cd mmcv
    pip install -r requirements/runtime.txt
    MMCV_WITH_OPS=1 MAX_JOBS=8 FORCE_NPU=1 python setup.py build_ext
    MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py develop
    ```

- 源码安装mmdet3d

  - 在模型根目录下，克隆mmdet3d仓

    ```
    git clone -b v1.0.0rc6 https://github.com/open-mmlab/mmdetection3d.git
    ```

  - 替换mmdetection3d中的文件

    ```
    cp -f mmdet3d_need/__init__.py mmdetection3d/mmdet3d/__init__.py
    cp -f mmdet3d_need/custom_3d.py mmdetection3d/mmdet3d/datasets/custom_3d.py
    cp -f mmdet3d_need/runtime.txt mmdetection3d/requirements/runtime.txt
    ```

  - 进入mmdetection3d目录，执行安装命令

    ```
    cd mmdetection3d
    pip install -v -e .
    ```

- 在模型根目录下执行以下命令，安装模型对应PyTorch版本需要的依赖。

  ```
  pip install -r requirements.txt
  ```

- 安装Driving SDK加速库，安装方法参考[原仓](https://gitee.com/ascend/DrivingSDK)，安装后手动source环境变量或将其配置在test/env_npu.sh中。
  【注意】当前版本配套Driving SDK RC3及以上版本，历史Driving SDK版本需要model仓代码回退到git reset --hard 91ac141ecfe5872f4835eef6aa4662f46ede80c3

### 准备数据集

- 根据原仓**Prepare Dataset**章节准备数据集，数据集目录及结构如下：

```
UniAD
├── data/
│   ├── nuscenes/
│   │   ├── can_bus/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
│   │   ├── v1.0-trainval/
│   ├── infos/
│   │   ├── nuscenes_infos_temporal_train.pkl
│   │   ├── nuscenes_infos_temporal_val.pkl
│   ├── others/
│   │   ├── motion_anchor_infos_mode6.pkl
```

> **说明：**
> 该数据集的训练过程脚本只作为一种参考示例。

### 准备预训练权重

- 根据原仓Installation章节下载预训练权重bevformer_r101_dcn_24ep.pth和uniad_base_track_map.pth，并放在模型根目录ckpts下：

```
UniAD
├── ckpts/
│   ├── bevformer_r101_dcn_24ep.pth
│   ├── uniad_base_track_map.pth
```

- （可选）可通过修改config文件中的load_from值来更改预训练权重
```
# projects/configs/stage1_track_map.py
load_from = "ckpts/bevformer_r101_dcn_24ep.pth"
```

## 快速开始

### 训练任务

本任务主要提供**单机**的**8卡**训练脚本。

#### 开始训练

1. 在模型根目录下，运行训练脚本。

   该模型支持单机8卡训练。

   - 单机8卡性能训练

   ```
   bash test/train_stage1_performance_8p.sh # stage1
   bash test/train_stage2_performance_8p.sh # stage2
   ```

   - 单机8卡精度训练

   ```
   bash test/train_stage1_full_8p.sh # stage1
   bash test/train_stage2_full_8p.sh # stage2
   ```

#### 训练结果

| 阶段     | 芯片          | 卡数 | global batch size | Precision | 性能-单步迭代耗时(ms) |
|--------| ------------- | ---- |-------------------| --------- |---------------|
| stage1 | 竞品A           | 8p   | 1                 | fp32      | 5883          |
| stage1 | Atlas 800T A2 | 8p   | 1                 | fp32      | 17654         |
| stage2 | 竞品A           | 8p   | 1                 | fp32      | 3990          |
| stage2 | Atlas 800T A2 | 8p   | 1                 | fp32      | 11374         |

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md

# 变更说明

2024.05.30：代码上仓，版本在研，暂不支持。

# FAQ

无