# UniAD for PyTorch

## 目录

- [UniAD for PyTorch](#uniad-for-pytorch)
  - [目录](#目录)
- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [代码实现](#代码实现)
- [UniAD](#uniad)
  - [准备训练环境](#准备训练环境)
    - [安装环境](#安装环境)
    - [安装昇腾环境](#安装昇腾环境)
    - [准备数据集](#准备数据集)
    - [准备预训练权重](#准备预训练权重)
  - [快速开始](#快速开始)
    - [训练任务](#训练任务)
      - [开始训练](#开始训练)
      - [训练结果](#训练结果)
- [变更说明](#变更说明)
- [FAQ](#faq)

# 简介

## 模型介绍

自动驾驶是一项高度复杂的技术，需要多个学科领域的知识和技能，包括传感器技术、机器学习、路径规划等方面。自动驾驶还需要适应不同的道路规则和交通文化，与其他车辆和行人进行良好的交互，以实现高度可靠和安全的自动驾驶系统。面对这种复杂的场景，大部分自动驾驶相关的工作都聚焦在具体的某个模块，关于框架性的研讨则相对匮乏。自动驾驶通用算法框架——Unified Autonomous Driving（UniAD）首次将检测、跟踪、建图、轨迹预测，占据栅格预测以及规划整合到一个基于Transformer的端到端网络框架下， 完美契合了”多任务”和“高性能”的特点，是自动驾驶中的重大技术突破。


## 代码实现

- 参考实现：

  ```
  url=https://github.com/OpenDriveLab/UniAD
  commit_id=7b5bf15e0e49522b6553ddc48e67833e8f5f0f52
  ```
- 适配昇腾 AI 处理器的实现：

    ```
    url=https://gitee.com/ascend/DrivingSDK.git
    code_path=model_examples/UniAD
    ```
# UniAD

## 准备训练环境

### 安装环境

**表 1** 三方库版本支持表

| 三方库  | 支持版本 |
| ------- | -------- |
| PyTorch | 2.1.0    |

### 安装昇腾环境

请参考昇腾社区中《[Pytorch框架训练环境准备](https://gitee.com/link?target=https%3A%2F%2Fwww.hiascend.com%2Fdocument%2Fdetail%2Fzh%2FModelZoo%2Fpytorchframework%2Fptes)》文档搭建昇腾环境，本仓已支持表2中软件版本。

**表 2** 昇腾软件版本支持表

| 软件类型          | 支持版本     |
| ----------------- |----------|
| FrameworkPTAdaper | 7.0.0    |
| CANN              | 8.1.RC1    |


- 激活 CANN 环境
  将 CANN 包目录记作 cann_root_dir，执行以下命令以激活环境

  ```
  source {cann_root_dir}/set_env.sh
  ```

- 安装Driving SDK加速库，安装方法参考[原仓](https://gitee.com/ascend/DrivingSDK)，安装后手动source环境变量或将其配置在test/env_npu.sh中。


- 安装mmcv
  ```
  git clone -b 1.x https://github.com/open-mmlab/mmcv.git
  cd mmcv
  cp -f ../mmcv.patch ./
  git apply --reject --whitespace=fix mmcv.patch
  pip install -r requirements/runtime.txt
  MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py install
  ```

- 源码安装mmdet3d
   ```
  git clone -b v1.0.0rc6 https://github.com/open-mmlab/mmdetection3d.git
  cp -f ../mmdet3d.patch mmdetection3d
  cd mmdetection3d
  git apply --reject --whitespace=fix mmdet3d.patch
  pip install -r requirements/runtime.txt
  pip install -e .
   ```

- 准备模型源码
  ```
  git clone https://github.com/OpenDriveLab/UniAD.git
  cp -f UniAD.patch UniAD
  cp -r test UniAD
  cd UniAD
  git checkout 7b5bf15e0e49522b6553ddc48e67833e8f5f0f52
  git apply UniAD.patch
  pip install -r requirements.txt
  ```



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

   该模型支持双机多卡训练。
   
   ```
   # 'XX.XX.XX.XX'为主节点的IP地址；端口号可以换成未被占用的可用端口

   # stage1
   bash test/train_stage1_multi_server.sh 2 0 ‘xx.xx.xx.xx’ '3389' #主节点
   bash test/train_stage1_multi_server.sh 2 1 ‘xx.xx.xx.xx’ '3389' #副节点

   # stage2
   bash test/train_stage2_multi_server.sh 2 0 ‘xx.xx.xx.xx’ '3389' #主节点
   bash test/train_stage2_multi_server.sh 2 1 ‘xx.xx.xx.xx’ '3389' #副节点
   ```

#### 训练结果

单机八卡
| 阶段     | 芯片          | 卡数 | global batch size | Precision | 性能-单步迭代耗时(ms) | FPS |amota |   L2   |
|--------| ------------- | ---- |-------------------| --------- |---------------|--------|------|---------|
| stage1 | 竞品A           | 8p   | 8                 | fp32      | 5883        |   1.359 | 0.380 | -      |
| stage1 | Atlas 800T A2 | 8p   | 8                 | fp32      | 9883         |  0.809  | 0.376 | -      |
| stage2 | 竞品A           | 8p   | 8                 | fp32      | 3990        |  2.000  | -     | 0.9127 |
| stage2 | Atlas 800T A2 | 8p   | 8                 | fp32      | 7220         |   1.108  | -     | 0.9014 |


多机多卡线性度
| 阶段     | 芯片          | 卡数 | global batch size | Precision | 性能-单步迭代耗时(ms) | FPS | 线性度 |  
|--------| ------------- | ---- |-------------------| --------- |---------------|--------|------|
| stage1 | Atlas 800T A2*2 | 16p   | 16               | fp32      | 10075        |  1.588  | 96.30%     | 
| stage2 | Atlas 800T A2*2 | 16p  | 16                | fp32      | 7403         |   2.161  | 95.12%     |

# 变更说明

2025.02.19: 代码上仓，stage1性能0.6倍竞品，stage2性能0.5倍竞品。

2025.02.26: stage2性能优化，消除部分free time, FA替换优化, stage2性能达到0.54倍竞品。

2025.04.25: 增加多机多卡训练脚本，增加多机多卡训练性能数据。


# FAQ

无