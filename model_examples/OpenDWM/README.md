# OpenDWM for PyTorch

## 目录

- [OpenDWM for PyTorch](#OpenDWM-for-pytorch)
  - [目录](#目录)
- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [支持任务列表](#支持任务列表)
  - [代码实现](#代码实现)
- [OpenDWM](#OpenDWM)
  - [准备训练环境](#准备训练环境)
    - [安装昇腾环境](#安装昇腾环境)
    - [准备数据集](#准备数据集)
    - [准备base_model](#准备base_model)
    - [准备预训练权重](#准备预训练权重)
  - [快速开始](#快速开始)
    - [训练任务](#训练任务)
      - [开始训练](#开始训练)
      - [训练结果](#训练结果)
    - [推理任务](#推理任务)
      - [开始推理](#开始推理)
      - [推理结果](#推理结果)
- [变更说明](#变更说明)
- [FAQ](#faq)

# 简介

## 模型介绍

OpenDWM是一种统一的多视角驾驶视频生成框架。通过融合单/多视角数据，结合DiT扩散模型与跨帧跨视图模块，分三阶段训练，提升生成视频的多样性和质量。创新的显式视角建模有效增强运动一致性，支持文本、图像等多类型输入，生成高质量、长时程、环绕视图一致的驾驶场景视频，在FID和FVD指标上显著优于现有模型。

## 支持任务列表

本仓已经支持以下模型任务类型

|    模型     | 任务列表 | 是否支持 |
| :---------: | :------: | :------: |
| OpenDWM |   训练   |    ✔     |
| OpenDWM |   推理   |    ✔     |

## 代码实现

- 参考实现：
  
  ```
  url=https://github.com/SenseTime-FVG/OpenDWM
  commit_id=b0ecc3d4020612376ea5a87500f98bc76893428f 
  ```

- 适配昇腾 AI 处理器的实现：
    ```
    url=https://gitee.com/ascend/DrivingSDK.git
    code_path=model_examples/OpenDWM
    ```

# OpenDWM

## 准备训练环境

### 安装环境

**表 1**  三方库版本支持表

| 三方库  | 支持版本 |
| :-----: | :------: |
| PyTorch |   2.5.1   |
| PyTorch |   2.6.0   |

### 安装昇腾环境

请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表2中软件版本。

**表 2**  昇腾软件版本支持表

|     软件类型      | 支持版本 |
| :---------------: | :------: |
| FrameworkPTAdapter | 7.1.0  |
|       CANN        | 8.2.RC1  |
|       Python        | 3.9  |

1. 激活 CANN 环境
    将 CANN 包目录记作 cann_root_dir，执行以下命令以激活环境
    ```
    source {cann_root_dir}/set_env.sh
    ```

2. 安装Driving SDK

    请参考昇腾[Driving SDK](https://gitee.com/ascend/DrivingSDK)代码仓说明编译安装Driving SDK

3. 安装MindSpeed
    
    源码安装：
    ```
    git clone https://gitee.com/ascend/MindSpeed.git
    pip install -e MindSpeed
    ```

4. 克隆代码仓到当前目录：

    ```
    git clone https://gitee.com/ascend/DrivingSDK.git -b master
    cd DrivingSDK/model_examples/OpenDWM
    git clone https://github.com/SenseTime-FVG/OpenDWM 
    cd OpenDWM
    git checkout b0ecc3d4020612376ea5a87500f98bc76893428f
    ```

    将模型根目录记作 `model-root-path`



5. 安装模型相关的依赖项。
    根据已安装的 PyTorch 版本，选择对应版本的 torchvision 进行安装：
    * 若 PyTorch 版本为 2.5.1：
    ```
    python -m pip install torchvision==0.20.1
    ```
    
    * 若 PyTorch 版本为 2.6.0：
    ```
    python -m pip install torchvision==0.21.0
    ```
    
    请确保 torch 与 torchvision 版本兼容。可通过 `python -c "import torch; print(torch.__version__)"` 查看当前 PyTorch 版本。
  
    ```
    # 安装其他依赖项
    python -m pip install -r requirements.txt

    # 安装tbe和hccl, 将 CANN 包目录记作 cann_root_dir
    pip uninstall te topi hccl -y
    pip install {cann_root_dir}/latest/lib64/te-*-py3-none-any.whl
    pip install {cann_root_dir}/latest/lib64/hccl-*-py3-none-any.whl
    ```
    
6. 使用 patch 文件：
    ```
    cp -f ../OpenDWM.patch .
    git apply --reject --whitespace=fix OpenDWM.patch
    cp -rf ../test .
    cp -rf ../tools/patch.py ./src/dwm/tools/
    ```

### 准备数据集

- 根据原仓**Train**章节准备数据集

  1. 下载[nuScenes数据集](https://www.nuscenes.org/download)到${model-root-path}/data/nuscenes，目录结构如下

      ```bash
      ${model-root-path}/data/nuscenes
      ├── interp_12Hz_trainval
      ├── v1.0-trainval01_blobs.tgz
      ├── v1.0-trainval02_blobs.tgz
      ├── v1.0-trainval03_blobs.tgz
      ├── v1.0-trainval04_blobs.tgz
      ├── v1.0-trainval05_blobs.tgz
      ├── v1.0-trainval06_blobs.tgz
      ├── v1.0-trainval07_blobs.tgz
      ├── v1.0-trainval08_blobs.tgz
      ├── v1.0-trainval09_blobs.tgz
      ├── v1.0-trainval10_blobs.tgz
      └── v1.0-trainval_meta.tgz
      ```

  2. 在model-root-path下执行如下命令处理数据集

      ```python
      python src/dwm/tools/tar2zip.py -i data/nuscenes/v1.0-trainval_meta.tgz -o data/nuscenes/v1.0-trainval_meta.zip
      python src/dwm/tools/tar2zip.py -i data/nuscenes/v1.0-trainval01_blobs.tgz -o data/nuscenes/v1.0-trainval01_blobs.zip
      python src/dwm/tools/tar2zip.py -i data/nuscenes/v1.0-trainval02_blobs.tgz -o data/nuscenes/v1.0-trainval02_blobs.zip
      ...
      python src/dwm/tools/tar2zip.py -i data/nuscenes/v1.0-trainval10_blobs.tgz -o data/nuscenes/v1.0-trainval10_blobs.zip
      ```

  3. 下载对应的[captions文件](https://huggingface.co/datasets/wzhgba/opendwm-data/resolve/main/nuscenes_v1.0-trainval_caption_v2.zip?download=true)

- 数据集目录及结构最终按照如下格式：

```bash
${model-root-path}/data/nuscenes
├── interp_12Hz_trainval.zip
├── nuScenes-map-expansion-v1.3.zip
├── nuscenes_v1.0-trainval_caption_v2_times_train.json
├── nuscenes_v1.0-trainval_caption_v2_times_val.json
├── nuscenes_v1.0-trainval_caption_v2_train.json
├── nuscenes_v1.0-trainval_caption_v2_val.json
├── v1.0-trainval01_blobs.zip
├── v1.0-trainval02_blobs.zip
├── v1.0-trainval03_blobs.zip
├── v1.0-trainval04_blobs.zip
├── v1.0-trainval05_blobs.zip
├── v1.0-trainval06_blobs.zip
├── v1.0-trainval07_blobs.zip
├── v1.0-trainval08_blobs.zip
├── v1.0-trainval09_blobs.zip
├── v1.0-trainval10_blobs.zip
└── v1.0-trainval_meta.zip
```

### 准备base_model

- 根据原仓**Models**章节准备SD3.5的[模型权重](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)，目录及结构如下：

```bash
${model-root-path}/base_model/
└── stable-diffusion-3.5-medium
```

### 准备预训练权重

- 推理需要[预训练权重](https://huggingface.co/wzhgba/opendwm-models/resolve/main/ctsd_35_tirda_nwao_20k.pth?download=true)，目录如下：
```bash
${model-root-path}/pretrained/
└── ctsd_35_tirda_nwao_20k.pth
```

## 快速开始

### 训练任务

本任务目前主要提供单机的8卡训练单数据集

#### 开始训练

1. 在模型根目录下，运行训练脚本。

   - 单机8卡精度训练
   
   ```
   # 单机8卡训练
   bash test/train.sh
   ```

   - 单机8卡的性能训练
   ```
   # 单机8卡训练
   bash test/train_performance.sh
   ```

#### 训练结果

| 芯片          | 卡数 | device_mesh | Precision | Loss | 性能-单步迭代耗时(s) |
| ------------- | :--: | :---------------: | :-------------------: | :-------------------: |:-------------------: |
| 竞品A           |  8p  |       [2, 4]         |   混精    | 0.1373 | 1.1|
| Atlas 800T A2 |  8p  |        [2, 4]         |   混精    | 0.1367 | 1.1|

### 推理任务
本任务目前主要提供单机单卡的推理

#### 开始推理
1. 在模型根目录下，运行推理指令。

   - 单卡推理

   ```
   PYTHONPATH=src python examples/ctsd_generation_example.py -c examples/ctsd_35_6views_image_generation.json -o output/ctsd_35_6views_image_generation
   ```

#### 推理结果
| 芯片          | 卡数 | 性能-单步迭代耗时(s) |
| ------------- | :--: |:-------------------: |
| 竞品A           |  1p  |11.2805|
| Atlas 800T A2 |  1p  |11.0295|

# 变更说明

2025.08.06：首次发布


# FAQ

1. 镜像中可能由于不支持awk的扩展正则表达式导致出现`syntax error at or near`，需要在镜像中安装gawk解决
```
# Debian/Ubuntu
apt-get update && apt-get install -y gawk

# CentOS/OpenEuler
yum install -y gawk
```

2. 训练过程会自动下载inception权重，如果遇到网络问题等下载失败，可以本地下载后，手动将该权重文件放到日志指定路径

3. 我们支持的训练方式是单数据集nuScenes，若在执行`python -m pip install -r requirements.txt` 时由于网络原因下载kitti仓失败，注释掉对应依赖即可