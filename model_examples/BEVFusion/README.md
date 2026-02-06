# BEVFusion

# 概述

`BEVFusion`是一个高效且通用的多任务多传感器融合框架，它在共享的鸟瞰图（BEV）表示空间中统一了多模态特征，这很好地保留了几何和语义信息，从而更好地支持 3D 感知任务。

- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmdetection3d/tree/main/projects/BEVFusion
  commit_id=0f9dfa97a35ef87e16b700742d3c358d0ad15452
  ```

# 支持模型

| Modality  | Voxel type (voxel size) | 训练方式 |
|-----------|-------------------------|------|
| lidar-cam | lidar-cam               | FP32、FP16 |

# 训练环境准备
## 昇腾环境安装
请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境。本仓已支持表1中软件版本。
  
  **表 1**  昇腾软件版本支持表

  |        软件类型        |   首次支持版本   |
  |:------------------:|:--------:|
  | FrameworkPTAdapter | 7.1.0  |
  |       CANN         | 8.2.rc1  |


## 模型环境安装
- 当前模型支持的`PyTorch`版本如下表所示。

  **表 2**  版本支持表

  | Torch_Version |
  |:-------------:|
  |  PyTorch 2.1、PyTorch 2.7  |

- 下载并编译安装`DrivingSDK`加速库，参考https://gitcode.com/Ascend/DrivingSDK

- 安装依赖。

  进入`BEVFusion`模型代码目录：

  ```
  cd DrivingSDK/model_examples/BEVFusion
  ```
  - 推荐使用依赖安装一键配置脚本，可使用如下指令安装后续`mmcv`和`mmdetection3d`：

  ```shell
  bash install_BEVFusion.sh
  ```

  1. 源码编译安装`mmcv`

  ```
  git clone -b main https://github.com/open-mmlab/mmcv.git
  cd mmcv
  pip install -r requirements/runtime.txt
  pip install ninja
  pip install "setuptools<=78.1.1"
  MMCV_WITH_OPS=1 MAX_JOBS=8 FORCE_NPU=1 python setup.py build_ext
  MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py develop
  cd ../
  ```

  2. 源码安装`mmdetection3d v1.2.0`版本

  ```
  git clone -b v1.2.0 https://github.com/open-mmlab/mmdetection3d.git
  cp -f bevfusion.patch mmdetection3d/
  cd mmdetection3d
  git apply bevfusion.patch --reject
  pip install mmengine==0.10.7 mmdet==3.1.0 numpy==1.23.5 yapf
  pip install -e . --no-build-isolation
  cd ../
  ```

## 数据准备
```
cd mmdetection3d/
```
1. 在`mmdetection3d`的`data`文件夹下新建`nuscenes`文件夹，`data`文件结构如下：
    ```
    data
    ├── lyft
    ├── nuscenes
    ├── s3dis
    ├── scannet
    └── sunrgbd
    ```
    请自行下载 [nuScenes 数据集](https://www.nuscenes.org/nuscenes#download) 或构建软连接到`nuscenes`文件夹下，模型运行的必要数据结构如下：
    ```
    nuscenes/
    ├── maps
    ├── samples
    ├── sweeps
    ├── v1.0-test
    ├── v1.0-trainval
    ```
2. 在`mmdetection3d`目录下进行数据预处理，处理方法参考原始`github`仓库：

   ```
   python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
   ```
    预处理后`nuscenes`文件结构如下：
    ```
    nuscenes/
    ├── maps
    ├── nuscenes_gt_database
    ├── nuscenes_infos_test.pkl
    ├── nuscenes_infos_train.pkl
    ├── nuscenes_infos_val.pkl
    ├── samples
    ├── sweeps
    ├── v1.0-test
    ├── v1.0-trainval

    ```
3. 下载预训练权重：在`mmdetection3d`目录下创建`pretrained`文件夹，参考 [BEVFusion Model](https://github.com/open-mmlab/mmdetection3d/tree/main/projects/BEVFusion)，下载预训练权重 [Swin pre-trained model](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/swint-nuimages-pretrained.pth) 和 [lidar-only pre-trained detector](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth)。将预训练权重放在`pretrained`文件夹中，目录样例如下：

    ```
    pretrained/
    ├── swint-nuimages-pretrained.pth
    ├── bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth
    ```

# 模型运行

数据预处理及预训练权重准备好后，回到`BEVFusion`模型目录：

```
cd ../
```

- 单机8卡训练
  
  运行脚本支持命令行参数（支持默认值+关键字参数+位置参数）
  - `--batch-size`：每卡batch size大小，默认值4；
  - `--num-npu`：每节点NPU卡数，默认值8；

  ```shell
  # 精度测试拉起脚本，默认训练6个epochs
  # FP32
  bash test/train_full_8p_base_fp32.sh --batch-size=4 --num-npu=8 # batch-size 和 num-npu 可不指定直接使用默认值，下同
  # FP16
  bash test/train_full_8p_base_fp16.sh --batch-size=4 --num-npu=8

  # 性能测试拉起脚本，默认训练1个epochs
  # FP32
  bash test/train_performance_8p_base_fp32.sh --batch-size=4 --num-npu=8 
  # FP16
  bash test/train_performance_8p_base_fp16.sh --batch-size=4 --num-npu=8
  ```
- 双机16卡性能（FP32）
  
  运行脚本支持命令行参数（支持默认值+关键字参数+位置参数）
  - `--batch-size`：每卡batch size大小，默认值4；
  - `--num-npu`：每节点NPU卡数，默认值8；
  - `--nnodes`：节点总数，默认值2；
  - `--node-rank`：当前节点编号（0 ~ nnodes-1），默认为主节点0；
  - `--port`：通信端口号，默认值29500；
  - `--master-addr`：主节点IP地址；
  ```shell
  # 主节点拉起脚本，默认训练1个epochs
  bash test/nnodes_train_performance_16p_base_fp32.sh --batch-size=4 --num-npu=8 --nnodes=2 --node-rank=0 --port=port --master-addr=master_addr # master-addr 必须指定，其余可省略以使用默认值
  # 副节点拉起脚本，默认训练1个epochs
  bash test/nnodes_train_performance_16p_base_fp32.sh --batch-size=4 --num-npu=8 --nnodes=2 --node-rank=1 --port=port --master-addr=master_addr # node-rank，master-addr 必须指定，其余可省略以使用默认值
  ```

# 训练结果
单机8卡
| NAME             | Modality  | Voxel type (voxel size) | 训练方式 | Epoch | global batch size | NDS   | mAP   | FPS   |
|------------------|-----------|-------------------------|------|-------|-------|-------|-------|-------|
| 8p-Atlas 800T A2 | lidar-cam | 0.075                   | FP32 | 6     | 32 | 69.98 | 67.36 | 26.46 |
| 8p-竞品A           | lidar-cam | 0.075                   | FP32 | 6     | 32 | 69.78 | 67.36 | 22.54 |
| 8p-Atlas 800T A2 | lidar-cam | 0.075                   | FP16 | 6     | 32 | 70.11 | 68.01 | 32.78 |
| 8p-竞品A           | lidar-cam | 0.075                   | FP16 | 6     | 32 | 68.50 | 64.89 | 26.59 |

双机16卡
| NAME             | Modality  | Voxel type (voxel size) | 训练方式 | Epoch | global batch size |FPS   | 线性度 |
|------------------|-----------|-------------------------|------|-------|-------|-------|-------|
| 8p-Atlas 800T A2 | lidar-cam | 0.075 | FP32 | 1     | 64 | 45.86 | 97.07%  |

# 版本说明

## 变更
2026.1.31：支持混精训练，更新模型性能。

2026.1.4：稀疏卷积类算子优化并加入一键Patch，更新模型性能，简化bevfusion.patch。

2025.8.29：模型优化，更新单机性能。

2025.8.1：模型性能优化，更新单机性能及精度。

2025.7.10：更新单机性能及精度。

2025.5.20：支持双机，更新单机及双机性能。

2024.12.5：首次发布。

## FAQ
1. `RuntimeError: The server socket has failed to listen on any local network address. The server socket has failed to bind to [::]:29500 (errno: 98 - Address already in use).`
   
   表示默认端口已被占用，自行修改`mmdetection3d`源码文件`tools/dist_train.sh`下的`PORT`默认值。

2. `FileNotFoundError: pretrained/swint-nuimages-pretrained.pth can not be found.`
   
   数据文件缺失，请对照[数据准备](##数据准备)检查数据是否完整。

3. `AssertionError: MMCV==2.2.0 is used but incompatible. Please install mmcv>=2.0.0rc4, <2.1.0.`
   
   MMCV版本冲突，修改`MMCV`源码文件`mmcv/version.py`中的`__version__ = '2.0.1'`

4. `Environment variable [HCCL_IF_IP] is invalid. Reason: it should be "ip[%ifname]".`
   
   将环境变量设置脚本`test/env_npu.sh`中的`export HCCL_IF_IP=...`注释即可。

5. 当前训练脚本采用`lidar-only`预训练权重，若需要基于`lidar-cam`预训练权重进行训练，仅需将脚本中的`bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth`更改为对应权重文件即可。

6. `RuntimeError: ACL stream synchronize failed, error code:507018`
​
大概率是有残余进程或者其他程序在模型预处理数据集时占用，全局清理一下进程并重跑即可。