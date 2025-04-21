# OpenPCDet for Pytorch

## 概述

`OpenPCDet` 是一个基于 LiDAR 的 3D 物体检测套件，包含PointPillar、PointRCNN、CenterPoint等多种自动驾驶模型。本仓库对 `OpenPCDet` 中的部分模型进行了NPU设备的适配。

- 参考实现：
```
https://github.com/open-mmlab/OpenPCDet.git
commit_id=255db8f02a8bd07211d2c91f54602d63c4c93356
```

- 适配昇腾AI处理器的实现：
```
url=https://gitee.com/ascend/DrivingSDK.git
code_path=model_examples/CenterPoint
```


## 模型适配情况
| 支持模型 | 支持数据集 |
| - | - |
| Centerpoint | Nuscenes |
| Centerpoint3d | Nuscenes |

## Centerpoint
### 准备环境
#### 安装昇腾环境
请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境。本仓已支持表1中软件版本。
  
  **表 1**  昇腾软件版本支持表

  |        软件类型        |   支持版本   |
  |:------------------:|:--------:|
  | FrameworkPTAdapter | 6.0.0  |
  |       CANN         | 8.0.0  |

### 安装模型环境
**表 2** 版本支持表

| Torch_Version | 三方库依赖版本            |
| - |--------------------|
| PyTorch 2.1.0 | torchvision 0.16.0 |

#### 0. 克隆代码仓到当前目录并使用patch文件

```
    git clone https://gitee.com/ascend/DrivingSDK.git
    cd DrivingSDK/model_examples/CenterPoint
    git clone https://github.com/open-mmlab/OpenPCDet.git
    cd OpenPCDet
    git checkout 255db8f02a8bd07211d2c91f54602d63c4c93356
    cp -f ../OpenPCDet_npu.patch .
    git apply --reject OpenPCDet_npu.patch
    cp -rf ../test tools/
```

#### 1. 基本环境准备

在应用过patch的模型源码包所在目录下执行相应命令，安装模型需要的依赖
```shell
pip install -r requirements.txt && cd ../   # PyTorch 2.1版本
```

#### 2. 手动编译安装cumm和spconv
在开始安装前，确保系统已安装GCC 7.5.0

#### 2.1 前置依赖安装
执行以下命令，安装前置依赖pccm==0.3.4，ccimport==0.3.7

```shell
pip install pccm==0.3.4
pip install ccimport==0.3.7
```

#### 2.2 编译安装cumm
执行以下命令编译安装cumm

```shell
git clone https://github.com/FindDefinition/cumm.git -b v0.2.9
export CUMM_CUDA_VERSION=""
export CUMM_DISABLE_JIT="1"
cd ./cumm/
python setup.py bdist_wheel
cd ../ && pip install cumm/dist/cumm-*.whl
```

【注意】安装完毕后建议运行以下命令，如无报错，证明安装无误，可继续安装流程
```shell
python -c "import cumm"
```

#### 2.3 编译安装spconv
1. 执行以下命令，拉取spconv源码用于本地编译

    ```shell
    git clone https://github.com/traveller59/spconv.git -b v2.1.25
    ```

2. 执行以下命令，删除冗余文件
    ```shell
    rm -rf spconv/spconv/core_cc/csrc/sparse/all/ops1d.pyi
    rm -rf spconv/spconv/core_cc/csrc/sparse/all/ops2d.pyi
    rm -rf spconv/spconv/core_cc/csrc/sparse/all/ops3d.pyi
    rm -rf spconv/spconv/core_cc/csrc/sparse/all/ops4d.pyi
    rm -rf spconv/spconv/core_cc/cumm/tools/
    rm -rf spconv/pyproject.toml
    ```

3. 执行以下命令，替换spconv三方库中的文件内容
    ```shell
    /bin/cp -rf OpenPCDet/third_party_patches/spconv_patches/spconv/core_cc/csrc/sparse/all/__init__.pyi spconv/spconv/core_cc/csrc/sparse/all/__init__.pyi
    ```

4. 将spconv/spconv/pytorch/ops.py文件第32行代码进行调整
    ```python
    # 将代码 if hasattr(_ext, "cumm"):
    # 调整为 if 0:
    ```

5. 注释spconv/spconv/utils/\_\_init\_\_.py文件第26-30行代码
    ```python
    if not CPU_ONLY_BUILD:
        from spconv.core_cc.csrc.sparse.all.ops1d import Point2Voxel as Point2VoxelGPU1d
        from spconv.core_cc.csrc.sparse.all.ops2d import Point2Voxel as Point2VoxelGPU2d
        from spconv.core_cc.csrc.sparse.all.ops3d import Point2Voxel as Point2VoxelGPU3d
        from spconv.core_cc.csrc.sparse.all.ops4d import Point2Voxel as Point2VoxelGPU4d
    ```

6. 执行以下命令编译安装spconv
    ```shell
    export SPCONV_DISABLE_JIT="1"
    cd ./spconv/
    python setup.py bdist_wheel
    cd ../ && pip install spconv/dist/spconv-*.whl
    ```

    【注意】安装完毕后建议运行以下命令，如无报错，证明安装无误，可继续安装流程
    ```shell
    python -c "import spconv"
    ```

#### 2.4 编译安装pytorch-scatter
执行以下命令编译安装pytorch-scatter
```shell
git clone https://github.com/rusty1s/pytorch_scatter.git -b 2.1.1
cd ./pytorch_scatter/
python setup.py bdist_wheel
cd ../ && pip install pytorch_scatter/dist/torch_scatter-*.whl
```

【注意】安装完毕后建议运行以下命令，如无报错，证明安装无误，可继续安装流程
```shell
python -c "import torch_scatter"
```

#### 2.5 编译安装Driving SDK
参考Driving SDK官方gitee仓README安装编译构建并安装Driving SDK包：[参考链接](https://gitee.com/ascend/DrivingSDK)

【注意】安装完毕后建议运行以下命令，如无报错，证明安装无误，可继续安装流程
```shell
python -c "import mx_driving"
```

#### 2.6 编译安装OpenPCDet
执行以下命令，应用过patch的模型根目录编译安装OpenPCDet
```shell
cd ./OpenPCDet/
python setup.py develop
```

### 准备数据集
1. 下载nuScenes数据集，请自行前往nuScenes官方网站下载3D目标检测数据集
2. 下载训练数据集data目录
   1. 克隆OpenPCDet源码：`git clone https://github.com/open-mmlab/OpenPCDet.git`
   2. 将OpenPCDet源码的data目录复制到本仓的OpenPCDet工程目录下
3. 下载解压的nuScenes数据集，并按照如下方式组织：
    ```
    OpenPCDet
    ├── data
    │   ├── nuscenes
    │   │   │── v1.0-trainval (or v1.0-mini if you use mini)
    │   │   │   │── samples
    │   │   │   │── sweeps
    │   │   │   │── maps
    │   │   │   │── v1.0-trainval
    ├── pcdet
    ├── tools
    ```
4. 安装数据处理相关依赖
    ```shell
    pip install nuscenes-devkit==1.0.5
    pip install av2
    pip install kornia==0.5.8
    pip install opencv-python-headless --force-reinstall
    ```
5. 进入应用过patch文件的OpenPCDet的根目录, 执行数据预处理脚本，生成序列化数据集
    ```shell
    cd ./OpenPCDet/
    python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml --version v1.0-trainval
    ```

### 模型训练
1. 进入应用过patch文件的OpenPCDet的根目录。
   ```shell
   cd ./OpenPCDet/
   ```
2. numpy版本降级为1.23.5
执行以下命令，将numpy版本降级为1.23.5，过高的numpy版本会导致代码中numpy部分被废弃用法不可用
    ```shell
    pip install numpy==1.23.5
    ```
3. 运行训练脚本。
   该模型支持单机单机8卡训练
   ```shell
   cd tools/test
   bash train_centerpoint_full_8p.sh # 8p精度训练
   bash train_centerpoint_performance_8p.sh # 8p性能训练
   ```
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息

### 训练结果对比
#### 精度
训练精度结果展示表
| Exp | mATE | mASE | mAOE | mAVE | mAAE | mAP | NDS |
| - | - | - | - | - | - | - | - |
| 8p-竞品A | 32.59 | 26.35 | 44.26 | 24.50 | 19.30 | 49.20 | 59.78 |
| 8p-Atlas 800T A2 | 32.50 | 26.34 | 45.05 | 24.23 | 19.39 | 50.06 | 60.45 |

#### 性能
训练性能结果展示表
| Exp | global batch size | FPS |
| - | - | - |
| 8p-竞品A | 96 | 85.712 |
| 8p-Atlas 800T A2| 96|  66.160 |



## CenterPoint3d
### 准备环境
**表1** 版本支持表

| Torch_Version | 三方库依赖版本            |
| - |--------------------|
| PyTorch 2.1.0 | torchvision 0.16.0 |

- 环境准备指导。

    请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

#### 0. 克隆代码仓到当前目录并使用patch文件

```
git clone https://gitee.com/ascend/DrivingSDK.git
cd DrivingSDK/model_examples/CenterPoint
git clone https://github.com/open-mmlab/OpenPCDet.git
cd OpenPCDet
git checkout 255db8f02a8bd07211d2c91f54602d63c4c93356
cp -f ../OpenPCDet_npu.patch .
git apply --reject OpenPCDet_npu.patch
cp -rf ../test tools/
```

#### 1. 基本环境准备

在应用过patch文件的模型源码包所在目录下执行相应命令，安装模型需要的依赖
```shell
pip install -r requirements.txt && cd ../   # PyTorch 2.1版本
```

#### 2. 手动编译安装cumm和spconv
在开始安装前，确保系统已安装GCC 7.5.0

#### 2.1 前置依赖安装
执行以下命令，安装前置依赖pccm==0.3.4，ccimport==0.3.7

```shell
pip install pccm==0.3.4
pip install ccimport==0.3.7
```

#### 2.2 编译安装cumm
执行以下命令编译安装cumm

```shell
git clone https://github.com/FindDefinition/cumm.git -b v0.2.9
export CUMM_CUDA_VERSION=""
export CUMM_DISABLE_JIT="1"
cd ./cumm/
python setup.py bdist_wheel
cd ../ && pip install cumm/dist/cumm-*.whl
```

【注意】安装完毕后建议运行以下命令，如无报错，证明安装无误，可继续安装流程
```shell
python -c "import cumm"
```

#### 2.3 编译安装spconv
1. 执行以下命令，拉取spconv源码用于本地编译

    ```shell
    git clone https://github.com/traveller59/spconv.git -b v2.1.25
    ```

2. 将spconv/spconv/pytorch/ops.py文件第32行代码进行调整
    ```python
    # 将代码 if hasattr(_ext, "cumm"):
    # 调整为 if 0:
    ```

3. 注释spconv/spconv/utils/__init__.py文件第26-30行代码
    ```python
    if not CPU_ONLY_BUILD:
        from spconv.core_cc.csrc.sparse.all.ops1d import Point2Voxel as Point2VoxelGPU1d
        from spconv.core_cc.csrc.sparse.all.ops2d import Point2Voxel as Point2VoxelGPU2d
        from spconv.core_cc.csrc.sparse.all.ops3d import Point2Voxel as Point2VoxelGPU3d
        from spconv.core_cc.csrc.sparse.all.ops4d import Point2Voxel as Point2VoxelGPU4d
    ```

4. 执行以下命令编译安装spconv
    ```shell
    export SPCONV_DISABLE_JIT="1"
    cd ./spconv/
    python setup.py bdist_wheel
    cd ../ && pip install spconv/dist/spconv-*.whl
    ```

    【注意】安装完毕后建议运行以下命令，如无报错，证明安装无误，可继续安装流程
    ```shell
    python -c "import spconv"
    ```

#### 2.5 编译安装Driving SDK
参考Driving SDK官方gitee仓README安装编译构建并安装Driving SDK包：[参考链接](https://gitee.com/ascend/DrivingSDK)

【注意】安装完毕后建议运行以下命令，如无报错，证明安装无误，可继续安装流程
```shell
python -c "import mx_driving"
```

#### 2.6 编译安装OpenPCDet
在应用过patch文件模型根目录，执行以下命令，编译安装OpenPCDet
```shell
cd ./OpenPCDet/
python setup.py develop
```

#### 2.7 高性能内存库替换
参考昇腾官方指导文档，下载高性能内存库并导入环境变量[参考链接](https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/performance_tuning_0067.html)

### 准备数据集
1. 下载nuScenes数据集，请自行前往nuScenes官方网站下载3D目标检测数据集
2. 下载训练数据集data目录
   1. 克隆OpenPCDet源码：`git clone https://github.com/open-mmlab/OpenPCDet.git`
   2. 将OpenPCDet源码的data目录复制到本仓的OpenPCDet工程目录下
3. 下载解压的nuScenes数据集，并按照如下方式组织：
    ```
    OpenPCDet
    ├── data
    │   ├── nuscenes
    │   │   │── v1.0-trainval (or v1.0-mini if you use mini)
    │   │   │   │── samples
    │   │   │   │── sweeps
    │   │   │   │── maps
    │   │   │   │── v1.0-trainval
    ├── pcdet
    ├── tools
    ```
4. 安装数据处理相关依赖
    ```shell
    pip install nuscenes-devkit==1.0.5
    pip install av2==0.2.1
    pip install kornia==0.5.8
    pip install opencv-python-headless --force-reinstall
    ```
5. 进入应用过patch文件的OpenPCDet的根目录, 执行数据预处理脚本，生成序列化数据集
    ```shell
    cd ./OpenPCDet/
    python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml --version v1.0-trainval
    ```

### 模型训练
1. 进入应用过patch文件的OpenPCDet的根目录。
   ```shell
   cd ./OpenPCDet/
   ```
2. numpy版本降级为1.23.5
执行以下命令，将numpy版本降级为1.23.5，过高的numpy版本会导致代码中numpy部分被废弃用法不可用
    ```shell
    pip install numpy==1.23.5
    ```
3. 运行训练脚本。
   该模型支持单机单机8卡训练
   ```shell
   cd tools/test
   bash train_centerpoint3d_full_8p.sh # 8p精度训练
   bash train_centerpoint3d_performance_8p.sh # 8p性能训练
   ```
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息

### 训练结果对比
#### 精度
训练精度结果展示表
| Exp | mATE | mASE | mAOE | mAVE | mAAE | mAP | NDS |
| - | - | - | - | - | - | - | - |
| 8p-竞品A | 28.80 | 25.43 | 37.27 | 21.55 | 18.24 | 58.65 | 66.22 |
| 8p-Atlas 800T A2 | 28.81 | 25.35 | 38.46 | 21.00 | 17.82 | 58.34 | 66.11 |

#### 性能
训练性能结果展示表
| Exp | global batchsize | FPS |
| - | - | - |
| 8p-竞品A | 32 | 48.48 |
| 8p-Atlas 800T A2| 32 | 28.881 |

## FAQ
### ImportError:/usr/local/gcc-7.5.0/lib64/libgomp.so.1:cannot allocate memory in static TLS block,
glibc版本兼容性问题，升级glibc版本或者手动导入环境变量export LD_PRELOAD=/usr/local/gcc-7.5.0/lib64/libgomp.so.1
### ImportError: {conda_env_path}/bin/../lib/libgomp.so.1:cannot allocate memory in static TLS block
出现上述报错时，将报错路径补充到环境变量LD_PRELOAD中即可，可参考以下指令
```shell
export LD_PRELOAD={conda_env_path}/bin/../lib/libgomp.so.1:$LD_PRELOAD # {conda_env_path}替换为实际使用python环境根目录
```
### ImportError: {conda_env_path}/lib/python3.8/site-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0
出现上述报错时，将报错路径补充到环境变量LD_PRELOAD中即可，可参考以下指令
```shell
export LD_PRELOAD={conda_env_path}/lib/python3.8/site-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0:$LD_PRELOAD # {conda_env_path}替换为实际使用python环境根目录
```
### ImportError: libblas.so.3: cannot open shared object file: No such file or directory
执行以下指令安装相关依赖即可
```shell
conda install -c conda-forge blas
```
### 数据预处理序列化时报错
可能时网络不稳定导致数据集下载时缺失部分image或者.bin文件
### 报错：KeyError:'road_plane'
修改`tools/cfgs/kitti_models/pointpillar.yaml`，`USE_ROAD_PLANE: False`
### 安装编译cumm时报错：TypeError: ccimport() got multiple values for argument 'std'
```python
pip install ccimport==0.3.7
```
### 训练卡住，日志记录`Wait 30 seconds for next check`
需要删除上次训练的存储，位置在`OpenPCDet/output/kitti_models/pointpillat/default`
### 运行报错：bc: command not found
bc命令是linux系统上的任意精度计算器语言, 有上述提示说明操作系统尚未安装bc，执行以下命令安装bc
```shell
yum -y install bc
```
### pkg_resources.DistributionNotFound: The 'protobuf' distribution was not found and is required by the application
执行以下指令在python环境中安装protobuf三方依赖
```shell
pip install protobuf
```
### 在Pytorh 2.5, Python 3.10环境下，需要在1 基本环境准备 中使用CenterPoint路径下的2.5_requirements.txt进行依赖安装
在CenterPoint路径下执行
```shell
pip install -r 2.5_requirements.txt
```

## 版本说明
[2024-12-23] **NEW:** CenterPoint3d模型在NPU设备首次适配.
[2025-02-18] **NEW:** CenterPoint2d模型增加PT2.5相关依赖.
[2025-03-12] **NEW:** CenterPoint仓CenterHead模块性能优化，并提供高性能内存库安装指导，更新了CenterPoint3d的性能数据
[2025-04-21] **NEW:** CenterPoint2d模型优化fps计算方式，乘以卡数，更新性能指标，readme中添加global batch size
