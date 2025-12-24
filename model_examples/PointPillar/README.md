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
url=https://gitcode.com/Ascend/DrivingSDK.git
code_path=model_examples/PointPillar
```


## 模型适配情况
| 支持模型 | 支持数据集 |
| - | - |
| PointPillar | Kitti |


## PointPillar

### 安装昇腾环境

请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表1中软件版本。

**表 1** 昇腾软件版本支持表

|     软件类型      | 首次支持版本 |
| :---------------: | :------: |
| FrameworkPTAdapter | 6.0.0 |
|       CANN        | 8.0.0 |

### 安装模型环境

  **表 2**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 2.1.0 | torchvision 0.16.0 |


- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

  #### 0. 克隆代码仓到当前目录并使用patch文件

    ```
    git clone https://github.com/open-mmlab/OpenPCDet.git
    cd OpenPCDet
    git checkout 255db8f02a8bd07211d2c91f54602d63c4c93356
    cp -f ../OpenPCDet_npu.patch .
    git apply --reject OpenPCDet_npu.patch
    cp -rf ../test tools/
    ```

  #### 1. 基本环境
  在应用过patch的模型源码包根目录下执行相应命令，安装模型需要的依赖。
  ```
  conda create -n env_name python=3.8
  pip install -r requirements.txt && cd ../  # PyTorch 2.1版本
  ```

  #### 2. 手动编译安装cumm和spconv
  #### 2.1 前置依赖安装
   执行以下命令，安装前置依赖pccm==0.3.4，ccimport==0.3.7

   ```
   pip install pccm==0.3.4
   pip install ccimport==0.3.7
   ```
  
  手动编译安装cumm==0.2.9，spconv=2.1.25。需要安装指定版本GCC，版本为GCC 7.5.0
  #### 2.2 编译安装cumm

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
      cp -rf OpenPCDet/third_party_patches/spconv_patches/spconv/core_cc/csrc/sparse/all/__init__.pyi spconv/spconv/core_cc/csrc/sparse/all/__init__.pyi
      ```

   4. 将spconv/spconv/pytorch/ops.py文件第32行代码进行调整
      ```python
      将代码 if hasattr(_ext, "cumm"):
      调整为 if 0:
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

   #### 3. 编译安装Driving SDK
   参考Driving SDK官方gitcode仓README安装编译构建并安装Driving SDK包：[参考链接](https://gitcode.com/Ascend/DrivingSDK)

   【注意】安装完毕后建议运行以下命令，如无报错，证明安装无误，可继续安装流程
   ```shell
   python -c "import mx_driving"
   ```

   #### 4. 编译安装OpenPCDet
   安装相关依赖
   ```shell
   pip install av2==0.2.1
   pip install opencv-python-headless --force-reinstall
   pip install numpy==1.23.5
   ```
   在应用过patch的模型源码包根目录下执行相应命令
   ```
   cd ./OpenPCDet/
   python setup.py develop
   ```


### 准备数据集
1. 下载kitti数据集，请自行前往Kitti官网下载3D检测数据集
2. 解压下载的kitti数据集，并按照如下方式组织
   ```
   OpenPCDet
   ├── data
   │   ├── kitti
   │   │   │── ImageSets
   │   │   │── training
   │   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes) & (optional: depth_2)
   │   │   │── testing
   │   │   │   ├──calib & velodyne & image_2
   ├── pcdet
   ├── tools
   ```
3. 修改`tools/cfgs/kitti_models/pointpillar.yaml`，`USE_ROAD_PLANE: False`
4. 序列化数据集生成数据信息
   ```python
   cd OpenPCDet/
   python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
   ```

### 模型训练
1. 进入应用过patch的OpenPCDet的根目录。
   ```
   cd OpenPCDet/
   ```
2. 运行训练脚本。
   该模型支持单机单机8卡训练
   
   运行脚本支持命令行参数：
   - '--num-npu'：NPU卡数，默认为8；
   - '--batch-size': 每卡batch-size大小，默认为4；
   ```
   cd tools/test
   # 8卡精度脚本
   bash train_pointpillar_full_8p.sh
   (option) bash train_pointpillar_full_8p.sh --num-npu 8 --batch-size 4
   # 8卡性能脚本
   bash train_pointpillar_performance_8p.sh
   (option) bash train_pointpillar_performance_8p.sh --num-npu 8 --batch-size 4
   ```
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息

### 训练结果对比
#### 精度
训练精度结果展示表
| Exp | mAP_bbox | mAP_3d | mAP_bev | mAP_aos |
| - | - | - | - | - |
| 8p-竞品A | 80.19 | 76.58 | 79.27 | 73.58 |
| 8p-Atlas 800T A2 | 80.93 | 76.67 | 79.52 | 74.91 |

#### 性能
训练性能结果展示表
| Exp | global batch size | FPS |
| - | - | - |
| 8p-竞品A | 32 | 60.75 |
| 8p-Atlas 800T A2| 32 | 70.79 |

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
需要删除上次训练的存储，位置在`OpenPCDet/output/kitti_models/pointpillar/default`
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

## 版本说明
[2024-01-24] **NEW:** PointPillar模型在NPU设备首次适配.

[2025-04-21] **NEW:** PointPillar模型优化fps计算方式，乘以卡数，更新性能指标，readme中添加global batch size

[2025-06-12] **NEW:** PointPillar模型更新fps计算方式，更新性能指标

[2025-08-25] **NEW:** PointPillar模型更新fps，更新脚本传参
