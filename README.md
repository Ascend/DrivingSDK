# Driving SDK

# 简介

Driving SDK是基于昇腾NPU平台开发的适用于自动驾驶场景的算子和模型加速库，提供了一系列高性能的算子和模型加速接口，支持PyTorch框架。


# 安装
## 前提条件
1. 本项目依赖昇腾提供的pytorch_npu包和CANN包，需要先安装对应版本的pytorch_npu和CANN软件包，具体配套关系见pytorch仓[README](https://gitcode.com/Ascend/pytorch)。
请参考昇腾官方文档[Pytorch框架训练环境准备](https://hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes/ptes_00001.html)。
2. 使用`pip3 install -r requirements.txt` 安装python依赖，`requirements.txt`文件位于项目根目录下。
3. 如果您需要编译`ONNX`插件，请安装`protobuf-devel-3.14.0`, 在`centos` 系统上可以执行`yum install protobuf-devel-3-14.0`，否则请将`CMakePresets.json`中的`ENABLE_ONNX`选项改为`FALSE`，`CMakePresets.json`文件位于项目根目录下。
4. 建议您在准备好环境后，使用`umask 0027`将umask调整为0027，以保证文件权限正确。
5. 建议您以非root用户身份执行以下操作。
6. 使用gcc编译本仓时，推荐使用gcc 10.2版本。

## 从发布包安装
当前并未正式发布whl包 ，请参考源码安装方式。
## 从源码安装
#### 1. 克隆原始仓。
```shell
git clone https://gitcode.com/Ascend/DrivingSDK.git -b master
```
#### 2. 编译Driving SDK。
> 注意：请在仓库根目录下执行编译命令
```shell
bash ci/build.sh --python=3.8
```
参数`--python`指定编译过程中使用的python版本，支持 3.8 及以上版本，缺省值为 3.8。请参考[编译指导](docs/get_started/compile.md)获取更多编译细节。

生成的whl包在`DrivingSDK/dist`目录下, 命名规则为`mx_driving-1.0.0+git{commit_id}-cp{python_version}-linux_{arch}.whl`。

#### 3. 安装Driving SDK。
```shell+
cd DrivingSDK/dist
pip3 install mx_driving-1.0.0+git{commit_id}-cp{python_version}-linux_{arch}.whl
```
如需要保存安装日志，可在`pip3 install`命令后添加`--log <PATH>`参数，并对您指定的目录<PATH>做好权限控制。


#### 配套关系

DrivingSDK支持的CPU架构，Python，PyTorch和torch_npu版本对应关系如下：

| Gitcode分支 |  CPU架构 |  支持的Python版本 | 支持的PyTorch版本 | 支持的torch_npu版本 |
|-----------|-----------|-------------------|-------------------|---------------------|
| master    | x86&aarch64|Python3.8.x,Python3.9.x,Python3.10.x,Python3.11.x|2.1.0|v2.1.0|
|           |       |Python3.8.x,Python3.9.x,Python3.10.x,Python3.11.x|2.5.1|v2.5.1|
|           |       |Python3.8.x,Python3.9.x,Python3.10.x,Python3.11.x|2.6.0|v2.6.0|
| branch_v7.1.RC1   | x86&aarch64|Python3.8.x,Python3.9.x,Python3.10.x,Python3.11.x|2.1.0|v2.1.0-7.2.0|
|           |       |Python3.8.x,Python3.9.x,Python3.10.x,Python3.11.x|2.6.0|v2.6.0-7.2.0|
| branch_v7.1.RC1   | x86&aarch64|Python3.8.x,Python3.9.x,Python3.10.x,Python3.11.x|2.1.0|v2.1.0-7.1.0|
|           |       |Python3.8.x,Python3.9.x,Python3.10.x,Python3.11.x|2.5.1|v2.5.1-7.1.0|
|           |       |Python3.8.x,Python3.9.x,Python3.10.x,Python3.11.x|2.6.0|v2.6.0-7.1.0|
| branch_v7.0.RC1    | x86&aarch64|Python3.8.x,Python3.9.x,Python3.10.x,Python3.11.x|2.1.0|v2.1.0-7.0.0|
|           |       |Python3.8.x,Python3.9.x,Python3.10.x,Python3.11.x|2.3.1|v2.3.1-7.0.0|
|           |       |Python3.8.x,Python3.9.x,Python3.10.x,Python3.11.x|2.4.0|v2.4.0-7.0.0|
| branch_v6.0.0    | x86&aarch64|Python3.8.x,Python3.9.x,Python3.10.x,Python3.11.x|2.1.0|v2.1.0-6.0.0|
|           |       |Python3.8.x,Python3.9.x,Python3.10.x,Python3.11.x|2.3.1|v2.3.1-6.0.0|
|           |       |Python3.8.x,Python3.9.x,Python3.10.x,Python3.11.x|2.4.0|v2.4.0-6.0.0|
| branch_v6.0.0-RC3    | x86&aarch64|Python3.8.x,Python3.9.x,Python3.10.x,Python3.11.x|2.1.0|v2.1.0-6.0.rc3|
|           |       |Python3.8.x,Python3.9.x,Python3.10.x,Python3.11.x|2.3.1|v2.3.1-6.0.rc3|
|           |       |Python3.8.x,Python3.9.x,Python3.10.x,Python3.11.x|2.4.0|v2.4.0-6.0.rc3|
| branch_v6.0.0-RC2    |x86&aarch64 |    Python3.7.x(>=3.7.5),Python3.8.x,Python3.9.x,Python3.10.x|1.11.0|v1.11.0-6.0.rc2|
|           |       |Python3.8.x,Python3.9.x,Python3.10.x|2.1.0|v2.1.0-6.0.rc2|
|           |       |Python3.8.x,Python3.9.x,Python3.10.x|2.2.0|v2.2.0-6.0.rc2|
|           |       |Python3.8.x,Python3.9.x,Python3.10.x|2.3.1|v2.3.1-6.0.rc2|
| branch_v6.0.0-RC1    |x86&aarch64 |    Python3.7.x(>=3.7.5),Python3.8.x,Python3.9.x,Python3.10.x|1.11.0|v1.11.0-6.0.rc1|
|           |       |Python3.8.x,Python3.9.x,Python3.10.x|2.1.0|v2.1.0-6.0.rc1|
|           |       |Python3.8.x,Python3.9.x,Python3.10.x|2.2.0|v2.2.0-6.0.rc1|



# 卸载
Pytorch 框架训练环境的卸载请参考昇腾官方文档[Pytorch框架训练环境卸载](https://hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes/ptes_00032.html)。
Driving SDK的卸载只需执行以下命令：
```shell
pip3 uninstall mx_driving
```

# 快速上手
```python
import torch, torch_npu
from mx_driving.common import scatter_max
updates = torch.tensor([[2, 0, 1, 3, 1, 0, 0, 4], [0, 2, 1, 3, 0, 3, 4, 2], [1, 2, 3, 4, 4, 3, 2, 1]], dtype=torch.float32).npu()
indices = torch.tensor([0, 2, 0], dtype=torch.int32).npu()
out = updates.new_zeros((3, 8))
out, argmax = scatter_max(updates, indices, out)
```

# 特性介绍
## 目录结构及说明
```
.
├── kernels                     # 算子实现
│  ├── op_host
│  ├── op_kernel
│  └── CMakeLists.txt
├── onnx_plugin                 # onnx框架适配层
├── mx_driving
│  ├── __init__.py
│  ├── csrc                     # 加速库API适配层
│  └── ...
├── model_examples              # 自动驾驶模型示例
│  └── BEVFormer                # BEVFormer模型示例
├── ci                          # ci脚本
├── cmake                       # cmake脚本
├── CMakeLists.txt              # cmake配置文件
├── CMakePresets.json           # cmake配置文件
├── docs                        # 文档
|  ├── api                      # 算子api调用文档
|  └── ...
├── include                     # 头文件
├── LICENSE                     # 开源协议
├── OWNERS                      # 代码审查
├── README.md                   # 项目说明
├── requirements.txt            # 依赖
├── scripts                     # 工程脚本
├── setup.py                    # whl打包配置
└── tests                       # 测试文件

```
## 算子清单
请参见[算子清单](./docs/api/README.md)。
## 支持特性
- [x] 支持PyTorch 2.1.0，2.5.1，2.6.0
- [x] 支持ONNX模型转换，训推一体
- [ ] 支持图模式
## onnx转换om
转换前需要手动添加环境变量
```shell
# 查看mx_driving安装路径
pip3 show mx_driving
export ASCEND_CUSTOM_OPP_PATH=xxx/site-packages/mx_driving/packages/vendors/customize/
export LD_LIBRARY_PATH=xxx/site-packages/mx_driving/packages/vendors/customize/op_api/lib/:$LD_LIBRARY_PATH
```

## 模型清单
DrivingSDK仓提供了包括感知、规划、端到端、VLA等自动驾驶模型基于昇腾机器的实操案例。每个模型都有详细的使用指导，后续将持续增加和优化典型模型。使用过程中，若遇到报错问题，可查看[自动驾驶模型FAQ](https://gitcode.com/Ascend/DrivingSDK/blob/master/docs/faq/model_faq.md)自助解决，或在[Issues](https://gitcode.com/Ascend/DrivingSDK/issues)中留言。如下列表中Released为Y的表示已经过测试验证，N的表示开发自验通过。

|  Model   | 8p-Atlas 800T A2性能(FPS)  | 8p-竞品性能(FPS)  | Released |
|  :----:  |  :----  | :----  | :----  |
| [BEVDepth](https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/BEVDepth)  | 32.29 | 22.11 |Y|
| [BEVDet](https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/BEVDet)  | 70.41 | 35.56 |Y|
| [BEVDet4D](https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/BEVDet4D)  | 7.04 | 5.59 |Y|
| [BevFormer](https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/BEVFormer)  | 3.66 | 3.32 |Y|
| [BEVFusion](https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/BEVFusion) | 23.62 | 22.54 |Y|
| [CenterNet](https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/CenterNet)  | 1257.444 | 542 |Y|
| [CenterPoint(2D)](https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/CenterPoint)  | 66.160 | 85.712 |Y|
| [CenterPoint(3D)](https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/CenterPoint)  | 39.41 | 48.48 |Y|
| [Deformable-DETR](https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/Deformable-DETR) | 63 | 65 |Y|
| [DenseTNT](https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/DenseTNT) | 166 | 237 |Y|
| [DETR](https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/DETR) | 122 | 126 |Y|
| [DETR3D](https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/DETR3D) | 14.35 | 14.28 |Y|
| [Diffusion-Planner](https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/Diffusion-Planner) | 5672.96 | 5304.32 |Y|
| [DiffusionDrive](https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/DiffusionDrive) | 28.43 | 30.53 |Y|
| [FCOS-resnet](https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/FCOS) | 196 | 196 |Y|
| [FCOS3D](https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/FCOS3D) | 44.31 | 44.30 |Y|
| [FlashOCC](https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/FlashOCC) | 104.85 | 67.98 |Y|
| [GameFormer](https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/GameFormer) | 7501.8 | 6400 |Y|
| [GameFormer-Planner](https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/GameFormer-Planner)  | 5319 | 5185 |Y|
| [LaneSegNet](https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/LaneSegNet) | 18.0 | 23.75 |Y|
| [MapTR](https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/MapTR) | 34.85 | 33.2 |Y|
| [MapTRv2](https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/MapTRv2)  | 23.03 | 21.91 |Y|
| [Mask2Former](https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/Mask2Former)  | 26.03 | 28.42 |Y|
| [MatrixVT](https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/MatrixVT)  | 46.19 | 36.89 |Y|
| [MultiPath++](https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/MultiPath++) | 149.53 | 198.14 |Y|
| [OpenDWM](https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/OpenDWM)  | 1.82 | 1.82 |Y|
| [OpenVLA](https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/OpenVLA)  | 56.14 | 73.12 |Y|
| [PanoOcc](https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/PanoOcc)  | 4.32 | 4.87 |Y|
| [Pi-0](https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/Pi-0)  | 116.36 | 136.17 |Y|
| [PivotNet](https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/PivotNet) | 9.75 | 13.8 |Y|
| [PointPillar(2D)](https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/PointPillar)  | 70.79 | 60.75 |Y|
| [SalsaNext](https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/SalsaNext) | 197.2 | 241.6 |Y|
| [Sparse4D](https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/Sparse4D)  | 70.59 | 65.75 |Y|
| [SparseDrive](https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/SparseDrive) | Stage1: 46.3<br>Stage2: 37.9 | Stage1: 41.0<br>Stage2: 35.2 |Y|
| [StreamPETR](https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/StreamPETR)  | 26.016 | 25.397 |Y|
| [SurroundOcc](https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/SurroundOcc)  | 7.59 | 7.78 |Y|
| [TPVFormer](https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/TPVFormer) | 6.69 | 10.32 |Y|
| [UniAD](https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/UniAD) | Stage1: 1.002<br>Stage2: 1.554 | Stage1: 1.359<br>Stage2: 2.000 |Y|
| [DexVLA](https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/Dexvla) | Stage2: 16.72<br>Stage3: 15.85 | Stage2: 18.88<br>Stage3: 18.67 |Y|
| [BEVNeXt](https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/BEVNeXt) | Stage1: 16.568<br>Stage2: 7.572 | Stage1: 36.643<br>Stage2: 11.651 |N|
| [HiVT](https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/HiVT) | 645 | 652 |N|
| [HPTR](https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/HPTR) | 25.12 | 36.07 |N|
| [LMDrive](https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/LMDrive)  | 8.02 | 13.85 |N|
| [MagicDriveDiT](https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/MagicDriveDiT) | Stage1: 0.83 | Stage1: 1.50 |N|
| [Panoptic-PolarNet](https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/Panoptic-PolarNet) | 1.28 | 1.69 |N|
| [PointTransformerV3](https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/PointTransformerV3)  | 11.92 | 35.56 |N|
| [QCNet](https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/QCNet) | 75.29 | 94.11 |N|
| [Senna](https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/Senna)  | 1.376 | 1.824 |N|
| [VAD](https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/VAD) | 2.847 | 7.476 |N|
| [VGGT](https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/VGGT)  | 25.04 | 15.30 |N|
| [YoloV8](https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/Yolov8)   | 214.64 | 479.73 |N|

# 支持的产品型号
- Atlas A2 训练系列产品

# 安全声明
## 系统安全加固

1. 建议您在运行系统配置时开启ASLR（级别2），又称**全随机地址空间布局随机化**，以提高系统安全性，可参考以下方式进行配置：
    ```shell
    echo 2 > /proc/sys/kernel/randomize_va_space
    ```
2. 由于Driving SDK需要用户自行编译，建议您对编译后生成的so文件开启`strip`, 又称**移除调试符号信息**, 开启方式如下：
    ```shell
    strip -s <so_file>
    ```
   具体so文件如下：
    - mx_driving/packages/vendors/customize/op_api/lib/libcust_opapi.so
    - mx_driving/packages/vendors/customize/op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so
    - mx_driving/packages/vendors/customize/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so
## 运行用户建议
出于安全性及权限最小化角度考虑，不建议使用`root`等管理员类型账户使用Driving SDK。

## 文件权限控制
在使用Driving SDK时，您可能会进行profiling、调试等操作，建议您对相关目录及文件做好权限控制，以保证文件安全。
1. 建议您在使用Driving SDK时，将umask调整为`0027`及以上，保障新增文件夹默认最高权限为`750`，文件默认最高权限为`640`。
2. 建议您对个人数据、商业资产、源文件、训练过程中保存的各类文件等敏感内容做好权限管控，可参考下表设置安全权限。
### 文件权限参考

|   类型                             |   Linux权限参考最大值   |
|----------------------------------- |-----------------------|
|  用户主目录                         |   750（rwxr-x---）     |
|  程序文件(含脚本文件、库文件等)       |   550（r-xr-x---）     |
|  程序文件目录                       |   550（r-xr-x---）     |
|  配置文件                           |   640（rw-r-----）     |
|  配置文件目录                       |   750（rwxr-x---）     |
|  日志文件(记录完毕或者已经归档)       |   440（r--r-----）     |
|  日志文件(正在记录)                  |   640（rw-r-----）    |
|  日志文件目录                       |   750（rwxr-x---）     |
|  Debug文件                         |   640（rw-r-----）      |
|  Debug文件目录                      |   750（rwxr-x---）     |
|  临时文件目录                       |   750（rwxr-x---）     |
|  维护升级文件目录                   |   770（rwxrwx---）      |
|  业务数据文件                       |   640（rw-r-----）      |
|  业务数据文件目录                   |   750（rwxr-x---）      |
|  密钥组件、私钥、证书、密文文件目录   |   700（rwx------）      |
|  密钥组件、私钥、证书、加密密文       |   600（rw-------）     |
|  加解密接口、加解密脚本              |   500（r-x------）      |

## 构建安全声明
在源码编译安装Driving SDK时，需要您自行编译，编译过程中会生成一些中间文件，建议您在编译完成后，对中间文件做好权限控制，以保证文件安全。
## 运行安全声明
1. 建议您结合运行环境资源状况编写对应训练脚本。若训练脚本与资源状况不匹配，如数据集加载内存大小超出内存容量限制、训练脚本在本地生成数据超过磁盘空间大小等情况，可能引发错误并导致进程意外退出。
2. Driving SDK在运行异常时(如输入校验异常（请参考api文档说明），环境变量配置错误，算子执行报错等)会退出进程并打印报错信息，属于正常现象。建议用户根据报错提示定位具体错误原因，包括通过设定算子同步执行、查看CANN日志、解析生成的Core Dump文件等方式。
## 公网地址声明

Driving SDK代码中包含公网地址声明如下表所示：
### 公网地址

|   类型   |   开源代码地址   | 文件名                                 |   公网IP地址/公网URL地址/域名/邮箱地址   | 用途说明                          |
|-------------------------|-------------------------|-------------------------------------|-------------------------|-------------------------------|
|   自研   |   不涉及   | ci/docker/ARM/Dockerfile            |  https://mirrors.aliyun.com/pypi/simple   | docker配置文件，用于配置pip源           |
|   自研   |   不涉及   | ci/docker/X86/Dockerfile            |   https://mirrors.huaweicloud.com/repository/pypi/simple   | docker配置文件，用于配置pip源           |   |
|   自研   |   不涉及   | ci/docker/ARM/install_cann.sh     |   https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN   | cann包下载地址    |
|   自研   |   不涉及   | ci/docker/x86/install_cann.sh     |   https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN   | cann包下载地址    |
|   自研   |   不涉及   | ci/docker/ARM/install_obs.sh     |   https://obs-community.obs.cn-north-1.myhuaweicloud.com/obsutil/current/obsutil_linux_arm64.tar.gz   | obs下载链接                 |
|   自研   |   不涉及   | ci/docker/X86/install_obs.sh     |   https://obs-community.obs.cn-north-1.myhuaweicloud.com/obsutil/current/obsutil_linux_amd64.tar.gz   | obs下载链接                 |
|   开源引入   |   https://gitee.com/it-monkey/protocolbuffers.git    | ci/docker/ARM/build_protobuf.sh     |   https://gitee.com/it-monkey/protocolbuffers.git   | 用于构建 protobuf                  |
|   开源引入   |   https://gitee.com/it-monkey/protocolbuffers.git    | ci/docker/X86/build_protobuf.sh     |   https://gitee.com/it-monkey/protocolbuffers.git   | 用于构建 protobuf                  |
|   开源引入   |   https://storage.googleapis.com/coco-dataset/external/PASCAL_VOC.zip    | model_examples/CenterNet/CenterNet.patch     |   https://storage.googleapis.com/coco-dataset/external/PASCAL_VOC.zip   | 源模型失效数据下载链接                  |
|   开源引入   |   https://s3.amazonaws.com/images.cocodataset.org/external/external_PASCAL_VOC.zip    | model_examples/CenterNet/CenterNet.patch     |   https://s3.amazonaws.com/images.cocodataset.org/external/external_PASCAL_VOC.zip   | 模型必要数据下载链接                |
| 开源引入 | https://download.pytorch.org/whl/torch_stable.html | model_examples/Diffusion-Planner/diffusionPlanner.patch | https://download.pytorch.org/whl/torch_stable.html | 模型依赖包下载 |
| 开源引入 | https://pypi.tuna.tsinghua.edu.cn/simple | model_examples/Diffusion-Planner/diffusionPlanner.patch | https://pypi.tuna.tsinghua.edu.cn/simple | 模型依赖包下载 |
| 开源引入 | https://github.com/Pointcept/Pointcept | model_examples/PointTransformerV3/Ptv3.patch | xiaoyang.wu.cs@gmail.com | 模型作者邮箱声明 |
| 开源引入 | https://www.argoverse.org/av2.html | model_examples/QCNet/patch/qcnet.patch | https://www.argoverse.org/av2.html | 模型数据集介绍 |

## 公开接口声明
参考[API清单](./docs/api/README.md)，Driving SDK提供了对外的自定义接口。如果一个函数在文档中有展示，则该接口是公开接口。否则，使用该功能前可以在社区询问该功能是否确实是公开的或意外暴露的接口，因为这些未暴露接口将来可能会被修改或者删除。
## 通信安全加固
Driving SDK在运行时依赖于`PyTorch`及`torch_npu`，您需关注通信安全加固，具体方式请参考[torch_npu通信安全加固](https://gitcode.com/Ascend/pytorch/blob/master/SECURITYNOTE.md#%E9%80%9A%E4%BF%A1%E5%AE%89%E5%85%A8%E5%8A%A0%E5%9B%BA)。
## 通信矩阵
Driving SDK在运行时依赖于`PyTorch`及`torch_npu`，涉及通信矩阵，具体信息请参考[torch_npu通信矩阵](https://gitcode.com/Ascend/pytorch/blob/master/SECURITYNOTE.md#%E9%80%9A%E4%BF%A1%E7%9F%A9%E9%98%B5)。


# 软件生命周期说明
## Driving SDK 分支维护策略

Driving SDK版本分支的维护阶段如下：

| **状态**            | **时间** | **说明**                                         |
| ------------------- | -------- | ------------------------------------------------ |
| 计划                | 1-3 个月 | 计划特性                                         |
| 开发                | 3 个月   | 开发特性                                         |
| 维护                | 6-12 个月| 合入所有已解决的问题并发布版本，针对不同的Driving SDK版本采取不同的维护策略，常规版本和长期支持版本维护周期分别为6个月和12个月 |
| 无维护              | 0-3 个月 | 合入所有已解决的问题，无专职维护人员，无版本发布 |
| 生命周期终止（EOL） | N/A      | 分支不再接受任何修改                             |


## Driving SDK 版本维护策略：

| **Driving SDK版本**     | **维护策略** | **当前状态** | **发布时间**   | **后续状态**           | **EOL日期** |
|---------------------|-----------|---------|------------|--------------------|-----------|
| v7.2.RC1  |  常规版本  | 维护      | 2025/09/30 | 预计2026/03/30起无维护	   |
| v7.1.RC1  |  常规版本  | 维护      | 2025/06/30 | 预计2025/12/30起无维护	   |
| v7.0.RC1  |  常规版本  | 维护      | 2025/03/30 | 预计2025/9/30起无维护	   |           |
| v6.0.0   |  常规版本  | 无维护      | 2024/12/30 | 2025/6/30起无维护	   |           |          |
| v6.0.0-RC3 |  常规版本  | 无维护      | 2024/09/30 | 2025/3/30起无维护	   |           |
| v6.0.0-RC2             |  常规版本  | 无维护      | 2024/06/30 | 2024/12/30起无维护	   |           |
| v6.0.0-RC1             |  常规版本  | 无维护  | 2024/03/30 | 2024/9/30起无维护           |           |


# 免责声明

## 致Driving SDK使用者
1. Driving SDK提供的模型仅供您用于非商业目的。
2. 对于各模型，Driving SDK平台仅提示性地向您建议可用于训练的数据集，华为不提供任何数据集，如您使用这些数据集进行训练，请您特别注意应遵守对应数据集的License，如您因使用数据集而产生侵权纠纷，华为不承担任何责任。
3. 如您在使用Driving SDK模型过程中，发现任何问题（包括但不限于功能问题、合规问题），请在Gitcode提交issue，我们将及时审视并解决。

## 致数据集所有者
如果您不希望您的数据集在Driving SDK中的模型被提及，或希望更新Driving SDK中的模型关于您的数据集的描述，请在Gitcode提交issue，我们将根据您的issue要求删除或更新您的数据集描述。衷心感谢您对Driving SDK的理解和贡献。