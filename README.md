# Driving SDK

# 简介

Driving SDK是基于昇腾NPU平台开发的适用于自动驾驶场景的算子和模型加速库，提供了一系列高性能的算子和模型加速接口，支持PyTorch框架。


# 安装
## 前提条件
1. 本项目依赖昇腾提供的pytorch_npu包和CANN包，需要先安装对应版本的pytorch_npu和CANN软件包，具体配套关系见pytorch仓[README](https://gitee.com/ascend/pytorch)。
请参考昇腾官方文档[Pytorch框架训练环境准备](https://hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes/ptes_00001.html)。
2. 使用`pip3 install -r requirements.txt` 安装python依赖，`requirements.txt`文件位于项目根目录下。
3. 如果您需要编译`ONNX`插件，请安装`protobuf-devel-3.14.0`, 在`centos` 系统上可以执行`yum install protobuf-devel-3-14.0`，否则请将`CMakePresets.json`中的`ENABLE_ONNX`选项改为`FALSE`，`CMakePresets.json`文件位于项目根目录下。
4. 建议您在准备好环境后，使用`umask 0027`将umask调整为0027，以保证文件权限正确。
5. 建议您以非root用户身份执行以下操作。

## 从发布包安装
当前并未正式发布whl包 ，请参考源码安装方式。
## 从源码安装
1. 克隆原始仓。
```shell
git clone https://gitee.com/ascend/DrivingSDK.git -b master
```
2. 编译Driving SDK。
> 注意：请在仓库根目录下执行编译命令
```shell
bash ci/build.sh --python=3.8
```
生成的whl包在`DrivingSDK/dist`目录下, 命名规则为`mx_driving-1.0.0+git{commit_id}-cp{python_version}-linux_{arch}.whl`。
请参考[编译指导](docs/get_started/compile.md)获取更多编译细节。
参数`--python`指定编译过程中使用的python版本，支持3.8及以上：

| 参数   | 取值范围                                                     | 说明                           | 缺省值 | 备注                                           |
| ------ | ------------------------------------------------------------ | ------------------------------ | ------ | ---------------------------------------------- |
| python | pytorch2.1.0、2.3.1及以上版本，支持3.8及以上 | 指定编译过程中使用的python版本 | 3.8    |

支持的CPU架构，Python，PyTorch和torch_npu版本对应关系如下：

| Gitee分支 |  CPU架构 |  支持的Python版本 | 支持的PyTorch版本 | 支持的torch_npu版本 |
|-----------|-----------|-------------------|-------------------|---------------------|
| master    | x86&aarch64|Python3.8.x,Python3.9.x,Python3.10.x,Python3.11.x|2.1.0|v2.1.0|
|           |       |Python3.8.x,Python3.9.x,Python3.10.x,Python3.11.x|2.3.1|v2.3.1|
|           |       |Python3.8.x,Python3.9.x,Python3.10.x,Python3.11.x|2.4.0|v2.4.0|
| branch_v6.0.0-RC1    |x86&aarch64 |    Python3.7.x(>=3.7.5),Python3.8.x,Python3.9.x,Python3.10.x|1.11.0|v1.11.0-6.0.rc1|
|           |       |Python3.8.x,Python3.9.x,Python3.10.x|2.1.0|v2.1.0-6.0.rc1|
|           |       |Python3.8.x,Python3.9.x,Python3.10.x|2.2.0|v2.2.0-6.0.rc1|
| branch_v6.0.0-RC2    |x86&aarch64 |    Python3.7.x(>=3.7.5),Python3.8.x,Python3.9.x,Python3.10.x|1.11.0|v1.11.0-6.0.rc2|
|           |       |Python3.8.x,Python3.9.x,Python3.10.x|2.1.0|v2.1.0-6.0.rc2|
|           |       |Python3.8.x,Python3.9.x,Python3.10.x|2.2.0|v2.2.0-6.0.rc2|
|           |       |Python3.8.x,Python3.9.x,Python3.10.x|2.3.1|v2.3.1-6.0.rc2|
| branch_v6.0.0-RC3    | x86&aarch64|Python3.8.x,Python3.9.x,Python3.10.x,Python3.11.x|2.1.0|v2.1.0-6.0.rc3|
|           |       |Python3.8.x,Python3.9.x,Python3.10.x,Python3.11.x|2.3.1|v2.3.1-6.0.rc3|
|           |       |Python3.8.x,Python3.9.x,Python3.10.x,Python3.11.x|2.4.0|v2.4.0-6.0.rc3|
| branch_v6.0.0    | x86&aarch64|Python3.8.x,Python3.9.x,Python3.10.x,Python3.11.x|2.1.0|v2.1.0-6.0.0|
|           |       |Python3.8.x,Python3.9.x,Python3.10.x,Python3.11.x|2.3.1|v2.3.1-6.0.0|
|           |       |Python3.8.x,Python3.9.x,Python3.10.x,Python3.11.x|2.4.0|v2.4.0-6.0.0|


3. 安装Driving SDK。
```shell+
cd DrivingSDK/dist
pip3 install mx_driving-1.0.0+git{commit_id}-cp{python_version}-linux_{arch}.whl
```
如需要保存安装日志，可在`pip3 install`命令后添加`--log <PATH>`参数，并对您指定的目录<PATH>做好权限控制。
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
- [x] 支持PyTorch 2.1.0，2.3.1，2.4.0
- [x] 支持ONNX模型转换，训推一体
- [ ] 支持图模式

## 模型清单
|  Model   | 链接  | Released |
|  :----:  |  :----  | :----  |
| YoloV8  |https://gitee.com/ascend/DrivingSDK/tree/master/model_examples/Yolov8 |N|
| BEVDepth  |https://gitee.com/ascend/DrivingSDK/tree/master/model_examples/BEVDepth |Y|
| Sparse4D  |  https://gitee.com/ascend/DrivingSDK/tree/master/model_examples/Sparse4D |N|
| CenterNet  |  https://gitee.com/ascend/DrivingSDK/tree/master/model_examples/CenterNet |Y|
| PointPillar(2D)  |  https://gitee.com/ascend/DrivingSDK/tree/master/model_examples/PointPillar |Y|
| CenterPoint(2D)  | https://gitee.com/ascend/DrivingSDK/tree/master/model_examples/CenterPoint |Y|
| BevFormer  |  https://gitee.com/ascend/DrivingSDK/tree/master/model_examples/BEVFormer |Y|
| SurroundOcc  | https://gitee.com/ascend/DrivingSDK/tree/master/model_examples/SurroundOcc |Y|
| GameFormer-Planner  |  https://gitee.com/ascend/DrivingSDK/tree/master/model_examples/GameFormer-Planner |Y|
| StreamPETR  | https://gitee.com/ascend/DrivingSDK/tree/master/model_examples/StreamPETR |Y|
| Senna  | https://gitee.com/ascend/DrivingSDK/tree/master/model_examples/Senna |N|
| BEVDet  |  https://gitee.com/ascend/DrivingSDK/tree/master/model_examples/BEVDet |Y|
| PanoOcc  |  https://gitee.com/ascend/DrivingSDK/tree/master/model_examples/PanoOcc |N|
| TPVFormer  |  https://gitee.com/ascend/DrivingSDK/tree/master/model_examples/TPVFormer |Y|
| DETR | https://gitee.com/ascend/DrivingSDK/tree/master/model_examples/DETR |Y|
| Deformable-DETR | https://gitee.com/ascend/DrivingSDK/tree/master/model_examples/Deformable-DETR |Y|
| LaneSegNet | https://gitee.com/ascend/DrivingSDK/tree/master/model_examples/LaneSegNet |Y|
| BEVFusion | https://gitee.com/ascend/DrivingSDK/tree/master/model_examples/BEVFusion |Y|
| FCOS-resnet | https://gitee.com/ascend/DrivingSDK/tree/master/model_examples/FCOS |Y|
| FCOS3D | https://gitee.com/ascend/DrivingSDK/tree/master/model_examples/FCOS3D |Y|
| MapTR |https://gitee.com/ascend/DrivingSDK/tree/master/model_examples/MapTR|Y|
| UniAD | https://gitee.com/ascend/DrivingSDK/tree/master/model_examples/UniAD |Y|
| PivotNet|https://gitee.com/ascend/DrivingSDK/tree/master/model_examples/PivotNet|Y|
| CenterPoint(3D)  | https://gitee.com/ascend/DrivingSDK/tree/master/model_examples/CenterPoint |Y|
| LMDrive  | https://gitee.com/ascend/DrivingSDK/tree/master/model_examples/LMDrive |N|
| DETR3D | https://gitee.com/ascend/DrivingSDK/tree/master/model_examples/DETR3D |Y|
| DenseTNT | https://gitee.com/ascend/DrivingSDK/tree/master/model_examples/DenseTNT |Y|
| Mask2Former | https://gitee.com/ascend/DrivingSDK/tree/master/model_examples/Mask2Former |Y|
| GameFormer | https://gitee.com/ascend/DrivingSDK/tree/master/model_examples/GameFormer |Y|
| VAD | https://gitee.com/ascend/DrivingSDK/tree/master/model_examples/VAD |N|
| QCNet | https://gitee.com/ascend/DrivingSDK/tree/master/model_examples/QCNet |N|
| BEVNeXt | https://gitee.com/ascend/DrivingSDK/tree/master/model_examples/BEVNeXt |N|
| MultiPath++ | https://gitee.com/ascend/DrivingSDK/tree/master/model_examples/MultiPath++ |Y|
| SalsaNext | https://gitee.com/ascend/DrivingSDK/tree/master/model_examples/SalsaNext |N|
| Panoptic-PolarNet | https://gitee.com/ascend/DrivingSDK/tree/master/model_examples/Panoptic-PolarNet |N|
| HPTR | https://gitee.com/ascend/DrivingSDK/tree/master/model_examples/HPTR |N|
| MatrixVT | https://gitee.com/ascend/DrivingSDK/tree/master/model_examples/MatrixVT |Y|
| FlashOCC | https://gitee.com/ascend/DrivingSDK/tree/master/model_examples/FlashOCC |N|
| HiVT | https://gitee.com/ascend/DrivingSDK/tree/master/model_examples/HiVT |N|
| MagicDriveDiT | https://gitee.com/ascend/DrivingSDK/tree/master/model_examples/MagicDriveDiT |N|
| SparseDrive | https://gitee.com/ascend/DrivingSDK/tree/master/model_examples/SparseDrive |N|

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


## 公开接口声明
参考[API清单](./docs/api/README.md)，Driving SDK提供了对外的自定义接口。如果一个函数在文档中有展示，则该接口是公开接口。否则，使用该功能前可以在社区询问该功能是否确实是公开的或意外暴露的接口，因为这些未暴露接口将来可能会被修改或者删除。
## 通信安全加固
Driving SDK在运行时依赖于`PyTorch`及`torch_npu`，您需关注通信安全加固，具体方式请参考[torch_npu通信安全加固](https://gitee.com/ascend/pytorch/blob/master/SECURITYNOTE.md#%E9%80%9A%E4%BF%A1%E5%AE%89%E5%85%A8%E5%8A%A0%E5%9B%BA)。
## 通信矩阵
Driving SDK在运行时依赖于`PyTorch`及`torch_npu`，涉及通信矩阵，具体信息请参考[torch_npu通信矩阵](https://gitee.com/ascend/pytorch/blob/master/SECURITYNOTE.md#%E9%80%9A%E4%BF%A1%E7%9F%A9%E9%98%B5)。


# 软件生命周期说明
## Driving SDK 分支维护策略

Driving SDK版本分支的维护阶段如下：

| **状态**            | **时间** | **说明**                                         |
| ------------------- | -------- | ------------------------------------------------ |
| 计划                | 1—3 个月 | 计划特性                                         |
| 开发                | 3 个月   | 开发特性                                         |
| 维护                | 6-12 个月| 合入所有已解决的问题并发布版本，针对不同的Driving SDK版本采取不同的维护策略，常规版本和长期支持版本维护周期分别为6个月和12个月 |
| 无维护              | 0—3 个月 | 合入所有已解决的问题，无专职维护人员，无版本发布 |
| 生命周期终止（EOL） | N/A      | 分支不再接受任何修改                             |


## Driving SDK 版本维护策略：

| **Driving SDK版本**     | **维护策略** | **当前状态** | **发布时间**   | **后续状态**           | **EOL日期** |
|---------------------|-----------|---------|------------|--------------------|-----------|
| v7.0.RC1  |  常规版本  | 开发      | 2025/03/30 | 预计2025/9/30起无维护	   |           |
| v6.0.0   |  常规版本  | 维护      | 2024/12/30 | 预计2025/6/30起无维护	   |           |          |
| v6.0.0-RC3 |  常规版本  | 维护      | 2024/09/30 | 预计2025/3/30起无维护	   |           |
| v6.0.0-RC2             |  常规版本  | 无维护      | 2024/06/30 | 2024/12/30起无维护	   |           |
| v6.0.0-RC1             |  常规版本  | 无维护  | 2024/03/30 | 2024/9/30起无维护           |           |


# 免责声明

## 致Driving SDK使用者
1. Driving SDK提供的模型仅供您用于非商业目的。
2. 对于各模型，Driving SDK平台仅提示性地向您建议可用于训练的数据集，华为不提供任何数据集，如您使用这些数据集进行训练，请您特别注意应遵守对应数据集的License，如您因使用数据集而产生侵权纠纷，华为不承担任何责任。
3. 如您在使用Driving SDK模型过程中，发现任何问题（包括但不限于功能问题、合规问题），请在Gitee提交issue，我们将及时审视并解决。

## 致数据集所有者
如果您不希望您的数据集在Driving SDK中的模型被提及，或希望更新Driving SDK中的模型关于您的数据集的描述，请在Gitee提交issue，我们将根据您的issue要求删除或更新您的数据集描述。衷心感谢您对Driving SDK的理解和贡献。