# ADS-Accelerator

# 简介

ADS-Accelerator是基于昇腾NPU平台开发的适用于自动驾驶场景的算子和模型加速库，提供了一系列高性能的算子和模型加速接口，支持PyTorch框架。


# 安装
本项目依赖昇腾提供的pytorch_npu包和CANN包，需要先安装对应版本的pytorch_npu和CANN软件包，具体配套关系见pytorch仓[README](https://gitee.com/ascend/pytorch)。
> 基于安全考虑，建议您以非root用户身份执行以下操作。
## 准备环境
请参考昇腾官方文档[Pytorch框架训练环境准备](https://hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes/ptes_00001.html)。建议您在准备好环境后，将umask调整为`0027`，以保证文件权限正确。
## 从发布包安装
当前并未正式发布whl包 ，请参考源码安装方式。
## 从源码安装
1. 克隆原始仓
```shell
git clone https://gitee.com/ascend/ads.git
```
2. 编译ADS
> 注意：请在仓库根目录下执行编译命令
```shell
bash ci/build.sh --python=3.7
```
生成的whl包在`ads/dist`目录下, 命名规则为`ads_accelerator-1.0.0+git{commit_id}-cp{python_version}-linux_{arch}.whl`。
参数`--python`指定编译过程中使用的python版本，支持3.7及以上：

| 参数   | 取值范围                                                     | 说明                           | 缺省值 | 备注                                           |
| ------ | ------------------------------------------------------------ | ------------------------------ | ------ | ---------------------------------------------- |
| python | pytorch1.11，支持3.7及以上；pytorch1.11以上版本，支持3.8及以上 | 指定编译过程中使用的python版本 | 3.7    | 仅pytorch版本为1.11时才支持指定python版本为3.7 |

支持的CPU架构，python和torch版本对应关系如下：

| 架构    | pytorch版本  | 出包版本                                                 |
| ------- | ------------ | -------------------------------------------------------- |
| x86     | pytorch1.11  | Python3.7(\>=3.7.5)， Python3.8， Python3.9， Python3.10 |
| x86     | pytorch2.1.0 | Python3.8， Python3.9， Python3.10                       |
| x86     | pytorch2.2.0 | Python3.8， Python3.9， Python3.10                       |
| aarch64 | pytorch1.11  | Python3.7(\>=3.7.5)， Python3.8， Python3.9， Python3.10 |
| aarch64 | pytorch2.1.0 | Python3.8， Python3.9， Python3.10                       |
| aarch64 | pytorch2.2.0 | Python3.8， Python3.9， Python3.10                       |
3. 安装ADS
```shell
cd ads/dist
pip3 install ads_accelerator-1.0.0+git{commit_id}-cp{python_version}-linux_{arch}.whl
```
如需要保存安装日志，可在`pip3 install`命令后添加`--log <PATH>`参数，并对您指定的目录<PATH>做好权限控制。
# 卸载
Pytorch 框架训练环境的卸载请参考昇腾官方文档[Pytorch框架训练环境卸载](https://hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes/ptes_00032.html)。
ADS-Accelerator的卸载只需执行以下命令：
```shell
pip3 uninstall ads-accelerator
```

# 快速上手
1. source 环境变量
```shell
# 查看ads安装路径
pip3 show ads-accelerator
export ASCEND_CUSTOM_OPP_PATH=xxx/site-packages/ads/packages/vendors/customize/
export LD_LIBRARY_PATH=xxx/site-packages/ads/packages/vendors/customize/op_api/lib/:$LD_LIBRARY_PATH
```
2. 算子调用
```python
import torch
import torch_npu
import numpy as np
import ads.common
device = torch.device("npu:5")
a=torch.rand([8, 2048]).half().npu()
b=torch.rand([8, 2048]).half().npu()
c = ads.common.npu_ads_add(a,b)
print(c)
```

# 特性介绍
## 目录结构及说明
```
.
├── ads
│  ├── __init__.py
│  ├── common                   # 通用模块
│  │  ├── __init__.py
│  │  ├── CMakeLists.txt
│  │  ├── components            # 通用组件
│  │  └── ops                   # 通用算子
│  ├── motion                   # 运动模块
│  │  ├── __init__.py
│  │  ├── CMakeLists.txt   
│  │  ├── components            # 运动组件
│  │  └── ops                   # 运动算子
│  └── perception               # 感知模块
│     ├── __init__.py
│     ├── CMakeLists.txt
│     ├── fused                 # 融合模块
│     ├── point                 # 点云模块
│     └── vision                # 视觉模块
├── bind                        # torch 绑定
├── ci                          # ci脚本
├── cmake                       # cmake脚本
├── CMakeLists.txt              # cmake配置文件
├── CMakePresets.json           # cmake配置文件
├── docs                        # 文档
├── include                     # 头文件
├── LICENSE                     # 开源协议
├── MANIFEST.in                 # whl打包配置
├── OWNERS                      # 代码审查
├── README.md                   # 项目说明
├── requirements.txt            # 依赖
├── scripts                     # 工程脚本
├── setup.py                    # whl打包配置
├── tests                       # 测试文件
└── utils                       # 工具脚本
```
## 算子清单
请参见[算子清单](./docs/api/README.md)。
## 支持特性
- [x] 支持PyTorch 1.11.0，2.0.1，2.1.0
- [x] 支持ONNX模型转换，训推一体
- [ ] 支持图模式


# 安全声明
## 系统安全加固

1. 建议您在运行系统配置时开启ASLR（级别2），又称**全随机地址空间布局随机化**，以提高系统安全性，可参考以下方式进行配置：
    ```shell
    echo 2 > /proc/sys/kernel/randomize_va_space
    ```
2. 由于ADS-Accelerator需要用户自行编译，建议您对编译后生成的so文件开启`strip`, 又称**移除调试符号信息**, 开启方式如下：
    ```shell
    strip -s <so_file>
    ```
   具体so文件如下：
    - ads/packages/vendors/customize/op_api/lib/libcust_opapi.so
    - ads/packages/vendors/customize/op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so
    - ads/packages/vendors/customize/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opsproto_rt2.0.so
## 运行用户建议
出于安全性及权限最小化角度考虑，不建议使用`root`等管理员类型账户使用ads。

## 文件权限控制
在使用`ADS`时，您可能会进行profiling、调试等操作，建议您对相关目录及文件做好权限控制，以保证文件安全。
1. 建议您在使用`ADS`时，将umask调整为`0027`及以上，保障新增文件夹默认最高权限为`750`，文件默认最高权限为`640`。
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
在源码编译安装ADS-Accelerator时，需要您自行编译，编译过程中会生成一些中间文件，建议您在编译完成后，对中间文件做好权限控制，以保证文件安全。
## 运行安全声明
1. 建议您结合运行环境资源状况编写对应训练脚本。若训练脚本与资源状况不匹配，如数据集加载内存大小超出内存容量限制、训练脚本在本地生成数据超过磁盘空间大小等情况，可能引发错误并导致进程意外退出。
2. ADS在运行异常时(如输入校验异常（请参考api文档说明），环境变量配置错误，算子执行报错等)会退出进程并打印报错信息，属于正常现象。建议用户根据报错提示定位具体错误原因，包括通过设定算子同步执行、查看CANN日志、解析生成的Core Dump文件等方式。
## 公网地址声明

在ads的配置文件和脚本中存在[公网地址](#公网地址)

### 公网地址

|   类型   |   开源代码地址   | 文件名                                 |   公网IP地址/公网URL地址/域名/邮箱地址   | 用途说明                          |
|-------------------------|-------------------------|-------------------------------------|-------------------------|-------------------------------|
|   自研   |   不涉及   | ci/docker/ARM/Dockerfile            |   https://mirrors.huaweicloud.com/repository/pypi/simple   | docker配置文件，用于配置pip源           |
|   自研   |   不涉及   | ci/docker/X86/Dockerfile            |   https://mirrors.huaweicloud.com/repository/pypi/simple   | docker配置文件，用于配置pip源           |
|   自研   |   不涉及   | ci/docker/ARM/Dockerfile            |   https://dl.fedoraproject.org/pub/epel/7/aarch64/Packages/n/ninja-build-1.7.2-2.el7.aarch64.rpm   | docker配置文件，用于下载ninja-build    |
|   自研   |   不涉及   | ci/docker/ARM/build_protobuf.sh     |   https://gitee.com/it-monkey/protocolbuffers.git   | 用于打包whl的url入参                 |
|   自研   |   不涉及   | setup.cfg                           |   https://gitee.com/ascend/pytorch/tags   | 用于打包whl的download_url入参        |
|   自研   |   不涉及   | third_party\op-plugin\ci\build.sh   |   https://gitee.com/ascend/pytorch.git   | 编译脚本根据torch_npu仓库地址拉取代码进行编译   |
|   自研   |   不涉及   | third_party\op-plugin\ci\exec_ut.sh |   https://gitee.com/ascend/pytorch.git   | UT脚本根据torch_npu仓库地址下拉取代码进行UT测试 |
|   开源引入   |   https://gitee.com/it-monkey/protocolbuffers.git    | ci/docker/ARM/build_protobuf.sh     |   https://gitee.com/it-monkey/protocolbuffers.git   | 用于构建protobuf                  |
|   开源引入   |   https://gitee.com/it-monkey/protocolbuffers.git    | ci/docker/X86/build_protobuf.sh     |   https://gitee.com/it-monkey/protocolbuffers.git   | 用于构建protobuf                  |

## 公开接口声明
参考[API清单](./docs/api/README.md)，Ads提供了对外的自定义接口。如果一个函数在文档中有展示，则该接口是公开接口。否则，使用该功能前可以在社区询问该功能是否确实是公开的或意外暴露的接口，因为这些未暴露接口将来可能会被修改或者删除。
## 通信安全加固
ADS在运行时依赖于`pytorch`及`torch_npu`，您需关注通信安全加固，具体方式请参考[torch_npu通信安全加固](https://gitee.com/ascend/pytorch/blob/master/SECURITYNOTE.md#%E9%80%9A%E4%BF%A1%E5%AE%89%E5%85%A8%E5%8A%A0%E5%9B%BA)。
## 通信矩阵
ADS在运行时依赖于`pytorch`及`torch_npu`，涉及通信矩阵，具体信息请参考[torch_npu通信矩阵](https://gitee.com/ascend/pytorch/blob/master/SECURITYNOTE.md#%E9%80%9A%E4%BF%A1%E7%9F%A9%E9%98%B5)。