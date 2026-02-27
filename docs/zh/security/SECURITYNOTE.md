# Driving SDK 安全声明

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
2. 当Driving SDK运行异常，如输入校验异常（请参考API文档说明）、环境变量配置错误、算子执行报错等，会退出进程并打印报错信息，属于正常现象。建议用户根据报错提示定位具体错误原因，包括通过设定算子同步执行、查看CANN日志、解析生成的Core Dump文件等方式。

## 公网地址声明

Driving SDK代码中包含公网地址声明如下表所示：
### 公网地址

|   类型   |   开源代码地址   | 文件名                                 |   公网IP地址/公网URL地址/域名/邮箱地址   | 用途说明                          |
|-------------------------|-------------------------|-------------------------------------|-------------------------|-------------------------------|
|   自研   |   不涉及   | ci/docker/ARM/Dockerfile            |  https://mirrors.aliyun.com/pypi/simple   | docker配置文件，用于配置pip源           |
|   自研   |   不涉及   | ci/docker/X86/Dockerfile            |   https://mirrors.huaweicloud.com/repository/pypi/simple   | docker配置文件，用于配置pip源           |   |
|   自研   |   不涉及   | ci/docker/ARM/install_cann.sh     |   https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN   | CANN包下载地址    |
|   自研   |   不涉及   | ci/docker/x86/install_cann.sh     |   https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN   | CANN包下载地址    |
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
| 开源引入 | https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0 | model_examples/Cosmos-Predict2/cosmos_predict2.patch | https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0 | 模型依赖包下载 |
| 开源引入 | https://github.com/yifanlu0227/viser | model_examples/Cosmos-Drive-Dreams/Cosmos-Drive-Dreams.patch | https://github.com/yifanlu0227/viser | 模型依赖包下载 |
| 开源引入 | https://github.com/vllm-project/vllm-ascend/pull/1482 | model_examples/Cosmos-Reason1/cosmos-rl.patch | https://github.com/vllm-project/vllm-ascend/pull/1482 | 模型依赖包下载 |
| 开源引入 | https://github.com/huggingface/transformers/pull/38366 | model_examples/Cosmos-Reason1/cosmos-rl.patch | https://github.com/huggingface/transformers/pull/38366 | 模型依赖包下载 |


## 公开接口声明
参考[API清单](./docs/api/README.md)，Driving SDK提供了对外的自定义接口。如果一个函数在文档中有展示，则该接口是公开接口。否则，使用该功能前可以在社区询问该功能是否确实是公开的或意外暴露的接口，因为这些未暴露接口将来可能会被修改或者删除。

## 通信安全加固
Driving SDK在运行时依赖于`PyTorch`及`torch_npu`，您需关注通信安全加固，具体方式请参考[torch_npu通信安全加固](https://gitcode.com/Ascend/pytorch/blob/master/SECURITYNOTE.md#%E9%80%9A%E4%BF%A1%E5%AE%89%E5%85%A8%E5%8A%A0%E5%9B%BA)。

## 通信矩阵
Driving SDK在运行时依赖于`PyTorch`及`torch_npu`，涉及通信矩阵，具体信息请参考[torch_npu通信矩阵](https://gitcode.com/Ascend/pytorch/blob/master/SECURITYNOTE.md#%E9%80%9A%E4%BF%A1%E7%9F%A9%E9%98%B5)。

