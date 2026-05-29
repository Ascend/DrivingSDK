# 部署Driving SDK环境

本文介绍部署Driving SDK环境的三种方式：**官方镜像部署**、**自定义镜像构建**和**源码编译安装**。推荐使用官方镜像部署，以便快速上手。

## 安装说明

### 产品硬件支持列表

在部署Driving SDK环境前，请确保选择以下经过验证的昇腾硬件：

| NPU 类型 | 架构 | 说明 | 状态 |
|----------|------|------|------|
| **Atlas A2 训练系列产品** | x86_64, aarch64 | A2 | 生产就绪 |
| **Atlas A3 训练系列产品** | x86_64, aarch64 | A3 | 生产就绪 |
| **Atlas A5 训练系列产品** | x86_64, aarch64 | A5 | 预览 |

- 各硬件产品对应物理机部署场景支持的操作系统请参考[兼容性查询助手](https://www.hiascend.com/hardware/compatibility)。

- 各硬件产品对应虚拟机及容器部署场景支持的操作系统请参考《CANN 软件安装指南》（[商用版](https://www.hiascend.com/document/detail/zh/canncommercial/) / [社区版](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/)）的“操作系统兼容性说明“章节。

### 软件版本配套说明

请参见《版本说明》中的“[相关产品版本配套说明](../release_note/release_notes.md#相关产品版本配套说明)”章节，下载安装对应的软件版本。

## 安装前准备

使用容器或在已有环境（物理机/容器）源码安装的部署场景下，都需要在物理机上先安装NPU[固件与驱动](https://hiascend.com/hardware/firmware-drivers/community)，请根据系统和硬件产品型号选择对应版本的社区版本或商用版本的固件与驱动。
参考《CANN 软件安装指南》（[商用版](https://www.hiascend.com/document/detail/zh/canncommercial/) / [社区版](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/)）的”安装NPU驱动和固件“章节或执行如下命令安装：

```shell
chmod +x Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run
chmod +x Ascend-hdk-<chip_type>-npu-firmware_<version>.run
./Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run --full --force
./Ascend-hdk-<chip_type>-npu-firmware_<version>.run --full
```

## 部署Driving SDK环境

### 方式一：官方镜像部署

Driving SDK 在昇腾社区 AscendHub 上提供预置 Docker 镜像，已内置 CANN、PyTorch 适配插件（PTA）和 Driving SDK 运行环境。镜像基于 openEuler 或 Ubuntu 构建，开箱即用。

镜像标签命名规范：`<DrivingSDK版本>-<CANN版本>-<NPU类型>-<操作系统类型>`

#### 字段说明

| 字段 | 说明 | 可选值 |
|------|------|--------|
| **DrivingSDK_VERSION** | Driving SDK 版本 | `26.0.0`, `master` |
| **CANN_VERSION** | CANN 版本 | `8.5.1`, `9.0.0` |
| **NPU_TYPE** | NPU 类型 | `A2`, `A3`, `950`|
| **OS_TYPE** | 操作系统类型 | `ubuntu22.04`, `openeuler24.03` |

#### 示例

- `26.0.0-cann8.5.1-910b-ubuntu22.04`: CANN 8.5.1, A2, Ubuntu 22.04
- `26.0.0-cann9.0.0-a3-openeuler24.03`: CANN 9.0.0, A3, openEuler 24.03

---

> 更多可用镜像版本请访问 [AscendHub DrivingSDK 页面](https://www.hiascend.com/developer/ascendhub/detail/696b50584fa04d4a8e99f7894f8eb176)。

#### 前置依赖

宿主机需已完成[安装前准备](#安装前准备)中的驱动与固件安装，并已安装 Docker（Docker 网络可用）。

#### 安装步骤

本文以 `26.0.0-cann9.0.0-910b-openeuler24.03` 版本为例，介绍Driving SDK官方镜像的安装步骤。

1. 拉取镜像。

    根据您的硬件型号和所需软件版本执行类似以下命令，拉取对应的镜像。

    ```shell
    docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/drivingsdk:26.0.0-cann9.0.0-910b-openeuler24.03
    ```

    执行如下命令查看镜像名和标签，确认拉取成功。将镜像名记作 `$IMAGE`，镜像标签记作 `$TAG`。

    ```shell
    docker images
    ```

2. 创建容器。

    创建 `run_drivingsdk_docker.sh` 脚本：

    ```shell
    #!/usr/bin/bash

    # 需要保证宿主机已经安装好了昇腾驱动，并将/usr/local/Ascend/driver挂载到容器中。
    # 容器中自带CANN包，位于/usr/local/Ascend/ascend-toolkit路径下。
    # 可根据需要挂载自定义工作目录。
    # 执行时需指定镜像名与镜像标签。

    IMAGE=$1
    TAG=$2

    docker run -it --ipc=host \
    --network=host \
    --privileged -u=root \
    --device=/dev/davinci0 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/bin/hccn_tool:/usr/bin/hccn_tool \
    ${IMAGE}:${TAG} \
    /bin/bash
    ```

    > **说明：** 以上脚本以单卡为例。如使用多卡，请追加 `--device=/dev/davinci1` 、 `--device=/dev/davinci2`… 等设备挂载。

    执行脚本创建并进入容器：

    ```shell
    #bash run_drivingsdk_docker.sh $IMAGE $TAG
    bash run_drivingsdk_docker.sh swr.cn-south-1.myhuaweicloud.com/ascendhub/drivingsdk 26.0.0-cann9.0.0-910b-openeuler24.03
    ```

3. 进入 conda 环境。

    ```shell
    conda activate torch2.7.1
    ```

    镜像中提供的 conda 环境、Python 和 PyTorch 配套关系如下表：

    | conda 环境 | Python | PyTorch | 用途 |
    |-----------|--------|---------|------|
    | torch2.1 | 3.8 | 2.1.0 | 通用 PyTorch 开发 |
    | torch2.7.1 | 3.10 | 2.7.1 | 最新 PyTorch 特性 |
    | bevformer | 3.10 | 2.7.1 | BEVFormer 模型训练 |
    | bevfusion | 3.10 | 2.7.1 | BEVFusion 模型训练 |
    | sparse4d | 3.10 | 2.7.1 | Sparse4D 模型训练 |

### 方式二：自定义镜像构建

除了使用社区提供的预置镜像外，Driving SDK 还支持用户根据自身需求自定义构建 Docker 镜像。仓库的 `docker/` 目录下提供了多种 CANN 版本（8.5.1 / 9.0.0）、NPU 类型（A2 / A3 / 950）和操作系统（Ubuntu 22.04 / openEuler 24.03）的 Dockerfile 组合，用户可基于这些 Dockerfile 进行本地构建或二次开发。适用于以下场景：

- 需要使用特定 CANN 版本或 NPU 类型组合，而社区预置镜像未覆盖
- 需要在基础镜像上添加额外的系统依赖或 Python 包
- 需要基于 Driving SDK 镜像进行二次定制开发

详细的镜像标签说明、Dockerfile 归档路径、构建步骤、硬件支持信息及常见问题，请参考[自定义镜像构建指导](../../../docker/OVERVIEW.zh.md)。

### 方式三：源码编译安装

容器部署和自定义镜像构建方式中已内置 CANN 及基础运行环境，无需额外安装。源码编译安装方式则需用户自行准备以下环境。

#### 安装CANN

从[CANN 下载页](https://www.hiascend.com/cann/download)获取配套版本的 Toolkit 并安装。若需要安装NNAL包或获取更多详细步骤请参考《CANN 软件安装指南》（[商用版](https://www.hiascend.com/document/detail/zh/canncommercial/) / [社区版](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/)）。

安装完成后配置 CANN 环境变量：

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh  # 可选：如安装了NNAL包
```

#### 前置依赖

Driving SDK 仓编译依赖以下组件：

1. 本项目依赖昇腾提供的 CANN 包和 torch_npu 包，需要先安装对应版本的 CANN 和 torch_npu 。具体配套关系和安装步骤，请参考 [Ascend Extension for PyTorch 插件](https://gitcode.com/Ascend/pytorch#%E7%89%88%E6%9C%AC%E8%AF%B4%E6%98%8E)和 [PyTorch 框架训练环境准备](https://www.hiascend.com/document/detail/zh/Pytorch/)。

2. 使用 `pip3 install -r requirements.txt` 命令安装 Python 依赖，`requirements.txt` 文件需位于项目根目录下。

3. （可选）如需编译 ONNX 插件，请安装 `protobuf-devel-3.14.0`。在 CentOS 系统上可执行 `yum install protobuf-devel-3-14.0`，否则请将 `CMakePresets.json` 中的 `ENABLE_ONNX` 选项改为 `FALSE`，`CMakePresets.json` 文件需位于项目根目录下。

4. 安装运行程序建议使用非root用户，且建议对安装程序的目录文件做好权限管控：文件夹权限设置为750，文件权限设置为640。可以通过设置umask控制安装后文件的权限，如使用 `umask 0027` 将 umask 调整为 `0027`，以保证文件权限正确。更多安全相关内容请参见《[安全声明](../../../SECURITYNOTE.md)》的说明。

5. 编译本仓推荐使用 GCC 10.2 版本。

#### 安装步骤

1. 下载 Driving SDK 源码：

    ```shell
    git clone https://gitcode.com/Ascend/DrivingSDK.git
    ```

2. 编译源码。

    支持 Release 和 Develop 两种编译模式，请按需选择。

    > **说明：** 如遇到编译问题，可查看 [FAQ](../faq/faq.md) 或在 Issue 中留言。

    **Release 模式**

    适用于生产环境。本文以 Python 3.8 为例：

    ```shell
    # 请在仓库根目录下执行编译命令。
    bash ci/build.sh --python=3.8
    # 或者
    python3.8 setup.py bdist_wheel

    # 若需要在 A5 设备上编译算子，需增加 '--a5'
    bash ci/build.sh --a5 --python=3.8
    # 或者
    python3.8 setup.py bdist_wheel --a5
    ```

    `--python` 参数指定编译过程中使用的 Python 版本，支持 Python 3.8 及以上，缺省值为 3.8。生成的 whl 包位于 `DrivingSDK/dist` 目录下，命名规则为 `mx_driving-1.0.0+git{commit_id}-cp{Python_version}-linux_{arch}.whl`。

    **Develop 模式**

    适用于开发调试环境，暂不支持 A5。本文以 Python 3.8 为例：

    ```shell
    python3.8 setup.py develop
    ```

    **在 Develop 模式下编译特定算子**

    如需编译特定算子（如 `DeformableConv2d` 和 `MultiScaleDeformableAttn`），算子名为 `op_host/xx.cpp` 中 `OpDef` 定义的名字，可使用 `--kernel-name` 参数：

    ```shell
    python3.8 setup.py develop --kernel-name="DeformableConv2d;MultiScaleDeformableAttn"
    # 多个 kernel-name 之间用 ";" 分隔。
    ```

3. 安装 Driving SDK：

    ```shell
    cd dist
    pip3 install mx_driving-1.0.0+git{commit_id}-cp{Python_version}-linux_{arch}.whl
    ```

    如需保存安装日志，可在 `pip3 install` 命令后添加 `--log <PATH>` 参数，并对指定目录做好权限控制。

## 卸载（可选）

执行以下命令卸载 Driving SDK：

```shell
pip3 uninstall mx_driving
```

## 下一步

环境部署完成后，请参考[快速上手](../get_started/quick_start_guide.md)开始使用 Driving SDK。
