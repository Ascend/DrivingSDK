# 部署Driving SDK环境

 本文主要介绍部署Driving SDK环境的两种方式，分别镜像部署和源码编译安装。推荐使用镜像部署Driving SDK环境，以便能够快速上手。



## 安装说明
在部署Driving SDK环境前，请确保选择以下经过验证的昇腾硬件：

表1：产品硬件支持列表

| 产品系列               | 产品型号                         |
|-----------------------|----------------------------------|
| Atlas A2 训练系列产品  | Atlas 800T A2 训练服务器          |
|                       | Atlas 900 A2 PoD 集群基础单元     |

## 部署Driving SDK环境
### 方式一：容器部署
#### 镜像及其版本介绍
Driving SDK镜像是基于openEulerOS构建的，包含Driving SDK模型和算子运行的基础环境，可实现模型训练快速上手。
<!-- 镜像版本说明需要询问,根据社区上的版本号为准(如7.3.0) -->
表2：镜像版本、CANN版本、PTA版本和Driving SDK版本配套关系

|镜像版本	|CANN版本	|PTA版本	|Driving SDK版本	       |备注|
|----------|-----------|-----------|--------------------|----|
|8.3.RC1_alpha001	|8.3.rc1.alpha001	|20250908	|20250908	|支持arm64架构，A2系列|
|8.5.0_alpha001	|8.5.0.alpha001	|20251125	|20251125	|支持arm64架构，A2系列|
#### 前置依赖
宿主机上有配套Ascend NPU芯片且已经安装好固件与驱动。宿主机已安装Docker，且Docker网络可用。

#### 安装步骤
1. 拉取镜像。
    ```shell
    docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/drivingsdk:8.5.0_alpha001-arm64
    ```
    > **说明：**
    > - 本文以拉取`8.5.0_alpha001-arm64`版本的镜像为例进行演示。
    > - 如需下载其它版本的镜像，请前往[镜像版本](https://www.hiascend.com/developer/ascendhub/detail/696b50584fa04d4a8e99f7894f8eb176)页面进行下载。
2. 创建容器。

    2.1 创建`run_drivingsdk_docker.sh`脚本。

    ```shell
    #!/usr/bin/bash

    # 需要保证宿主机已经安装好了昇腾驱动，并将/usr/local/Ascend/driver挂载到容器中。
    # 容器中自带CANN包，位于/usr/local/Ascend/ascend-toolkit路径下。
    # 镜像标签与镜像版本一致，例如8.5.0_alpha001-arm64。

    TAG=$1

    docker run -it --ipc=host \
    --network=host \
    --privileged -u=root \
    --device=/dev/davinci0 \
    --device=/dev/davinci1 \
    --device=/dev/davinci2 \
    --device=/dev/davinci3 \
    --device=/dev/davinci4 \
    --device=/dev/davinci5 \
    --device=/dev/davinci6 \
    --device=/dev/davinci7 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/bin/hccn_tool:/usr/bin/hccn_tool \
    drivingsdk:${TAG} \
    /bin/bash
    ```
    2.2 执行脚本创建并进入容器。
    ```shell
    bash run_drivingsdk_docker.sh 8.5.0_alpha001-arm64
    ```
    2.3 进入模型所需conda环境。
    ```shell
    conda activate torch2.1.0_py38
    ```
    表3：镜像中提供三个conda环境、Python和torch的配套关系
    |conda环境             |python      |torch|
    |----------------------|-----------|-----|
    |torch2.1.0_py38	   |3.8	       |2.1.0|
    |torch2.6.0_py310	   |3.10	   |2.6.0|
    |torch2.7.1_py310	   |3.10    	|2.7.1|


### 方式二：源码编译安装

#### 前置依赖
Driving SDK仓编译依赖以下组件：
1. 本项目依赖昇腾提供的torch_npu包和CANN包，需要先安装对应版本的torch_npu包和CANN包。具体配套关系和安装步骤，请参考[Ascend Extension for PyTorch插件](https://gitcode.com/Ascend/pytorch#%E7%89%88%E6%9C%AC%E8%AF%B4%E6%98%8E)和[PyTorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/Pytorch/720/configandinstg/instg/insg_0001.html)。

2. 使用`pip3 install -r requirements.txt`命令安装Python依赖，`requirements.txt`文件需位于项目根目录下。

3. （可选）如果您需要编译ONNX插件，请安装`protobuf-devel-3.14.0`, 在centos系统上可以执行`yum install protobuf-devel-3-14.0`，否则请将`CMakePresets.json`中的`ENABLE_ONNX`选项改为`FALSE`，`CMakePresets.json`文件需位于项目根目录下。

4. （可选）建议您以非root用户身份执行以下操作。

5. 建议您在准备好环境后，使用`umask 0027`将umask调整为0027，以保证文件权限正确。

6. 使用gcc编译本仓时，推荐使用gcc 10.2版本。

#### 安装步骤

1. 下载Driving SDK源码
    ```shell
    git clone https://gitcode.com/Ascend/DrivingSDK.git
    ```
2. 编译源码
支持Release和Develop两种编译模式，请按需选择。
    > **说明：**
    >
    > 如果你遇到编译问题，可查看[FAQ](../faq/model_faq.md)或者去issue中留言。

   **Release模式**

    该模式适用于生产环境。本文以Python 3.8为例进行演示。
    ```shell
    bash ci/build --python=3.8
    或者
    python3.8 setup.py bdist_wheel
    # 请在仓库根目录下执行编译命令。
    ```
    `--python`参数为指定编译过程中使用的Python版本，支持Python 3.8及以上版本，缺省值为 3.8。生成的whl包在`DrivingSDK/dist`目录下, 命名规则为`mx_driving-1.0.0+git{commit_id}-cp{Python_version}-linux_{arch}.whl`。

    **Develop模式**

    该模式适用于开发调试环境。本文以Python 3.8为例进行演示。
    ```shell
    python3.8 setup.py develop
    ```
    **在Develop模式下编译特定算子**

    如果你想编译一个或多个算子，比如`DeformableConv2d`和`MultiScaleDeformbaleAttn`,算子名为`op_host/xx.cpp`中的`OpDef`定义的名字，可以使用`--kernel-name`参数。示例如下：
    ```shell
    python3.8 setup.py develop --kernel-name="DeformableConv2d;MultiScaleDeformableAttn"
    # 每个kernel-name之间需要用 “;” 分隔。
    ```

3. 安装Driving SDK。
    ```shell
    cd dist
    pip3 install mx_driving-1.0.0+git{commit_id}-cp{Python_version}-linux_{arch}.whl
    ```
    如需要保存安装日志，可在`pip3 install`命令后添加`--log <PATH>`参数，并对您指定的目录做好权限控制。

## 卸载（可选）
Driving SDK的卸载只需执行以下命令。
```shell
pip3 uninstall mx_driving
```