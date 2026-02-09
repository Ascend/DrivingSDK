# Driving SDK 安装部署

## 快速部署
推荐使用 [Driving SDK镜像](https://www.hiascend.com/developer/ascendhub/detail/696b50584fa04d4a8e99f7894f8eb176) 快速完成环境部署。该镜像基于 openEulerOS 构建，预置 8.5.0 版本的 CANN 环境，以及包含版本配套的 torch_npu 和 mx_driving 的 conda 环境，可实现快速上手。

## 源码编译安装

### 1. 安装 CANN 和 torch_npu
参考昇腾官方文档 [PyTorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/Pytorch/730/configandinstg/instg/docs/zh/installation_guide/installation_description.md)。

### 2. 克隆 Driving SDK 代码仓
```shell
git clone https://gitcode.com/Ascend/DrivingSDK.git
```

### 3. 安装 Python 依赖 
```shell
cd DrivingSDK
pip3 install -r requirements.txt
```

### 4. 编译 Driving SDK
> **注意** ：
> - 推荐使用 gcc 10.2 版本
> - 请在仓库根目录下执行编译命令

以 Python 3.8 为例
```shell
bash ci/build.sh --python=3.8
```
参数 `--python` 指定编译过程中使用的 Python 版本，支持 Python 3.8 及以上版本，缺省值为 3.8。请参考[编译指导](./compile.md)获取更多编译细节。
生成的 whl 包在 `DrivingSDK/dist` 目录下, 命名规则为 `mx_driving-1.0.0+git{commit_id}-cp{Python_version}-linux_{arch}.whl`。

### 5. 安装 Driving SDK
```shell
cd dist
pip3 install mx_driving-1.0.0+git{commit_id}-cp{Python_version}-linux_{arch}.whl
```
如需要保存安装日志，可在 `pip3 install` 命令后添加 `--log <PATH>` 参数，并对您指定的目录<PATH>做好权限控制。

## 卸载（可选）
Driving SDK的卸载只需执行以下命令：
```shell
pip3 uninstall mx_driving
```