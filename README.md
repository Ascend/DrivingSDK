### 简介

本项目基于昇腾NPU开发了用于自动驾驶场景的高性能算子

### 编译、安装ADS

#### 发布包安装

暂未正式发布

#### 源码安装

**安装依赖**

> 安装对应的版本的torch、torch_npu、cann包，具体配套关系见pytorch仓(https://gitee.com/ascend/pytorch)首页readme 
>
> 并source cann包环境变量

##### 下载ADS

```shell
# 下载ads仓
git clone https://gitee.com/ascend/ads.git
```

##### 编译ADS

```shell
# 编译
# NOTE: 请在仓库根目录下执行编译命令
cd ads
bash ci/build.sh --python=3.7
```

| 架构    | pytorch版本  | 出包版本                                                 |
| ------- | ------------ | -------------------------------------------------------- |
| x86     | pytorch1.11  | Python3.7(\>=3.7.5)， Python3.8， Python3.9， Python3.10 |
| x86     | pytorch2.0.1 | Python3.8， Python3.9， Python3.10                       |
| x86     | pytorch2.1.0 | Python3.8， Python3.9， Python3.10                       |
| aarch64 | pytorch1.11  | Python3.7(\>=3.7.5)， Python3.8， Python3.9， Python3.10 |
| aarch64 | pytorch2.0.1 | Python3.8， Python3.9， Python3.10                       |
| aarch64 | pytorch2.1.0 | Python3.8， Python3.9， Python3.10                       |

| 参数   | 取值范围                                                     | 说明                           | 缺省值 | 备注                                           |
| ------ | ------------------------------------------------------------ | ------------------------------ | ------ | ---------------------------------------------- |
| python | pytorch1.11，支持3.7及以上；pytorch1.11以上版本，支持3.8及以上 | 指定编译过程中使用的python版本 | 3.7    | 仅pytorch版本为1.11时才支持指定python版本为3.7 |

##### 安装ADS

```shell
cd ads/dist
pip3 install ads-1.0-cp37-cp37m-linux_aarch64.whl
```

#### CMC取包安装

当前ADS包还未商发，需到https://cmc-szv.clouddragon.huawei.com/cmcversion/index/search 搜索 FrameworkPTAdapter V100R001C01B001 取最新的包即可，注意需要根据环境的torch版本和python版本选择下载，如 ADS_v1.11.0_py37.tar.gz，其中v1.11.0表示torch版本，py37表示python版本。

后续计划发包版本

| 架构    | pytorch版本  | 出包版本                                                 |
| ------- | ------------ | -------------------------------------------------------- |
| x86     | pytorch1.11  | Python3.7(\>=3.7.5)， Python3.8， Python3.9， Python3.10 |
| x86     | pytorch2.0.1 | Python3.8， Python3.9， Python3.10                       |
| x86     | pytorch2.1.0 | Python3.8， Python3.9， Python3.10                       |
| aarch64 | pytorch1.11  | Python3.7(\>=3.7.5)， Python3.8， Python3.9， Python3.10 |
| aarch64 | pytorch2.0.1 | Python3.8， Python3.9， Python3.10                       |
| aarch64 | pytorch2.1.0 | Python3.8， Python3.9， Python3.10                       |

### ADS算子调用

##### 设置环境变量

注意：其中xxx表示当前环境上的python安装路径

```bash
# 查看ads安装路径
pip3 show ads-accelerator
export ASCEND_CUSTOM_OPP_PATH=xxx/site-packages/ads/packages/vendors/customize/
export LD_LIBRARY_PATH=xxx/site-packages/ads/packages/vendors/customize/op_api/lib/:$LD_LIBRARY_PATH
```

算子调用

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

