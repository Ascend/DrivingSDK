# StreamPETR for PyTorch

## 目录

- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [支持任务列表](#支持任务列表)
  - [代码实现](#代码实现)
- [StreamPETR本地部署](#StreamPETR本地部署)
  - [准备训练环境](#准备训练环境)
    - [安装昇腾环境](#安装昇腾环境)
    - [安装模型环境](#安装模型环境)
    - [准备数据集](#准备数据集)
    - [准备预训练权重](#准备预训练权重)
  - [快速开始](#快速开始)
    - [开始训练](#开始训练)
    - [训练结果](#训练结果)
- [公网地址说明](#公网地址说明)
- [变更说明](#变更说明)
- [FAQ](#FAQ)



# 简介


## 模型介绍
StreamPETR 是一种创新性的端到端 3D 目标检测模型，专注于高效实时的点云处理。它利用流式推理机制和基于 Transformer 的特征建模能力，首次将 3D 目标检测优化为一个高效、低延迟的全流程网络框架。StreamPETR 的设计理念完美契合了自动驾驶中“实时感知”和“高性能”的技术需求，是 3D 目标检测领域的重要突破。



## 支持任务列表

本仓已经支持以下模型任务类型

| 模型       |   任务列表  | 是否支持  |
| :--------: | :--------: | :------: |
| StreamPETR |   train    |    ✔     |



## 代码实现

- 参考实现：
```
url=https://github.com/exiawsh/StreamPETR
commit_id=95f64702306ccdb7a78889578b2a55b5deb35b2a
```

- 适配昇腾 AI 处理器的实现：
```
url=https://gitee.com/ascend/DrivingSDK.git
code_path=model_examples/StreamPETR
```



# StreamPETR本地部署

## 准备训练环境

### 安装昇腾环境

请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表1中软件版本。

**表 1** 昇腾软件版本支持表

|     软件类型      | 支持版本 |
| :---------------: | :------: |
| FrameworkPTAdaper | 7.0.RC1 |
|       CANN        | 8.1.RC1 |



### 安装模型环境

**表 2** 三方库版本支持表

| 三方库  | 支持版本 |
| :-----: | :------: |
| PyTorch |  2.1.0   |

0. 激活 CANN 环境

将 CANN 包所在目录记作 cann_root_dir，执行以下命令以激活环境

```
source {cann_root_dir}/set_env.sh
```

1. 参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》安装 2.1.0 版本的 PyTorch 框架和 torch_npu 插件。

2. 创建环境并激活环境
```
conda create -n streampetr python=3.8
conda activate streampetr
```

3. 克隆代码仓并使用patch文件
```
git clone https://gitee.com/ascend/DrivingSDK.git -b master
cd DrivingSDK/model_examples/StreamPETR
chmod -R 777 run.sh
./run.sh
```

安装依赖：
将Driving SDK文件夹所在位置记作 DrivingSDK_root_dir，执行以下命令
```
cd StreamPETR/patch
pip install -r requirements.txt
```

4. 安装mmcv-full
源码安装：
```
cd {DrivingSDK_root_dir}/model_examples/StreamPETR
git clone https://github.com/open-mmlab/mmcv.git -b 1.x
cd mmcv
cp -f {DrivingSDK_root_dir}/model_examples/StreamPETR/StreamPETR/patch/distributed.py mmcv/parallel/distributed.py
cp -f {DrivingSDK_root_dir}/model_examples/StreamPETR/StreamPETR/patch/text.py mmcv/runner/hooks/logger/text.py
source {cann_root_dir}/set_env.sh
MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py install
pip show mmcv-full
```

5. 安装mmdet
源码安装：
```
cd {DrivingSDK_root_dir}/model_examples/StreamPETR
git clone -b v2.28.2 https://github.com/open-mmlab/mmdetection.git
cd mmdetection
cp -f {DrivingSDK_root_dir}/model_examples/StreamPETR/StreamPETR/patch/match_cost.py mmdet/core/bbox/match_costs/match_cost.py
cp -f {DrivingSDK_root_dir}/model_examples/StreamPETR/StreamPETR/patch/pseudo_sampler.py mmdet/core/bbox/samplers/pseudo_sampler.py
pip install -e .
```


6. 安装mmdet3d
源码安装：
在模型根目录下，克隆mmdet3d仓，并进入mmdetection3d目录
```
cd {DrivingSDK_root_dir}/model_examples/StreamPETR/StreamPETR
git clone -b v1.0.0rc6 https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
```

在mmdetection3d目录下，修改代码

（1）删除requirements/runtime.txt中第3行 numba==0.53.0和第4行numpy

（2）修改mmdet3d/init.py中第22行 mmcv_maximum_version = '1.7.0'为mmcv_maximum_version = '1.7.2'

安装包：
在mmdetection3d根目录下执行：
```
pip install -v -e .
```

最后在mmdetection3d/configs/_base_/default_runtime.py目录下，修改代码
（1）第13行: dist_params = dict(backend='nccl') 改为 dist_params = dict(backend='hccl')


7. 安装mindspeed
源码安装：
```
cd {DrivingSDK_root_dir}/model_examples/StreamPETR
git clone https://gitee.com/ascend/MindSpeed.git
pip install -e MindSpeed
```


8. 安装tcmalloc动态库
源码安装：
```
cd {DrivingSDK_root_dir}/model_examples/StreamPETR/StreamPETR
mkdir gperftools
cd gperftools
wget https://github.com/gperftools/gperftools/releases/download/gperftools-2.16/gperftools-2.16.tar.gz
tar -zvxf gperftools-2.16.tar.gz
cd gperftools-2.16
./configure --prefix=/usr/local/lib --with-tcmalloc-pagesize=64
make
make install
echo '/usr/local/lib/lib/' >> /etc/ld.so.conf
ldconfig
export LD_LIBRARY_PATH=/usr/local/lib/lib/:$LD_LIBRARY_PATH
export PATH=/usr/local/lib/bin:$PATH
export LD_PRELOAD=/usr/local/lib/lib/libtcmalloc.so.4
```


9. 加入test相关文件
```
cd {DrivingSDK_root_dir}/model_examples/StreamPETR
cp -rf test StreamPETR
```

10. 编译优化
(1)python编译优化
源码下载毕昇编译器：
```
ulimit -n 4096
wget --no-check-certificate  https://kunpeng-repo.obs.cn-north-4.myhuaweicloud.com/BiSheng%20Enterprise/BiSheng%20Enterprise%20203.0.0/BiShengCompiler-4.1.0-aarch64-linux.tar.gz
tar -xvf BiShengCompiler-4.1.0-aarch64-linux.tar.gz
export PATH=$(pwd)/BiShengCompiler-4.1.0-aarch64-linux/bin:$PATH
export LD_LIBRARY_PATH=$(pwd)/BiShengCompiler-4.1.0-aarch64-linux/lib:$LD_LIBRARY_PATH
```

下载所需依赖：
```
wget --no-check-certificate https://www.bytereef.org/software/mpdecimal/releases/mpdecimal-2.5.1.tar.gz
tar -xvf mpdecimal-2.5.1.tar.gz
cd mpdecimal-2.5.1
bash ./configure --prefix=/path/to/install/mpdecimal
make -j
make install
```

使用python源码进行编译出包：
源码下载地址：https://www.python.org/downloads/source/
- 以3.8.17为例
```
tar -xvf Python-3.8.17.tgz
cd Python-3.8.17
export CC=clang
export CXX=clang++
```

使用指令which python查看现有python安装路径/path/to/python
- 以3.8.17为例
python安装路径为/home/miniconda3/envs/petr/bin/python

```
./configure --prefix=/home/miniconda3/envs/petr > --with-lto --enable-optimizations
make -j
make install
```

(2)pytorch、torchnpu编译优化
参考编译优化官方文档完成pytorch\torch_npu编译优化
官方文档：https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/performance_tuning_0061.html



### 准备数据集

进入原仓data_preparation章节(https://github.com/exiawsh/StreamPETR/blob/main/docs/data_preparation.md)准备数据集，数据集目录及结构如下：
```
StreamPETR
├── projects/
├── mmdetection3d/
├── tools/
├── configs/
├── ckpts/
├── data/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── panoptic/
│   │   ├── lidarseg/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
|   |   ├── nuscenes2d_temporal_infos_train.pkl
|   |   ├── nuscenes2d_temporal_infos_val.pkl
```



### 准备预训练权重

进入原仓data_preparation章节(https://github.com/exiawsh/StreamPETR/blob/main/docs/data_preparation.md)下载权重文件：
```
fcos3d_vovnet_imgbackbone-remapped.pth
```

```
cd /path/to/StreamPETR
mkdir ckpts
```

目录结构如下：
```
StreamPETR
├── projects/
├── mmdetection3d/
├── tools/
├── configs/
├── ckpts/
│   ├── fcos3d_vovnet_imgbackbone-remapped.pth
├── data/
```



## 快速开始

### 训练任务
本任务主要提供**单机**的**8卡**训练脚本。

#### 开始训练

- 运行训练脚本。

单机8卡精度训练：
```
cd {DrivingSDK_root_dir}/model_examples/StreamPETR/StreamPETR
source tools/env_model.sh
bash test/train_8p.sh
```

单机8卡性能训练：
```
bash test/train_8p_performance.sh
```

若以上命令执行出错，执行以下命令：
```
source tools/env_model.sh
tools/dist_train.sh projects/configs/StreamPETR/stream_petr_vov_flash_800_bs2_seq_24e.py 8 --work-dir work_dirs/report_vision
```


#### 训练结果（使用10. 编译优化）

|     芯片      | 卡数 | global batch size | epoch   |   mAP    |      NDS     |    性能-单步迭代耗时(s)   |    FPS    |
| :-----------: | :--: | :---------------: | :----: | :-------: | :---------: | :----------------------: | :------: |
|     竞品A     |  8p   |        16         |  24    |  0.4822  |    0.5708    |           0.63          |     25.397    |
| Atlas 800T A2 |  8p   |        16         |  24   |   0.4803  |    0.5703   |           0.70          |   22.857      |


训练结果（不使用10. 编译优化）
|     芯片      | 卡数 | global batch size | epoch   |   mAP    |      NDS     |    性能-单步迭代耗时(s)   |    FPS    |
| :-----------: | :--: | :---------------: | :----: | :-------: | :---------: | :----------------------: | :------: |
|     竞品A     |  8p   |        16         |  24    |  0.4822  |    0.5708    |           0.63          |     25.397    |
| Atlas 800T A2 |  8p   |        16         |  24   |   0.4803  |    0.5703   |           0.84          |   19.047      |



# 变更说明

2024.12.24：首次发布

# FAQ

暂无