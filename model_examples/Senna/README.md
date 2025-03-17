# Senna for PyTorch

## 目录

- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [支持任务列表](#支持任务列表)
  - [代码实现](#代码实现)
- [Senna本地部署](#Senna本地部署)
  - [准备训练环境](#准备训练环境)
    - [安装昇腾环境](#安装昇腾环境)
    - [安装模型环境](#安装模型环境)
    - [准备数据集](#准备数据集)
    - [准备预训练权重](#准备预训练权重)
  - [快速开始](#快速开始)
    - [开始训练](#开始训练)
    - [训练结果](#训练结果)
- [变更说明](#变更说明)
- [FAQ](#FAQ)



# 简介


## 模型介绍
Senna 是一套自动驾驶系统，它将大规模视觉-语言模型与端到端规划框架无缝集成，从而提升了规划的安全性、鲁棒性和泛化能力。该系统实现了业界最先进的规划性能，展现出卓越的跨场景泛化能力和强大的可迁移性。


## 支持任务列表

本仓已经支持以下模型任务类型

| 模型       |   任务列表  | 是否支持  |
| :--------: | :--------: | :------: |
| Senna |   train    |    ✔     |



## 代码实现

- 参考实现：
```
url=https://github.com/hustvl/Senna
commit_id=5f202ce84dc4fe52949934ab0921e287d733ff8f
```

- 适配昇腾 AI 处理器的实现：
```
url=https://gitee.com/ascend/DrivingSDK.git
code_path=model_examples/Senna
```



# Senna本地部署

## 准备训练环境

### 安装昇腾环境

请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表1中软件版本。

**表 1** 昇腾软件版本支持表

|     软件类型      | 支持版本 |
| :---------------: | :------: |
| FrameworkPTAdaper | 7.0.0 |
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
conda create -n senna python=3.10
conda activate senna
```


3. 克隆代码仓并使用patch文件
```
git clone https://gitee.com/ascend/DrivingSDK.git -b master
cd DrivingSDK/model_examples/Senna
chmod -R 777 run.sh
./run.sh
```

安装依赖：
```
cd Senna
pip install -r requirements.txt
```

安装NPU适配bitsandbytes版本：
(1)下载源码并创建所需文件
将使用的昇腾芯片型号记作 Ascend_version
```
cd ..
./bitsandbytes_npu.sh "{Ascend_version}"
```
(2)安装：
将环境目录记作 env_root_dir
在bitsandbytes目录下：
在setup.py文件中：第11行setup()函数, 增加入参version="0.45.4"
```
source {cann_root_dir}/set_env.sh
bash deploy.sh
python setup.py install
```
最后在路径：/{env_root_dir}/lib/python3.10/site-packages/bitsandbytes-0.45.4-py3.10-linux-aarch64.egg/bitsandbytes/下创建文件_version.py，并加入__version__ = "0.45.4" 


### 准备数据集
参考原仓data_preparation章节(https://github.com/hustvl/Senna/tree/main)使用llava模型推理生成数据集llava_output

还需下载：
LLaVA训练数据集：https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain
目录结构如下：
```
Senna
├── assets/
├── data_tools/
├── eval_tools/
├── llaVa-Pretrain/
├── llava_output/
```


### 准备预训练权重
1. 下载vicuna-7b-v1.5权重：https://huggingface.co/lmsys/vicuna-7b-v1.5
2. 下载clip-vit-large-patch14-336权重：https://huggingface.co/openai/clip-vit-large-patch14-336
目录结构如下：
```
Senna
├── assets/
├── data_tools/
├── eval_tools/
├── llaVa-Pretrain/
├── vicuna-7b-v1.5/
├── clip-vit-large-patch14-336/
```



## 快速开始

### 训练任务
本任务主要提供**单机**的**8卡**训练脚本。

#### 开始训练
- 添加文件路径
(1) 在train_tools/pretrain_senna_llava.sh中第3-6行更换文件路径
将Senna模型所在目录记作 Senna_root_dir
```
MODEL="{Senna_root_dir}/vicuna-7b-v1.5/"
DATA="{Senna_root_dir}/llaVa-Pretrain/blip_laion_cc_sbu_558k.json"
IMAGE_DATA="/{Senna_root_dir}/llaVa-Pretrain"
OUT_DIR="/{Senna_root_dir}/output1/"
```

(2) 在train_tools/train_senna_llava.sh中第3-5行更换文件路径
```
MODEL="/{Senna_root_dir}/output1/"
DATA="/{Senna_root_dir}/lava_output/"
OUT_DIR="/{Senna_root_dir}/output2/"
```

- 运行训练脚本。

cd {DrivingSDK_root_dir}/model_examples/Senna/Senna
单机8卡精度训练：
```
bash test/train_8p.sh
```

单机8卡性能训练：
```
bash test/train_8p_performance.sh
```




#### 训练结果

|     芯片       |   卡数  | global batch size |   epoch   |   accuracy   |    FPS    |
|:-------------:| :------: | :---------------: | :----: | :-------:  | :------: |
|     竞品A      |    8p  |        16         |    1    |  82.35%  |        1.824    |
| Atlas 800T A2 |    8p  |        16         |      1   |   82.35%  |     1.376      |



# 变更说明

2025.3.5：首次发布


# FAQ
暂无