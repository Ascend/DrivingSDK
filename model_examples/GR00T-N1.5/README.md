# GR00T-N1.5 for PyTorch

## 目录
- [GR00T-N1.5 for PyTorch](#gr00t-n15-for-pytorch)
  - [目录](#目录)
- [简介](#简介)
  - [模型介绍](#模型介绍)
- [准备训练环境](#准备训练环境)
  - [安装昇腾环境](#安装昇腾环境)
  - [安装模型环境](#安装模型环境)
- [准备数据集](#准备数据集)
  - [获取预训练权重](#获取预训练权重)
  - [获取数据集](#获取数据集)
- [快速开始](#快速开始)
  - [训练结果展示](#训练结果展示)
- [版本说明](#版本说明)
  - [变更](#变更)
  - [FAQ](#faq)


# 简介

## 模型介绍

Isaac GR00T-N1 是 NVIDIA 在2025年初发布的视觉 - 语言 - 动作（VLA）模型，GR00T-N1.5 则是在模型架构、数据等方面对N1做了改进，提升了任务理解与泛化能力；模型具有双系统架构，接收多模态输入（包括语言和图像），能够在多样化环境中执行多种操作任务。

- 参考实现：https://github.com/NVIDIA/Isaac-GR00T/tree/main

- 适配昇腾 AI 处理器的实现：https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/GR00T-N1.5

# 准备训练环境

## 安装昇腾环境

请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表1中软件版本。

**表 1**  昇腾软件版本支持表

|     软件类型      | 支持版本 |
| :---------------: | :------: |
| FrameworkPTAdapter | 7.2.0  |
|       CANN        | 8.3.RC1  |

## 安装模型环境

当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

**表 2**  版本支持表

|      三方库       |  支持版本  |
|:--------------:|:------:|
|    Python      | 3.10 |
|    PyTorch     |  2.7.1   |

0. 激活 CANN 环境

1. 创建环境

    参考原仓下载 Driving SDK 加速库：https://gitcode.com/Ascend/DrivingSDK

    随后创建conda环境
    ```sh
    conda create -n gr00t python=3.10
    conda activate gr00t
    cd ./DrivingSDK/model_examples/GR00T-N1.5
    ```

2. 准备模型源码，安装gr00t

      在 GR00T-N1.5根目录下，克隆原始仓，替换其中部分代码并安装
      ```sh
      git clone https://github.com/NVIDIA/Isaac-GR00T
      cd Isaac-GR00T
      git checkout a86e9159a7e6acf771479007a79d4aeaa408c4c0
      cp -f ../gr00t.patch ./
      git apply --reject gr00t.patch
      pip install --upgrade setuptools
      pip install -e .[base]
      cp -f ../patch.py ./gr00t/utils/
      cp -f ../test/train* ./
    ```

3. 安装ffmpeg与decord库

      a. 安装ffmpeg
    ```sh
    # 源码编译ffmpeg
    wget https://ffmpeg.org/releases/ffmpeg-4.4.2.tar.bz2
    tar -xvf ffmpeg-4.4.2.tar.bz2
    cd ffmpeg-4.4.2
    ./configure --enable-shared  --prefix=/usr/local/ffmpeg    # --enable-shared is needed for sharing libavcodec with decord
    make -j 64
    make install
    ffmpeg   #验证安装成功
    cd ..
    ```

    如果运行ffmpeg后没有输出或者后续出现ffmpeg相关依赖错误，可能是环境变量未添加：

    ```sh
    # 编辑全局配置文件
    vim /etc/profile.d/ffmpeg.sh

    # 添加以下内容
    export PATH="/usr/local/ffmpeg/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/ffmpeg/lib:$LD_LIBRARY_PATH"

    # 使配置立即生效
    source /etc/profile
    ```

    b. 安装decord

    ```sh
    # 源码编译decord
    git clone --recursive https://github.com/dmlc/decord --depth 1
    cd decord
    mkdir build && cd build
    cmake ..  -DCMAKE_BUILD_TYPE=Release -DFFMPEG_DIR:PATH="/usr/local/ffmpeg/"
    make

    # 编译whl包
    cd ../python
    python setup.py sdist bdist_wheel
    pip install ./dist/decord-0.6.0-cp310-cp310-linux_aarch64.whl
    cd ../..
    ```


# 准备数据集

## 获取预训练权重

下载权重至Isaac-GR00T/GR00T-N1.5-3B，Huggingface链接: [GR00T-N1.5-3B](https://huggingface.co/nvidia/GR00T-N1.5-3B)

```sh
pip install huggingface-hub
hf download nvidia/GR00T-N1.5-3B --local-dir ./GR00T-N1.5-3B
```

## 获取数据集

下载数据集至Isaac-GR00T/demo_data/so101-table-cleanup，Huggingface链接:
 [so101-table-cleanup](https://huggingface.co/spaces/lerobot/visualize_dataset?dataset=youliangtan%2Fso101-table-cleanup&episode=0)
```sh
hf download \
    --repo-type dataset youliangtan/so101-table-cleanup \
    --local-dir ./demo_data/so101-table-cleanup
```

```sh
cp examples/SO-100/so100_dualcam__modality.json ./demo_data/so101-table-cleanup/meta/modality.json
```

# 快速开始

* 单机8卡训练
```sh
bash train_8p.sh --batch_size=64 --num_npu=8 --max_steps=10000 --dataset_path=./demo_data/so101-table-cleanup --base_model_path=./GR00T-N1.5-3B
```

* 单机8卡训练性能
```sh
bash train_performance_8p.sh --batch_size=64 --num_npu=8 --max_steps=1000 --dataset_path=./demo_data/so101-table-cleanup --base_model_path=./GR00T-N1.5-3B
```


## 训练结果展示

**表 3**  训练结果展示表

|     芯片      | 卡数 | global batch size | max steps | Final loss | FPS  |
| :-----------: | :--: | :---------------: | :---: | :--------------------: | :--------------------|
|     竞品A     |  8p  |         512       |  10000 |  0.0032 |  276.38  |    
| Atlas 800T A2 |  8p  |         512       |  10000 |  0.0031 |  337.35  | 

# 版本说明

## 变更

2025.10.28: 首次发布。


## FAQ
Q: 在无法访问 Hugging Face hub 的情况下运行模型报错？

A: 用户可以前往官网或使用 Hugging Face 镜像源在有网络的情况下自主下载，文件结构如下：

模型权重
```sh
GR00T-N1.5-3B
├── experiment_cfg
│   └── metadata.json
├── .gitattributes
├── BIAS.md
├── EXPLAINABILITY.md
├── LICENSE
├── PRIVACY.md
├── README.md
├── SAFETY_and_SECURITY.md
├── config.json
├── model-00001-of-00003.safetensors
├── model-00002-of-00003.safetensors
├── model-00003-of-00003.safetensors
└── model.safetensors.index.json
```
训练数据集
```sh
so101-table-cleanup
├── data
│   └── chunk-000
├── meta
├── videos
│   └── chunk-000
│       ├── observation.images.front
│       └── observation.images.wrist
├── .gitattributes
└── README.md
```

Q: 训练完成后如何开环评估？

A: 运行如下命令
```sh
python scripts/eval_policy.py --plot \
   --embodiment_tag new_embodiment \
   --model_path <YOUR_CHECKPOINT_PATH>  \
   --data_config so100_dualcam \
   --dataset_path ./demo_data/so101-table-cleanup/ \
   --video_backend torchvision_av \
   --modality_keys single_arm gripper
```
其中<YOUR_CHECKPOINT_PATH>需要替换为训练完成的checkpoint路径（例如/DrivingSDK/model_examples/GR00T-N1.5/Isaac-GR00T/so101-checkpoints/checkpoint-1000/）