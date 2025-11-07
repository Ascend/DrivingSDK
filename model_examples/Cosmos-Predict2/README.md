# Cosmos-Predict2 for PyTorch

## 目录

- [Cosmos-Predict2 for PyTorch](#cosmos-predict2-for-pytorch)
  - [目录](#目录)
- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [支持任务列表](#支持任务列表)
  - [代码实现](#代码实现)
- [准备训练环境](#准备训练环境)
  - [安装昇腾环境](#安装昇腾环境)
  - [安装模型环境](#安装模型环境)
- [准备训练数据](#准备训练数据)
  - [准备模型权重](#准备模型权重)
  - [准备数据集](#准备数据集)
- [快速开始](#快速开始)
  - [Video2World-14B](#video2world-14b)
  - [Video2World-2B](#video2world-2b)
  - [Text2Image-14B](#text2image-14b)
  - [Text2Image-2B](#text2image-2b)
  - [训练结果展示](#训练结果展示)
- [版本说明](#版本说明)
  - [变更](#变更)
  - [FAQ](#faq)

# 简介

## 模型介绍

Cosmos-Predict2 是 Cosmos 世界基础模型（WFMs）生态中专注于物理 AI 未来状态预测的关键分支，具备两大核心能力：Text-to-Image 生成（从文本描述创建高质量图像）和 Video-to-World 生成（从视频输入结合文本提示生成视觉仿真序列）。包含 0.6B/2B/14B 等多规模模型，支持单模态或文本与视觉结合的多模态输入。同时提供后训练等脚本，便于定制优化，适用于物理 AI 场景的未来视觉世界模拟。

## 支持任务列表

本仓已经支持以下模型任务类型

|    模型     | 任务列表 | 是否支持 |
| :---------: | :------: | :------: |
| Cosmos-Predict2-2B-Text2Image |   训练&推理   |    ✔     |
| Cosmos-Predict2-14B-Text2Image |   训练&推理   |    ✔     |
| Cosmos-Predict2-2B-Video2World |   训练&推理   |    ✔     |
| Cosmos-Predict2-14B-Video2World |   训练&推理   |    ✔     |

## 代码实现

- 参考实现：
  
  ```
  url=https://github.com/nvidia-cosmos/cosmos-predict2
  commit_id=ccb40411471d7e37cad7c8a4b4b9f7f088edbdf1 
  ```

- 适配昇腾 AI 处理器的实现：
    ```
    url=https://gitcode.com/Ascend/DrivingSDK.git
    code_path=model_examples/Cosmos-Predict2
    ```

# 准备训练环境

## 安装昇腾环境

请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/Pytorch/720/configandinstg/instg/insg_0001.html)》文档搭建昇腾环境，本仓已支持表1中软件版本。

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

1. 安装Driving SDK

    请参考昇腾[Driving SDK](https://gitcode.com/Ascend/DrivingSDK)代码仓说明编译安装Driving SDK


2. 源码安装decord

    ```sh
    # 源码编译ffmpeg
    wget https://ffmpeg.org/releases/ffmpeg-4.4.2.tar.bz2
    tar -xvf ffmpeg-4.4.2.tar.bz2
    cd ffmpeg-4.4.2
    ./configure --enable-shared  --prefix=/usr/local/ffmpeg    # --enable-shared is needed for sharing libavcodec with decord
    make -j 64
    make install
    cd ..

    # 源码编译decord
    git clone  --recursive https://github.com/dmlc/decord --depth 1
    cd decord
    mkdir build && cd build
    cmake ..  -DCMAKE_BUILD_TYPE=Release -DFFMPEG_DIR:PATH="/usr/local/ffmpeg/"
    make

    # 编译whl包并安装
    cd ../python
    python setup.py sdist bdist_wheel
    cd ../..
    pip install decord/python/dist/decord-0.6.0-cp310-cp310-linux_aarch64.whl
    ```


3. 安装apex
    
    ```sh
    # 下载适配源码
    git clone https://gitcode.com/Ascend/apex.git
    cd apex/

    # 编译apex的二进制包
    pip install pyyaml
    bash scripts/build.sh --python=3.10

    # 安装apex
    pip install apex/dist/apex-0.1+ascend-cp310-cp310-linux_aarch64.whl
    cd ..
    ```

4. 准备模型源码

    在 Cosmos-Predict2 根目录下，克隆原始仓，使用patch文件替换其中部分代码并安装
    ```sh
    git clone https://github.com/nvidia-cosmos/cosmos-predict2.git
    cd cosmos-predict2
    git checkout ccb40411471d7e37cad7c8a4b4b9f7f088edbdf1
    cp -f ../cosmos_predict2.patch .
    git apply --reject --whitespace=fix cosmos_predict2.patch
    cp -f ../test/train* ./
    cp -f ../tools/patch.py ./scripts/
    ```


5. 安装其他依赖项
    
    ```sh
    pip install -r requirements.txt
    pip install opencv-python-headless==4.12.0.88
    pip install cosmos-guardrail==0.1.0 --no-deps
    pip install numpy==1.26.4
    # 在安装之后，需检查torch及torchvision版本，若版本被覆盖，需再次安装torch及torchvision==0.22.1
    cd ..
    ```

# 准备训练数据

1. 生成一个[Hugging Face](https://huggingface.co/settings/tokens)访问令牌，将访问令牌设置为 'Read' 权限

2. 使用该令牌登录Hugging Face
    
    ```sh
    huggingface-cli login
    ```

## 准备模型权重

下载权重至 Cosmos-Predict2/checkpoints/nvidia/

1. 处理数据集所需模型（必要）

    ```sh
    # 处理数据集
    hf download google-t5/t5-11b --local-dir ./checkpoints/google-t5/t5-11b --exclude "tf_model.h5"
    ```

2. Cosmos-Predict2系列模型（按需下载对应模型即可，非必要全下载）
    
    ```sh
    # 根据需求下载模型权重
    # Video2World-14B
    hf download nvidia/Cosmos-Predict2-14B-Video2World  --local-dir ./checkpoints/nvidia/Cosmos-Predict2-14B-Video2World
    
    # Video2World-2B
    hf download nvidia/Cosmos-Predict2-2B-Video2World  --local-dir ./checkpoints/nvidia/Cosmos-Predict2-2B-Video2World
    
    # Text2Image-14B
    hf download nvidia/Cosmos-Predict2-14B-Text2Image  --local-dir ./checkpoints/nvidia/Cosmos-Predict2-14B-Text2Image
    
    # Text2Image-2B
    hf download nvidia/Cosmos-Predict2-2B-Text2Image  --local-dir ./checkpoints/nvidia/Cosmos-Predict2-2B-Text2Image

    # 安全检查模型（仅推理时使用，且可以选择不启用）
    hf download nvidia/Cosmos-1.0-Guardrail --local-dir ./checkpoints/nvidia/Cosmos-1.0-Guardrail
    ```

    若全下载，则模型权重目录结构如下

    ```sh
    ${model_root_path}/checkpoints/
    ├── google-t5
    │   └── t5-11b
    └── nvidia
        ├── Cosmos-1.0-Guardrail
        ├── Cosmos-Predict2-2B-Text2Image
        ├── Cosmos-Predict2-2B-Video2World
        ├── Cosmos-Predict2-14B-Text2Image
        └── Cosmos-Predict2-14B-Video2World
    ```

    注：官方代码提供一键安装脚本，会下载多余附属模型，对网络要求较高，效果与逐个`hf download`一致。

    ```sh
    # 一键安装脚本，下载Cosmos-Predict2-14B-Video2World、t5-11b等模型
    python ./scripts/download_checkpoints.py --model_types "video2world" --model_sizes "14B"
    
    # 再补充安全检查模型
    hf download nvidia/Cosmos-1.0-Guardrail --local-dir ./checkpoints/nvidia/Cosmos-1.0-Guardrail
    ```

## 准备数据集

下载数据集至 Cosmos-Predict2/cosmos-predict2/datasets，Huggingface链接：[Cosmos-NeMo-Assets](https://huggingface.co/datasets/nvidia/Cosmos-NeMo-Assets)

    ```sh
    mkdir -p datasets/cosmos_nemo_assets/
    hf download nvidia/Cosmos-NeMo-Assets --repo-type dataset --local-dir datasets/cosmos_nemo_assets/ --include "*.mp4*"
    mv datasets/cosmos_nemo_assets/nemo_diffusion_example_data datasets/cosmos_nemo_assets/videos
    ```

数据集预处理:

1. Video2World

    运行预处理脚本

    ```sh
    python scripts/get_t5_embeddings_from_cosmos_nemo_assets.py --dataset_path datasets/cosmos_nemo_assets --prompt "A video of sks teal robot."
    ```

    Video2World 数据集目录结构如下

    ```sh
    datasets/cosmos_nemo_assets/
    ├── metas/
    │   ├── *.txt
    ├── videos/
    │   ├── *.mp4
    ├── t5_xxl/
    │   ├── *.pickle
    ```

2. Text2Image

    运行预处理脚本

    ```sh
    python scripts/extract_images_from_videos.py --input_dataset_dir datasets/cosmos_nemo_assets --output_dataset_dir datasets/cosmos_nemo_assets_images --stride 30
    python scripts/get_t5_embeddings_from_cosmos_nemo_assets.py --dataset_path datasets/cosmos_nemo_assets_images --prompt "An image of sks teal robot." --is_image
    ```
    Text2Image 数据集目录结构如下

    ```sh
    datasets/cosmos_nemo_assets_images/
    ├── metas/
    │   ├── *.txt
    ├── images/
    │   ├── *.jpg
    ├── t5_xxl/
    │   ├── *.pickle
    ```


# 快速开始

## Video2World-14B

* 双机16卡训练
    ```sh
    # 主节点 Master
    bash train.sh --EXP=predict2_video2world_training_14b_cosmos_nemo_assets --nproc_per_node=8 --master_addr=<Master_IP> --nnodes=2 --master_port=<Master_PORT> --hccl_if_ip=<Master_IP> --node_rank=0
    # 从节点 Worker
    bash train.sh --EXP=predict2_video2world_training_14b_cosmos_nemo_assets --nproc_per_node=8 --master_addr=<Master_IP> --nnodes=2 --master_port=<Master_PORT> --hccl_if_ip=<Worker_IP> --node_rank=1
    ```

* 单卡推理
    ```sh
    python ./scripts/hf_video2world.py output/hf_video2world14b --prompt "assets/video2world/example_prompt.txt" --image "assets/video2world/example_input.jpg" -v --model "./checkpoints/nvidia/Cosmos-Predict2-14B-Video2World"
    ```

## Video2World-2B

* 单机8卡训练
    ```sh
    bash train.sh --EXP=predict2_video2world_training_2b_cosmos_nemo_assets --nproc_per_node=8
    ```

* 单卡推理
    ```sh
    python ./scripts/hf_video2world.py output/hf_video2world2b --prompt "assets/video2world/example_prompt.txt" --image "assets/video2world/example_input.jpg" -v --model "./checkpoints/nvidia/Cosmos-Predict2-2B-Video2World"
    ```

## Text2Image-14B

* 单机8卡训练
    ```sh
    bash train.sh --EXP=predict2_text2image_training_14b_cosmos_nemo_assets --nproc_per_node=8
    ```

* 单卡推理
    ```sh
    python ./scripts/hf_text2image.py output/hf_text2image14b --prompt "assets/text2image/example_prompt.txt" -v --model "./checkpoints/nvidia/Cosmos-Predict2-14B-Text2Image/"
    ```

## Text2Image-2B

* 单机8卡训练
    ```sh
    bash train.sh --EXP=predict2_text2image_training_2b_cosmos_nemo_assets --nproc_per_node=8 
    ```

* 单卡推理
    ```sh
    python ./scripts/hf_text2image.py output/hf_text2image2b --prompt "assets/text2image/example_prompt.txt" -v --model "./checkpoints/nvidia/Cosmos-Predict2-2B-Text2Image/"
    ```


## 训练结果展示

**表 3**  Video2World-14B 训练结果展示表

|     芯片      | 卡数 | global batch size | max steps | Final loss |
| :-----------: | :--: | :---------------: | :---: | :--------------------: |
|     竞品A     |  16p  |         16       |  500 |  0.0402 |  
| Atlas 800T A2 |  16p  |         16       |  500 |   0.0409 | 



# 版本说明

## 变更

2025.11.7：首次发布


## FAQ

Q：`hf download` 下载数据集过程中报错或速度过慢？

A：用户可以前往官网或使用 Hugging Face 镜像源在有网络的情况下自主下载，按照前面的结构组织文件即可。


Q：在各参数配置一致的情况下 NPU 训练结果与 GPU 差异较大？

A：可能是由于 EMA 模型初始化过程中随机数生成器状态不一致导致，NPU 对该过程进行了优化；原始逻辑中，EMA 模型 (`cosmos-predict2/cosmos_predict2/pipelines/video2world.py`中的`pipe.dit_ema`模型) 会先在 CPU 上随机初始化 (消耗大量随机数) 再被主模型预训练权重覆盖；而 NPU 版本优化后跳过该冗余步骤，直接加载预训练权重，加快训练启动速度。可对 GPU 也应用相同修改，统一随机数状态，且不影响训练效果。


Q：推理时如何启用安全检查？

A：运行脚本时设置`--use_safety_checker`为`True`即可，为能够单卡运行，脚本中已默认将安全检查模型设置为在 CPU 上加载和运行，用户可根据资源自行修改`CosmosSafetyChecker`类的内部参数。


Q：出现 decord 或 ffmpeg 相关依赖报错？

A：可能是环境变量未添加，在添加后若仍报错则尝试重新安装 decord：

```sh
# 编辑全局配置文件
vim /etc/profile.d/ffmpeg.sh

# 添加以下内容
export PATH="/usr/local/ffmpeg/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/ffmpeg/lib:$LD_LIBRARY_PATH"

# 使配置立即生效
source /etc/profile

# 若仍报错则重新编译 decord 并安装
```


Q：多机训练运行出现 hccl 报错且 error code 为1或7？

A：优先考虑进程残留没杀干净，可以参考如下命令终止服务器上其余进程：

```sh
pkill -9 python; pkill -9 torchrun
```