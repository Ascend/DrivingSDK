# Cosmos-Transfer1 for PyTorch

## 目录

- [Cosmos-Transfer1 for PyTorch](#Cosmos-Transfer1-for-pytorch)
  - [目录](#目录)
- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [支持任务列表](#支持任务列表)
  - [代码实现](#代码实现)
- [Cosmos-Transfer1](#Cosmos-Transfer1)
  - [准备训练环境](#准备训练环境)
    - [安装昇腾环境](#安装昇腾环境)
    - [准备模型权重](#准备模型权重)
    - [准备数据集](#准备数据集)
  - [快速开始](#快速开始)
    - [训练任务](#训练任务)
      - [开始训练](#开始训练)
      - [训练结果](#训练结果)
    - [推理任务](#推理任务)
      - [开始推理](#开始推理)
- [变更说明](#变更说明)
- [FAQ](#faq)

# 简介

## 模型介绍

Cosmos-Transfer1是一个支持多模态条件控制的视频生成模型，可基于分割、深度、边缘、LiDAR等多种输入生成高质量视频。支持单模态或组合式多模态控制，并结合文本与图像提示。包含4K上采样功能及预/后训练脚本，便于定制和优化，适用于自动驾驶等物理AI场景的仿真模拟。

## 支持任务列表

本仓已经支持以下模型任务类型

|    模型     | 任务列表 | 是否支持 |
| :---------: | :------: | :------: |
| Cosmos-Transfer1-7B |   训练   |    ✔     |
| Cosmos-Transfer1-7B |   推理   |    ✔     |
| Cosmos-Transfer1-7B-Sample-AV |   训练   |    ✔     |
| Cosmos-Transfer1-7B-Sample-AV |   推理   |    ✔     |
| Cosmos-Transfer1-7B-Sample-AV-Single2MultiView |   训练   |    ✔     |
| Cosmos-Transfer1-7B-Sample-AV-Single2MultiView |   推理   |    ✔     |
| Cosmos-Transfer1-7B-4KUpscaler |   推理   |    ✔     |

## 代码实现

- 参考实现：
  
  ```
  url=https://github.com/nvidia-cosmos/cosmos-transfer1
  commit_id=fc6376b0315e0f915a5349e09c1ad622a974c99f 
  ```

- 适配昇腾 AI 处理器的实现：
    ```
    url=https://gitcode.com/Ascend/DrivingSDK.git
    code_path=model_examples/Cosmos-Transfer1
    ```

# Cosmos-Transfer1

## 准备训练环境

### 安装环境

**表 1**  三方库版本支持表

| 三方库  | 支持版本 |
| :-----: | :------: |
| PyTorch |   2.7.1   |

### 安装昇腾环境

请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表2中软件版本。

**表 2**  昇腾软件版本支持表

|     软件类型      | 支持版本 |
| :---------------: | :------: |
| FrameworkPTAdapter | 7.2.0  |
|       CANN        | 8.3.RC1  |
|       Python        | 3.10  |

1. 安装tbe
    
    ```
    # 安装tbe和hccl, 将 CANN 包目录记作 cann_root_dir
    pip uninstall te topi hccl -y
    pip install {cann_root_dir}/latest/lib64/te-*-py3-none-any.whl
    pip install {cann_root_dir}/latest/lib64/hccl-*-py3-none-any.whl
    ```

2. 安装Driving SDK

    请参考昇腾[Driving SDK](https://gitcode.com/Ascend/DrivingSDK)代码仓说明编译安装Driving SDK


3. 源码安装decord
    ```
    # 源码编译ffmpeg
    wget https://ffmpeg.org/releases/ffmpeg-4.4.2.tar.bz2 --no-check-certificate
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

    # 编译whl包
    cd ../python
    python setup.py sdist bdist_wheel

    # 安装对应whl包
    cd ..
    pip install python/dist/decord-0.6.0-cp310-cp310-linux_aarch64.whl
    cd ..
    ```
    
4. 安装apex
    ```
    # 下载适配源码
    git clone https://gitee.com/ascend/apex.git
    cd apex/
    bash scripts/build.sh --python=3.10

    # 安装apex
    pip install apex/dist/apex-0.1+ascend-{version}.whl # version为python版本和cpu架构
    cd ..
    ```

5. 克隆代码仓到当前目录并使用 patch 文件：
    ```
    git clone https://github.com/nvidia-cosmos/cosmos-transfer1.git
    cd cosmos-transfer1
    git checkout fc6376b0315e0f915a5349e09c1ad622a974c99f
    cp -f ../Cosmos_transfer1.patch .
    git apply --reject --whitespace=fix Cosmos_transfer1.patch
    cp -rf ../test .
    cp -rf ../tools/patch.py ./cosmos_transfer1/utils/
    ```

    将模型根目录记作 `model_root_path`


6. 安装其他依赖项
    ```
    pip install -r requirements.txt
    # 在安装之后，需检查torch及torchvision版本，若版本被覆盖，需再次安装torch及torchvision(0.22.1)
    cd ..
    ```

7. vllm和vllm-asend安装
    ```
    # 下载vllm
    git clone https://github.com/vllm-project/vllm.git
    cd vllm
    git checkout 5bc1ad6
    cd ../

    # 下载vllm-ascend
    git clone https://github.com/vllm-project/vllm-ascend.git
    cd vllm-ascend
    git checkout 75c10ce
    cd ../


    # 安装VLLM
    cd vllm
    VLLM_TARGET_DEVICE=empty pip install -v -e .
    cd ..

    # VLLM安装可能会升级numpy版本，numpy版本要求为1.26.4
    pip install numpy==1.26.4

    # 安装VLLM-ASCEND，需导入CANN
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    cd vllm-ascend

    # 关闭编译
    export COMPILE_CUSTOM_KERNELS=0

    # 安装依赖包
    pip install setuptools_scm pybind11 cmake msgpack numba quart

    # 安装vllm-ascend，由于之前安装了torchvision，需要把requirements.txt中的torchvision<0.21.0注释掉
    python setup.py develop

    # vllm-ascend源码安装过程中遇到相关依赖包因网络问题安装不成功，可以先尝试pip install xxx安装对应失败的依赖包，再执行上一句命令
    cd ..

    # 在安装完VLLM及VLLM-ASCEND后，需检查torch、torch_npu、torchvision、transformers版本，若版本被覆盖，需再次安装
    pip install torch-2.7.1-xxx.whl
    pip install torch_npu-2.7.1-xxx.whl
    pip install torchvision==0.22.1 transformers==4.51.0
    ```

### 准备模型权重

- 根据原仓**inference_cosmos_transfer1_7b**部分准备权重到${model_root_path}/checkpoints，目录结构如下
    ```bash
    ${model_root_path}/checkpoints/
    ├── depth-anything
    │   └── Depth-Anything-V2-Small-hf
    ├── facebook
    │   └── sam2-hiera-large
    ├── google-t5
    │   └── t5-11b
    ├── IDEA-Research
    │   └── grounding-dino-tiny
    ├── meta-llama
    │   └── Llama-Guard-3-8B
    └── nvidia
        ├── Cosmos-Guardrail1
        ├── Cosmos-Tokenize1-CV8x8x8-720p
        ├── Cosmos-Transfer1-7B
        ├── Cosmos-Transfer1-7B-Sample-AV
        ├── Cosmos-Transfer1-7B-Sample-AV-Single2MultiView
        └── Cosmos-UpsamplePrompt1-12B-Transfer
    ```



  1. 生成一个[Hugging Face](https://huggingface.co/settings/tokens)访问令牌，将访问令牌设置为'Read'权限。

  2. 使用该令牌登录Hugging Face
    ```
    huggingface-cli login
    ```

  3. 获取[Llama-Guard-3-8B](https://huggingface.co/meta-llama/Llama-Guard-3-8B)的访问权限

  4. 从Hugging Face上下载Cosmos模型的权重
    ```
    cd ${model_root_path}
    PYTHONPATH=$(pwd) python scripts/download_checkpoints.py --output_dir checkpoints/
    ```
  

### 准备数据集

- 根据原仓**cosmos-transfer1/examples/training_cosmos_transfer_7b**部分准备[HD-VILA-100M](https://github.com/microsoft/XPretrain/tree/main/hd-vila-100m)的子集，目录及结构如下：

    ```bash
    ${model_root_path}/datasets/hdvila/
    ├── metas/
    │   ├── *.json
    │   ├── *.txt
    ├── videos/
    │   ├── *.mp4
    ├── t5_xxl/
    │   ├── *.pickle
    ├── keypoint/
    │   ├── *.pickle
    ├── depth/
    │   ├── *.mp4
    ├── seg/
    │   ├── *.pickle
    └── <your control input modality>/
        ├── <your files>
    ```

    1. 准备Videos和Captions

    ```
    # Download metadata with video urls and captions
    mkdir -p datasets/hdvila
    cd datasets/hdvila
    wget https://huggingface.co/datasets/TempoFunk/hdvila-100M/resolve/main/hdvila-100M.jsonl

    # download the sample videos used for training
    pip install pytubefix ffmpeg

    # downlaod the original HD-VILA-100M videos, save the corresponding clips, the captions and the metadata.
    PYTHONPATH=$(pwd) python scripts/download_diffusion_example_data.py --dataset_path datasets/hdvila --N_videos 128 --do_download --do_clip
    ```
    
    2. 计算T5 Text Embedding
    ```
    # The script will read the captions, save the T5-XXL embeddings in pickle format.
    PYTHONPATH=$(pwd) python scripts/get_t5_embeddings.py --dataset_path datasets/hdvila
    ```


## 快速开始

### 训练任务-cosmos_transfer_7b

本任务目前主要提供cosmos_transfer_7b模型的单机8卡训练

#### 开始训练

1. 将模型Checkpoints拆分为TensorParallel Checkpoints
    ```
    cd ${model_root_path}
    # 将 Base model checkpoint 拆分成8个 TP checkpoints
    PYTHONPATH=. python scripts/convert_ckpt_fsdp_to_tp.py checkpoints/nvidia/Cosmos-Transfer1-7B/base_model.pt
    # EdgeControl checkpoint 拆分进行post-train
    PYTHONPATH=. python scripts/convert_ckpt_fsdp_to_tp.py checkpoints/nvidia/Cosmos-Transfer1-7B/edge_control.pt
    ```

2. 在模型根目录下，运行训练脚本。

   - 单机8卡精度训练
   
   ```
   # 单机8卡训练
   bash test/train_cosmos_transfer_7b.sh
   ```

#### 训练结果

| 芯片          | 卡数 | TP_Size | Precision | Loss | 性能-单步迭代耗时(s) |
| ------------- | :--: | :---------------: | :-------------------: | :-------------------: |:-------------------: |
| 竞品A           |  8p  |       8         |   混精    | -0.4805 | 8.93 |
| Atlas 800T A2 |  8p  |        8         |   混精    | -0.4570 | 10.34 |


### 训练任务-cosmos_transfer_7b_sample_AV
本任务目前主要提供cosmos_transfer_7b_sample_AV模型的单机8卡训练

#### 开始训练

1. 将模型Checkpoints拆分为TensorParallel Checkpoints
    ```
    cd ${model_root_path}
    # 将 Base model checkpoint 拆分成8个 TP checkpoints
    PYTHONPATH=. python scripts/convert_ckpt_fsdp_to_tp.py checkpoints/nvidia/Cosmos-Transfer1-7B-Sample-AV/t2w_base_model.pt
    # LidarControl checkpoint 拆分进行post-train
    PYTHONPATH=. python scripts/convert_ckpt_fsdp_to_tp.py checkpoints/nvidia/Cosmos-Transfer1-7B-Sample-AV/t2w_lidar_control.pt
    ```

2. 在模型根目录下，运行训练脚本。

   - 单机8卡精度训练
   
   ```
   # 单机8卡训练
   bash test/train_cosmos_transfer_7b_sample_AV.sh
   ```

#### 训练结果

| 芯片          | 卡数 | TP_Size | Precision | Loss | 性能-单步迭代耗时(s) |
| ------------- | :--: | :---------------: | :-------------------: | :-------------------: |:-------------------: |
| 竞品A           |  8p  |       8         |   混精    | 0.4707 | 10.91 |
| Atlas 800T A2 |  8p  |        8         |   混精    | 0.4648 | 11.13 |


### 训练任务-cosmos_transfer1_7b_sample_AV_single2multiview
本任务目前主要提供cosmos_transfer1_7b_sample_AV_single2multiview模型的单机8卡训练

#### 开始训练

1. 将模型Checkpoints拆分为TensorParallel Checkpoints
    ```
    cd ${model_root_path}
    # 将 Base model checkpoint 拆分成8个 TP checkpoints
    PYTHONPATH=. python scripts/convert_ckpt_fsdp_to_tp.py checkpoints/nvidia/Cosmos-Transfer1-7B-Sample-AV-Single2MultiView/t2w_base_model.pt
    # HDMapControl checkpoint 拆分进行post-train
    PYTHONPATH=. python scripts/convert_ckpt_fsdp_to_tp.py checkpoints/nvidia/Cosmos-Transfer1-7B-Sample-AV-Single2MultiView/t2w_hdmap_control.pt
    ```

2. 在模型根目录下，运行训练脚本。

   - 单机8卡精度训练
   
   ```
   # 单机8卡训练
   bash test/train_cosmos_transfer_7b_sample_AV_single2multiview.sh
   ```

#### 训练结果

| 芯片          | 卡数 | TP_Size | Precision | Loss | 性能-单步迭代耗时(s) |
| ------------- | :--: | :---------------: | :-------------------: | :-------------------: |:-------------------: |
| 竞品A           |  8p  |       8         |   混精    | 0.0608 | 23.02 |
| Atlas 800T A2 |  8p  |        8         |   混精    | 0.0757 | 20.90|


### 推理任务-inference_cosmos_transfer1_7b

本任务目前主要提供单卡和多卡的推理

#### 推理-单模态控制 (Edge)
1. 在模型根目录{model_root_path}下，运行推理指令。

   - 单卡推理

   ```
   bash test/inference_cosmos_transfer_7b.sh 1
   ```

   - 多卡推理
   ```
   bash test/inference_cosmos_transfer_7b.sh 4
   ```


#### 推理-蒸馏模型单模态控制 (Edge)

1. 在模型根目录{model_root_path}下，运行推理指令。

   - 单卡推理

   ```
   bash test/inference_cosmos_transfer_7b_distilled.sh
   ```

#### 推理-多模态控制
1. 在模型根目录{model_root_path}下，运行推理指令。

   - 单卡推理

   ```
   bash test/inference_cosmos_transfer_7b_uniform.sh
   ```


### 推理任务-inference_cosmos_transfer1_7b_sample_av
本任务目前主要提供单卡和多卡的推理
1. 在模型根目录{model_root_path}下，运行推理指令。

   - 单卡推理

   ```
   bash test/inference_cosmos_transfer1_7b_sample_av.sh 1
   ```

   - 多卡推理
   ```
   bash test/inference_cosmos_transfer1_7b_sample_av.sh 4
   ```




### 推理任务-inference_cosmos_transfer1_7b_sample_av_single2multiview

本任务目前主要提供单卡推理

1. 在模型根目录{model_root_path}下，运行推理指令。

   - 单卡推理

   ```
   bash test/inference_cosmos_transfer1_7b_sample_av_single2multiview.sh
   ```

### 推理任务-inference_cosmos_transfer1_7b_4kupscaler
本任务目前主要提供单卡和多卡的推理
1. 在模型根目录{model_root_path}下，运行推理指令。

   - 单卡推理

   ```
   bash test/inference_cosmos_transfer1_7b_4kupscaler.sh 1
   ```

   - 多卡推理
   ```
   bash test/inference_cosmos_transfer1_7b_4kupscaler.sh 4
   ```



# 变更说明

2025.10.30：首次发布


# FAQ

1. 镜像中可能没有装GL库导致报错，需要安装对应的GL库

```
yum install libglvnd-glx
```