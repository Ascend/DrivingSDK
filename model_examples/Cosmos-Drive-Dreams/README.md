# Cosmos-Drive-Dreams

## 目录

- [Cosmos-Drive-Dreams](#cosmos-drive-dreams)
  - [目录](#目录)
- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [支持任务列表](#支持任务列表)
  - [代码实现](#代码实现)
  - [准备环境](#准备环境)
    - [安装环境](#安装环境)
    - [安装昇腾环境](#安装昇腾环境)
    - [准备模型权重](#准备模型权重)
  - [快速开始](#快速开始)
    - [推理任务](#推理任务)
- [变更说明](#变更说明)
- [FAQ](#faq)

# 简介

## 模型介绍

Cosmos-Drive-Dreams 是一个基于 Cosmos 世界基础模型构建的合成数据生成（SDG）流水线，专为可基于 HD 地图、LiDAR 点云、世界场景等多模态条件生成自动驾驶场景的高质量合成数据。支持单视图 / 多视图视频生成、LiDAR 数据生成等任务，结合文本提示实现场景定制，包含数据集可视化、轨迹编辑、格式转换等工具套件及预训练 / 后训练脚本，适用于自动驾驶算法的训练与测试场景仿真。

## 支持任务列表

本仓已经支持以下模型任务类型

|        模型         | 任务列表 | 是否支持 |
| :-----------------: | :------: | :------: |
| Cosmos-Drive-Dreams |   推理   |    ✔     |


## 代码实现

- 参考实现：

  ```
  url=https://github.com/nv-tlabs/Cosmos-Drive-Dreams
  commit_id=b9a156e23e894517e433bb28b5074de7ac8e1614
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitcode.com/Ascend/DrivingSDK.git
  code_path=model_examples/Cosmos-Drive-Dreams
  ```

## 准备环境

### 安装环境

**表 1**  三方库版本支持表

| 三方库  | 支持版本 |
| :-----: | :------: |
| PyTorch |  2.7.1   |

### 安装昇腾环境

请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表2中软件版本。

**表 2**  昇腾软件版本支持表

|      软件类型      | 首次支持版本 |
| :----------------: | :------: |
| FrameworkPTAdapter |  7.2.0   |
|        CANN        | 8.5.0  |
|       Python       |   3.10   |

1. 克隆代码仓到当前目录并使用 patch 文件：

   ```
   git clone https://github.com/nv-tlabs/Cosmos-Drive-Dreams
   cd Cosmos-Drive-Dreams
   git checkout b9a156e23e894517e433bb28b5074de7ac8e1614
   cp -f ../Cosmos_Drive_Dreams.patch .
   git apply --reject --whitespace=fix Cosmos_Drive_Dreams.patch
   cp -rf ../test .
   cd cosmos-transfer1
   git checkout b25a3ac8efe277931c62b4e23a5437afb468ef0c
   cp -f ../../Cosmos_Transfer1.patch .
   git apply --reject --whitespace=fix Cosmos_Transfer1.patch
   cd ..
   ```

   将模型根目录记作 model_root_path

2. 安装依赖项

   ```
   pip install -r requirements.txt
   ```

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

5. vllm和vllm-asend安装

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
   ```

   ```
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
   pip install setuptools_scm pybind11 cmake msgpack numba quart
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

      huggingface-cli login

   3. 获取[Llama-Guard-3-8B](https://huggingface.co/meta-llama/Llama-Guard-3-8B)的访问权限

   4. 从Hugging Face上下载Cosmos模型的权重

      cd ${model_root_path}
      PYTHONPATH=$(pwd) python scripts/download_checkpoints.py --output_dir checkpoints/



## 快速开始

### 推理任务

在模型根目录下，运行推理指令

```
bash test/inference_pipeline.sh
```


# 变更说明

2025.11.12：首次发布


# FAQ

1. 镜像中可能没有装yaml库导致apex安装时报错，需要安装对应的yaml库

```
pip install pyyaml
```

2. 在运行render_from_rds_hq.py脚本时，如果出现报错：`Failed to register worker to Raylet: IOError: Failed to read data from the socket: End of file worker_id=01000000ffffffffffffffffffffffffffffffffffffffffffffffff`是由于asyncio相关依赖库冲突导致，可尝试调整相关依赖库版本解决：

```
pip uninstall aiohttp-cors
pip install aiofiles==24.1.0 aiohttp==3.12.15
pip install tornado==6.5.1
```