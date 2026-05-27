# GR00T-N1.6 for PyTorch

## 目录

- [GR00T-N1.6 for PyTorch](#gr00t-n16-for-pytorch)
  - [目录](#目录)
- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [支持任务列表](#支持任务列表)
- [准备训练环境](#准备训练环境)
  - [安装昇腾环境](#安装昇腾环境)
  - [安装模型环境](#安装模型环境)
- [准备数据集](#准备数据集)
  - [获取预训练权重](#获取预训练权重)
  - [准备数据集](#准备数据集-1)
- [快速开始](#快速开始)
  - [单机8卡训练](#单机8卡训练)
  - [单卡推理](#单卡推理)
- [版本说明](#版本说明)
  - [变更](#变更)
  - [FAQ](#faq)

# 简介

## 模型介绍

Isaac GR00T-N1.6 为 GR00T-N1.5升级版

- 参考实现：<https://github.com/NVIDIA/Isaac-GR00T/tree/main>

- 适配昇腾 AI 处理器的实现：<https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/GR00T-N1.6>

## 支持任务列表

本仓已经支持以下模型任务类型。如下列表中Released为Y的表示已经过测试验证，N的表示开发自验通过。

|    模型     | 任务列表 | 是否支持 | Released |
| :---------: | :------: | :------: | :------: |
| GR00T-N1.6 |   SFT训练   |    ✔     |    N     |

# 准备训练环境

## 安装昇腾环境

请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表1中软件版本。

**表 1**  昇腾软件版本支持表

|     软件类型      | 首次支持版本 |
| :---------------: | :------: |
| FrameworkPTAdapter | 26.0.0  |
|       CANN        | 9.0.0  |

## 安装模型环境

当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

**表 2**  版本支持表

|      三方库       |  支持版本  |
|:--------------:|:------:|
|    Python      | 3.10 |
|    PyTorch     |  2.7.1   |

0. 激活 CANN 环境

1. 创建环境

    创建conda环境

    ```sh
    conda create -n gr00t python=3.10
    conda activate gr00t
    ```

    参考原仓编译安装 Driving SDK 加速库：<https://gitcode.com/Ascend/DrivingSDK>

2. 准备模型源码，安装gr00t

    在 GR00T-N1.6根目录下，克隆原始仓，替换其中部分代码并安装

    ```sh
    git clone https://github.com/NVIDIA/Isaac-GR00T.git
    cd Isaac-GR00T
    git checkout e29d8fc50b0e4745120ae3fb72447986fe638aa6
    cp -f ../gr00t_n1d6.patch ./
    git apply --reject gr00t_n1d6.patch
    pip install -e .
    cp -f ../test/train* ./
    ```

3. 安装ffmpeg

    - 推荐基于conda安装

    ```sh
    # 安装ffmpeg
    conda install -c conda-forge ffmpeg=4.4.2
    export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH
    ```

    - 若采用源码安装，则步骤如下

    ```sh
    # 下载源码
    wget https://ffmpeg.org/releases/ffmpeg-4.4.2.tar.bz2
    tar -xvf ffmpeg-4.4.2.tar.bz2
    cd ffmpeg-4.4.2
    # 执行此步时环境中可能需要手动下载部分依赖包
    ./configure --prefix=/usr/local/ffmpeg --disable-doc --disable-openssl --enable-avresample --enable-demuxer=dash --enable-hardcoded-tables --enable-libfreetype --enable-libfontconfig --enable-libopenh264 --enable-gnutls --enable-libmp3lame --enable-libvpx --enable-pthreads --enable-gpl --enable-libx264 --enable-libx265 --enable-libaom --enable-libsvtav1 --enable-libxml2 --enable-pic --enable-shared --disable-static --enable-version3 --enable-zlib
    make -j 64
    make install
    cd ..

    # 编辑全局配置文件
    vim /etc/profile.d/ffmpeg.sh

    # 添加以下内容
    export PATH="/usr/local/ffmpeg/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/ffmpeg/lib:$LD_LIBRARY_PATH"

    # 使配置立即生效
    source /etc/profile
    # 运行命令后应正常输出相关配置等信息
    ffmpeg
    ```

4. 安装torchcodec

    - 推荐基于conda安装

    ```sh
    # 安装pybind11
    conda install -c conda-forge pybind11
    # 源码安装torchcodec
    git clone https://github.com/meta-pytorch/torchcodec.git
    cd torchcodec
    git checkout v0.5.0
    pip install -e . --no-build-isolation
    cd ..
    ```

    - 若采用源码安装，则步骤如下

    ```sh
    # 源码安装pybind11
    git clone https://github.com/pybind/pybind11.git
    cd pybind11
    mkdir build && cd build
    cmake .. -DPYBIND11_TEST=OFF -DCMAKE_INSTALL_PREFIX=/usr/local
    make -j$(nproc)
    make install
    cd ..
    # 源码安装torchcodec
    git clone https://github.com/meta-pytorch/torchcodec.git
    cd torchcodec
    git checkout v0.5.0
    pip install -e . --no-build-isolation
    cd ..
    ```

5. 安装decord（用于推理阶段）

    ```sh
    # 安装decord
    git clone --recursive https://github.com/dmlc/decord --depth 1
    cd decord
    mkdir build && cd build
    cmake ..  -DCMAKE_BUILD_TYPE=Release -DFFMPEG_DIR:PATH=$CONDA_PREFIX
    make
    # 编译whl包
    cd ../python
    python setup.py sdist bdist_wheel
    cd ../..
    pip install decord/python/dist/decord-0.6.0-cp310-cp310-linux_aarch64.whl
    ```

6. 安装triton-ascend（用于推理阶段）

    ```sh
    # 通过pip安装Triton-Ascend的最新稳定版本
    pip install triton-ascend==3.2.0
    ```

    注意：triton-ascend 3.2.0 及以下 Triton-Ascend和Triton 不能同时存在。需要先卸载社区 Triton，再安装 Triton-Ascend, 详细参考triton-ascend原仓安装介绍: <https://gitcode.com/Ascend/triton-ascend/blob/main/docs/zh/installation_guide.md>

7. 安装git-lfs（用于推理阶段）

    ```sh
    # 通过源码安装git-lfs
    wget --no-check-certificate https://github.com/git-lfs/git-lfs/releases/download/v3.6.1/git-lfs-linux-arm64-v3.6.1.tar.gz
    tar -zxf git-lfs-linux-arm64-v3.6.1.tar.gz
    cd git-lfs-3.6.1
    cp git-lfs /usr/bin/
    git lfs install
    git lfs version
    ```

# 准备数据集

## 获取预训练权重

下载权重至Isaac-GR00T/GR00T-N1.6-3B，Huggingface链接: [GR00T-N1.6-3B](https://huggingface.co/nvidia/GR00T-N1.6-3B)

```sh
pip install huggingface-hub
hf download nvidia/GR00T-N1.6-3B --local-dir ./GR00T-N1.6-3B
```

## 准备数据集

- 以LIBERO 10微调为例，安装数据集

  ```sh
  huggingface-cli download \
      --repo-type dataset IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot \
      --local-dir examples/LIBERO/libero_10_no_noops_1.0.0_lerobot/

  cp -r examples/LIBERO/modality.json examples/LIBERO/libero_10_no_noops_1.0.0_lerobot/meta/
  ```

- 以gr1.PickNPlace的推理为例，安装数据集

  ```sh
  cd /home/workspace/DrivingSDK/model_examples/GR00T-N1.6/Isaac-GR00T
  git lfs pull
  ```

# 快速开始

## 单机8卡训练

需先进入Isaac-GR00T目录

```sh
cd Isaac-GR00T
```

**训练脚本**

```sh
bash train_8p.sh --num_gpus=8 --global_batch_size=640 --max_steps=20000 --dataset_path=examples/LIBERO/libero_10_no_noops_1.0.0_lerobot/ --base_model_path=./GR00T-N1.6-3B
```

**性能测试脚本**

```sh
bash train_performance_8p.sh --num_gpus=8 --global_batch_size=640 --max_steps=1000 --dataset_path=examples/LIBERO/libero_10_no_noops_1.0.0_lerobot/ --base_model_path=./GR00T-N1.6-3B
```

**训练结果展示**

**表 3**  训练结果展示表

|     芯片      | 卡数 | global batch size | max steps | Final loss | FPS  |
| :-----------: | :--: | :---------------: | :---: | :--------------------: | :--------------------|
|     竞品A     |  8p  |         640       |  20000 |  0.0084 |  457  |
| Atlas 800T A2 |  8p  |         640       |  20000 |  0.0082 |  449  |

## 单卡推理

需先进入Isaac-GR00T目录

```sh
cd Isaac-GR00T
```

**推理脚本**

```sh
# 使用GR00T N1.6官方的base Model进行推理，GR00T-N1.6-3B
taskset -c 0-7 python scripts/deployment/standalone_inference_script.py --model-path ./GR00T-N1.6-3B --dataset-path demo_data/gr1.PickNPlace --embodiment-tag GR1 --traj-ids 0 1 2 --inference-mode pytorch --action-horizon 8 --video_backend decord
```

**推理结果展示**

**表 4**  推理结果展示表

|     芯片      | 卡数 | 数据集 | Average MSE | Average MAE | Avg time/step(s) |  SPS  |
| :-----------: | :--: | :----- | :---------: | :---------: | :--------------: | :---: |
|     竞品A     |  1p  |  ...   |   1.179     |   0.673     |      0.147       | 6.80  |
| Atlas 800T A2 |  1p  |  ...   |   1.185     |   0.674     |      0.145       | 6.90  |

# 版本说明

## 变更

2026.5.14: 适配一键Patcher 2.0。

2026.5.7: 新增推理适配。

2026.4.3: 首次发布。

## FAQ

Q: 在无法访问 Hugging Face hub 的情况下运行模型报错？

A: 用户可以前往官网或使用 Hugging Face 镜像源在有网络的情况下自主下载。

Q: 若运行过程中出现torchcodec相关报错，如decoder等？

A：可能是受到环境内系统原有ffmpeg的影响，需将原有的ffmpeg目录更名（如`mv ffmpeg ffmpeg_bak`）来避免冲突，从而确保只依赖于conda版本，随后可重新编译安装torchcodec

Q: 运行过程中性能劣化很明显，特别是日志中的shard dataset部分耗时较高？

A: 可能是ffmpeg的配置问题，建议检查环境内的ffmpeg是否为当前README推荐的方式安装所得。

Q: 若推理过程中出现`/lib/python3.10/site-packages/torch/utils/_triton.py`文件中cuda版本校验失败？

A：torch.compile 在 NPU 上运行时， `torch._dynamo` 会调用 `_triton.py` 中的函数检查 CUDA 设备能力，但 NPU 环境下 CUDA 相关接口返回 None ，导致 None >= (9, 0) 和 None >= 7 的类型比较错误。可尝试注释该校验，或者参考下面的修改:

```python
 _cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
        if (
            _cap is not None
            and _cap >= (9, 0)
            and not torch.version.hip
        ):
```
