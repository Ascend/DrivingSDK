# OpenVLA

# 目录
- [OpenVLA](#OpenVLA)
- [目录](#目录)
- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [代码实现](#代码实现)
- [准备训练环境](#准备训练环境)
  - [安装昇腾环境](#安装昇腾环境)
  - [安装模型环境](#安装模型环境)
- [准备数据集](#准备数据集)
  - [获取训练数据集](#获取训练数据集)
  - [获取预训练权重](#获取预训练权重)
- [快速开始](#快速开始)
  - [微调模型](#微调模型)
  - [训练结果](#训练结果)
- [版本说明](#版本说明)
  - [变更](#变更)
  - [FAQ](#faq)


# 简介

## 模型介绍

OpenVLA 是一个 70 亿参数的开源视觉 - 语言 - 动作模型，基于 Open X-Embodiment 数据集的 97 万条机器人演示进行训练。它采用 DinoV2 和 SigLIP 双分支视觉编码器与 Llama 2 语言模型骨干，能将图像观察和语言指令映射为机器人控制动作，在多物体多任务场景中泛化性强，语言理解能力优异。本仓适配全参微调流程。

## 代码实现
- 参考实现：

  ```
  url=https://github.com/openvla/openvla
  commit_id=c8f03f48af692657d3060c19588038c7220e9af9
  ```
  
- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/DrivingSDK.git
  code_path=model_examples/OpenVLA
  ```

# 准备训练环境
## 安装昇腾环境
请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境。本仓已支持表1中软件版本。
  
  **表 1**  昇腾软件版本支持表

  |        软件类型        |   支持版本   |
  |:------------------:|:--------:|
  | FrameworkPTAdapter | 7.1.0  |
  |       CANN         | 8.2.RC1  |

## 安装模型环境

 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 2**  版本支持表

  |      三方库       |  支持版本  |
  |:--------------:|:------:|
  |    Python      | 3.10 |
  |    PyTorch     |  2.6.0   |

- 安装Driving SDK

  请参考昇腾[Driving SDK](https://gitee.com/ascend/DrivingSDK)代码仓说明编译安装Driving SDK

- 克隆代码仓到当前目录
  ```shell
  git clone https://gitee.com/ascend/DrivingSDK.git -b master
  ```
- 安装基础依赖

  在模型根目录下执行命令，安装模型需要的依赖
  
  ```shell
  conda create -n openvla python=3.10
  conda activate openvla
  cd DrivingSDK/model_examples/OpenVLA
  export VLA_HOME=`pwd`
  pip install -r requirements.txt
  ```
- 源码安装tensorflow-addons
  ```shell
  git clone https://github.com/tensorflow/addons.git
  cd addons
  git checkout d208d752e98c310280938efa939117bf635a60a8
  pip install -e .
  cd ..
  ```
- 模型代码使用Patch
  ```shell
  git clone https://github.com/openvla/openvla.git
  cp OpenVLA.patch openvla
  cp patch.py openvla/vla-scripts
  cd openvla
  git checkout c8f03f48af692657d3060c19588038c7220e9af9
  git apply OpenVLA.patch
  cp -rf ../test .
  mkdir logs
  pip install -e .
  ```

# 准备数据集

## 获取训练数据集
用户自行获取 *Bridge* 数据集
  ```shell
  cd $VLA_HOME
  mkdir datasets && cd datasets
  wget -r -nH --cut-dirs=4 --reject="index.html*" https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/
  mv bridge_dataset bridge_orig
  ```

## 获取预训练权重
下载预训练权重
  ```shell
  cd $VLA_HOME
  mkdir models && cd models
  git clone git@hf.co:openvla/openvla-7b-prismatic
  ```
如遇到网络问题，请到 [openvla-7b-prismatic](https://huggingface.co/openvla/openvla-7b-prismatic/tree/main) 手动下载文件到 `$VLA_HOME/model/openvla-7b-prismatic`

## 使用高性能内存库
安装tcmalloc（适用OS: __openEuler__）
```shell
mkdir gperftools
cd gperftools
wget https://github.com/gperftools/gperftools/releases/download/gperftools-2.16/gperftools-2.16.tar.gz --no-check-certificate
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
注意：需要安装OS对应tcmalloc版本（以下以 __Ubuntu__ 为例）
```shell
# 安装autoconf和libtool
apt-get update
apt install autoconf
apt install libtool
git clone https://github.com/libunwind/libunwind.git
cd libunwind
autoreconf -i
./configure --prefix=/usr/local
make -j128
make install
cd ..

# 安装tcmalloc
wget https://github.com/gperftools/gperftools/releases/download/gperftools-2.16/gperftools-2.16.tar.gz
tar -xf gperftools-2.16.tar.gz && cd gperftools-2.16
./configure --prefix=/usr/local/lib --with-tcmalloc-pagesize=64
make -j128
make install
export LD_PRELOAD="$LD_PRELOAD:/usr/local/lib/lib/libtcmalloc.so"
```

# 快速开始

## 微调模型
进入模型根目录 `$VLA_HOME/openvla`

按照 [Hugging Face user access token](https://huggingface.co/docs/hub/en/security-tokens) 创建 `hf_token` 记作 `hf_...`
  ```shell
  # 用自己的 hf_token 替换  "hf_..." 
  echo hf_... >>> .hf_token
  ```

- 单机8卡精度
```shell
bash test/train_full_8p.sh $VLA_HOME/models $VLA_HOME/datasets
```
- 单机8卡性能
```shell
bash test/train_performance_8p.sh $VLA_HOME/models $VLA_HOME/datasets
```

## 训练结果

**表 3** fully-finetune 结果展示表


|      芯片       | 卡数 | global batchsize  | max steps  |loss | FPS|
|:-------------:|:----:|:----:|:----------:|:----------:|:----:|
|      竞品A      | 8p | 256 |1000|0.1873  |  73.12  |
| Atlas 800T A2   | 8p | 256 |1000|0.1876  |   56.14 |




# 版本说明
## 变更
2025.8.21: 首次发布。

## FAQ

