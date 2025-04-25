
# LaneSegNet: Map Learning with Lane Segment Perception for Autonomous Driving

# 环境准备

## 准备环境

### 安装昇腾环境

  请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表1中软件版本。

  **表 1**  昇腾软件版本支持表

|     软件类型      | 支持版本 |
| :---------------: | :------: |
| FrameworkPTAdaper | 7.0.0  |
|       CANN        | 8.1.RC1 |

### 安装模型环境

- 当前模型支持的 PyTorch 版本如下表所示。

  **表 2**  版本支持表

  | Torch_Version |
  | :--------: |
  | PyTorch 2.1 |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》搭建torch环境。

## 代码实现

- 参考实现：

```
url=https://github.com/OpenDriveLab/LaneSegNet
commit 699e5862ba2c173490b7e1f47b06184be8b7306e
```

- 适配昇腾 AI 处理器的实现：

```
url=https://gitee.com/ascend/DrivingSDK.git
code_path=DrivingSDK/model_examples/LaneSegNet
```

- 安装依赖。

1. 安装基础依赖
  ```
  pip install mmsegmentation==0.29.1
  ```

2. 源码安装 mmcv
  ```
  git clone -b 1.x https://github.com/open-mmlab/mmcv.git
  cd mmcv
  cp ../mmcv_config.patch ./
  git apply --reject --whitespace=fix mmcv_config.patch
  pip install -r requirements/runtime.txt
  MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py install
  ```

3. 源码安装 mmdet 2.26.0
  ```
  git clone -b v2.26.0 https://github.com/open-mmlab/mmdetection.git
  cp mmdet_config.patch mmdetection
  cd mmdetection
  git apply --reject mmdet_config.patch
  pip install -e .
  ```

4. 安装 mmdet3d
  ```
  git clone -b v1.0.0rc6 https://github.com/open-mmlab/mmdetection3d.git
  cd mmdetection3d
  cp ../mmdet3d_config.patch ./
  git apply --reject --whitespace=fix mmdet3d_config.patch
  pip install -e .
  ```

5. 安装Driving SDK加速库
  ```
  git clone https://gitee.com/ascend/DrivingSDK.git -b master
  cd mx_driving
  bash ci/build.sh --python=3.8
  cd dist
  pip3 install mx_driving-1.0.0+git{commit_id}-cp{python_version}-linux_{arch}.whl
  ```

6. 根据操作系统，安装tcmalloc动态库。

  - OpenEuler系统

  在当前python环境和路径下执行以下命令，安装并使用tcmalloc动态库。
  ```
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

7. Python编译优化

  编译优化是指通过毕昇编译器的LTO和PGO编译优化技术，源码构建编译Python、PyTorch、torch_npu（Ascend Extension for PyTorch）三个组件，有效提升程序性能。

  本节介绍Python LTO编译优化方式。

  - 安装毕昇编译器

  将CANN包安装目录记为cann_root_dir，执行下列命令安装毕昇编译器。
  ```
  wget https://kunpeng-repo.obs.cn-north-4.myhuaweicloud.com/BiSheng%20Enterprise/BiSheng%20Enterprise%20203.0.0/BiShengCompiler-4.1.0-aarch64-linux.tar.gz
  tar -xvf BiShengCompiler-4.1.0-aarch64-linux.tar.gz
  export PATH=$(pwd)/BiShengCompiler-4.1.0-aarch64-linux/bin:$PATH
  export LD_LIBRARY_PATH=$(pwd)/BiShengCompiler-4.1.0-aarch64-linux/lib:$LD_LIBRARY_PATH
  source {cann_root_dir}/set_env.sh
  ```

  - 安装依赖，将安装mpdecimal依赖包的目录记为mpdecimal_install_path。
  ```
  wget --no-check-certificate https://www.bytereef.org/software/mpdecimal/releases/mpdecimal-2.5.1.tar.gz
  tar -xvf mpdecimal-2.5.1.tar.gz
  cd mpdecimal-2.5.1
  bash ./configure --prefix=mpdecimal_install_path
  make -j
  make install
  ```

  - 获取Python源码并编译优化

  执行以下指令获取Python版本及安装目录，将Python安装路径记为python_path。
  ```
  python -V
  which python
  ```

  在[Python源码下载地址](https://www.python.org/downloads/source/)下载对应版本的Python源码并解压。

  以Python 3.8.17为例：
  ```
  tar -xvf Python-3.8.17.tgz
  cd Python-3.8.17
  export CC=clang
  export CXX=clang++
  ./configure --prefix=python_path > --with-lto --enable-optimizations
  make -j
  make install
  ```

8. 设置LaneSegNet
  ```
  git clone https://github.com/OpenDriveLab/LaneSegNet.git
  cp -f lane_seg_net_config.patch LaneSegNet
  cd LaneSegNet
  git checkout 699e5862ba2c173490b7e1f47b06184be8b7306e
  git apply --reject --whitespace=fix lane_seg_net_config.patch
  pip install -r requirements.txt
  ```

9. 依赖配置
  ```
  pip install networkx==3.1
  pip install numba
  pip install torchvision==0.16.0
  pip install numpy==1.24.0
  ```

## 准备数据集

根据[OpenLane-V2 repo](https://github.com/OpenDriveLab/OpenLane-V2/blob/v2.1.0/data)下载**Image**和**Map Element Bucket**文件。根据下列脚本生成模型所需数据。

> [!IMPORTANT]
>
> :exclamation: 请注意，用于生成LaneSegNet数据的脚本与OpenLane-V2中的`Map Element Bucket`不同。`*_lanesegnet.pkl`文件与`*_ls.pkl`文件不相同。
>
> :bell: `Map Element Bucket` 已于2023年10月更新。请务必下载最新数据。

```bash
cd LaneSegNet
mkdir data

ln -s {Path to OpenLane-V2 repo}/data/OpenLane-V2 ./data/
python ./tools/data_process.py
```

经过数据处理步骤后，`data`目录结构如下：
```
data/OpenLane-V2
├── train
|   └── ...
├── val
|   └── ...
├── test
|   └── ...
├── data_dict_subset_A_train_lanesegnet.pkl
├── data_dict_subset_A_val_lanesegnet.pkl
├── ...
```

## 开始训练

### Train

- 单机8卡性能

  ```
  bash test/train_8p_performance.sh
  ```

- 单机8卡精度

  ```
  bash test/train_8p_full.sh
  ```

# 结果

|  NAME       | Backbone    |   训练方式     |     Epoch    |    global_batch_size      |    mAP      |     FPS      |
|-------------|-------------------|-----------------|---------------|--------------|--------------|--------------|
|  8p-竞品A   | R50       |       FP32    |        24     |      8    |        32.27   |      23.75    |
|  8p-Atlas 800T A2   | R50      |       FP32    |        24     |      8    |        32.44   |      13.76    |


## 变更

2025.04.25：更新模型性能数据。

2025.04.24：优化模型性能。

2025.02.5：更新模型性能数据。

2024.12.5：首次发布。