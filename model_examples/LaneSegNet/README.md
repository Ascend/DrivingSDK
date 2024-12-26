
# LaneSegNet: Map Learning with Lane Segment Perception for Autonomous Driving

# 概述
[![arXiv](https://img.shields.io/badge/arXiv-2312.16108-479ee2.svg)](https://arxiv.org/abs/2312.16108)
[![OpenLane-V2](https://img.shields.io/badge/GitHub-OpenLane--V2-blueviolet.svg)](https://github.com/OpenDriveLab/OpenLane-V2)
[![LICENSE](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](./LICENSE)

![lanesegment](figs/lane_segment.jpg "Diagram of Lane Segment")


### Performance in LaneSegNet paper

|   Model    | Epoch |  mAP  | TOP<sub>lsls</sub> | Memory | Config | Download |
| :--------: | :---: | :---: | :----------------: | :----: | :----: | :------: |
| LaneSegNet | 24 | 33.5 | 25.4 | 9.4G | [config](projects/configs/lanesegnet_r50_8x1_24e_olv2_subset_A.py) | [ckpt](https://huggingface.co/OpenDriveLab/lanesegnet_r50_8x1_24e_olv2_subset_A/resolve/main/lanesegnet_r50_8x1_24e_olv2_subset_A.pth) / [log](https://huggingface.co/OpenDriveLab/lanesegnet_r50_8x1_24e_olv2_subset_A/resolve/main/20231225_213951.log) |

# 环境准备

## 准备环境

- 当前模型支持的 PyTorch 版本如下表所示。

  **表 1**  版本支持表

  | Torch_Version | Python_Version|
  | :--------: | :--------:|
  | PyTorch 2.1 | Python 3.9 |
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》搭建torch环境。
  
- 安装依赖。
  1. 安装基础依赖
    ```
    pip install mmdet==2.26.0
    pip install mmsegmentation==0.29.1
    ```
  2. 源码安装 mmcv
    ```
    git clone -b 1.x https://github.com/open-mmlab/mmcv.git
    cd mmcv

    vi ./mmcv/parallel/distributed.py"
    将 module_to_run = self._replicated_tensor_module if self._use_replicated_tensor_module else self.module
    改为 module_to_run = self.module

    vi ./mmcv/parallel/_functions.py
    将 streams = [_get_stream(device) for device in target_gpus]
    改为 streams = [_get_stream(torch.device("cuda", device)) for device in target_gpus]

    pip install -r requirements/runtime.txt
    MMCV_WITH_OPS=1 MAX_JOBS=8 FORCE_NPU=1 python setup.py build_ext
    MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py develop
    ```

  3. 安装 mmdet3d
    ```
    git clone -b v1.0.0rc6 https://github.com/open-mmlab/mmdetection3d.git

    cd mmdetection3d
    修改mmdet3d/init.py mmcv_maximum_version版本范围
    修改 requirements/runtime.txt 注释numba部分
    pip install -e .
    ```

  4. 安装其他依赖
    ```
    pip install -r requirements.txt
    ```

  5. 安装mxDriving加速库

  6. 非兼容修改[可选]
    ```
    vi ../python3.9/site-packages/networkx/algorithms/dag.py
    将 from fractions import gcd 改为 from math import gcd
    ```

## 准备数据集

Following [OpenLane-V2 repo](https://github.com/OpenDriveLab/OpenLane-V2/blob/v2.1.0/data) to download the **Image** and the **Map Element Bucket** data. Run the following script to collect data for this repo. 

> [!IMPORTANT]
> 
> :exclamation: Please note that the script for generating LaneSegNet data is not the same as the OpenLane-V2 Map Element Bucket. The `*_lanesegnet.pkl` is not the same as the `*_ls.pkl`.
> 
> :bell: The `Map Element Bucket` has been updated as of October 2023. Please ensure you download the most recent data.

```bash
cd LaneSegNet
mkdir data

ln -s {Path to OpenLane-V2 repo}/data/OpenLane-V2 ./data/
python ./tools/data_process.py
```

After setup, the hierarchy of folder `data` is described below:
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

We recommend using 8 GPUs for training. If a different number of GPUs is utilized, you can enhance performance by configuring the `--autoscale-lr` option. The training logs will be saved to `work_dirs/lanesegnet`.

```bash
cd LaneSegNet
mkdir -p work_dirs/lanesegnet

bash ./tools/dist_train.sh 8 [--autoscale-lr]
```

### Evaluate
You can set `--show` to visualize the results.

```bash
bash ./tools/dist_test.sh 8 [--show]
```

# 结果

|  NAME       | Backbone    |   训练方式     |     Epoch    |    mAP      |     FPS      |
|-------------|-------------------|-----------------|---------------|--------------|--------------|
|  8p-竞品A   | R50       |       FP32    |        24     |        33.5   |      10.84    |
|  8p-Atlas 800T A2   | R50      |       FP32    |        24     |        32.9   |      11.00    |

# 公网地址说明
代码涉及公网地址参考 public_address_statement.md

# 版本说明

## 变更

2024.12.5：首次发布。

## FAQ