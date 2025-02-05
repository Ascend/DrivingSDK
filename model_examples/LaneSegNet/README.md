
# LaneSegNet: Map Learning with Lane Segment Perception for Autonomous Driving

# 环境准备

## 准备环境

- 当前模型支持的 PyTorch 版本如下表所示。

  **表 1**  版本支持表

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
  pip install mmdet==2.26.0
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

3. 安装 mmdet3d
  ```
  git clone -b v1.0.0rc6 https://github.com/open-mmlab/mmdetection3d.git
  cd mmdetection3d
  cp ../mmdet3d_config.patch ./
  git apply --reject --whitespace=fix mmdet3d_config.patch
  pip install -e .
  ```

4. 设置LaneSegNet
  ```
  git clone https://github.com/OpenDriveLab/LaneSegNet.git
  cp -f lane_seg_net_config.patch LaneSegNet
  cd LaneSegNet
  git checkout 699e5862ba2c173490b7e1f47b06184be8b7306e
  git apply --reject --whitespace=fix lane_seg_net_config.patch
  pip install -r requirements.txt
  ```

5. 依赖配置
  ```
  pip install networkx==3.1
  pip insatll torchvision==0.16.0
  pip install numba
  pip install torchvision==0.16.0
  pip install numpy==1.24.0
  ```

6. 安装Driving SDK加速库
  ```
  git clone https://gitee.com/ascend/DrivingSDK.git -b master
  cd mx_driving
  bash ci/build.sh --python=3.8
  cd dist
  pip3 install mx_driving-1.0.0+git{commit_id}-cp{python_version}-linux_{arch}.whl
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

- 单机8卡性能

  ```
  bash test/train_8p_performance.sh
  ```

- 单机8卡精度

  ```
  bash test/train_8p_full.sh
  ```

# 结果

|  NAME       | Backbone    |   训练方式     |     Epoch    |    mAP      |     FPS      |
|-------------|-------------------|-----------------|---------------|--------------|--------------|
|  8p-竞品V   | R50       |       FP32    |        24     |        32.27   |      23.75    |
|  8p-Atlas 800T A2   | R50      |       FP32    |        24     |        32.44   |      14.25    |


## 变更
2024.12.5：首次发布。