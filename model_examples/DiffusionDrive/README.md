# DiffusionDrive

## 目录 
- [DiffusionDrive](#diffusiondrive)
  - [目录](#目录)
- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [支持任务列表](#支持任务列表)
  - [代码实现](#代码实现)
- [准备训练环境](#准备训练环境)
  - [安装昇腾环境](#安装昇腾环境)
  - [安装模型环境](#安装模型环境)
- [准备数据集与权重](#准备数据集与权重)
  - [Nuscenes数据集](#nuscenes数据集)
  - [数据集预处理](#数据集预处理)
  - [下载权重](#下载权重)
- [快速开始](#快速开始)
  - [训练模型](#训练模型)
  - [验证性能](#验证性能)
  - [训练脚本支持的命令行参数](#训练脚本支持的命令行参数)
  - [训练结果](#训练结果)
- [版本说明](#版本说明)
  - [变更](#变更)
  - [FAQ](#faq)


# 简介

## 模型介绍

**DiffusionDrive**是一种基于**截断扩散策略**的端到端自动驾驶模型，其核心通过**锚定高斯分布**重构多模态轨迹生成范式，显著提升了模型的推理效率与决策多样性。模型利用**级联扩散解码器**深度融合场景感知特征，仅需2步去噪即可生成物理合理的驾驶轨迹，在保证实时性的同时解决了传统扩散模型的计算瓶颈与模式重叠问题

## 支持任务列表

| 模型           | 任务列表 | 是否支持           |
| -------------- | -------- | ------------------ |
| DiffusionDrive | 训练     | :heavy_check_mark: |

## 代码实现

* 参考的官方实现：

  ```
  url=https://github.com/hustvl/DiffusionDrive/tree/nusc
  commit_id=ae54fd87b32b3762f20e63ffd0af91d343cade85
  ```

* 适配昇腾AI处理器的实现

  ```
  url=https://gitcode.com/Ascend/DrivingSDK.git
  code_path=model_examples/DiffusionDrive
  ```


# 准备训练环境

## 安装昇腾环境

请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes/ptes_00001.html)》文档搭建昇腾环境，本仓已支持表1中软件版本。

**表 1** 昇腾软件版本支持表

| 软件类型           | 支持版本 |
| ------------------ | -------- |
| FrameworkPTAdapter | 7.1.0    |
| CANN               | 8.2.RC1  |

## 安装模型环境

当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示， Python版本建议使用Python3.8。

**表 2** 版本支持表

| 三方库      | 支持版本 |
| ----------- | -------- |
| PyTorch     | 2.1.0    |
| mmcv        | 1.x      |
| mmdet       | 2.28.2   |

(需按照安装以下安装顺序进行安装)

- 安装Driving SDK：请参考昇腾[Driving SDK](https://gitcode.com/Ascend/DrivingSDK)代码仓说明编译安装Driving SDK，在完成README安装步骤后，应当完成了以下包的安装：

  - pyyaml
  - setuptools
  - CANN包
  - torch_npu包
  - DrivingSDK根目录下requirements.txt里列出的依赖
  - 源码编译并安装了的drivingsdk包

- 源码安装geos

  ```bash
  git clone https://github.com/libgeos/geos.git
  cd geos
  mkdir build
  cd build
  cmake ../
  make
  DRIVING_ENV_PATH=`pip3 show mx_driving | grep "Location" | awk -F "Location: " '{print $2}' | awk -F "python" '{print $1}'`
  cp lib/libgeos* ${DRIVING_ENV_PATH}
  cd ../../
  ```

- 克隆模型官方仓

  ```bash
  # 仅以克隆至当前目录举例，实际路径选择无影响
  git clone https://github.com/hustvl/DiffusionDrive
  cd DiffusionDrive
  git checkout ae54fd87b32b3762f20e63ffd0af91d343cade85
  ```

- 拷贝迁移昇腾的补丁至已克隆本地的官方仓内

  ```
  cp -r ../migrate_to_ascend ./
  ```

- 安装模型依赖与迁移依赖：一并打包进了`migrate_to_ascend`目录下的`requirements.txt`

  ```bash
  cd migrate_to_ascend
  pip install -r requirements.txt
  cd ..
  ```

- 源码安装mmcv

  ```
    git clone -b 1.x https://github.com/open-mmlab/mmcv.git
    cd mmcv
    MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py install
    cd ..
  ```

# 准备数据集与权重

## Nuscenes数据集

用户自行获取*nuscenes*数据集，在源码目录创建软连接`data/nuscenes`指向存放数据的具体路径，nuscenes数据目录结构如下：

```shell
DiffusionDrive/
├── data/
│   └── nuscenes/                 # 主数据集目录
│       ├── can_bus/              # 车辆总线信号数据
│       ├── lidarseg/             # 激光雷达点云语义分割
│       ├── maps/                 # 高精地图
│       ├── nuscenes_gt_database/ # 真值数据
│       ├── samples/              # 关键帧传感器数据
│       ├── sweeps/               # 非关键帧连续数据
│       ├── v1.0-test/            # 测试集元数据
│       └── v1.0-trainval/        # 训练验证集元数据
```


```bash
mkdir -p ./data/nuscenes
```

创建软链接：

```bash
ln -s [path/to/nuscenes] ./data/nuscenes
```

例：

```
export DATA_PATH=[path/to/nuscenes]
ln -s $DATA_PATH/can_bus ./data/nuscenes/can_bus
ln -s $DATA_PATH/lidarseg ./data/nuscenes/lidarseg
ln -s $DATA_PATH/maps ./data/nuscenes/maps
ln -s $DATA_PATH/nuscenes_gt_database ./data/nuscenes/nuscenes_gt_database
ln -s $DATA_PATH/samples ./data/nuscenes/samples
ln -s $DATA_PATH/sweeps ./data/nuscenes/sweeps
ln -s $DATA_PATH/v1.0-test ./data/nuscenes/v1.0-test
ln -s $DATA_PATH/v1.0-trainval ./data/nuscenes/v1.0-trainval
```

## 数据集预处理

运行数据预处理脚本生成DiffusionDrive模型训练需要的pkl文件与初始锚框（预处理耗时约5h左右）

```bash
python ./migrate_to_ascend/preprocess.py
```

## 下载权重

```bash
mkdir ckpts
cd ./ckpts/
wget https://download.pytorch.org/models/resnet50-19c8e357.pth
wget https://github.com/swc-17/SparseDrive/releases/download/v1.0/sparsedrive_stage1.pth
cd ..
```



# 快速开始

## 训练模型

```bash
# 8卡长跑全量epoch，训练结束后打印模型精度
bash migrate_to_ascend/train_8p.sh
```

## 验证性能

```bash
# 仅跑1000个step后打印8卡训练性能（AvgStepTime、FPS）
bash migrate_to_ascend/train_8p.sh --performance
```

## 训练脚本支持的命令行参数
`train_8p.sh`
* `--performance`：添加该参数，训练脚本仅验机器性能；未添加时，正常长跑训练完整epochs数
* `--num_npu`: 可调整训练使用的npu卡数
* `--batch_size`: 可调整每张卡的batch size




## 训练结果

**表 3** 训练结果展示表

| 芯片          | 卡数 | global batch size | FPS   | 平均step耗时(s) | L2     |
| ------------- | ---- | ----------------- | ----- | --------------- | ------ |
| 竞品A         | 8p   | 48                | 30.53 | 1.572           | 0.5897 |
| Atlas 800T A2 | 8p   | 48                | 28.43 | 1.688           | 0.5971 |



# 版本说明

## 变更

2025.06.16：首次发布。

2025.07.29: 修复预处理脚本和训练脚本统计step time的bug，更新GPU基线数据，更新DrivingSDK版本得到性能增益，刷新性能数据。

## FAQ

暂无。

