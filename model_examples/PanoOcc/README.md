# PanoOcc for PyTorch

## 目录

- [PanoOcc for PyTorch](#panoocc-for-pytorch)
  - [目录](#目录)
- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [支持任务列表](#支持任务列表)
  - [代码实现](#代码实现)
- [PanoOcc](#panoocc)
  - [准备训练环境](#准备训练环境)
    - [安装昇腾环境](#安装昇腾环境)
    - [安装模型环境](#安装模型环境)
    - [准备数据集](#准备数据集)
    - [准备预训练权重](#准备预训练权重)
  - [快速开始](#快速开始)
    - [单机8卡验证训练性能](#单机8卡验证训练性能)
    - [单机8卡完整训练验证精度](#单机8卡完整训练验证精度)
    - [训练脚本支持的命令行参数](#训练脚本支持的命令行参数)
- [训练结果](#训练结果)
- [变更说明](#变更说明)
- [FAQ](#faq)

# 简介

## 模型介绍

现有的感知任务（如对象检测、道路结构分割等）都只关注整体 3D 场景理解任务的一小部分。这种分而治之的策略简化了算法开发过程，但代价是失去了问题的端到端统一解决方案。PanoOcc 利用体素查询以从粗到细的方案聚合来自多帧和多视图图像的时空信息，将特征学习和场景表示集成到统一的占用表示中，为仅依赖相机的 3D 场景理解实现统一的占用表示，实现了基于相机的 3D 全景分割。

## 支持任务列表

本仓已经支持以下模型任务类型

|    模型     | 任务列表 | 是否支持 |
| :---------: | :------: | :------: |
| PanoOcc |   训练   |    ✔     |

## 代码实现

- 参考实现：

    ```
    url=https://github.com/Robertwyq/PanoOcc
    commit_id=3d93b119fcced35612af05587b395e8b38d8271f
    ```

# PanoOcc

## 准备训练环境
环境默认采用 Python 3.8

### 安装昇腾环境

请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表1中软件版本。

**表 1**  昇腾软件版本支持表

|     软件类型      | 支持版本 |
| :---------------: | :------: |
| FrameworkPTAdapter | 7.1.0  |
|       CANN        | 8.2.RC1  |

### 安装模型环境

**表 2**  三方库版本支持表

| 三方库  | 支持版本 |
| :-----: | :------: |
| PyTorch |   2.1.0   |
| mmcv |   1.7.2 |
| mmdet |   2.24.0   |
| mmdet3d |   1.0.0rc4   |


0. 激活 CANN 环境

1. 安装Driving SDK：请参考昇腾[Driving SDK](https://gitcode.com/Ascend/DrivingSDK)代码仓说明编译安装Driving SDK，在完成README安装步骤后，应当完成了以下包的安装：

     - CANN包
     - torch_npu包
        - torch_npu安装步骤包含了pyyaml和setuptools的安装，如果是通过whl包安装，请先安装`pip install pyyaml setuptools`
     - 根目录下requirements.txt里列出的依赖
     - 源码编译并安装了的drivingsdk包
  
2. 准备模型源码

    克隆并准备 PanoOcc 源码
    * 仅以clone至`DrivingSDK/model_example/PanoOcc/`目录举例
      * 完成clone后路径会包含`.../PanoOcc/PanoOcc/`连续两个PanoOcc字段。
      * 前者的PanoOcc主要存放DrivingSDK仓内的昇腾迁移优化补丁文件
      * 后者的PanoOcc是模型源码仓目录
    * 实际工程路径可用户自行选择，只需将`migrate_to_ascend`文件夹拷贝到实际的PanoOcc源码目录下即可

    ```
    git clone https://github.com/Robertwyq/PanoOcc.git
    cd PanoOcc
    git checkout 3d93b119fcced35612af05587b395e8b38d8271f
    ```

3. - 拷贝该模型专用的昇腾迁移补丁文件`migrate_to_ascend`至PanoOcc源码仓内
    
    如果PanoOcc源码仓clone至`DrivingSDK/model_example/PanoOcc/`目录下，在该路径里面的那个`PanoOcc`源码仓路径下运行：    
    ```
    cp -r ../migrate_to_ascend ./
    ```
    如果在其他路径下，则:
    ```
    cp -r DrivingSDK/model_example/PanoOcc/migrate_to_ascend/ [PATH_TO_PANOOCC_SOURCE_CODE]/PanoOcc/ 
    ```
    补丁文件主要包含以下内容：
    * 通过一键patcher特性实现的纯Python文件补丁，通过动态函数/类的替换对源码打迁移优化补丁
    * .patch补丁文件
    * requirements.txt 整合了模型原本的需要安装的依赖以及迁移到昇腾环境下所需的额外依赖
    * 在昇腾环境下运行模型的脚本
4. 安装依赖

    ```bash
    cd migrate_to_ascend
    pip install -r requirements.txt
    cd ..
    ```


5. 源码编译安装 mmcv

    在PanoOcc源码仓目录下，克隆 mmcv 仓，应用patch替换其中部分代码，并进入 mmcv 目录使用NPU编译选项编译安装（路径并非必须在PanoOcc目录下，仅以此举例）

    ```
    git clone -b 1.x https://github.com/open-mmlab/mmcv
    cp migrate_to_ascend/mmcv.patch mmcv/
    cd mmcv
    git apply --reject mmcv.patch
    MMCV_WITH_OPS=1 MAX_JOBS=8 FORCE_NPU=1 python setup.py build_ext
    MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py develop
    cd ../
    ```

6. 应用patch并源码编译安装 mmdet3d

    在PanoOcc源码仓目录下克隆 mmdet3d 仓，应用patch替换其中部分代码，并进入 mmdet3d 目录安装（路径并非必须在PanoOcc目录下，仅以此举例）

    ```
    git clone -b v1.0.0rc4 https://github.com/open-mmlab/mmdetection3d.git
    cp migrate_to_ascend/mmdetection3d.patch mmdetection3d
    cd mmdetection3d
    git apply --reject mmdetection3d.patch
    pip install -v -e .
    cd ../
    ```

### 准备数据集

1. 根据原仓数据集准备中的 [NuScenes LiDAR Benchmark](https://github.com/Robertwyq/PanoOcc/blob/main/docs/dataset.md#1-nuscenes-lidar-benchmark) 章节在模型源码根目录下准备数据集，参考数据集结构如下（数据预处理部分查看下文，预处理脚本会生成额外的json文件，没有列在该数据集结构里，不影响运行）：

    ```
    PanoOcc
    ├── data/
    │   ├── nuscenes/
    │   │   ├── can_bus/
    │   │   ├── maps/
    │   │   ├── lidarseg/
    │   │   ├── panoptic/
    │   │   ├── samples/
    │   │   ├── sweeps/
    │   │   ├── v1.0-trainval/
    │   │   ├── v1.0-test/
    │   │   ├── nuscenes_infos_temporal_train.pkl (经数据预处理后生成)
    │   │   ├── nuscenes_infos_temporal_val.pkl (经数据预处理后生成)
    │   │   ├── nuscenes_infos_temporal_test.pkl (经数据预处理后生成)
    │   │   ├── nuscenes.yaml
    ```
    可通过创建软链接的方式链接用户实际的下载解压后存放数据集的路径，假设实际存放数据集的路径已设为环境变量`$DATA_PATH`（例：`export DATA_PATH=/home/data/nuscenes`）在PanoOcc源码仓目录下（例：`DrivingSDK/model_example/PanoOcc/PanoOcc/`），运行以下命令创建软连接：
    ```
    ln -s $DATA_PATH/can_bus ./data/nuscenes/can_bus
    ln -s $DATA_PATH/maps ./data/nuscenes/maps
    ln -s $DATA_PATH/lidarseg ./data/nuscenes/lidarseg
    ln -s $DATA_PATH/panoptic ./data/nuscenes/panoptic
    ln -s $DATA_PATH/samples ./data/nuscenes/samples
    ln -s $DATA_PATH/sweeps ./data/nuscenes/sweeps
    ln -s $DATA_PATH/v1.0-trainval ./data/nuscenes/v1.0-trainval
    ln -s $DATA_PATH/v1.0-test ./data/nuscenes/v1.0-test
    ln -s $DATA_PATH/nuscenes.yaml ./data/nuscenes/nuscenes.yaml
    ```
2. 在模型源码根目录下进行数据预处理

   （预计耗时1个小时左右）
   ``` 
   # 运行数据预处理脚本，运行结束后会在./data/nuscenes里生成.pkl文件
   python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0 --canbus ./data/nuscenes
   ```
    （如果遇到`AssertionError: MMCV==1.7.2 is used but incompatible`报错，参考下方FAQ章节）
### 准备预训练权重

在模型源码根目录下创建 ckpts 文件夹，将预训练权重 r101_dcn_fcos3d_pretrain.pth 放入其中
   ```
   ckpts/
   ├── r101_dcn_fcos3d_pretrain.pth
   ```

## 快速开始


本任务主要提供**单机**的**8卡**训练脚本。

在模型源码根目录下，运行训练脚本:

* 以下`${CASE_NAME}`变量指代由脚本自动生成的目录名，包含了
  * 模型网络名称（对应config文件名称，有small、base、large等几种变种）
  * 训练卡数
  * batch size
  * epoch数
  * 时间戳
* 以下示例均以PanoOcc_Base_4f举例，可通过--config入参来指定不同的config文件并使用对应变种的模型


### 单机8卡验证训练性能

仅训练少量迭代检验8卡训练性能，1000个训练step后会早停（预计耗时45分钟）
```
bash migrate_to_ascend/train_8p.sh --performance # 8卡性能
```
脚本默认通过nohup于后台不挂断进行训练，训练日志默认存放在`output/${CASE_NAME}/train_8p_performance.log`


### 单机8卡完整训练验证精度
全量长跑config文件里所配置的epoch数（默认24个epochs）的训练（预计耗时约3天）
```
bash migrate_to_ascend/train_8p.sh # 8卡精度
```
脚本默认通过nohup于后台不挂断进行训练，训练日志默认存放在`output/${CASE_NAME}/train_8p_full.log`（${CASE_NAME}目录的命名包含了batch size、训练卡数、时间戳等信息）

完整训练获得`latest.pth`后，脚本会自动进行模型推理验证精度（latest.pth存放在`output/${CASE_NAME}/work_dir`下），推理精度结果会添加到`output/${CASE_NAME}/train_result.log`


### 训练脚本支持的命令行参数
`train_8p.sh`
* `--performance`：添加该参数，训练脚本仅验机器性能；未添加时，正常长跑训练完整epochs数
* `--epochs`: 可调整训练epochs数
* `--num_npu`: 可调整训练使用的npu卡数
* `--workers_per_npu`：可调整每张卡的数据加载子进程的数量
* `--batch_size`: （当前版本暂不支持bs大于1，仅作为预埋参数，待后续更新）可调整每张卡的batch size

# 训练结果

| 芯片          | 卡数 | global batch size | Precision | epoch | mIoU | mAP | NDS | 性能-单步迭代耗时(ms) |
| ------------- | :--: | :---------------: | :-------: | :---: | :----: | :----: | :----: | :-------------------: |
| 竞品A           |  8p  |         8         |   fp32    |  24   | 0.6879 | 0.3976 | 0.4791 |         1590          |
| Atlas 800T A2 |  8p  |         8         |   fp32    |  24   | 0.6838 | 0.3988 | 0.4810 |         1849          |

# 变更说明

2024.09.10：首次发布。

2025.8.18: 迁移至一键patcher实现，性能优化，脚本优化，更新基线。

2025.9.18: 修复npu随机性固定，更新基线，修复test脚本因import顺序导致的libtorch_npu.so找不到的问题



# FAQ

* 当前版本暂不支持bs大于1，仅作为预埋参数，待后续更新
* 运行`PanoOcc/tools/`里的脚本如遇到`AssertionError: MMCV==1.7.2 is used but incompatible`，可通过以下任意一种方式解决：
  * 在脚本最上方添加以下两行解决
    ```
    from mx_driving.patcher import patch_mmcv_version
    patch_mmcv_version("1.6.0")
    ```
  * 安装mmcv 1.6.0，使用完脚本后，再重新使用npu编译选项编译安装回mmcv 1.7.2