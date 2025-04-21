# MatrixVT

## 目录

- [MatrixVT](#matrixvt)
  - [目录](#目录)
- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [支持任务列表](#支持任务列表)
  - [代码实现](#代码实现)
- [准备训练环境](#准备训练环境)
  - [安装昇腾环境](#安装昇腾环境)
  - [准备环境](#准备环境)
  - [准备数据集](#准备数据集)
- [开始训练](#开始训练)
- [训练结果](#训练结果)
  - [精度](#精度)
  - [性能](#性能)
- [变更说明](#变更说明)
- [FAQ](#faq)

# 简介

## 模型介绍

MatrixVT是一个基于Transformer结构的BEV 3D检测模型，没有定制化算子。针对目前BEV中更有优势的Lift-Splat类方法中关键模块（Vision Transformation），MatrixVT实现了非常优雅的优化，在保持模型性能（甚至略微提高）的同时，能大幅降低计算量和内存消耗。

## 支持任务列表
本仓已经支持以下模型任务类型

| 模型 | 任务列表 | 是否支持 |
|:---:|:---:|:----:|
|MatrixVT | 训练 |  ✔   |

## 代码实现

- 参考实现：

  ```
  url=https://github.com/Megvii-BaseDetection/BEVDepth/
  commit_id=d78c7b58b10b9ada940462ba83ab24d99cae5833
  ```
- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/DrivingSDK.git
  code_path=model_examples/MatrixVT
  ```

# 准备训练环境

## 安装昇腾环境

请参考昇腾社区中《 [Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes) 》文档搭建昇腾环境。本仓已支持表1中软件版本。

  **表 1**  昇腾软件版本支持表

|        软件类型       |   支持版本   |
|:--------:|:--------:|
| FrameworkPTAdapter | 7.0.0  |
|       CANN        | 8.1.RC1  |

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 2**  版本支持表

  |  三方库  |  支持版本  |
  |:------:|:--------:| 
  |  PyTorch  | 2.1.0  |
  |  TorchVision  | 0.12.0  |
  |    mmcv   | 1.7.2  |
  |   mmdet3d | 1.0.0rc4  |
  |   mmdet   |   2.28.0  |
  | mmsegmentation | 0.30.0 |
  | pytorch-lightning | 1.6.5 |
  
- 安装依赖

  首先进入模型目录：

  ```
  cd DrivingSDK/model_examples/MatrixVT
  ```
  
  1. 下载包含 `BEVDepth` 模型的原始仓库代码，并执行如下命令：
      
      ```
      git clone https://github.com/Megvii-BaseDetection/BEVDepth.git
      # 拷贝训练脚本和环境配置文件
      cp -r test/* BEVDepth/test/
      # 拷贝MatrixVT的patch文件
      cp -f MatrixVT.patch BEVDepth/
      cd BEVDepth
      git reset --hard d78c7b58b10b9ada940462ba83ab24d99cae5833
      git apply MatrixVT.patch
      # require pip==23.3.1
      pip install pip==23.3.1
      pip install -r requirements.txt
      cd ../
      ```

  2. 源码编译安装 mmcv 1.7.2
     ```
     git clone -b 1.x https://github.com/open-mmlab/mmcv.git
     cp -f mmcv.patch mmcv
     cd mmcv
     git reset --hard 5e2b9a7b837d903bca00daf929ca5a461a8c7f50
     git apply mmcv.patch
     MMCV_WITH_OPS=1 pip install -e . -v
     cd ../
      ```

  3. 源码安装 mmdetection3d v1.0.0rc4
      ```
      git clone -b v1.0.0rc4 https://github.com/open-mmlab/mmdetection3d.git
      cp -f mmdetection3d.patch mmdetection3d
      cd mmdetection3d
      git reset --hard c9541b0db89498fdea5cafd05b7b17f7b625b858
      git apply mmdetection3d.patch
      pip install -v -e .
      cd ../
      ```

  4. pytorch-lightning 兼容性配置
      ```
      pip show pytorch_lightning
      ```
     找到pytorch_lightning的安装路径，初始化pytorch_lightning代码仓并提交一个commit记录，然后将源码根目录下面lightning.patch文件复制到pytorch_lightning安装路径下。
      ```
      git init
      git add .
      git commit -m "Initialize pytorch-lightning" 
      ```
     
      ```
      cp -f lightning.patch {pytorch_lightning_install_path}/pytorch_lightning/
      cd {pytorch_lightning_install_path}/pytorch_lightning/
      git apply lightning.patch
      ``` 

  5. 请参考 [Driving SDK 使用说明](https://gitee.com/ascend/DrivingSDK/blob/master/README.md) 编译并安装 `Driving SDK` 包, 参考`从源码安装`章节。

  6.  返回BEVDepth模型根目录
      ```shell
      cd DrivingSDK/model_examples/MatrixVT/BEVDepth
      python setup.py develop
      ```

## 准备数据集

   1. 请用户自行获取并解压nuScenes数据集，并将数据集的路径软链接到 `./data/`。
       ```
       ln -s [nuscenes root] ./data/
       ```

   2. 在源码根目录下进行数据集预处理。

       ```
       python scripts/gen_info.py
       python scripts/gen_depth_gt.py
       ```

      参考数据集结构如下：

      ```
       MatrixVT for PyTorch
       ├── data
       │   ├── nuScenes
       │   │   ├── maps
       │   │   ├── samples
       │   │   ├── sweeps
       │   │   ├── v1.0-test
       |   |   ├── v1.0-trainval
       ```
       
       > **说明：**  
       该数据集的训练过程脚本只作为一种参考示例。      


# 开始训练

本任务主要提供**混精fp16**的**8卡**训练脚本。

1. 进入源码根目录。

    ```
    cd /${模型文件夹名称}
    ```

2. 运行训练脚本。

     该模型支持单机8卡训练。

     - 单机8卡训练

     ```
     bash ./test/train_full_8p.sh # 8卡精度，混精fp16
     bash ./test/train_performance_8p.sh # 8卡性能，混精fp16
     ```

     > 注：当前配置下，不需要修改train_full_8p.sh中的ckpt路径，如果涉及到epoch的变化，请用户根据路径自行配置ckpt。

     模型训练脚本参数说明如下。
   
     ```
     matrixvt_bev_depth_lss_r50_256x704_128x128_24e_ema.py
     --seed                              // 随机种子
     --learning_rate                     // 学习率
     --max_epoch                         // 最大迭代回合数
     --amp_backend                       // 混精策略
     --gpus                              // 卡数
     --precision                         // 训练精度模式
     --batch_size_per_device             // 每张卡的批大小
     ```


# 训练结果
## 精度

| 芯片       | 卡数 | global batch size | mAP   |  NDS   | AMP_Type |
|----------|:--:|:------:|:------:|:----------:|:--------:|
| 竞品A     | 8p | 64 | 0.3299 | 0.4141 | fp16 |
| Atlas 800T A2 | 64 | 8p | 0.3275 | 0.4049 | fp16 |

## 性能

| 芯片       | 卡数 | global batch size | FPS  | Each epoch time  | AMP_Type |
|----------|:--:|:----:|:----:|:----:|:--------:|
| 竞品A      | 8p | 64| 36.89 | 1.00 h | fp16 |
| Atlas 800T A2 | 8p | 64| 46.19 | 0.80 h | fp16 |

# 变更说明

2025.03.14：首次发布。

# FAQ

1. 报错scikit_learning.libs/libgomp-d22c30c5.so.1.0.0: cannot allocate memory in static TLS block的问题，解决方案为：

   ```
   # 手动导入环境变量
   export LD_PRELOAD={libgomp-d22c30c5.so.1.0.0_path}/libgomp-d22c30c5.so.1.0.0:$LD_PRELOAD
   ```

