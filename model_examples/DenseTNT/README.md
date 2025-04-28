#  DenseTNT for PyTorch

## 目录

- [简介](#简介)
- [支持任务列表](#支持任务列表)
- [准备训练环境](#准备训练环境)
- [准备数据集](#准备数据集)
- [快速开始](#快速开始)
    - [模型训练](#模型训练)
    - [训练结果](#训练结果)
-  [变更说明](#变更说明)
-  [FAQ](#FAQ)

## 简介

- 论文名称： DenseTNT: End-to-end Trajectory Prediction from Dense Goal Sets

- DenseTNT是一种基于密集目标的轨迹预测方法，可以直接从地图上采样多个目标点的概率分布，而不需要手动设置目标点。

- 原始代码仓库：https://github.com/Tsinghua-MARS-Lab/DenseTNT
- commit id：a07c237ea883b320aedf5e505185589ec4f88d76
- 昇腾适配代码仓库：https://gitee.com/ascend/DrivingSDK/tree/master/model_examples/DenseTNT

## 支持的任务列表

| 模型            | 任务列表       | 精度     | Backbone | 是否支持  |
| --------------- | -------------- | -------- | -------- | --------- |
| DenseTNT | 轨迹预测 | FP32精度 | VectorNet | $\sqrt{}$ |

## 准备训练环境

- 当前模型支持的 PyTorch 版本：`PyTorch 2.1`

- 搭建 PyTorch 环境参考：https://gitee.com/link?target=https%3A%2F%2Fwww.hiascend.com%2Fdocument%2Fdetail%2Fzh%2FModelZoo%2Fpytorchframework%2Fptes

**表1** 昇腾软件版本支持列表

| 软件类型 | 支持列表 |
| ------- | ------- |
| FrameworkPTAdapter | 7.0.0 |
| CANN | 8.1.RC1 |

1、激活 CANN 包环境：将 CANN 包所在目录记为 cann_root_dir，执行以下命令以激活环境：
```
source {cann_root_dir}/set_env.sh
```

2、创建 conda 环境并激活：
```
conda create -n densetnt python=3.9
conda activate densetnt
```

3、克隆模型原始代码仓到当前目录并切换到目标 commit ID
```
git clone https://github.com/Tsinghua-MARS-Lab/DenseTNT -b argoverse2
cd DenseTNT
git checkout a07c237ea883b320aedf5e505185589ec4f88d76
```
将模型根目录记作`path-to-DenseTNT`

4、克隆 Driving SDK 仓并使用 patch 文件
```
git clone https://gitee.com/ascend/DrivingSDK.git -b master
cd DrivingSDK/model_examples/DenseTNT
cp -f DenseTNT_npu.patch ${path-to-DenseTNT}
cp -rf test ${path-to-DenseTNT}
cd ${path-to-DenseTNT}
git apply DenseTNT_npu.patch
```

5、安装依赖包
```
pip install -r requirements.txt
pip install av2
```

6、安装gperftools，使能高性能库
```
wget https://github.com/gperftools/gperftools/releases/download/gperftools-2.16/gperftools-2.16.tar.gz
tar -zxvf gperftools-2.16.tar.gz && cd gperftools-2.16
./configure --prefix=/usr/local/lib --with-tcmalloc-pagesize=64
make
make install
```

## 准备数据集

进入 [Argoverse 2](https://www.argoverse.org/av2.html#download-link) 官网，下载 Argoverse 2 Motion Forecasting Dataset 数据集。将数据集上传到服务器任意路径下并解压，数据集结构排布成如下格式：

```shell
av2-motion-forecasting-dataset-path/
  train/  # training data
  test/   # test data
  val/     # validation data
```

## 快速开始

### 模型训练：

主要提供单机 8 卡训练脚本：

- 在模型根目录下运行训练脚本

```shell
bash test/train_8p_full.sh --data_path='data/train' --output_path='argoverse2.densetnt.1'  # data_path 替换成实际的数据集路径，进行 8 卡训练
bash test/train_8p_performance.sh --data_path='data/train' --output_path='argoverse2.densetnt.1'  # data_path 替换成实际的数据集路径，进行 8 卡性能测试
```

训练脚本参数说明：

```shell
--data_path    # 数据集路径，必填
--output_path  # 处理后数据和模型参数保存路径，必填
--epochs       # 训练迭代次数，可选项，默认 16
```

### 训练结果：

| 芯片          | 卡数 | epoch | global batch size | FDE | it/s  | FPS |
| ------------- | ---- | ----- | -----------------| --- | ---- |---|
| 竞品A         | 8p   | 16    | 64 | 5.158           | 3.712   |237 |
| Atlas 800T A2 | 8p   | 16    | 64 | 5.007            | 2.6   |166 |

## 变更说明

2025.1.6：首次发布
2025.3.4：性能优化，取消使用融合优化器
2025.4.22：性能优化
2025.4.28：性能优化，等价代码替换

## FQA

暂无