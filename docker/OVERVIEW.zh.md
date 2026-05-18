# Driving SDK Docker 镜像概览

## 快速参考

- **维护者**: Driving SDK 团队
- **问题反馈**: [Driving SDK 问题追踪](https://gitcode.com/Ascend/DrivingSDK/issues)
- **支持的架构**: `x86_64` (AMD64), `aarch64` (ARM64)
- **支持的操作系统**: Ubuntu 22.04, openEuler 24.03
- **基础镜像**: CANN 9.0.0, CANN 8.5.1

---

## 镜像标签关键词说明

镜像标签遵循以下命名规范：

`<DrivingSDK版本>-<CANN版本>-<NPU类型>-<操作系统类型>`

### 字段说明

| 字段 | 说明 | 可选值 |
|------|------|--------|
| **DrivingSDK_VERSION** | Driving SDK 版本 | `26.0.0` |
| **CANN_VERSION** | CANN 版本 | `8.5.1`, `9.0.0` |
| **NPU_TYPE** | NPU 类型 | `A2`, `a3`|
| **OS_TYPE** | 操作系统类型 | `ubuntu22.04`, `openeuler24.03` |

### 示例

- `26.0.0-cann8.5.1-910b-ubuntu22.04`: CANN 8.5.1, A2, Ubuntu 22.04
- `26.0.0-cann9.0.0-a3-openeuler24.03`: CANN 9.0.0, A3, openEuler 24.03

---

## Dockerfile 归档路径

所有 Dockerfile 均归档在 `docker/` 目录下，结构如下：

```shell
docker/
├── 8.5.1-910b-openeuler24.03/
│   └── Dockerfile
├── 8.5.1-910b-ubuntu22.04/
│   └── Dockerfile
├── 8.5.1-a3-openeuler24.03/
│   └── Dockerfile
├── 8.5.1-a3-ubuntu22.04/
│   └── Dockerfile
├── 9.0.0-910b-openeuler24.03/
│   └── Dockerfile
├── 9.0.0-910b-ubuntu22.04/
│   └── Dockerfile
├── 9.0.0-a3-openeuler24.03/
│   └── Dockerfile
├── 9.0.0-a3-ubuntu22.04/
│   └── Dockerfile
├── install_bevformer.sh
├── install_bevfusion.sh
├── install_drivingsdk.sh
├── install_sparse4d.sh
├── OVERVIEW.md
└── OVERVIEW.zh.md
```

---

## 快速开始

### 1. 本地构建镜像

从源码构建 Docker 镜像：

```bash
# 克隆仓库
git clone https://gitcode.com/Ascend/DrivingSDK.git
cd DrivingSDK

# 构建镜像
docker build -f docker/8.5.1-910b-ubuntu22.04/Dockerfile -t drivingsdk:26.0.0-cann8.5.1-910b-ubuntu22.04 .

# 使用挂载卷运行，便于开发
docker run -it --rm \
  --device=/dev/davinci0 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/Ascend:/usr/local/Ascend \
  -v $(pwd):/workspace \
  drivingsdk:26.0.0-cann8.5.1-910b-ubuntu22.04 \
  /bin/bash
```

### 2. 二次开发

用于自定义开发和修改：

```bash
# 克隆仓库
git clone https://gitcode.com/Ascend/DrivingSDK.git
cd DrivingSDK

# 根据需要修改 Dockerfile
vim docker/8.5.1-910b-ubuntu22.04/Dockerfile

# 使用您的修改构建
docker build -f docker/8.5.1-910b-ubuntu22.04/Dockerfile -t drivingsdk:dev .

# 使用挂载卷运行，便于开发
docker run -it --rm \
  --device=/dev/davinci0 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/Ascend:/usr/local/Ascend \
  -v $(pwd):/workspace \
  drivingsdk:dev \
  /bin/bash
```

---

## 硬件支持信息

### 支持的 NPU 类型

| NPU 类型 | 架构 | 说明 | 状态 |
|----------|------|------|------|
| **A2** | x86_64, aarch64 | A2 | 生产就绪 |
| **A3** | x86_64, aarch64 | A3 | 生产就绪 |

### 支持的操作系统

| 操作系统 | 版本 | 架构 | 包管理器 |
|----------|------|------|----------|
| Ubuntu | 22.04 LTS | x86_64, aarch64 | apt |
| openEuler | 24.03 LTS | x86_64, aarch64 | yum/dnf |

### Python 环境支持

Docker 镜像通过 Miniconda 提供多种 Python 环境：

| 环境 | Python 版本 | PyTorch 版本 | 用途 |
|------|-------------|--------------|------|
| `torch2.1` | 3.8 | 2.1.0 | 通用 PyTorch 开发 |
| `torch2.7.1` | 3.10 | 2.7.1 | 最新 PyTorch 特性 |
| `bevformer` | 3.10 | 2.7.1 | BEVFormer 模型训练 |
| `bevfusion` | 3.10 | 2.7.1 | BEVFusion 模型训练 |
| `sparse4d` | 3.10 | 2.7.1 | Sparse4D 模型训练 |

### 硬件要求

- **最低配置**: 1 个 NPU 设备 (A2 或 A3)
- **推荐配置**: 2 个以上 NPU 设备用于分布式训练
- **内存**: 最低 32GB RAM，推荐 64GB 以上
- **存储**: 最低 100GB，用于 Docker 镜像和数据集

---

## 包含的组件

### 核心组件

- **DrivingSDK**: Driving SDK 版本 26.0.0
- **CANN**: 神经网络计算架构 (8.5.1 / 9.0.0)
- **PyTorch**: 深度学习框架 (2.1.0 / 2.7.1)
- **torch-npu**: 昇腾 PyTorch NPU 后端
- **Miniconda**: Python 环境管理

### 模型示例

- **BEVFormer**: 用于 3D 目标检测的鸟瞰图 Transformer
- **BEVFusion**: 基于 BEV 融合的多模态 3D 检测
- **Sparse4D**: 稀疏 4D 检测框架

### 系统依赖

- GCC/G++ 编译器
- CMake 构建系统
- Git, wget, curl 工具
- Protocol Buffers
- 网络工具

---

## 使用示例

### 激活特定环境

```bash
# 在容器内部
source /opt/conda/etc/profile.d/conda.sh
conda activate torch2.7.1
```

### 安装额外包

```bash
# 激活环境
conda activate torch2.7.1

# 安装包
pip install your-package
```

---

## 故障排除

### 常见问题

1. **NPU 设备未找到**

   ```bash
   # 检查 NPU 设备
   npu-smi info

   # 确保设备已挂载
   docker run --device=/dev/davinci0 ...
   ```

2. **CANN 环境未设置**

   ```bash
   # 加载 CANN 环境
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```

3. **Conda 环境未找到**

   ```bash
   # 初始化 conda
   source /opt/conda/etc/profile.d/conda.sh
   conda env list
   ```

---

## 许可证

本项目根据仓库根目录下 LICENSE 文件中指定的条款进行许可。

### 第三方许可证

- **CANN**: 华为昇腾软件许可
- **PyTorch**: BSD 3 条款许可证
- **Miniconda**: Anaconda 服务条款

---

## 免责声明

**重要提示**: 本软件按"原样"提供，不提供任何明示或暗示的担保。

- 使用风险自负
- 维护者不对因使用本软件而引起的任何损害负责
- 部署前请务必在非生产环境中进行测试
- 确保遵守华为昇腾许可条款
- NPU 硬件必须正确配置并可访问

对于生产部署，请参考华为昇腾官方文档，并遵循安全与性能优化的最佳实践。

---

## 支持与资源

- **CANN 文档**: [昇腾文档](https://www.hiascend.com/document)
- **问题追踪**: [Driving SDK 问题](https://gitcode.com/Ascend/DrivingSDK/issues)

---

**最后更新**: 2026-04-23
**维护者**: Driving SDK 团队
