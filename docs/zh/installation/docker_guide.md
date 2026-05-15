# Driving SDK 自定义镜像构建指导

## 快速参考

- **维护团队**：Driving SDK Team
- **问题反馈**：[Driving SDK Issue Tracker](https://gitcode.com/Ascend/DrivingSDK/issues)
- **支持架构**：`x86_64` (AMD64), `aarch64` (ARM64)
- **支持操作系统**：Ubuntu 22.04, openEuler 24.03
- **基础镜像**：CANN 9.0.0, CANN 8.5.1

---

## 镜像标签说明

镜像标签遵循以下命名规则：

`<DrivingSDK_VERSION>-<CANN_VERSION>-<NPU_TYPE>-<OS_TYPE>`

### 字段说明

| 字段 | 说明 | 可选值 |
|------|------|--------|
| **Driving SDK VERSION** | Driving SDK 版本 | `26.0.0` |
| **CANN_VERSION** | CANN 版本 | `8.5.1`, `9.0.0` |
| **NPU_TYPE** | NPU 类型 | `910b`, `a3` |
| **OS_TYPE** | 操作系统类型 | `ubuntu22.04`, `openeuler24.03` |

### 示例

- `26.0.0-cann8.5.1-910b-ubuntu22.04`：CANN 8.5.1, 910B, Ubuntu 22.04
- `26.0.0-cann9.0.0-a3-openeuler24.03`：CANN 9.0.0, A3, openEuler 24.03

---

## Dockerfile 归档路径

所有 Dockerfile 归档在 `docker/` 目录下，目录结构如下：

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
└── install_sparse4d.sh
```

---

## 快速上手

### 1. 本地构建镜像

从源码构建 Docker 镜像：

```bash
# 克隆仓库
git clone https://gitcode.com/Ascend/DrivingSDK.git
cd DrivingSDK

# 构建镜像
docker build -f docker/8.5.1-910b-ubuntu22.04/Dockerfile -t drivingsdk:26.0.0-cann8.5.1-910b-ubuntu22.04 .

# 挂载卷运行容器进行开发
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

如需自定义开发和修改：

```bash
# 克隆仓库
git clone https://gitcode.com/Ascend/DrivingSDK.git
cd DrivingSDK

# 根据需要修改 Dockerfile
vim docker/8.5.1-910b-ubuntu22.04/Dockerfile

# 使用修改后的 Dockerfile 构建
docker build -f docker/8.5.1-910b-ubuntu22.04/Dockerfile -t drivingsdk:dev .

# 挂载卷运行容器进行开发
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
| **910b** | x86_64, aarch64 | 910B | 已就绪 |
| **a3** | x86_64, aarch64 | A3 | 已就绪 |

### 支持的操作系统

| 操作系统 | 版本 | 架构 | 包管理器 |
|----------|------|------|----------|
| Ubuntu | 22.04 LTS | x86_64, aarch64 | apt |
| openEuler | 24.03 LTS | x86_64, aarch64 | yum/dnf |

### Python 环境支持

Docker 镜像通过 Miniconda 提供多个 Python 环境：

| 环境名称 | Python 版本 | PyTorch 版本 | 适用场景 |
|----------|-------------|-------------|----------|
| `torch2.1` | 3.8 | 2.1.0 | 通用 PyTorch 开发 |
| `torch2.7.1` | 3.10 | 2.7.1 | 最新 PyTorch 特性 |
| `bevformer` | 3.10 | 2.7.1 | BEVFormer 模型训练 |
| `bevfusion` | 3.10 | 2.7.1 | BEVFusion 模型训练 |
| `sparse4d` | 3.10 | 2.7.1 | Sparse4D 模型训练 |

### 硬件要求

- **最低配置**：1 个 NPU 设备（910B 或 A3）
- **推荐配置**：2 个及以上 NPU 设备用于分布式训练
- **内存**：最低 32GB RAM，推荐 64GB+
- **存储**：最低 100GB 用于 Docker 镜像和数据集

---

## 包含的组件

### 核心组件

- **Driving SDK**：Driving SDK 版本 26.0.0
- **CANN**：异构计算架构（8.5.1 / 9.0.0）
- **PyTorch**：深度学习框架（2.1.0 / 2.7.1）
- **torch-npu**：PyTorch 昇腾 NPU 后端
- **Miniconda**：Python 环境管理

### 模型示例

- **BEVFormer**：用于 3D 目标检测的鸟瞰图 Transformer
- **BEVFusion**：基于 BEV 融合的多模态 3D 检测
- **Sparse4D**：稀疏 4D 检测框架

### 系统依赖

- GCC/G++ 编译器
- CMake 构建系统
- Git、wget、curl 工具
- Protocol Buffers
- 网络工具

---

## 使用示例

### 激活指定环境

```bash
# 在容器内执行
source /opt/conda/etc/profile.d/conda.sh
conda activate torch2.7.1
```

### 安装额外依赖包

```bash
# 激活环境
conda activate torch2.7.1

# 安装依赖包
pip install your-package
```

---

## 常见问题

以下是使用 Driving SDK Docker 镜像时可能遇到的常见问题及解决方法：

1. **找不到 NPU 设备**

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

3. **找不到 Conda 环境**

   ```bash
   # 初始化 conda
   source /opt/conda/etc/profile.d/conda.sh
   conda env list
   ```

---

## 免责声明

**重要提示**：本文档描述的 Docker 镜像及构建文件按"原样"提供，不提供任何形式的明示或暗示担保。

- 使用风险自负
- 维护者不对因使用本镜像而产生的任何损害负责
- 在生产环境部署前，请务必在非生产环境中进行测试
- 确保遵守华为昇腾许可条款
- NPU 硬件必须正确配置并可访问

对于生产环境部署，请查阅华为昇腾官方文档，并遵循安全和性能优化的最佳实践。

---

## 支持与资源

- **CANN 文档**：[昇腾文档](https://www.hiascend.com/document)
- **PyTorch 昇腾适配**：[Ascend Extension for PyTorch](https://gitcode.com/Ascend/pytorch)
- **问题反馈**：[Driving SDK Issues](https://gitcode.com/Ascend/DrivingSDK/issues)
- **环境部署**：返回[部署Driving SDK环境](./installation.md)

---

**最后更新**：2026-04-23
**维护者**：Driving SDK Team
