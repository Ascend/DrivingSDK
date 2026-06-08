# 跨芯片性能预测工具

基于昇腾 NPU 的模型训练采集的 Profiling 数据，根据不同硬件平台的芯片规格，预测模型在不同平台上的性能数据。

## 目录结构

```text
performance_predict/
├── performance_predictor.py   # 核心预测脚本
└── devices_info.csv           # 芯片硬件规格数据
```

## 快速使用

```bash
python performance_predictor.py <profiling目录路径> [--source <基线芯片>]
```

**示例：**

```bash
python performance_predictor.py D:\profile\gr00t-n1.6
python performance_predictor.py D:\profile\gr00t-n1.6 --source "Huawei Ascend 910B"
```

## 输入要求

- Profiling 数据目录需包含 `*_ascend_pt` 子目录
- 每个子目录下需有 `ASCEND_PROFILER_OUTPUT/kernel_details.csv`
- 基线芯片默认为 `Huawei Ascend 910B`

## 输出文件

| 文件 | 说明 |
|------|------|
| `performance_prediction.csv` | 各芯片汇总对比（预估耗时、相对性能倍数） |
| `kernel_prediction_detail.csv` | 逐算子预测明细 |

## 已收录芯片

| 芯片 | 架构 | Cube FP16 | Cube FP32 | Vector FP16 | Vector FP32 | HBM带宽 |
|------|------|-----------|-----------|-------------|-------------|---------|
| Huawei Ascend 910B | NPU | 354 TFLOPS | 88 TFLOPS | 22 TFLOPS | 11 TFLOPS | 1.6 TB/s |
| Huawei Ascend 910C | NPU | 708 TFLOPS | 176 TFLOPS | 44 TFLOPS | 22 TFLOPS | 3.2 TB/s |
| Huawei Ascend 950DT | NPU | 432 TFLOPS | 27 TFLOPS | 54 TFLOPS | 27 TFLOPS | 4.0 TB/s |
| Huawei Ascend 950PR | NPU | 371 TFLOPS | 27 TFLOPS | 54 TFLOPS | 27 TFLOPS | 1.4 TB/s |

## 添加新芯片

在 `devices_info.csv` 中追加一行，格式如下：

```csv
vendor,model,architecture_type,cube_fp16_tflops,cube_fp32_tflops,vector_fp16_tflops,vector_fp32_tflops,hbm_bandwidth_gbs,memory_gb,tdp_w,notes
```

**字段说明：**

| 字段 | 说明 |
|------|------|
| `vendor` | 芯片厂商（Huawei、NVIDIA 等） |
| `model` | 芯片型号 |
| `architecture_type` | 架构类型：NPU 或 GPU |
| `cube_fp16_tflops` | Tensor Core / Cube 的 FP16/BF16 算力（TFLOPS） |
| `cube_fp32_tflops` | Tensor Core / Cube 的 FP32 算力（TFLOPS） |
| `vector_fp16_tflops` | CUDA Core / Vector 的 FP16/BF16 算力（TFLOPS） |
| `vector_fp32_tflops` | CUDA Core / Vector 的 FP32 算力（TFLOPS） |
| `hbm_bandwidth_gbs` | HBM 内存带宽（TB/s） |
| `memory_gb` | HBM 显存容量（GB） |
| `interconnect_bandwidth` | 互联带宽（GB/s） |
| `notes` | 备注 |

**术语对应：**

| NPU (昇腾) | GPU (NVIDIA) | 说明 |
|-------------|-------------|------|
| Cube | Tensor Core | 矩阵乘法单元 |
| Vector | CUDA Core | 向量计算单元 |
| MTE1/MTE2 | L2→SM 读 | 数据搬运（读） |
| MTE3 | SM→L2 写 | 数据搬运（写） |

## 预测算法

### 核心思路

对每个计算类算子，提取**计算耗时**和**访存耗时**，根据目标芯片的算力比和带宽比进行换算。

### 计算与访存提取

| 算子类型 | 计算时间 | 访存时间 |
|---------|---------|---------|
| Cube | `aic_mac_time` | `max(aic_mte1_time, aic_mte2_time)` |
| Vector | `aiv_vec_time` | `max(aiv_mte2_time, aiv_mte3_time)` |
| Mix | `max(aic_mac_time, aiv_vec_time)` | `max(aic_mte1, aic_mte2, aiv_mte2, aiv_mte3)` |

> **注意：** `scalar` 和 `fixpipe` 时间在不同架构间不可直接比较，预测时忽略。

### 换算公式

```text
计算占比 = 计算时间 / (计算时间 + 访存时间)
访存占比 = 访存时间 / (计算时间 + 访存时间)

新耗时 = 计算时间 × compute_ratio + 访存时间 × memory_ratio
scale  = 新耗时 / (计算时间 + 访存时间)
预测耗时 = 原始 Duration × scale
```

其中：

- `compute_ratio` = 基线芯片算力 / 目标芯片算力（根据算子类型和精度选择 cube 或 vector 算力）
- `memory_ratio` = 基线芯片 HBM 带宽 / 目标芯片 HBM 带宽

### 精度检测

从 `Input Data Types` 列取**第一个**数据类型判断精度：

- `DT_BF16;INT64;INT64` → BF16 → fp16
- `DT_FP32;INT64` → FP32 → fp32

### 统计范围

- **只统计计算类算子**（cube / vector / mix）
- **排除** AICPU、DSA、通信类算子

## 综合测算规则

原始性能预估值仅考虑了纯计算耗时与访存耗时的对比，实际性能还需结合 Step Trace 中计算、未掩盖通信和 FreeTime 的比例进行综合测算。以下为各芯片的换算规则：

### 1. 结合 Step Trace 综合测算

调用 `ascend-profiling-analysis` skill 可获取 Step 1 输出的 Computing / Comm(Not Overlapped) / FreeTime 占比。综合性能预估公式：

```text
综合性能 = 计算部分性能 × Computing占比 + 通信部分性能 × Comm占比 + FreeTime占比
```

其中：

- 计算部分性能 = 1 / 预测 scale（由算力比和带宽比换算得到）
- 通信部分性能 = 基线互联带宽 / 目标互联带宽
- FreeTime 视为不变（占比直接保留）

### 2. FreeTime 与 CPU 性能换算

FreeTime 与 CPU 算子下发性能强相关。昇腾系列以 ARM 为主，友商竞品以 x86 为主，x86 的算子下发性能更好。换算规则：

- 同代际 x86 竞品相比 910B，FreeTime 部分性能提升约 **20%**
- 910C 相比 910B，FreeTime 部分性能提升约 **20%**
- 具体性能系数参考 `devices_info.csv` 中的 `cpu` 列

| 芯片 | cpu 系数 | 说明 |
|------|---------|------|
| Ascend 910B | 1.0 | 基线（ARM） |
| Ascend 910C | 1.2 | 双 die，算子下发效率提升 |
| H200 | 1.3 | x86，算子下发效率更高 |
| H20 | 1.2 | x86 |
| PPU1.5 | 1.2 | x86 |
| Ascend 950DT | 1.3 | 新架构，算子下发效率提升 |
| Ascend 950PR | 1.2 | 新架构 |

> FreeTime 预估 = 原始 FreeTime / cpu系数

### 3. 通信耗时拆分：真实通信 vs 等待

Step Trace 中的「未掩盖通信」包含两部分：**真实通信耗时**和**通信等待时间**。两者需分别测算。

#### 3.1 从 communication_matrix.json 提取真实通信数据

Profiling 数据中 `ASCEND_PROFILER_OUTPUT/communication_matrix.json` 记录了每个 step 的通信详情。提取规则：

1. 选择与 step_trace 相同的 step（取最后一个 step）
2. 遍历 `collective` 下所有通信算子（如 `allreduce-top1@xxx`、`allgather-top1@xxx` 等）
3. 对每个通信算子，取当前卡到其他所有卡中 **通信时长最长** 的那条记录，作为该算子的通信量和通信时长
4. 将一个 step 中所有通信算子的通信量和通信时长分别求和，作为该 step 的总通信量和总真实通信时长

```json
// 示例：communication_matrix.json 结构
{
    "step22": {
        "collective": {
            "allreduce-top1@5862276097510448909": {
                "0-0": { "Transit Size(MB)": 45.65, "Transit Time(ms)": 0.067, ... },
                "0-1": { "Transit Size(MB)": 75.50, "Transit Time(ms)": 3.812, ... },
                "0-7": { "Transit Size(MB)": 75.50, "Transit Time(ms)": 3.914, ... },  ← 最长，取此条
                ...
            },
            "allgather-top1@5862276097510448909": {
                "0-7": { "Transit Size(MB)": 404.85, "Transit Time(ms)": 20.834, ... },  ← 最长，取此条
                ...
            }
        }
    }
}
```

#### 3.2 通信耗时拆分公式

```text
等待时间 = Step Trace 未掩盖通信时长 - 真实通信时长
```

#### 3.3 跨芯片通信换算

- **真实通信耗时**：按互联带宽比换算（`基线互联带宽 / 目标互联带宽`）
- **等待时间**：视为不变，直接保留

#### 3.4 输出格式

在结果中分别列出：

```text
未掩盖通信: 149468.23 us（真实通信: 89200.00 us, 等待: 60268.23 us）
```

### 4. Ascend 910C 双die换算

910C 为双 die 合封结构，相当于两张 910B 封装在一起。在进行计算耗时换算时：

- **计算部分**：可直接按 **2 倍**换算（即 compute_ratio = 910B算力 / (910B算力 × 2)）
- **访存部分**：HBM 带宽同样翻倍（3.2 TB/s vs 1.6 TB/s）
- **互联带宽**：910C HCCL 带宽为 784 GB/s，是 910B 392 GB/s 的 2 倍

### 5. Ascend 950 系列微架构换算

950 系列相比 910B/910C 除了算力和带宽变化，**微架构也发生了变化**，部分算子的计算效率有显著提升：

| 算子类型 | 换算系数 | 说明 |
|---------|---------|------|
| Cube MatMul | 等算力 × **1.2** | 同 FLOPS 下 MatMul 算子效率为 910B/910C 的 1.2 倍 |
| Cube FlashAttention | 等算力 × **2.0** | 同 FLOPS 下 FA 算子效率为 910B/910C 的 2 倍 |
| 其他算子 | 等算力 × 1.0 | 按原始算力比换算 |

> **示例**：950DT Cube FP16 = 432 TFLOPS，MatMul 算子等效算力 = 432 × 1.2 = 518.4 TFLOPS

### 6. Ascend 950PR 带宽等价处理

950PR 的 HBM 带宽为 1.4 TB/s，910B 为 1.6 TB/s，但实际效果中等价。在昇腾系列芯片间换算时：

- **950PR 的 1.4 TB/s 与 910B 的 1.6 TB/s 按等价处理**
- 即 memory_ratio 中 950PR 的带宽取值与 910B 相同（1.6 TB/s）

## 示例输出

```text
================================================================================
预测结果: 各芯片相对性能对比 (基线=1.0)
================================================================================

设备                               架构   CubeFP16   CubeFP32    VecFP16    VecFP32     带宽GB/s       预估耗时(us)         相对性能
------------------------------------------------------------------------------------------------------------------------
NVIDIA H100                     GPU      989.0       67.0     134.00      67.00       3350     3043854.88       2.50x
Huawei Ascend 910C              NPU      708.0      188.0      44.00      22.00       3200     3801293.57       2.00x
NVIDIA H20                      GPU      148.0       44.0      88.00      44.00       4000     6602274.66       1.15x

基线 (Huawei Ascend 910B) = 1.00x (计算类耗时 7602587.14 us)
```
