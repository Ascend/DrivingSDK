---
name: "ascend-profiling-analysis"
description: "Analyze Ascend NPU profiling data to identify training performance bottlenecks. Breaks down step-level time into compute, unoverlapped communication, and freetime; within compute, analyzes compute vs memory-bound ratios and cube vs vector utilization to summarize the model's performance bottleneck."
---

# 模型 Profiling 数据分析

对基于 Ascend NPU 训练的模型，通过 `torch.profile` 采集的 profiling 数据进行自动化分析，拆解模型性能瓶颈，统计 cube/vector 算子耗时比例，计算和访存比例。

## 触发条件

当用户提供 profiling 数据目录路径并要求以下任一分析时调用此 skill：

- 分析模型训练耗时瓶颈
- 分析 step trace 中计算/通信/空闲占比
- 分析 kernel 算子耗时分布
- 分析计算 vs 访存带宽比例

## 执行步骤

### 1. 确认路径

用户需提供 profiling 数据根目录（包含 `*_ascend_pt` 子目录的路径），例如：

```text
D:\profile\gr00t-n1.6
```

如果用户未提供路径，使用 `AskUserQuestion` 询问。

### 2. 运行分析脚本

执行以下命令（使用 miniconda3 的 Python 环境）：

```powershell
& "$env:USERPROFILE\miniconda3\python.exe" "D:\code\DrivingSDK\mx_driving\tools\profile_analyse\profile_analyse.py" "<用户提供的路径>"
```

### 3. 解读结果

脚本会输出以下分析内容，将结果整理后呈现给用户：

**Section 1 — Step Trace 分析：**

- 各卡耗时明细表（Computing / Comm / FreeTime in us）
- 数据分析表（耗时占比 + 耗时波动 min~max）

**Section 2 — Kernel Details 分析：**

- 每种算子计算耗时与访存带宽耗时占比（Top 20，含计算类型 cube/vector）

**Section 3 — 整网计算 vs 访存比例：**

- 计算:访存百分比比值（如 43.20% : 56.80%）
- Cube:Vector 百分比比值（如 64.85% : 35.15%）

**CSV 输出文件**（保存在 `mx_driving\tools\profile_analyse\` 目录下，所有数值保留两位小数）：

| 文件 | 说明 |
|------|------|
| `step_trace_analysis.csv` | Step 1 — 各卡耗时明细 + Summary（耗时占比与波动） |
| `top20_ops_compute_memory_breakdown.csv` | Step 2 — Top 20 算子计算/访存耗时占比 |
| `computing_summary.csv` | Step 3 — cube/vector 大类汇总 + 计算:访存 / Cube:Vector 比例 |

## 注意事项

- Kernel Details 分析只取第一张卡的 kernel_details.csv 以加速计算
- 若 step_trace_time.csv 包含多个 Step，脚本自动取最后一个 Step 进行分析

## 依赖

- Python 环境：miniconda3（路径 `$env:USERPROFILE\miniconda3\python.exe`）
- 三方库：pandas, numpy
