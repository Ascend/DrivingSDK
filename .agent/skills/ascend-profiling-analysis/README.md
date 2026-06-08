# Ascend Profiling 数据分析

对基于 Ascend NPU 训练的模型，通过 `torch.profile` 采集的 profiling 数据，通过该skill可进行自动化分析，拆解模型性能瓶颈，统计cube,vector算子耗时比例，计算和访存比例。

## 使用方法

向 AI 助手调用该skill，并提供 profiling 数据目录路径即可触发分析，例如：

```text
请分析 D:\profile\gr00t-n1.6 的 profiling 数据
```

## 分析内容

### 1. Step Trace 分析

- 各卡 Computing / Communication(Not Overlapped) / FreeTime 耗时明细
- 耗时占比与耗时波动（min~max）汇总表

### 2. Kernel Details 分析

- 按算子类型汇总计算耗时与访存带宽耗时（Top 20 表格）
- 自动将 MIX 算子重分类到 cube 或 vector
- 输出每类算子的计算/访存占比

### 3. 整网计算 vs 访存比例

- 计算:访存 = XX% : XX%
- cube:vector = XX% : XX%

## 输出文件

分析完成后，CSV 结果保存在 `mx_driving\tools\profile_analyse\` 目录下，所有数值保留两位小数：

| 文件 | 说明 |
|------|------|
| `step_trace_analysis.csv` | Step 1 —  各卡耗时明细，Summary（耗时占比与波动汇总） |
| `top20_ops_compute_memory_breakdown.csv` | Step 2 — Top 20 算子计算/访存耗时占比 |
| `computing_summary.csv` | Step 3 — cube / vector 大类汇总，整网计算/访存比例 |

## 数据要求

Profiling 数据需要时L1代算子信息的，目录需包含以 `_ascend_pt` 结尾的卡数据子目录，每个子目录下应有 `ASCEND_PROFILER_OUTPUT` 文件夹，包含：

- `step_trace_time.csv` — 单步计算/通信/freetime 耗时
- `kernel_details.csv` — 每个算子的执行耗时详情

## 注意事项

- 在进行Kernel Details 分析时，为了加快计算效率，可以只取其中一个文件夹的 kernel_details.csv 文件进行分析。
- 若分析数据包含多个 step，脚本会自动取最后一个 step 进行分析。

## 环境依赖

- Python 3（通过 miniconda3 管理）
- pandas
- numpy
