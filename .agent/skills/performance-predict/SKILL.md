---
name: "performance-predict"
description: "Predict cross-chip performance from Ascend NPU profiling data. Invoke when user provides a profiling directory path and asks to predict/estimate performance on other chips."
---

# Cross-Chip Performance Prediction

Predict training/inference performance on different hardware chips based on Ascend NPU profiling data.

## When to Invoke

Invoke this skill when the user:

- Provides an Ascend profiling data directory path and asks to predict performance on other chips
- Asks to estimate/calculate cross-platform performance
- Mentions "性能预测", "性能测算", "跨芯片", "performance predict"

## Core Files

- Script: `d:\code\DrivingSDK\mx_driving\tools\performance_predict\performance_predictor.py`
- Hardware specs: `d:\code\DrivingSDK\mx_driving\tools\performance_predict\devices_info.csv`
- Documentation: `d:\code\DrivingSDK\.agent\skills\performance-predict\README.md`
- Output directory: `d:\code\DrivingSDK\mx_driving\tools\performance_predict\`

## Workflow

### Step 1: Load Hardware Specs & Confirm with User

Read `devices_info.csv` and display all available chips to the user:

```text
当前已收录的芯片规格:
  1. Huawei Ascend 910B (NPU) - Cube FP16: 354T, Vec FP16: 22T, HBM: 1.6 TB/s, 互联: 392 GB/s
  2. Huawei Ascend 910C (NPU) - Cube FP16: 708T, Vec FP16: 44T, HBM: 3.2 TB/s, 互联: 784 GB/s
  3. NVIDIA H200 (GPU) - Cube FP16: 989T, Vec FP16: 134T, HBM: 4.8 TB/s, 互联: 900 GB/s
  4. NVIDIA H20 (GPU) - Cube FP16: 148T, Vec FP16: 88T, HBM: 4.0 TB/s, 互联: 900 GB/s
```

Use `AskUserQuestion` to ask:

1. Which chip is the **baseline** (the profiling data source)? Default: auto-detect from profiling directory.
2. Which chips do you want to predict performance on? (multi-select from available chips)
3. If the user's baseline or target chip is NOT in the list, ask the user to provide the missing chip's specs in the following CSV format:

```csv
vendor,model,architecture_type,cube_fp16_tflops,cube_fp32_tflops,vector_fp16_tflops,vector_fp32_tflops,hbm_bandwidth,memory_gb,interconnect_bandwidth,cpu,notes
```

Field descriptions:

- `vendor`: Chip vendor (Huawei, NVIDIA, etc.)
- `model`: Chip model name
- `architecture_type`: NPU or GPU
- `cube_fp16_tflops`: Tensor Core / Cube FP16/BF16 compute (TFLOPS)
- `cube_fp32_tflops`: Tensor Core / Cube FP32 compute (TFLOPS)
- `vector_fp16_tflops`: CUDA Core / Vector FP16/BF16 compute (TFLOPS)
- `vector_fp32_tflops`: CUDA Core / Vector FP32 compute (TFLOPS)
- `hbm_bandwidth`: HBM memory bandwidth (TB/s)
- `memory_gb`: HBM capacity (GB)
- `interconnect_bandwidth`: Interconnect bandwidth (GB/s)
- `cpu`: CPU performance coefficient (1.0 = baseline ARM, higher = better dispatch)
- `notes`: Any notes

If the user provides new chip data, append it to `devices_info.csv` before proceeding.

### Step 2: Auto-detect Baseline Chip

If the user doesn't specify the baseline chip, auto-detect from the profiling directory:

- If directory name contains `ascend_pt` → Ascend 910B (default)
- Ask user to confirm

### Step 3: Run Prediction

Execute the prediction script:

```powershell
& "C:\Users\Jing\miniconda3\python.exe" "d:\code\DrivingSDK\mx_driving\tools\performance_predict\performance_predictor.py" "<profile_dir>" --source "<baseline_chip>"
```

Parameters:

- `<profile_dir>`: User-provided profiling directory path (must contain `*_ascend_pt` subdirectories)
- `--source`: Baseline chip identifier (e.g., "Huawei Ascend 910B")

The script automatically:

1. Loads `step_trace_time.csv` to extract Computing/Comm/FreeTime proportions
2. Loads `communication_matrix.json` to extract real communication time and split into real_comm + wait_time
3. Loads `kernel_details.csv` for per-kernel compute/memory prediction
4. Applies comprehensive calculation rules (see below)

### Step 4: Present Results

After prediction completes, present the results to the user:

- **Summary table**: Show both compute-only and comprehensive performance comparison
- **Step Trace breakdown**: Show Computing/Comm/FreeTime proportions and communication split
- **Key insights**: Which chip is best and why; whether compute-bound or memory-bound; real communication vs wait time; impact of FreeTime/CPU performance
- **Output files**: `performance_prediction.csv` (summary); `kernel_prediction_detail.csv` (per-kernel)

## Prediction Algorithm

### Compute Prediction (Kernel-level)

The prediction uses a **Max(Compute, Memory) bottleneck model**:

1. For each kernel, extract compute and memory times:
   - **Cube kernel**: compute = `aic_mac_time`, memory = `max(aic_mte1, aic_mte2)`
   - **Vector kernel**: compute = `aiv_vec_time`, memory = `max(aiv_mte2, aiv_mte3)`
   - **Mix kernel**: compute = `max(aic_mac, aiv_vec)`, memory = `max(aic_mte1, aic_mte2, aiv_mte2, aiv_mte3)`

2. Scale per target chip: `new_time = compute × compute_ratio + memory × memory_ratio`
   - `compute_ratio` = source_compute / target_compute (based on precision: fp16 or fp32)
   - `memory_ratio` = source_bandwidth / target_bandwidth

3. `scale = new_time / (compute + memory)`, `predicted_duration = original_duration × scale`

4. Only compute kernels (cube/vector/mix) are counted; AICPU/DSA/communication kernels are excluded.

5. Precision detection: Use only the **first** data type from "Input Data Types" column (e.g., "DT_BF16;INT64" → BF16 → fp16)

### Comprehensive Performance (Step Trace Integration)

The comprehensive performance combines compute, communication, and FreeTime:

```text
综合性能 = step_total_per_card / pred_total_per_card
```

Where:

- `pred_compute_us` = predicted_total / baseline_total × step_compute_mean
- `pred_comm_us` = real_comm_us / comm_speedup + wait_comm_us
  - `comm_speedup` = tgt_interconnect / src_interconnect
  - `wait_comm_us` = max(0, step_comm_mean - real_comm_us)
- `pred_free_us` = step_free_mean / free_speedup
  - `free_speedup` = tgt_cpu / src_cpu

### Special Rules

1. **910C dual-die**: Compute and bandwidth are 2× of 910B, interconnect is 784 GB/s (2× of 910B's 392 GB/s)
2. **950 microarchitecture**: MatMul kernels get ×1.2 efficiency boost, FlashAttention kernels get ×2.0 efficiency boost on 950 series
3. **950PR bandwidth equivalence**: When comparing within Huawei, 950PR's 1.4 TB/s HBM is treated as equivalent to 910B's 1.6 TB/s

## Important Notes

- scalar and fixpipe times are **ignored** in cross-architecture comparison as they are not directly comparable
- The profiling directory must be from Ascend NPU (containing `*_ascend_pt` subdirectories with `kernel_details.csv`)
- If `step_trace_time.csv` is not found, only compute-level prediction is performed (no comprehensive performance)
- If `communication_matrix.json` is not found, communication is not split (all treated as real communication)
- Python path: `C:\Users\Jing\miniconda3\python.exe`
- For detailed rules documentation, refer to `d:\code\DrivingSDK\.agent\skills\performance-predict\README.md`
