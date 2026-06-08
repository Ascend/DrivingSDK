import os
import sys
import json
import argparse
import importlib
import pandas as pd

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

PROFILE_ANALYSE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "profile_analyse"))
if PROFILE_ANALYSE_DIR not in sys.path:
    sys.path.insert(0, PROFILE_ANALYSE_DIR)

profiling_common = importlib.import_module("profiling_common")
extract_step_trace_stats = profiling_common.extract_step_trace_stats
find_card_dirs = profiling_common.find_card_dirs
get_profile_dir = profiling_common.get_profile_dir
load_step_trace = profiling_common.load_step_trace
pad = profiling_common.pad


def load_devices_info(csv_path=None):
    if csv_path is None:
        csv_path = os.path.join(OUTPUT_DIR, "devices_info.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"硬件规格文件不存在: {csv_path}")
    df = pd.read_csv(csv_path)
    df["identifier"] = df["vendor"] + " " + df["model"]
    print(f"加载 {len(df)} 款芯片的硬件规格数据")

    device_cache = {}
    for _, row in df.iterrows():
        ident = row["identifier"]
        device_cache[ident] = {
            "cube_fp16": row["cube_fp16_tflops"] if not pd.isna(row["cube_fp16_tflops"]) else 0,
            "cube_fp32": row["cube_fp32_tflops"] if not pd.isna(row["cube_fp32_tflops"]) else 0,
            "vector_fp16": row["vector_fp16_tflops"] if not pd.isna(row["vector_fp16_tflops"]) else 0,
            "vector_fp32": row["vector_fp32_tflops"] if not pd.isna(row["vector_fp32_tflops"]) else 0,
            "hbm_bandwidth": row["hbm_bandwidth"] if not pd.isna(row["hbm_bandwidth"]) else 0,
            "interconnect_bandwidth": row["interconnect_bandwidth"]
            if "interconnect_bandwidth" in row and not pd.isna(row["interconnect_bandwidth"])
            else 0,
            "cpu": row["cpu"] if "cpu" in row and not pd.isna(row["cpu"]) else 1.0,
            "row": row,
        }

    return df, device_cache


def load_communication_matrix(card_dirs):
    for d in card_dirs:
        json_path = os.path.join(d, "ASCEND_PROFILER_OUTPUT", "communication_matrix.json")
        if not os.path.exists(json_path):
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    return None


def extract_comm_stats(comm_data, step_key=None):
    if comm_data is None:
        return None
    steps = list(comm_data.keys())
    if not steps:
        return None
    if step_key is None:
        step_key = steps[-1]
    if step_key not in comm_data:
        return None
    step_data = comm_data[step_key]
    collective = step_data.get("collective", {})
    if not collective:
        return None

    total_size_mb = 0.0
    total_time_ms = 0.0
    op_details = []

    for op_name, links in collective.items():
        if "-total@" not in op_name and not op_name.endswith("-total"):
            continue
        max_time = 0.0
        max_size = 0.0
        for link_key, link_data in links.items():
            t = link_data.get("Transit Time(ms)", 0)
            s = link_data.get("Transit Size(MB)", 0)
            if t > max_time:
                max_time = t
                max_size = s
        total_size_mb += max_size
        total_time_ms += max_time
        op_details.append(
            {
                "op_name": op_name,
                "size_mb": round(max_size, 2),
                "time_ms": round(max_time, 3),
            }
        )

    return {
        "step": step_key,
        "total_size_mb": round(total_size_mb, 2),
        "total_time_ms": round(total_time_ms, 3),
        "total_time_us": round(total_time_ms * 1000, 2),
        "ops": op_details,
    }


def load_kernel_details(card_dirs):
    csv_path = os.path.join(card_dirs[0], "ASCEND_PROFILER_OUTPUT", "kernel_details.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到 {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"加载单卡 kernel 记录: {len(df)} 条 (来自 {os.path.basename(card_dirs[0])})")
    return df


def get_data_type_precision(input_data_types_str):
    if not isinstance(input_data_types_str, str):
        return "fp16"
    first_type = input_data_types_str.split(";")[0].strip().replace("DT_", "").upper()
    fp16_kw = ["FP16", "BF16", "HALF", "FLOAT16", "BFLOAT16"]
    fp32_kw = ["FP32", "FLOAT", "FLOAT32", "TF32"]
    if any(kw in first_type for kw in fp16_kw):
        return "fp16"
    elif any(kw in first_type for kw in fp32_kw):
        return "fp32"
    return "fp16"


def get_compute_tflops(device_cache, device_identifier, core_type, precision):
    if device_identifier not in device_cache:
        raise ValueError(f"未找到设备: {device_identifier}")
    cache = device_cache[device_identifier]
    key = f"{core_type}_{precision}"
    val = cache.get(key, 0)
    if pd.isna(val) or val <= 0:
        if precision == "fp32":
            return cache.get(f"{core_type}_fp16", 0)
        return 0
    return val


def get_hbm_bandwidth(device_cache, device_identifier):
    return device_cache[device_identifier]["hbm_bandwidth"]


def is_device_valid(device_cache, device_identifier):
    if device_identifier not in device_cache:
        return False
    cache = device_cache[device_identifier]
    if pd.isna(cache["cube_fp16"]) or cache["cube_fp16"] <= 0:
        return False
    if pd.isna(cache["hbm_bandwidth"]) or cache["hbm_bandwidth"] <= 0:
        return False
    return True


def is_compute_kernel(acc_core):
    if not isinstance(acc_core, str):
        return False
    return acc_core.upper().strip() in ["AI_CORE", "AI_VECTOR_CORE", "MIX_AIC", "MIX_AIV"]


def get_kernel_core_type(acc_core):
    if not isinstance(acc_core, str):
        return "skip"
    acc = acc_core.upper().strip()
    if acc == "AI_CORE":
        return "cube"
    elif acc == "AI_VECTOR_CORE":
        return "vector"
    elif acc in ("MIX_AIC", "MIX_AIV"):
        return "mix"
    return "skip"


def get_microarch_boost(kernel_type, target_id):
    if "950" not in target_id:
        return 1.0
    t = kernel_type.lower()
    if "matmul" in t or "mm" in t:
        return 1.2
    if "flashattention" in t or "flash_attention" in t or "fa" == t:
        return 2.0
    return 1.0


def get_effective_hbm_bandwidth(device_cache, device_identifier, source_id):
    bw = get_hbm_bandwidth(device_cache, device_identifier)
    src_vendor = device_cache[source_id]["row"].get("vendor", "")
    tgt_vendor = device_cache[device_identifier]["row"].get("vendor", "")
    if src_vendor == "Huawei" and tgt_vendor == "Huawei":
        if "950PR" in device_identifier:
            return get_hbm_bandwidth(device_cache, source_id)
    return bw


def run_prediction(df_kernels, devices_df, device_cache, source_id, step_stats=None, comm_stats=None):
    print("\n" + "=" * 80)
    print(f"性能预测: 基线设备 = {source_id}")
    print("=" * 80)

    compute_mask = df_kernels["Accelerator Core"].apply(is_compute_kernel)
    df_compute = df_kernels[compute_mask].copy()
    compute_count = len(df_compute)
    print(f"\n计算类算子 (cube/vector/mix): {compute_count} 条")

    baseline_total = df_compute["Duration(us)"].sum()
    print(f"基线计算类总耗时: {baseline_total:>15.2f} us")

    kernel_core_types = df_compute["Accelerator Core"].apply(get_kernel_core_type).values
    kernel_precisions = df_compute.get("Input Data Types", "").apply(get_data_type_precision).values
    kernel_durations = df_compute["Duration(us)"].values
    kernel_types = df_compute.get("Type", "").values
    kernel_names = df_compute.get("Name", df_compute.get("Task Type", "")).values
    kernel_acc_cores = df_compute.get("Accelerator Core", "").values
    kernel_input_types = df_compute.get("Input Data Types", "").values

    aic_mac_vals = df_compute.get("aic_mac_time(us)", pd.Series([0] * compute_count)).fillna(0).values
    aic_mte1_vals = df_compute.get("aic_mte1_time(us)", pd.Series([0] * compute_count)).fillna(0).values
    aic_mte2_vals = df_compute.get("aic_mte2_time(us)", pd.Series([0] * compute_count)).fillna(0).values
    aiv_vec_vals = df_compute.get("aiv_vec_time(us)", pd.Series([0] * compute_count)).fillna(0).values
    aiv_mte2_vals = df_compute.get("aiv_mte2_time(us)", pd.Series([0] * compute_count)).fillna(0).values
    aiv_mte3_vals = df_compute.get("aiv_mte3_time(us)", pd.Series([0] * compute_count)).fillna(0).values

    src_cube_fp16 = get_compute_tflops(device_cache, source_id, "cube", "fp16")
    src_cube_fp32 = get_compute_tflops(device_cache, source_id, "cube", "fp32")
    src_vec_fp16 = get_compute_tflops(device_cache, source_id, "vector", "fp16")
    src_vec_fp32 = get_compute_tflops(device_cache, source_id, "vector", "fp32")
    src_bw = get_hbm_bandwidth(device_cache, source_id)
    src_interconnect = device_cache[source_id].get("interconnect_bandwidth", 0)
    src_cpu = device_cache[source_id].get("cpu", 1.0)

    step_computing_pct = 1.0
    step_comm_pct = 0.0
    step_free_pct = 0.0
    step_comm_us = 0.0
    step_free_us = 0.0

    if step_stats:
        step_total_per_card = (
            step_stats["Computing"]["mean"]
            + step_stats["Communication(Not Overlapped)"]["mean"]
            + step_stats["Free"]["mean"]
        )
        step_computing_pct = step_stats["Computing"]["pct_mean"] / 100.0
        step_comm_pct = step_stats["Communication(Not Overlapped)"]["pct_mean"] / 100.0
        step_free_pct = step_stats["Free"]["pct_mean"] / 100.0
        step_comm_us = step_stats["Communication(Not Overlapped)"]["mean"]
        step_free_us = step_stats["Free"]["mean"]
        step_compute_us = step_stats["Computing"]["mean"]

    real_comm_us = 0.0
    wait_comm_us = 0.0
    if comm_stats:
        real_comm_us = comm_stats["total_time_us"]
        wait_comm_us = max(0, step_comm_us - real_comm_us)

    if step_stats:
        print(
            f"\nStep Trace 占比: 计算={step_computing_pct * 100:.2f}%, 通信={step_comm_pct * 100:.2f}%, Free={step_free_pct * 100:.2f}%"
        )
        if comm_stats:
            print(f"通信拆分: 未掩盖={step_comm_us:.2f}us (真实通信={real_comm_us:.2f}us, 等待={wait_comm_us:.2f}us)")

    target_devices = devices_df[devices_df["identifier"] != source_id]
    results = []
    all_detail_rows = [{} for _ in range(compute_count)]

    for _, dev_row in target_devices.iterrows():
        target_id = dev_row["identifier"]
        if not is_device_valid(device_cache, target_id):
            print(f"  跳过 {target_id}: 关键规格数据缺失")
            continue

        tgt_cube_fp16 = get_compute_tflops(device_cache, target_id, "cube", "fp16")
        tgt_cube_fp32 = get_compute_tflops(device_cache, target_id, "cube", "fp32")
        tgt_vec_fp16 = get_compute_tflops(device_cache, target_id, "vector", "fp16")
        tgt_vec_fp32 = get_compute_tflops(device_cache, target_id, "vector", "fp32")
        tgt_bw = get_effective_hbm_bandwidth(device_cache, target_id, source_id)
        tgt_interconnect = device_cache[target_id].get("interconnect_bandwidth", 0)
        tgt_cpu = device_cache[target_id].get("cpu", 1.0)

        cube_ratio_fp16 = src_cube_fp16 / tgt_cube_fp16 if tgt_cube_fp16 > 0 else 1.0
        cube_ratio_fp32 = src_cube_fp32 / tgt_cube_fp32 if tgt_cube_fp32 > 0 else cube_ratio_fp16
        vec_ratio_fp16 = src_vec_fp16 / tgt_vec_fp16 if tgt_vec_fp16 > 0 else 1.0
        vec_ratio_fp32 = src_vec_fp32 / tgt_vec_fp32 if tgt_vec_fp32 > 0 else vec_ratio_fp16
        memory_ratio = src_bw / tgt_bw if tgt_bw > 0 else 1.0

        print(f"  预测 {target_id}...")

        predicted_total = 0.0
        target_col = f"{target_id}_Duration(us)"

        for i in range(compute_count):
            core_type = kernel_core_types[i]
            if core_type == "skip":
                all_detail_rows[i][target_col] = round(kernel_durations[i], 2)
                continue

            precision = kernel_precisions[i]
            k_type = kernel_types[i] if isinstance(kernel_types[i], str) else ""

            microarch_boost = get_microarch_boost(k_type, target_id)

            if core_type == "cube":
                cube_cr = cube_ratio_fp32 if precision == "fp32" else cube_ratio_fp16
                effective_cr = cube_cr / microarch_boost
                comp_time = aic_mac_vals[i]
                mem_time = max(aic_mte1_vals[i], aic_mte2_vals[i])
                total_time = comp_time + mem_time
                if total_time > 0:
                    new_time = comp_time * effective_cr + mem_time * memory_ratio
                    scale = new_time / total_time
                else:
                    scale = 1.0
            elif core_type == "vector":
                vec_cr = vec_ratio_fp32 if precision == "fp32" else vec_ratio_fp16
                comp_time = aiv_vec_vals[i]
                mem_time = max(aiv_mte2_vals[i], aiv_mte3_vals[i])
                total_time = comp_time + mem_time
                if total_time > 0:
                    new_time = comp_time * vec_cr + mem_time * memory_ratio
                    scale = new_time / total_time
                else:
                    scale = 1.0
            elif core_type == "mix":
                cube_cr = cube_ratio_fp32 if precision == "fp32" else cube_ratio_fp16
                vec_cr = vec_ratio_fp32 if precision == "fp32" else vec_ratio_fp16
                effective_cube_cr = cube_cr / microarch_boost
                comp_time = max(aic_mac_vals[i], aiv_vec_vals[i])
                mem_time = max(aic_mte1_vals[i], aic_mte2_vals[i], aiv_mte2_vals[i], aiv_mte3_vals[i])
                total_time = comp_time + mem_time
                if total_time > 0:
                    compute_cr = effective_cube_cr if aic_mac_vals[i] >= aiv_vec_vals[i] else vec_cr
                    new_time = comp_time * compute_cr + mem_time * memory_ratio
                    scale = new_time / total_time
                else:
                    scale = 1.0
            else:
                scale = 1.0

            pred = kernel_durations[i] * scale
            predicted_total += pred
            all_detail_rows[i][target_col] = round(pred, 2)

        compute_speedup = baseline_total / predicted_total if predicted_total > 0 else 1.0

        if step_stats:
            comm_speedup = tgt_interconnect / src_interconnect if src_interconnect > 0 else 1.0
            free_speedup = tgt_cpu / src_cpu if src_cpu > 0 else 1.0

            pred_real_comm_us = real_comm_us / comm_speedup
            pred_comm_us = pred_real_comm_us + wait_comm_us
            pred_free_us = step_free_us / free_speedup
            pred_compute_us = predicted_total / baseline_total * step_compute_us

            pred_total_per_card = pred_compute_us + pred_comm_us + pred_free_us
            overall_speedup = step_total_per_card / pred_total_per_card if pred_total_per_card > 0 else compute_speedup

            comm_detail = f"通信={pred_real_comm_us:.2f}us, 等待={wait_comm_us:.2f}us" if comm_stats else ""
        else:
            overall_speedup = compute_speedup
            pred_total_per_card = predicted_total
            pred_compute_us = predicted_total
            pred_comm_us = 0
            pred_free_us = 0
            comm_detail = ""

        pred_total_us = pred_compute_us + pred_comm_us + pred_free_us

        results.append(
            {
                "设备": target_id,
                "架构": dev_row["architecture_type"],
                "Computing(us)": round(pred_compute_us, 2),
                "Comm Not Overlapped(us)": round(pred_comm_us, 2),
                "FreeTime(us)": round(pred_free_us, 2),
                "Total Time(us)": round(pred_total_us, 2),
                "Speedup": round(overall_speedup, 2),
                "通信详情": comm_detail,
            }
        )

    results_df = pd.DataFrame(results)
    if results_df.empty:
        print("\n警告: 没有可预测的目标设备")
        return results_df

    results_df = results_df.sort_values("Speedup", ascending=False)

    baseline_comm_detail = ""
    if comm_stats:
        baseline_comm_detail = f"通信={real_comm_us:.2f}us, 等待={wait_comm_us:.2f}us"

    baseline_total_us = (step_compute_us + step_comm_us + step_free_us) if step_stats else baseline_total

    baseline_row = pd.DataFrame(
        [
            {
                "设备": f"{source_id} (基线)",
                "架构": device_cache[source_id]["row"].get("architecture_type", ""),
                "Computing(us)": round(step_compute_us, 2) if step_stats else round(baseline_total, 2),
                "Comm Not Overlapped(us)": round(step_comm_us, 2) if step_stats else 0,
                "FreeTime(us)": round(step_free_us, 2) if step_stats else 0,
                "Total Time(us)": round(baseline_total_us, 2),
                "Speedup": 1.0,
                "通信详情": baseline_comm_detail,
            }
        ]
    )
    results_df = pd.concat([results_df, baseline_row], ignore_index=True)

    COL_W = [24, 5, 13, 20, 13, 13, 8]
    COL_HEADERS = ["设备", "架构", "Computing", "Comm Not Overlapped", "FreeTime", "Total Time", "Speedup"]

    print("\n" + "=" * 80)
    print("预测结果: 各芯片相对性能对比 (基线=1.0)")
    print("=" * 80)
    print()

    header_parts = [pad(h, w, "right" if h != "设备" else "left") for h, w in zip(COL_HEADERS, COL_W)]
    print("  ".join(header_parts))
    print("-" * (sum(COL_W) + 2 * (len(COL_W) - 1)))

    def _fmt_row(name, arch, comp, comm, free, total, speedup):
        vals = [
            pad(name[:24], COL_W[0], "left"),
            pad(arch, COL_W[1], "right"),
            pad(f"{comp:.0f}", COL_W[2], "right"),
            pad(f"{comm:.0f}", COL_W[3], "right"),
            pad(f"{free:.0f}", COL_W[4], "right"),
            pad(f"{total:.0f}", COL_W[5], "right"),
            pad(f"{speedup:.2f}x", COL_W[6], "right"),
        ]
        return "  ".join(vals)

    for _, r in results_df.iterrows():
        print(
            _fmt_row(
                r['设备'],
                r['架构'],
                r['Computing(us)'],
                r['Comm Not Overlapped(us)'],
                r['FreeTime(us)'],
                r['Total Time(us)'],
                r['Speedup'],
            )
        )

    print()
    print(f"基线 ({source_id}) = 1.00x (计算类耗时 {baseline_total:.2f} us)")
    if step_stats:
        print(
            f"Step Trace: 计算={step_computing_pct * 100:.2f}%, 通信={step_comm_pct * 100:.2f}%, Free={step_free_pct * 100:.2f}%"
        )
        if comm_stats:
            print(
                f"通信拆分: 未掩盖通信 {step_comm_us:.2f}us (真实通信={real_comm_us:.2f}us, 等待={wait_comm_us:.2f}us)"
            )

    results_df.to_csv(os.path.join(OUTPUT_DIR, "performance_prediction.csv"), encoding="utf-8-sig", index=False)

    detail_rows = []
    for i in range(compute_count):
        row_data = {
            "算子名称": kernel_names[i],
            "类型(Type)": kernel_types[i],
            "Accelerator Core": kernel_acc_cores[i],
            "核心分类": kernel_core_types[i],
            "Input Data Types": kernel_input_types[i],
            "精度": kernel_precisions[i],
            "原始Duration(us)": round(kernel_durations[i], 2),
        }
        row_data.update(all_detail_rows[i])
        detail_rows.append(row_data)

    detail_df = pd.DataFrame(detail_rows)
    detail_df.to_csv(os.path.join(OUTPUT_DIR, "kernel_prediction_detail.csv"), encoding="utf-8-sig", index=False)
    print("\n逐算子预测明细已保存至: kernel_prediction_detail.csv")

    return results_df


def main():
    parser = argparse.ArgumentParser(description="跨芯片性能预测工具")
    parser.add_argument("profile_dir", nargs="?", default=None, help="Profiling 数据目录路径")
    parser.add_argument("--source", "-s", default="Huawei Ascend 910B", help="基线设备标识 (默认: Huawei Ascend 910B)")
    parser.add_argument("--devices", "-d", default=None, help="硬件规格 CSV 文件路径 (默认: devices_info.csv)")
    args = parser.parse_args()

    profile_dir = get_profile_dir(args.profile_dir, script_name="performance_predictor.py")

    print("=" * 80)
    print("跨芯片性能预测工具")
    print(f"数据目录: {profile_dir}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 80)

    devices_df, device_cache = load_devices_info(args.devices)
    source_id = args.source

    if source_id not in devices_df["identifier"].values:
        print(f"错误: 基线设备 '{source_id}' 不在硬件规格表中")
        print("可用设备:")
        for ident in devices_df["identifier"]:
            print(f"  {ident}")
        sys.exit(1)

    source_row = devices_df[devices_df["identifier"] == source_id].iloc[0]
    print(f"\n基线设备: {source_id}")
    print(f"  架构: {source_row['architecture_type']}")
    print(f"  Cube FP16: {source_row['cube_fp16_tflops']} TFLOPS")
    print(f"  Vector FP16: {source_row['vector_fp16_tflops']} TFLOPS")
    print(f"  HBM带宽: {source_row['hbm_bandwidth']} TB/s")
    print(f"  互联带宽: {source_row.get('interconnect_bandwidth', 0)} GB/s")
    print(f"  CPU系数: {source_row.get('cpu', 1.0)}")

    card_dirs = find_card_dirs(profile_dir)
    if not card_dirs:
        print("错误: 未找到任何 ascend_pt 数据目录")
        sys.exit(1)

    step_df = load_step_trace(card_dirs)
    step_stats = extract_step_trace_stats(step_df)
    if step_stats:
        print(
            f"\nStep Trace: 计算={step_stats['Computing']['pct_mean']:.2f}%, "
            f"通信={step_stats['Communication(Not Overlapped)']['pct_mean']:.2f}%, "
            f"Free={step_stats['Free']['pct_mean']:.2f}%"
        )
    else:
        print("\n未找到 step_trace_time.csv, 仅进行计算类预测")

    comm_data = load_communication_matrix(card_dirs)
    comm_stats = None
    if comm_data:
        step_key = None
        if step_df is not None and "Step" in step_df.columns:
            step_key = f"step{int(step_df['Step'].iloc[0])}"
        comm_stats = extract_comm_stats(comm_data, step_key)
        if comm_stats:
            print(
                f"通信矩阵: Step={comm_stats['step']}, "
                f"真实通信={comm_stats['total_time_us']:.2f}us, "
                f"通信量={comm_stats['total_size_mb']:.2f}MB"
            )
    else:
        print("未找到 communication_matrix.json")

    df_kernels = load_kernel_details(card_dirs)
    run_prediction(df_kernels, devices_df, device_cache, source_id, step_stats=step_stats, comm_stats=comm_stats)

    print("\n" + "=" * 80)
    print("预测完成!")
    print(f"结果文件已保存至: {OUTPUT_DIR}")
    print("  - performance_prediction.csv  (各芯片汇总对比)")
    print("  - kernel_prediction_detail.csv (逐算子明细)")
    print("=" * 80)


if __name__ == "__main__":
    main()
