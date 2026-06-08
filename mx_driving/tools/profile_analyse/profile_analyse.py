import os
import sys
import argparse
from pathlib import Path
import pandas as pd

from profiling_common import find_card_dirs, get_profile_dir, load_step_trace, pad

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = str(SCRIPT_DIR)


def analyze_step_trace(df):
    print("\n" + "=" * 80)
    print("1. Step Trace 分析 — 计算 / 未掩盖通信 / FreeTime 耗时统计")
    print("=" * 80)

    cols = {
        "Computing": "Computing",
        "Communication(Not Overlapped)": "Comm(Not Overlapped)",
        "Free": "FreeTime",
    }

    total = df[list(cols.keys())].sum(axis=1)
    for key, label in cols.items():
        df[f"{label}_占比"] = df[key] / total * 100

    print(f"\n--- 各卡耗时明细 ({len(df)} 张卡, Step={int(df['Step'].iloc[0])}) ---")
    print()
    COL_W1 = [10, 15, 15, 15]
    COL_H1 = ["指标", "Computing", "Comm", "FreeTime"]
    header_parts = [pad(h, w, "right" if i > 0 else "left") for i, (h, w) in enumerate(zip(COL_H1, COL_W1))]
    print("  ".join(header_parts))
    print("-" * (sum(COL_W1) + 2 * (len(COL_W1) - 1)))
    for _, row in df.iterrows():
        card_short = row["Card"].split("_")[1]
        vals = [
            pad(f"Card {card_short}", COL_W1[0], "left"),
            pad(f"{row['Computing']:.2f}", COL_W1[1], "right"),
            pad(f"{row['Communication(Not Overlapped)']:.2f}", COL_W1[2], "right"),
            pad(f"{row['Free']:.2f}", COL_W1[3], "right"),
        ]
        print("  ".join(vals))
    print("-" * (sum(COL_W1) + 2 * (len(COL_W1) - 1)))
    for label, key in [("Sum", "sum"), ("Mean", "mean"), ("Min", "min"), ("Max", "max")]:
        vals = [
            pad(label, COL_W1[0], "left"),
            pad(f"{getattr(df['Computing'], key)():.2f}", COL_W1[1], "right"),
            pad(f"{getattr(df['Communication(Not Overlapped)'], key)():.2f}", COL_W1[2], "right"),
            pad(f"{getattr(df['Free'], key)():.2f}", COL_W1[3], "right"),
        ]
        print("  ".join(vals))

    agg_stats = {}
    for key, label in cols.items():
        vals = df[key]
        pct_col = f"{label}_占比"
        pcts = df[pct_col]
        agg_stats[label] = {
            "sum": vals.sum(),
            "mean": vals.mean(),
            "min": vals.min(),
            "max": vals.max(),
            "pct_mean": pcts.mean(),
            "pct_min": pcts.min(),
            "pct_max": pcts.max(),
        }

    print("\n--- 数据分析 ---")
    print()
    COL_W2 = [22, 12, 22]
    COL_H2 = ["指标", "耗时占比", "波动(min~max)"]
    header_parts = [pad(h, w, "right" if i > 0 else "left") for i, (h, w) in enumerate(zip(COL_H2, COL_W2))]
    print("  ".join(header_parts))
    print("-" * (sum(COL_W2) + 2 * (len(COL_W2) - 1)))

    for key, label in cols.items():
        s = agg_stats[label]
        range_str = f"{s['pct_min']:.2f}% ~ {s['pct_max']:.2f}%"
        vals = [
            pad(label, COL_W2[0], "left"),
            pad(f"{s['pct_mean']:.2f}%", COL_W2[1], "right"),
            pad(range_str, COL_W2[2], "right"),
        ]
        print("  ".join(vals))

    detail_rows = []
    for _, row in df.iterrows():
        card_short = row["Card"].split("_")[1]
        detail_rows.append(
            {
                "Card": f"Card {card_short}",
                "Computing(us)": round(row["Computing"], 2),
                "Comm(us)": round(row["Communication(Not Overlapped)"], 2),
                "FreeTime(us)": round(row["Free"], 2),
            }
        )
    for label, key in [("Sum", "sum"), ("Mean", "mean"), ("Min", "min"), ("Max", "max")]:
        detail_rows.append(
            {
                "Card": label,
                "Computing(us)": round(getattr(df["Computing"], key)(), 2),
                "Comm(us)": round(getattr(df["Communication(Not Overlapped)"], key)(), 2),
                "FreeTime(us)": round(getattr(df["Free"], key)(), 2),
            }
        )

    summary_rows = [
        {
            "指标": "耗时占比(%)",
            "Computing(%)": round(agg_stats["Computing"]["pct_mean"], 2),
            "Comm(%)": round(agg_stats["Comm(Not Overlapped)"]["pct_mean"], 2),
            "FreeTime(%)": round(agg_stats["FreeTime"]["pct_mean"], 2),
        },
        {
            "指标": "波动min(%)",
            "Computing(%)": round(agg_stats["Computing"]["pct_min"], 2),
            "Comm(%)": round(agg_stats["Comm(Not Overlapped)"]["pct_min"], 2),
            "FreeTime(%)": round(agg_stats["FreeTime"]["pct_min"], 2),
        },
        {
            "指标": "波动max(%)",
            "Computing(%)": round(agg_stats["Computing"]["pct_max"], 2),
            "Comm(%)": round(agg_stats["Comm(Not Overlapped)"]["pct_max"], 2),
            "FreeTime(%)": round(agg_stats["FreeTime"]["pct_max"], 2),
        },
    ]

    detail_df = pd.DataFrame(detail_rows)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.columns = detail_df.columns

    title_row = pd.DataFrame([["Summary", "", "", ""]], columns=detail_df.columns)
    step1_csv = pd.concat([detail_df, title_row, summary_df], axis=0, ignore_index=True)
    step1_csv.to_csv(os.path.join(OUTPUT_DIR, "step_trace_analysis.csv"), encoding="utf-8-sig", index=False)

    return df, agg_stats


AIC_TIME_COLS = [
    "aicore_time(us)",
    "aic_mac_time(us)",
    "aic_scalar_time(us)",
    "aic_mte1_time(us)",
    "aic_mte2_time(us)",
    "aic_fixpipe_time(us)",
]
AIV_TIME_COLS = [
    "aiv_time(us)",
    "aiv_vec_time(us)",
    "aiv_scalar_time(us)",
    "aiv_mte2_time(us)",
    "aiv_mte3_time(us)",
]


def classify_and_breakdown(row):
    acc_core = row["Accelerator Core"]
    dur = row["Duration(us)"]

    aic_mac = row.get("aic_mac_time(us)", 0) or 0
    aic_mte1 = row.get("aic_mte1_time(us)", 0) or 0
    aic_mte2 = row.get("aic_mte2_time(us)", 0) or 0
    aiv_vec = row.get("aiv_vec_time(us)", 0) or 0
    aiv_mte2 = row.get("aiv_mte2_time(us)", 0) or 0
    aiv_mte3 = row.get("aiv_mte3_time(us)", 0) or 0

    if acc_core == "AI_VECTOR_CORE":
        core_type = "vector"
        compute_time = aiv_vec
        memory_time = max(aiv_mte2, aiv_mte3)
    elif acc_core == "AI_CORE":
        core_type = "cube"
        compute_time = aic_mac
        memory_time = max(aic_mte1, aic_mte2)
    elif acc_core in ("MIX_AIC", "MIX_AIV"):
        aic_total = sum(row.get(c, 0) or 0 for c in AIC_TIME_COLS)
        aiv_total = sum(row.get(c, 0) or 0 for c in AIV_TIME_COLS)
        if aic_total > aiv_total:
            core_type = "cube"
            compute_time = aic_mac
            memory_time = max(aic_mte1, aic_mte2)
        else:
            core_type = "vector"
            compute_time = aiv_vec
            memory_time = max(aiv_mte2, aiv_mte3)
    else:
        core_type = "other"
        compute_time = 0
        memory_time = 0

    return pd.Series(
        {
            "core_category": core_type,
            "compute_time": compute_time,
            "memory_time": memory_time,
            "duration": dur,
        }
    )


def analyze_kernel_details(card_dirs):
    print("\n" + "=" * 80)
    print("2. Kernel Details 分析 (仅 cube / vector 计算类算子)")
    print("=" * 80)

    csv_path = os.path.join(card_dirs[0], "ASCEND_PROFILER_OUTPUT", "kernel_details.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到 {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"\n加载单卡 kernel 记录: {len(df)} 条 (来自 {os.path.basename(card_dirs[0])})")

    step_col = None
    for candidate in ["Step", "step", "Step ID", "StepID", "step_id", "StepId", "Step_ID"]:
        if candidate in df.columns:
            step_col = candidate
            break
    if step_col is not None and not df.empty:
        step_series = pd.to_numeric(df[step_col], errors="coerce")
        if step_series.notna().any() and step_series.nunique(dropna=True) > 1:
            last_step = int(step_series.max())
            df = df[step_series == last_step].copy()
            print(f"检测到多个 Step，取最后一个 Step={last_step} 进行分析")

    breakdowns = df.apply(classify_and_breakdown, axis=1)
    df = pd.concat([df, breakdowns], axis=1)

    mixed_count = df[df["Accelerator Core"].isin(["MIX_AIC", "MIX_AIV"])].shape[0]
    mixed_to_cube = df[(df["Accelerator Core"].isin(["MIX_AIC", "MIX_AIV"])) & (df["core_category"] == "cube")].shape[0]
    mixed_to_vector = df[
        (df["Accelerator Core"].isin(["MIX_AIC", "MIX_AIV"])) & (df["core_category"] == "vector")
    ].shape[0]
    print(f"\nMixed 算子重分类: 共 {mixed_count} 条, {mixed_to_cube} 条归入 cube, {mixed_to_vector} 条归入 vector")

    df_compute = df[df["core_category"].isin(["cube", "vector"])].copy()
    print(f"过滤后计算类算子 (cube + vector): {len(df_compute)} 条")

    all_type_stats = df_compute.groupby("Type").agg(
        total_duration=("duration", "sum"),
        count=("duration", "count"),
        compute_time=("compute_time", "sum"),
        memory_time=("memory_time", "sum"),
    )
    all_type_stats["compute_pct"] = all_type_stats["compute_time"] / all_type_stats["total_duration"] * 100
    all_type_stats["memory_pct"] = all_type_stats["memory_time"] / all_type_stats["total_duration"] * 100

    type_core = df_compute.groupby("Type")["core_category"].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else "mixed"
    )
    all_type_stats["计算类型"] = all_type_stats.index.map(type_core)
    all_type_stats = all_type_stats.sort_values("total_duration", ascending=False)
    top20_types = all_type_stats.head(20)

    print("\n--- 每种算子计算耗时与访存带宽耗时占比 ---")
    print()
    COL_W = [24, 7, 7, 13, 13, 13, 9, 9]
    COL_H = ["算子", "类型", "次数", "总耗时", "计算耗时", "访存耗时", "计算%", "访存%"]
    header_parts = [pad(h, w, "right" if i > 0 else "left") for i, (h, w) in enumerate(zip(COL_H, COL_W))]
    print("  ".join(header_parts))
    print("-" * (sum(COL_W) + 2 * (len(COL_W) - 1)))

    for type_name, row_data in top20_types.iterrows():
        vals = [
            pad(type_name[:24], COL_W[0], "left"),
            pad(row_data["计算类型"], COL_W[1], "right"),
            pad(str(int(row_data["count"])), COL_W[2], "right"),
            pad(f"{row_data['total_duration']:.2f}", COL_W[3], "right"),
            pad(f"{row_data['compute_time']:.2f}", COL_W[4], "right"),
            pad(f"{row_data['memory_time']:.2f}", COL_W[5], "right"),
            pad(f"{row_data['compute_pct']:.2f}%", COL_W[6], "right"),
            pad(f"{row_data['memory_pct']:.2f}%", COL_W[7], "right"),
        ]
        print("  ".join(vals))

    top20_save = top20_types.copy()
    top20_save = top20_save.reset_index()
    top20_save = top20_save.rename(columns={"Type": "算子", "index": "算子"})
    for col in ["total_duration", "compute_time", "memory_time", "compute_pct", "memory_pct"]:
        top20_save[col] = top20_save[col].round(2)
    top20_save["count"] = top20_save["count"].astype(int)
    top20_save = top20_save[
        ["算子", "计算类型", "count", "total_duration", "compute_time", "memory_time", "compute_pct", "memory_pct"]
    ]
    top20_save.columns = [
        "算子",
        "计算类型",
        "调用次数",
        "总耗时(us)",
        "计算耗时(us)",
        "访存耗时(us)",
        "计算占比(%)",
        "访存占比(%)",
    ]
    top20_save.to_csv(
        os.path.join(OUTPUT_DIR, "top20_ops_compute_memory_breakdown.csv"), encoding="utf-8-sig", index=False
    )

    core_summary = df_compute.groupby("core_category").agg(
        total_duration=("duration", "sum"),
        count=("duration", "count"),
        compute_time=("compute_time", "sum"),
        memory_time=("memory_time", "sum"),
    )
    core_summary["compute_pct"] = core_summary["compute_time"] / core_summary["total_duration"] * 100
    core_summary["memory_pct"] = core_summary["memory_time"] / core_summary["total_duration"] * 100
    core_summary["总耗时占比(%)"] = core_summary["total_duration"] / core_summary["total_duration"].sum() * 100
    core_summary = core_summary.round(2)

    total_compute = df_compute["compute_time"].sum()
    total_memory = df_compute["memory_time"].sum()
    total_dur = df_compute["duration"].sum()
    comp_pct = total_compute / (total_compute + total_memory) * 100
    mem_pct = total_memory / (total_compute + total_memory) * 100

    cube_dur = core_summary.loc["cube", "total_duration"] if "cube" in core_summary.index else 0
    vector_dur = core_summary.loc["vector", "total_duration"] if "vector" in core_summary.index else 0
    cube_pct = cube_dur / total_dur * 100 if total_dur > 0 else 0
    vector_pct = vector_dur / total_dur * 100 if total_dur > 0 else 0

    print("\n" + "=" * 80)
    print("3. 整网计算耗时 vs 访存带宽耗时比例")
    print("=" * 80)
    print(f"\n  计算类算子总 Duration:  {total_dur:>12.2f} us")
    print()
    print("  +-- 计算 & 访存对比")
    print(f"  |   +-- 计算耗时:  {total_compute:>12.2f} us  --- {comp_pct:.2f}%")
    print(f"  |   +-- 访存耗时:  {total_memory:>12.2f} us  --- {mem_pct:.2f}%")
    print("  +-- Cube & Vector 对比")
    print(f"  |   +-- Cube 耗时:  {cube_dur:>12.2f} us  --- {cube_pct:.2f}%")
    print(f"  |   +-- Vector 耗时:{vector_dur:>12.2f} us  --- {vector_pct:.2f}%")

    section1_rows = []
    for cat in ["cube", "vector"]:
        if cat in core_summary.index:
            r = core_summary.loc[cat]
            section1_rows.append(
                {
                    "类别": cat,
                    "总耗时(us)": r["total_duration"],
                    "算子数": int(r["count"]),
                    "计算耗时(us)": r["compute_time"],
                    "访存耗时(us)": r["memory_time"],
                    "计算占比(%)": r["compute_pct"],
                    "访存占比(%)": r["memory_pct"],
                    "总耗时占比(%)": r["总耗时占比(%)"],
                }
            )
    s1_df = pd.DataFrame(section1_rows)

    ratio_rows = [
        {
            "对比项": "计算:访存",
            "耗时A(us)": round(total_compute, 2),
            "耗时B(us)": round(total_memory, 2),
            "占比A(%)": round(comp_pct, 2),
            "占比B(%)": round(mem_pct, 2),
            "比例": f"{comp_pct:.2f}% : {mem_pct:.2f}%",
        },
        {
            "对比项": "Cube:Vector",
            "耗时A(us)": round(cube_dur, 2),
            "耗时B(us)": round(vector_dur, 2),
            "占比A(%)": round(cube_pct, 2),
            "占比B(%)": round(vector_pct, 2),
            "比例": f"{cube_pct:.2f}% : {vector_pct:.2f}%",
        },
    ]
    s2_df = pd.DataFrame(ratio_rows)

    max_cols = max(len(s1_df.columns), len(s2_df.columns))
    s1_cols = list(s1_df.columns)
    s2_cols = list(s2_df.columns)

    with open(os.path.join(OUTPUT_DIR, "computing_summary.csv"), "w", encoding="utf-8-sig", newline="") as f:
        f.write(",".join(s1_cols) + "\n")
        for _, row_data in s1_df.iterrows():
            f.write(",".join(str(v) for v in row_data.values) + "\n")
        f.write(",".join(["Ratio Summary"] + [""] * (max_cols - 1)) + "\n")
        f.write(",".join(s2_cols) + "\n")
        for _, row_data in s2_df.iterrows():
            f.write(",".join(str(v) for v in row_data.values) + "\n")

    return df_compute


def main():
    parser = argparse.ArgumentParser(description="Ascend Profiling 数据分析")
    parser.add_argument(
        "profile_dir", nargs="?", default=None, help="Profiling 数据目录路径 (例如: D:\\profile\\gr00t-n1.6)"
    )
    args = parser.parse_args()

    profile_dir = get_profile_dir(
        args.profile_dir,
        script_name="profile_analyse.py",
        extra_usage_lines=[
            "用法: python profile_analyse.py <profile_dir>",
            "  或设置环境变量: set PROFILE_DIR=D:\\profile\\gr00t-n1.6",
        ],
    )

    print("=" * 80)
    print("Ascend Profiling 数据分析")
    print(f"数据目录: {profile_dir}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 80)

    card_dirs = find_card_dirs(profile_dir)
    if not card_dirs:
        print("错误: 未找到任何 ascend_pt 数据目录")
        sys.exit(1)

    step_df = load_step_trace(card_dirs, require=True)
    analyze_step_trace(step_df)

    analyze_kernel_details(card_dirs)

    print("\n" + "=" * 80)
    print("分析完成!")
    print(f"结果文件已保存至: {OUTPUT_DIR}")
    print("  - step_trace_analysis.csv")
    print("  - top20_ops_compute_memory_breakdown.csv")
    print("  - computing_summary.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()
