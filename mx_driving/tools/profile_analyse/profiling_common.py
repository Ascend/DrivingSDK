import glob
import os
import sys
import unicodedata

import pandas as pd


def display_width(s):
    w = 0
    for ch in str(s):
        if unicodedata.east_asian_width(ch) in ("W", "F"):
            w += 2
        else:
            w += 1
    return w


def pad(s, width, align="left"):
    dw = display_width(s)
    padding = max(0, width - dw)
    if align == "left":
        return str(s) + " " * padding
    return " " * padding + str(s)


def find_card_dirs(base_dir, *, verbose=True):
    pattern = os.path.join(base_dir, "*_ascend_pt")
    dirs = sorted(glob.glob(pattern))
    if verbose:
        print(f"发现 {len(dirs)} 张卡的数据目录:")
        for d in dirs:
            print(f"  {os.path.basename(d)}")
    return dirs


def load_step_trace(card_dirs, *, verbose=True, require=False):
    all_data = []
    for d in card_dirs:
        csv_path = os.path.join(d, "ASCEND_PROFILER_OUTPUT", "step_trace_time.csv")
        if not os.path.exists(csv_path):
            if verbose:
                print(f"警告: {csv_path} 不存在，跳过")
            continue
        df = pd.read_csv(csv_path)
        df["Card"] = os.path.basename(d)
        all_data.append(df)

    if not all_data:
        if require:
            raise FileNotFoundError("未找到任何 step_trace_time.csv 文件")
        return None

    combined = pd.concat(all_data, ignore_index=True)
    if "Step" in combined.columns and combined["Step"].nunique() > 1:
        last_step = combined["Step"].max()
        if verbose:
            print(f"检测到多个 Step，取最后一个 Step={int(last_step)} 进行分析")
        combined = combined[combined["Step"] == last_step].copy()
    return combined


def extract_step_trace_stats(df):
    if df is None or df.empty:
        return None
    cols = ["Computing", "Communication(Not Overlapped)", "Free"]
    total = df[cols].sum(axis=1)
    stats = {}
    for col in cols:
        vals = df[col]
        pcts = vals / total * 100
        stats[col] = {"sum": vals.sum(), "mean": vals.mean(), "pct_mean": pcts.mean()}
    return stats


def get_profile_dir(profile_dir_arg, *, script_name, extra_usage_lines=None):
    profile_dir = profile_dir_arg if profile_dir_arg is not None else os.environ.get("PROFILE_DIR", "")
    if profile_dir:
        return profile_dir

    print("错误: 请指定 profiling 数据目录路径")
    if extra_usage_lines:
        for line in extra_usage_lines:
            print(line)
    else:
        print(f"用法: python {script_name} <profile_dir>")
    sys.exit(1)
