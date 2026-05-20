#!/usr/bin/env python

# pylint: disable=logging-fstring-interpolation, logging-not-lazy, too-many-lines, consider-using-dict-items, unnecessary-dunder-call, duplicate-code

"""
视频解码器性能对比脚本
支持三种对比模式:
  1. 抽帧解码对比 (默认) - 测试指定时间戳的帧提取性能
  2. 全量解码对比 (--full_decode) - 测试完整视频所有帧的解码性能

使用方法:
    # 抽帧解码对比
    python video_backend.py --video_path <video_path> --num_iterations 100

    # 全量解码对比
    python video_backend.py --video_path <video_path> --full_decode --num_iterations 10

示例:
    python video_backend.py \
        --video_path /path/to/video.mp4 \
        --num_iterations 100 \
        --timestamps 0.0 0.5 1.0 1.5 2.0

    python video_backend.py \
        --video_path /path/to/video.mp4 \
        --full_decode \
        --num_iterations 5
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

from torchcodec_decoder import test_performance as test_torchcodec
from torchcodec_decoder import test_full_decode_performance as test_torchcodec_full

from pyav_decoder import test_performance as test_pyav
from pyav_decoder import test_full_decode_performance as test_pyav_full

from npu_decoder import test_performance as test_dvpp
from npu_decoder import test_full_decode_performance as test_dvpp_full

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compare_decoders(
    video_path: str,
    timestamps: List[float],
    tolerance_s: float = 1e-4,
    num_iterations: int = 100,
) -> None:
    """
    对比 torchcodec、pyav 和 dvpp 抽帧解码性能

    Args:
        video_path: 视频文件路径
        timestamps: 时间戳列表
        tolerance_s: 时间容差
        num_iterations: 迭代次数
    """
    logger.info("=" * 80)
    logger.info("视频解码器性能对比（抽帧解码模式）")
    logger.info("=" * 80)
    logger.info(f"视频文件:   {video_path}")
    logger.info(f"时间戳:     {timestamps}")
    logger.info(f"迭代次数:   {num_iterations}")
    logger.info("=" * 80)

    results = []

    pyav_result = test_pyav(video_path, timestamps, tolerance_s, num_iterations)
    results.append(pyav_result)

    torchcodec_cache_result = test_torchcodec(video_path, timestamps, tolerance_s, num_iterations, use_cache=True)
    results.append(torchcodec_cache_result)

    torchcodec_no_cache_result = test_torchcodec(video_path, timestamps, tolerance_s, num_iterations, use_cache=False)
    results.append(torchcodec_no_cache_result)

    dvpp_result = test_dvpp(video_path, timestamps, tolerance_s, num_iterations)
    if dvpp_result is not None:
        results.append(dvpp_result)

    _print_frame_comparison_table(results, pyav_result['avg_time'])


def compare_full_decode(
    video_path: str,
    num_iterations: int = 10,
) -> None:
    """
    对比 torchcodec、pyav 和 dvpp 全量视频解码性能

    Args:
        video_path: 视频文件路径
        num_iterations: 迭代次数
    """
    logger.info("=" * 80)
    logger.info("视频解码器性能对比（全量解码模式）")
    logger.info("=" * 80)
    logger.info(f"视频文件:   {video_path}")
    logger.info(f"迭代次数:   {num_iterations}")
    logger.info("=" * 80)

    results = []

    logger.info("\n>>> 测试 PyAV 全量解码 ...")
    pyav_result = test_pyav_full(video_path, num_iterations)
    if pyav_result:
        results.append(pyav_result)

    logger.info("\n>>> 测试 TorchCodec (cache) 全量解码 ...")
    torchcodec_cache_result = test_torchcodec_full(video_path, num_iterations, use_cache=True)
    if torchcodec_cache_result:
        results.append(torchcodec_cache_result)

    logger.info("\n>>> 测试 TorchCodec (no cache) 全量解码 ...")
    torchcodec_no_cache_result = test_torchcodec_full(video_path, num_iterations, use_cache=False)
    if torchcodec_no_cache_result:
        results.append(torchcodec_no_cache_result)

    logger.info("\n>>> 测试 DVPP 全量解码 ...")
    dvpp_result = test_dvpp_full(video_path, num_iterations)
    if dvpp_result is not None:
        results.append(dvpp_result)

    _print_full_decode_comparison_table(results)


def _print_frame_comparison_table(results: list, base_time: float) -> None:
    logger.info("\n" + "=" * 80)
    logger.info("抽帧解码性能对比汇总")
    logger.info("=" * 80)

    print(f"\n{'Backend':<30} {'Avg (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12} {'Speedup':<10}")
    print("-" * 80)

    for result in results:
        speedup = base_time / result['avg_time']
        backend_name = result['backend']

        if 'use_cache' in result:
            backend_name = f"torchcodec ({'cache' if result['use_cache'] else 'no cache'})"

        print(
            f"{backend_name:<30} "
            f"{result['avg_time'] * 1000:<12.2f} "
            f"{result['min_time'] * 1000:<12.2f} "
            f"{result['max_time'] * 1000:<12.2f} "
            f"{speedup:.2f}x"
        )

    print("\n" + "=" * 80)


def _print_full_decode_comparison_table(results: list) -> None:
    logger.info("\n" + "=" * 80)
    logger.info("全量解码性能对比汇总")
    logger.info("=" * 80)

    if not results:
        logger.error("没有有效的测试结果")
        return

    print(
        f"\n{'Backend':<25} {'Total Frames':<13} {'Avg (s)':<10} "
        f"{'Min (s)':<10} {'Max (s)':<10} {'Decode FPS':<12} {'Speedup':<10}"
    )
    print("-" * 100)

    base_time = results[0]['avg_time']

    for r in results:
        speedup = base_time / r['avg_time']
        backend_name = r['backend']
        if 'use_cache' in r:
            backend_name += f" ({'cache' if r['use_cache'] else 'no cache'})"

        print(
            f"{backend_name:<25} "
            f"{r['total_frames']:<13} "
            f"{r['avg_time']:<10.2f} "
            f"{r['min_time']:<10.2f} "
            f"{r['max_time']:<10.2f} "
            f"{r['decode_fps']:<12.1f} "
            f"{speedup:.2f}x"
        )

    print("\n" + "=" * 100)


def main():
    parser = argparse.ArgumentParser(description="视频解码器性能对比")
    parser.add_argument("--video_path", type=str, required=True, help="视频文件路径")
    parser.add_argument(
        "--timestamps", type=float, nargs='+', default=[0.0, 0.5, 1.0, 1.5, 2.0], help="要解码的时间戳列表（秒）"
    )
    parser.add_argument("--tolerance_s", type=float, default=1e-4, help="时间容差（秒）")
    parser.add_argument("--num_iterations", type=int, default=100, help="迭代次数")
    parser.add_argument("--full_decode", action="store_true", help="全量视频解码对比模式")

    args = parser.parse_args()

    if not Path(args.video_path).exists():
        logger.error(f"视频文件不存在: {args.video_path}")
        sys.exit(1)

    if args.full_decode:
        compare_full_decode(
            args.video_path,
            args.num_iterations,
        )
    else:
        compare_decoders(
            args.video_path,
            args.timestamps,
            args.tolerance_s,
            args.num_iterations,
        )


if __name__ == "__main__":
    main()
