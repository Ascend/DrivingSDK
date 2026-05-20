#!/usr/bin/env python

# pylint: disable=logging-fstring-interpolation, logging-not-lazy, too-many-lines, consider-using-dict-items, unnecessary-dunder-call, duplicate-code

"""
PyAV 视频解码器脚本
基于 torchvision 的 pyav 后端，可独立调用进行视频解码和性能测试

使用方法:
    python pyav_decoder.py --video_path <video_path> --timestamps 0.0 0.5 1.0 1.5 2.0

示例:
    python pyav_decoder.py \
        --video_path /path/to/video.mp4 \
        --timestamps 0.0 0.5 1.0 1.5 2.0 \
        --num_iterations 100
"""

import argparse
import time
import logging
import sys
from pathlib import Path
from typing import List

import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from lerobot.datasets.video_utils import decode_video_frames_torchvision

    PYAV_AVAILABLE = True
except ImportError as e:
    PYAV_AVAILABLE = False
    logger.error(f"lerobot.datasets.video_utils 导入失败: {e}")

try:
    import av

    PYAV_NATIVE_AVAILABLE = True
except ImportError:
    PYAV_NATIVE_AVAILABLE = False
    logger.warning("av 原生库不可用，全量解码将回退到时间戳模式")


def decode_video_frames(
    video_path: str,
    timestamps: List[float],
    tolerance_s: float = 1e-4,
) -> torch.Tensor:
    """
    使用 PyAV 解码视频帧

    Args:
        video_path: 视频文件路径
        timestamps: 时间戳列表（秒）
        tolerance_s: 时间容差（秒）

    Returns:
        解码后的视频帧张量
    """
    if not PYAV_AVAILABLE:
        raise RuntimeError("PyAV 后端不可用，请检查 lerobot 安装")

    if not Path(video_path).exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    frames = decode_video_frames_torchvision(
        video_path,
        timestamps,
        tolerance_s,
        backend="pyav",
    )

    logger.info(f"解码完成: {len(timestamps)} 帧, shape={frames.shape}")
    return frames


def get_video_metadata(video_path: str) -> dict:
    """获取视频元数据（时长、FPS、总帧数）"""
    if not PYAV_NATIVE_AVAILABLE:
        return {'duration_s': 0, 'fps': 0, 'total_frames': 0}

    container = av.open(video_path)
    video_stream = container.streams.video[0]
    metadata = {
        'duration_s': float(video_stream.duration * video_stream.time_base) if video_stream.duration else 0,
        'fps': float(video_stream.average_rate) if video_stream.average_rate else 0,
        'total_frames': video_stream.frames if video_stream.frames else 0,
    }
    container.close()
    return metadata


def decode_video_all_frames(video_path: str) -> torch.Tensor:
    """
    使用 PyAV 全量解码视频所有帧

    Args:
        video_path: 视频文件路径

    Returns:
        解码后的所有帧张量 [total_frames, C, H, W]
    """
    if not PYAV_AVAILABLE:
        raise RuntimeError("PyAV 后端不可用，请检查 lerobot 安装")

    if not Path(video_path).exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    if PYAV_NATIVE_AVAILABLE:
        container = av.open(video_path)
        video_stream = container.streams.video[0]
        frames = []

        for packet in container.demux(video_stream):
            for frame in packet.decode():
                img = frame.to_ndarray(format='rgb24')
                frames.append(torch.from_numpy(img).permute(2, 0, 1))

        container.close()
        frames_tensor = torch.stack(frames, dim=1)
        logger.info(f"全量解码完成: {len(frames)} 帧, shape={frames_tensor.shape}")
        return frames_tensor
    else:
        metadata = get_video_metadata(video_path)
        timestamps = [i / metadata['fps'] for i in range(metadata['total_frames'])]
        return decode_video_frames(video_path, timestamps, 1e-4)


def test_full_decode_performance(
    video_path: str,
    num_iterations: int = 10,
) -> dict:
    """
    测试 PyAV 全量视频解码性能

    Args:
        video_path: 视频文件路径
        num_iterations: 迭代次数

    Returns:
        性能统计字典
    """
    if not PYAV_AVAILABLE:
        raise RuntimeError("PyAV 后端不可用")

    metadata = get_video_metadata(video_path)
    logger.info(
        f"PyAV 全量解码测试: iterations={num_iterations}, "
        f"duration={metadata['duration_s']:.1f}s, "
        f"total_frames={metadata['total_frames']}"
    )

    logger.info("预热中...")
    decode_video_all_frames(video_path)

    logger.info(f"运行 {num_iterations} 次全量解码...")
    start_time = time.time()
    decode_times = []

    for i in range(num_iterations):
        iter_start = time.time()
        frames = decode_video_all_frames(video_path)
        iter_time = time.time() - iter_start
        decode_times.append(iter_time)

        logger.info(f"第 {i + 1}/{num_iterations} 次: {iter_time:.2f}s, fps={frames.shape[1] / iter_time:.1f}")

    total_time = time.time() - start_time

    avg_time = sum(decode_times) / len(decode_times)
    min_time = min(decode_times)
    max_time = max(decode_times)

    result = {
        'backend': 'pyav',
        'mode': 'full_decode',
        'total_time': total_time,
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time,
        'num_iterations': num_iterations,
        'total_frames': frames.shape[1],
        'frames_shape': tuple(frames.shape),
        'video_duration_s': metadata['duration_s'],
        'video_fps': metadata['fps'],
        'decode_fps': frames.shape[1] / avg_time,
    }

    logger.info(
        f"PyAV 全量解码结果: avg={avg_time:.2f}s, "
        f"min={min_time:.2f}s, max={max_time:.2f}s, "
        f"decode_fps={result['decode_fps']:.1f}"
    )

    return result


def test_performance(
    video_path: str,
    timestamps: List[float],
    tolerance_s: float = 1e-4,
    num_iterations: int = 100,
) -> dict:
    """
    测试 PyAV 解码器性能

    Args:
        video_path: 视频文件路径
        timestamps: 时间戳列表
        tolerance_s: 时间容差
        num_iterations: 迭代次数

    Returns:
        性能统计字典
    """
    if not PYAV_AVAILABLE:
        raise RuntimeError("PyAV 后端不可用")

    logger.info(f"PyAV 性能测试: iterations={num_iterations}")

    logger.info("预热中...")
    decode_video_frames(video_path, timestamps, tolerance_s)

    logger.info(f"运行 {num_iterations} 次迭代...")
    start_time = time.time()
    decode_times = []

    for i in range(num_iterations):
        iter_start = time.time()
        frames = decode_video_frames(video_path, timestamps, tolerance_s)
        iter_time = time.time() - iter_start
        decode_times.append(iter_time)

        if (i + 1) % 10 == 0:
            logger.info(f"已完成 {i + 1}/{num_iterations} 次迭代")

    total_time = time.time() - start_time

    avg_time = sum(decode_times) / len(decode_times)
    min_time = min(decode_times)
    max_time = max(decode_times)

    result = {
        'backend': 'pyav',
        'total_time': total_time,
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time,
        'num_iterations': num_iterations,
        'frames_shape': frames.shape,
    }

    logger.info(f"PyAV 结果: avg={avg_time * 1000:.2f}ms, min={min_time * 1000:.2f}ms, max={max_time * 1000:.2f}ms")

    return result


def main():
    parser = argparse.ArgumentParser(description="PyAV 视频解码器")
    parser.add_argument("--video_path", type=str, required=True, help="视频文件路径")
    parser.add_argument(
        "--timestamps", type=float, nargs='+', default=[0.0, 0.5, 1.0, 1.5, 2.0], help="要解码的时间戳列表（秒）"
    )
    parser.add_argument("--tolerance_s", type=float, default=1e-4, help="时间容差（秒）")
    parser.add_argument("--num_iterations", type=int, default=100, help="性能测试迭代次数")
    parser.add_argument("--test", action="store_true", help="运行性能测试模式")
    parser.add_argument("--full_decode", action="store_true", help="全量视频解码模式")

    args = parser.parse_args()

    if not PYAV_AVAILABLE:
        logger.error("PyAV 后端不可用，无法继续")
        sys.exit(1)

    if not Path(args.video_path).exists():
        logger.error(f"视频文件不存在: {args.video_path}")
        sys.exit(1)

    if args.full_decode:
        result = test_full_decode_performance(
            args.video_path,
            args.num_iterations,
        )
        print(f"\n{'=' * 60}")
        print("PyAV 全量解码测试结果")
        print(f"{'=' * 60}")
        print(f"视频时长:      {result['video_duration_s']:.1f}s")
        print(f"视频FPS:       {result['video_fps']:.1f}")
        print(f"总帧数:        {result['total_frames']}")
        print(f"迭代次数:      {result['num_iterations']}")
        print(f"帧形状:        {result['frames_shape']}")
        print(f"总耗时:        {result['total_time']:.2f}s")
        print(f"平均耗时:      {result['avg_time']:.2f}s")
        print(f"最小耗时:      {result['min_time']:.2f}s")
        print(f"最大耗时:      {result['max_time']:.2f}s")
        print(f"解码FPS:       {result['decode_fps']:.1f}")
        print(f"{'=' * 60}\n")
    elif args.test:
        result = test_performance(
            args.video_path,
            args.timestamps,
            args.tolerance_s,
            args.num_iterations,
        )
        print(f"\n{'=' * 60}")
        print("PyAV 性能测试结果")
        print(f"{'=' * 60}")
        print(f"迭代次数:      {result['num_iterations']}")
        print(f"帧形状:        {result['frames_shape']}")
        print(f"总耗时:        {result['total_time'] * 1000:.2f}ms")
        print(f"平均耗时:      {result['avg_time'] * 1000:.2f}ms")
        print(f"最小耗时:      {result['min_time'] * 1000:.2f}ms")
        print(f"最大耗时:      {result['max_time'] * 1000:.2f}ms")
        print(f"{'=' * 60}\n")
    else:
        frames = decode_video_frames(
            args.video_path,
            args.timestamps,
            args.tolerance_s,
        )
        print(f"解码成功: {frames.shape}")


if __name__ == "__main__":
    main()
