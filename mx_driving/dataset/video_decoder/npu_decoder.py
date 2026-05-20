#!/usr/bin/env python

# pylint: disable=logging-fstring-interpolation, logging-not-lazy, too-many-lines, consider-using-dict-items, unnecessary-dunder-call, duplicate-code

"""
DVPP 视频解码器脚本
基于华为昇腾 DVPP 硬件加速，可独立调用进行视频解码和性能测试

注意：使用 DVPP 后端需要先 source CANN 环境变量

使用方法:
    python npu_decoder.py --video_path <video_path> --timestamps 0.0 0.5 1.0 1.5 2.0

示例:
    python npu_decoder.py \
        --video_path /path/to/video.mp4 \
        --timestamps 0.0 0.5 1.0 1.5 2.0 \
        --num_iterations 100
"""

import argparse
import time
import logging
import sys
from pathlib import Path
from typing import List, Optional

import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import torch_npu
    import torchvision_npu

    torchvision_npu._set_video_backend('npu')
    torch_npu.npu.current_stream().set_data_preprocess_stream(True)
    DVPP_AVAILABLE = True
    logger.info("DVPP 后端可用")
except ImportError as e:
    DVPP_AVAILABLE = False
    logger.warning(f"DVPP 后端不可用: {e}")

try:
    from lerobot.datasets.video_utils import decode_video_frames_npu

    LEROBOT_AVAILABLE = True
except ImportError as e:
    LEROBOT_AVAILABLE = False
    logger.error(f"lerobot.datasets.video_utils 导入失败: {e}")

try:
    import torchcodec

    DVPP_METADATA_AVAILABLE = True
except ImportError:
    DVPP_METADATA_AVAILABLE = False
    logger.warning("torchcodec 不可用，DVPP 全量解码无法获取视频元数据")


def decode_video_frames(
    video_path: str,
    timestamps: List[float],
    tolerance_s: float = 1e-4,
) -> torch.Tensor:
    """
    使用 DVPP 解码视频帧

    Args:
        video_path: 视频文件路径
        timestamps: 时间戳列表（秒）
        tolerance_s: 时间容差（秒）

    Returns:
        解码后的视频帧张量
    """
    if not DVPP_AVAILABLE:
        raise RuntimeError("DVPP 后端不可用，请先 source CANN 环境变量并确保 torch_npu 已安装")
    if not LEROBOT_AVAILABLE:
        raise RuntimeError("lerobot 库不可用，无法调用 DVPP 解码函数")

    if not Path(video_path).exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    frames = decode_video_frames_npu(video_path, timestamps, tolerance_s)

    logger.info(f"解码完成: {len(timestamps)} 帧, shape={frames.shape}")
    return frames


def get_video_metadata(video_path: str) -> dict:
    """获取视频元数据（时长、FPS、总帧数）"""
    if not DVPP_METADATA_AVAILABLE:
        return {'duration_s': 0, 'fps': 0, 'total_frames': 0}

    decoder = torchcodec.decoders.VideoDecoder(video_path)
    metadata = decoder.metadata
    return {
        'duration_s': metadata.duration_seconds,
        'fps': metadata.average_fps,
        'total_frames': len(decoder),
    }


def decode_video_all_frames(
    video_path: str,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """
    使用 DVPP 全量解码视频所有帧
    通过分块解码避免 DVPP 显存 OOM，每块解码后立即移到 CPU

    Args:
        video_path: 视频文件路径
        chunk_size: 每次解码的帧数上限，默认 2048

    Returns:
        解码后的所有帧张量 [total_frames, C, H, W]，存储在 CPU 上
    """
    if not DVPP_AVAILABLE:
        raise RuntimeError("DVPP 后端不可用，请先 source CANN 环境变量并确保 torch_npu 已安装")
    if not LEROBOT_AVAILABLE:
        raise RuntimeError("lerobot 库不可用，无法调用 DVPP 解码函数")

    if not Path(video_path).exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    metadata = get_video_metadata(video_path)
    total_frames = metadata['total_frames']
    fps = metadata['fps']

    if total_frames == 0:
        raise RuntimeError(f"无法获取视频元数据: {video_path}")

    tolerance = max(1 / fps + 1e-4, 1e-3)
    num_chunks = (total_frames + chunk_size - 1) // chunk_size
    logger.info(f"全量解码: {total_frames} 帧, chunk_size={chunk_size}, 共 {num_chunks} 块")

    chunks = []
    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, total_frames)
        chunk_ts = [i / fps for i in range(start, end)]

        logger.debug(f"解码第 {chunk_idx + 1}/{num_chunks} 块: 帧 {start}-{end - 1}")
        chunk_frames = decode_video_frames_npu(video_path, chunk_ts, tolerance)
        chunks.append(chunk_frames.cpu())

    frames = torch.cat(chunks, dim=0)
    logger.info(f"全量解码完成: {total_frames} 帧, shape={frames.shape}")
    return frames


def test_full_decode_performance(
    video_path: str,
    num_iterations: int = 10,
) -> Optional[dict]:
    """
    测试 DVPP 全量视频解码性能

    Args:
        video_path: 视频文件路径
        num_iterations: 迭代次数

    Returns:
        性能统计字典，DVPP 不可用时返回 None
    """
    if not DVPP_AVAILABLE:
        logger.warning("DVPP 后端不可用，跳过全量解码测试")
        return None
    if not LEROBOT_AVAILABLE:
        logger.warning("lerobot 库不可用，跳过全量解码测试")
        return None

    metadata = get_video_metadata(video_path)
    logger.info(
        f"DVPP 全量解码测试: iterations={num_iterations}, "
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

        logger.info(f"第 {i + 1}/{num_iterations} 次: {iter_time:.2f}s, fps={frames.shape[0] / iter_time:.1f}")

    total_time = time.time() - start_time

    avg_time = sum(decode_times) / len(decode_times)
    min_time = min(decode_times)
    max_time = max(decode_times)

    result = {
        'backend': 'dvpp',
        'mode': 'full_decode',
        'total_time': total_time,
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time,
        'num_iterations': num_iterations,
        'total_frames': frames.shape[0],
        'frames_shape': tuple(frames.shape),
        'video_duration_s': metadata['duration_s'],
        'video_fps': metadata['fps'],
        'decode_fps': frames.shape[0] / avg_time,
    }

    logger.info(
        f"DVPP 全量解码结果: avg={avg_time:.2f}s, "
        f"min={min_time:.2f}s, max={max_time:.2f}s, "
        f"decode_fps={result['decode_fps']:.1f}"
    )

    return result


def test_performance(
    video_path: str,
    timestamps: List[float],
    tolerance_s: float = 1e-4,
    num_iterations: int = 100,
) -> Optional[dict]:
    """
    测试 DVPP 解码器性能

    Args:
        video_path: 视频文件路径
        timestamps: 时间戳列表
        tolerance_s: 时间容差
        num_iterations: 迭代次数

    Returns:
        性能统计字典，DVPP 不可用时返回 None
    """
    if not DVPP_AVAILABLE:
        logger.warning("DVPP 后端不可用，跳过性能测试")
        return None
    if not LEROBOT_AVAILABLE:
        logger.warning("lerobot 库不可用，跳过性能测试")
        return None

    logger.info(f"DVPP 性能测试: iterations={num_iterations}")

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
        'backend': 'dvpp',
        'total_time': total_time,
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time,
        'num_iterations': num_iterations,
        'frames_shape': frames.shape,
    }

    logger.info(f"DVPP 结果: avg={avg_time * 1000:.2f}ms, min={min_time * 1000:.2f}ms, max={max_time * 1000:.2f}ms")

    return result


def main():
    parser = argparse.ArgumentParser(description="DVPP 视频解码器")
    parser.add_argument("--video_path", type=str, required=True, help="视频文件路径")
    parser.add_argument(
        "--timestamps", type=float, nargs='+', default=[0.0, 0.5, 1.0, 1.5, 2.0], help="要解码的时间戳列表（秒）"
    )
    parser.add_argument("--tolerance_s", type=float, default=1e-4, help="时间容差（秒）")
    parser.add_argument("--num_iterations", type=int, default=100, help="性能测试迭代次数")
    parser.add_argument("--test", action="store_true", help="运行性能测试模式")
    parser.add_argument("--full_decode", action="store_true", help="全量视频解码模式")

    args = parser.parse_args()

    if not DVPP_AVAILABLE:
        logger.error("DVPP 后端不可用，请先 source CANN 环境变量并确保 torch_npu 已安装")
        sys.exit(1)

    if not LEROBOT_AVAILABLE:
        logger.error("lerobot 库不可用，无法继续")
        sys.exit(1)

    if not Path(args.video_path).exists():
        logger.error(f"视频文件不存在: {args.video_path}")
        sys.exit(1)

    if args.full_decode:
        result = test_full_decode_performance(
            args.video_path,
            args.num_iterations,
        )
        if result is None:
            logger.error("全量解码测试失败")
            sys.exit(1)

        print(f"\n{'=' * 60}")
        print("DVPP 全量解码测试结果")
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
        if result is None:
            logger.error("性能测试失败")
            sys.exit(1)

        print(f"\n{'=' * 60}")
        print("DVPP 性能测试结果")
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
