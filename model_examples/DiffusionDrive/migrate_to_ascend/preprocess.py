import os

# 设置线程相关环境变量
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["GOTO_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

# pylint: disable=huawei-wrong-import-position
import sys
import subprocess
import types
import runpy
from pathlib import Path


import pickle

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans

import mmcv
import mx_driving


def execute_script(script_path, args, cwd):
    """
    在当前进程执行指定脚本（使用runpy.run_path）
    :param script_path: 脚本相对路径（如"tools/data_converter/nuscenes_converter.py"）
    :param args: 命令行参数列表
    :param cwd: 工作目录
    """
    # 保存原始状态
    original_cwd = os.getcwd()
    original_argv = sys.argv.copy()
    
    try:
        # 切换工作目录并设置sys.argv
        os.chdir(cwd)
        sys.argv = [script_path] + args
        
        # 直接使用run_path执行脚本
        runpy.run_path(script_path, run_name="__main__")
        
    except Exception as e:
        print(f"执行失败: {script_path} with args {args}")
        print(f"错误详情: {str(e)}")
        raise
    finally:
        # 恢复原始状态
        os.chdir(original_cwd)
        sys.argv = original_argv


# pylint: disable=huawei-redefined-outer-name, lambda-assign
def patch_mock_gpu_flash_attn():
    flash_attn = types.ModuleType('flash_attn')

    flash_attn_interface = types.ModuleType('flash_attn.flash_attn_interface')
    flash_attn_interface.flash_attn_unpadded_kvpacked_func = lambda *args, **kwargs: None
    flash_attn_interface.flash_attn_varlen_kvpacked_func = lambda *args, **kwargs: None

    bert_padding = types.ModuleType('flash_attn.bert_padding')
    bert_padding.unpad_input = lambda *args, **kwargs: (None, None)  
    bert_padding.pad_input = lambda *args, **kwargs: None
    bert_padding.index_first_axis = lambda *args, **kwargs: None

    flash_attn.flash_attn_interface = flash_attn_interface
    flash_attn.bert_padding = bert_padding

    sys.modules['flash_attn'] = flash_attn
    sys.modules['flash_attn.flash_attn_interface'] = flash_attn_interface
    sys.modules['flash_attn.bert_padding'] = bert_padding


# Mock deform_aggreg in projects.mmdet3d_plugin.ops.deformable_aggregation, replace by mx_driving's deform_aggreg within the mock class
def patch_deform_aggreg():
    
    
    class MockDeformableAggregationFunction:
        @staticmethod
        def apply(*args, **kwargs):
            return mx_driving.deformable_aggregation(*args, **kwargs)
    
    
    mock_module = types.ModuleType("projects.mmdet3d_plugin.ops.deformable_aggregation")
    sys.modules["projects.mmdet3d_plugin.ops.deformable_aggregation"] = mock_module
    mock_module.DeformableAggregationFunction = MockDeformableAggregationFunction


def kmeans_plan_bugfree():
    K = 6

    fp = 'data/infos/nuscenes_infos_train.pkl'
    
    if not os.path.exists(fp):
        raise FileNotFoundError(f"{fp} does not exist")
    
    data = mmcv.load(fp)
    data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
    navi_trajs = [[], [], []]
    for idx in tqdm(range(len(data_infos))):
        info = data_infos[idx]
        plan_traj = info['gt_ego_fut_trajs'].cumsum(axis=-2)
        plan_mask = info['gt_ego_fut_masks']
        cmd = info['gt_ego_fut_cmd'].astype(np.int32)
        cmd = cmd.argmax(axis=-1)
        if not plan_mask.sum() == 6:
            continue
        navi_trajs[cmd].append(plan_traj)

    clusters = []
    for trajs in navi_trajs:
        trajs = np.concatenate(trajs, axis=0).reshape(-1, 12)
        cluster = KMeans(n_clusters=K).fit(trajs).cluster_centers_
        cluster = cluster.reshape(-1, 6, 2)
        clusters.append(cluster)
        for j in range(K):
            plt.scatter(cluster[j, :, 0], cluster[j, :, 1])
    plt.savefig(f'vis/kmeans/plan_{K}', bbox_inches='tight')
    plt.close()

    clusters = np.stack(clusters, axis=0)
    np.save(f'data/kmeans/kmeans_plan_{K}.npy', clusters)


def main():
    # 设置环境变量
    current_dir = Path(__file__).parent.absolute()
    parent_dir = current_dir.parent
    os.environ["PYTHONPATH"] = f"{parent_dir}:{os.environ.get('PYTHONPATH', '')}"
    
    patch_mock_gpu_flash_attn()
    patch_deform_aggreg()
    commands = [
        # NuScenes转换命令
        [
            "tools/data_converter/nuscenes_converter.py",
            "nuscenes",
            "--root-path", "./data/nuscenes",
            "--canbus", "./data/nuscenes",
            "--out-dir", "./data/infos/",
            "--extra-tag", "nuscenes",
            "--version", "v1.0"
        ],
        # K-means相关命令
        ["tools/kmeans/kmeans_det.py"],
        ["tools/kmeans/kmeans_map.py"],
        ["tools/kmeans/kmeans_motion.py"],
        #["tools/kmeans/kmeans_plan.py"] # 有bug
    ]

    # 执行所有命令
    for cmd in commands:
        script_path = cmd[0]
        
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"{script_path} does not exist")
        
        args = cmd[1:] if len(cmd) > 1 else []
        try:
            execute_script(script_path, args, parent_dir)
        except Exception as e:
            print(f"命令执行终止: {' '.join(cmd)}")
            sys.exit(1)

    # 执行修复版的kmeans plan
    kmeans_plan_bugfree()

if __name__ == "__main__":
    main()
