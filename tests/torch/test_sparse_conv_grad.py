from typing import Tuple, Optional
import torch
import torch_npu
from data_cache import golden_data_cache

import mx_driving


# 'pylint: disable=too-many-arguments,huawei-too-many-arguments
@golden_data_cache(__file__)
def generate_input_data(
    seed: int = 0,
    device: str = "npu",
    min_val: int = 1485,
    max_val: int = 2481,
    num_samples: int = 425,
    features_shape: Tuple[int, int] = (827, 128),
    weight_shape: Tuple[int, int, int, int, int] = (3, 1, 1, 128, 128),
    grad_out_shape: Tuple[int, int] = (424, 128)
) -> Tuple[torch.Tensor, ...]:
    """
    生成可重复的随机输入数据
    
    参数:
    - seed: 随机种子 (默认0)
    - device: 目标设备 (默认'npu')
    - min_val/max_val: 索引范围 (包含端点)
    - num_samples: 采样数量
    - features_shape: 特征矩阵形状
    - weight_shape: 权重张量形状
    - grad_out_shape: 梯度输出形状
    
    返回:
    Tuple containing:
    - unique_indices_offset
    - sorted_idx_to_former_indices
    - features
    - weight
    - grad_out_features
    """
    # 参数校验
    assert max_val > min_val, "max_val 必须大于 min_val"
    total_numbers = max_val - min_val + 1
    assert num_samples <= total_numbers, f"采样数量不能超过 {total_numbers}"
    
    # 固定随机种子
    torch.manual_seed(seed)
    
    # 生成设备映射函数
    def to_device(tensor):
        return tensor.to(device) if device else tensor
    
    # 生成唯一随机索引
    random_indices = torch.sort(
        torch.randperm(total_numbers, generator=torch.Generator().manual_seed(seed))[:num_samples]
    )[0]
    
    # 映射到目标范围
    unique_indices_offset = to_device(
        (random_indices + min_val).to(torch.int32).reshape(-1, 1)
    )
    
    # 生成排序后的随机索引
    sorted_idx_to_former_indices = to_device(
        torch.sort(torch.randint(
            low=1,
            high=max_val,
            size=(max_val,),
            generator=torch.Generator().manual_seed(seed),
            dtype=torch.int32
        ))[0]
    )
    
    # 生成特征矩阵
    features = to_device(
        torch.rand(*features_shape, generator=torch.Generator().manual_seed(seed)) * 10 - 5
    )
    
    # 生成权重张量
    weight = to_device(
        torch.rand(*weight_shape, generator=torch.Generator().manual_seed(seed)) * 2 - 1
    )
    
    # 生成梯度输出
    grad_out_features = to_device(
        torch.rand(*grad_out_shape, generator=torch.Generator().manual_seed(seed)) * 2 - 1
    )
    
    return (
        unique_indices_offset,
        sorted_idx_to_former_indices,
        features,
        weight,
        grad_out_features
    )


if __name__ == "__main__":
    # 定义测试用例列表
    test_cases = [
        # 第一个测试用例（原自定义参数）
        {
            "seed": 42,
            "device": "npu",
            "min_val": 1485,
            "max_val": 2481,
            "num_samples": 425,
            "features_shape": (827, 128),
            "grad_out_shape": (424, 128),
            "weight_shape": (3, 1, 1, 128, 128)
        },
        # 第二个测试用例（大数值参数）
        {
            "seed": 42,
            "device": "npu",
            "min_val": 4508310,
            "max_val": 4516128,
            "num_samples": 1831,
            "features_shape": (167264, 16),
            "grad_out_shape": (1830, 32),
            "weight_shape": (3, 3, 3, 16, 32)
        }
    ]

    # 遍历执行所有测试用例
    for _, case_params in enumerate(test_cases, 1):
        
        # 生成输入数据
        inputs = generate_input_data(**case_params)
        unique_indices_offset, sorted_idx_to_former_indices, features, weight, grad_out_features = inputs
        
        # 调用目标函数
        feature_grad, weight_grad = mx_driving._C.npu_sparse_conv3d_grad(
            unique_indices_offset, sorted_idx_to_former_indices, features, weight, grad_out_features
        )