import torch
import mx_driving._C


def npu_fake_tensor_quant(
    inputs: torch.Tensor, amax: torch.Tensor, num_bits: int = 8, is_unsigned: bool = False, narrow_range: bool = True
) -> torch.Tensor:
    return mx_driving._C.npu_fake_tensor_quant(inputs, amax, num_bits, is_unsigned, narrow_range)


def npu_fake_tensor_quant_inplace(
    inputs: torch.Tensor, amax: torch.Tensor, num_bits: int = 8, is_unsigned: bool = False, narrow_range: bool = True
) -> torch.Tensor:
    return mx_driving._C.npu_fake_tensor_quant_inplace(inputs, amax, num_bits, is_unsigned, narrow_range)


def npu_fake_tensor_quant_with_axis(
    inputs: torch.Tensor,
    amax: torch.Tensor,
    axis: int,
    num_bits: int = 8,
    is_unsigned: bool = False,
    narrow_range: bool = True,
) -> torch.Tensor:
    return mx_driving._C.npu_fake_tensor_quant_with_axis(inputs, amax, axis, num_bits, is_unsigned, narrow_range)
