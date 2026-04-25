import numpy as np
import torch
import torch_npu
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests
from mx_driving import npu_fake_tensor_quant
from mx_driving import npu_fake_tensor_quant_inplace
from mx_driving import npu_fake_tensor_quant_with_axis


def bits_to_bound(num_bits, is_unsigned):
    bound = (1 << (num_bits - 1 + int(is_unsigned))) - 1
    return bound


@golden_data_cache(__file__)
def cpu_gen_inputs(inputs_shape):
    cpu_inputs = torch.randn(inputs_shape, dtype=torch.float)
    cpu_amax = cpu_inputs.max()
    return cpu_inputs, cpu_amax


@golden_data_cache(__file__)
def cpu_gen_inputs_with_axis(inputs_shape, axis):
    cpu_inputs = torch.randn(inputs_shape, dtype=torch.float)
    dims_to_reduce = [i for i in range(len(inputs_shape)) if i != axis]
    cpu_amax = cpu_inputs.abs().amax(dim=dims_to_reduce)
    return cpu_inputs, cpu_amax


class TestFakeTensorQuant(TestCase):
    def fake_tensor_quant(self, inputs, amax, num_bits=8, is_unsigned=False, narrow_range=True):

        if not inputs.is_contiguous():
            inputs = inputs.contiguous()
        if not amax.is_contiguous():
            amax = amax.contiguous()

        bound = bits_to_bound(num_bits, is_unsigned)
        max_bound = bound
        min_bound = -(bound + int(not narrow_range))

        scale = max_bound / amax
        outputs = torch.round(inputs * scale)
        outputs = torch.clamp(outputs, min_bound, max_bound)
        outputs = outputs / scale

        return outputs

    def fake_tensor_quant_with_axis(self, inputs, amax, axis, num_bits=8, is_unsigned=False, narrow_range=True):
        if not inputs.is_contiguous():
            inputs = inputs.contiguous()
        if not amax.is_contiguous():
            amax = amax.contiguous()

        bound = bits_to_bound(num_bits, is_unsigned)
        max_bound = bound
        min_bound = -(bound + int(not narrow_range))
        expanded_max_bound = torch.full_like(amax, max_bound)
        scale = expanded_max_bound / amax
        scale = scale.view([-1 if i == axis else 1 for i in range(inputs.dim())])

        outputs = torch.round(inputs * scale)
        outputs = torch.clamp(outputs, min_bound, max_bound)
        outputs = outputs / scale
        return outputs

    def test_npu_fake_tensor_quant_case(self):
        input_list = [[[10000], 8, False, True]]

        for input_info in input_list:
            inputs_shape = input_info[0]
            bits_num = input_info[1]
            is_unsigned = input_info[2]
            narrow_range = input_info[3]

            cpu_inputs, cpu_amax = cpu_gen_inputs(inputs_shape)
            npu_inputs = cpu_inputs.clone().to("npu")
            npu_amax = cpu_amax.clone().to("npu")

            cpu_output = self.fake_tensor_quant(cpu_inputs, cpu_amax, bits_num, is_unsigned, narrow_range)
            npu_output = npu_fake_tensor_quant(npu_inputs, npu_amax, bits_num, is_unsigned, narrow_range)

            self.assertRtolEqual(cpu_output, npu_output.cpu())

    def test_npu_fake_tensor_quant_inplace_case(self):
        input_list = [[[10000], 8, False, True]]

        for input_info in input_list:
            inputs_shape = input_info[0]
            bits_num = input_info[1]
            is_unsigned = input_info[2]
            narrow_range = input_info[3]

            cpu_inputs, cpu_amax = cpu_gen_inputs(inputs_shape)
            npu_inputs = cpu_inputs.clone().to("npu")
            npu_amax = cpu_amax.clone().to("npu")

            cpu_output = self.fake_tensor_quant(cpu_inputs, cpu_amax, bits_num, is_unsigned, narrow_range)
            npu_output = npu_fake_tensor_quant_inplace(npu_inputs, npu_amax, bits_num, is_unsigned, narrow_range)

            self.assertRtolEqual(cpu_output, npu_output.cpu())

    def test_npu_fake_tensor_quant_with_axis_case(self):
        input_list = [[[40, 3, 10, 500], 1, 8, False, True]]

        for input_info in input_list:
            inputs_shape = input_info[0]
            axis = input_info[1]
            bits_num = input_info[2]
            is_unsigned = input_info[3]
            narrow_range = input_info[4]

            cpu_inputs, cpu_amax = cpu_gen_inputs_with_axis(inputs_shape, axis)
            npu_inputs = cpu_inputs.clone().to("npu")
            npu_amax = cpu_amax.clone().to("npu")

            cpu_output = self.fake_tensor_quant_with_axis(
                cpu_inputs, cpu_amax, axis, bits_num, is_unsigned, narrow_range
            )
            npu_output = npu_fake_tensor_quant_with_axis(
                npu_inputs, npu_amax, axis, bits_num, is_unsigned, narrow_range
            )
            self.assertRtolEqual(cpu_output, npu_output.cpu())

    def test_npu_fake_tensor_quant_float16_case(self):
        input_list = [[[10000], 8, False, True]]

        for input_info in input_list:
            inputs_shape = input_info[0]
            bits_num = input_info[1]
            is_unsigned = input_info[2]
            narrow_range = input_info[3]

            cpu_inputs, cpu_amax = cpu_gen_inputs(inputs_shape)
            cpu_inputs = cpu_inputs.to(torch.float16)
            cpu_amax = cpu_amax.to(torch.float16)

            npu_inputs = cpu_inputs.clone().to("npu")
            npu_amax = cpu_amax.clone().to("npu")

            cpu_output = self.fake_tensor_quant(cpu_inputs, cpu_amax, bits_num, is_unsigned, narrow_range)
            npu_output = npu_fake_tensor_quant(npu_inputs, npu_amax, bits_num, is_unsigned, narrow_range)

            self.assertRtolEqual(cpu_output, npu_output.cpu())

    def test_npu_fake_tensor_quant_with_axis_last_dim_case(self):
        input_list = [[[40, 3, 10, 500], 3, 8, False, True]]

        for input_info in input_list:
            inputs_shape = input_info[0]
            axis = input_info[1]
            bits_num = input_info[2]
            is_unsigned = input_info[3]
            narrow_range = input_info[4]

            cpu_inputs, cpu_amax = cpu_gen_inputs_with_axis(inputs_shape, axis)
            npu_inputs = cpu_inputs.clone().to("npu")
            npu_amax = cpu_amax.clone().to("npu")

            cpu_output = self.fake_tensor_quant_with_axis(
                cpu_inputs, cpu_amax, axis, bits_num, is_unsigned, narrow_range
            )
            npu_output = npu_fake_tensor_quant_with_axis(
                npu_inputs, npu_amax, axis, bits_num, is_unsigned, narrow_range
            )
            self.assertRtolEqual(cpu_output, npu_output.cpu())

    def test_npu_fake_tensor_quant_case_2(self):
        input_list = [[[100663296], 8, False, True]]

        for input_info in input_list:
            inputs_shape = input_info[0]
            bits_num = input_info[1]
            is_unsigned = input_info[2]
            narrow_range = input_info[3]

            cpu_inputs, cpu_amax = cpu_gen_inputs(inputs_shape)
            npu_inputs = cpu_inputs.clone().to("npu")
            npu_amax = cpu_amax.clone().to("npu")

            cpu_output = self.fake_tensor_quant(cpu_inputs, cpu_amax, bits_num, is_unsigned, narrow_range)
            npu_output = npu_fake_tensor_quant(npu_inputs, npu_amax, bits_num, is_unsigned, narrow_range)

            self.assertRtolEqual(cpu_output, npu_output.cpu())

    def test_npu_fake_tensor_quant_with_axis_case_2(self):
        input_list = [[[32, 8, 512, 768], 1, 8, False, True]]

        for input_info in input_list:
            inputs_shape = input_info[0]
            axis = input_info[1]
            bits_num = input_info[2]
            is_unsigned = input_info[3]
            narrow_range = input_info[4]

            cpu_inputs, cpu_amax = cpu_gen_inputs_with_axis(inputs_shape, axis)
            npu_inputs = cpu_inputs.clone().to("npu")
            npu_amax = cpu_amax.clone().to("npu")

            cpu_output = self.fake_tensor_quant_with_axis(
                cpu_inputs, cpu_amax, axis, bits_num, is_unsigned, narrow_range
            )
            npu_output = npu_fake_tensor_quant_with_axis(
                npu_inputs, npu_amax, axis, bits_num, is_unsigned, narrow_range
            )
            self.assertRtolEqual(cpu_output, npu_output.cpu())


if __name__ == "__main__":
    run_tests()
