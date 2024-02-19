import torch
import numpy as np
import torch_scatter

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor
import ads.common


class TestScatterMaxWithArgmax(TestCase):

    def cpu_op_exec(self, updates, indices):
        updates.requires_grad = True

        output, output_argmax = torch_scatter.scatter_max(updates, indices.long(), dim=0)
        output.backward(torch.ones_like(output))

        output_grad = updates.grad
        output_grad = output_grad.detach().numpy()
        output = output.detach().numpy()
        output_argmax = output_argmax.to(torch.int32).numpy()

        return output, output_argmax, output_grad

    def npu_op_exec(self, updates, indices):
        updates.requires_grad = True

        output, output_argmax = ads.common.scatter_max(updates, indices)
        output.backward(torch.ones_like(output))

        output_grad = updates.grad.cpu()
        output_grad = output_grad.detach().numpy()
        output = output.cpu()
        output = output.detach().numpy()
        output_argmax = output_argmax.cpu().numpy()

        return output, output_argmax, output_grad

    def test_scatter_max_with_argmax_1(self):
        shape_updates = (262144, 16)
        shape_indices = (262144, 1)
        cpu_updates_input, npu_updates_input = create_common_tensor(["float32", 2, shape_updates], 0, 262144)
        cpu_indices_input, npu_indices_input = create_common_tensor(["int32", 2, shape_indices], 0, 262144)
        cpu_output = self.cpu_op_exec(cpu_updates_input, cpu_indices_input)
        npu_output = self.npu_op_exec(npu_updates_input, npu_indices_input)
        self.assertRtolEqual(cpu_output[0], npu_output[0])
        self.assertRtolEqual(cpu_output[1], npu_output[1])
        self.assertRtolEqual(cpu_output[2], npu_output[2])

    def test_scatter_max_with_argmax_2(self):
        shape_updates = (78848, 16)
        shape_indices = (78848, 1)
        cpu_updates_input, npu_updates_input = create_common_tensor(["float32", 2, shape_updates], 0, 78848)
        cpu_indices_input, npu_indices_input = create_common_tensor(["int32", 2, shape_indices], 0, 78848)
        cpu_output = self.cpu_op_exec(cpu_updates_input, cpu_indices_input)
        npu_output = self.npu_op_exec(npu_updates_input, npu_indices_input)
        self.assertRtolEqual(cpu_output[0], npu_output[0])
        self.assertRtolEqual(cpu_output[1], npu_output[1])
        self.assertRtolEqual(cpu_output[2], npu_output[2])

    def test_scatter_max_with_argmax_3(self):
        shape_updates = (1024, 16)
        shape_indices = (1024, 1)
        cpu_updates_input, npu_updates_input = create_common_tensor(["float32", 2, shape_updates], 0, 100)
        cpu_indices_input, npu_indices_input = create_common_tensor(["int32", 2, shape_indices], 0, 100)
        cpu_output = self.cpu_op_exec(cpu_updates_input, cpu_indices_input)
        npu_output = self.npu_op_exec(npu_updates_input, npu_indices_input)
        self.assertRtolEqual(cpu_output[0], npu_output[0])
        self.assertRtolEqual(cpu_output[1], npu_output[1])
        self.assertRtolEqual(cpu_output[2], npu_output[2])


if __name__ == "__main__":
    run_tests()
