import numpy as np
import torch
import torch_npu
import torch_scatter
from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.testing.testcase import TestCase, run_tests

import mx_driving.common


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

        output, output_argmax = mx_driving.common.scatter_max(updates, indices)
        output, output_argmax = mx_driving.scatter_max(updates, indices)
        output.backward(torch.ones_like(output))

        output_grad = updates.grad.cpu()
        output_grad = output_grad.detach().numpy()
        output = output.cpu()
        output = output.detach().numpy()
        output_argmax = output_argmax.cpu().numpy()

        return output, output_argmax, output_grad

    def test_scatter_max_dim3_1(self):
        shape_updates = (100, 3, 16)
        shape_indices = (100, 1, 1)
        cpu_updates, npu_updates = create_common_tensor(["float32", 2, shape_updates], 0, 100)
        cpu_indices, npu_indices = create_common_tensor(["int32", 2, shape_indices], 0, 100)
        output_npu, output_argmax_npu = mx_driving.common.scatter_max(npu_updates, npu_indices)
        output_cpu, output_argmax_cpu = torch_scatter.scatter_max(cpu_updates, cpu_indices.to(torch.int64), dim=0)
        self.assertRtolEqual(output_cpu, output_npu)
        self.assertRtolEqual(output_argmax_cpu.to(torch.int32), output_argmax_npu)

    def test_scatter_max_dim5_2(self):
        shape_updates = (100, 4, 3, 16, 3)
        shape_indices = (100, 1, 1)
        cpu_updates, npu_updates = create_common_tensor(["float32", 2, shape_updates], 0, 100)
        cpu_indices, npu_indices = create_common_tensor(["int32", 2, shape_indices], 0, 100)
        output_npu, output_argmax_npu = mx_driving.common.scatter_max(npu_updates, npu_indices)
        output_cpu, output_argmax_cpu = torch_scatter.scatter_max(cpu_updates, cpu_indices.to(torch.int64), dim=0)
        self.assertRtolEqual(output_cpu, output_npu)
        self.assertRtolEqual(output_argmax_cpu.to(torch.int32), output_argmax_npu)

    def test_scatter_max_bigtail_3(self):
        shape_updates = (100, 8192)
        shape_indices = (100, 1)
        cpu_updates, npu_updates = create_common_tensor(["float32", 2, shape_updates], 0, 100)
        cpu_indices, npu_indices = create_common_tensor(["int32", 2, shape_indices], 0, 100)
        output_npu, output_argmax_npu = mx_driving.common.scatter_max(npu_updates, npu_indices)
        output_cpu, output_argmax_cpu = torch_scatter.scatter_max(cpu_updates, cpu_indices.to(torch.int64), dim=0)
        self.assertRtolEqual(output_cpu, output_npu)
        self.assertRtolEqual(output_argmax_cpu.to(torch.int32), output_argmax_npu)

    def test_scatter_max_dim3_and_unaligned_4(self):
        shape_updates = (1024, 123, 5)
        shape_indices = (1024, 1, 1)
        cpu_updates, npu_updates = create_common_tensor(["float32", 2, shape_updates], 0, 100)
        cpu_indices, npu_indices = create_common_tensor(["int32", 2, shape_indices], 0, 100)
        output_npu, output_argmax_npu = mx_driving.common.scatter_max(npu_updates, npu_indices)
        output_cpu, output_argmax_cpu = torch_scatter.scatter_max(cpu_updates, cpu_indices.to(torch.int64), dim=0)
        self.assertRtolEqual(output_cpu, output_npu)
        self.assertRtolEqual(output_argmax_cpu.to(torch.int32), output_argmax_npu)

    def test_scatter_max_with_out_1(self):
        shape_updates = (1024, 16)
        shape_indices = (1024, 1)
        shape_out = (1024, 16)
        cpu_updates, npu_updates = create_common_tensor(["float32", 2, shape_updates], 0, 100)
        cpu_indices, npu_indices = create_common_tensor(["int32", 2, shape_indices], 0, 1000)
        cpu_out, npu_out = create_common_tensor(["float32", 2, shape_out], 0, 100)
        output_npu, output_argmax_npu = mx_driving.common.scatter_max(npu_updates, npu_indices, npu_out)
        output_npu, output_argmax_npu = mx_driving.scatter_max(npu_updates, npu_indices, npu_out)
        output_cpu, output_argmax_cpu = torch_scatter.scatter_max(cpu_updates, cpu_indices.to(torch.int64), dim=0, out=cpu_out)
        self.assertRtolEqual(output_cpu, output_npu)
        self.assertRtolEqual(output_argmax_cpu.to(torch.int32), output_argmax_npu)

    def test_scatter_max_with_out_2(self):
        shape_updates = (100, 3, 16)
        shape_indices = (100, 1, 1)
        shape_out = (20, 3, 16)
        cpu_updates, npu_updates = create_common_tensor(["float32", 2, shape_updates], 0, 100)
        cpu_indices, npu_indices = create_common_tensor(["int32", 2, shape_indices], 0, 20)
        cpu_out, npu_out = create_common_tensor(["float32", 2, shape_out], 0, 100)
        output_npu, output_argmax_npu = mx_driving.common.scatter_max(npu_updates, npu_indices, npu_out)
        output_npu, output_argmax_npu = mx_driving.scatter_max(npu_updates, npu_indices, npu_out)
        output_cpu, output_argmax_cpu = torch_scatter.scatter_max(cpu_updates, cpu_indices.to(torch.int64), dim=0, out=cpu_out)
        self.assertRtolEqual(output_cpu, output_npu)
        self.assertRtolEqual(output_argmax_cpu.to(torch.int32), output_argmax_npu)
    
    def test_scatter_max_with_grad_1(self):
        shape_updates = (262144, 16)
        shape_indices = (262144, 1)
        cpu_updates_input, npu_updates_input = create_common_tensor(["float32", 2, shape_updates], 0, 262144)
        cpu_indices_input, npu_indices_input = create_common_tensor(["int32", 2, shape_indices], 0, 262144)
        cpu_output = self.cpu_op_exec(cpu_updates_input, cpu_indices_input)
        npu_output = self.npu_op_exec(npu_updates_input, npu_indices_input)
        self.assertRtolEqual(cpu_output[0], npu_output[0])
        self.assertRtolEqual(cpu_output[1], npu_output[1])
        self.assertRtolEqual(cpu_output[2], npu_output[2])

    def test_scatter_max_with_grad_2(self):
        shape_updates = (78848, 16)
        shape_indices = (78848, 1)
        cpu_updates_input, npu_updates_input = create_common_tensor(["float32", 2, shape_updates], 0, 78848)
        cpu_indices_input, npu_indices_input = create_common_tensor(["int32", 2, shape_indices], 0, 78848)
        cpu_output = self.cpu_op_exec(cpu_updates_input, cpu_indices_input)
        npu_output = self.npu_op_exec(npu_updates_input, npu_indices_input)
        self.assertRtolEqual(cpu_output[0], npu_output[0])
        self.assertRtolEqual(cpu_output[1], npu_output[1])
        self.assertRtolEqual(cpu_output[2], npu_output[2])

    def test_scatter_max_with_grad_3(self):
        shape_updates = (1024, 16)
        shape_indices = (1024, 1)
        cpu_updates_input, npu_updates_input = create_common_tensor(["float32", 2, shape_updates], 0, 100)
        cpu_indices_input, npu_indices_input = create_common_tensor(["int32", 2, shape_indices], 0, 100)
        cpu_output = self.cpu_op_exec(cpu_updates_input, cpu_indices_input)
        npu_output = self.npu_op_exec(npu_updates_input, npu_indices_input)
        self.assertRtolEqual(cpu_output[0], npu_output[0])
        self.assertRtolEqual(cpu_output[1], npu_output[1])
        self.assertRtolEqual(cpu_output[2], npu_output[2])

    def test_scatter_max_with_grad_4(self):
        shape_updates = (1024, 128)
        shape_indices = (1024, 1)
        cpu_updates_input, npu_updates_input = create_common_tensor(["float32", 2, shape_updates], 0, 100)
        cpu_indices_input, npu_indices_input = create_common_tensor(["int32", 2, shape_indices], 0, 100)
        cpu_output = self.cpu_op_exec(cpu_updates_input, cpu_indices_input)
        npu_output = self.npu_op_exec(npu_updates_input, npu_indices_input)
        self.assertRtolEqual(cpu_output[0], npu_output[0])
        self.assertRtolEqual(cpu_output[1], npu_output[1])
        self.assertRtolEqual(cpu_output[2], npu_output[2])

    def test_scatter_max_with_grad_bigtail(self):
        shape_updates = (1024, 4096)
        shape_indices = (1024, 1)
        cpu_updates_input, npu_updates_input = create_common_tensor(["float32", 2, shape_updates], 0, 100)
        cpu_indices_input, npu_indices_input = create_common_tensor(["int32", 2, shape_indices], 0, 100)
        cpu_output = self.cpu_op_exec(cpu_updates_input, cpu_indices_input)
        npu_output = self.npu_op_exec(npu_updates_input, npu_indices_input)
        self.assertRtolEqual(cpu_output[0], npu_output[0])
        self.assertRtolEqual(cpu_output[1], npu_output[1])
        self.assertRtolEqual(cpu_output[2], npu_output[2])

    def test_scatter_max_with_grad_unaligned(self):
        shape_updates = (1024, 17)
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
