import os
import time
import unittest
import torch
import torch_npu
import numpy as np
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests

import mx_driving

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:9]

def sigmoid_focal_loss(logit, target, gamma=2.0, alpha=0.25, weight=None, reduction='mean'):
    logit_size = logit.shape
    output = torch.zeros_like(logit)
    for i in range(logit_size[0]):
        target_i = target[i].item()
        for j in range(logit_size[1]):
            sigmoid_x = torch.sigmoid(logit[i, j])
            entropy_p = torch.pow(1 - sigmoid_x, gamma) * torch.log(sigmoid_x)
            entropy_n = torch.pow(sigmoid_x, gamma) * torch.log(1 - sigmoid_x)
            if j == target_i:
                output[i, j] += -alpha * entropy_p
            else:
                output[i, j] += (alpha - 1) * entropy_n
            if weight is not None:
                output[i, j] *= weight[target_i]

    if reduction == 'mean':
        output = output.sum() / logit.shape[0]
    elif reduction == 'sum':
        output = output.sum()
    return output

def sigmoid_focal_loss_grad(logit, target, grad_output, gamma=2.0, alpha=0.25, weight=None, reduction='mean'):
    logit_size = logit.shape
    grad_input = torch.zeros_like(logit)
    for i in range(logit_size[0]):
        target_i = target[i].item()
        for j in range(logit_size[1]):
            sigmoid_x = torch.sigmoid(logit[i, j])
            entropy_p = alpha * torch.pow(1 - sigmoid_x, gamma) * (gamma * sigmoid_x * torch.log(sigmoid_x) - (1. - sigmoid_x))
            entropy_n = (1 - alpha) * torch.pow(sigmoid_x, gamma) * (sigmoid_x - gamma * (1 - sigmoid_x) * torch.log(1 - sigmoid_x))
            if j == target_i:
                grad_input[i, j] += entropy_p
            else:
                grad_input[i, j] += entropy_n
            if weight is not None:
                grad_input[i, j] *= weight[target_i]
    grad_input *= grad_output
    if reduction == 'mean':
        grad_input /= logit_size[0]
    return grad_input
 

@golden_data_cache(__file__)
def gen_data(N, NC):
    logit = torch.rand(N, NC, dtype=torch.float32) * 10 - 5
    logit.requires_grad = True
    target = torch.randint(low=0, high=NC, size=(N,), dtype=torch.int64)
    weight = torch.rand(NC, dtype=torch.float32) * 10 - 5
    return logit, target, weight

class TestSigmoidFocalLoss(TestCase):
    @unittest.skipIf(DEVICE_NAME not in ['Ascend950'], "OP `SigmoidFocalLoss` is not supported, skip this ut!")
    def test_sigmoid_focal_loss(self):
        N_list = [2700, 5400, 6300, 4761]
        NC_list = [10, 79]
        for N in N_list:
            for NC in NC_list:
                logit, target, weight = gen_data(N, NC)
                logit_npu, target_npu, weight_npu = logit.npu(), target.npu(), weight.npu()
                output_golden = sigmoid_focal_loss(logit_npu, target_npu, 2.0, 0.25, weight_npu, 'mean')
                grad_golden = sigmoid_focal_loss_grad(logit_npu, target_npu, torch.ones_like(output_golden), 2.0, 0.25, weight_npu, 'mean')
                torch.npu.synchronize()

                output_mxdriving = mx_driving.sigmoid_focal_loss(logit_npu, target_npu, 2.0, 0.25, weight_npu, 'mean')
                output_mxdriving.backward()
                grad_mxdriving = logit.grad
                torch.npu.synchronize()
                self.assertRtolEqual(output_golden.cpu(), output_mxdriving.cpu())
                self.assertRtolEqual(grad_golden.cpu(), grad_mxdriving.cpu())


if __name__ == "__main__":
    run_tests()
