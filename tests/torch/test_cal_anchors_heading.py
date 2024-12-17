import numpy as np
import torch
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests

import mx_driving


class TestCalAnchorsHeading(TestCase):
    @golden_data_cache(__file__)
    def cal_anchors_heading_cpu(self, anchors, origin_pos=None):
        if origin_pos is None:
            input_add_start = torch.cat((torch.zeros_like(anchors[:, :, 0:1, :]), anchors), dim=-2)
        elif len(origin_pos.shape) == 2:
            input_add_start = torch.cat((origin_pos.unsqueeze(1).unsqueeze(1).repeat(1, anchors.shape[1], 1, 1), anchors), dim=-2)
        
        xy_diff = input_add_start[:, :, 1:, :] - input_add_start[:, :, :-1, :]
        heading_valid = torch.logical_or(xy_diff[..., 0] > 0.1, xy_diff[..., 1] > 0.1)
        heading = torch.atan2(xy_diff[..., 1], xy_diff[..., 0])

        for t in range(heading.shape[2]):
            heading_t = heading[:, :, t]
            heading_valid_t = heading_valid[:, :, t]
            if t == 0:
                heading_t[heading_valid_t == False] = 0
            else:
                heading_t[heading_valid_t == False] = heading[:, :, t - 1][heading_valid_t == False]
                
        return heading.numpy()

    def cal_anchors_heading_npu(self, anchors, origin_pos=None):
        anchors = anchors.npu()
        origin_pos = None if origin_pos is None else origin_pos.npu()
        heading = mx_driving.cal_anchors_heading(anchors, origin_pos)
        return heading.cpu().numpy()

    @golden_data_cache(__file__)
    def gen_data(self, batch_size, anchors_num, seq_length):
        anchors = np.random.uniform(-5, 5, (batch_size, anchors_num, seq_length, 2))
        origin_pos = np.random.uniform(-5, 5, (batch_size, 2))
        return torch.from_numpy(anchors).float(), torch.from_numpy(origin_pos).float()
    
    def one_case(self, batch_size, anchors_num, seq_length, none_origin_pos=False):
        anchors, origin_pos = self.gen_data(batch_size, anchors_num, seq_length)
        origin_pos = origin_pos if none_origin_pos is False else None
        heading_cpu = self.cal_anchors_heading_cpu(anchors, origin_pos)
        heading_npu = self.cal_anchors_heading_npu(anchors, origin_pos)
        self.assertRtolEqual(heading_cpu, heading_npu)
    
    def test_cal_anchors_heading(self):
        # test min border
        self.one_case(1, 1, 1, True)
        self.one_case(1, 1, 1, False)
        
        # test batch_size max border
        self.one_case(2048, 256, 32, True)
        self.one_case(2048, 256, 32, False)
        
        # test anchor max border
        self.one_case(2, 10240, 32, True)
        self.one_case(2, 10240, 32, False)
        
        # test seq_legnth max border
        self.one_case(2, 256, 256, True)
        self.one_case(2, 256, 256, False)

if __name__ == "__main__":
    run_tests()
