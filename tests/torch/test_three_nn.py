import numpy as np
import torch
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests

import mx_driving
import mx_driving.common


@golden_data_cache(__file__)
def cpu_gen_inputs(batch, N, npoint):
    source = np.ones((batch, N, 3)).astype(np.float32)
    target = np.zeros((batch, npoint, 3)).astype(np.float32)
    return source, target


class TestThreeNN(TestCase):
    @golden_data_cache(__file__)
    def cpu_op_exec(self,
        batch,
        npoint,
        source,
        target):
        idx = np.zeros((batch, npoint, 3), dtype=np.int32)
        dist2 = np.zeros((batch, npoint, 3), dtype=np.float32)

        for b in range(batch):
            for m in range(npoint):
                new_x = target[b][m][0]
                new_y = target[b][m][1]
                new_z = target[b][m][2]

                x = source[b, :, 0]
                y = source[b, :, 1]
                z = source[b, :, 2]

                dist = (x - new_x) ** 2 + (y - new_y) ** 2 + (z - new_z) ** 2

                sorted_indices_and_values = sorted(enumerate(dist), key=lambda x: (x[1], x[0]))
                for i in range(3):
                    idx[b][m][i], dist2[b][m][i] = sorted_indices_and_values[i]
        return np.sqrt(dist2), idx

    def test_three_nn(self):
        batch = 1
        npoint = 1
        N = 200
        source, target = cpu_gen_inputs(batch, N, npoint)

        expected_dist, expected_idx = self.cpu_op_exec(batch, npoint, source, target)
        dist, idx = mx_driving.three_nn(torch.from_numpy(target).npu(), torch.from_numpy(source).npu())

        self.assertRtolEqual(expected_dist, dist.cpu().numpy())
        self.assertRtolEqual(expected_idx, idx.cpu().numpy())

        dist_verify, idx_verify = mx_driving.common.three_nn(torch.from_numpy(target).npu(), torch.from_numpy(source).npu())
        self.assertRtolEqual(expected_dist, dist_verify.cpu().numpy())
        self.assertRtolEqual(expected_idx, idx_verify.cpu().numpy())

    def test_three_nn_1(self):
        source = np.array(
            [[[-2.0238, -3.3207, 8.1171],
              [-1.7629, 0.8950, 1.8025],
              [5.4141, -0.4934, -15.7047],
              [-2.9846, 13.8238, 3.3200],
              [12.3173, 1.2928, 13.7832],
              [-0.4871, -12.8897, 3.9310],
              [1.3828, 13.5287, 6.6030],
              [2.0981, 13.4126, 0.6788],
              [-1.5090, -9.5088, 15.5907],
              [-0.3600, -2.2742, -4.2612],
              [-10.7938, -7.7781, 8.5862],
              [-7.1659, -13.5676, -4.6127],
              [2.1858, 7.6612, -9.7124],
              [21.7843, -1.3703, 27.4645],
              [6.1499, -16.0455, -12.5857],
              [-19.1001, -2.6565, 11.3803],
              [-19.3571, -9.4445, -12.9314],
              [8.2156, 8.4073, 1.6766],
              [19.4642, -2.8906, -2.3793]],

             [[-7.0372, 4.1289, -9.9067],
              [6.4930, -1.7466, 1.1633],
              [-22.6341, -5.4099, 8.3307],
              [-13.1931, -0.9765, -13.7386],
              [-23.3002, -5.7986, 4.4145],
              [5.5258, -21.0376, 5.4568],
              [6.4506, -17.7697, -11.0068],
              [-10.4950, -10.3471, 5.6770],
              [13.7752, 6.7100, -4.8219],
              [-7.8780, 3.1963, -0.1692],
              [-7.3232, 4.3203, -6.0770],
              [0.2542, 5.1513, -3.5310],
              [7.6403, -13.0677, -0.5636],
              [1.9074, -6.7681, 6.0172],
              [-1.7251, -1.5686, 1.7895],
              [17.7471, -3.9607, -11.1501],
              [3.2450, 9.3232, 7.4576],
              [10.1135, -11.6206, -1.4590],
              [18.1644, -6.7828, -8.3379]],

             [[10.9152, 10.4122, -3.4534],
              [-6.2864, -4.0358, -5.6467],
              [2.5500, -3.9311, 14.4367],
              [-10.0484, 5.8571, -6.5559],
              [7.8157, -4.0994, 3.6841],
              [-18.5262, -1.7756, -1.1456],
              [-9.6069, 5.0123, 0.7830],
              [9.9818, 4.6848, 6.8564],
              [0.1195, 5.5399, 6.4940],
              [-3.1857, -6.6081, 19.7650],
              [1.9646, -6.7072, -19.4115],
              [-1.9893, -0.1298, 4.7661],
              [-2.6206, -4.9635, 15.4014],
              [2.9496, 11.2910, -1.2555],
              [9.5890, -2.0793, 15.3348],
              [1.0805, 7.4695, -8.3777],
              [7.3841, -10.5168, -17.1081],
              [-7.7960, 9.7074, 4.6976],
              [10.6985, -3.1990, 2.6828]],

             [[7.9618, -0.8266, -1.7552],
              [-8.3311, 10.2373, -0.3869],
              [-5.9823, 10.5979, -7.8076],
              [2.2180, 14.0860, 13.6502],
              [2.4314, 11.7121, 6.4348],
              [-0.4151, 7.2973, 6.0398],
              [-4.1299, -8.9361, -7.6534],
              [3.8277, 1.8265, -8.3354],
              [-10.3117, 13.5711, 6.3501],
              [-10.4253, -2.0336, -1.4464],
              [-1.6153, 9.3242, 8.7789],
              [8.2936, -3.3222, 5.3288],
              [-3.2867, 4.5483, 4.8678],
              [-9.8688, 6.7665, -12.3278],
              [18.2852, 13.3130, -2.6990],
              [-1.4564, 7.7977, -3.2762],
              [-5.4958, -9.1329, -7.3159],
              [2.5063, -4.2323, -3.7325],
              [-1.5683, -5.2733, -4.4030]],

             [[-5.9978, -6.6638, 17.9294],
              [-7.0945, -21.9682, 3.6423],
              [7.2417, -26.7891, 14.3742],
              [6.3299, -4.5266, 6.9212],
              [1.3738, 4.2038, 19.1443],
              [9.6290, -8.5555, -4.7785],
              [-0.5148, 7.7239, 1.3804],
              [7.7055, -1.6901, 4.6251],
              [4.3315, 6.5598, -1.3172],
              [1.0159, 12.8314, 10.3839],
              [14.0982, -3.8149, 2.3777],
              [6.1465, -0.8435, 10.8809],
              [-0.7985, -4.8287, 12.7530],
              [0.7275, -0.6962, 3.6355],
              [-3.6257, 3.0410, 7.1237],
              [10.9248, -3.1069, 2.9934],
              [-1.1401, -12.3300, -4.9909],
              [12.5418, 7.1750, -6.7905],
              [8.7212, -17.8335, -6.1030]],

             [[3.0320, -15.9645, -18.2718],
              [7.4597, -4.7232, 0.7846],
              [-2.8595, 3.8792, 11.7725],
              [1.6641, 4.4071, -11.8249],
              [15.8837, -18.1406, 6.6712],
              [14.7876, -14.0845, -12.0774],
              [-6.7706, -4.9462, 5.0924],
              [-12.8305, -0.7784, -6.2183],
              [9.3316, 16.6452, 9.5345],
              [16.3296, 2.6188, -3.6169],
              [10.5791, 10.2543, -5.5496],
              [-6.6624, -23.4441, 21.3813],
              [3.7359, 6.4775, 8.5231],
              [7.6542, 11.0349, -0.4124],
              [0.1497, 1.1762, 7.0101],
              [0.2828, 12.0548, -0.2077],
              [25.4812, -20.3991, -3.1827],
              [-2.0954, 19.6107, -3.1691],
              [0.5224, -1.2179, 0.8885]],

             [[8.8782, 13.3272, -1.2416],
              [3.6355, 8.4196, 24.4064],
              [-12.4760, -3.5521, 9.8329],
              [-6.6276, -4.6736, 2.5339],
              [3.7698, 13.3429, 1.6758],
              [-1.7233, 9.2620, -7.9604],
              [-8.0612, -3.4397, -17.7093],
              [1.8540, 3.9471, 11.5713],
              [23.3233, 13.0063, -10.8297],
              [-1.2089, 4.1231, -0.0570],
              [-8.5042, 5.5494, 0.8967],
              [-13.8665, -2.2168, 4.2592],
              [-5.1688, 8.0387, -6.2596],
              [3.5258, -5.5064, 16.1700],
              [-7.4856, 8.7537, -10.3982],
              [-10.3084, 0.8473, -19.7287],
              [14.7944, -5.6747, -19.0772],
              [0.6918, 9.7165, 0.1957],
              [16.3732, -4.0999, -3.9985]]]).astype(np.float32)
        target = np.array(
            [[[8.7439, -27.7414, -9.7220],
              [0.2520, -8.6042, -11.4539],
              [-14.9610, -14.3345, -11.5521],
              [-13.2282, 6.8182, 10.6166],
              [-16.3290, -17.9947, 10.0489],
              [-9.6678, -5.6852, -27.6616],
              [-24.5317, -17.8157, -7.6455]],

             [[-8.1713, 2.0374, -14.0803],
              [-11.8026, -26.9108, -9.4289],
              [-20.5227, -6.4087, -19.2017],
              [-8.7339, 12.7233, -10.4712],
              [-16.0912, -13.8846, 3.1460],
              [0.5381, -6.2621, -7.7534],
              [-12.8680, -3.4454, 7.4327]],

             [[-12.9428, -2.4699, -10.3607],
              [-18.0424, -3.6509, 3.5435],
              [3.2357, -0.7375, -11.1968],
              [-19.1509, -18.1645, -20.9853],
              [-1.1792, -22.6442, -12.5420],
              [-13.3646, -18.7175, 0.6305],
              [8.8235, -15.3310, -0.8600]],

             [[1.4339, -12.6501, -7.9228],
              [-2.2071, -26.2669, -12.1616],
              [-28.4725, -9.3618, 3.6961],
              [-15.9940, -19.1420, -19.8135],
              [9.6876, -9.9449, -17.4420],
              [-12.0320, -0.7099, -5.3813],
              [-2.2564, -13.2958, -16.8111]],

             [[-11.5133, -13.0389, -15.0602],
              [-30.6160, 2.2305, 2.2046],
              [-1.0405, -8.1890, -7.5102],
              [-21.9827, -8.6073, -20.7981],
              [-24.9296, -17.4062, -8.7698],
              [-20.9245, -15.1580, -14.3011],
              [-18.4567, -2.5647, -33.5925]],

             [[-0.4307, -18.0852, -15.4936],
              [-21.5592, -9.8085, -11.8501],
              [-7.8881, 0.4018, -16.6824],
              [-17.2060, -7.4546, 0.3126],
              [-23.8340, -0.4459, -24.1054],
              [-10.6462, -12.4018, -27.9820],
              [-16.6482, -19.8319, -12.3085]],

             [[-4.9994, -7.9994, -11.2154],
              [3.5213, -19.1980, -5.4001],
              [-9.1273, -11.8657, -3.3748],
              [12.6587, 7.4803, -16.1441],
              [2.3833, -8.9995, -20.8745],
              [-0.9850, -8.3160, -9.6637],
              [-4.7395, -5.6236, 2.6582]]]).astype(np.float32)
        batch = 7
        N = 19
        npoint = 7
        expected_dist, expected_idx = self.cpu_op_exec(batch, npoint, source, target)
        dist, idx = mx_driving.three_nn(torch.from_numpy(target).npu(), torch.from_numpy(source).npu())

        self.assertRtolEqual(expected_dist, dist.cpu().numpy())
        self.assertRtolEqual(expected_idx, idx.cpu().numpy())

        dist_verify, idx_verify = mx_driving.common.three_nn(torch.from_numpy(target).npu(), torch.from_numpy(source).npu())
        self.assertRtolEqual(expected_dist, dist_verify.cpu().numpy())
        self.assertRtolEqual(expected_idx, idx_verify.cpu().numpy())

    def test_three_nn_2(self):
        batch = 21
        N = 3
        npoint = 1

        source, target = cpu_gen_inputs(batch, N, npoint)

        expected_dist, expected_idx = self.cpu_op_exec(batch, npoint, source, target)
        dist, idx = mx_driving.three_nn(torch.from_numpy(target).npu(), torch.from_numpy(source).npu())

        self.assertRtolEqual(expected_dist, dist.cpu().numpy())
        self.assertRtolEqual(expected_idx, idx.cpu().numpy())

        dist_verify, idx_verify = mx_driving.common.three_nn(torch.from_numpy(target).npu(), torch.from_numpy(source).npu())
        self.assertRtolEqual(expected_dist, dist_verify.cpu().numpy())
        self.assertRtolEqual(expected_idx, idx_verify.cpu().numpy())

    def test_three_nn_3(self):
        batch = 256
        N = 12
        npoint = 21

        source, target = cpu_gen_inputs(batch, N, npoint)

        expected_dist, expected_idx = self.cpu_op_exec(batch, npoint, source, target)
        dist, idx = mx_driving.three_nn(torch.from_numpy(target).npu(), torch.from_numpy(source).npu())

        self.assertRtolEqual(expected_dist, dist.cpu().numpy())
        self.assertRtolEqual(expected_idx, idx.cpu().numpy())

        dist_verify, idx_verify = mx_driving.common.three_nn(torch.from_numpy(target).npu(), torch.from_numpy(source).npu())
        self.assertRtolEqual(expected_dist, dist_verify.cpu().numpy())
        self.assertRtolEqual(expected_idx, idx_verify.cpu().numpy())

if __name__ == "__main__":
    run_tests()