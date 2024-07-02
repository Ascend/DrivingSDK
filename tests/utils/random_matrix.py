#!/usr/bin/env python3
import random
import numpy as np


# float16[-14,16], float32[-126,128], float64[-1022,1024], int16[0,15], int32[0,31], int64[0,63]
# random_value(-7, 8, (1, 2, 3), np.float32, True, True, False, False)
def random_value(min_log, max_log, size, dtype=np.float32,
    nega_flag=True, zero_flag=True, inf_flag=False, nan_flag=False):
    matrix_log = np.random.uniform(low=min_log, high=max_log, size=size).astype(np.float32)
    matrix = np.exp2(matrix_log).astype(dtype)
    flag_value = int(zero_flag) + int(inf_flag) + int(nan_flag)
    size_value = np.prod(size)
    p0 = 0.1
    if (flag_value > 0) and (size_value > 0):
        p0 = 0.1 / flag_value / size_value # 10%
    if nega_flag:
        matrix *= np.random.choice(a=[1, -1], size=size, p=[0.5, 0.5])
    if zero_flag:
        matrix *= np.random.choice(a=[1, 0], size=size, p=[1 - p0, p0])
    if inf_flag:
        np_inf = np.array([np.inf]).astype(dtype)[0]
        matrix += np.random.choice(a=[0, np_inf], size=size, p=[1 - p0, p0])
    if nan_flag:
        np_nan = np.array([np.nan]).astype(dtype)[0]
        matrix += np.random.choice(a=[0, np_nan], size=size, p=[1 - p0, p0])
    return matrix


# random_size(-7, 8, ([1, 4], [1, 4], [3]), np.float32, True, True, False, False)
def random_size(min_log, max_log, size_range, dtype=np.float32,
    nega_flag=True, zero_flag=True, inf_flag=False, nan_flag=False):
    size = []
    dim = len(size_range)
    for i in range(dim):
        if len(size_range[i]) == 1:
            size.append(size_range[i][0])
        else:
            random.seed()
            size.append(random.randint(size_range[i][0], size_range[i][1]))
    return random_value(min_log, max_log, size, dtype, nega_flag, zero_flag, inf_flag, nan_flag)
