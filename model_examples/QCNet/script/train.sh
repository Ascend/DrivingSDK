#!/usr/bin/env bash

# 使用tcmalloc内存分配库
export LD_PRELOAD=/usr/local/lib/libtcmalloc.so.4

#设置是否开启taskque,0-关闭/1-开启/2-流水优化
export TASK_QUEUE_ENABLE=2
#设置是否开启均匀绑核,0-关闭/1-开启粗粒度绑核/2-开启细粒度绑核
export CPU_AFFINITY_CONF=1
# 设置算子缓存数量，取值范围[1, 10000000]，默认值为10000，一般情况下保持默认即可
export ACLNN_CACHE_LIMIT=500000
#设置Shape数据缓存，默认值为0，配置为非零正整数N时，系统会缓存N个频繁出现的Shape
export HOST_CACHE_CAPACITY=50

# /path/to/datasets 请更改为存放数据的路径
python QCNet/train_qcnet.py --root /path/to/datasets --train_batch_size 4 \
    --val_batch_size 4 --test_batch_size 4 --devices 8 --num_workers 8 --dataset argoverse_v2 \
    --num_historical_steps 50 --num_future_steps 60 --num_recurrent_steps 3 \
    --pl2pl_radius 150 --time_span 10 --pl2a_radius 50 --a2a_radius 50 \
    --num_t2m_steps 30 --pl2m_radius 150 --a2m_radius 150 --T_max 64 --max_epochs 64
