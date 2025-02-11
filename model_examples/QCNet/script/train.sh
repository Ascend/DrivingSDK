#!/usr/bin/env bash

# 使用tcmalloc内存分配库
export LD_PRELOAD=/usr/local/lib/libtcmalloc.so.4

# 使用算子下发队列
export TASK_QUEUE_ENABLE=2
# 使用绑核
export CPU_AFFINITY_CONF=1
# 设置算子缓存数量
export ACLNN_CACHE_LIMIT=500000
export HOST_CACHE_CAPACITY=50

# /path/to/datasets 请更改为存放数据的路径
python QCNet/train_qcnet.py --root /path/to/datasets --train_batch_size 4 \
    --val_batch_size 4 --test_batch_size 4 --devices 8 --num_workers 8 --dataset argoverse_v2 \
    --num_historical_steps 50 --num_future_steps 60 --num_recurrent_steps 3 \
    --pl2pl_radius 150 --time_span 10 --pl2a_radius 50 --a2a_radius 50 \
    --num_t2m_steps 30 --pl2m_radius 150 --a2m_radius 150 --T_max 64 --max_epochs 64
