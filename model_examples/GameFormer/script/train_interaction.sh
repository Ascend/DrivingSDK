#!/usr/bin/env bash
# -------------------------------------------------- #
GPUS=$1                                              #    
# -------------------------------------------------- #
GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))

MASTER_PORT=${MASTER_PORT:-28596}

# 使用tcmalloc进行内存资源分配
export LD_PRELOAD=/usr/local/lib/libtcmalloc.so.4
# 配置算子二进制文件缓存数量
export ACLNN_CACHE_LIMIT=1000000
# 配置算子shape缓存数量
export HOST_CACHE_CAPACITY=50
# 使能算子下发流水
export TASK_QUEUE_ENABLE=2
# 使能CPU绑核
export CPU_AFFINITY_CONF=1

cd GameFormer/interaction_prediction
python3 -m torch.distributed.launch \
        --nproc_per_node=$GPUS_PER_NODE \
        --master_port=$MASTER_PORT \
        train.py \
        --batch_size=512 \
        --learning_rate=5e-4 \
        --train_set=waymo/motion/training_processed \
        --valid_set=waymo/motion/validation_processed \
        --name=train_log_512_5e-4 \
        --workers=8 \
        