#!/usr/bin/env bash
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export HCCL_WHITELIST_DISABLE=1
export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE

CONFIG=$1
GPUS=$2
NNODES=$3
RANK=$4
MASTER_ADDR=$5
PORT=$6
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 python3 -m torch.distributed.run --nnodes=$NNODES --node_rank=$RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT \
    ./tools/train.py $CONFIG --launcher pytorch ${@:7}
