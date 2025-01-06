#!/usr/bin/env bash
if [ "$#" -lt 5 ]; then
    echo "Usage: $0 GPUS EPOCHS NODE_RANK MASTER_ADDR MASTER_PORT [ADDITIONAL_ARGS...]"
    exit 1
fi

# -------------------------------------------------- #
GPUS=$1                                              #
EPOCHS=$2                                            #
NODE_RANK=$3                                         #    
MASTER_ADDR=$4                                       #
MASTER_PORT=$5                                       #
# -------------------------------------------------- #
GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))

export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export HCCL_WHITELIST_DISABLE=1
export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE

torchrun --nnodes=2 \
        --nproc_per_node=$GPUS_PER_NODE \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        train_predictor.py \
        --batch_size=512 \
        --train_epochs=$EPOCHS \
        --learning_rate=1e-4 \
        --train_set=nuplan/nuplan_processed/train \
        --valid_set=nuplan/nuplan_processed/val \
        --name="log_8x512" \
