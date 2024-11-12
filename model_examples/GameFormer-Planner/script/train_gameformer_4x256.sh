#!/usr/bin/env bash
# -------------------------------------------------- #
GPUS=$1
EPOCHS=$2                                              #    
# -------------------------------------------------- #
GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))

export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export HCCL_WHITELIST_DISABLE=1
export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE

torchrun --nnodes=1 \
        --nproc_per_node=$GPUS_PER_NODE \
        train_predictor.py \
        --batch_size=256 \
        --train_epochs=$EPOCHS \
        --learning_rate=1e-4 \
        --train_set=nuplan/nuplan_processed/train \
        --valid_set=nuplan/nuplan_processed/val \
        --name="log_4x256" \

