#!/usr/bin/env bash
export HCCL_WHITELIST_DISABLE=1
export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE

T=`date +%m%d%H%M`

# -------------------------------------------------- #
# Usually you only need to customize these variables #
CFG=$1                                               #
GPUS=$2                                              #
NNODES=$3
RANK=$4
MASTER_ADDR=$5
MASTER_PORT=$6
# -------------------------------------------------- #
GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))


WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/
# Intermediate files and logs will be saved to UniAD/projects/work_dirs/

if [ ! -d ${WORK_DIR}logs ]; then
    mkdir -p ${WORK_DIR}logs
fi

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --nnodes=${NNODES} \
    --node_rank=${RANK} \
    $(dirname "$0")/perf.py \
    $CFG \
    --launcher pytorch ${@:7} \
    --deterministic \
    --work-dir ${WORK_DIR} \
    2>&1 | tee ${WORK_DIR}logs/train.$T
