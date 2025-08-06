#!/bin/bash

# 设置环境变量
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=src:externals/waymo-open-dataset/src:externals/TATS/tats/fvd


#设置是否开启taskque,0-关闭/1-开启/2-流水优化
export TASK_QUEUE_ENABLE=2
#设置是否开启均匀绑核,0-关闭/1-开启粗粒度绑核/2-开启细粒度绑核
export CPU_AFFINITY_CONF=2
# #HCCL白名单开关,1-关闭/0-开启
export HCCL_WHITELIST_DISABLE=1
# HCCL确定性计算
export HCCL_DETERMINISTIC=true
# Matmul确定性计算
export CLOSE_MATMUL_K_SHIFT=1

# 设置分布式训练参数
NNODES=1
NPROC_PER_NODE=8
NODE_RANK=0
MASTER_ADDR=$1
MASTER_PORT=23500

# 配置文件路径和输出目录
CONFIG_FILE="configs/ctsd/single_dataset/ctsd_35_crossview_tirda_bm_nusc_a.json"
OUTPUT_DIR="output/train_out"
mkdir -p "$OUTPUT_DIR"
export CACHE_ENV_VAR=0

# 启动分布式训练
python -m torch.distributed.run \
    --nnodes $NNODES \
    --nproc-per-node $NPROC_PER_NODE \
    --node-rank $NODE_RANK \
    --master-addr $MASTER_ADDR \
    --master-port $MASTER_PORT \
    src/dwm/train.py \
    -c $CONFIG_FILE \
    -o $OUTPUT_DIR 2>&1 | tee "${OUTPUT_DIR}/train.log"
    