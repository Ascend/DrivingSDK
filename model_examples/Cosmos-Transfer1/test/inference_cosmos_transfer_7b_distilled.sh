#!/usr/bin/bash

#设置是否开启taskque,0-关闭/1-开启/2-流水优化
export TASK_QUEUE_ENABLE=2
#设置是否开启均匀绑核,0-关闭/1-开启粗粒度绑核/2-开启细粒度绑核
export CPU_AFFINITY_CONF=2
#HCCL白名单开关,1-关闭/0-开启
export HCCL_WHITELIST_DISABLE=1
# HCCL确定性计算
export HCCL_DETERMINISTIC=true
# Matmul确定性计算
export CLOSE_MATMUL_K_SHIFT=1

export CHECKPOINT_DIR="${CHECKPOINT_DIR:=./checkpoints}"
export NUM_GPU="${NUM_GPU:=1}"
PYTHONPATH=$(pwd)${PYTHONPATH:+:$PYTHONPATH} torchrun --nproc_per_node=$NUM_GPU --nnodes=1 --node_rank=0 cosmos_transfer1/diffusion/inference/transfer.py \
    --checkpoint_dir $CHECKPOINT_DIR \
    --video_save_folder outputs/example1_single_control_edge_distilled \
    --controlnet_specs assets/inference_cosmos_transfer1_single_control_edge.json \
    --offload_text_encoder_model \
    --offload_guardrail_models \
    --num_gpus $NUM_GPU \
    --use_distilled 2>&1 | tee "outputs/inference_example1_single_control_edge_distilled.log"
