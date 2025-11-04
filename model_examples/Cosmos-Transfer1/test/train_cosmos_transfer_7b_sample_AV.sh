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

export OUTPUT_ROOT=checkpoints

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=8 -m cosmos_transfer1.diffusion.training.train --config=cosmos_transfer1/diffusion/config/config_train.py -- experiment=CTRL_7Bv1pt3_t2w_121frames_control_input_lidar_block3_pretrain