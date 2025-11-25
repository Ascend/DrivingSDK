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

export PROMPT="The video is captured from a camera mounted on a car. The camera is facing forward. The video captures a driving scene on a multi-lane highway during the day. The sky is clear and blue, indicating good weather conditions. The road is relatively busy with several cars and trucks in motion. A red sedan is driving in the left lane, followed by a black pickup truck in the right lane. The vehicles are maintaining a safe distance from each other. On the right side of the road, there are speed limit signs indicating a limit of 65 mph. The surrounding area includes a mix of greenery and industrial buildings, with hills visible in the distance. The overall environment appears to be a typical day on a highway with moderate traffic. The golden light of the late afternoon bathes the highway, casting long shadows and creating a warm, serene atmosphere. The sky is a mix of orange and blue, with the sun low on the horizon. The red sedan in the left lane reflects the golden hues, while the black pickup truck in the right lane casts a distinct shadow on the pavement. The speed limit signs stand out clearly under the fading sunlight. The surrounding greenery glows with a rich, warm tone, and the industrial buildings take on a softened appearance in the sunset."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:=0}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:=./checkpoints}"
export NUM_GPUS=1

PYTHONPATH=$(pwd)${PYTHONPATH:+:$PYTHONPATH} torchrun --nproc_per_node=${NUM_GPUS} cosmos_transfer1/diffusion/inference/transfer_multiview.py \
--checkpoint_dir $CHECKPOINT_DIR \
--video_save_name output_video_1_30_0 \
--video_save_folder outputs/sample_av_multiview \
--offload_text_encoder_model \
--guidance 3 \
--controlnet_specs assets/sample_av_hdmap_multiview_spec.json --num_gpus ${NUM_GPUS} --num_steps 30 \
--view_condition_video assets/sample_av_mv_input_rgb.mp4 \
--prompt "$PROMPT" 2>&1 | tee "outputs/inference_sample_av_multi_control.log"