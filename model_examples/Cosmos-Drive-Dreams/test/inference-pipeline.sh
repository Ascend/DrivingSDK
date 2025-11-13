#!/bin/bash

#设置是否开启taskque,0-关闭/1-开启/2-流水优化
export TASK_QUEUE_ENABLE=1
#设置是否开启均匀绑核,0-关闭/1-开启粗粒度绑核/2-开启细粒度绑核
export CPU_AFFINITY_CONF=1
# 使能内存池扩展段功能，由PyTorch管理虚拟地址和物理地址的映射关系，降低内存碎片化
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

echo "=== Step 1: Preprocessing Condition Videos ==="
cd cosmos-drive-dreams-toolkits
python render_from_rds_hq.py \
    -i ../assets/example \
    -o ../outputs \
    -d rds_hq_mv \
    --skip lidar
if [ $? -ne 0 ]; then
    echo "预处理失败，终止流程"
    exit 1
fi
cd ..

echo -e "\n=== Step 2: Prompt Rewriting ==="
python scripts/rewrite_caption.py \
    -i assets/example/captions \
    -o outputs/captions 
if [ $? -ne 0 ]; then
    echo "提示词改写失败，终止流程"
    exit 1
fi


echo -e "\n=== Step 3: Front-view Video Generation ==="
PYTHONPATH="cosmos-transfer1" python scripts/generate_video_single_view.py \
    --caption_path outputs/captions \
    --input_path outputs \
    --video_save_folder outputs/single_view \
    --checkpoint_dir checkpoints \
    --is_av_sample \
    --controlnet_specs assets/sample_av_hdmap_spec.json
if [ $? -ne 0 ]; then
    echo "前视图生成失败，终止流程"
    exit 1
fi


echo -e "\n=== Step 4: Multiview Video Generation ==="
CUDA_HOME=$CONDA_PREFIX PYTHONPATH="cosmos-transfer1" python scripts/generate_video_multi_view.py \
    --caption_path outputs/captions \
    --input_path outputs \
    --input_view_path outputs/single_view \
    --video_save_folder outputs/multi_view \
    --checkpoint_dir checkpoints \
    --is_av_sample \
    --controlnet_specs assets/sample_av_hdmap_multiview_spec.json
if [ $? -ne 0 ]; then
    echo "多视图生成失败，终止流程"
    exit 1
fi


echo -e "\n=== 全流程执行完成 ==="
echo "输出路径:outputs"