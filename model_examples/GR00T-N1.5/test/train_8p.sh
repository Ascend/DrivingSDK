#!/bin/bash
export CPU_AFFINITY_CONF=1
export TASK_QUEUE_ENABLE=2

LOG_DIR="./so101-checkpoints/$(date +%Y%m%d)_logs"
mkdir -p ${LOG_DIR}

batch_size=64
num_npu=8
max_steps=10000
dataset_path=./demo_data/so101-table-cleanup
base_model_path=./GR00T-N1.5-3B

# 解析命令行参数
for para in $*
do
    if [[ $para == --batch_size* ]];then
        batch_size=`echo ${para#*=}`
    elif [[ $para == --num_npu* ]];then
        num_npu=`echo ${para#*=}`
    elif [[ $para == --max_steps* ]];then
        max_steps=`echo ${para#*=}`
    elif [[ $para == --dataset_path* ]];then
        dataset_path=`echo ${para#*=}`
    elif [[ $para == --base_model_path* ]];then
        base_model_path=`echo ${para#*=}`
    fi
done

# 参数检查
if [[ "$dataset_path" == "" ]];then
    echo "[Error] para \"dataset_path\" must be configured."
    exit 1
fi
if [ ! -d "$dataset_path" ]; then
    echo "[Error] dataset path \"$dataset_path\" does not exist."
    exit 1
fi
if [ ! -d "$base_model_path" ]; then
    echo "[Error] base model path \"$base_model_path\" does not exist."
    exit 1
fi

LOG_FILE="${LOG_DIR}/test_${num_npu}p.log"

python scripts/gr00t_finetune.py \
   --dataset-path ${dataset_path} \
   --num-gpus ${num_npu} \
   --output-dir ./so101-checkpoints  \
   --max-steps ${max_steps} \
   --data-config so100_dualcam \
   --video-backend torchvision_av \
   --batch-size ${batch_size} \
   --base-model-path ${base_model_path} \
   --save-steps ${max_steps} \
   2>&1 | tee -a ${LOG_FILE}     # 记录log同时打印至终端


# 检查日志文件是否存在
if [ ! -f "${LOG_FILE}" ]; then
    echo "Log Error: 日志文件 ${LOG_FILE} 未生成"
    exit 1
fi

MeanLoss=$(grep -o "train_loss': [0-9.]*" ${LOG_FILE} | awk 'END {print $NF}')
FinalLoss=$(grep -o "{'loss': [0-9.]*" ${LOG_FILE} | awk 'END {print $NF}')

# 打印Loss
printf "Mean Loss: %.4f\n" "$MeanLoss"
echo "Final Loss: ${FinalLoss}"