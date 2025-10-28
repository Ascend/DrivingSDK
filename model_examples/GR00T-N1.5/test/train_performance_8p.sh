#!/bin/bash
export CPU_AFFINITY_CONF=1
export TASK_QUEUE_ENABLE=2

LOG_DIR="./so101-checkpoints/$(date +%Y%m%d)_logs"
mkdir -p ${LOG_DIR}

batch_size=64
num_npu=8
max_steps=1000
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

LOG_FILE="${LOG_DIR}/test_${num_npu}p_performance.log"

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
   > ${LOG_DIR}/test_${num_npu}p_performance.log 2>&1 


# 检查日志文件是否存在
if [ ! -f "${LOG_FILE}" ]; then
    echo "Log Error: 日志文件 ${LOG_FILE} 未生成"
    exit 1
fi

# 从日志中提取时间信息计算FPS
stepstart_time=$(grep " 100/${max_steps} " ${LOG_FILE} | tail -n1 | awk -F '[\\[<]' '{print $2}' | xargs)
stepend_time=$(grep " 1000/${max_steps} " ${LOG_FILE} | tail -n1 | awk -F '[\\[<]' '{print $2}' | xargs)

# 检查时间是否获取成功
if [ -z "$stepstart_time" ] || [ -z "$stepend_time" ]; then
    echo "Log Error: 未找到时间记录"
    exit 1
fi

# 计算时间
convert_time_to_sec() {
    local time_str=$1
    local IFS=':'
    local parts=($time_str)
    local sec=0
    if [ ${#parts[@]} -eq 3 ]; then
        sec=$((10#${parts[0]} * 3600 + 10#${parts[1]} * 60 + 10#${parts[2]})) 
    elif [ ${#parts[@]} -eq 2 ]; then
        sec=$((10#${parts[0]} * 60 + 10#${parts[1]}))
    else
        echo "0" 
    fi
    echo ${sec}
}

start_time=$(convert_time_to_sec "${stepstart_time}")
end_time=$(convert_time_to_sec "${stepend_time}")
total_steps=900
total_time=$((end_time - start_time))

step_time=$(echo "scale=4; ${total_time} / ${total_steps}" | bc)    # 保留4位小数
FPS=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'* '${num_npu}'/'${step_time}'}'`

# 打印性能
echo "Step start time in seconds: ${start_time}"
echo "Step end time in seconds: ${end_time}"
echo "Step time: ${step_time}"
echo "FPS: ${FPS}"
echo "FPS: $FPS" >>${LOG_FILE}