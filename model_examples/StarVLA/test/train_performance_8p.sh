#!/bin/bash
export WANDB_MODE=offline
export WANDB_OFFLINE=true
export TASK_QUEUE_ENABLE=2
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export CPU_AFFINITY_CONF=1
export TORCH_HCCL_ZERO_COPY=1   # A2 需要注释这一配置
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

###########################################################################################
# Default parameters
num_processes=8
max_train_steps=1000
per_device_batch_size=4
data_mix=robotwin
base_vlm=./playground/Pretrained_models/Qwen3-VL-4B-Instruct
config_yaml=./examples/Robotwin/train_files/starvla_cotrain_robotwin_abs.yaml
Framework_name=QwenOFT
freeze_module_list=''
run_root_dir=./results/Checkpoints
# === End of default parameters ===
###########################################################################################

for para in $*
do
    if [[ $para == --num_processes* ]];then
        num_processes=`echo ${para#*=}`
    elif [[ $para == --max_train_steps* ]];then
        max_train_steps=`echo ${para#*=}`
    elif [[ $para == --per_device_batch_size* ]];then
        per_device_batch_size=`echo ${para#*=}`
    elif [[ $para == --data_mix* ]];then
        data_mix=`echo ${para#*=}`
    elif [[ $para == --base_vlm* ]];then
        base_vlm=`echo ${para#*=}`
    fi
done

if [ ! -d "$base_vlm" ]; then
    echo "[Error] base_vlm path \"$base_vlm\" does not exist."
    exit 1
fi

global_batch_size=$((num_processes * per_device_batch_size))
run_id=perf_${num_processes}p_bs${global_batch_size}_${data_mix}

LOG_DIR="./results/Logs/$(date +%Y%m%d)"
mkdir -p ${LOG_DIR}
LOG_FILE="${LOG_DIR}/train_${num_processes}p_performance.log"

measure_start=400
measure_end=900

echo "============================================"
echo "StarVLA Performance Test"
echo "  num_processes:       ${num_processes}"
echo "  per_device_batch_size: ${per_device_batch_size}"
echo "  global_batch_size:   ${global_batch_size}"
echo "  max_train_steps:     ${max_train_steps}"
echo "  measure range:       [${measure_start}, ${measure_end}]"
echo "  data_mix:            ${data_mix}"
echo "  base_vlm:            ${base_vlm}"
echo "  log_file:            ${LOG_FILE}"
echo "============================================"

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${num_processes} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.per_device_batch_size ${per_device_batch_size} \
  --datasets.vla_data.data_mix ${data_mix} \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps ${max_train_steps} \
  --trainer.save_interval ${max_train_steps} \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval ${max_train_steps} \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  2>&1 | tee ${LOG_FILE}

if [ ! -f "${LOG_FILE}" ]; then
    echo "Log Error: log file ${LOG_FILE} not generated"
    exit 1
fi

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

stepstart_time=$(grep "Step ${measure_start}," ${LOG_FILE} | tail -n1 | awk -F '[<[]' '{print $2}' | xargs)
stepend_time=$(grep "Step ${measure_end}," ${LOG_FILE} | tail -n1 | awk -F '[<[]' '{print $2}' | xargs)

if [ -z "$stepstart_time" ] || [ -z "$stepend_time" ]; then
    echo "Log Error: time records not found in range [${measure_start}, ${measure_end}]"
    echo "Trying to extract from tqdm progress bar..."
    stepstart_time=$(grep "${measure_start}/${max_train_steps}" ${LOG_FILE} | tail -n1 | awk -F '[<[]' '{print $2}' | xargs)
    stepend_time=$(grep "${measure_end}/${max_train_steps}" ${LOG_FILE} | tail -n1 | awk -F '[<[]' '{print $2}' | xargs)
fi

if [ -z "$stepstart_time" ] || [ -z "$stepend_time" ]; then
    echo "Log Error: still cannot find time records, FPS calculation skipped"
    exit 1
fi

start_sec=$(convert_time_to_sec "${stepstart_time}")
end_sec=$(convert_time_to_sec "${stepend_time}")
measure_steps=$((measure_end - measure_start))
total_time=$((end_sec - start_sec))

if [ ${total_time} -le 0 ]; then
    echo "Log Error: invalid time range (${start_sec} -> ${end_sec})"
    exit 1
fi

step_time=$(echo "scale=4; ${total_time} / ${measure_steps}" | bc)
FPS=$(awk 'BEGIN{printf "%.2f\n", '${global_batch_size}'/'${step_time}'}')

echo ""
echo "============================================"
echo "Performance Results"
echo "============================================"
echo "Measure steps:       [${measure_start}, ${measure_end}] = ${measure_steps} steps"
echo "Start time:          ${stepstart_time} (${start_sec}s)"
echo "End time:            ${stepend_time} (${end_sec}s)"
echo "Total time:          ${total_time}s"
echo "Step time:           ${step_time}s/step"
echo "Global batch size:   ${global_batch_size}"
echo "FPS:                 ${FPS}"
echo "============================================"

echo "FPS: ${FPS}" >> ${LOG_FILE}
echo "Step time: ${step_time}s/step" >> ${LOG_FILE}
