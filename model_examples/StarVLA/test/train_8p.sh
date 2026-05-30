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
max_train_steps=150000
per_device_batch_size=4
data_mix=robotwin
base_vlm=./playground/Pretrained_models/Qwen3-VL-4B-Instruct
config_yaml=./examples/Robotwin/train_files/starvla_cotrain_robotwin_abs.yaml
Framework_name=QwenOFT
freeze_module_list=''
run_root_dir=./results/Checkpoints
save_interval=10000
logging_frequency=100
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
    elif [[ $para == --save_interval* ]];then
        save_interval=`echo ${para#*=}`
    fi
done

if [ ! -d "$base_vlm" ]; then
    echo "[Error] base_vlm path \"$base_vlm\" does not exist."
    exit 1
fi

global_batch_size=$((num_processes * per_device_batch_size))
run_id=accuracy_${num_processes}p_bs${global_batch_size}_${data_mix}

LOG_DIR="./results/Logs/$(date +%Y%m%d)"
mkdir -p ${LOG_DIR}
LOG_FILE="${LOG_DIR}/train_${num_processes}p_accuracy.log"

echo "============================================"
echo "StarVLA Accuracy Long-run Test"
echo "  num_processes:       ${num_processes}"
echo "  per_device_batch_size: ${per_device_batch_size}"
echo "  global_batch_size:   ${global_batch_size}"
echo "  max_train_steps:     ${max_train_steps}"
echo "  save_interval:       ${save_interval}"
echo "  data_mix:            ${data_mix}"
echo "  base_vlm:            ${base_vlm}"
echo "  log_file:            ${LOG_FILE}"
echo "============================================"

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/

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
  --trainer.save_interval ${save_interval} \
  --trainer.logging_frequency ${logging_frequency} \
  --trainer.eval_interval 1000 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_Robotwin \
  --wandb_entity starvla \
  2>&1 | tee ${LOG_FILE}

if [ ! -f "${LOG_FILE}" ]; then
    echo "Log Error: log file ${LOG_FILE} not generated"
    exit 1
fi

FinalLoss=$(grep "Step ${max_train_steps}," ${LOG_FILE} | tail -n1 | grep -oP "action_loss['\"]?\s*:\s*[0-9.]+" | grep -oP "[0-9.]+$")

if [ -z "$FinalLoss" ]; then
    FinalLoss=$(grep "Step.*Loss:" ${LOG_FILE} | tail -n1 | grep -oP "action_loss['\"]?\s*:\s*[0-9.]+" | grep -oP "[0-9.]+$")
fi


echo ""
echo "============================================"
echo "Accuracy Results"
echo "============================================"
echo "Total steps:         ${max_train_steps}"
echo "Final step loss:     ${FinalLoss:-N/A}"
echo "============================================"

echo "Final Loss: ${FinalLoss:-N/A}" >> ${LOG_FILE}
