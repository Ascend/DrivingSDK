#!/bin/bash
export CPU_AFFINITY_CONF=1
export TASK_QUEUE_ENABLE=2
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MULTI_STREAM_MEMORY_REUSE=2
export TOKENIZERS_PARALLELISM=false
DEFAULT_ASCEND_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-${DEFAULT_ASCEND_DEVICES}}

ROOT_DIR=$(pwd)
LOG_DIR="./lingbot_checkpoints/$(date +%Y%m%d)_logs"
mkdir -p ${LOG_DIR}

num_npus=16
micro_batch_size=32
global_batch_size=512
config_path=${ROOT_DIR}/configs/vla/robotwin_load20000h.yaml
dataset_path=${ROOT_DIR}/click_bell_aloha_repo
model_path=${ROOT_DIR}/lingbot-vla-4b
tokenizer_path=${ROOT_DIR}/Qwen2.5-VL-3B-Instruct
output_dir=${ROOT_DIR}/output
nnodes=${NNODES:-1}
node_rank=${NODE_RANK:-0}
master_addr=${MASTER_ADDR:-0.0.0.0}
master_port=${MASTER_PORT:-62500}

# 解析命令行参数
for para in $*
do
    if [[ $para == --num_npus* ]];then
        num_npus=`echo ${para#*=}`
    elif [[ $para == --micro_batch_size* ]];then
        micro_batch_size=`echo ${para#*=}`
    elif [[ $para == --global_batch_size* ]];then
        global_batch_size=`echo ${para#*=}`
    elif [[ $para == --config_path* ]];then
        config_path=`echo ${para#*=}`
    elif [[ $para == --dataset_path* ]];then
        dataset_path=`echo ${para#*=}`
    elif [[ $para == --model_path* ]];then
        model_path=`echo ${para#*=}`
    elif [[ $para == --tokenizer_path* ]];then
        tokenizer_path=`echo ${para#*=}`
    elif [[ $para == --output_dir* ]];then
        output_dir=`echo ${para#*=}`
    elif [[ $para == --nnodes* ]];then
        nnodes=`echo ${para#*=}`
    elif [[ $para == --node_rank* ]];then
        node_rank=`echo ${para#*=}`
    elif [[ $para == --master_addr* ]];then
        master_addr=`echo ${para#*=}`
    elif [[ $para == --master_port* ]];then
        master_port=`echo ${para#*=}`
    fi
done

# 参数检查
if [[ "$config_path" == "" ]];then
    echo "[Error] para \"config_path\" must be configured."
    exit 1
fi
if [ ! -f "$config_path" ]; then
    echo "[Error] config path \"$config_path\" does not exist."
    exit 1
fi
if [[ "$dataset_path" == "" ]];then
    echo "[Error] para \"dataset_path\" must be configured."
    exit 1
fi
if [ ! -d "$dataset_path" ]; then
    echo "[Error] dataset path \"$dataset_path\" does not exist."
    exit 1
fi
if [ ! -d "$model_path" ]; then
    echo "[Error] model path \"$model_path\" does not exist."
    exit 1
fi
if [ ! -d "$tokenizer_path" ]; then
    echo "[Error] tokenizer path \"$tokenizer_path\" does not exist."
    exit 1
fi

mkdir -p ${output_dir}
LOG_FILE="${LOG_DIR}/train_${num_npus}p_lingbot.log"

torchrun \
    --nnodes=${nnodes} \
    --nproc-per-node=${num_npus} \
    --node-rank=${node_rank} \
    --master-addr=${master_addr} \
    --master-port=${master_port} \
    tasks/vla/train_lingbotvla.py \
    ${config_path} \
    --model.model_path ${model_path} \
    --data.train_path ${dataset_path} \
    --train.output_dir ${output_dir} \
    --model.tokenizer_path ${tokenizer_path} \
    --train.micro_batch_size ${micro_batch_size} \
    --train.global_batch_size ${global_batch_size} \
    --train.use_compile false \
    2>&1 | tee ${LOG_FILE}     # 记录log同时打印至终端


# 检查日志文件是否存在
if [ ! -f "${LOG_FILE}" ]; then
    echo "Log Error: 日志文件 ${LOG_FILE} 未生成"
    exit 1
fi

FinalLoss=$(grep -aE "Step [0-9]+/[0-9]+, .*Loss " ${LOG_FILE} | tail -n1 | sed -nE "s/.*Loss ([0-9.eE+-]+),.*/\1/p")


# 打印Loss
echo "Final Loss: ${FinalLoss}"
