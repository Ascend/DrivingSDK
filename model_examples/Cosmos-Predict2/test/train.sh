#!/bin/bash

#开启taskque
export TASK_QUEUE_ENABLE=1
#开启粗粒度绑核
export CPU_AFFINITY_CONF=2
#HCCL白名单开关,1-关闭/0-开启
export HCCL_WHITELIST_DISABLE=1
# HCCL确定性计算
export HCCL_DETERMINISTIC=true
# Matmul确定性计算
export CLOSE_MATMUL_K_SHIFT=1
# 开启内存池扩展段功能
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

LOG_DIR="./checkpoints/posttraining/$(date +%Y%m%d)_logs"
mkdir -p ${LOG_DIR}

# 默认参数
EXP=predict2_video2world_training_2b_cosmos_nemo_assets
nproc_per_node=8
nnodes=2
node_rank=0
hccl_if_ip=1.1.1.1
master_addr=1.1.1.1
master_port=12345

# 解析命令行参数
for para in $*
do
    if [[ $para == --EXP* ]];then
        EXP=`echo ${para#*=}`
    elif [[ $para == --nproc_per_node* ]];then
        nproc_per_node=`echo ${para#*=}`
    elif [[ $para == --nnodes* ]];then
        nnodes=`echo ${para#*=}`
    elif [[ $para == --node_rank* ]];then
        node_rank=`echo ${para#*=}`
    elif [[ $para == --master_addr* ]];then
        master_addr=`echo ${para#*=}`
    elif [[ $para == --hccl_if_ip* ]];then
        hccl_if_ip=`echo ${para#*=}`
    elif [[ $para == --master_port* ]];then
        master_port=`echo ${para#*=}`
    fi
done

# 参数检查
if [[ "$EXP" == "" ]];then
    echo "[Error] para \"EXP\" must be configured."
    exit 1
fi

# 判断脚本分支
# video2world-2b 单机8卡运行
if [[ "$EXP" == "predict2_video2world_training_2b_cosmos_nemo_assets" ]];then
    echo "start with video2world-2b"
    LOG_FILE="${LOG_DIR}/video2world_2b_${nproc_per_node}p.log"
    
    torchrun --nproc_per_node=${nproc_per_node} --master_port=${master_port} -m scripts.train --config=cosmos_predict2/configs/base/config.py \
    -- experiment=${EXP} 2>&1 | tee "${LOG_FILE}"

# video2world-14b 双机16卡运行
elif [[ "$EXP" == "predict2_video2world_training_14b_cosmos_nemo_assets" ]];then
    export HCCL_IF_IP=${hccl_if_ip}

    echo "start with video2world-14b"
    LOG_FILE="${LOG_DIR}/video2world_14b_${nproc_per_node}p_${node_rank}.log"
    
    torchrun --nproc_per_node=${nproc_per_node} --nnodes=${nnodes} --node_rank=${node_rank} --master_addr=${master_addr} \
    --master_port=${master_port} -m scripts.train --config=cosmos_predict2/configs/base/config.py  -- experiment=${EXP} 2>&1 | tee "${LOG_FILE}"

# text2image-2b 单机8卡运行
elif [[ "$EXP" == "predict2_text2image_training_2b_cosmos_nemo_assets" ]];then
    echo "start with text2image-2b"
    LOG_FILE="${LOG_DIR}/text2image_2b_${nproc_per_node}p.log"
    
    torchrun --nproc_per_node=${nproc_per_node} --master_port=${master_port} -m scripts.train --config=cosmos_predict2/configs/base/config.py \
    -- experiment=${EXP} 2>&1 | tee "${LOG_FILE}"

# text2image-14b 单机8卡运行
elif [[ "$EXP" == "predict2_text2image_training_14b_cosmos_nemo_assets" ]];then
    echo "start with text2image-14b"
    LOG_FILE="${LOG_DIR}/text2image_14b_${nproc_per_node}p.log"
    
    torchrun --nproc_per_node=${nproc_per_node} --master_port=${master_port} -m scripts.train --config=cosmos_predict2/configs/base/config.py \
    -- experiment=${EXP} 2>&1 | tee "${LOG_FILE}"
fi


# 从日志中提取时间信息计算FPS
start_step=5  # 起始步骤
end_step=25    # 结束步骤
total_steps=$((end_step - start_step))  # 总步骤数

# 提取时间
stepstart_time=$(grep "cosmos_predict2/callbacks/iter_speed.py.* Iteration ${start_step}: "  ${LOG_FILE} | tail -n1 | awk -F '[\\[ |]' '{print $3}' | xargs)
stepend_time=$(grep "cosmos_predict2/callbacks/iter_speed.py.* ${end_step} :"  ${LOG_FILE} | tail -n1 | awk -F '[\\[ |]' '{print $3}' | xargs)

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
total_time=$((end_time - start_time))

step_time=$(echo "scale=4; ${total_time} / ${total_steps}" | bc)    # 保留4位小数
FPS=`awk 'BEGIN{printf "%.4f\n", '${nproc_per_node}'/'${step_time}'}'`

# 打印性能
echo "Step time: ${step_time}" | tee -a "${LOG_FILE}"
echo "FPS: ${FPS}" | tee -a "${LOG_FILE}"

if [ ! -f "${LOG_FILE}" ]; then
    echo "Log Error: 日志文件 ${LOG_FILE} 未生成"
    exit 1
fi

# 打印最终Loss
FinalLoss=$(grep "Loss: " "${LOG_FILE}" | tail -1 | awk 'match($0, /Loss: ([0-9]+\.?[0-9]*)/, arr) {print arr[1]}')

awk '
    /Loss: [0-9.]+/ {
        # 提取Loss数值
        loss = substr($0, index($0, "Loss: ") + 6)
        t += loss; c++
    }
    END {
        if (c > 0) {
            printf "Mean Loss: %.4f\n", t/c
        } else {
            print "Loss Not Found"
        }
    }
' "$LOG_FILE" | tee -a "${LOG_FILE}"

echo "Final Loss: ${FinalLoss}" | tee -a "${LOG_FILE}"