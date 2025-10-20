#!/bin/bash

# 此脚本用于多卡分布式训练，须在模型根目录下运行

# Config文件路径
CONFIG_FILE="./projects/configs/PanoOcc/Panoptic/PanoOcc_base_4f.py"

# 训练用例信息
NETWORK="PanoOcc_Base_4f"
DEVICE_TYPE=$(uname -m)

# 默认单机8卡
RANK_SIZE=8 # Number of NPUs/GPUs per cluster



BATCH_SIZE=1 # 单个GPU/NPU上的BS

WORKERS_PER_GPU=6 #每个GPU/NPU分配的数据加载子进程数量

TOTAL_EPOCHS=24

PERFORMANCE_MODE=0

# 获取传入的参数
for para in $*
do  
    # 如果设置了--performance，仅验性能（仅需训练1k步），不验精度。
    # 不设置时默认为精度模式，将运行默认的TOTAL_EPOCHS数量的epochs，较为耗时
    if [[ $para == --performance ]]; then
        PERFORMANCE_MODE=1
    fi

    # 可通过入参修改单机场景下使用多少块NPU，即RANK_SIZE
    if [[ $para == --num_npu=* ]]; then
        RANK_SIZE=`echo ${para#*=}`
    fi

    # 可通过入参修改TOTAL_EPOCHS
    if [[ $para == --epochs=* ]]; then
        TOTAL_EPOCHS=`echo ${para#*=}`
    fi

    # 可通过入参设置单卡的Batch Size（对应config文件里的samples_per_gpu）
    if [[ $para == --batch_size=* ]]; then
        BATCH_SIZE=`echo ${para#*=}`
    fi

    # 可通过入参设置单卡的数据加载子进程数
    if [[ $para == --workers_per_npu=* ]]; then
        WORKERS_PER_GPU=`echo ${para#*=}`
    fi

    # 可指定其他的config文件路径
    if [[ $para == --config=* ]]; then
        CONFIG_FILE=`echo ${para#*=}`
    fi
done


GLOBAL_BATCH_SIZE=$(expr $RANK_SIZE \* $BATCH_SIZE)

# 训练用例名称
TIMESTAMP=$(date +%m%d_%H%M)
CASE_NAME=${NETWORK}_${RANK_SIZE}p_bs${BATCH_SIZE}_e${TOTAL_EPOCHS}_${TIMESTAMP}
echo "[PanoOcc] CASE_NAME = ${CASE_NAME}"

# 创建输出目录
OUTPUT_PATH=./output/${CASE_NAME}


mkdir -p ${OUTPUT_PATH}
echo "[PanoOcc] OUTPUT_PATH = ${OUTPUT_PATH}"

if [[ ${PERFORMANCE_MODE} == 0 ]]; then
    LOG_FILE=${OUTPUT_PATH}/train_8p_full.log
else
    LOG_FILE=${OUTPUT_PATH}/train_8p_performance.log
fi


# 配置环境变量
export ACLNN_CACHE_LIMIT=100000

# 将 Host 日志输出到串口, 0-关闭/1-开启
export ASCEND_SLOG_PRINT_TO_STDOUT=0
# 设置默认日志级别, 0-debug/1-info/2-warning/3-error
export ASCEND_GLOBAL_LOG_LEVEL=3
# 设置Event日志开启标志, 0-关闭/1-开启
export ASCEND_GLOBAL_EVENT_ENABLE=0

# HCCL 白名单开关, 1-关闭/0-开启
export HCCL_WHITELIST_DISABLE=1
#配置HCCL的初始化root通信网卡IP
export HCCL_IF_IP=$(hostname -I |awk '{print $1}')
#配置不同设备之间socket建链过程的等待时间，取值范围[120, 7200]，默认120，单位s
export HCCL_CONNECT_TIMEOUT=1200

#设置是否开启taskque,0-关闭/1-开启/2-流水优化
export TASK_QUEUE_ENABLE=2
#设置是否开启均匀绑核,0-关闭/1-开启粗粒度绑核/2-开启细粒度绑核
export CPU_AFFINITY_CONF=1

# 设置是否开启 combined 标志, 0-关闭/1-开启
export COMBINED_ENABLE=1

#减少显存占用
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

# 设置ACL_OP_INIT_MODE=0，防止pickle报错
export ACL_OP_INIT_MODE=0


echo "DeviceType = ${DEVICE_TYPE}"
echo "RankSize = ${RANK_SIZE}" 
echo "TotalEpochs = ${TOTAL_EPOCHS}"
echo "BatchSizePerGPU = ${BATCH_SIZE}"
echo "WorkersPerGPU = ${WORKERS_PER_GPU}"




# 训练开始时间
start_time=$(date +%s)

# 开始训练

echo "[PanoOcc] Training..."
echo "Path to realtime training logs: ${OUTPUT_PATH}"

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

# 长跑全量运行/验精度
if [[ ${PERFORMANCE_MODE} == 0 ]]; then
    nohup torchrun \
        --nnodes=${NNODES} \
        --node_rank=${NODE_RANK} \
        --master_addr=${MASTER_ADDR} \
        --nproc_per_node=${RANK_SIZE} \
        --master_port=${PORT} \
        $(dirname "$0")/train.py \
        ${CONFIG_FILE} \
        --cfg-options \
        total_epochs=${TOTAL_EPOCHS} \
        runner.max_epochs=${TOTAL_EPOCHS} \
        data.samples_per_gpu=${BATCH_SIZE} \
        data.workers_per_gpu=${WORKERS_PER_GPU} \
        --seed 0 \
        --work-dir ${OUTPUT_PATH}/work_dir \
        --launcher pytorch \
        > ${LOG_FILE} 2>&1 &

    wait

# 短跑验性能
else
    nohup torchrun \
        --nnodes=${NNODES} \
        --node_rank=${NODE_RANK} \
        --master_addr=${MASTER_ADDR} \
        --nproc_per_node=${RANK_SIZE} \
        --master_port=${PORT} \
        $(dirname "$0")/train.py \
        ${CONFIG_FILE} \
        --cfg-options \
        total_epochs=${TOTAL_EPOCHS} \
        runner.max_epochs=${TOTAL_EPOCHS} \
        data.samples_per_gpu=${BATCH_SIZE} \
        data.workers_per_gpu=${WORKERS_PER_GPU} \
        --seed 0 \
        --work-dir ${OUTPUT_PATH}/work_dir \
        --launcher pytorch \
        --performance > ${LOG_FILE} 2>&1 &

    wait
fi



# 训练结束时间
end_time=$(date +%s)


# 训练结果
echo "------------------ Final result ------------------"

# 总训练时长
e2e_time=$(($end_time - $start_time))
echo "[PanoOcc] E2E Training Time (sec) : ${e2e_time}"




# 单迭代训练时长
if [[ ${PERFORMANCE_MODE} == 0 ]]; then
    NUM_SAMPLES=30
else
    NUM_SAMPLES=5
fi

per_step_time=$(grep -o ", time: [0-9.]*" ${LOG_FILE} | tail -n ${NUM_SAMPLES} | grep -o "[0-9.]*" | awk '{sum += $1} END {print sum/NR}')
echo "[PanoOcc] Average Per Step Training Time (sec) : ${per_step_time}"

# 吞吐量
actual_fps=$(awk BEGIN'{print ('$GLOBAL_BATCH_SIZE') / '$per_step_time'}')
echo "[PanoOcc] Final Performance images/sec : ${actual_fps}"

# loss 值
actual_loss=$(grep -o "loss: [0-9.]*" ${LOG_FILE} | awk 'END {print $NF}')
echo "[PanoOcc] Final Train Loss : ${actual_loss}"


# 将关键信息打印到 log 中
echo "Network = ${NETWORK}" > ${OUTPUT_PATH}/train_result.log
echo "DeviceType = ${DEVICE_TYPE}" >> ${OUTPUT_PATH}/train_result.log
echo "RankSize = ${RANK_SIZE}" >> ${OUTPUT_PATH}/train_result.log
echo "BatchSize = ${BATCH_SIZE}" >> ${OUTPUT_PATH}/train_result.log
echo "CaseName = ${CASE_NAME}" >> ${OUTPUT_PATH}/train_result.log
echo "E2ETrainingTime = ${e2e_time}" >> ${OUTPUT_PATH}/train_result.log
echo "TrainingTime = ${per_step_time}" >> ${OUTPUT_PATH}/train_result.log
echo "ActualFPS = ${actual_fps}" >> ${OUTPUT_PATH}/train_result.log
echo "ActualLoss = ${actual_loss}" >> ${OUTPUT_PATH}/train_result.log

# 验精度
if [[ ${PERFORMANCE_MODE} == 0 ]]; then
    bash $(dirname "$0")/eval.sh ${OUTPUT_PATH}/work_dir/epoch_${TOTAL_EPOCHS}.pth ${OUTPUT_PATH} ${RANK_SIZE} 
fi
