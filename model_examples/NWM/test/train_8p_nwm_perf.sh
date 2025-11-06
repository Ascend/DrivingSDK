#!/bin/bash

# 训练用例信息
NETWORK="NWM"
DEVICE_TYPE=$(uname -m)

WORLD_SIZE=8
BATCH_SIZE=12

# 训练用例名称
CASE_NAME=${NETWORK}_${WORLD_SIZE}p_perf
echo "[${NETWORK}] CASE_NAME = ${CASE_NAME}"

# 创建输出目录
OUTPUT_PATH=./test/output/${CASE_NAME}

if [ -d "${OUTPUT_PATH}" ]; then
  rm -rf "${OUTPUT_PATH}"
fi

mkdir -p "${OUTPUT_PATH}"
echo "[${NETWORK}] OUTPUT_PATH = ${OUTPUT_PATH}"


# 配置环境变量

# 将 Host 日志输出到串口, 0-关闭/1-开启
export ASCEND_SLOG_PRINT_TO_STDOUT=0
# 设置默认日志级别, 0-debug/1-info/2-warning/3-error
export ASCEND_GLOBAL_LOG_LEVEL=3
# 设置Event日志开启标志, 0-关闭/1-开启
export ASCEND_GLOBAL_EVENT_ENABLE=0

# 算子下发队列优化
export TASK_QUEUE_ENABLE=2
# 绑核
export CPU_AFFINITY_CONF=1
# 使能内存池扩展段功能，由PyTorch管理虚拟地址和物理地址的映射关系，降低内存碎片化
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

while [[ $# -gt 0 ]]; do 
  case $1 in
    --MAX-EPOCHS|--max-epochs)
      MAX_EPOCHS="$2"
      shift 2
      ;;
    --help)
      echo "用法: $0 [选项]"
      echo "选项:"
      echo " --MAX-EPOCHS 设置 MAX_EPOCHS (默认: 20)"
      echo " --help 显示帮助信息"
      exit 0
      ;;
    *)
      echo "未知选项: $1"
      echo "使用 --help 查看帮助"
      exit 1
      ;;
  esac
done

echo "MAX_EPOCHS: $MAX_EPOCHS"

# 修改参数
sed -i "s/max_epochs: [0-9]\+/max_epochs: $MAX_EPOCHS/g" config/nwm_cdit_xl.yaml

# 训练开始时间
start_time=$(date +%s)

# 开始训练
echo "[${NETWORK}] Training..."
torchrun --nproc_per_node=${WORLD_SIZE} \
  train.py --config config/nwm_cdit_xl.yaml --ckpt-every 2000 --eval-every 10000 --bfloat16 1 --epochs $MAX_EPOCHS --torch-compile 0 \
  2>&1 | tee ${OUTPUT_PATH}/train.log
wait

# 训练结束时间
end_time=$(date +%s)

# 训练结果
echo "------------------ Final result ------------------"

# 总训练时长
e2e_time=$(($end_time - $start_time))
echo "[${NETWORK}] E2E Training Time (sec) : ${e2e_time}"

# 计算平均批次时间和SPS
samples_per_sec=$(grep -a 'Samples/Sec: ' "${OUTPUT_PATH}/train.log" | tail -n 1 | awk -F 'Samples/Sec: ' '{print $2}' | awk -F ', ' '{print $1}')
echo "[${NETWORK}] Final Performance Samples/Sec : ${samples_per_sec}"

# 将关键信息打印到 ${CASE_NAME}.log 中
echo "Network = ${NETWORK}" > "${OUTPUT_PATH}/${CASE_NAME}.log"
echo "DeviceType = ${DEVICE_TYPE}" >> "${OUTPUT_PATH}/${CASE_NAME}.log"
echo "RankSize = ${WORLD_SIZE}" >> "${OUTPUT_PATH}/${CASE_NAME}.log"
echo "BatchSize = ${BATCH_SIZE}" >> "${OUTPUT_PATH}/${CASE_NAME}.log"
echo "CaseName = ${CASE_NAME}" >> "${OUTPUT_PATH}/${CASE_NAME}.log"
echo "E2ETrainingTime = ${e2e_time}" >> "${OUTPUT_PATH}/${CASE_NAME}.log"
echo "ActualSPS = ${samples_per_sec}" >> "${OUTPUT_PATH}/${CASE_NAME}.log"