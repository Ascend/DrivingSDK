#!/bin/bash

# 训练用例信息
NETWORK="FlashOCC_R50"
DEVICE_TYPE=$(uname -m)

WORLD_SIZE=8
BATCH_SIZE=24
TOTAL_EPOCHS=1

# 训练用例名称
CASE_NAME=${NETWORK}_${WORLD_SIZE}p_bs${BATCH_SIZE}_e${TOTAL_EPOCHS}_perf
echo "[FlashOCC] CASE_NAME = ${CASE_NAME}"

# 创建输出目录
OUTPUT_PATH=./test/output/${CASE_NAME}

if [ -d ${OUTPUT_PATH} ]; then
  rm -rf ${OUTPUT_PATH}
fi

mkdir -p ${OUTPUT_PATH}
echo "[FlashOCC] OUTPUT_PATH = ${OUTPUT_PATH}"


# 配置环境变量

# 设置 device 侧日志登记为 error
msnpureport -g error -d 0
msnpureport -g error -d 1
msnpureport -g error -d 2
msnpureport -g error -d 3
msnpureport -g error -d 4
msnpureport -g error -d 5
msnpureport -g error -d 6
msnpureport -g error -d 7
# 关闭 Device 侧 Event 日志
msnpureport -e disable

# 将 Host 日志输出到串口, 0-关闭/1-开启
export ASCEND_SLOG_PRINT_TO_STDOUT=0
# 设置默认日志级别, 0-debug/1-info/2-warning/3-error
export ASCEND_GLOBAL_LOG_LEVEL=3
# 设置Event日志开启标志, 0-关闭/1-开启
export ASCEND_GLOBAL_EVENT_ENABLE=0

# HCCL 白名单开关, 1-关闭/0-开启
export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=$(hostname -I |awk '{print $1}')
export HCCL_CONNECT_TIMEOUT=1200

# 设置是否开启 taskque, 0-关闭/1-开启
export TASK_QUEUE_ENABLE=2
# 绑核
export CPU_AFFINITY_CONF=1

# 设置是否开启 combined 标志, 0-关闭/1-开启
export COMBINED_ENABLE=1

# bevdet_occ.py 适配 npu
sed -i 's/^from multiprocessing.dummy import Pool as ThreadPool/# from multiprocessing.dummy import Pool as ThreadPool/' projects/mmdet3d_plugin/models/detectors/bevdet_occ.py
sed -i 's/^from ...ops import nearest_assign/# from ...ops import nearest_assign/' projects/mmdet3d_plugin/models/detectors/bevdet_occ.py
sed -i 's/^\(\s*\)is_cuda\s*=\s*True/\1is_cuda = False/' projects/mmdet3d_plugin/models/detectors/bevdet_occ.py

# 每个step打印时间
sed -i 's/interval=50,/interval=1,/g' mmdetection3d/configs/_base_/default_runtime.py

# 训练开始时间
start_time=$(date +%s)

# 开始训练
echo "[FlashOCC] Training..."
bash ./tools/dist_train_fp16_backbone.sh ./projects/configs/flashocc/flashocc-r50-perf.py ${WORLD_SIZE} --work-dir ${OUTPUT_PATH}/work_dir > ${OUTPUT_PATH}/train.log 2>&1 &
wait

# 训练结束时间
end_time=$(date +%s)

# 训练结果
echo "------------------ Final result ------------------"

# 总训练时长
e2e_time=$(($end_time - $start_time))
echo "[FlashOCC] E2E Training Time (sec) : ${e2e_time}"

avg_time=`grep -a 'mmdet - INFO - Epoch ' ${OUTPUT_PATH}/train.log |awk -F "time: " '{print $2}' | awk -F ", " '{print $1}' | awk 'NR>10 {sum+=$1; count++} END {if (count != 0) printf("%.3f",sum/count)}'`
fps_value=$(awk BEGIN'{print ('$BATCH_SIZE' * '$WORLD_SIZE')/'$avg_time'}')

# 吞吐量
echo "[FlashOCC] Final Performance images/sec : ${fps_value}"

# 将关键信息打印到 ${CASE_NAME}.log 中
echo "Network = ${NETWORK}" > ${OUTPUT_PATH}/${CASE_NAME}.log
echo "DeviceType = ${DEVICE_TYPE}" >> ${OUTPUT_PATH}/${CASE_NAME}.log
echo "RankSize = ${WORLD_SIZE}" >> ${OUTPUT_PATH}/${CASE_NAME}.log
echo "BatchSize = ${BATCH_SIZE}" >> ${OUTPUT_PATH}/${CASE_NAME}.log
echo "CaseName = ${CASE_NAME}" >> ${OUTPUT_PATH}/${CASE_NAME}.log
echo "E2ETrainingTime = ${e2e_time}" >> ${OUTPUT_PATH}/${CASE_NAME}.log
echo "ActualFPS = ${fps_value}" >> ${OUTPUT_PATH}/${CASE_NAME}.log
