#!/bin/bash

# 训练用例信息
NETWORK="PanoOcc_Base_4f"
DEVICE_TYPE=$(uname -m)

WORLD_SIZE=8
BATCH_SIZE=1
TOTAL_EPOCHS=24

# 获取传入的参数，重新赋值 TOTAL_EPOCHS
for para in $*
do
    if [[ $para == --epochs=* ]]; then
        TOTAL_EPOCHS=`echo ${para#*=}`
    fi
done

# 训练用例名称
CASE_NAME=${NETWORK}_${WORLD_SIZE}p_bs${BATCH_SIZE}_e${TOTAL_EPOCHS}
echo "[PanoOcc] CASE_NAME = ${CASE_NAME}"

# 创建输出目录
OUTPUT_PATH=./test/output/${CASE_NAME}

if [ -d ${OUTPUT_PATH} ]; then
  rm -rf ${OUTPUT_PATH}
fi

mkdir -p ${OUTPUT_PATH}
echo "[PanoOcc] OUTPUT_PATH = ${OUTPUT_PATH}"


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
#配置HCCL的初始化root通信网卡IP
export HCCL_IF_IP=$(hostname -I |awk '{print $1}')
#配置不同设备之间socket建链过程的等待时间，取值范围[120, 7200]，默认120，单位s
export HCCL_CONNECT_TIMEOUT=1200

#设置是否开启taskque,0-关闭/1-开启/2-流水优化
export TASK_QUEUE_ENABLE=2
#设置是否开启均匀绑核,0-关闭/1-开启粗粒度绑核/2-开启细粒度绑核
export CPU_AFFINITY_CONF=1

#设置Shape数据缓存，默认值为0，配置为非零正整数N时，系统会缓存N个频繁出现的Shape
export HOST_CACHE_CAPACITY=20
# 设置是否开启 combined 标志, 0-关闭/1-开启
export COMBINED_ENABLE=1


# 修改配置文件中的 total_epochs
sed -i "s|total_epochs = .*|total_epochs = ${TOTAL_EPOCHS}|g" ./projects/configs/PanoOcc/Panoptic/PanoOcc_base_4f.py

# 训练开始时间
start_time=$(date +%s)

# 开始训练
echo "[PanoOcc] Training..."
bash ./tools/dist_train.sh ./projects/configs/PanoOcc/Panoptic/PanoOcc_base_4f.py ${WORLD_SIZE} ${OUTPUT_PATH}/work_dir > ${OUTPUT_PATH}/train.log 2>&1 &
wait

# 训练结束时间
end_time=$(date +%s)

# 恢复配置文件中的 total_epochs
sed -i "s|total_epochs = .*|total_epochs = 24|g" ./projects/configs/PanoOcc/Panoptic/PanoOcc_base_4f.py


# 训练结果
echo "------------------ Final result ------------------"

# 总训练时长
e2e_time=$(($end_time - $start_time))
echo "[PanoOcc] E2E Training Time (sec) : ${e2e_time}"

# 单迭代训练时长
per_step_time=$(grep -o ", time: [0-9.]*" ${OUTPUT_PATH}/train.log | tail -n 30 | grep -o "[0-9.]*" | awk '{sum += $1} END {print sum/NR}')
echo "[PanoOcc] Average Per Step Training Time (sec) : ${per_step_time}"

# 吞吐量
actual_fps=$(awk BEGIN'{print ('$BATCH_SIZE' * '$WORLD_SIZE') / '$per_step_time'}')
echo "[PanoOcc] Final Performance images/sec : ${actual_fps}"

# loss 值
actual_loss=$(grep -o "loss: [0-9.]*" ${OUTPUT_PATH}/train.log | awk 'END {print $NF}')
echo "[PanoOcc] Final Train Loss : ${actual_loss}"

# 验证精度
if [[ ${TOTAL_EPOCHS} == 24 ]]; then
    # 验证 segmentation 精度
    echo "[PanoOcc] Evaluating Segmentation..."
    bash ./tools/dist_test_seg.sh ./projects/configs/PanoOcc/Panoptic/PanoOcc_base_4f.py ${OUTPUT_PATH}/work_dir/epoch_24.pth ${WORLD_SIZE} > ${OUTPUT_PATH}/eval_segmentation_result.log 2>&1 &
    wait
    mIoU=$(grep -o "16 categores mIoU: [0-9.]*" ${OUTPUT_PATH}/eval_segmentation_result.log | awk 'END {print $NF}')
    echo "[PanoOcc] mIoU : ${mIoU}"

    # 验证 detection 精度
    echo "[PanoOcc] Evaluating Detection..."
    bash ./tools/dist_test.sh ./projects/configs/PanoOcc/Panoptic/PanoOcc_base_4f.py ${OUTPUT_PATH}/work_dir/epoch_24.pth ${WORLD_SIZE} > ${OUTPUT_PATH}/eval_detection_result.log 2>&1 &
    wait
    NDS=$(grep -o "NDS: [0-9.]*" ${OUTPUT_PATH}/eval_detection_result.log | awk 'END {print $NF}')
    mAP=$(grep -o "mAP: [0-9.]*" ${OUTPUT_PATH}/eval_detection_result.log | awk 'END {print $NF}')
    echo "[PanoOcc] NDS : ${NDS}"
    echo "[PanoOcc] mAP : ${mAP}"
fi

# 将关键信息打印到 ${CASE_NAME}.log 中
echo "Network = ${NETWORK}" > ${OUTPUT_PATH}/${CASE_NAME}.log
echo "DeviceType = ${DEVICE_TYPE}" >> ${OUTPUT_PATH}/${CASE_NAME}.log
echo "RankSize = ${WORLD_SIZE}" >> ${OUTPUT_PATH}/${CASE_NAME}.log
echo "BatchSize = ${BATCH_SIZE}" >> ${OUTPUT_PATH}/${CASE_NAME}.log
echo "CaseName = ${CASE_NAME}" >> ${OUTPUT_PATH}/${CASE_NAME}.log
echo "E2ETrainingTime = ${e2e_time}" >> ${OUTPUT_PATH}/${CASE_NAME}.log
echo "TrainingTime = ${per_step_time}" >> ${OUTPUT_PATH}/${CASE_NAME}.log
echo "ActualFPS = ${actual_fps}" >> ${OUTPUT_PATH}/${CASE_NAME}.log
echo "ActualLoss = ${actual_loss}" >> ${OUTPUT_PATH}/${CASE_NAME}.log
if [[ ${TOTAL_EPOCHS} == 24 ]]; then
    echo "mIoU = ${mIoU}" >> ${OUTPUT_PATH}/${CASE_NAME}.log
    echo "NDS = ${NDS}" >> ${OUTPUT_PATH}/${CASE_NAME}.log
    echo "mAP = ${mAP}" >> ${OUTPUT_PATH}/${CASE_NAME}.log
fi
