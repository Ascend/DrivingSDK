#!/bin/bash

# 训练用例信息
NETWORK="FlashOCC_R50"
DEVICE_TYPE=$(uname -m)

WORLD_SIZE=8
BATCH_SIZE=4
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

# 设置 Shape 数据缓存
export HOST_CACHE_CAPACITY=20
# 设置是否开启 combined 标志, 0-关闭/1-开启
export COMBINED_ENABLE=1

# bevdet_occ.py 适配 npu
sed -i 's/^from multiprocessing.dummy import Pool as ThreadPool/# from multiprocessing.dummy import Pool as ThreadPool/' projects/mmdet3d_plugin/models/detectors/bevdet_occ.py
sed -i 's/^from ...ops import nearest_assign/# from ...ops import nearest_assign/' projects/mmdet3d_plugin/models/detectors/bevdet_occ.py
sed -i 's/^\(\s*\)is_cuda\s*=\s*True/\1is_cuda = False/' projects/mmdet3d_plugin/models/detectors/bevdet_occ.py

# 修改配置文件中的 max_epoch
sed -i 's/runner = dict(type='\''EpochBasedRunner'\'', max_epoch=24)/runner = dict(type='\''EpochBasedRunner'\'', max_epoch=1)/' "./projects/configs/flashocc/flashocc-r50.py"

# 训练开始时间
start_time=$(date +%s)

# 开始训练
echo "[FlashOCC] Training..."
bash ./tools/dist_train.sh ./projects/configs/flashocc/flashocc-r50.py ${WORLD_SIZE} --work-dir ${OUTPUT_PATH}/work_dir > ${OUTPUT_PATH}/train.log 2>&1 &
wait

# 训练结束时间
end_time=$(date +%s)

# 恢复配置文件中的 total_epochs
sed -i 's/runner = dict(type='\''EpochBasedRunner'\'', max_epoch=1)/runner = dict(type='\''EpochBasedRunner'\'', max_epoch=24)/' "./projects/configs/flashocc/flashocc-r50.py"


# 训练结果
echo "------------------ Final result ------------------"

# 总训练时长
e2e_time=$(($end_time - $start_time))
echo "[FlashOCC] E2E Training Time (sec) : ${e2e_time}"

# 单迭代训练时长
per_step_time=$(grep -o ", time: [0-9.]*" ${OUTPUT_PATH}/train.log | tail -n 30 | grep -o "[0-9.]*" | awk '{sum += $1} END {print sum/NR}')
echo "[FlashOCC] Average Per Step Training Time (sec) : ${per_step_time}"

# 吞吐量
actual_fps=$(awk BEGIN'{print ('$BATCH_SIZE' * '$WORLD_SIZE') / '$per_step_time'}')
echo "[FlashOCC] Final Performance images/sec : ${actual_fps}"

# loss 值
actual_loss=$(grep -o "loss: [0-9.]*" ${OUTPUT_PATH}/train.log | awk 'END {print $NF}')
echo "[FlashOCC] Final Train Loss : ${actual_loss}"

# 验证精度
if [[ ${TOTAL_EPOCHS} == 24 ]]; then
    # 验证精度
    echo "[FlashOCC] Evaluating ..."
    bash ./tools/dist_test.sh ./projects/configs/flashocc/flashocc-r50.py ${OUTPUT_PATH}/work_dir/epoch_24.pth ${WORLD_SIZE} --eval mAP > ${OUTPUT_PATH}/eval_result.log 2>&1 &
    wait
    mIoU=$(grep -o "mIoU of 6019 samples: [0-9.]*" ${OUTPUT_PATH}/eval_result.log | awk 'END {print $NF}')
    echo "[FlashOCC] mIoU : ${mIoU}"
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
fi
