#!/bin/bash

# Config文件路径
CONFIG_FILE="./projects/configs/PanoOcc/Panoptic/PanoOcc_base_4f.py"

# 设置ACL_OP_INIT_MODE=0，防止pickle报错
export ACL_OP_INIT_MODE=0

TOTAL_EPOCHS=24

CHECKPOINT_FILE=$1

OUTPUT_PATH=$2

# Number of NPUs/GPUs 
RANK_SIZE=$3 

echo "RankSize = ${RANK_SIZE}" 



# 动态生成并检测本地可用高位端口（10000-59151），直至找到未被占用的空闲端口
while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done


# 验证 segmentation 精度
echo "[PanoOcc] Evaluating Segmentation..."
PORT=${PORT:-29500}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun --nproc_per_node=${RANK_SIZE} --master_port=$PORT \
    $(dirname "$0")/test.py ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    --launcher pytorch --out 'seg_result.pkl' > ${OUTPUT_PATH}/eval_segmentation_result.log #2>&1 &
wait

# 验证 detection 精度
echo "[PanoOcc] Evaluating Detection..."
PORT=${PORT:-29500}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun --nproc_per_node=${RANK_SIZE} --master_port=$PORT \
    $(dirname "$0")/test.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    --launcher pytorch --eval bbox > ${OUTPUT_PATH}/eval_detection_result.log #2>&1 &
wait


mIoU=$(grep -o "16 categores mIoU: [0-9.]*" ${OUTPUT_PATH}/eval_segmentation_result.log | awk 'END {print $NF}')
NDS=$(grep -o "NDS: [0-9.]*" ${OUTPUT_PATH}/eval_detection_result.log | awk 'END {print $NF}')
mAP=$(grep -o "mAP: [0-9.]*" ${OUTPUT_PATH}/eval_detection_result.log | awk 'END {print $NF}')
echo "[PanoOcc] mIoU : ${mIoU}"
echo "[PanoOcc] NDS : ${NDS}"
echo "[PanoOcc] mAP : ${mAP}"

echo "mIoU = ${mIoU}" >> ${OUTPUT_PATH}/train_result.log
echo "NDS = ${NDS}" >> ${OUTPUT_PATH}/train_result.log
echo "mAP = ${mAP}" >> ${OUTPUT_PATH}/train_result.log