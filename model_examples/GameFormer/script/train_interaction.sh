#!/usr/bin/env bash
# -------------------------------------------------- #
GPUS=$1                                              #    
# -------------------------------------------------- #
GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))
MASTER_PORT=${MASTER_PORT:-28596}

# 由于同时安装了torch和tensorflow，可能导致gomp库版本不一致的问题，使用以下环境变量防止冲突
PYTHON_PATH=$(python -c "import site; print(site.getsitepackages()[0])")
LIBGOMP_PATH=$(find "$PYTHON_PATH" -name "libgomp-d22c30c5.so.1.0.0" | grep tensorflow)
export LD_PRELOAD=$LIBGOMP_PATH

BATCH_SIZE=512
LEARNING_RATE=5e-4
TRAIN_EPOCHS=30
OUTPUT_PATH="train_log"
# 请将训练集和测试集的路径改为实际路径
TRAINING_SET="/path/to/waymo/motion/training_processed/"
VALID_SET="/path/to/waymo/motion/validation_processed"

# 使用tcmalloc进行内存资源分配
export LD_PRELOAD=/usr/local/lib/libtcmalloc.so.4:$LIBGOMP_PATH
# 配置算子二进制文件缓存数量
export ACLNN_CACHE_LIMIT=1000000
# 设置Shape数据缓存，默认值为0，配置为非零正整数N时，系统会缓存N个频繁出现的Shape
export HOST_CACHE_CAPACITY=50
# 设置是否开启taskque,0-关闭/1-开启/2-流水优化
export TASK_QUEUE_ENABLE=2
# 设置是否开启均匀绑核,0-关闭/1-开启粗粒度绑核/2-开启细粒度绑核
export CPU_AFFINITY_CONF=1

cd GameFormer/interaction_prediction
if [ ! -d $OUTPUT_PATH ]; then
    mkdir -p ${OUTPUT_PATH}
fi
echo "Save train log in GameFormer/interaction_prediction/${OUTPUT_PATH}/train_${GPUS_PER_NODE}p_bs${BATCH_SIZE}_ep${TRAIN_EPOCHS}.log"
nohup python -m torch.distributed.launch \
        --nproc_per_node=$GPUS_PER_NODE \
        --master_port=$MASTER_PORT \
        train.py \
        --batch_size=${BATCH_SIZE} \
        --learning_rate=${LEARNING_RATE} \
        --train_set=${TRAINING_SET} \
        --valid_set=${VALID_SET} \
        --training_epochs=${TRAIN_EPOCHS} \
        --name=${OUTPUT_PATH} \
        --workers=8 > ${OUTPUT_PATH}/train_${GPUS_PER_NODE}p_bs${BATCH_SIZE}_ep${TRAIN_EPOCHS}.log 2>&1 &

wait
echo "-----------------------------Result-------------------------------"
tail -n 3 ${OUTPUT_PATH}/train_${GPUS_PER_NODE}p_bs${BATCH_SIZE}_ep${TRAIN_EPOCHS}.log