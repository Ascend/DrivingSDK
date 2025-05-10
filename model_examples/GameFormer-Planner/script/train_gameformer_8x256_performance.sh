#!/usr/bin/env bash
# -------------------------------------------------- #
GPUS=$1
EPOCHS=$2                                              #    
# -------------------------------------------------- #
GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))

BATCH_SIZE=256

#设置是否开启taskque,0-关闭/1-开启/2-流水优化
export TASK_QUEUE_ENABLE=2
#设置是否开启均匀绑核,0-关闭/1-开启粗粒度绑核/2-开启细粒度绑核
export CPU_AFFINITY_CONF=1
#HCCL白名单开关,1-关闭/0-开启
export HCCL_WHITELIST_DISABLE=1

torchrun --nnodes=1 \
        --nproc_per_node=$GPUS_PER_NODE \
        GameFormer-Planner/train_predictor.py \
        --batch_size=$BATCH_SIZE \
        --train_epochs=$EPOCHS \
        --learning_rate=1e-4 \
        --train_set=nuplan/nuplan_processed/train \
        --valid_set=nuplan/nuplan_processed/val \
        --name="log_8x256" \
    
echo "------------------ Final result ------------------"
# 提取并计算平均训练时间（去除前10个step的不稳定数据）
avg_time=$(grep 'avg_train_time' training_log/log_8x256/train.log | awk '
{
    # 提取 Step 编号和 avg_train_time 的值
    if (match($0, /Step \[([0-9]+).*avg_train_time: ([0-9.]+)/, arr)) {
        step = arr[1]
        time = arr[2]
        # 排除 Step 10 的数据
        if (step != 10) {
            sum += time
            count++
        }
    }
}
END {
    if (count > 0) {
        # print(count)
        printf "%.4f", sum / count
    } else {
        printf "0"
    }
}')

# 输出结果
avg_fps_awk=$(awk "BEGIN {printf \"%.4f\", $GPUS_PER_NODE * $BATCH_SIZE / $avg_time}")
echo "Model Performance(平均单步耗时): $avg_time"
echo "Model Performance(平均FPS): $avg_fps_awk"

