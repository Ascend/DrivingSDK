#!/bin/bash

# 训练
python train.py --train_set ./Train_stop_and_go.mat --val_set ./Val_stop_and_go.mat

# 验证
python evaluate.py --test_set ./Test_stop_and_go.mat

epochs=30
samples=129591

# 训练总时长
TrainingTime=`grep -a '耗时'  ./trained_models/npu_train/train.log|awk -F "耗时: " '{print $2}'|awk -F " " '{print $1}'| awk '{a+=$1} END {printf("%.3f",a)}'`

echo "TrainingTime = ${TrainingTime} 秒"

# 计算每个 epoch 的平均时间
epochTime=$(echo "scale=3; $TrainingTime / $epochs" | bc)
FPS=$(echo "scale=3; $samples / $epochTime" | bc)

echo "平均每个 Epoch: ${epochTime} 秒"
echo "FPS: ${FPS}"

# 提取最后一个RMSE的Mean值
RMSE_Mean=$(grep 'RMSE.*Mean=' ./trained_models/npu_train/evaluation.log \
  | tail -n 1 \
  | awk -F "Mean=" '{print $2}' \
  | awk '{print $1}')

echo "RMSE Mean: ${RMSE_Mean}"