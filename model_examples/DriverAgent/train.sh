#!/bin/bash

# 默认参数
MODE=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --precision)
            MODE="precision"
            shift
            ;;
        --performance)
            MODE="performance"
            shift
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: bash train.sh --precision | --performance"
            exit 1
            ;;
    esac
done

# 检查是否提供了参数
if [ -z "$MODE" ]; then
    echo "错误: 请指定运行模式"
    echo "用法: bash train.sh --precision | --performance"
    exit 1
fi

# 根据模式设置参数
if [ "$MODE" = "precision" ]; then
    echo "运行模式: 精度测试"
    epochs=30
elif [ "$MODE" = "performance" ]; then
    echo "运行模式: 性能测试"
    epochs=1
fi

samples=129591

# 训练
if [ "$MODE" = "precision" ]; then
    python train.py --train_set ./Train_stop_and_go.mat --val_set ./Val_stop_and_go.mat
else
    python train.py --train_set ./Train_stop_and_go.mat --val_set ./Val_stop_and_go.mat --pretrain_epochs 1 --train_epochs 0
fi

# 验证 (仅精度测试需要)
if [ "$MODE" = "precision" ]; then
    python evaluate.py --test_set ./Test_stop_and_go.mat
fi

# 训练总时长
TrainingTime=`grep -a '耗时'  ./trained_models/npu_train/train.log|awk -F "耗时: " '{print $2}'|awk -F " " '{print $1}'| awk '{a+=$1} END {printf("%.3f",a)}'`

echo "TrainingTime = ${TrainingTime} 秒"

# 计算每个 epoch 的平均时间
epochTime=$(echo "scale=3; $TrainingTime / $epochs" | bc)
FPS=$(echo "scale=3; $samples / $epochTime" | bc)

echo "平均每个 Epoch: ${epochTime} 秒"
echo "FPS: ${FPS}"

# 精度测试时提取RMSE
if [ "$MODE" = "precision" ]; then
    # 提取最后一个RMSE的Mean值
    RMSE_Mean=$(grep 'RMSE.*Mean=' ./trained_models/npu_train/evaluation.log \
      | tail -n 1 \
      | awk -F "Mean=" '{print $2}' \
      | awk '{print $1}')
    
    echo "RMSE Mean: ${RMSE_Mean}"
fi
