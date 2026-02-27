#!/bin/bash
# 启动 Cosmos-Reason1 的 SFT 或 RL 训练

# 检查参数个数
if [ $# -ne 1 ]; then
    echo "ERROR：需要一个参数"
    echo "$0 [--sft | --rl]"
    exit 1
fi

MODE="$1"

# 开启内存池扩展段功能
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# 多流内存复用
export MULTI_STREAM_MEMORY_REUSE=2
#开启粗粒度绑核
export CPU_AFFINITY_CONF=1
# 二级流水
export TASK_QUEUE_ENABLE=2
# vllm-ascend 配置
export VLLM_VERSION=0.11.0
export USE_OPTIMIZED_MODEL=0
export VLLM_USE_V1=1

# 进入训练目录
TRAIN_DIR="cosmos-reason1/examples/post_training"
if [ ! -d "$TRAIN_DIR" ]; then
    echo "ERROR：训练目录 $TRAIN_DIR 不存在"
    exit 1
fi
cd "$TRAIN_DIR" || exit 1

# 创建日志目录（放在脚本所在目录的 logs 文件夹下）
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

echo "日志文件: $LOG_FILE"

case "$MODE" in
    --sft)
        export DEVICES_POLICY=0,1,2,3,4,5,6,7
        echo "启动监督微调 (SFT) 训练，Policy训练分配NPU: $DEVICES_POLICY"
        cosmos-rl --config configs/sft.toml tools/dataset/cosmos_sft.py 2>&1 | tee "$LOG_FILE"
        ;;
    --rl)
        # PORT_POLICY和PORT_ROLLOUT不能一致
        export PORT_ROLLOUT=65104
        export PORT_POLICY=64104
        # policy训练所用NPU
        export DEVICES_POLICY=0,1,2,3
        # rollout 所用NPU
        export DEVICES_ROLLOUT=4,5,6,7
        echo "启动强化学习 (RL) 训练，Policy训练分配NPU: $DEVICES_POLICY，Rollout分配NPU: $DEVICES_ROLLOUT"
        cosmos-rl --config configs/rl.toml tools/dataset/cosmos_grpo.py 2>&1 | tee "$LOG_FILE"
        ;;
    *)
        echo "无效参数 '$MODE'，请使用 --sft 或 --rl"
        exit 1
        ;;
esac

# 解析 RL 日志（训练步骤行包含 "Step: 数字/数字"）
parse_rl_log() {
    local logfile=$1
    echo "===== RL 训练统计 ====="
    awk '
    $0 ~ /Step: [0-9]+\// {   # 只匹配带有斜杠的 Step 行
        match($0, /Step: ([0-9]+)\//, a)
        step = a[1]
        match($0, /Reward Mean: ([0-9.]+)/, a)
        reward = a[1]
        match($0, /Iteration time: ([0-9.]+)s/, a)
        itime = a[1]

        steps[step] = 1
        reward_mean[step] = reward
        iter_time[step] = itime
        if (step > max_step) max_step = step
    }
    END {
        if (max_step == 0) {
            print "未找到任何有效的 Step 信息"
            exit 1
        }
        printf "最后一个 Step: %d, Reward Mean: %s\n", max_step, reward_mean[max_step]
        sum = 0; cnt = 0
        for (i = 6; i <= 15; i++) {
            if (i in steps) {
                sum += iter_time[i]
                cnt++
            }
        }
        if (cnt > 0) {
            printf "平均 Iteration time: %.2f 秒\n", sum/cnt, cnt
        } else {
            print "迭代步数不足"
        }
    }' "$logfile"
}

# 解析 SFT 日志
parse_sft_log() {
    local logfile=$1
    echo "===== SFT 训练统计 ====="
    awk '
    $0 ~ /Step: [0-9]+\// {   # 排除 checkpoint 日志
        match($0, /Step: ([0-9]+)\//, a)
        step = a[1]
        match($0, /Loss: ([0-9.]+)/, a)
        loss = a[1]
        match($0, /Iteration time: ([0-9.]+)s/, a)
        itime = a[1]

        steps[step] = 1
        loss_val[step] = loss
        iter_time[step] = itime
        if (step > max_step) max_step = step
    }
    END {
        if (max_step == 0) {
            print "未找到任何有效的 Step 信息"
            exit 1
        }
        printf "最后一个 Step: %d, Loss: %s\n", max_step, loss_val[max_step]
        sum = 0; cnt = 0
        for (i = 51; i <= 100; i++) {
            if (i in steps) {
                sum += iter_time[i]
                cnt++
            }
        }
        if (cnt > 0) {
            printf "平均 Iteration time: %.2f 秒\n", sum/cnt
        } else {
            print "迭代步数不足"
        }
    }' "$logfile"
}

# 执行解析
case "$MODE" in
    --sft) parse_sft_log "$LOG_FILE" ;;
    --rl)  parse_rl_log "$LOG_FILE" ;;
esac

echo "日志文件已保存至: $LOG_FILE"