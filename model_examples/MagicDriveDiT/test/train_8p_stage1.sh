#!/usr/bin/env bash
################基础配置参数，需要模型审视修改##################
# 网络名称，同目录名称
Network="MagicDriveDiT"

# -------------------------------------------------- #
# 训练卡数，开启sp需要至少4张卡
if [ $# -lt 1 ]; then
    echo "Usage: $0 <GPUS> [NUM_WORKERS] [prefetch_factor] [sp_size] [epochs]"
    exit 1
fi

GPUS=$1
NUM_WORKERS=${2:-2}
prefetch_factor=${3:-2}
sp_size=${4:-4}
epochs=${5:-450}
# -------------------------------------------------- #
GPUS_PER_NODE=$(($GPUS < 8 ? $GPUS : 8))

###############指定训练脚本执行路径###############
# cd到与test文件夹同层级目录下执行脚本
cur_path=$(pwd)
cur_path_last_dirname=${cur_path##*/}
if [ "${cur_path_last_dirname}" = "test" ]; then
    test_path_dir=$cur_path
    cd ..
    cur_path=$(pwd)
else
    test_path_dir=${cur_path}/test
fi

# 创建输出目录
output_dir="${test_path_dir}/output/${ASCEND_DEVICE_ID}"
mkdir -p "$output_dir"

# 非平台场景时source环境变量
check_etp_flag=$(env | grep etp_running_flag)
etp_flag=${check_etp_flag#*=}
if [ "${etp_flag}" != "true" ]; then
    source "${test_path_dir}/env_npu.sh"
fi

# 训练开始时间
start_time=$(date +%s)

# 同步执行训练命令
torchrun --nproc-per-node=$GPUS_PER_NODE \
         --nnode=1 \
         --node_rank=0 \
         scripts/train_magicdrive.py \
         configs/magicdrive/train/stage1_1x224x400_stdit3_CogVAE_noTemp_xCE_wSST_bs4_lr8e-5.py \
         --cfg-options \
             num_workers=$NUM_WORKERS \
             prefetch_factor=$prefetch_factor \
             sp_size=$sp_size \
             epochs=$epochs 2>&1 | tee "${output_dir}/train_${ASCEND_DEVICE_ID}.log"

# 训练结束时间
end_time=$(date +%s)
e2e_time=$((end_time - start_time))

# 结果处理
echo "------------------ Final result ------------------"
if [ -f "${output_dir}/train_${ASCEND_DEVICE_ID}.log" ]; then
    avg_time=$(grep -E "[0-9]+\.[0-9]+s/it.*step=[0-9]+" "${output_dir}/train_${ASCEND_DEVICE_ID}.log" | 
      awk '
      {
          # 提取时间(s/it前的浮点数)
          if (match($0, /([0-9]+\.[0-9]+)s\/it/, time_match)) {
              current_time = time_match[1]
          }
          # 提取step(step=后的整数)
          if (match($0, /step=([0-9]+)/, step_match)) {
              current_step = step_match[1]
          }
          # 统计20-100步
          if (current_step >= 20 && current_step <= 100) {
              sum += current_time
              count++
              print(current_time)
          }
      }
      END {
          printf "%.2f", (count > 0 ? sum / count : 0)
          
      }')  # 保留两位小数

    echo "Average iteration time (step20-100): ${avg_time:-0}s"
    echo "Total training time: ${e2e_time}s"
else
    echo "Log file not found!"
fi