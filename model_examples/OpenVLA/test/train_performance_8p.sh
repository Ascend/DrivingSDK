export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=2
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export COMBINED_ENABLE=1

export DYNAMIC_OP="ADD#MUL"
export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE
export MULTI_STREAM_MEMORY_REUSE=1

num_npu=8
batch_size=32

PORT=${PORT:-28666}
cur_path=$(pwd)
CURRENT_TIME=$(date +"%Y-%m-%d_%H-%M-%S")

weights_path=$1
dataset_path=$2
log_dir=$cur_path/logs

sed -i '89s/self.max_steps = self.vla.max_steps/self.max_steps =260/' vla-scripts/train.py

torchrun --standalone --nnodes 1 --nproc-per-node $num_npu --master_port=$PORT \
        vla-scripts/train.py --pretrained_checkpoint "$weights_path/openvla-7b-prismatic/checkpoints/step-295000-epoch-40-loss=0.2200.pt" \
        --vla.type prism-dinosiglip-224px+mx-bridge \
        --data_root_dir "$dataset_path" \
        --run_root_dir "work_dir" \
        --image_aug False \
        --save_interval 10000 \
        --is_resume False \
        --trackers ["jsonl"] \
        2>&1 | tee $log_dir/train_${CURRENT_TIME}.log

sed -i '89s/self.max_steps =260/self.max_steps = self.vla.max_steps/' vla-scripts/train.py

log_file=`find ${log_dir} -regex ".*\.log" | sort -r | head -n 1`

step_time=$(
    sed 's/\r/\n/g' "$log_file" | awk -v start=200 -v end=250 '
        BEGIN { count = 0; sum_time = 0 }
        /Global Step [0-9]+ =>>/ && /[0-9]+\.[0-9]+s\/it/ {
            if (match($0, /Global Step ([0-9]+) =>>/, step_arr)) {
                step = step_arr[1] + 0
                if (match($0, /([0-9]+\.[0-9]+)s\/it/, time_arr)) {
                    time = time_arr[1] + 0
                    if (step >= start && step <= end && !(step in steps)) {
                        steps[step] = 1
                        sum_time += time
                        count++
                    }
                }
            }
        }
        END {
            if (count > 0) {
                printf "%.3f\n", sum_time / count  # 仅输出数值（供变量捕获）
            } else {
                printf "0\n"  # 无数据时输出默认值
            }
        }
    '
)

FPS=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'* '${num_npu}'/'${step_time}'}'`

echo "step time: $step_time"
echo "FPS: $FPS"
