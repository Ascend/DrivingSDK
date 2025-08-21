export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=2
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export COMBINED_ENABLE=1

export DYNAMIC_OP="ADD#MUL"
export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE
export MULTI_STREAM_MEMORY_REUSE=1


PORT=${PORT:-28666}
cur_path=$(pwd)
CURRENT_TIME=$(date +"%Y-%m-%d_%H-%M-%S")

weights_path=$1
dataset_path=$2
log_dir=$cur_path/logs

sed -i '89s/self.max_steps = self.vla.max_steps/self.max_steps =1010/' vla-scripts/train.py

torchrun --standalone --nnodes 1 --nproc-per-node 8 --master_port=$PORT \
        vla-scripts/train.py --pretrained_checkpoint "$weights_path/openvla-7b-prismatic/checkpoints/step-295000-epoch-40-loss=0.2200.pt" \
        --vla.type prism-dinosiglip-224px+mx-bridge \
        --data_root_dir "$dataset_path" \
        --run_root_dir "work_dir" \
        --image_aug False \
        --save_interval 10000 \
        --is_resume False \
        --trackers ["jsonl"] \
        2>&1 | tee $log_dir/train_${CURRENT_TIME}.log

sed -i '89s/self.max_steps =1010/self.max_steps = self.vla.max_steps/' vla-scripts/train.py

log_file=`find ${log_dir} -regex ".*\.log" | sort -r | head -n 1`

average_loss=$(
    sed 's/\r/\n/g' "$log_file" | awk -v start=500 -v end=1000 '
        BEGIN {
            count = 0
            sum = 0
        }
        /Global Step [0-9]+ =>>/ && /Loss :: [0-9]+\.[0-9]+:/ {
            if (match($0, /Global Step ([0-9]+) =>>/, step_arr)) {
                step = step_arr[1] + 0
                if (match($0, /Loss :: ([0-9]+\.[0-9]+):/, loss_arr)) {
                    loss = loss_arr[1] + 0
                    if (step >= start && step <= end && !(step in steps)) {
                        steps[step] = 1
                        sum += loss
                        count++
                    }
                }
            }
        }
        END {
            if (count > 0) {
                printf "%.4f\n", sum / count
            } else {
                printf "0\n"
            }
        }
    '
)

echo "Average Loss: $average_loss" 