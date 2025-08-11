#将Host日志输出到串口,0-关闭/1-开启
export ASCEND_SLOG_PRINT_TO_STDOUT=0
#设置默认日志级别,0-debug/1-info/2-warning/3-error
export ASCEND_GLOBAL_LOG_LEVEL=3
#设置Event日志开启标志,0-关闭/1-开启
export ASCEND_GLOBAL_EVENT_ENABLE=0
#设置是否开启taskque,0-关闭/1-开启
export TASK_QUEUE_ENABLE=2
#设置是否开启PTCopy,0-关闭/1-开启
export PTCOPY_ENABLE=1
#设置是否开启combined标志,0-关闭/1-开启
export COMBINED_ENABLE=1
#设置特殊场景是否需要重新编译,不需要修改
export DYNAMIC_OP="ADD#MUL"
#HCCL白名单开关,1-关闭/0-开启
export HCCL_WHITELIST_DISABLE=1
#export HCCL_IF_IP=$(hostname -I |awk '{print $1}')
#开启绑核
export CPU_AFFINITY_CONF=1

export ACLNN_CACHE_LIMIT=100000

export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE
export MULTI_STREAM_MEMORY_REUSE=1

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

gpu_num=$1
echo "number of gpus: "${gpu_num}

batch_size=6
global_batch_size=$((gpu_num * batch_size))

config=projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py
work_dir=work_dirs/sparse4dv3_temporal_r50_1x8_bs6_256x704

sed -i "s/total_batch_size = 48/total_batch_size = ${global_batch_size}/" "$config"
sed -i "s/num_gpus = 8/num_gpus = ${gpu_num}/" "$config"

#训练开始时间
start_time=$(date +%s)

if [ ${gpu_num} -gt 1 ]
then
    bash ./tools/dist_train.sh \
        ${config} \
        ${gpu_num} \
        --work-dir=${work_dir} > train_full.log 2>&1
else
    python ./tools/train.py \
        ${config} > train_full_1p.log 2>&1
fi

#训练结束时间
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

sed -i "s/total_batch_size = ${global_batch_size}/total_batch_size = 48/" "$config"
sed -i "s/num_gpus = ${gpu_num}/num_gpus = 8/" "$config"

#evaluation
bash local_test.sh sparse4dv3_temporal_r50_1x8_bs6_256x704 ${gpu_num} \
    work_dirs/sparse4dv3_temporal_r50_1x8_bs6_256x704/latest.pth >work_dirs/val.log

log_file=`find ${work_dir} -regex ".*\.log" | sort -r | head -n 1`
batch_size=6
result=$(grep 'mAP:' work_dirs/val.log | awk '{print $2}')

#结果打印
echo "------------------ Final result ------------------"
#输出性能FPS
time_per_iter=$(grep -E 'mmdet - INFO - (Iter|Epoch)' "${log_file}" | awk -F " time: " '!/Iter \[1\// {print $NF}' | awk -F "," '{print $1}' | awk '{ if ($0 < 1) { sum += $0; n++ } } END { if (n > 0) printf "%.2f\n", sum/n }')
FPS=`awk 'BEGIN{printf "%.2f\n", '${batch_size}' * '${gpu_num}' / '${time_per_iter}'}'`
#打印
echo "Step time per iteration sec : $time_per_iter"
echo "Final Performance images/sec : $FPS"
echo "E2E Training Duration sec : $e2e_time"
echo "Accuracy mAP: $result"