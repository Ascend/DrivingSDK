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
export HOST_CACHE_CAPACITY=20

export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE
export MULTI_STREAM_MEMORY_REUSE=1

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

gpu_num=$1
echo "number of gpus: "${gpu_num}

config=projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py
work_dir=work_dirs/sparse4dv3_temporal_r50_1x8_bs6_256x704

#使用断点
sed -i.bak '
/^[[:space:]]*main()[[:space:]]*$/ {
    i\
    from mx_driving.patcher.patcher import PatcherBuilder, Patch
    i\
    pb = PatcherBuilder().brake_at(1000)
    i\
    with pb.build():
    a\
        main()
    d
}' ./tools/train.py
#训练开始时间
start_time=$(date +%s)

if [ ${gpu_num} -gt 1 ]
then
    bash ./tools/dist_train.sh \
        ${config} \
        ${gpu_num} \
        --work-dir=${work_dir}
else
    python ./tools/train.py \
        ${config}
fi

#训练结束时间
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

# 将备份文件恢复为原文件
if [ -f "./tools/train.py.bak" ]; then
    mv -f "./tools/train.py.bak" "./tools/train.py"
else
    echo "Error: could not find the file train.py.bak, restoring failed."
fi

log_file=`find ${work_dir} -regex ".*\.log" | sort -r | head -n 1`
batch_size=6

#结果打印
echo "------------------ Final result ------------------"
#输出性能FPS
time_per_iter=$(grep -E 'mmdet - INFO - (Iter|Epoch)' "${log_file}" | awk -F " time: " '! (/Iter \[1\//) {print $NF}' | awk -F "," '{print $1}' | awk '{ sum += $0; n++ } END { if (n > 0) printf "%.2f\n", int((sum / n) * 100) / 100 }' )
FPS=`awk 'BEGIN{printf "%.2f\n", '${batch_size}' * '${gpu_num}' / '${time_per_iter}'}'`
#打印
echo "Step time per iteration sec : $time_per_iter"
echo "Final Performance images/sec : $FPS"
echo "E2E Training Duration sec : $e2e_time"