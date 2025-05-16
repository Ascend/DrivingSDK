# 配置环境变量
msnpureport -g error -d 0
msnpureport -g error -d 1
msnpureport -g error -d 2
msnpureport -g error -d 3
msnpureport -g error -d 4
msnpureport -g error -d 5
msnpureport -g error -d 6
msnpureport -g error -d 7
#设置Device侧日志等级为error
msnpureport -g error
#关闭Device侧Event日志
msnpureport -e disable

#将Host日志输出到串口,0-关闭/1-开启
export ASCEND_SLOG_PRINT_TO_STDOUT=0
#设置默认日志级别,0-debug/1-info/2-warning/3-error
export ASCEND_GLOBAL_LOG_LEVEL=3
#设置Host侧Event日志开启标志,0-关闭/1-开启
export ASCEND_GLOBAL_EVENT_ENABLE=0

#设置是否开启taskque,0-关闭/1-开启/2-优化
export TASK_QUEUE_ENABLE=2
#设置是否开启均匀绑核,0-关闭/1-开启粗粒度绑核/2-开启细粒度绑核
export CPU_AFFINITY_CONF=1

batch_node_size=8

cur_path=`pwd`
output_path_dir=${cur_path}/log

if [ -d ${output_path_dir} ]; then
  rm -rf ${output_path_dir}
fi
mkdir -p ${output_path_dir}

start_time=$(date +%s)

#训练
python tools/train.py --config-file configs/nuscenes/semseg-pt-v3m1-0-base.py --num-gpus 8 --options save_path=output > ${output_path_dir}/train.log 2>&1 &

wait
#日志
log_file=`find ${output_path_dir} -regex ".*\.log" | sort -r | head -n 1`

end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

# 从 log 中获取性能
avg_time=`grep -oP 'Batch \d+\.\d+ \(\K\d+\.\d+' ${log_file} | tail -n 100 | awk '{sum+=$1; count++} END {if(count>0) print sum/count}'`
FPS=`awk 'BEGIN{printf "%.3f\n", '$batch_node_size'/'$avg_time'}'`

mIou=`grep -oP 'Best validation mIoU updated to: \K\d+\.\d+' ${log_file} | head -n 1`

# 输出结果
echo "[INFO] Final Result"
echo " - Time avg per batch :  ${avg_time}s"
echo " - Final Performance images/sec :  ${FPS}"
echo " - mIou : ${mIou}"