#!/bin/sh

CONFIG=projects/configs/diffusiondrive_configs/diffusiondrive_small_stage2.py
GLOBAL_BATCH_SIZE=48
RANK_SIZE=8
#export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


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
#减少显存占用
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"



start_time=$(date +%s)
echo "start_time=$(date -d @${start_time} "+%Y-%m-%d %H:%M:%S")"

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

GPUS=$RANK_SIZE
MPORT=${PORT:-28651}


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -W ignore -m torch.distributed.run \
    --nproc_per_node=$GPUS \
    --master_port=$MPORT \
    migrate_to_ascend/train.py $CONFIG \
    --launcher pytorch \
    --deterministic --performance

end_time=$(date +%s)
echo "end_time=$(date -d @${end_time} "+%Y-%m-%d %H:%M:%S")"
e2e_time=$(( $end_time - $start_time ))

log_file=`find work_dirs/diffusiondrive_small_stage2 -regex ".*\.log" | sort -r | head -n 1`

avg_time=`grep "Iter" ${log_file} | tail -n 10 | awk -F "time: " '{print $2}' | awk '{sum+=$1; count++} END {if(count>0) print sum/count}'`
echo "avg_time : ${avg_time}"

avg_fps=`awk 'BEGIN{printf "%.3f\n", '$GLOBAL_BATCH_SIZE'/'$avg_time'}'`

# 输出结果
echo "[INFO] Final Result"
echo " - End to End Time : ${e2e_time}s"
echo " - Time avg per batch :  ${avg_time}s"
echo " - Final Performance images/sec :  ${avg_fps}"