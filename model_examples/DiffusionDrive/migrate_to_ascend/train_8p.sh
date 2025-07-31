#!/bin/sh

# 创建输出目录（mmcv框架会输出日志到work_dir下，此处的输出目录为合并stdout和stderr的重定向文件路径）
OUTPUT_PATH=./migrate_to_ascend/output/
# 清理旧目录
if [ -d ${OUTPUT_PATH} ]; then
  rm -rf ${OUTPUT_PATH}
fi
mkdir -p ${OUTPUT_PATH}

# Default路径
CONFIG=projects/configs/diffusiondrive_configs/diffusiondrive_small_stage2.py

GLOBAL_BATCH_SIZE=48

# 设置NPU卡数
RANK_SIZE=8


# 传参：
# 如果设置了--performance，仅验性能（仅需少量epochs，建议同步设置--epoch为小一点的数字），不验精度。
# 不设置时默认为验精度模式，将运行config文件里设置的全量epochs，较为耗时
# 如果设置了--config，则可以指定config文件路径覆盖default路径
PERFORMANCE_MODE=0
TEE_TO_STDOUT=0
for para in $*
do    
    if [[ $para == --performance ]]; then
        PERFORMANCE_MODE=1
    fi

    if [[ $para == --config=* ]]; then
        CONFIG_FILE=`echo ${para#*=}`
    fi
done



##################################################################
# 配置环境变量
echo "[INFO] Start setting ENV VAR"

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

#减少显存占用
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
##################################################################



# 动态生成并检测本地可用高位端口（10000-59151），直至找到未被占用的空闲端口
while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

MPORT=${PORT:-28651}


# 训练
#################################################
start_time=$(date +%s)
echo "start_time=$(date -d @${start_time} "+%Y-%m-%d %H:%M:%S")"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

# 验精度
if [[ ${PERFORMANCE_MODE} == 0 ]]; then
    nohup python -m torch.distributed.run \
        --nproc_per_node=$RANK_SIZE \
        --master_port=$MPORT \
        migrate_to_ascend/train.py $CONFIG \
        --launcher pytorch \
        --deterministic > ${OUTPUT_PATH}/train_8p_full.log 2>&1 &
    wait

# 验性能
else
    nohup python -m torch.distributed.run \
        --nproc_per_node=$RANK_SIZE \
        --master_port=$MPORT \
        migrate_to_ascend/train.py $CONFIG \
        --launcher pytorch \
        --deterministic --performance > ${OUTPUT_PATH}/train_8p_performance.log 2>&1 &
    wait
fi


end_time=$(date +%s)
echo "end_time=$(date -d @${end_time} "+%Y-%m-%d %H:%M:%S")"
e2e_time=$(( $end_time - $start_time ))
#################################################




log_file=`find work_dirs/diffusiondrive_small_stage2 -regex ".*\.log" | sort -r | head -n 1`
echo "Log file Path: ${log_file}"

per_step_time=$(grep -o ", time: [0-9.]*" ${log_file} | tail -n 10 | grep -o "[0-9.]*" | awk '{sum += $1} END {print sum/NR}')
echo "avg_per_step_time : ${per_step_time}"

avg_fps=`awk 'BEGIN{printf "%.3f\n", '$GLOBAL_BATCH_SIZE'/'$per_step_time'}'`

# 输出结果
echo "[INFO] Final Result"
echo " - End to End Time : ${e2e_time}s"
echo " - Time avg per batch :  ${per_step_time}s"
echo " - Final Performance images/sec :  ${avg_fps}"

# 验精度
if [[ ${PERFORMANCE_MODE} == 0 ]]; then
    L2=`grep "val" ${log_file}  | awk -F "L2: " '{print $2}' | awk 'END {print}'`
    echo " - L2 : ${L2}"
fi