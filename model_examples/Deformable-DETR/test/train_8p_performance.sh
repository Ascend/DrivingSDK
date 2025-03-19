#!/bin/bash

node_size=8
batch_size=8
epochs=1
data_path=""

for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
    if [[ $para == --epochs* ]];then
        epochs=`echo ${para#*=}`
    fi
done

if [[ "$data_path" == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

echo "[INFO] Start setting ENV VAR"

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
#设置是否开启taskque,0-关闭/1-开启/2-流水优化
export TASK_QUEUE_ENABLE=2
#设置aclnn cache，使能host侧算子下发缓存机制
export DISABLE_L2_CACHE=0
#设置是否开启均匀绑核,0-关闭/1-开启粗粒度绑核/2-开启细粒度绑核
export CPU_AFFINITY_CONF=1
#HCCL白名单开关,配置在使用HCCL时是否关闭通信白名单。1-关闭/0-开启
export HCCL_WHITELIST_DISABLE=1
#启用可扩展内存段分配策略
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

echo "[INFO] Finish setting ENV VAR"

cur_path=`pwd`
cur_path_last_diename=${cur_path##*/}

if [ x"${cur_path_last_diename}" == x"test" ];then
    test_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi
output_path_dir=${test_path_dir}/output
if [ -d ${output_path_dir} ]; then
  rm -rf ${output_path_dir}
fi
mkdir -p ${output_path_dir}

# 训练
start_time=$(date +%s)
echo "start_time=$(date -d @${start_time} "+%Y-%m-%d %H:%M:%S")"

# train
nohup torchrun --standalone --nnodes=1 --nproc_per_node=${node_size} main.py \
    --output_dir=${output_path_dir} \
    --batch_size=${batch_size} \
    --epochs=${epochs} \
    --coco_path=${data_path} > ${output_path_dir}/train_8p_performance.log 2>&1 &

wait
end_time=$(date +%s)
echo "end_time=$(date -d @${end_time} "+%Y-%m-%d %H:%M:%S")"
e2e_time=$(( $end_time - $start_time ))

# 从 log 中获取性能
avg_time=`grep "Epoch: .* Total time"  ${output_path_dir}/train_8p_performance.log | tail -n 5 | grep -oP "[0-9]+\.[0-9]+" | awk '{sum+=$1; count++} END {if(count>0) print sum/count}'`
# 从 log 中获取精度
mAP=`grep "Average Precision.* IoU=0\.50\:0\.95.* all" ${output_path_dir}/train_8p_performance.log |awk -F "=" '{print $NF}'|awk 'END {print}'`
avg_fps=`awk 'BEGIN{printf "%.3f\n", '$batch_size'*'${node_size}'/'$avg_time'}'`

# 输出结果
echo "[INFO] Final Result"
echo " - End to End Time : ${e2e_time}s"
echo " - Final Performance images/sec :  ${avg_fps}"
echo " - Final mAP(IoU=0.50:0.95) : ${mAP}"