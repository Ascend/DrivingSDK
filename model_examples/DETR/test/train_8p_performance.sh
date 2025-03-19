#!/bin/bash
###############基本配置###############
# 模型配置
export RANK_SIZE=8
batch_size=8
epochs=20
data_path=""

# 参数校验，data_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
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
    echo "[DETR] [Error] para \"data_path\" must be confing"
    exit 1
fi

# 训练用例信息
network="DETR"
device_type=`uname -m`
case_name=${network}_${RANK_SIZE}p_bs${batch_size}_epochs${epochs}
echo "[DETR] case_name = ${case_name}"

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
#设置是否开启均匀绑核,0-关闭/1-开启粗粒度绑核/2-开启细粒度绑核
export CPU_AFFINITY_CONF=1
#HCCL白名单开关,配置在使用HCCL时是否关闭通信白名单。1-关闭/0-开启
export HCCL_WHITELIST_DISABLE=1


###############指定训练脚本执行路径###############
# cd到与test文件夹同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
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
mkdir -p ${output_path_dir}


###############开始训练###############
# 训练开始时间
start_time=$(date +%s)
echo "[DETR] start_time=$(date -d @${start_time} "+%Y-%m-%d %H:%M:%S")"

nohup torchrun --standalone --nnodes=1 \
    --nproc_per_node=${RANK_SIZE} \
    main.py \
    --batch_size=${batch_size} \
    --epochs=${epochs} \
    --coco_path=${data_path} \
    --output_dir=${output_path_dir} > ${output_path_dir}/train_${RANK_SIZE}p_bs${batch_size}_epochs${epochs}.log 2>&1 &

wait

# 训练结束时间
end_time=$(date +%s)
echo "[DETR] end_time=$(date -d @${end_time} "+%Y-%m-%d %H:%M:%S")"
e2e_time=$(( $end_time - $start_time ))

# 输出性能
avg_time=`grep -P 'Epoch: \[\d+\] Total time' ${output_path_dir}/train_${RANK_SIZE}p_bs${batch_size}_epochs${epochs}.log | tail -n 5 | grep -oP '(?<=\()[0-9]+\.[0-9]+' | awk '{sum+=$1; count++} END {if(count>0) print sum/count}'`
avg_fps=`awk 'BEGIN{printf "%.3f\n", '$batch_size'*'${RANK_SIZE}'/'$avg_time'}'`

# 输出训练精度
mAP=`grep "Average Precision.* IoU=0.50\:0.95.* all" ${output_path_dir}/train_${RANK_SIZE}p_bs${batch_size}_epochs${epochs}.log |awk -F "=" '{print $NF}'|awk 'END {print}'`

# 结果打印
echo "------------------ Final result ------------------"
echo "[DETR] E2E Training Duration sec : ${e2e_time}"
echo "[DETR] Final Performance images/sec : ${avg_fps}"
echo "[DETR] Final mAP(IoU=0.50:0.95) : ${mAP}"

# 将关键信息打印到 ${CASE_NAME}.log 中
echo "Network = ${network}" > ${output_path_dir}/${case_name}.log
echo "DeviceType = ${device_type}" >> ${output_path_dir}/${case_name}.log
echo "RankSize = ${RANK_SIZE}" >> ${output_path_dir}/${case_name}.log
echo "BatchSize = ${batch_size}" >> ${output_path_dir}/${case_name}.log
echo "CaseName = ${case_name}" >> ${output_path_dir}/${case_name}.log
echo "E2ETrainingTime = ${e2e_time}" >> ${output_path_dir}/${case_name}.log
echo "TrainingTime = ${avg_time}" >> ${output_path_dir}/${case_name}.log
echo "averageFPS = ${avg_fps}" >> ${output_path_dir}/${case_name}.log
echo "mAP = ${mAP}" >> ${output_path_dir}/${case_name}.log
