#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`

#集合通信参数,不需要修改
export RANK_SIZE=1

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="MultiPath++"
#训练batch_size
batch_size=128

num_data=$(find $(sed -n '6p' ./configs/final_RoP_Cov_Single.yaml | awk -F': ' '{print $2}' | tr -d '"') -type f | wc -l)
num_step=$((($num_data + $batch_size - 1) / $batch_size))

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

#################创建日志输出目录，不需要修改#################
ASCEND_DEVICE_ID=0
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi

# 非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/env_npu.sh
fi

KERNEL_NUM=$(($(nproc)/8))
PID_START=0
PID_END=$((PID_START + KERNEL_NUM - 1))

sed -i '17 s/120/1/'  ./configs/final_RoP_Cov_Single.yaml

#训练开始时间，不需要修改
start_time=$(date +%s)

nohup taskset -c $PID_START-$PID_END python3 train.py 2>&1 | tee ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log &

wait

sed -i '17 s/1/120/'  ./configs/final_RoP_Cov_Single.yaml

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

##################获取训练数据################
#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能，需要模型审视修改
avg_training_time=`grep -a "$num_step/$num_step" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log |awk -F '[' '{print $2}'|awk -F '<' '{print $1}'|awk '{split($0,a,":");b+=a[1]*60+a[2];} END{printf "%d\n",int(b)}'`
training_time_per_step=$(awk "BEGIN {printf \"%.3f\", $avg_training_time / $num_step}")
#打印，不需要修改
echo "Final Training Time per epoch : $avg_training_time"
echo "Final Training Time per step : $training_time_per_step"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

#获取性能数据，不需要修改
#吞吐量
AvgTrainingTime=${avg_training_time}
TrainingTimePerStep=${training_time_per_step}

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "AvgTrainingTime = ${AvgTrainingTime}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTimePerStep = ${TrainingTimePerStep}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log

