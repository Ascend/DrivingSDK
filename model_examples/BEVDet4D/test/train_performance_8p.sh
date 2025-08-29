#!/bin/bash

#当前路径
cur_path=`pwd`

#集合通信参数
export RANK_SIZE=8
export JOB_ID=10087
RANK_ID_START=0

#基础参数
batch_size=8
#训练step
max_epochs=24

# 帮助信息
if [[ $1 == --help || $1 == -h ]];then
    echo "usage: ./train_8p.sh <args>"
    echo " "
    echo "parameter explain:
    --py_config               train config
    --test              switch to test mode when != 0
    --work_dir                 set output dir for training
    --fp16                     enable fp16 mode (set BEVDET4D_FP16=1)
    -h/--help                  show help message
    "
    exit 1
fi

# 参数校验
# 初始化fp16标志
fp16=0

for para in $*
do
    if [[ $para == --py_config* ]];then
        py_config=`echo ${para#*=}`
    elif [[ $para == --test* ]];then
        test=`echo ${para#*=}`
    elif [[ $para == --work_dir* ]];then
        work_dir=`echo ${para#*=}`
    elif [[ $para == --fp16 ]];then
        fp16=1
    fi
done

#校验是否传入py_config
if [[ $py_config == "" ]];then
    echo "[Error] para \"py_config\" must be config"
    exit 1
fi

#配置名称
config_name=`echo $py_config | awk -F "/" '{print $NF}' | awk -F "." '{print $1}'`
#网络名称，同配置名称
Network=$config_name

if [[ $work_dir == "" ]];then
    work_dir="output/train_8p/$config_name"
else
    work_dir="${work_dir}/train_8p/$config_name"
fi

test_path_dir=$cur_path
ASCEND_DEVICE_ID=0

if [ ! -d ${test_path_dir}/output ];then
    mkdir ${test_path_dir}/output
fi
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID/ckpt
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID/ckpt
fi

#训练开始时间
start_time=$(date +%s)
# 非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/test/env_npu.sh
fi

#设置环境变量
echo "Device ID: $ASCEND_DEVICE_ID"
export RANK_ID=$RANK_ID
export WORLD_SIZE=8
export BEVDET4D_PERFORMANCE_FLAG=1
if [[ $fp16 -eq 1 ]];then
    export BEVDET4D_FP16=1
    echo "Enabled fp16 mode (BEVDET4D_FP16=1)"
fi

bash ./tools/dist_train.sh ${py_config} ${WORLD_SIZE} \
--work-dir ${work_dir} \
--cfg-options runner.max_epochs=$max_epochs data.samples_per_gpu=$batch_size > train_performance.log 2>&1

#训练结束时间
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

log_file=`find ${work_dir} -regex ".*\.log" | sort -r | head -n 1`

#结果打印
echo "------------------ Final result ------------------"
#输出性能FPS
TrainingTime=`grep -a 'Epoch '  ${log_file}|awk -F " time: " '!/Epoch \[1\]\[1/ {print $NF}'|awk -F " " '{print $1}' | awk '{ sum += $0; n++} END { if (n > 0) print sum / n;}'`
FPS=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'* '${WORLD_SIZE}'/'${TrainingTime}'}'`
#打印
echo "Final Performance images/sec : $FPS"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息
DeviceType=`uname -m`
CaseName=${Network}_bs${batch_size}_${RANK_SIZE}'p'

##获取性能数据
#吞吐量
ActualFPS=${FPS}

#关键信息打印到${CaseName}.log中
echo "Network = ${Network}" >  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${batch_size}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
