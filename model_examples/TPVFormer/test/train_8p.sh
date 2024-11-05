#!/bin/bash

#当前路径
cur_path=`pwd`
# 指定训练所使用的npu device卡id
device_id=0

#集合通信参数
export RANK_SIZE=8
export JOB_ID=10087
RANK_ID_START=0

performance=0

#设置默认日志级别
#export ASCEND_GLOBAL_LOG_LEVEL=3

#基础参数
batch_size=1
#训练step
max_epochs=24

# 帮助信息
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_8p.sh <args>"
    echo " "
    echo "parameter explain:
    --py_config               train config
    --performance              switch to performance mode when != 0
    --work_dir                 set output dir for training
    -h/--help		             show help message
    "
    exit 1
fi

#参数校验
for para in $*
do
    if [[ $para == --py_config* ]];then
        py_config=`echo ${para#*=}`
    elif [[ $para == --performance* ]];then
        performance=`echo ${para#*=}`
    elif [[ $para == --work_dir* ]];then
        work_dir=`echo ${para#*=}`
    fi
done

if (($performance!=0)); then
    max_epochs=1
fi

#校验是否传入py_config
if [[ $py_config == "" ]];then
    echo "[Error] para \"py_config\" must be config"
    exit 1
fi

#配置名称
config_name=`echo $py_config | awk -F "/" '{print $NF}' | awk -F "." '{print $1}'`
#网络名称，同配置名称
Network=$config_name

# 校验是否指定了device_id,分动态分配device_id与手动指定device_id
if [ $ASCEND_DEVICE_ID ];then
    echo "device id is ${ASCEND_DEVICE_ID}"
elif [ ${device_id} ];then
    export ASCEND_DEVICE_ID=${device_id}
    echo "device id is ${ASCEND_DEVICE_ID}"
else
    "[Error] device id must be config"
    exit 1
fi

work_dir="output/train_8p/$config_name"
test_path_dir=$cur_path
ASCEND_DEVICE_ID=$device_id

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
export WORLD_SIZE=1

python train.py \
--py-config ${py_config} \
--work-dir ${work_dir} \
--max-epochs ${max_epochs}


#训练结束时间
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

log_file=`find ${work_dir} -regex ".*\.log" | sort -r | head -n 1`

#结果打印
echo "------------------ Final result ------------------"
#输出性能FPS
FPS=`grep -a '\[TRAIN\] Epoch '  ${log_file}|awk -F " time: " '!/Iter     0/ {print $NF}'|awk -F " " '{print $1}' | awk '{ sum += $0; n++} END { if (n > 0) print sum / n;}'`
FPS=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*8/'${FPS}'}'`
#打印
echo "Final Performance images/sec : $FPS"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息
DeviceType=`uname -m`
CaseName=${Network}_bs${batch_size}_${RANK_SIZE}'p'_'miou'

##获取性能数据
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000*8/'${FPS}'}'`
echo "TrainingTime for step(ms) : $TrainingTime"

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中
grep "\[TRAIN\] Epoch " ${log_file}|awk -F "Loss: " '!/Iter     0/ {print $NF}' | awk -F "," '{print $1}' >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值
ActualLoss=`awk 'END {print}' ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中
echo "Network = ${Network}" >  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${batch_size}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log

# 性能任务控制
if (($performance==0));then
  #输出训练精度
  train_accuracy=`grep -a 'best val miou pts is' ${log_file}|awk 'END {print}'|awk -F "best val miou pts is" '{print $NF}'|awk -F " " '{print $1}'`
  #打印
  echo "Final Train Accuracy : ${train_accuracy}"
  echo "miou = ${train_accuracy}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
fi
