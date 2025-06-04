#!/bin/bash


# 数据集路径,保持为空,不需要修改
data_path=""

#网络名称,同目录名称,需要模型审视修改
Network="HiVT"

#训练batch_size,需要模型审视修改
batch_size=32

world_size=8
#参数校验，不需要修改
# 参数校验，data_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --batch_size* ]];then
        batch_size=`echo ${para#*=}`
    fi
done

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

###############指定训练脚本执行路径###############
# cd到与test文件夹同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
cur_path=`pwd`
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ];then
    test_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi

#创建DeviceID输出目录，不需要修改
ASCEND_DEVICE_ID=0
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/$ASCEND_DEVICE_ID
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi

#################启动训练脚本#################
cd HiVT
start_time=$(date +%s)
echo "start_time=$(date -d @${start_time} "+%Y-%m-%d %H:%M:%S")"
# 非平台场景时source 环境变量
check_etp_flag=$(env | grep etp_running_flag)
etp_flag=$(echo ${check_etp_flag#*=})
if [ x"${etp_flag}" != x"true" ]; then
  source ${test_path_dir}/env_npu.sh
fi

python train.py --root $data_path --embed_dim 64 --train_batch_size ${batch_size} --gpus ${world_size} > ${test_path_dir}/output/train_performance_8p_bs${batch_size}.log 2>&1

wait

##################获取训练数据################
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(($end_time - $start_time))

#结果打印，不需要修改
echo "------------------ Final result ------------------"

#获取性能数据，不需要修改
#单迭代训练时长，不需要修改
TrainingTime=$(grep -oP '\d+\.\d+(?=it/s, loss)' ${test_path_dir}/output/train_performance_8p_bs${batch_size}.log | tail -n 10 | grep -o "[0-9.]*" | awk '{sum += $1} END {print sum/NR}')
echo "TrainingTime : $TrainingTime"
#吞吐量
ActualFPS=$(awk BEGIN'{print ('$batch_size' * '$world_size') * '$TrainingTime'}')

#打印，不需要修改
echo "Final Performance images/sec : $ActualFPS"

#loss值，不需要修改
ActualLoss=$(grep -oP '(?<=loss=)\d+\.\d+' ${test_path_dir}/output/train_performance_8p_bs${batch_size}.log | awk 'END {print $NF}')

#minADE值
minADE=$(grep -oP '(?<=val_minADE=)\d+\.\d+' ${test_path_dir}/output/train_performance_8p_bs${batch_size}.log | awk 'END {print $NF}')


#minFDE值
minFDE=$(grep -oP '(?<=val_minFDE=)\d+\.\d+' ${test_path_dir}/output/train_performance_8p_bs${batch_size}.log | awk 'END {print $NF}')

#打印，不需要修改
echo "Final Train Loss : ${ActualLoss}"
echo "minADE : ${minADE}"
echo "minFDE : ${minFDE}"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
WORLD_SIZE=${world_size}
DeviceType=$(uname -m)
CaseName=${Network}_bs${BatchSize}_${WORLD_SIZE}'p'_'acc'

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >${test_path_dir}/output/${CaseName}.log
echo "RankSize = ${WORLD_SIZE}" >>${test_path_dir}/output/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>${test_path_dir}/output/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>${test_path_dir}/output/${CaseName}.log
echo "CaseName = ${CaseName}" >>${test_path_dir}/output/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >>${test_path_dir}/output/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >>${test_path_dir}/output/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >>${test_path_dir}/output/${CaseName}.log
echo "minADE = ${minADE}" >>${test_path_dir}/output/${CaseName}.log
echo "minFDE = ${minFDE}" >>${test_path_dir}/output/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>${test_path_dir}/output/${CaseName}.log