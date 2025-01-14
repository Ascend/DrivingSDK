#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`

#集合通信参数,不需要修改
export RANK_SIZE=8
export JOB_ID=10087
RANK_ID_START=0

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="mask2former_swin"
batch_size=2
num_workers=2
max_iters=1000
val_interval=1000


# 帮助信息，不需要修改
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_performance_8p.shh <args>"
    echo " "
    echo "parameter explain:
    --batch_size               batch size for train dataloader
    --num_workers              num workers for train dataloader
    -h/--help		           show help message
    "
    exit 1
fi

#参数校验，不需要修改
for para in $*
do
    if [[ $para == --batch_size* ]];then
        batch_size=`echo ${para#*=}`
    elif [[ $para == --num_workers* ]];then
        num_workers=`echo ${para#*=}`
    fi
done

# 创建输出目录
test_path_dir=${cur_path}/test
if [ ! -d ${test_path_dir}/output/ckpt ];then
    mkdir -p ${test_path_dir}/output/ckpt
fi

# source 环境变量
source ${test_path_dir}/env_npu.sh

work_dir=${test_path_dir}/output/ckpt
log_file=${test_path_dir}/output/train_performance_${RANK_SIZE}p.log

#训练开始时间，不需要修改
start_time=$(date +%s)

cd mmsegmentation
bash ./tools/dist_train.sh ./configs/mask2former/mask2former_swin-b-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024.py ${RANK_SIZE} \
    --work-dir=$work_dir \
    --cfg-options train_cfg.max_iters=$max_iters \
    --cfg-options train_cfg.val_interval=$val_interval \
    --cfg-options train_dataloader.batch_size=$batch_size \
    --cfg-options train_dataloader.num_workers=$num_workers \
    > ${log_file} 2>&1 &
cd ..
wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=`grep -a 'Iter(train)'  ${log_file}|awk -F " time: " '{print $NF}'|awk -F " " '{print $1}' | awk '{ sum += $0; n++} END { if (n > 0) print sum / n;}'`
FPS=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*'${RANK_SIZE}'/'${FPS}'}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=`grep -a 'mIoU:' ${log_file}|awk 'END {print}'|awk -F "mIoU:" '{print $NF}'|awk -F " " '{print $1}'`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
DeviceType=`uname -m`
CaseName=${Network}_${RANK_SIZE}'p'_'miou'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*'${RANK_SIZE}'/'${FPS}'}'`
echo "TrainingTime for step: $TrainingTime"

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "Iter(train)" ${log_file}|awk -F "loss:" '{print $NF}' | awk -F " " '{print $1}' > ${test_path_dir}/output/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' ${test_path_dir}/output/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >  ${test_path_dir}/output/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >>  ${test_path_dir}/output/${CaseName}.log
echo "BatchSize = ${batch_size}" >>  ${test_path_dir}/output/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>  ${test_path_dir}/output/${CaseName}.log
echo "CaseName = ${CaseName}" >>  ${test_path_dir}/output/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >>  ${test_path_dir}/output/${CaseName}.log
echo "TrainingStepTime = ${TrainingTime}" >>  ${test_path_dir}/output/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >>  ${test_path_dir}/output/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>  ${test_path_dir}/output/${CaseName}.log
echo "miou = ${train_accuracy}" >>  ${test_path_dir}/output/${CaseName}.log
