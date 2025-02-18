#!/bin/bash

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="BEVDepth_for_PyTorch"
#训练epoch
train_epochs=1
#训练batch_size
batch_size=8
#学习率
learning_rate=0.000003125


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

#集合通信参数,不需要修改

export RANK_SIZE=8
RANK_ID_START=0

# 数据集路径,保持为空,不需要修改
data_path=""

#删除outputs文件夹，不需要修改
if [ -d ${cur_path}/outputs/ ];then
    rm -rf ${cur_path}/outputs/
fi

ckpt_path="./outputs/bev_depth_lss_r50_256x704_128x128_24e_2key/lightning_logs/version_0/checkpoints/epoch=23-step=10536.ckpt"


#设置环境变量，不需要修改
ASCEND_DEVICE_ID=0


#创建DeviceID输出目录，不需要修改
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID

else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi

#非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source  ${test_path_dir}/env_npu.sh
fi

#训练开始时间，不需要修改
start_time=$(date +%s)

#进入训练脚本目录，需要模型审视修改
cd $cur_path/
nohup python3 ./bevdepth/exps/nuscenes/mv/bev_depth_lss_r50_256x704_128x128_24e_2key.py \
        --seed 0 \
        --learning_rate ${learning_rate} \
        --max_epoch ${train_epochs} \
        --amp_backend 'native' \
        --gpus ${RANK_SIZE} \
        --precision 16 \
        --batch_size_per_device ${batch_size} > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))


#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
step_time=`grep -oP "(?:Epoch \d+: 100%.+,  )(\d+\.\d+)(?:s/it)" "$test_path_dir/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log" | awk -F "," '{print $2}' | grep -oP "\d+\.\d+" |  awk '{sum += $1; count++} END {print sum/count}'`
FPS=`awk 'BEGIN{printf "%.2f\n", '$batch_size'/'$step_time'*'$RANK_SIZE'}'`
#每回合训练迭代数，计数从0开始，需补1
train_iteration=`grep 'step_time' $test_path_dir/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | tail -1 | awk -F 'id ' '{print $2}' | awk -F ' step' '{print $1}'`
train_iteration=$((train_iteration + 1))
EpochTime=`awk 'BEGIN{printf "%.2f\n", '$step_time'*'$train_iteration'/'3600'}'`
#排除功能问题导致计算溢出的异常，增加健壮性
if [ x"${FPS}" == x"2147483647" ] || [ x"${FPS}" == x"-2147483647" ];then
    FPS=""
fi
#打印，不需要修改
echo "Final Performance images/sec : $FPS"
echo "Final Performance each_epoch_time : $EpochTime h"

#打印，不需要修改
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%df\n",''3600*'${EpochTime}'}'`
#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要模型审视修改
grep  "loss=" $test_path_dir/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F "loss=" '{print $4}' | awk -F "," '{print $1}' | sed  '/^$/d' > $test_path_dir/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' $test_path_dir/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`


#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
