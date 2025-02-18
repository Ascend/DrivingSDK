#!/bin/bash

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="BEVDepth_for_PyTorch"
#训练epoch
train_epochs=24
#训练batch_size
batch_size=32
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

export RANK_SIZE=1
RANK_ID_START=0

# 数据集路径,保持为空,不需要修改
data_path=""


ckpt_path="./outputs/bev_depth_lss_r50_256x704_128x128_24e_2key/lightning_logs/version_2/checkpoints/epoch=23-step=10536.ckpt"

#设置环境变量，不需要修改
ASCEND_DEVICE_ID=0

#创建DeviceID输出目录，不需要修改
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID/
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi

#非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source  ${test_path_dir}/env_npu.sh
fi


echo "------------------ Start eval ------------------"
nohup python3 ./bevdepth/exps/nuscenes/mv/bev_depth_lss_r50_256x704_128x128_24e_2key.py \
        -e \
        --seed 0 \
        --learning_rate ${learning_rate} \
        --max_epoch ${train_epochs} \
        --amp_backend 'native' \
        --gpus 1 \
        --precision 16 \
        --ckpt_path ${ckpt_path} \
        -b ${batch_size} > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/eval_${ASCEND_DEVICE_ID}.log 2>&1 &
wait

###打印精度数据
metric_names=("mAP" "mATE" "mASE" "mAOE" "mAVE" "mAAE" "NDS")
for metric in "${metric_names[@]}"; do
    grep -i "$metric" "${test_path_dir}/output/${ASCEND_DEVICE_ID}/eval_${ASCEND_DEVICE_ID}.log"
done | sort -u