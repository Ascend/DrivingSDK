#!/bin/bash

export RANK_SIZE=8
batch_size=4
num_workers=4
max_epochs=12
data_root=''
for para in $*
do
    if [[ $para == --batch_size* ]];then
        batch_size=`echo ${para#*=}`
    elif [[ $para == --num_workers* ]];then
        num_workers=`echo ${para#*=}`
    elif [[ $para == --max_epochs* ]];then
        max_epochs=`echo ${para#*=}`
    elif [[ $para == --data_root* ]];then
        data_root=`echo ${para#*=}`
    fi
done

if [[ "$data_root" == "" ]];then
    echo "[Error] para \"data_root\" must be confing"
    exit 1
fi
if [ ! -d "$data_root" ]; then
    echo "[Error] para \"$data_root\" must be exist"
    exit 1
fi
if [ ! -d "data" ]; then
    mkdir -p data
fi
ln -nsf $data_root data/coco

# 训练用例信息
network="FCOS"
device_type=`uname -m`
case_name=${network}_${RANK_SIZE}p_bs${batch_size}_epochs${max_epochs}
echo "[FCOS] case_name = ${case_name}"

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

source ${test_path_dir}/env_npu.sh

# 训练开始时间
start_time=$(date +%s)
echo "[FCOS] start_time=$(date -d @${start_time} "+%Y-%m-%d %H:%M:%S")"

PORT=29888 bash ./tools/dist_train.sh ./configs/fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py 8 \
    --cfg-options train_dataloader.batch_size=${batch_size} train_dataloader.num_workers=${num_workers} train_dataloader.pin_memory=True train_cfg.max_epochs=${max_epochs} data_root=${data_root} \
    > ${output_path_dir}/train_${RANK_SIZE}p_bs${batch_size}_epochs${max_epochs}.log 2>&1 &
wait

# 训练结束时间
end_time=$(date +%s)
echo "[FCOS] end_time=$(date -d @${end_time} "+%Y-%m-%d %H:%M:%S")"
e2e_time=$(( $end_time - $start_time ))

# # 输出性能
avg_fps=`grep "Epoch(train) \[average\]" ${output_path_dir}/train_${RANK_SIZE}p_bs${batch_size}_epochs${max_epochs}.log | awk '{printf "%.2f\n", $NF}'`
avg_time=`awk 'BEGIN{printf "%.4f\n", '${batch_size}'*'${RANK_SIZE}'/'${avg_fps}'}'`
# 输出训练精度
mAP=`grep "Average Precision.* IoU=0.50\:0.95.* all" ${output_path_dir}/train_${RANK_SIZE}p_bs${batch_size}_epochs${max_epochs}.log | awk -F "=" '{print $NF}' | awk 'END {print}'`

# 结果打印
echo "------------------ Final result ------------------"
echo "[FCOS] E2E Training Duration sec : ${e2e_time}"
echo "[FCOS] Final Performance images/sec : ${avg_fps}"
echo "[FCOS] Final mAP(IoU=0.50:0.95) : ${mAP}"

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
