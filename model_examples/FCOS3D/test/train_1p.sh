#!/bin/bash

# test_path_dir 为包含 test 文件夹的路径
cur_path=`pwd`
cur_path_last_diename=${cur_path##*/}
if [ x"${cur_path_last_diename}" == x"test" ];then
    test_path_dir=${cur_path}
    # 若当前在 test 目录下，cd 到 test 同级目录
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi

source ${test_path_dir}/env_npu.sh

batch_size=2
max_epochs=1
data_root=''
for para in $*
do
    if [[ $para == --batch_size* ]];then
        batch_size=`echo ${para#*=}`
    elif [[ $para == --max_epochs* ]];then
        max_epochs=`echo ${para#*=}`
    elif [[ $para == --data_root* ]];then
        data_root=`echo ${para#*=}`
    fi
done

if [[ "$data_root" == "" ]];then
    echo "[Error] para \"data_root\" must be configured."
    exit 1
fi
if [ ! -d "$data_root" ]; then
    echo "[Error] para \"$data_root\" must exist."
    exit 1
fi
if [ ! -d "data" ]; then
    mkdir -p data
fi
ln -nsf $data_root data/nuscenes

# 在 test 目录下创建 output，用于存放日志文件
output_path_dir=${test_path_dir}/output
mkdir -p ${output_path_dir}
log_path=${output_path_dir}/train_1p_${batch_size}bs_${max_epochs}epochs.log
echo "log path is ${log_path}"
rm -f ${log_path}

# 训练开始时间
start_time=$(date +%s)
echo "[FCOS3D] start_time=$(date -d @${start_time} "+%Y-%m-%d %H:%M:%S")"

export RANK_SIZE=1
PORT=29888 bash tools/dist_train.sh configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d.py 1 \
    --cfg-options train_dataloader.batch_size=${batch_size} train_cfg.max_epochs=${max_epochs} \
    > ${log_path} 2>&1 &
wait

# 训练结束时间
end_time=$(date +%s)
echo "[FCOS3D] end_time=$(date -d @${end_time} "+%Y-%m-%d %H:%M:%S")"
e2e_time=$(( $end_time - $start_time ))
echo "E2ETrainingTime = ${e2e_time}"

step_time=$(grep -o "  time: [0-9.]*" ${log_path} | tail -n 19 | grep -o "[0-9.]*" | awk '{sum += $1} END {print sum/NR}')
FPS=$(awk BEGIN'{print ('$batch_size' * '$world_size') / '$step_time'}')

# 打印性能
echo "Final Performance images/sec (FPS) : ${FPS}"
echo "FPS = ${FPS}" >>${log_path}
