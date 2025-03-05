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

export RANK_SIZE=8
PORT=29888 bash tools/dist_train.sh configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d.py 8 --amp --cfg-options train_dataloader.batch_size=${batch_size} train_cfg.max_epochs=${max_epochs}
