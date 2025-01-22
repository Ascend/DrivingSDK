#!/bin/bash

#设置是否开启taskque,0-关闭/1-开启/2-优化
export TASK_QUEUE_ENABLE=2
#开启绑核
export CPU_AFFINITY_CONF=1
#设置hccl超时时间，单位：秒
export HCCL_EXEC_TIMEOUT=8000

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

export RANK_SIZE=1
PORT=29888 bash tools/dist_train.sh configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d.py 1 --amp --cfg-options train_dataloader.batch_size=${batch_size} train_cfg.max_epochs=${max_epochs}
