#!/bin/bash

#设置是否开启taskque,0-关闭/1-开启/2-优化
export TASK_QUEUE_ENABLE=2

batch_size=8
num_workers=4
max_epochs=1
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

export RANK_SIZE=1
PORT=29888 bash ./tools/dist_train.sh ./configs/fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py 1 --amp --cfg-options train_dataloader.batch_size=${batch_size} train_dataloader.num_workers=${num_workers} train_cfg.max_epochs=${max_epochs} data_root=${data_root}
