#!/bin/bash
# 模型配置
export RANK_SIZE=8
batch_size=8
epochs=2
work_dir="work_dirs/bevnext-stage1"

# 获取传入的参数，重新赋值 work_dir
for para in $*
do
    if [[ $para == --work_dir* ]];then
        work_dir=`echo ${para#*=}`
    fi
done

# 训练用例信息
network="BEVNeXt"
device_type=`uname -m`
case_name=${network}_stage1_${RANK_SIZE}p_bs${batch_size}_epochs${epochs}
echo "[BEVNeXt] case_name = ${case_name}"

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

source ${test_path_dir}/env_npu.sh

# 开始训练
echo "[BEVNeXt] Training stage1..."
bash ./tools/dist_train.sh configs/bevnext/bevnext-stage1.py ${RANK_SIZE} --work-dir ${work_dir} --seed 0 &
wait

echo "[BEVNeXt] Train stage1 success."