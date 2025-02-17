#!/bin/bash
# 模型配置
export RANK_SIZE=8
batch_size=8
epochs=1
work_dir="work_dirs/bevnext-stage2"
stage1_ckpts_path="work_dirs/bevnext-stage1/epoch_2_ema.pth"

# 获取传入的参数，重新赋值对应参数
for para in $*
do
    if [[ $para == --work_dir* ]];then
        work_dir=`echo ${para#*=}`
    fi
    if [[ $para == --stage1_ckpts_path* ]];then
        stage1_ckpts_path=`echo ${para#*=}`
    fi
done

# 训练用例信息
network="BEVNeXt"
device_type=`uname -m`
case_name=${network}_stage2_${RANK_SIZE}p_bs${batch_size}_epochs${epochs}
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

# 修改配置文件中的 load_from
sed -i "s|load_from = .*|load_from = '${stage1_ckpts_path}'|g" ./configs/bevnext/bevnext-stage2.py
# 修改配置文件中的 max_epochs
sed -i "s|max_epochs=.*|max_epochs=${epochs})|g" ./configs/bevnext/bevnext-stage2.py

# 开始训练
echo "[BEVNeXt] Training stage2..."
bash ./tools/dist_train.sh configs/bevnext/bevnext-stage2.py ${RANK_SIZE} --work-dir ${work_dir} --seed 0 &
wait

# 恢复配置文件中的 total_epochs
sed -i "s|max_epochs=.*|max_epochs=12)|g" ./configs/bevnext/bevnext-stage2.py

echo "[BEVNeXt] Train stage2 success."