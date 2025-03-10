#!/bin/bash
# 模型配置
export RANK_SIZE=8
epochs=12
work_dir="work_dirs/bevnext-stage2"

# 获取传入的参数，重新赋值对应参数
for para in $*
do
    if [[ $para == --work_dir* ]];then
        work_dir=`echo ${para#*=}`
    fi
done

# 训练用例信息
network="BEVNeXt"
device_type=`uname -m`
case_name=${network}_${RANK_SIZE}p_epochs${epochs}_eval
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

echo "[BEVNeXt] Evaluating..."
bash tools/dist_test.sh $work_dir/*.py $work_dir/epoch_${epochs}_ema.pth ${RANK_SIZE} --eval mAP --no-aavt \
    > ${work_dir}/bevnext_epoch${epochs}_eval.log 2>&1 &
wait

NDS=$(grep -o "NDS: [0-9.]*" ${work_dir}/bevnext_epoch${epochs}_eval.log | awk 'END {print $NF}')
mAP=$(grep -o "mAP: [0-9.]*" ${work_dir}/bevnext_epoch${epochs}_eval.log | awk 'END {print $NF}')
echo "[BEVNeXt] NDS : ${NDS}"
echo "[BEVNeXt] mAP : ${mAP}"

# 将关键信息打印到 ${case_name}.log 中
echo "Network = ${network}" > ${work_dir}/${case_name}.log
echo "DeviceType = ${device_type}" >> ${work_dir}/${case_name}.log
echo "NDS = ${NDS}" >> ${work_dir}/${case_name}.log
echo "mAP = ${mAP}" >> ${work_dir}/${case_name}.log
