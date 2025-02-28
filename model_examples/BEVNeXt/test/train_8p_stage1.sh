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

mkdir -p ${work_dir}

source ${test_path_dir}/env_npu.sh

# 开始训练
# 训练开始时间
start_time=$(date +%s)
echo "[BEVNeXt] Stage1 start_time=$(date -d @${start_time} "+%Y-%m-%d %H:%M:%S")"

bash ./tools/dist_train.sh configs/bevnext/bevnext-stage1.py ${RANK_SIZE} --work-dir ${work_dir} --seed 0 \
    > ${work_dir}/train_stage1_${RANK_SIZE}p_bs${batch_size}_epochs${epochs}.log 2>&1 &
wait

# 训练结束时间
end_time=$(date +%s)
echo "[BEVNeXt] Stage1 end_time=$(date -d @${end_time} "+%Y-%m-%d %H:%M:%S")"
e2e_time=$(( $end_time - $start_time ))

# 输出性能
avg_time=`grep -o ", time: [0-9.]*" ${work_dir}/train_stage1_${RANK_SIZE}p_bs${batch_size}_epochs${epochs}.log | grep -o "[0-9.]*" | awk '{sum+=$1; count++} END {if(count>0) print sum/count}'`
avg_fps=`awk 'BEGIN{printf "%.3f\n", '$batch_size'*'${RANK_SIZE}'/'$avg_time'}'`

# 结果打印
echo "------------------ Final result ------------------"
echo "[BEVNeXt] Stage1 E2E Training Duration sec : ${e2e_time}"
echo "[BEVNeXt] Stage1 Final Performance images/sec : ${avg_fps}"
echo "[BEVNeXt] Stage1 Train stage1 success."

# 将关键信息打印到 ${case_name}.log 中
echo "Network = ${network}" > ${work_dir}/${case_name}.log
echo "DeviceType = ${device_type}" >> ${work_dir}/${case_name}.log
echo "RankSize = ${RANK_SIZE}" >> ${work_dir}/${case_name}.log
echo "BatchSize = ${batch_size}" >> ${work_dir}/${case_name}.log
echo "CaseName = ${case_name}" >> ${work_dir}/${case_name}.log
echo "E2ETrainingTime = ${e2e_time}" >> ${work_dir}/${case_name}.log
echo "TrainingTime = ${avg_time}" >> ${work_dir}/${case_name}.log
echo "averageFPS = ${avg_fps}" >> ${work_dir}/${case_name}.log