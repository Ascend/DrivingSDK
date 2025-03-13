#!/bin/bash
################基础配置参数，需要模型审视修改##################
# 网络名称，同目录名称
Network="HPTR"
WORLD_SIZE=8
Dataset="womd" #womd, av2

###############指定训练脚本执行路径###############
# cd到与test文件夹同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
cur_path=$(pwd)
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ]; then
  test_path_dir=${cur_path}
  cd ..
  cur_path=$(pwd)
else
  test_path_dir=${cur_path}/test
fi

ASCEND_DEVICE_ID=0

if [ -d ${cur_path}/test/output/${ASCEND_DEVICE_ID} ]; then
  rm -rf ${cur_path}/test/output/${ASCEND_DEVICE_ID}
  mkdir -p ${cur_path}/test/output/${ASCEND_DEVICE_ID}
else
  mkdir -p ${cur_path}/test/output/${ASCEND_DEVICE_ID}
fi

# 非平台场景时source 环境变量
check_etp_flag=$(env | grep etp_running_flag)
etp_flag=$(echo ${acheck_etp_flag#*=})
if [ x"${etp_flag}" != x"true" ]; then
  source ${test_path_dir}/env_npu.sh
fi

sed -i '7 s/null/41/'  ./configs/trainer/${Dataset}.yaml

start_time=$(date +%s)

bash ./bash/train_{$Dataset}.sh 2>&1 | tee $cur_path/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log &
wait

end_time=$(date +%s)

sed -i '7 s/41/null/'  ./configs/trainer/${Dataset}.yaml

e2e_time=$(( $end_time - $start_time ))

# 训练用例信息，不需要修改
BatchSize=8
DeviceType=$(uname -m)
CaseName=${Network}_${Dataset}_bs${BatchSize}_${WORLD_SIZE}'p'_'acc'

# 结果打印，不需要修改
echo "------------------ Final result ------------------"
# 输出训练metric,需要模型审视修改
result=$(sed 's/\r/\n/g' $cur_path/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | grep -E 'waymo_pred/min_ade|waymo_pred/min_fde' | tail -n 2 | sed 's/^wandb: *//')
echo "${result}"
TrainAccuracy=$(echo "$result" | sed 's#waymo_pred/##g' | awk '{printf "%s %s ", $1, $2}' | xargs)
echo "E2E Duration sec : $e2e_time"

# 训练总时长
txt=$(sed 's/\r/\n/g' $cur_path/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log |tac| awk '/Validating: 0it/{flag=1; next} flag && /Epoch 0/{print; exit}')
TrainingTime=$(echo "$txt" | awk -F'[][]' '{split($2, a, "<"); split(a[1], t, ":"); if (length(t) == 3) {print t[1]*3600 + t[2]*60 + t[3];} else if (length(t) == 2) {print t[1]*60 + t[2];}}')
echo "Training time per epoch : $TrainingTime"

# 输出单步耗时，需要模型审视修改
num_step=$(echo $txt | cut -d'|' -f3 | cut -d'/' -f1)
time_per_step=$(awk "BEGIN {printf \"%.3f\", $TrainingTime / $num_step}")
Iteration_time=$time_per_step
echo "Iteration time : $Iteration_time"

# 关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${WORLD_SIZE}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${TrainAccuracy}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "Iteration time = ${Iteration_time}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log