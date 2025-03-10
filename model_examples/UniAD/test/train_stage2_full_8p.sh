#!/bin/bash
################基础配置参数，需要模型审视修改##################
# 网络名称，同目录名称
Network="UniAD"
WORLD_SIZE=8
WORK_DIR=""
LOAD_FROM=""

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

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

if [ -d ${cur_path}/test/output/full/stage2 ]; then
  rm -rf ${cur_path}/test/output/full/stage2
  mkdir -p ${cur_path}/test/output/full/stage2
else
  mkdir -p ${cur_path}/test/output/full/stage2
fi

start_time=$(date +%s)
# 非平台场景时source 环境变量
check_etp_flag=$(env | grep etp_running_flag)
etp_flag=$(echo ${check_etp_flag#*=})
if [ x"${etp_flag}" != x"true" ]; then
  source ${test_path_dir}/env_npu.sh
fi

python_path=$(pip show torch |grep Location|awk -F ': ' '{print $2}')
cp -f ./nuscenes_need/mot.py ${python_path}/nuscenes/eval/tracking/mot.py

bash ./tools/uniad_dist_train.sh ./projects/configs/stage2_e2e/base_e2e.py 8 \
    >$cur_path/test/output/full/stage2/train_full.log 2>&1 &
wait

# 训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

# 训练用例信息，不需要修改
BatchSize=1
DeviceType=$(uname -m)
CaseName=${Network}_bs${BatchSize}_${WORLD_SIZE}'p'

# 结果打印，不需要修改
echo "------------------ Final result ------------------"
# 输出性能FPS，需要模型审视修改
avg_time=`grep -a 'mmdet - INFO - Epoch '  ${test_path_dir}/output/full/stage2/train_full.log|awk -F "time: " '{print $2}' | awk -F ", " '{print $1}' | awk 'NR>10 {sum+=$1; count++} END {if (count != 0) printf("%.3f",sum/count)}'`
Iteration_time=$avg_time
# 打印，不需要修改
echo "Iteration time : $Iteration_time"

# 输出训练精度amota,需要模型审视修改
amota=$(grep -a "mmdet - INFO - Epoch(val)" ${test_path_dir}/output/full/stage2/train_full.log  |tail -1|awk -F "amota: " '{print $2}' |awk -F ", " '{print $1}'| awk -F ", " '{print $1}')
# 输出训练精度lanes_iou,需要模型审视修改
lanes_iou=$(grep -a "mmdet - INFO - Epoch(val)" ${test_path_dir}/output/full/stage2/train_full.log  |tail -1|awk -F "lanes_iou: " '{print $2}' |awk -F ", " '{print $1}'| awk -F ", " '{print $1}')
# 输出训练精度amotp,需要模型审视修改
amotp=$(grep -a "mmdet - INFO - Epoch(val)" ${test_path_dir}/output/full/stage2/train_full.log  |tail -1|awk -F "amotp: " '{print $2}' |awk -F ", " '{print $1}'| awk -F ", " '{print $1}')

L2_eval=$(grep -a "|      L2     |" ${test_path_dir}/output/full/stage2/train_full.log | tail -1 | awk -F"[| ]+" '{sum=0; count=0; for(i=2; i<=NF; i++) {if($i ~ /^[0-9]+(\.[0-9]+)?$/) {sum+=$i; count++}}; if(count>0) print sum/count}')

# 打印，不需要修改
#echo "amota : ${amota}"
#echo "lanes_iou : ${lanes_iou}"
#echo "amotp : ${amotp}"
#echo "E2E Training Duration sec : $e2e_time"
echo "L2 : " ${L2_eval}

# 训练总时长
TrainingTime=`grep -a 'Time'  ${test_path_dir}/output/full/stage2/train_full.log|awk -F "Time: " '{print $2}'|awk -F "," '{print $1}'| awk '{a+=$1} END {printf("%.3f",a)}'`

# 关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >${test_path_dir}/output/full/stage2/${CaseName}.log
echo "RankSize = ${WORLD_SIZE}" >>${test_path_dir}/output/full/stage2/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>${test_path_dir}/output/full/stage2/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>${test_path_dir}/output/full/stage2/${CaseName}.log
echo "CaseName = ${CaseName}" >>${test_path_dir}/output/full/stage2/${CaseName}.log
#echo "AMOTA = ${amota}" >>${test_path_dir}/output/full/stage2/${CaseName}.log
#echo "IoU-lane = ${lanes_iou}" >>${test_path_dir}/output/full/stage2/${CaseName}.log
#echo "AMOTP = ${amotp}" >>${test_path_dir}/output/full/stage2/${CaseName}.log
echo "Iteration time = ${Iteration_time}" >>${test_path_dir}/output/full/stage2/${CaseName}_perf_report.log
echo "TrainingTime = ${TrainingTime}" >>${test_path_dir}/output/full/stage2/${CaseName}_perf_report.log
echo "E2ETrainingTime = ${e2e_time}" >>${test_path_dir}/output/full/stage2/${CaseName}_perf_report.log