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

if [ -d ${cur_path}/test/output/perf/stage1 ]; then
  rm -rf ${cur_path}/test/output/perf/stage1
  mkdir -p ${cur_path}/test/output/perf/stage1
else
  mkdir -p ${cur_path}/test/output/perf/stage1
fi

start_time=$(date +%s)
# 非平台场景时source 环境变量
check_etp_flag=$(env | grep etp_running_flag)
etp_flag=$(echo ${check_etp_flag#*=})
if [ x"${etp_flag}" != x"true" ]; then
  source ${test_path_dir}/env_npu.sh
fi


bash ./tools/uniad_dist_perf.sh ./projects/configs/stage1_track_map/base_track_map.py 8 \
    >$cur_path/test/output/perf/stage1/train_perf.log 2>&1 &
wait


# 训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

# 训练用例信息，不需要修改
BatchSize=1
DeviceType=$(uname -m)
CaseName=${Network}_bs${BatchSize}_${WORLD_SIZE}'p'_'perf'

# 结果打印，不需要修改
echo "------------------ Final result ------------------"
# 输出性能FPS，需要模型审视修改
avg_time=`grep -a 'mmdet - INFO - Epoch '  ${test_path_dir}/output/perf/stage1/train_perf.log|awk -F "time: " '{print $2}' | awk -F ", " '{print $1}' | awk 'NR>10 {sum+=$1; count++} END {if (count != 0) printf("%.3f",sum/count)}'`
Iteration_time=$avg_time
# 打印，不需要修改
echo "Iteration time : $Iteration_time"

# 打印，不需要修改
echo "E2E Training Duration sec : $e2e_time"


# 训练总时长
TrainingTime=`grep -a 'Time'  ${test_path_dir}/output/perf/stage1/train_perf.log|awk -F "Time: " '{print $2}'|awk -F "," '{print $1}'| awk '{a+=$1} END {printf("%.3f",a)}'`

# 关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >${test_path_dir}/output/perf/stage1/${CaseName}.log
echo "RankSize = ${WORLD_SIZE}" >>${test_path_dir}/output/perf/stage1/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>${test_path_dir}/output/perf/stage1/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>${test_path_dir}/output/perf/stage1/${CaseName}.log
echo "CaseName = ${CaseName}" >>${test_path_dir}/output/perf/stage1/${CaseName}.log
echo "Iteration time = ${Iteration_time}" >>${test_path_dir}/output/perf/stage1/${CaseName}_perf_report.log
echo "TrainingTime = ${TrainingTime}" >>${test_path_dir}/output/perf/stage1/${CaseName}_perf_report.log
echo "E2ETrainingTime = ${e2e_time}" >>${test_path_dir}/output/perf/stage1/${CaseName}_perf_report.log