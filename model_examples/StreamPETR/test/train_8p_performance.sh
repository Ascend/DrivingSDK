#!/bin/bash
################基础配置参数，需要模型审视修改##################
# 网络名称，同目录名称
Network="StreamPETR"
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

ASCEND_DEVICE_ID=0

if [ -d ${cur_path}/test/output/${ASCEND_DEVICE_ID} ]; then
  rm -rf ${cur_path}/test/output/${ASCEND_DEVICE_ID}
  mkdir -p ${cur_path}/test/output/${ASCEND_DEVICE_ID}
else
  mkdir -p ${cur_path}/test/output/${ASCEND_DEVICE_ID}
fi

# 配置环境变量

# 设置 device 侧日志登记为 error
msnpureport -g error -d 0
msnpureport -g error -d 1
msnpureport -g error -d 2
msnpureport -g error -d 3
msnpureport -g error -d 4
msnpureport -g error -d 5
msnpureport -g error -d 6
msnpureport -g error -d 7
# 关闭 Device 侧 Event 日志
msnpureport -e disable

# 将 Host 日志输出到串口, 0-关闭/1-开启
export ASCEND_SLOG_PRINT_TO_STDOUT=0
# 设置默认日志级别, 0-debug/1-info/2-warning/3-error
export ASCEND_GLOBAL_LOG_LEVEL=3
# 设置Event日志开启标志, 0-关闭/1-开启
export ASCEND_GLOBAL_EVENT_ENABLE=0

# HCCL 白名单开关, 1-关闭/0-开启
export HCCL_WHITELIST_DISABLE=1
#配置HCCL的初始化root通信网卡IP
export HCCL_IF_IP=$(hostname -I |awk '{print $1}')
#配置不同设备之间socket建链过程的等待时间，取值范围[120, 7200]，默认120，单位s
export HCCL_CONNECT_TIMEOUT=1200

#设置是否开启taskque,0-关闭/1-开启/2-流水优化
export TASK_QUEUE_ENABLE=2
#设置是否开启均匀绑核,0-关闭/1-开启粗粒度绑核/2-开启细粒度绑核
export CPU_AFFINITY_CONF=1


start_time=$(date +%s)
# 非平台场景时source 环境变量

bash tools/dist_train.sh projects/configs/StreamPETR/stream_petr_vov_flash_800_bs2_seq_24e_performance.py 8 --work-dir work_dirs/report_vision --no-validate \
    >$cur_path/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
wait


# 训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

# 训练用例信息，不需要修改
BatchSize=2
DeviceType=$(uname -m)
CaseName=${Network}_bs${BatchSize}_${WORLD_SIZE}'p'_'acc'

# 结果打印，不需要修改
echo "------------------ Final result ------------------"
# 输出性能FPS，需要模型审视修改
avg_time=`grep -a ' - mmdet - INFO - Iter '  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F "time: " '{print $2}' | awk -F "," '{print $1}' | awk 'NR>=2 && NR<=11' | awk '{sum+=$1; count++} END {if(count>0) print sum/count}'`

Iteration_time=$avg_time
FPS=$(awk BEGIN'{print ('$BatchSize' * '$WORLD_SIZE') / '$Iteration_time'}')
# 打印，不需要修改
echo "Iteration time : $Iteration_time"
echo "FPS : $FPS"

# 打印，不需要修改
echo "E2E Training Duration sec : $e2e_time"


# 关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${WORLD_SIZE}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "Iterationtime = ${Iteration_time}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}_perf_report.log
echo "E2ETrainingTime = ${e2e_time}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}_perf_report.log