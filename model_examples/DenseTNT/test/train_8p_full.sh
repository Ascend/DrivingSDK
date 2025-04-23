#!/bin/bash

npu_num=8
hidden_size=128
batch_size=64
core_num=16
epochs=16
output_path=""
data_path=""

for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
    if [[ $para == --output_path* ]];then
        output_path=`echo ${para#*=}`
    fi
    if [[ $para == --epochs* ]];then
        epochs=`echo ${para#*=}`
    fi
done

if [[ "$data_path" == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

echo "[INFO] Start setting ENV VAR"

# 配置环境变量
msnpureport -g error -d 0
msnpureport -g error -d 1
msnpureport -g error -d 2
msnpureport -g error -d 3
msnpureport -g error -d 4
msnpureport -g error -d 5
msnpureport -g error -d 6
msnpureport -g error -d 7
#设置Device侧日志等级为error
msnpureport -g error
#关闭Device侧Event日志
msnpureport -e disable

#将Host日志输出到串口,0-关闭/1-开启
export ASCEND_SLOG_PRINT_TO_STDOUT=0
#设置默认日志级别,0-debug/1-info/2-warning/3-error
export ASCEND_GLOBAL_LOG_LEVEL=3
#设置Host侧Event日志开启标志,0-关闭/1-开启
export ASCEND_GLOBAL_EVENT_ENABLE=0
#设置是否开启taskque,0-关闭/1-开启/2-流水优化
export TASK_QUEUE_ENABLE=2
#设置是否开启均匀绑核,0-关闭/1-开启粗粒度绑核/2-开启细粒度绑核
export CPU_AFFINITY_CONF=2
#缓存aclnn算子数量
export ACLNN_CACHE_LIMIT=100000
#启动动态shape缓存
export HOST_CACHE_CAPACITY=20
#启用hccl ffts+模式
export ASCEND_ENHANCE_enable=1
#缓存分配器创建特定的内存块
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
#使能高性能内存库
export LD_PRELOAD=$LD_PRELOAD:/usr/local/lib/lib/libtcmalloc.so

echo "[INFO] Finish setting ENV VAR"

cur_path=`pwd`
cur_path_last_diename=${cur_path##*/}

if [ x"${cur_path_last_diename}" == x"test" ];then
    test_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi
output_path_dir=${test_path_dir}/output
if [ -d ${output_path_dir} ]; then
  rm -rf ${output_path_dir}
fi
mkdir -p ${output_path_dir}

# 训练
start_time=$(date +%s)
echo "start_time=$(date -d @${start_time} "+%Y-%m-%d %H:%M:%S")"

nohup python src/run.py --argoverse --argoverse2 --future_frame_num 60 \
  --do_train --data_dir ${data_path} --output_dir ${output_path} --num_train_epochs ${epochs} \
  --hidden_size ${hidden_size} --train_batch_size ${batch_size} --use_map \
  --distributed_training ${npu_num} --core_num ${core_num} --use_centerline \
  --other_params \
    semantic_lane direction l1_loss \
    goals_2D enhance_global_graph subdivide goal_scoring laneGCN point_sub_graph \
    lane_scoring complete_traj complete_traj-3 \
> ${output_path_dir}/train_8p_full.log 2>&1 &

wait
end_time=$(date +%s)
echo "end_time=$(date -d @${end_time} "+%Y-%m-%d %H:%M:%S")"
e2e_time=$(( $end_time - $start_time ))

# 从 log 中获取性能
avg_time=`sed 's/\r/\n/g' ${output_path_dir}/train_8p_full.log | grep "loss=" | tail -n 200 | grep -oP "[0-9]+\.[0-9]+it/s" | awk '{sum+=$1; count++} END {if(count>0) print sum/count}'`
# 从 log 中获取精度
FDE=`grep "FDE:" ${output_path_dir}/train_8p_full.log | awk -F " " '{print $2}'| awk 'END {print}'`
MR=`grep "FDE:" ${output_path_dir}/train_8p_full.log | awk -F ":" '{print $NF}'| awk 'END {print}'`

# 输出结果
echo "[INFO] Final Result"
echo " - End to End Time is ${e2e_time}s"
echo " - Final Performance iter/sec ${avg_time}"
echo " - Final FDE : ${FDE}"
echo " - Final MR(2m,4m,6m) : ${MR}"
