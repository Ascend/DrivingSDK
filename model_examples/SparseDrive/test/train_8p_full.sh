#!/bin/sh

get_abs_filename() {
  echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
}

helpFunction()
{
   echo "TODO"
   exit 1
}

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

#设置是否开启taskque,0-关闭/1-开启/2-优化
export TASK_QUEUE_ENABLE=2
#设置是否开启均匀绑核,0-关闭/1-开启粗粒度绑核/2-开启细粒度绑核
export CPU_AFFINITY_CONF=1
#减少显存占用
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

batch_node_size_stage1=64
batch_node_size_stage2=48
rank_size=8

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

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

## stage1
nohup bash ./tools/dist_train.sh \
   projects/configs/sparsedrive_small_stage1.py \
   ${rank_size} \
   --deterministic > ${output_path_dir}/stage1_train_8p_full.log 2>&1 &

wait

cp -f work_dirs/sparsedrive_small_stage1/iter_43900.pth ckpt/sparsedrive_stage1.pth

## stage2
nohup bash ./tools/dist_train.sh \
   projects/configs/sparsedrive_small_stage2.py \
   ${rank_size} \
   --deterministic > ${output_path_dir}/stage2_train_8p_full.log 2>&1 &

wait

log_file1=`find work_dirs/sparsedrive_small_stage1 -regex ".*\.log" | sort -r | head -n 1`
log_file2=`find work_dirs/sparsedrive_small_stage2 -regex ".*\.log" | sort -r | head -n 1`

end_time=$(date +%s)
echo "end_time=$(date -d @${end_time} "+%Y-%m-%d %H:%M:%S")"
e2e_time=$(( $end_time - $start_time ))

# 从 log 中获取性能
avg_time1=`grep "time" ${log_file1} | tail -n 10 | awk -F "time: " '{print $2}' | awk '{sum+=$1; count++} END {if(count>0) print sum/count}'`
avg_time2=`grep "time" ${log_file2} | tail -n 10 | awk -F "time: " '{print $2}' | awk '{sum+=$1; count++} END {if(count>0) print sum/count}'`
# 从 log 中获取精度
amota=`grep "val" ${log_file1} | awk -F "amota: " '{print $2}' | awk -F  "," '{print $1}' | awk 'END {print}'`
L2=`grep "val" ${log_file2}  | awk -F "L2: " '{print $2}' | awk 'END {print}'`

avg_fps1=`awk 'BEGIN{printf "%.3f\n", '$batch_node_size_stage1'/'$avg_time1'}'`
avg_fps2=`awk 'BEGIN{printf "%.3f\n", '$batch_node_size_stage2'/'$avg_time2'}'`

# 输出结果
echo "[INFO] Final Result"
echo " - End to End Time : ${e2e_time}s"
echo " - Stage1 Time avg per batch :  ${avg_time1}s"
echo " - Stage1 Final Performance images/sec :  ${avg_fps1}"
echo " - Stage2 Time avg per batch :  ${avg_time2}s"
echo " - Stage2 Final Performance images/sec :  ${avg_fps2}"
echo " - Stage1 AMOTA : ${amota}"
echo " - Stage2 L2 : ${L2}"