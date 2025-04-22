#!/bin/sh

get_abs_filename() {
  echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
}

helpFunction()
{
   echo "TODO"
   exit 1
}

# ****************************ADD**********************

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
#设置aclnn cache，使能host侧算子下发缓存机制
export DISABLE_L2_CACHE=0
#设置是否开启均匀绑核,0-关闭/1-开启粗粒度绑核/2-开启细粒度绑核
export CPU_AFFINITY_CONF=1
#HCCL白名单开关,配置在使用HCCL时是否关闭通信白名单。1-关闭/0-开启
export HCCL_WHITELIST_DISABLE=1
#启用可扩展内存段分配策略
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

cur_path=`pwd`
cur_path_last_diename=${cur_path##*/}

batch_node_size=192

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

# ****************************ADD**********************

while getopts "d:a:l:n:c:p:u:" opt
do
   case "$opt" in
      d ) d="$OPTARG" ;;
      a ) a="$OPTARG" ;;
      l ) l="$OPTARG" ;;
      n ) n="$OPTARG" ;;
      c ) c="$OPTARG" ;;
      p ) p="$OPTARG" ;;
      u ) u="$OPTARG" ;;
      ? ) helpFunction ;;
   esac
done

if [ -z "$a" ] || [ -z "$d" ] || [ -z "$l" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi
if [ -z "$u" ]
then u='false'
fi
d=$(get_abs_filename "$d")
a=$(get_abs_filename "$a")
l=$(get_abs_filename "$l")
if [ -z "$p" ]
then
 p=""
else
  p=$(get_abs_filename "$p")
fi

export RANK_SIZE=8
cd ./train/tasks/semantic;

# ****************************ADD**********************
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

me=20
nohup python -m torch.distributed.launch \
          --nproc_per_node=${RANK_SIZE} \
          --rdzv_endpoint=localhost:${PORT} \
          ./train.py -d "$d"  -ac "$a" -l "$l" -n "$n" -p "$p" -u "$u" -me $me \
          --launcher pytorch > ${output_path_dir}/train_8p_performance.log 2>&1 &
wait

# ****************************ADD**********************
end_time=$(date +%s)
echo "end_time=$(date -d @${end_time} "+%Y-%m-%d %H:%M:%S")"
e2e_time=$(( $end_time - $start_time ))

# 从 log 中获取性能
avg_time=`grep "Epoch" ${output_path_dir}/train_8p_performance.log | tail -n 5 | awk -F "Time " '{print $2}' | awk '{sum+=$1; count++} END {if(count>0) print sum/count}'`
echo "avg_time : ${avg_time}"
# 从 log 中获取精度
mAP=`grep "IoU avg *" ${output_path_dir}/train_8p_performance.log |awk -F "=" '{print $NF}'|awk 'END {print}'`
echo "mAP : ${mAP}"
avg_fps=`awk 'BEGIN{printf "%.3f\n", '$batch_node_size'/'$avg_time'}'`

# 输出结果
echo "[INFO] Final Result"
echo " - End to End Time : ${e2e_time}s"
echo " - Time avg per batch :  ${avg_time}s"
echo " - Final Performance images/sec :  ${avg_fps}"
echo " - Final mAP(IoU=0.50:0.95) : ${mAP}"