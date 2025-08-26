#!/bin/bash

###############指定训练脚本执行路径###############
# cd到与test文件夹同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
cur_path=`pwd`
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ];then
    test_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi

#集合通信参数,不需要修改

RANK_ID_START=0

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="PointPillar"
#训练batch_size
batch_size=4
#训练NPU卡数
num_npu=8
#训练模型配置文件
cfg_file="cfgs/kitti_models/pointpillar.yaml"
#计算mAP时整除的倍率（检测类别数3，每个类别对应的指标数4）
num_metric=12

while [[ $# -gt 0 ]]; do 
  case $1 in
    --NUM-NPU|--num-npu)
      num_npu="$2"
      shift 2
      ;;
    --BATCH-SIZE|--batch-size)
      batch_size="$2"
      shift 2
      ;;
    --help)
      echo "用法: $0 [选项]"
      echo "选项:"
      echo " --NUM-NPU 设置 NUM-NPU (默认: 8)"
      echo " --BATCH-SIZE 设置 BATCH_SIZE (默认: 24)"
      echo " --help 显示帮助信息"
      exit 0
      ;;
    *)
      echo "未知选项: $1"
      echo "使用 --help 查看帮助"
      exit 1
      ;;
  esac
done

echo "num_npu: $num_npu"
echo "batch_size: $batch_size"

#设置环境变量，不需要修改
ASCEND_DEVICE_ID=0


#创建DeviceID输出目录，不需要修改
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${test_path_dir}/output/${ASCEND_DEVICE_ID}

else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi

#非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source  ${test_path_dir}/env_npu.sh
fi

# 备份config文件
cp ${cfg_file} ${cfg_file}.bak

#修改batchsize
sed -i "s/BATCH_SIZE_PER_GPU:[[:space:]]*[0-9]\{1,\}/BATCH_SIZE_PER_GPU: $batch_size/g" ${cfg_file}

#复原callback
restore_config() {
    if [ -f ${cfg_file}.bak ]; then
        mv -f ${cfg_file}.bak ${cfg_file}
    fi
}

#异常复原
trap restore_config EXIT SIGINT SIGTERM ERR

#训练开始时间，不需要修改
start_time=$(date +%s)

#进入训练脚本目录，需要模型审视修改
cd $cur_path/

# DDP通信端口
while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

nohup python -m torch.distributed.launch \
          --nproc_per_node=${num_npu} \
          --rdzv_endpoint=localhost:${PORT} \
          train.py \
          --launcher pytorch \
          --cfg_file ${cfg_file} \
          --epochs 1 \
          --logger_iter_interval 1 > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
avg_time=`grep -a 'Batch time: ' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F 'Batch time: ' '{print $2}'|awk 'NR>10'|awk -F '(' '{print $1}'|tail -100|awk '{a+=$1} END {if (NR != 0) printf("%.3f",a/NR)}'`
FPS=`awk 'BEGIN{printf "%.3f\n", '$batch_size'*'${num_npu}'/'$avg_time'}'`

#打印，不需要修改
echo "Final Performance images/sec : $FPS"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${num_npu}'p'_'acc'

#获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${BatchSize}'*'${num_npu}'*1000/'${FPS}'}'`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${num_npu}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
