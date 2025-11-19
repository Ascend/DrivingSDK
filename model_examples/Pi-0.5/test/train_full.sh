# 网络名称,同目录名称,需要模型审视修改
Network="Pi05"
batch_size=64
world_size=16
log_interval=20

dataset_dir=$1
policy_dir=$2

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

source ${test_path_dir}/env_npu.sh

#创建DeviceID输出目录，不需要修改
output_path=${cur_path}/test/output/
mkdir -p ${output_path}

#训练开始时间，不需要修改
start_time=$(date +%s)
 
accelerate launch --num_processes=${world_size} $(which lerobot-train) \
  --dataset.repo_id=${dataset_dir} \
  --policy.type=act \
  --policy.push_to_hub=false \
  --policy.repo_id=${policy_dir} \
  --steps=300000 --log_freq=${log_interval} --save_freq=2000000 --batch_size=${batch_size} \
  --output_dir=outputs/train/$start_time > ${test_path_dir}/output/train_full_fp32.log 2>&1

wait
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(($end_time - $start_time))

cd ..

#结果打印，不需要修改
echo "------------------ Final result ------------------"

#获取性能数据，不需要修改
#单迭代训练时长，不需要修改
epoch_start_time=`grep 'ot_train.py:' ${test_path_dir}/output/train_full_fp32.log | grep 'loss' | grep 'step:20' | grep -o [0-9][0-9]:[0-9][0-9]:[0-9][0-9] | head -1`
epoch_end_time=`grep 'ot_train.py:' ${test_path_dir}/output/train_full_fp32.log | grep 'loss' | grep 'step:1k' | grep -o [0-9][0-9]:[0-9][0-9]:[0-9][0-9] | head -1`

TrainingTime=$(($(date +%s -d $epoch_end_time) - $(date +%s -d $epoch_start_time)))

#吞吐量
ActualFPS=$(awk BEGIN'{print ('$batch_size') * '$world_size' / ('$TrainingTime' / 980)}')


#loss值，不需要修改
ActualLoss=$(grep -o "loss:[0-9.]*" ${test_path_dir}/output/train_full_fp32.log | awk 'END {print $NF}')

#打印，不需要修改
echo "Final Train Loss : ${ActualLoss}"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
WORLD_SIZE=${world_size}
DeviceType=$(uname -m)
CaseName=${Network}_bs${BatchSize}_${WORLD_SIZE}'p'_'acc'

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >${test_path_dir}/output/${CaseName}.log
echo "RankSize = ${WORLD_SIZE}" >>${test_path_dir}/output/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>${test_path_dir}/output/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>${test_path_dir}/output/${CaseName}.log
echo "CaseName = ${CaseName}" >>${test_path_dir}/output/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >>${test_path_dir}/output/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >>${test_path_dir}/output/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >>${test_path_dir}/output/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>${test_path_dir}/output/${CaseName}.log