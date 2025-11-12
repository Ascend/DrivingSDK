# 网络名称,同目录名称,需要模型审视修改
Network="DinoV3"
world_size=8
batch_size=64
num_step=10000

data_path=$1
extra_path=$2
output_dir=$3

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
torchrun --nproc_per_node=$world_size  --nnodes=1  --node-rank=0 --master_addr=127.0.0.1  --master_port=29501  dinov3/train/train.py --config-file dinov3/configs/train/vitl_im1k_lin834_performance.yaml \
      	--output-dir $output_dir train.dataset_path=ImageNet:split=TRAIN:root=$data_path:extra=$extra_path \
	> ${test_path_dir}/output/train_performance_8p_base_fp32.log 2>&1 &
wait
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(($end_time - $start_time))

cd ..

#结果打印，不需要修改
echo "------------------ Final result ------------------"

#获取性能数据，不需要修改
#吞吐量
avg_time=$(grep ")  time:" ${test_path_dir}/output/train_performance_8p_base_fp32.log | tail -5 | awk -F'time: ' '{print $2}' | awk '{print $1}' | awk '{sum+=$1} END {print sum/NR}')
ActualFPS=$(awk BEGIN'{print ('$world_size' * '$batch_size') / '$avg_time'}')
#打印，不需要修改
echo "Final Performance images/sec : $ActualFPS"

#loss值，不需要修改
ActualLoss=$(grep -o "total_loss: [0-9.]*" ${test_path_dir}/output/train_performance_8p_base_fp32.log | awk 'END {print $NF}')

#打印，不需要修改
echo "Final Train Loss : ${ActualLoss}"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
WORLD_SIZE=${world_size}
DeviceType=$(uname -m)
CaseName=${Network}_bs${BatchSize}_${WORLD_SIZE}'p'_'perf'

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >${test_path_dir}/output/${CaseName}.log
echo "RankSize = ${WORLD_SIZE}" >>${test_path_dir}/output/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>${test_path_dir}/output/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>${test_path_dir}/output/${CaseName}.log
echo "CaseName = ${CaseName}" >>${test_path_dir}/output/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >>${test_path_dir}/output/${CaseName}.log
echo "TrainingTime = ${avg_time}" >>${test_path_dir}/output/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >>${test_path_dir}/output/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>${test_path_dir}/output/${CaseName}.log
