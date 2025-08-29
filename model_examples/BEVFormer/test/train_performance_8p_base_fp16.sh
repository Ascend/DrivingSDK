# 网络名称,同目录名称,需要模型审视修改
Network="BEVFormer_Base"

# 初始化参数
batch_size=1
world_size=8

# 帮助信息
if [[ $1 == --help || $1 == -h ]];then
    echo "usage: ./train_*.sh <args>"
    echo " "
    echo "parameter explain:
    --batch-size               specify batch size
    --num-npu                  specify the number of NPUs
    -h/--help                  show help message
    "
    exit 1
fi

# 参数校验
for para in "$@"
do
    if [[ $para == --batch-size* ]];then
        batch_size=$(echo "${para#*=}")
    elif [[ $para == --num-npu* ]];then
        world_size=$(echo "${para#*=}")
    fi
done

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

if [ -d ${output_path} ]; then
  rm -rf ${output_path}
fi

mkdir -p ${output_path}
cd BEVFormer
config=projects/configs/bevformer_fp16/bevformer_base_fp16.py

# 备份config文件
cp ${config} ${config}.bak

sed -i "253s/interval=50/interval=1/g" projects/configs/bevformer_fp16/bevformer_base_fp16.py
sed -i "200s|samples_per_gpu=1|samples_per_gpu=$batch_size|g" projects/configs/bevformer_fp16/bevformer_base_fp16.py
sed -i "s|total_epochs = 24|total_epochs = 1|g" projects/configs/bevformer_fp16/bevformer_base_fp16.py
sed -i "7s/^/#/" ./projects/mmdet3d_plugin/bevformer/detectors/bevformer_fp16.py

# 定义复原config文件的callback
restore_config() { 
    if [ -f ${config}.bak ]; then
        mv -f ${config}.bak ${config}
    fi
}

# 设置信号捕获，如果训练
#   正常退出（EXIT）
#   用户中断（SIGINT）
#   Kill终止请求（SIGTERM）
#   命令执行失败（ERR）
# 可以自动还原对config文件的修改
trap restore_config EXIT SIGINT SIGTERM ERR

#训练开始时间，不需要修改
start_time=$(date +%s)

bash ./tools/fp16/dist_train.sh ./projects/configs/bevformer_fp16/bevformer_base_fp16.py ${world_size} > ${test_path_dir}/output/train_performance_8p_base_fp16.log 2>&1 &

wait
sed -i "253s/interval=1/interval=50/g" projects/configs/bevformer_fp16/bevformer_base_fp16.py
sed -i "200s|samples_per_gpu=$batch_size|samples_per_gpu=1|g" projects/configs/bevformer_fp16/bevformer_base_fp16.py
sed -i "s|total_epochs = 1|total_epochs = 24|g" projects/configs/bevformer_fp16/bevformer_base_fp16.py

cd ..
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(($end_time - $start_time))

#结果打印，不需要修改
echo "------------------ Final result ------------------"

#获取性能数据，不需要修改
#单迭代训练时长，不需要修改
TrainingTime=$(grep -o ", time: [0-9.]*" ${test_path_dir}/output/train_performance_8p_base_fp16.log | tail -n 1 | grep -o "[0-9.]*" | awk '{sum += $1} END {print sum/NR}')

#吞吐量
ActualFPS=$(awk BEGIN'{print ('$batch_size' * '$world_size') / '$TrainingTime'}')

#打印，不需要修改
echo "Final Performance images/sec : $ActualFPS"

#loss值，不需要修改
ActualLoss=$(grep -o "loss: [0-9.]*" ${test_path_dir}/output/train_performance_8p_base_fp16.log | awk 'END {print $NF}')

#NDS值
NDS=$(grep -o "pts_bbox_NuScenes/NDS: [0-9.]*" ${test_path_dir}/output/train_performance_8p_base_fp16.log | awk 'END {print $NF}')

#mAP值
mAP=$(grep -o "pts_bbox_NuScenes/mAP: [0-9.]*" ${test_path_dir}/output/train_performance_8p_base_fp16.log | awk 'END {print $NF}')

#打印，不需要修改
echo "Final Train Loss : ${ActualLoss}"
echo "NDS : ${NDS}"
echo "mAP : ${mAP}"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
WORLD_SIZE=${world_size}
DeviceType=$(uname -m)
CaseName=${Network}_bs${BatchSize}_${WORLD_SIZE}'p'_'performance'

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >${test_path_dir}/output/${CaseName}.log
echo "RankSize = ${WORLD_SIZE}" >>${test_path_dir}/output/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>${test_path_dir}/output/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>${test_path_dir}/output/${CaseName}.log
echo "CaseName = ${CaseName}" >>${test_path_dir}/output/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >>${test_path_dir}/output/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >>${test_path_dir}/output/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >>${test_path_dir}/output/${CaseName}.log
echo "NDS = ${NDS}" >>${test_path_dir}/output/${CaseName}.log
echo "mAP = ${mAP}" >>${test_path_dir}/output/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>${test_path_dir}/output/${CaseName}.log