# 网络名称,同目录名称,需要模型审视修改
Network="PivotNet"
batch_size=1
world_size=8

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
output_path_dir=${test_path_dir}/output
if [ -d ${output_path_dir} ]; then
  rm -rf ${output_path_dir}
fi
mkdir -p ${output_path_dir}


###############开始训练###############
# 训练开始时间
cd PivotNet
start_time=$(date +%s)
echo "start_time=$(date -d @${start_time} "+%Y-%m-%d %H:%M:%S")"

bash run.sh train pivotnet_nuscenes_swint 1 > ${output_path_dir}/train_8p_performance.log 2>&1 &
wait

# 训练结束时间
end_time=$(date +%s)
echo "end_time=$(date -d @${end_time} "+%Y-%m-%d %H:%M:%S")"
e2e_time=$(( $end_time - $start_time ))

# 模型单epoch的step数量
total_step=3517

# 单迭代训练时长
avg_time=$(echo "scale=3; $e2e_time / $total_step" | bc | sed 's/^\./0./')

#吞吐量
ActualFPS=$(awk BEGIN'{print ('$batch_size' * '$world_size') / '$avg_time'}')

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
WORLD_SIZE=${world_size}
DeviceType=$(uname -m)
CaseName=${Network}_bs${BatchSize}_${WORLD_SIZE}'p'_'performance'

#关键信息打印到${CaseName}.log中，不需要修改
echo "------------------ Final result ------------------"
echo "Network: ${Network}"
echo "RankSize: ${WORLD_SIZE}"
echo "BatchSize: ${BatchSize}"
echo "DeviceType: ${DeviceType}"
echo "CaseName: ${CaseName}"
echo "E2E Training Duration sec: ${e2e_time}"
echo "Final Performance sec/iter: ${avg_time}"
echo "ActualFPS: ${ActualFPS}"

echo "Network: ${Network}" >>${test_path_dir}/output/${CaseName}.log
echo "RankSize: ${WORLD_SIZE}" >>${test_path_dir}/output/${CaseName}.log
echo "BatchSize: ${BatchSize}" >>${test_path_dir}/output/${CaseName}.log
echo "DeviceType: ${DeviceType}" >>${test_path_dir}/output/${CaseName}.log
echo "CaseName: ${CaseName}" >>${test_path_dir}/output/${CaseName}.log
echo "E2E Training Duration sec: ${e2e_time}" >>${test_path_dir}/output/${CaseName}.log
echo "Final Performance sec/iter: ${avg_time}" >>${test_path_dir}/output/${CaseName}.log
echo "ActualFPS: ${ActualFPS}" >>${test_path_dir}/output/${CaseName}.log