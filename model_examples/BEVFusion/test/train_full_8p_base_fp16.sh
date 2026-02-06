#!/bin/bash
# 网络名称,同目录名称,需要模型审视修改
Network="BEVFusion"
batch_size=4
num_npu=8
epochs=6

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

#避免torch2.7权重加载报错，设置weights_only=False
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# FP16下的默认优化
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export BATCH_NORM_PATCH=1
export SUBM_FP16_ENABLED=1

# 解析参数
source ${test_path_dir}/parse_args.sh
declare_required_params batch_size num_npu # 接收参数顺序
parse_common_args "$@"

base_batch_size=$(($batch_size * $num_npu))

#创建DeviceID输出目录，不需要修改
output_path=${cur_path}/test/output/
log_name="train_full_${num_npu}p_base_fp16.log"
mkdir -p ${output_path}

for para in $*
do
    if [[ $para == --epochs* ]];then
        epochs=`echo ${para#*=}`
    fi
done

cd mmdetection3d

#训练开始时间，不需要修改
start_time=$(date +%s)
#设置amp参数以打开混精
bash tools/dist_train.sh \
    projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py ${num_npu} \
    --cfg-options train_dataloader.batch_size=${batch_size} auto_scale_lr.base_batch_size=${base_batch_size} \
    load_from=pretrained/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth model.img_backbone.init_cfg.checkpoint=pretrained/swint-nuimages-pretrained.pth --amp \
    > ${test_path_dir}/output/${log_name} 2>&1 &

wait
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(($end_time - $start_time))

cd ..

#结果打印，不需要修改
echo "------------------ Final result ------------------"

#获取性能数据，不需要修改
#单迭代训练时长，不需要修改
TrainingTime=$(grep -v val ${test_path_dir}/output/${log_name} | grep -o " time: [0-9.]*"  | tail -n +200 | grep -o "[0-9.]*" | awk '{sum += $1} END {print sum/NR}')

#吞吐量
ActualFPS=$(awk BEGIN'{print ('$batch_size' * '$num_npu') / '$TrainingTime'}')

#打印，不需要修改
echo "Final Performance images/sec : $ActualFPS"

#loss值，不需要修改
ActualLoss=$(grep -o "loss: [0-9.]*" ${test_path_dir}/output/${log_name} | awk 'END {print $NF}')

#NDS值
NDS=$(grep -o "pred_instances_3d_NuScenes/NDS: [0-9.]*" ${test_path_dir}/output/${log_name} | awk 'END {print $NF}')

#mAP值
mAP=$(grep -o "pred_instances_3d_NuScenes/mAP: [0-9.]*" ${test_path_dir}/output/${log_name} | awk 'END {print $NF}')

#打印，不需要修改
echo "Final Train Loss : ${ActualLoss}"
echo "NDS : ${NDS}"
echo "mAP : ${mAP}"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
WORLD_SIZE=${num_npu}
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
echo "NDS = ${NDS}" >>${test_path_dir}/output/${CaseName}.log
echo "mAP = ${mAP}" >>${test_path_dir}/output/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>${test_path_dir}/output/${CaseName}.log