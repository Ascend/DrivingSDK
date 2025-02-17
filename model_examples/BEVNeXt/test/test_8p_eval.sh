#!/bin/bash
# 模型配置
export RANK_SIZE=8
epochs=1
stage2_ckpts_path="work_dirs/bevnext-stage2"

# 获取传入的参数，重新赋值对应参数
for para in $*
do
    if [[ $para == --stage2_ckpts_path* ]];then
        stage2_ckpts_path=`echo ${para#*=}`
    fi
done

# cd到与test文件夹同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
cur_path=`pwd`
cur_path_last_diename=${cur_path##*/}
if [ x"${cur_path_last_diename}" == x"test" ];then
    test_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi

source ${test_path_dir}/env_npu.sh

echo "[BEVNeXt] Evaluating..."
bash tools/dist_test.sh $stage2_ckpts_path/*.py $stage2_ckpts_path/epoch_${epochs}_ema.pth ${RANK_SIZE} --eval mAP --no-aavt > ${stage2_ckpts_path}/bevnext_eval_epoch${epochs}.log 2>&1 &
wait

NDS=$(grep -o "NDS: [0-9.]*" ${stage2_ckpts_path}/bevnext_eval_epoch${epochs}.log | awk 'END {print $NF}')
mAP=$(grep -o "mAP: [0-9.]*" ${stage2_ckpts_path}/bevnext_eval_epoch${epochs}.log | awk 'END {print $NF}')
echo "[BEVNeXt] NDS : ${NDS}"
echo "[BEVNeXt] mAP : ${mAP}"