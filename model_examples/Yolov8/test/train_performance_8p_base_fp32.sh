#网络名称,同目录名称,需要模型审视修改
Network="Yolov8"
batch_size=256
epochs=1
RANK_SIZE=1

cur_path=`pwd`
for para in $*
do
   if [[ $para == --batch_size* ]];then
      	batch_size=`echo ${para#*=}`
   elif [[ $para == --data_path* ]];then
       data_path=`echo ${para#*=}`
   elif [[ $para == --epochs* ]];then
       epochs=`echo ${para#*=}`
   fi
done

ASCEND_DEVICE_ID=0, 1, 2, 3, 4, 5, 6, 7

#创建DeviceID输出目录，不需要修改
if [ -d ${cur_path}/test/output/${ASCEND_DEVICE_ID} ];
	then
	   rm -rf ${cur_path}/test/output/${ASCEND_DEVICE_ID}
		mkdir -p ${cur_path}/test/output/${ASCEND_DEVICE_ID}
	else
	   mkdir -p ${cur_path}/test/output/${ASCEND_DEVICE_ID}
	fi

#训练开始时间，不需要修改
start_time=$(date +%s)
echo "start_time: ${start_time}"

source ${cur_path}/test/env_npu.sh

python3 -u train.py --data ./ultralytics/cfg/datasets/coco.yaml \
                    --weights ./yolov8n.pt \
                    --device $ASCEND_DEVICE_ID \
                    --epochs $epochs > $cur_path/test/output/${ASCEND_DEVICE_ID}/train_full_8p.log 2>&1 &

wait

# #训练结束时间，不需要修改
end_time=$(date +%s)
echo "end_time: ${end_time}"
e2e_time=$(( $end_time - $start_time ))

# 计算FPS的平均值
total_FPS=`grep -oP 'FPS:\K\s*(\d+\.?\d*)' ${cur_path}/test/output/$ASCEND_DEVICE_ID/train_performance_8p.log | tail -n 78 | awk '{sum += $1} END {print sum}'`
averageFPS=$(echo "scale=2; $total_FPS/78" | bc)

# 取mAP50的值
mAP50=$(grep -w 'all' ${cur_path}/test/output/$ASCEND_DEVICE_ID/train_performance_8p.log | awk -F "all" '{print $2}' | awk -F "    " '{print $7}' |  tail -n1)
# 取mAP50-95的值
mAP50_95=$(grep -w 'all' ${cur_path}/test/output/$ASCEND_DEVICE_ID/train_performance_8p.log | awk -F "all" '{print $2}' | awk -F "    " '{print $8}' |  tail -n1)

#打印，不需要修改
echo "Average Performance images/sec : $averageFPS"
echo "mAP50 : $mAP50"
echo "mAP50-95 : $mAP50_95"
echo "E2E Training Duration sec : $e2e_time"

#稳定性精度看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

##获取性能数据，不需要修改
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${averageFPS}'}'`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "averageFPS = ${averageFPS}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "mAP50 = ${mAP50}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "mAP50-95 = ${mAP50_95}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log