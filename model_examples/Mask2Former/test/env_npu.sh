#设置Shape数据缓存，默认值为0，配置为非零正整数N时，系统会缓存N个频繁出现的Shape
export HOST_CACHE_CAPACITY=20
#将Host日志输出到串口,0-关闭/1-开启
export ASCEND_SLOG_PRINT_TO_STDOUT=0
#设置默认日志级别,0-debug/1-info/2-warning/3-error
export ASCEND_GLOBAL_LOG_LEVEL=3
#设置Event日志开启标志,0-关闭/1-开启
export ASCEND_GLOBAL_EVENT_ENABLE=0
#设置是否开启taskque,0-关闭/1-开启/2-流水优化
export TASK_QUEUE_ENABLE=2
#设置是否开启combined标志,0-关闭/1-开启
export COMBINED_ENABLE=1
#设置是否开启均匀绑核,0-关闭/1-开启粗粒度绑核/2-开启细粒度绑核
export CPU_AFFINITY_CONF=1
#设置特殊场景是否需要重新编译,不需要修改
export DYNAMIC_OP="ADD#MUL"
#HCCL白名单开关,1-关闭/0-开启
export HCCL_WHITELIST_DISABLE=1
#配置HCCL的初始化root通信网卡IP
export HCCL_IF_IP=$(hostname -I |awk '{print $1}')
#配置不同设备之间socket建链过程的等待时间，取值范围[120, 7200]，默认120，单位s
export HCCL_CONNECT_TIMEOUT=1200

# config PYTHONPATH
CURRENT_DIR=$(cd $(dirname "${BASH_SOURCE[0]}"); pwd)
ROOT_DIR=$(readlink -f "${CURRENT_DIR}/../mmsegmentation/")
echo $ROOT_DIR
export PYTHONPATH=$ROOT_DIR:$PYTHONPATH

#设置device侧日志登记为error
msnpureport -g error -d 0
msnpureport -g error -d 1
msnpureport -g error -d 2
msnpureport -g error -d 3
msnpureport -g error -d 4
msnpureport -g error -d 5
msnpureport -g error -d 6
msnpureport -g error -d 7
#关闭Device侧Event日志
msnpureport -e disable
