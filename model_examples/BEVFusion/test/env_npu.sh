#!/bin/bash
# 使能内存池扩展段功能，由PyTorch管理虚拟地址和物理地址的映射关系，降低内存碎片化
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# 增加ACLNN算子缓存数量上限
export ACLNN_CACHE_LIMIT=100000
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
export CPU_AFFINITY_CONF=2
#HCCL白名单开关,1-关闭/0-开启
export HCCL_WHITELIST_DISABLE=1
#配置HCCL的初始化root通信网卡IP
export HCCL_IF_IP=$(hostname -I |awk '{print $1}')
#配置不同设备之间socket建链过程的等待时间，取值范围[120, 7200]，默认120，单位s
export HCCL_CONNECT_TIMEOUT=1200
#模型不提前初始化，默认为1
export ACL_OP_INIT_MODE=0

path_lib=$(python3 -c """
import sys
import re
result=''
for index in range(len(sys.path)):
    match_sit = re.search('-packages', sys.path[index])
    if match_sit is not None:
        match_lib = re.search('lib', sys.path[index])

        if match_lib is not None:
            end=match_lib.span()[1]
            result += sys.path[index][0:end] + ':'

        result+=sys.path[index] + '/torch/lib:'
print(result)"""
)

echo ${path_lib}
