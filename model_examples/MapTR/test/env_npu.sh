#!/bin/bash
CANN_INSTALL_PATH_CONF='/etc/Ascend/ascend_cann_install.info'

if [ -f CANN_INSTALL_PATH_CONF ]; then
    CANN_INSTALL_PATH=$(cat $CANN_INSTALL_PATH_CONF | grep Install_Path | cut -d "=" -f 2)
else
    CANN_INSTALL_PATH="/usr/local/Ascend"
fi

if [ -d {CANN_INSTALL_PATH}/ascend-toolkit/latest ]; then
    source ${CANN_INSTALL_PATH}/ascend-toolkit/set_env.sh
else
    source ${CANN_INSTALL_PATH}/nnae/set_env.sh
fi

# 设置是否开启均匀绑核,0-关闭/1-开启粗粒度绑核/2-开启细粒度绑核
export CPU_AFFINITY_CONF=1
# 设置是否开启taskque,0-关闭/1-开启/2-流水优化
export TASK_QUEUE_ENABLE=2
# 设置是否开启combined标志,0-关闭/1-开启
export COMBINED_ENABLE=1
# 启用可扩展内存段分配策略
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# 启用多流复用，降低集合通信跨流依赖内存延迟释放带来的内存峰值上升
export MULTI_STREAM_MEMORY_REUSE=1
#配置初始化模式
export ACL_OP_INIT_MODE=0

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
