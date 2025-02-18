#！/bin/bash

#设置device侧日志登记为error
msnpureport -g error -d 0
msnpureport -g error -d 1
msnpureport -g error -d 2
msnpureport -g error -d 3
msnpureport -g error -d 4
msnpureport -g error -d 5
msnpureport -g error -d 6
msnpureport -g error -d 7
# #关闭Device侧Event日志
msnpureport -e disable

export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_GLOBAL_EVENT_ENABLE=0
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export TASK_QUEUE_ENABLE=2
export BMMV2_ENABLE=1
export COMBINED_ENABLE=1
export PTCOPY_ENABLE=1
export HCCL_WHITELIST_DISABLE=1
export SCALAR_TO_HOST_MEM=1
export HCCL_CONNECT_TIMEOUT=1200
export TORCH_HCCL_ENABLE_MONITORING=0