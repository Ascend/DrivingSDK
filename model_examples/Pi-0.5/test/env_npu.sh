#!/bin/bash

#将Host日志输出到串口,0-关闭/1-开启
export ASCEND_SLOG_PRINT_TO_STDOUT=0
#设置默认日志级别,0-debug/1-info/2-warning/3-error
export ASCEND_GLOBAL_LOG_LEVEL=3
#设置Event日志开启标志,0-关闭/1-开启
export ASCEND_GLOBAL_EVENT_ENABLE=0
# 设置是否开启taskque,0-关闭/1-开启/2-流水优化
export TASK_QUEUE_ENABLE=2
# 设置是否开启均匀绑核,0-关闭/1-开启粗粒度绑核/2-开启细粒度绑核
export CPU_AFFINITY_CONF=2
# 设置是否开启 combined 标志, 0-关闭/1-开启
export COMBINED_ENABLE=1

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export DYNAMIC_OP="ADD#MUL"
export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE
export HCCL_WHITELIST_DISABLE=1
export MULTI_STREAM_MEMORY_REUSE=1

#关闭Device侧Event日志
msnpureport -e disable