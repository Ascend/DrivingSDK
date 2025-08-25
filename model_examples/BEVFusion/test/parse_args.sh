#!/bin/bash
# ================================
# 公共参数解析器（支持默认值 + 关键字 + 位置参数）
# ================================

# 用法：declare_required_params batch_size npus
declare_required_params() {
    local params=("$@")
    export REQUIRED_PARAMS=("${params[@]}")
}

# 补全默认值
setup_defaults() {
    local param
    for param in "${REQUIRED_PARAMS[@]}"; do
        eval "$param=\${$param}"
    done
}

# 帮助函数
show_help() {
    cat << EOF
用法: $0 [参数] 或 $0 [选项]
示例：
  a.sh 4 8
  a.sh --batch-size=4 --npus=8
  b.sh 4 8 --nnodes=2 --node-rank=1 --port=30000 --master_addr=192.168.1.100
EOF
}

# 解析命令行参数（支持 --key=value 和 位置参数）
parse_common_args() {
    # Step 1: 确保 REQUIRED_PARAMS 已定义
    if [[ -z "${REQUIRED_PARAMS+set}" ]]; then
        echo "错误：未调用 declare_required_params，请先声明所需参数！" >&2
        exit 1
    fi

    # Step 2: 补全默认值
    setup_defaults

    # Step 3: 提取命令行参数
    local args=("$@")
    local pos_args=()
    local key_value_args=()

    for arg in "${args[@]}"; do
        if [[ "$arg" == --* ]]; then
            key_value_args+=("$arg") # 提取长选项参数（-- 开头的）
        else
            pos_args+=("$arg") # 提取位置参数
        fi
    done

    # Step 4: 处理位置参数（按 REQUIRED_PARAMS 顺序赋值）
    local idx=0
    declare -A required_param_map
    for param in "${REQUIRED_PARAMS[@]}"; do
        opt_name="${param//_/-}"
        required_param_map["$opt_name"]="$param"
        if [[ $idx -lt ${#pos_args[@]} ]] && [[ -n "${pos_args[$idx]}" ]]; then
            eval "$param=\"\${pos_args[$idx]}\""
        fi
        ((idx++))
    done

    # Step 5: 处理长选项（所有 key_value_args 都是 --param=value 格式）
    for arg in "${key_value_args[@]}"; do
        # 跳过空值
        [[ -z "$arg" ]] && continue

        # 提取 --param=value 中的 param 和 value
        local param_key="${arg#--}"           # 去掉 --
        local param_name="${param_key%%=*}"   # 如 batch-size
        local value="${param_key#*=}"         # 如 8

        # 查表：查找 param_name 是否在映射表中
        if [[ -n "${required_param_map[$param_name]}" ]]; then
            local var_name="${required_param_map[$param_name]}"
            eval "$var_name=\"\$value\""
        else
            echo "未知参数: $arg" >&2
            exit 1
        fi
    done

    # Step 6: 检查必填参数是否都提供了
    for param in "${REQUIRED_PARAMS[@]}"; do
        if [[ -z "${!param}" ]]; then
            echo "缺少必要参数: --${param//-/_}=<value>" >&2
            show_help
            exit 1
        fi
    done

    # Step 7: 成功提示
    echo "参数解析成功！"
    for param in "${REQUIRED_PARAMS[@]}"; do
        local name=$(echo "$param" | cut -d'=' -f1)
        echo "$name: ${!name}"
    done
}