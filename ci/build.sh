# Copyright 2023 Huawei Technologies Co., Ltd
CUR_DIR=$(dirname $(readlink -f $0))
SUPPORTED_PY_VERSION=(3.7 3.8 3.9 3.10)
PY_VERSION='3.7'
DEFAULT_SCRIPT_ARGS_NUM=1

function check_python_version() {
    matched_py_version='false'
    for ver in ${SUPPORTED_PY_VERSION[*]}; do
        if [ "${PY_VERSION}" = "${ver}" ]; then
            matched_py_version='true'
            return 0
        fi
    done
    if [ "${matched_py_version}" = 'false' ]; then
        echo "${PY_VERSION} is an unsupported python version, we suggest ${SUPPORTED_PY_VERSION[*]}"
        exit 1
    fi
}

function parse_script_args() {
    local args_num=0
    if [[ "x${1}" = "x" ]]; then
        # default: bash build.sh (python3.7)
        return 0
    fi

    while true; do
        if [[ "x${1}" = "x" ]]; then
            break
        fi
        if [[ "$(echo "${1}"|cut -b1-|cut -b-2)" == "--" ]]; then
            args_num=$((args_num+1))
        fi
        if [[ ${args_num} -eq ${DEFAULT_SCRIPT_ARGS_NUM} ]]; then
            break
        fi
        shift
    done

    # if num of args are not fully parsed, throw an error.
    if [[ ${args_num} -lt ${DEFAULT_SCRIPT_ARGS_NUM} ]]; then
        return 1
    fi

    while true; do
        case "${1}" in
        --python=*)
            PY_VERSION=$(echo "${1}"|cut -d"=" -f2)
            args_num=$((args_num-1))
            shift
            ;;
        --tocpu=*)
            export 'NPU_TOCPU'=${1:8}
            args_num=$((args_num-1))
            shift
            ;;
        -*)
            echo "ERROR Unsupported parameters: ${1}"
            return 1
            ;;
        *)
            if [ "x${1}" != "x" ]; then
                echo "ERROR Unsupported parameters: ${1}"
                return 1
            fi
            break
            ;;
        esac
    done

    # if some "--param=value" are not parsed correctly, throw an error.
    if [[ ${args_num} -ne 0 ]]; then
        return 1
    fi
}

function main()
{
    if [ -z "$ASCEND_OPP_PATH" ]; then
        echo "ImportError: libhccl.so: cannot open shared object file: No such file or directory. Please check that the cann package is installed. Please run 'source set_env.sh' in the CANN installation path."
        exit 1
    else
        echo "ASCEND_OPP_PATH = $ASCEND_OPP_PATH"
    fi
    chmod -R 777 ${CUR_DIR}/..
    cd ${CUR_DIR}/../ads/common/ops/kernels/ads_op
    bash build.sh
    cd ${CUR_DIR}/../ads/common/ops/kernels/ads_op/build_out
    cp custom_opp_*.run ${CUR_DIR}/../ads/common/ops/kernels
    cd ..
    rm -rf build_out
    cd ..
    
    
    ./custom_opp_*.run --extract=./ads_op_kernel
    rm -rf custom_opp_*.run
    if [ -d "ads_op_kernel/" ]; then
        echo "kernel compile success"
    else
        echo "kernel did not compile success"
        exit 1
    fi
    cd ${CUR_DIR}/../ads/common/ops/kernels/ads_op_kernel
    rm -rf install.sh
    rm -rf upgrade.sh
    
    cd ${CUR_DIR}/..
    rm -rf build
    if [ -d "ads.egg-info" ]; then
        echo "ads.egg-info exist"
        rm -rf ads.egg-info
    else
        echo "ads.egg-info not exist"
    fi

    if ! parse_script_args "$@"; then
        echo "Failed to parse script args. Please check your inputs."
        exit 1
    fi

    check_python_version

    cd ${CUR_DIR}/..
    python"${PY_VERSION}" setup.py build bdist_wheel
    if [ $? != 0 ]; then
        echo "Failed to compile the wheel file. Please check the source code by yourself."
        exit 1
    fi

    exit 0
}

main "$@"
