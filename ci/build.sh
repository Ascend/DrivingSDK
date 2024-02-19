# Copyright 2023 Huawei Technologies Co., Ltd
CUR_DIR=$(dirname $(readlink -f $0))
SCRIPTS_DIR=${CUR_DIR}/../scripts
BUILD_PACKAGES_DIR=${CUR_DIR}/../build_out/packages
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
    chmod -R 777 ${SCRIPTS_DIR}
    bash ${SCRIPTS_DIR}/build_kernel.sh

    cd ${CUR_DIR}/..
    rm -rf build
    if [ -d "ads_accelerator.egg-info" ]; then
        echo "ads_accelerator.egg-info exist"
        rm -rf ads_accelerator.egg-info
    else
        echo "ads_accelerator.egg-info not exist"
    fi

    if ! parse_script_args "$@"; then
        echo "Failed to parse script args. Please check your inputs."
        exit 1
    fi

    check_python_version

    python"${PY_VERSION}" setup.py build bdist_wheel
    if [ $? != 0 ]; then
        echo "Failed to compile the wheel file. Please check the source code by yourself."
        exit 1
    fi

    exit 0
}

main "$@"
