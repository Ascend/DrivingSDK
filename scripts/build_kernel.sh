#!/bin/bash
script_path=$(realpath $(dirname $0))
root_path=$(realpath $script_path/..)
rm -rf build_out
mkdir build_out
cd build_out
SINGLE_OP=""
BUILD_TYPE="Release"

function parse_script_args() {
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --single_op=*)
                SINGLE_OP="${1#*=}"
                shift 1
                ;;
            --build_type=*)
                BUILD_TYPE="${1#*=}"
                shift 1
                ;;
            *)
              echo "Usage: $0 --single_op=xxx --build_type=xxx"
              return 1
            ;;
        esac
    done
    return 0
}

parse_script_args $@

cmake_version=$(cmake --version | grep "cmake version" | awk '{print $3}')
if [ "$cmake_version" \< "3.19.0" ]; then
  opts=$(python3 $root_path/cmake/util/preset_parse.py $root_path/CMakePresets.json)
  echo $opts
  cmake .. $opts -DSINGLE_OP=$SINGLE_OP -DCMAKE_BUILD_TYPE=$BUILD_TYPE
else
  cmake .. --preset=default -DSINGLE_OP=$SINGLE_OP -DCMAKE_BUILD_TYPE=$BUILD_TYPE
fi

cmake --build . -j16
if [ $? -ne 0 ]; then exit 1; fi
