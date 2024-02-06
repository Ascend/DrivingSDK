#!/bin/bash
script_path=$(realpath $(dirname $0))


mkdir -p build_out
rm -rf build_out/*
cd build_out

if [ $ASCEND_AICPU_PATH ]; then
    jq --arg field "configurePresets" --arg value "$ASCEND_AICPU_PATH" '.[$field][0].cacheVariables.ASCEND_CANN_PACKAGE_PATH.value = $value' $script_path/CMakePresets.json > $script_path/CMakePresets_bat.json

    if [ $? -eq 0 ]; then
        mv $script_path/CMakePresets_bat.json $script_path/CMakePresets.json -f
    else
        echo "Error: please install jq with yum or apt-get"
        exit 1
    fi
else
    echo "Error: please source env.sh"
    exit 1
fi

cmake_version=$(cmake --version | grep "cmake version" | awk '{print $3}')
if [ "$cmake_version" \< "3.19.0" ] ; then
    opts=$(python3 $script_path/cmake/util/preset_parse.py $script_path/CMakePresets.json)
    echo $opts
    cmake .. $opts
else
    cmake .. --preset=default
fi
target=package
if [ "$1"x != ""x ]; then target=$1; fi

cmake --build . --target $target -j16
if [ $? -ne 0 ]; then exit 1; fi

if [ $target = "package" ]; then
  if test -d ./op_kernel/binary ; then
    ./cust*.run
    if [ $? -ne 0 ]; then exit 1; fi
    cmake --build . --target binary -j16
    if [ $? -ne 0 ]; then exit 1; fi
    cmake --build . --target $target -j16
  fi
fi

# for debug
# cd build_out
# make
# cpack
# verbose append -v
