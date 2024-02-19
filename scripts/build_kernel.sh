#!/bin/bash
script_path=$(realpath $(dirname $0))
root_path=$(realpath $script_path/..)
rm -rf build_out
mkdir build_out
cd build_out

if [ $ASCEND_AICPU_PATH ]; then
  jq --arg field "configurePresets" --arg value "$ASCEND_AICPU_PATH" '.[$field][0].cacheVariables.ASCEND_CANN_PACKAGE_PATH.value = $value' $root_path/CMakePresets.json >$root_path/CMakePresets_bat.json

  if [ $? -eq 0 ]; then
    mv $root_path/CMakePresets_bat.json $root_path/CMakePresets.json -f
  else
    echo "Error: please install jq with yum or apt-get"
    exit 1
  fi
else
  echo "Error: please source env.sh"
  exit 1
fi

cmake_version=$(cmake --version | grep "cmake version" | awk '{print $3}')
if [ "$cmake_version" \< "3.19.0" ]; then
  opts=$(python3 $root_path/cmake/util/preset_parse.py $root_path/CMakePresets.json)
  echo $opts
  cmake .. $opts
else
  cmake .. --preset=default
fi

cmake --build . -j16
if [ $? -ne 0 ]; then exit 1; fi
