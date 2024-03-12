#!/bin/bash
script_path=$(realpath $(dirname $0))
root_path=$(realpath $script_path/..)
rm -rf build_out
mkdir build_out
cd build_out

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
