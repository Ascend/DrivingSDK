#!/bin/bash

# test_path_dir 为包含 test 文件夹的路径
cur_path=`pwd`
cur_path_last_diename=${cur_path##*/}
if [ x"${cur_path_last_diename}" == x"test" ];then
    test_path_dir=${cur_path}
    # 若当前在 test 目录下，cd 到 test 同级目录
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi

source ${test_path_dir}/env_npu.sh

python train.py &
wait
