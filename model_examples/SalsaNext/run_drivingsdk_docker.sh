#!/usr/bin/bash

# 需要保证宿主机已经安装好了昇腾驱动，并将/usr/local/Ascend/driver挂载到容器中
# 容器中自带CANN包，位于/usr/local/Ascend/ascend-toolkit路径下
# 镜像标签与镜像版本一致，例如7.0.RC1
TAG=$1

docker run -it --ipc=host \
--network=host \
--privileged -u=root \
--device=/dev/davinci0 \
--device=/dev/davinci1 \
--device=/dev/davinci2 \
--device=/dev/davinci3 \
--device=/dev/davinci4 \
--device=/dev/davinci5 \
--device=/dev/davinci6 \
--device=/dev/davinci7 \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/bin/hccn_tool:/usr/bin/hccn_tool \
drivingsdk:${TAG} \
/bin/bash