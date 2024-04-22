#!/usr/bin/bash

repo_host="https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN"
download_file="ai_cann_arm.tar.gz"
folder_name="ai_cann_arm"
rm -rf ${folder_name}
get_day() {
  local day=$(date +%Y%m%d)
  day_branch="${day}_newest"
}

get_day
download_url="${repo_host}/${day_branch}/${download_file}"
curl -O ${download_url}
if [[ $? -ne 0 ]]; then
  echo "Failed to download ${download_url}"
  exit 1
fi
tar -xzvf ${download_file}
rm -f ${download_file}
cd ${folder_name}
chmod +x *.run
## loop through the packages ending in .run and run them with the --full and --quiet flags
for install_file in $(ls *.run)
do
  ./${install_file} --full --install-path=/usr/local/Ascend --quiet
done
