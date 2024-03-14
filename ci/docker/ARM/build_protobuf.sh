#!/usr/bin/bash
yum install -y autoconf automake libtool curl make g++ unzip
git clone -b 3.14.x https://gitee.com/it-monkey/protocolbuffers.git
cd protocolbuffers
./autogen.sh
./configure
make -j$(nproc)
make install
ldconfig