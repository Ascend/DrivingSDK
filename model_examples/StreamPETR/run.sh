#!/bin/bash
git clone https://github.com/exiawsh/StreamPETR.git
cp -f streampetr.patch StreamPETR
cd StreamPETR
git checkout 95f64702306ccdb7a78889578b2a55b5deb35b2a
git apply streampetr.patch