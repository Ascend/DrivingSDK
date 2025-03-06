#!/bin/bash
git clone https://github.com/hustvl/Senna.git
cp -f Senna.patch Senna
cd Senna
git checkout 5f202ce84dc4fe52949934ab0921e287d733ff8f
git apply Senna.patch