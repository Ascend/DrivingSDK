#!/bin/bash
git clone https://github.com/fundamentalvision/Deformable-DETR.git
cp -f Deformable-DETR_npu.patch Deformable-DETR
cd Deformable-DETR
git apply Deformable-DETR_npu.patch