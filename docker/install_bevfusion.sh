#!/bin/bash

# ============================================================================
# BEVFusion Installation Script for Docker
# ============================================================================
# This script installs BEVFusion and its dependencies in a Python environment
#
# Usage:
#   bash install_bevfusion.sh
#
# Prerequisites:
#   - Python environment with PyTorch 2.7.1
#   - DrivingSDK already installed
# ============================================================================

set -e

echo "========================================"
echo "BEVFusion Installation"
echo "========================================"
echo ""

# Create workspace directory for BEVFusion
WORK_DIR=/workspace/bevfusion
mkdir -p $WORK_DIR
cd $WORK_DIR

# Copy BEVFusion installation files from model_examples
echo "1. Copying BEVFusion installation files..."
BEVFUSION_SRC="/tmp/BEVFusion"
if [ -d "$BEVFUSION_SRC" ]; then
    cp -r $BEVFUSION_SRC/* $WORK_DIR/
    echo "✓ Files copied from $BEVFUSION_SRC"
else
    echo "Error: BEVFusion directory not found at $BEVFUSION_SRC"
    echo "Please ensure DrivingSDK source code is copied to /workspace/DrivingSDK"
    exit 1
fi

# Install mmcv
echo ""
echo "2. Installing mmcv..."
if [ ! -d "mmcv" ]; then
    git clone -b main https://github.com/open-mmlab/mmcv.git
    cd mmcv
    pip install -r requirements/runtime.txt
    pip install ninja
    MMCV_WITH_OPS=1 MAX_JOBS=8 FORCE_NPU=1 python setup.py build_ext
    MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py develop
    cd ..
    echo "✓ mmcv installed"
else
    echo "mmcv already exists, skipping..."
fi

# Install mmdetection3d
echo ""
echo "3. Installing mmdetection3d..."
if [ ! -d "mmdetection3d" ]; then
    echo "mmdet3d开始安装"
    git clone -b v1.2.0 https://github.com/open-mmlab/mmdetection3d.git
    cp -f bevfusion.patch mmdetection3d/
    cd mmdetection3d
    git apply bevfusion.patch --reject
    pip install mmengine==0.10.7 mmdet==3.1.0 numpy==1.23.5 yapf
    pip install -e . --no-build-isolation
    cd ..
    echo "✓ mmdetection3d installed"
else
    echo "mmdetection3d already exists, skipping..."
fi

echo ""
echo "========================================"
echo "BEVFusion Installation Summary"
echo "========================================"
echo "Installation directory: $WORK_DIR"
echo "Installed components:"
echo "  - mmcv (with NPU support)"
echo "  - mmdetection3d"
echo "========================================"

