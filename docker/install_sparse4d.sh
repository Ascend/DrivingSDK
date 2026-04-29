#!/bin/bash

# ============================================================================
# Sparse4D Installation Script for Docker
# ============================================================================
# This script installs Sparse4D and its dependencies in a Python environment
#
# Usage:
#   bash install_sparse4d.sh
#
# Prerequisites:
#   - Python environment with PyTorch 2.7.1
#   - DrivingSDK already installed
# ============================================================================

set -e

echo "========================================"
echo "Sparse4D Installation"
echo "========================================"
echo ""

# Create workspace directory for Sparse4D
WORK_DIR=/workspace/sparse4d
mkdir -p $WORK_DIR
cd $WORK_DIR

# Copy Sparse4D installation files from model_examples
echo "1. Copying Sparse4D installation files..."
SPARSE4D_SRC="/tmp/Sparse4D"
if [ -d "$SPARSE4D_SRC" ]; then
    cp -r $SPARSE4D_SRC/* $WORK_DIR/
    echo "✓ Files copied from $SPARSE4D_SRC"
else
    echo "Error: Sparse4D directory not found at $SPARSE4D_SRC"
    echo "Please ensure DrivingSDK source code is copied to /workspace/DrivingSDK"
    exit 1
fi

# Install basic dependencies
echo ""
echo "2. Installing basic dependencies..."
pip install -r requirements_pytorch2.7.1.txt
echo "✓ Basic dependencies installed"

# Install mmcv
echo ""
echo "3. Installing mmcv..."
if [ ! -d "mmcv" ]; then
    git clone -b 1.x https://github.com/open-mmlab/mmcv.git
    cp mmcv.patch mmcv
    cd mmcv
    git apply mmcv.patch
    pip install -r requirements/runtime.txt
    pip install ninja
    MMCV_WITH_OPS=1 MAX_JOBS=8 FORCE_NPU=1 python setup.py build_ext
    MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py develop
    cd ..
    echo "✓ mmcv installed"
else
    echo "mmcv already exists, skipping..."
fi

# Install mmdetection
echo ""
echo "4. Installing mmdetection3d..."
if [ ! -d "mmdetection" ]; then
    echo "mmdet开始安装"
    git clone -b v2.28.2 https://github.com/open-mmlab/mmdetection.git
    cp mmdet.patch mmdetection
    cd mmdetection
    git apply mmdet.patch
    pip install -e . --no-build-isolation
    cd ..
    echo "✓ mmdetection3d installed"
else
    echo "mmdetection3d already exists, skipping..."
fi

# Install Sparse4D
echo ""
echo "5. Installing Sparse4D..."
if [ ! -d "Sparse4D" ]; then
    git clone https://github.com/HorizonRobotics/Sparse4D.git
    cp Sparse4D.patch Sparse4D
    cp patch.py Sparse4D/tools
    cd Sparse4D
    git checkout c41df4bbf7bc82490f11ff55173abfcb3fb91425
    git apply Sparse4D.patch
    cp -rf ../test .
    cd ..
    echo "✓ Sparse4D installed"
else
    echo "Sparse4D already exists, skipping..."
fi

echo ""
echo "========================================"
echo "Sparse4D Installation Summary"
echo "========================================"
echo "Installation directory: $WORK_DIR"
echo "Installed components:"
echo "  - mmcv (with NPU support)"
echo "  - mmdetection"
echo "========================================"

