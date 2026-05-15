#!/bin/bash

# ============================================================================
# BEVFormer Installation Script for Docker
# ============================================================================
# This script installs BEVFormer and its dependencies in a Python environment
#
# Usage:
#   bash install_bevformer.sh
#
# Prerequisites:
#   - Python environment with PyTorch 2.7.1
#   - Driving SDK already installed
# ============================================================================

set -e

echo "========================================"
echo "BEVFormer Installation"
echo "========================================"
echo ""

# Create workspace directory for BEVFormer
WORK_DIR=/workspace/bevformer
mkdir -p $WORK_DIR
cd $WORK_DIR

# Copy BEVFormer installation files from model_examples
echo "1. Copying BEVFormer installation files..."
BEVFORMER_SRC="/tmp/BEVFormer"
if [ -d "$BEVFORMER_SRC" ]; then
    cp -r $BEVFORMER_SRC/* $WORK_DIR/
    echo "✓ Files copied from $BEVFORMER_SRC"
else
    echo "Error: BEVFormer directory not found at $BEVFORMER_SRC"
    echo "Please ensure Driving SDK source code is copied to /workspace/DrivingSDK"
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
    cp mmcv_config.patch mmcv
    cd mmcv
    git apply --reject mmcv_config.patch || true
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
echo "4. Installing mmdetection3d..."
if [ ! -d "mmdetection3d" ]; then
    git clone -b v1.0.0rc4 https://github.com/open-mmlab/mmdetection3d.git
    cp mmdet3d_config.patch mmdetection3d
    cd mmdetection3d
    git apply --reject mmdet3d_config.patch || true
    pip install -e . --no-build-isolation
    cd ..
    echo "✓ mmdetection3d installed"
else
    echo "mmdetection3d already exists, skipping..."
fi

# Install mmdetection
echo ""
echo "5. Installing mmdetection..."
if [ ! -d "mmdetection" ]; then
    git clone -b v2.24.0 https://github.com/open-mmlab/mmdetection.git
    cp mmdet_config.patch mmdetection
    cd mmdetection
    git apply --reject mmdet_config.patch || true
    pip install -e . --no-build-isolation
    cd ..
    echo "✓ mmdetection installed"
else
    echo "mmdetection already exists, skipping..."
fi

# Install detectron2
echo ""
echo "6. Installing detectron2..."
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation
echo "✓ detectron2 installed"

# Install BEVFormer
echo ""
echo "7. Installing BEVFormer..."
if [ ! -d "BEVFormer" ]; then
    git clone https://github.com/fundamentalvision/BEVFormer.git
    cp bev_former_config.patch BEVFormer
    cd BEVFormer
    git checkout 66b65f3a1f58caf0507cb2a971b9c0e7f842376c
    git apply --reject --whitespace=fix bev_former_config.patch || true
    cd ..
    echo "✓ BEVFormer installed"
else
    echo "BEVFormer already exists, skipping..."
fi

echo ""
echo "========================================"
echo "BEVFormer Installation Summary"
echo "========================================"
echo "Installation directory: $WORK_DIR"
echo "Installed components:"
echo "  - mmcv (with NPU support)"
echo "  - mmdetection3d"
echo "  - mmdetection"
echo "  - detectron2"
echo "  - BEVFormer"
echo "========================================"
