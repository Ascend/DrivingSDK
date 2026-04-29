#!/bin/bash

# ============================================================================
# DrivingSDK Pre-built Package Installation Script
# ============================================================================
# This script downloads and installs pre-built DrivingSDK wheel packages
#
# Usage:
#   bash install_drivingsdk.sh <BUILD_DATE> <TORCH_VERSION> <PYTHON_VERSION> <ARCH>
#
# Arguments:
#   BUILD_DATE     - Build date in format YYYYMMDD (e.g., 20260421)
#   TORCH_VERSION  - PyTorch version (e.g., 2.1.0, 2.7.1)
#   PYTHON_VERSION - Python version (e.g., 38, 3.10, 311)
#   ARCH           - Architecture (x86_64 or aarch64)
#   BUILD_NUMBER   - Optional: Build number (default: 2)
#
# Example:
#   bash install_drivingsdk.sh 20260421 2.7.1 3.10 x86_64
# ============================================================================

set -e

# Parse arguments
if [ $# -lt 4 ]; then
    echo "Error: Insufficient arguments"
    echo "Usage: bash install_drivingsdk.sh <BUILD_DATE> <TORCH_VERSION> <PYTHON_VERSION> <ARCH> [BUILD_NUMBER]"
    echo "Example: bash install_drivingsdk.sh 20260421 2.7.1 3.10 x86_64"
    exit 1
fi

BUILD_DATE=$1
TORCH_VERSION=$2
PYTHON_VERSION=$3
ARCH=$4
BUILD_NUMBER=${5:-2}  # Default to 2 if not provided

# Convert Python version format (3.10 -> cp310)
PYTHON_TAG="cp${PYTHON_VERSION//./}"

# Convert architecture format
if [ "$ARCH" = "x86_64" ]; then
    ARCH_TAG="x86_64"
elif [ "$ARCH" = "aarch64" ]; then
    ARCH_TAG="aarch64"
else
    echo "Error: Unsupported architecture: $ARCH"
    exit 1
fi

# Construct download URL
BASE_URL="https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/DrivingSDK/Daily/branch_v26.0.0"
TARBALL_NAME="mx_driving-1.0.${BUILD_DATE}-${PYTHON_TAG}-${PYTHON_TAG}-linux_${ARCH_TAG}.whl"
DOWNLOAD_URL="${BASE_URL}/torch${TORCH_VERSION}/${BUILD_DATE}.${BUILD_NUMBER}/${TARBALL_NAME}"

echo "========================================"
echo "DrivingSDK Installation"
echo "========================================"
echo "Build Date:      ${BUILD_DATE}.${BUILD_NUMBER}"
echo "PyTorch Version: ${TORCH_VERSION}"
echo "Python Version:  ${PYTHON_VERSION}"
echo "Architecture:    ${ARCH} (${ARCH_TAG})"
echo "Download URL:    ${DOWNLOAD_URL}"
echo "========================================"
echo ""

# Create temporary directory
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

# Download the tarball
echo "1. Downloading DrivingSDK wheel package..."
if ! wget --timeout=300 --tries=3 "$DOWNLOAD_URL"; then
    echo "Error: Failed to download from $DOWNLOAD_URL"
    echo "Please check:"
    echo "  - Build date is correct: ${BUILD_DATE}"
    echo "  - Build number exists: ${BUILD_NUMBER}"
    echo "  - PyTorch version is available: ${TORCH_VERSION}"
    echo "  - Python version is available: ${PYTHON_VERSION}"
    exit 1
fi

# Find the correct wheel file for the architecture
echo "2. Finding wheel file for ${ARCH_TAG}..."
WHEEL_FILE=$(find . -name "*${ARCH_TAG}*.whl" | head -1)

if [ -z "$WHEEL_FILE" ]; then
    echo "Error: No wheel file found for architecture ${ARCH_TAG}"
    echo "Available files:"
    find . -name "*.whl"
    exit 1
fi

echo "✓ Found wheel file: $WHEEL_FILE"
echo ""

# Install the wheel
echo "3. Installing DrivingSDK..."
pip install --no-cache-dir "$WHEEL_FILE"

if [ $? -eq 0 ]; then
    echo "✓ Installation completed successfully"
    echo ""
else
    echo "Error: Installation failed"
    exit 1
fi

# Cleanup
cd /
rm -rf "$TEMP_DIR"

echo "✓ Cleanup completed"

