# Driving SDK Docker Images Overview

## Quick Reference

- **Maintained by**: Driving SDK Team
- **Where to file issues**: [Driving SDK Issue Tracker](https://gitcode.com/Ascend/DrivingSDK/issues)
- **Supported architectures**: `x86_64` (AMD64), `aarch64` (ARM64)
- **Supported OS**: Ubuntu 22.04, openEuler 24.03
- **Base image**: CANN 9.0.0, CANN 8.5.1

---

## Image Tag Keywords Description

The image tags follow this naming convention:

`<DrivingSDK_VERSION>-<CANN_VERSION>-<NPU_TYPE>-<OS_TYPE>`

### Field Descriptions

| Field | Description | Available Values |
|-------|-------------|------------------|
| **DrivingSDK_VERSION** | Driving SDK version | `26.0.0` |
| **CANN_VERSION** | CANN version | `8.5.1`, `9.0.0` |
| **NPU_TYPE** | NPU type | `A2`, `A3`|
| **OS_TYPE** | Operating System type | `ubuntu22.04`, `openeuler24.03` |

### Examples

- `26.0.0-cann8.5.1-910b-ubuntu22.04`: CANN 8.5.1, A2, Ubuntu 22.04
- `26.0.0-cann9.0.0-a3-openeuler24.03`: CANN 9.0.0, A3, openEuler 24.03

---

## Dockerfile Archive Path

All Dockerfiles are archived under the `docker/` directory with the following structure:

```shell
docker/
├── 8.5.1-910b-openeuler24.03/
│   └── Dockerfile
├── 8.5.1-910b-ubuntu22.04/
│   └── Dockerfile
├── 8.5.1-a3-openeuler24.03/
│   └── Dockerfile
├── 8.5.1-a3-ubuntu22.04/
│   └── Dockerfile
├── 9.0.0-910b-openeuler24.03/
│   └── Dockerfile
├── 9.0.0-910b-ubuntu22.04/
│   └── Dockerfile
├── 9.0.0-a3-openeuler24.03/
│   └── Dockerfile
├── 9.0.0-a3-ubuntu22.04/
│   └── Dockerfile
├── install_bevformer.sh
├── install_bevfusion.sh
├── install_drivingsdk.sh
├── install_sparse4d.sh
├── OVERVIEW.md
└── OVERVIEW.zh.md
```

---

## Quick Start

### 1. Building Images Locally

Build the Docker image from source:

```bash
# Clone the repository
git clone https://gitcode.com/Ascend/DrivingSDK.git
cd DrivingSDK

# Build the image
docker build -f docker/8.5.1-910b-ubuntu22.04/Dockerfile -t drivingsdk:26.0.0-cann8.5.1-910b-ubuntu22.04 .

# Run with mounted volumes for development
docker run -it --rm \
  --device=/dev/davinci0 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/Ascend:/usr/local/Ascend \
  -v $(pwd):/workspace \
  drivingsdk:26.0.0-cann8.5.1-910b-ubuntu22.04 \
  /bin/bash
```

### 2. Secondary Development

For custom development and modifications:

```bash
# Clone the repository
git clone https://gitcode.com/Ascend/DrivingSDK.git
cd DrivingSDK

# Modify Dockerfile as needed
vim docker/8.5.1-910b-ubuntu22.04/Dockerfile

# Build with your changes
docker build -f docker/8.5.1-910b-ubuntu22.04/Dockerfile -t drivingsdk:dev .

# Run with mounted volumes for development
docker run -it --rm \
  --device=/dev/davinci0 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/Ascend:/usr/local/Ascend \
  -v $(pwd):/workspace \
  drivingsdk:dev \
  /bin/bash
```

---

## Hardware Support Information

### Supported NPU Types

| NPU Type | Architecture | Description | Status |
|----------|--------------|-------------|--------|
| **A2** | x86_64, aarch64 | A2 | Production Ready |
| **A3** | x86_64, aarch64 | A3 | Production Ready |

### Supported Operating Systems

| OS | Version | Architecture | Package Manager |
|----|---------|--------------|-----------------|
| Ubuntu | 22.04 LTS | x86_64, aarch64 | apt |
| openEuler | 24.03 LTS | x86_64, aarch64 | yum/dnf |

### Python Environment Support

The Docker images provide multiple Python environments via Miniconda:

| Environment | Python Version | PyTorch Version | Use Case |
|-------------|----------------|-----------------|----------|
| `torch2.1` | 3.8 | 2.1.0 | General PyTorch development |
| `torch2.7.1` | 3.10 | 2.7.1 | Latest PyTorch features |
| `bevformer` | 3.10 | 2.7.1 | BEVFormer model training |
| `bevfusion` | 3.10 | 2.7.1 | BEVFusion model training |
| `sparse4d` | 3.10 | 2.7.1 | Sparse4D model training |

### Hardware Requirements

- **Minimum**: 1 NPU device (A2 or A3)
- **Recommended**: 2+ NPU devices for distributed training
- **Memory**: Minimum 32GB RAM, recommended 64GB+
- **Storage**: Minimum 100GB for Docker images and datasets

---

## Included Components

### Core Components

- **DrivingSDK**: Driving SDK version 26.0.0
- **CANN**: Compute Architecture for Neural Networks (8.5.1 / 9.0.0)
- **PyTorch**: Deep learning framework (2.1.0 / 2.7.1)
- **torch-npu**: PyTorch NPU backend
- **Miniconda**: Python environment management

### Model Examples

- **BEVFormer**: Bird's Eye View Transformer for 3D object detection
- **BEVFusion**: Multi-modal 3D detection with BEV fusion
- **Sparse4D**: Sparse 4D detection framework

### System Dependencies

- GCC/G++ compiler
- CMake build system
- Git, wget, curl utilities
- Protocol Buffers
- Network tools

---

## Usage Examples

### Activate a Specific Environment

```bash
# Inside the container
source /opt/conda/etc/profile.d/conda.sh
conda activate torch2.7.1
```

### Install Additional Packages

```bash
# Activate environment
conda activate torch2.7.1

# Install package
pip install your-package
```

---

## Troubleshooting

### Common Issues

1. **NPU Device Not Found**

   ```bash
   # Check NPU devices
   npu-smi info

   # Ensure devices are mounted
   docker run --device=/dev/davinci0 ...
   ```

2. **CANN Environment Not Set**

   ```bash
   # Source CANN environment
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```

3. **Conda Environment Not Found**

   ```bash
   # Initialize conda
   source /opt/conda/etc/profile.d/conda.sh
   conda env list
   ```

---

## License

This project is licensed under the terms specified in the LICENSE file at the root of the repository.

### Third-Party Licenses

- **CANN**: CANN Software License
- **PyTorch**: BSD 3-Clause License
- **Miniconda**: Anaconda Terms of Service

---

## Disclaimer

**IMPORTANT**: This software is provided "as is" without warranty of any kind, express or implied.

- Use at your own risk
- The maintainers are not responsible for any damages arising from the use of this software
- Always test in a non-production environment before deployment
- Ensure compliance with Atlas licensing terms
- NPU hardware must be properly configured and accessible

For production deployments, please consult the official Atlas documentation and follow best practices for security and performance optimization.

---

## Support and Resources

- **CANN Documentation**: [Ascend Documentation](https://www.hiascend.com/document)
- **Issue Tracker**: [Driving SDK Issues](https://gitcode.com/Ascend/DrivingSDK/issues)

---

**Last Updated**: 2026-04-23
**Maintainer**: Driving SDK Team
