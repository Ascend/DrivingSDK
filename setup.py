import os
import glob
import subprocess
from pathlib import Path
from typing import Union
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension
from utils import extension
import imp

from torch.utils.cpp_extension import BuildExtension
torch_npu_dir = extension.PYTORCH_NPU_INSTALL_PATH

source_file = []
source_file += glob.glob(os.path.join("./ads/common/ops/csrc/", "*.cpp"))
source_file += glob.glob(os.path.join("./bind/", "*.cpp"))

include_dirs = []
include_dirs.append(torch_npu_dir + "/include/third_party/acl/inc/")

exts = []
ext1 = extension.NpuExtension(
    name="ads_c",
    sources=source_file,
    extra_compile_args=[
      '-D__FILENAME__=\"$$(notdir $$(abspath $$<))\"',
      '-I' + imp.find_module('torch_npu')[1] + "/include/third_party/acl/inc",
      '-fprofile-arcs',
      '-ftest-coverage'],
    libraries=['gcov'],
)
exts.append(ext1)


def get_sha(pytorch_root: Union[str, Path]) -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=pytorch_root)  # Compliant
            .decode("ascii")
            .strip()
        )
    except Exception:
        return "Unknown"

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
VERSION = "1.0-" + torch.__version__[0:6]
torch_npu_root = Path(__file__).parent
sha = get_sha(torch_npu_root)
if not os.getenv("BUILD_WITHOUT_SHA"):
    VERSION += "+git" + sha[:7]

setup(
    name="ads_accelerator",
    version=VERSION,
    description='Cpp Extension Include ascend_ads',
    keywords='ads',
    ext_modules=exts,
    author='Ascend Contributors',
    cmdclass={"build_ext": BuildExtension},
    packages=find_packages(),
    include_package_data=True,
)