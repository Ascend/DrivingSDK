import glob
import os
import subprocess
from pathlib import Path
from typing import Union

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension 

from utils import extension

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
VERSION = torch.__version__
full_components = ["common", "motion", "perception/fused", "perception/point", "perception/vision", "spconv"]
source_file = glob.glob(os.path.join("./bind/", "*.cpp"))
include_dirs = [os.path.join(BASE_DIR, "include")]
for part in full_components:
    source_file += glob.glob(os.path.join(f"./ads/{part}/ops/csrc/", "*.cpp"))

exts = []
ext1 = extension.NpuExtension(
    name="ads_c",
    sources=source_file,
    include_dirs=include_dirs,
    extra_compile_args=[
        '-D__FILENAME__="$$(notdir $$(abspath $$<))"',
        "-fprofile-arcs",
        "-ftest-coverage",
        "-fPIC",
        "-fstack-protector-all",
    ],
    extra_link_args=[
        "-Wl,-z,relro",
        "-Wl,-z,now",
        "-s"
    ],
    libraries=["gcov"],
)
exts.append(ext1)


def get_sha(pytorch_root: Union[str, Path]) -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=pytorch_root).decode("ascii").strip()  # Compliant
        )
    except Exception:
        return "Unknown"


VERSION = "1.0.0"
ads_root = Path(__file__).parent
sha = get_sha(ads_root)
if not os.getenv("BUILD_WITHOUT_SHA"):
    VERSION += "+git" + sha[:7]

setup(
    name="ads_accelerator",
    version=VERSION,
    description="A Library of acceleration for autonomous driving systems on Ascend-NPU.",
    keywords="ads",
    ext_modules=exts,
    author="Ascend Contributors",
    cmdclass={"build_ext": BuildExtension},
    packages=find_packages(),
    include_package_data=True,
)
