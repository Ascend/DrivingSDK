import multiprocessing
import os
import platform
import stat
import subprocess
import sys
from pathlib import Path
from typing import Union

import torch
from setuptools import Extension, find_packages, setup
from setuptools._distutils.version import LooseVersion
from setuptools.command.build_clib import build_clib
from setuptools.command.build_ext import build_ext
from setuptools.command.develop import develop

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
VERSION = "1.0.0"


def which(thefile):
    path = os.environ.get("PATH", os.defpath).split(os.pathsep)
    for d in path:
        fname = os.path.join(d, thefile)
        fnames = [fname]
        if sys.platform == "win32":
            exts = os.environ.get("PATHEXT", "").split(os.pathsep)
            fnames += [fname + ext for ext in exts]
        for name in fnames:
            if os.access(name, os.F_OK | os.X_OK) and not os.path.isdir(name):
                return name
    return None


def get_cmake_command():
    def _get_version(cmd):
        for line in subprocess.check_output([cmd, "--version"]).decode("utf-8").split("\n"):
            if "version" in line:
                return LooseVersion(line.strip().split(" ")[2])
        raise RuntimeError("no version found")

    "Returns cmake command."
    cmake_command = "cmake"
    if platform.system() == "Windows":
        return cmake_command
    cmake3 = which("cmake3")
    cmake = which("cmake")
    if cmake3 is not None and _get_version(cmake3) >= LooseVersion("3.19.0"):
        cmake_command = "cmake3"
        return cmake_command
    elif cmake is not None and _get_version(cmake) >= LooseVersion("3.19.0"):
        return cmake_command
    else:
        raise RuntimeError("no cmake or cmake3 with version >= 3.19.0 found")


class CPPLibBuild(build_clib):
    def initialize_options(self) -> None:
        super().initialize_options()
        self.kernel_name = None

    def run(self) -> None:
        cmake = get_cmake_command()
        if not cmake:
            raise RuntimeError("CMake must be installed to build the libraries")
        self.cmake = cmake

        build_py = self.get_finalized_command("build_py")
        mx_driving_dir = os.path.join(BASE_DIR, build_py.build_lib, build_py.get_package_dir("mx_driving"))
        if not os.path.exists(mx_driving_dir):
            os.makedirs(mx_driving_dir)

        cmake_args = [
            "--preset=default",
            f"-DCMAKE_BUILD_TYPE={'Debug' if self.debug else 'Release'}",
            "-B",
            self.build_temp,
            f"-DMX_DRIVING_PATH={mx_driving_dir}",
            f"-DKERNEL_NAME={self.kernel_name if self.kernel_name else '*'}",
        ]
        build_args = ["--build", self.build_temp, f"-j{multiprocessing.cpu_count()}"]

        for stage in range(2):
            subprocess.check_call(
                [self.cmake, BASE_DIR] + cmake_args + ["-DBUILD_STAGE=" + str(stage)],
                cwd=BASE_DIR,
                env=os.environ,
            )
            subprocess.check_call(
                [self.cmake] + build_args,
                cwd=BASE_DIR,
                env=os.environ,
            )


class ExtBuild(build_ext):
    def run(self) -> None:
        cmake = get_cmake_command()
        if not cmake:
            raise RuntimeError("CMake must be installed to build the libraries")
        self.cmake = cmake

        build_py = self.get_finalized_command("build_py")
        mx_driving_dir = os.path.join(BASE_DIR, build_py.build_lib, build_py.get_package_dir("mx_driving"))
        if not os.path.exists(mx_driving_dir):
            os.makedirs(mx_driving_dir)

        ext_cxx_flags = ["-std=c++17"]
        for name in ["COMPILER_TYPE", "STDLIB", "BUILD_ABI"]:
            val = getattr(torch._C, f"_PYBIND11_{name}")
            if val:
                ext_cxx_flags.append(f"-D_PYBIND11_{name}={val}")

        cmake_args = [
            "--preset=default",
            f"-DCMAKE_BUILD_TYPE={'Debug' if self.debug else 'Release'}",
            "-B",
            self.build_temp,
            f"-DMX_DRIVING_PATH={mx_driving_dir}",
            f"-DEXT_CXX_FLAGS={' '.join(ext_cxx_flags)}",
            f"-DPython3_EXECUTABLE={sys.executable}",
        ]
        if LooseVersion(torch.__version__) < LooseVersion("2.1.0"):
            cmake_args.append("-DCOMPILE_WITH_XLA:BOOL=ON")

        if LooseVersion(torch.__version__) >= LooseVersion("2.6.0"):
            cmake_args.append("-DABI=1")
        else:
            cmake_args.append("-DABI=0")
        build_args = ["--build", self.build_temp, f"-j{multiprocessing.cpu_count()}"]

        subprocess.check_call(
            [self.cmake, BASE_DIR] + cmake_args + ["-DBUILD_STAGE=2"],
            cwd=BASE_DIR,
            env=os.environ,
        )
        subprocess.check_call(
            [self.cmake] + build_args,
            cwd=BASE_DIR,
            env=os.environ,
        )


class DevelopBuild(develop):
    user_options = develop.user_options + [
        ("kernel-name=", None, "Build the single kernel with the specified name"),
        ("release", None, "Build the release version"),
    ]

    def initialize_options(self) -> None:
        super().initialize_options()
        self.kernel_name = None
        self.release = False

    def install_for_development(self) -> None:
        self.reinitialize_command("build_py", build_lib="")
        self.reinitialize_command("build_clib", kernel_name=self.kernel_name, debug=not self.release)
        self.reinitialize_command("build_ext", debug=not self.release)

        if self.kernel_name:
            self.run_command("build_clib")
            return

        self.run_command("egg_info")
        self.run_command("build_clib")
        self.run_command("build_ext")

        if not self.dry_run:
            with os.fdopen(
                os.open(self.egg_link, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IWUSR | stat.S_IRUSR),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(self.egg_path + "\n" + self.setup_path)
        self.process_distribution(None, self.dist, not self.no_deps)


def get_sha(pytorch_root: Union[str, Path]) -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=pytorch_root).decode("ascii").strip()  # Compliant
        )
    except Exception:
        return "Unknown"


sha = get_sha(BASE_DIR)
if not os.getenv("BUILD_WITHOUT_SHA"):
    VERSION += "+git" + sha[:7]

setup(
    name="mx_driving",
    version=VERSION,
    description="A Library of acceleration for autonomous driving systems on Ascend-NPU.",
    keywords="mx_driving",
    ext_modules=[Extension("mx_driving._C", sources=[])],
    author="Ascend Contributors",
    libraries=[("mx_driving", {"sources": []})],
    cmdclass={
        "build_clib": CPPLibBuild,
        "build_ext": ExtBuild,
        "develop": DevelopBuild,
    },
    packages=find_packages(),
    include_package_data=True,
)
