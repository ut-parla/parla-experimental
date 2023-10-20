from pathlib import Path
import sys
import re
import os
import skbuild

from setuptools import find_namespace_packages
import numpy as np
from string import Template


def main():

    cmake_args = []
    if env_conda := os.getenv("CONDA_PREFIX"):
        cmake_args.append(f"-DCONDA_PREFIX={env_conda}")

    if env_enable_logging := os.getenv("PARLA_ENABLE_LOGGING"):
        cmake_args.append(f"-DPARLA_ENABLE_LOGGING={env_enable_logging}")

    if env_enable_nvtx := os.getenv("PARLA_ENABLE_NVTX"):
        cmake_args.append(f"-DPARLA_ENABLE_NVTX={env_enable_nvtx}")

    if env_enable_cuda := os.getenv("PARLA_ENABLE_CUDA"):
        cmake_args.append(f"-DPARLA_ENABLE_CUDA={env_enable_cuda}")

    package_list = find_namespace_packages(where='src/python/')
    print("Found packages:", package_list)

    skbuild.setup(
        name="parla",
        description="Parla: A Python Parallel Programming Framework",
        author='UT Austin Parla Team',
        author_email="",
        packages=["parla", "parla.cython", "parla.common", "parla.utility", "parla.array", "parla.tasks", "parla.devices"],
        package_dir={"parla": "src/python/parla",
                     "parla.cython": "src/python/parla/cython",
                     "parla.common": "src/python/parla/common",
                     "parla.utility": "src/python/parla/utility", 
                     "parla.array": "src/python/parla/array",
                     "parla.tasks": "src/python/parla/tasks", 
                     "parla.devices": "src/python/parla/devices"},
        cmake_args=cmake_args
    )


if __name__ == "__main__":
    main()
