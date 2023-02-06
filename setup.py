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

    package_list = find_namespace_packages(where='src/python/')
    print("Found packages:", package_list)

    skbuild.setup(
        name="parla",
        version="0.0.0",
        description="Minimal Example of a C++/Python Extension with CMake, Unit Testing, and Documentation",
        author='William Ruys',
        author_email="will@oden.utexas.edu",
        packages=["parla", "parla.cython", "parla.common"],
        package_dir={"parla": "src/python/parla",
                     "parla.cython": "src/python/parla/cython",
                     "parla.common": "src/python/parla/common"},
        python_requires=">=3.8",
        cmake_args=cmake_args
    )


if __name__ == "__main__":
    main()
