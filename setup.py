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

    package_list = find_namespace_packages(where='src/python/')
    print("Found packages:", package_list)

    skbuild.setup(
        name="modtest",
        version="0.0.0",
        description="Minimal Example of a C++/Python Extension with CMake, Unit Testing, and Documentation",
        author='William Ruys',
        author_email="will@oden.utexas.edu",
        packages=["modtest", "modtest.cython", "modtest.common"],
        package_dir={"modtest": "src/python/modtest",
                     "modtest.cython": "src/python/modtest/cython",
                     "modtest.common": "src/python/modtest/common"},
        python_requires=">=3.8",
        cmake_args=cmake_args
    )


if __name__ == "__main__":
    main()
