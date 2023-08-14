import os
import skbuild

from setuptools import find_namespace_packages


def main():
    cmake_args = []
    if env_conda := os.getenv("CONDA_PREFIX"):
        cmake_args.append(f"-DCONDA_PREFIX={env_conda}")

    package_list = find_namespace_packages(where="src/")
    print("Found packages:", package_list)

    skbuild.setup(
        name="cufftmg",
        version="0.0.0",
        description="Minimal 2D CUFFTMG Wrapper",
        author="William Ruys",
        author_email="will@oden.utexas.edu",
        packages=["cufftmg", "cufftmg.common", "cufftmg.cython"],
        package_dir={
            "cufftmg": "src/cufftmg",
            "cufftmg.cython": "src/cufftmg/cython",
            "cufftmg.common": "src/cufftmg/common",
        },
        python_requires=">=3.8",
        cmake_args=cmake_args,
    )


if __name__ == "__main__":
    main()
