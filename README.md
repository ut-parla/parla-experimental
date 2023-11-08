#  Parla Experimental

Repository for refactoring Parla to a C++ runtime to avoid GIL during scheduling

# Installation and Use

Note: Ensure that the Cupy version and your CUDA Toolkit are compatible with each other and the architecture of your machine.
On TACC systems we recommend installing CuPy (and if needed numba) via pip against the system CUDA version instead of pulling in binaries from conda-forge's cudatoolkit.

Can be built with:
```
    git submodule update --init --recursive
    pip install .
```

You may need to set C compilers explicily on Frontera. 
```
module load gcc/12
make clean; CC=gcc CXX=g++ make
```

This will populate a `build` directory and an install into your active Python environment's site-packages. 
If you are a developer, it may sometimes be necessary to remove the build directory manually to apply changes and remove the cached version.
Changes to most files in the src tree will trigger a rebuild of the relevant files without needing to clear the build directory. 
Additionally, `scikit-build-core` supports editable installations that recompile changes on each run, please see this documentation. 

You may need to set C compilers explicily on Frontera. 
```
module load gcc/12
make clean; CC=gcc CXX=g++ make
```

Build options are set through the pyproject.toml:
PARLA_ENABLE_NVTX=ON/OFF :   Enables NVTX hooks in the C++ backend.
PARLA_ENABLE_LOGGING=ON/OFF : Enables Binlog hooks in the Python & C++ backend for logging. 
PARLA_ENABLE_CUDA=ON/OFF : Enables CUDA C++ backend for runahead scheduling
PARLA_ENABLE_HIP=ON/OFF : Enables HIP C++ backend for runahead scheduling

# Logging

Logging is done through the high-performance binlog library. We wrap the logging function to dump to the same logs from Python, Cython, & C++. 
Logging from Python & Cython is more expensive than in C++ due to the cost of calling a c-extension, be aware of performance penalties from logging many things. 

Logs are dumped when the Parla runtime exits. The name of the log file defaults to "parla.blog" but can be set with the PARLA_LOGFILE enviornment variable or as a parameter in the `with Parla(logfile=filename)` context. 

If a Parla program is interrupted with a KeyboardInterrupt Exception (aka Ctrl+C) it should dump the log as the runtime exits gracefully. The same *should* hopefully now be true of exceptions raised within tasks or workers. But please raise an issue if you found one that isn't handled. 
 
If a Parla program is interuppted with an error that causes a crash and core dump, the logfile can be recovered from the coredump directly with `brecovery`. See the binlog documentation for more information. 

To read a logfile run:
`bread <logfile>`


# Number of threads

 The number of workers in the Parla runtime is controlled by the PARLA_NUM_THREADS enviornment variable. If it is not set it defaults to the number of cores on the host machine. 

