#  Parla Experimental

Repository for refactoring Parla to a C++ runtime to avoid GIL during scheduling

# Installation and Use

Requires: 
- psutil, cython, scikit-build, Python>=3.8, nvtx (python module)
- A C++ compiler that supports C++20 float atomics
- An internet connection. I don't know how to make scikit-build stop building in its own venv. 


Can be built with:
```
    make clean; make
```

RUNNING `make clean` IS VERY IMPORTANT! 
You may need to set C compilers explicily on Frontera. 

```
module load gcc/12
make clean; CC=gcc CXX=g++ make
```

Build options are set through enviornment variables:

PARLA_ENABLE_NVTX=ON/OFF :   Enables NVTX hooks in the C++ backend.

PARLA_ENABLE_LOGGING=ON/OFF : Enables Binlog hooks in the Python & C++ backend for logging. 

There are currently no C++ tests for Parla but if there were, the CMAKE is configured to build and run them with `ctest`. 
There is a pytest driven folder for tests from Python. 

# Logging

Logging is done through the high-performance binlog library. We wrap the logging function to dump to the same logs from Python, Cython, & C++. 
Logging from Python & Cython is more expensive than in C++ due to the cost of calling a c-extension, be aware of performance penalties from logging many things. 

Logs are dumped when the Parla runtime exits. The name of the log file defaults to "parla.blog" but can be set with the PARLA_LOGFILE enviornment variable or as a parameter in the `with Parla(logfile=filename)` context. 

If a Parla program is interrupted with a KeyboardInteruppt (Ctrl+C) it should dump the log when the runtime exits gracefully. The same *should* hopefully now be true of exceptions raised within tasks or workers. But please raise an issue if you found one that isn't handled. 
 
If a Parla program is interuppted with an error that causes a crash and core dump, the logfile can be recovered with `brecovery`. See the binlog documentation for more information. 


To read a logfile run:
`bread <logfile>`

# Number of threads

 The number of workers in the Parla runtime is controlled by the PARLA_NUM_THREADS enviornment variable. If it is not set it defaults to the number of cores on the host machine. 

