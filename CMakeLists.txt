cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)

project(parla LANGUAGES C CXX)

set(3RD_PARTY_DIR ${PROJECT_SOURCE_DIR}/3rd_party)
option(PARLA_BUILD_TESTS "Enables building of C++ tests" OFF)
option(PARLA_DEBUG "Enables all debugging flags" ON)

option(PARLA_ENABLE_LOGGING "Enable binlog logging" OFF)
option(PARLA_ENABLE_NVTX "Enable tracing with nvtx" OFF)
option(PARLA_ENABLE_CUDA "Enable compilation for CUDA architectures" OFF)

message(STATUS "PARLA_BUILD_TESTS: " ${PARLA_BUILD_TESTS})
message(STATUS "PARLA_DEBUG: " ${PARLA_DEBUG})
message(STATUS "PARLA_ENABLE_LOGGING: " ${PARLA_ENABLE_LOGGING})
message(STATUS "PARLA_ENABLE_NVTX: " ${PARLA_ENABLE_NVTX})
message(STATUS "PARLA_ENABLE_CUDA: " ${PARLA_ENABLE_CUDA})

set(CMAKE_VERBOSE_MAKEFILE ON)
include(CTest)

find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
#include_directories(${NumPy_INCLUDE_DIRS})
include_directories(${NUMPY_INCLUDE})

if(PARLA_DEBUG)
    SET(PARLA_DEBUG_C_COMPILE_FLAGS "-fno-omit-frame-pointer -O0 -g -static-libstdc++")
    SET(PARLA_DEBUG_CYTHON_COMPILE_FLAGS "--gdb")
    SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${PARLA_DEBUG_C_COMPILE_FLAGS}")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${PARLA_DEBUG_C_COMPILE_FLAGS}")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
    set(CYTHON_FLAGS "${CYTHON_FLAGS} ${PARLA_DEBUG_CYTHON_COMPILE_FLAGS}")
endif(PARLA_DEBUG)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lrt -lpthread")
if(PARLA_ENABLE_NVTX)
    set(NVTX_INCLUDE ${3RD_PARTY_DIR}/nvtx/c/include)
endif(PARLA_ENABLE_NVTX)

if(PARLA_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif(PARLA_ENABLE_CUDA)

if(PARLA_ENABLE_LOGGING)
    #TODO: figure out binlog cmake support
    message(STATUS "Finding binlog...")
    list(APPEND CMAKE_PREFIX_PATH "${3RD_PARTY_DIR}/binlog/install")
    find_package(binlog 0.1.0)

    if (binlog_FOUND)
        install(IMPORTED_RUNTIME_ARTIFACTS binlog::bread binlog::brecovery DESTINATION bin)
        install(RUNTIME_DEPENDENCY_SET binlog::binlog DESTINATION lib)
    endif (binlog_FOUND)

    if(NOT binlog_FOUND)
        message(STATUS "Logging library `binlog` not found. Building binlog from source")
        set(BINLOG_BUILD_EXAMPLES OFF CACHE BOOL "Build binlog examples" FORCE)
        set(BINLOG_BUILD_UNIT_TESTS OFF CACHE BOOL "Build binlog unit tests" FORCE)
        set(BINLOG_BUILD_INTEGRATION_TESTS OFF CACHE BOOL "Build binlog integration tests" FORCE)
        add_subdirectory(${3RD_PARTY_DIR}/binlog)
        #find_package(binlog 0.1.0)
    endif(NOT binlog_FOUND)

endif(PARLA_ENABLE_LOGGING)

add_subdirectory(src/c/backend)
add_subdirectory(src/python/parla)

if(PARLA_BUILD_TESTS)
    add_subdirectory(testing)
endif(PARLA_BUILD_TESTS)

set(test_path_file ${CMAKE_SOURCE_DIR}/testing/run_tests.sh)
file(WRITE ${test_path_file} "export PARLA_TESTS=${CMAKE_BINARY_DIR}/testing\n")
file(APPEND ${test_path_file} "py.test $PARLA_TESTS\n")
file(APPEND ${test_path_file} "ctest --test-dir $PARLA_TESTS\n")
