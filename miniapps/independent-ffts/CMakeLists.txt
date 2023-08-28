cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)

project(wrapper LANGUAGES C CXX)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(3RD_PARTY_DIR ${PROJECT_SOURCE_DIR}/3rd_party)
option(CUFFTMG_BUILD_TESTS "Enables building of C++ tests" ON)
set(CMAKE_CTEST_ARGUMENTS "--verbose")
include(CTest)



enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDAToolkit_LIBRARY_DIR: " ${CUDAToolkit_LIBRARY_DIR})

find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
include_directories(${NumPy_INCLUDE_DIRS})
include_directories(${NUMPY_INCLUDE})

#Compile C++ Implementation Source
add_subdirectory(src/impl)

#Compute Cython Wrapper
add_subdirectory(src/cufftmg)

if(CUFFTMG_BUILD_TESTS)
    message(STATUS "Building tests")
    add_subdirectory(test)
endif(CUFFTMG_BUILD_TESTS)

set(test_path_file ${CMAKE_SOURCE_DIR}/test/run_tests.sh)
file(WRITE ${test_path_file} "export CUFFTMG_TESTS=${CMAKE_BINARY_DIR}/test\n")
file(APPEND ${test_path_file} "py.test $CUFFTMG_TESTS\n")
file(APPEND ${test_path_file} "ctest --verbose --test-dir $CUFFTMG_TESTS\n")
