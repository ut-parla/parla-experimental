cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)

project(modtest LANGUAGES C CXX)

set(3RD_PARTY_DIR ${PROJECT_SOURCE_DIR}/3rd_party)
option(MODTEST_BUILD_TESTS "Enables building of C++ tests" ON)

message(STATUS "MODTEST_BUILD_TESTS: " ${MODTEST_BUILD_TESTS})

set(CMAKE_VERBOSE_MAKEFILE ON)
include(CTest)

find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
#include_directories(${NumPy_INCLUDE_DIRS})
include_directories(${NUMPY_INCLUDE})

add_subdirectory(src/c/backend)
add_subdirectory(src/python/modtest/cython)

if(MODTEST_BUILD_TESTS)
    add_subdirectory(unittests)
endif(MODTEST_BUILD_TESTS)

