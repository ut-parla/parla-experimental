cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 20)

project(parla LANGUAGES C CXX)

set(3RD_PARTY_DIR ${PROJECT_SOURCE_DIR}/3rd_party)
option(PARLA_BUILD_TESTS "Enables building of C++ tests" OFF)
option(PARLA_DEBUG "Enables all debugging flags" ON)

message(STATUS "PARLA_BUILD_TESTS: " ${PARLA_BUILD_TESTS})

set(CMAKE_VERBOSE_MAKEFILE ON)
include(CTest)

find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
#include_directories(${NumPy_INCLUDE_DIRS})
include_directories(${NUMPY_INCLUDE})

if(PARLA_DEBUG)
    #add_compile_options("-Wall" "-Wpedantic" "-Wextra" "-fexceptions", "-fno-omit-frame-pointer", "-fsanitize=address", "-fsanitize=undefined", "-O0", "-g3", "-ggdb")
    #add_compile_options("-O0", "-g")
    SET(PARLA_DEBUG_C_COMPILE_FLAGS "-fno-omit-frame-pointer -O0 -g")
    SET(PARLA_DEBUG_CYTHON_COMPILE_FLAGS "--gdb")
    SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${PARLA_DEBUG_C_COMPILE_FLAGS}")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
    set(CYTHON_FLAGS "${CYTHON_FLAGS} ${PARLA_DEBUG_CYTHON_COMPILE_FLAGS}")
endif(PARLA_DEBUG)

add_subdirectory(src/c/backend)
add_subdirectory(src/python/parla)

if(PARLA_BUILD_TESTS)
    add_subdirectory(unittests)
endif(PARLA_BUILD_TESTS)

set(test_path_file ${CMAKE_SOURCE_DIR}/unittests/run_tests.sh)
file(WRITE ${test_path_file} "export PARLA_TESTS=${CMAKE_BINARY_DIR}/unittests\n")
file(APPEND ${test_path_file} "py.test $PARLA_TESTS\n")
file(APPEND ${test_path_file} "ctest --test-dir $PARLA_TESTS\n")
