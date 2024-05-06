cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)

project(${SKBUILD_PROJECT_NAME} LANGUAGES C CXX)

set(3RD_PARTY_DIR ${PROJECT_SOURCE_DIR}/3rd_party)
option(PARLA_BUILD_TESTS "Enables building of C++ tests" OFF)
option(PARLA_DEBUG "Enables all debugging flags" ON)

option(PARLA_ENABLE_LOGGING "Enable binlog logging" OFF)
option(PARLA_ENABLE_NVTX "Enable tracing with nvtx" OFF)
option(PARLA_ENABLE_CUDA "Enable compilation for CUDA architectures" OFF)

message(STATUS "PARLA_BUILD_TESTS: " ${PARLA_BUILD_TESTS})
message(STATUS "CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE})
message(STATUS "PARLA_ENABLE_LOGGING: " ${PARLA_ENABLE_LOGGING})
message(STATUS "PARLA_ENABLE_NVTX: " ${PARLA_ENABLE_NVTX})
message(STATUS "PARLA_ENABLE_CUDA: " ${PARLA_ENABLE_CUDA})

set(CMAKE_VERBOSE_MAKEFILE ON)
include(CTest)

find_package(
  Python
  COMPONENTS Interpreter Development.Module
  REQUIRED)

set(CYTHON cython)
message(STATUS "Cython: " ${CYTHON})
# Test if cython is installed
if(NOT CYTHON)
  message(
    FATAL_ERROR
      "Cython is required to build Parla. Please install cython and try again.")
endif()
# Test if cython exists
execute_process(
  COMMAND ${CYTHON} --version
  ERROR_QUIET
  OUTPUT_VARIABLE CYTHON_VERSION)
if(NOT CYTHON_VERSION)
  message(
    FATAL_ERROR
      "Cython is required to build Parla. Please install cython and try again.")
endif()

message(STATUS "Python: " ${Python_EXECUTABLE})
include_directories(${NUMPY_INCLUDE})

set(CYTHON_FLAGS "--cplus;--fast-fail")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(PARLA_DEBUG_C_COMPILE_FLAGS "-fno-omit-frame-pointer")
  set(PARLA_DEBUG_CYTHON_FLAGS "--line-directives;-p;--gdb")

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${PARLA_DEBUG_C_COMPILE_FLAGS}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${PARLA_DEBUG_C_COMPILE_FLAGS}")
  list(APPEND CYTHON_FLAGS ${PARLA_DEBUG_CYTHON_FLAGS})
  list(APPEND CYTHON_FLAGS "-X;linetrace=True")
endif()

if(UNIX)
  if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  else(NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  endif()
else(NOT UNIX)
  message(
    ERROR
    "Windows is not supported. Please raise an issue if you need Windows support."
  )
endif()

if(PARLA_ENABLE_NVTX)
  set(NVTX_INCLUDE ${3RD_PARTY_DIR}/nvtx/c/include)
endif(PARLA_ENABLE_NVTX)

if(PARLA_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif(PARLA_ENABLE_CUDA)

if(PARLA_ENABLE_HIP)
  enable_language(HIP)
endif(PARLA_ENABLE_HIP)

if(PARLA_ENABLE_LOGGING)
  # TODO: figure out binlog cmake support
  message(STATUS "Finding binlog...")
  list(APPEND CMAKE_PREFIX_PATH "${3RD_PARTY_DIR}/binlog/install")
  find_package(binlog 0.1.0)

  if(binlog_FOUND)
    install(IMPORTED_RUNTIME_ARTIFACTS binlog::bread binlog::brecovery
            DESTINATION ${SKBUILD_SCRIPTS_DIR})
    install(RUNTIME_DEPENDENCY_SET binlog::binlog DESTINATION lib)
  endif(binlog_FOUND)

  if(NOT binlog_FOUND)
    message(
      STATUS "Logging library `binlog` not found. Building binlog from source")
    set(BINLOG_BUILD_EXAMPLES
        OFF
        CACHE BOOL "Build binlog examples" FORCE)
    set(BINLOG_BUILD_UNIT_TESTS
        OFF
        CACHE BOOL "Build binlog unit tests" FORCE)
    set(BINLOG_BUILD_INTEGRATION_TESTS
        OFF
        CACHE BOOL "Build binlog integration tests" FORCE)
    add_subdirectory(${3RD_PARTY_DIR}/binlog)
    # find_package(binlog 0.1.0)
  endif(NOT binlog_FOUND)

endif(PARLA_ENABLE_LOGGING)

add_subdirectory(src/c/backend)
add_subdirectory(src/python/parla)