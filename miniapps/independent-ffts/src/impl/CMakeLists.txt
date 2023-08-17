
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

message(STATUS "CUDAToolkit_LIBRARY_DIR: " ${CUDAToolkit_LIBRARY_DIR})


file(GLOB IMPL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
file(GLOB IMPL_HDR ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

add_library(cufftmg SHARED ${IMPL_SRC})

set_target_properties(cufftmg PROPERTIES PUBLIC_HEADER "${IMPL_HDR}")
set_target_properties(cufftmg PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(cufftmg PUBLIC ${CUDAToolkit_LIBRARY_DIR})
target_include_directories(cufftmg PUBLIC ${CUDAToolkit_LIBRARY_ROOT})
target_include_directories(cufftmg PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
target_include_directories(cufftmg PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_directories(cufftmg PUBLIC ${CUDAToolkit_LIBRARY_ROOT})

target_compile_options(cufftmg PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda -DPROD>)
target_compile_options(cufftmg PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

target_link_libraries(cufftmg PUBLIC CUDA::cudart CUDA::cufft)

#Move the libraries to the conda lib folder. This puts them on the library path.
install(
        TARGETS cufftmg
        LIBRARY
            DESTINATION lib
        PUBLIC_HEADER
            DESTINATION include
    )
