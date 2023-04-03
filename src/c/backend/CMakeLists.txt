
file(GLOB BACKEND_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB BACKEND_HDR_CURRENT ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
file(GLOB BACKEND_HDR_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

#TODO(wlr): Fix this to be less hacky and cleaner :)
if(PARLA_ENABLE_CUDA)
    file(GLOB BACKEND_SRC ${BACKEND_SRC} ${CMAKE_CURRENT_SOURCE_DIR}/impl_cuda/*.cu)
    file(GLOB BACKEND_SRC ${BACKEND_SRC} ${CMAKE_CURRENT_SOURCE_DIR}/impl_cuda/*.cpp)
else()
    file(GLOB BACKEND_SRC ${BACKEND_SRC} ${CMAKE_CURRENT_SOURCE_DIR}/impl_none/*.cu)
    file(GLOB BACKEND_SRC ${BACKEND_SRC} ${CMAKE_CURRENT_SOURCE_DIR}/impl_none/*.cpp)
endif(PARLA_ENABLE_CUDA)

add_library(backend SHARED ${BACKEND_SRC})

if(PARLA_ENABLE_NVTX)
    message(STATUS "Enabling NVTX support.")
    target_include_directories(backend PUBLIC ${NVTX_INCLUDE})
    target_compile_definitions(backend PUBLIC PARLA_ENABLE_NVTX)
endif(PARLA_ENABLE_NVTX)

if(PARLA_ENABLE_LOGGING)
    message(STATUS "Enabling logging support.")
    target_link_libraries(backend binlog::headers)
    target_compile_definitions(backend PUBLIC PARLA_ENABLE_LOGGING)
endif(PARLA_ENABLE_LOGGING)

if(PARLA_ENABLE_CUDA)
    set_target_properties(backend PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set_target_properties(backend PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(backend PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    target_include_directories(backend PUBLIC ${CUDAToolkit_LIBRARY_DIR})
    target_include_directories(backend PUBLIC ${CUDAToolkit_LIBRARY_ROOT})
    target_include_directories(backend PUBLIC ${CUDAToolkit_INCLUDE_DIRS})

    target_link_directories(backend PUBLIC ${CUDAToolkit_LIBRARY_ROOT})
    target_compile_options(backend PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda -DPROD>)
    target_compile_options(backend PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

    target_link_libraries(backend PUBLIC CUDA::cudart)

    set(GPU_ARCH $ENV{CUDA_ARCH})
    if(GPU_ARCH)
        set_target_properties(backend PROPERTIES CUDA_ARCHITECTURES ${GPU_ARCH})
    else()
        set_target_properties(backend PROPERTIES CUDA_ARCHITECTURES OFF)
    endif()
endif(PARLA_ENABLE_CUDA)


target_include_directories(backend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(backend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(backend PROPERTIES PUBLIC_HEADER "${BACKEND_HDR_INCLUDE}")

#${PYTHON_RELATIVE_SITE_PACKAGES_DIR}/parla/lib/

install(
        TARGETS backend
        LIBRARY
            DESTINATION lib #${PYTHON_RELATIVE_SITE_PACKAGES_DIR}/parla/lib/
        PUBLIC_HEADER
            DESTINATION ${PYTHON_RELATIVE_SITE_PACKAGES_DIR}/parla/include/
    )

    
