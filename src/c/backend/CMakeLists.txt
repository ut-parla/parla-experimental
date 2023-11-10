set(BACKEND_SRC 
    device_manager.cpp
    device.cpp
    parray_state.cpp 
    parray.cpp
    parray_tracker.cpp
    phases.cpp
    policy.cpp
    resource_requirements.cpp
    resources.cpp
    scheduler.cpp
    task.cpp
)

set(BACKEND_HDR
    atomic_wrapper.hpp
    containers.hpp
    device_queues.hpp
    gpu_utility.hpp
    device_manager.hpp
    device.hpp
    parray_state.hpp
    parray.hpp
    parray_tracker.hpp
    phases.hpp
    policy.hpp
    resource_requirements.hpp
    resources.hpp
    runtime.hpp
)

#Add "include" to all header files
foreach(HEADER ${BACKEND_HDR})
    set(BACKEND_HDR_INCLUDE ${BACKEND_HDR_INCLUDE} ${CMAKE_CURRENT_SOURCE_DIR}/include/${HEADER})
endforeach()

set(SPECIALIZATIONS "utility")

if(PARLA_ENABLE_CUDA)
    foreach(SPEC ${SPECIALIZATIONS})
        message("SPEC_G: ${SPEC}")
        list(APPEND BACKEND_SRC impl_cuda/${SPEC}.cu)
    endforeach()
elseif(PARLA_ENABLE_HIP)
    foreach(SPEC ${SPECIALIZATIONS})
        message("SPEC_C: ${SPEC}")
        list(APPEND BACKEND_SRC impl_hip/${SPEC}.hip)
    endforeach()
else()
    foreach(SPEC ${SPECIALIZATIONS})
        message("SPEC_C: ${SPEC}")
        list(APPEND BACKEND_SRC impl_none/${SPEC}.cpp)
    endforeach()
endif(PARLA_ENABLE_CUDA)

message(STATUS "ALL FILES: ${BACKEND_SRC}")

add_library(backend STATIC ${BACKEND_SRC})

if(PARLA_ENABLE_NVTX)
    message(STATUS "Enabling NVTX support.")
    target_include_directories(backend PUBLIC ${NVTX_INCLUDE})
    target_compile_definitions(backend PUBLIC PARLA_ENABLE_NVTX)
endif(PARLA_ENABLE_NVTX)

if(PARLA_ENABLE_LOGGING)
    message(STATUS "Enabling logging support.")
    target_link_libraries(backend PUBLIC binlog::headers)
    target_compile_definitions(backend PUBLIC PARLA_ENABLE_LOGGING)
endif(PARLA_ENABLE_LOGGING)

set_target_properties(backend PROPERTIES POSITION_INDEPENDENT_CODE ON)


if(PARLA_ENABLE_CUDA)
    
    set_target_properties(backend PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(backend PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    target_compile_definitions(backend PUBLIC PARLA_ENABLE_CUDA)

    #target_include_directories(backend PUBLIC ${CUDAToolkit_LIBRARY_DIR})
    #target_include_directories(backend PUBLIC ${CUDAToolkit_LIBRARY_ROOT})
    #target_include_directories(backend PUBLIC ${CUDAToolkit_INCLUDE_DIRS})

    #target_link_directories(backend PUBLIC ${CUDAToolkit_LIBRARY_ROOT})
    #target_compile_options(backend PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda -DPROD>)
    #target_compile_options(backend PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

    #target_link_libraries(backend PUBLIC CUDA::cudart)

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

install(TARGETS backend
    EXPORT ParlaBackendTargets
    LIBRARY DESTINATION .
)

    
