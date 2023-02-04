
file(GLOB BACKEND_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB BACKEND_HDR_CURRENT ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
file(GLOB BACKEND_HDR_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)





add_library(backend STATIC ${BACKEND_SRC})

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



target_include_directories(backend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(backend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

set_target_properties(backend PROPERTIES PUBLIC_HEADER "${BACKEND_HDR_INCLUDE}")
set_target_properties(backend PROPERTIES POSITION_INDEPENDENT_CODE ON)

install(
        TARGETS backend
        LIBRARY
            DESTINATION ${PYTHON_RELATIVE_SITE_PACKAGES_DIR}/parla/lib/
        PUBLIC_HEADER
            DESTINATION ${PYTHON_RELATIVE_SITE_PACKAGES_DIR}/parla/include/
    )

    