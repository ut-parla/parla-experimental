
file(GLOB BACKEND_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB BACKEND_HDR ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

add_library(backend STATIC ${BACKEND_SRC})

target_include_directories(backend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(backend PROPERTIES PUBLIC_HEADER "${BACKEND_HDR}")
set_target_properties(backend PROPERTIES POSITION_INDEPENDENT_CODE ON)

install(
        TARGETS backend
        LIBRARY
            DESTINATION ${PYTHON_RELATIVE_SITE_PACKAGES_DIR}/modtest/lib/
        PUBLIC_HEADER
            DESTINATION ${PYTHON_RELATIVE_SITE_PACKAGES_DIR}/modtest/include/
    )