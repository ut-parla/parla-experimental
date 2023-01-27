
file(GLOB BACKEND_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB BACKEND_HDR_CURRENT ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
file(GLOB BACKEND_HDR_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

add_library(backend STATIC ${BACKEND_SRC})

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

    