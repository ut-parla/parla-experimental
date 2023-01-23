
set(CYTHON_FLAGS "${CYTHON_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_SHARED_MODULE_PREFIX "")

add_cython_target(core core.pyx CXX PY3)
#add_cython_target(__init__ __init__.pyx CXX PY3)

add_library(core MODULE ${core})
#add_library(__init__ MODULE ${__init__})


message(STATUS "PYTHON_RELATIVE_SITE_PACKAGES_DIR: ${PYTHON_RELATIVE_SITE_PACKAGES_DIR}")
message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")

install(TARGETS core LIBRARY DESTINATION ${PYTHON_RELATIVE_SITE_PACKAGES_DIR}/modtest/cython/)
#install(TARGETS __init__ LIBRARY DESTINATION ${PYTHON_RELATIVE_SITE_PACKAGES_DIR}/modtest/cython/)

target_link_libraries(core ${PYTHON_LIBRARIES})
target_link_libraries(core backend)

target_include_directories(core PUBLIC ${PROJECT_SOURCE_DIR}/src/c/backend)
target_include_directories(core PUBLIC ${NumPy_INCLUDE_DIRS})
target_include_directories(core PUBLIC ${PYTHON_INCLUDE_DIRS})
target_include_directories(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

#target_link_libraries(__init__ ${PYTHON_LIBRARIES})
#target_include_directories(__init__ PUBLIC ${NumPy_INCLUDE_DIRS})
#target_include_directories(__init__ PUBLIC ${PYTHON_INCLUDE_DIRS})
#target_include_directories(__init__ PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

python_extension_module(core)