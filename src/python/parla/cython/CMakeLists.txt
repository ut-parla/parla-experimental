
set(CYTHON_FLAGS "${CYTHON_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_SHARED_MODULE_PREFIX "")

add_cython_target(core core.pyx CXX PY3)
add_cython_target(tasks tasks.pyx CXX PY3)
add_cython_target(scheduler scheduler.pyx CXX PY3)

add_library(core MODULE ${core})
add_library(tasks MODULE ${tasks})
add_library(scheduler MODULE ${scheduler})


message(STATUS "PYTHON_RELATIVE_SITE_PACKAGES_DIR: ${PYTHON_RELATIVE_SITE_PACKAGES_DIR}")
message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")

install(TARGETS core LIBRARY DESTINATION ${PYTHON_RELATIVE_SITE_PACKAGES_DIR}/parla/cython/)
install(TARGETS tasks LIBRARY DESTINATION ${PYTHON_RELATIVE_SITE_PACKAGES_DIR}/parla/cython/)
install(TARGETS scheduler LIBRARY DESTINATION ${PYTHON_RELATIVE_SITE_PACKAGES_DIR}/parla/cython/)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py DESTINATION ${PYTHON_RELATIVE_SITE_PACKAGES_DIR}/parla/cython/)

target_link_libraries(core ${PYTHON_LIBRARIES})
target_link_libraries(core backend)
target_include_directories(core PUBLIC ${PROJECT_SOURCE_DIR}/src/c/backend)
target_include_directories(core PUBLIC ${PROJECT_SOURCE_DIR}/src/c/backend/include)
target_include_directories(core PUBLIC ${NumPy_INCLUDE_DIRS})
target_include_directories(core PUBLIC ${PYTHON_INCLUDE_DIRS})
target_include_directories(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(tasks ${PYTHON_LIBRARIES})
target_link_libraries(tasks backend)
target_include_directories(tasks PUBLIC ${PROJECT_SOURCE_DIR}/src/c/backend)
target_include_directories(tasks PUBLIC ${PROJECT_SOURCE_DIR}/src/c/backend/include)
target_include_directories(tasks PUBLIC ${NumPy_INCLUDE_DIRS})
target_include_directories(tasks PUBLIC ${PYTHON_INCLUDE_DIRS})
target_include_directories(tasks PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})


target_link_libraries(scheduler ${PYTHON_LIBRARIES})
target_link_libraries(scheduler backend)
target_include_directories(scheduler PUBLIC ${PROJECT_SOURCE_DIR}/src/c/backend)
target_include_directories(scheduler PUBLIC ${PROJECT_SOURCE_DIR}/src/c/backend/include)
target_include_directories(scheduler PUBLIC ${NumPy_INCLUDE_DIRS})
target_include_directories(scheduler PUBLIC ${PYTHON_INCLUDE_DIRS})
target_include_directories(scheduler PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

python_extension_module(core)
python_extension_module(tasks)
python_extension_module(scheduler)