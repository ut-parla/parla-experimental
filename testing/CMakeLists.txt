message(STATUS "Building unit tests")
message(STATUS "CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")
file(COPY ${CMAKE_SOURCE_DIR}/testing/python DESTINATION ${CMAKE_BINARY_DIR}/unittests/)
#file(COPY ${CMAKE_SOURCE_DIR}/testing/run_test.py DESTINATION ${CMAKE_BINARY_DIR}/unittests/)
add_subdirectory(c)
