
enable_testing()

include(${3RD_PARTY_DIR}/doctest/doctest.cmake)

list(APPEND MODTEST_UNITTESTS 
  test_cpp
)

foreach(unittest IN LISTS MODTEST_UNITTESTS)
  add_executable(${unittest} ${unittest}.cpp)
  target_link_libraries(${unittest} backend)
  target_include_directories(${unittest} PRIVATE ${3RD_PARTY_DIR}/doctest)
  doctest_discover_tests(${unittest})
  install(TARGETS ${unittest} DESTINATION ${CMAKE_INSTALL_PREFIX}/testing/c/)
endforeach()
