enable_testing()

include(${3RD_PARTY_DIR}/doctest/doctest.cmake)

list(APPEND CUFFTMG_UNITTESTS 
  test_2_GPU
)

foreach(unittest IN LISTS CUFFTMG_UNITTESTS)
  message(STATUS "Adding unittest: ${unittest}")
  add_executable(${unittest} ${unittest}.cpp)
  target_link_libraries(${unittest} PRIVATE cufftmg)
  target_include_directories(${unittest} PRIVATE ${3RD_PARTY_DIR}/doctest)
  target_include_directories(${unittest} PRIVATE ${CUDAToolkit_LIBRARY_DIR})
  target_include_directories(${unittest} PRIVATE ${CUDAToolkit_LIBRARY_ROOT})
  target_include_directories(${unittest} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  #target_include_directories(${unittest} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_directories(${unittest} PRIVATE ${CUDAToolkit_LIBRARY_ROOT})
  #target_compile_options(${unittest} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda -DPROD>)
  #target_compile_options(${unittest} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  target_link_libraries(${unittest} PRIVATE CUDA::cudart CUDA::cufft)
  doctest_discover_tests(${unittest})
  install(TARGETS ${unittest} DESTINATION ${CMAKE_INSTALL_PREFIX}/test/cpp/)
endforeach()
