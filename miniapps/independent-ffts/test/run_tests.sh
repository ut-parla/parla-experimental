export CUFFTMG_TESTS=/work2/06081/wlruys/frontera/fft-wrapper-private/_skbuild/linux-x86_64-3.10/cmake-build/test
py.test $CUFFTMG_TESTS
ctest --verbose --test-dir $CUFFTMG_TESTS
