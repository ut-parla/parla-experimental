export PARLA_TESTS=/work2/06398/hochan/frontera/workspace/parla-experimental/_skbuild/linux-x86_64-3.10/cmake-build/testing
py.test $PARLA_TESTS
ctest --test-dir $PARLA_TESTS
