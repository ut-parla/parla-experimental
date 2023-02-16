export PARLA_TESTS=/home/wlruys/workspace/parla-experimental/_skbuild/linux-x86_64-3.10/cmake-build/unittests
py.test $PARLA_TESTS
ctest --test-dir $PARLA_TESTS
