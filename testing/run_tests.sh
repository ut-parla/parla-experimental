export PARLA_TESTS=/home/wlruys/syncthing/workspace/parla/parla-experimental/_skbuild/linux-x86_64-3.11/cmake-build/testing
py.test $PARLA_TESTS
ctest --test-dir $PARLA_TESTS
