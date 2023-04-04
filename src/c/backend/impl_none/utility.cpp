#include <utility.hpp>

void gpu_busy_sleep(const int device, const unsigned long t,
                    uintptr_t stream_ptr) {
  printf("gpu_busy_sleep() is not implemented for this backend.\n");
}