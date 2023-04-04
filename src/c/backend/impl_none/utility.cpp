#include <gpu_utility.hpp>

void gpu_busy_sleep(const int device, const unsigned long t,
                    uintptr_t stream_ptr) {
  printf("gpu_busy_sleep() is not implemented for this backend.\n");
}

void event_synchronize(uintptr_t event_ptr){};
void event_wait(uintptr_t event_ptr, uintptr_t stream_ptr){};
void stream_synchronize(uintptr_t stream_ptr){};
