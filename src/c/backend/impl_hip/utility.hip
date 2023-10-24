#include <cstdint>
#include <gpu_utility.hpp>
#include<hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>


__device__ void gpu_sleep_impl(unsigned long sleep_cycles) {
  unsigned long start = clock64();
  volatile unsigned long cycles_elapsed;
  do {
    cycles_elapsed = clock64() - start;
  } while (cycles_elapsed < sleep_cycles);
}

__global__ void gpu_sleep(clock_t clock_count) {
  gpu_sleep_impl(clock_count);
}

void gpu_busy_sleep(const int device, const unsigned long cycles,
                    uintptr_t stream_ptr) {
  auto res = hipSetDevice(device);
  hipStream_t stream = reinterpret_cast<hipStream_t>(stream_ptr);
  gpu_sleep<<<1, 1,0, stream>>>(cycles); 
}

void event_synchronize(uintptr_t event_ptr) {
  hipEvent_t event = reinterpret_cast<hipEvent_t>(event_ptr);
  auto res = hipEventSynchronize(event);
}
void event_wait(uintptr_t event_ptr, uintptr_t stream_ptr) {
  hipEvent_t event = reinterpret_cast<hipEvent_t>(event_ptr);
  hipStream_t stream = reinterpret_cast<hipStream_t>(stream_ptr);

  // The 0 is for the flags.
  // 0 means that the event will be waited on in the default manner.
  // 1 has to do with CUDA graphs.
  auto res = hipStreamWaitEvent(stream, event, 0);
};

void stream_synchronize(uintptr_t stream_ptr) {
  hipStream_t stream = reinterpret_cast<hipStream_t>(stream_ptr);
  auto res = hipStreamSynchronize(stream);
};
