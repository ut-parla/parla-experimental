#include <cstdint>
#include <cuda_runtime_api.h>
#include <gpu_utility.hpp>

// __device__ void gpu_sleep_0_bak(volatile clock_t *d_o, clock_t clock_count) {
//   clock_t start = clock();
//   clock_t now;
//   clock_t cycles;

//   for (;;) {
//     now = clock();
//     cycles = now > start ? now - start : now + (0xffffffff - start);
//     if (cycles >= clock_count) {
//       break;
//     }
//   }

//   *d_o = cycles;
// }

// __global__ void gpu_sleep_0(clock_t clock_count) {
//   static volatile clock_t buffer;
//   gpu_sleep_0_bak(&buffer, clock_count);
// }

__device__ void gpu_sleep_1_bak(unsigned long sleep_cycles) {
  unsigned long start = clock64();
  volatile unsigned long cycles_elapsed;
  do {
    cycles_elapsed = clock64() - start;
  } while (cycles_elapsed < sleep_cycles);
}

__global__ void gpu_sleep_1(clock_t clock_count) {
  gpu_sleep_1_bak(clock_count);
}

void gpu_busy_sleep(const int device, const unsigned long cycles,
                    uintptr_t stream_ptr) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  gpu_sleep_1<<<1, 1, device, stream>>>(cycles);
}

void event_synchronize(uintptr_t event_ptr) {
  cudaEvent_t event = reinterpret_cast<cudaEvent_t>(event_ptr);
  cudaEventSynchronize(event);
}
void event_wait(uintptr_t event_ptr, uintptr_t stream_ptr) {
  cudaEvent_t event = reinterpret_cast<cudaEvent_t>(event_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  // The 0 is for the flags.
  // 0 means that the event will be waited on in the default manner.
  // 1 has to do with CUDA graphs.
  cudaStreamWaitEvent(stream, event, 0);
};

void stream_synchronize(uintptr_t stream_ptr) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  cudaStreamSynchronize(stream);
};