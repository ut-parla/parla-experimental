#ifndef PARLA_CUDA_UTILITY_H
#define PARLA_CUDA_UTILITY_H

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>

#if defined(PARLA_ENABLE_CUDA)
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

using namespace std;
using namespace chrono;

void gpu_busy_sleep(const int device, const unsigned long cycles,
                    uintptr_t stream_ptr);

// Busy sleep for a given number of microseconds
inline void cpu_busy_sleep(unsigned int micro) {
  // compute_range r("sleep::busy", nvtx3::rgb{0, 127, 127});
  // int count = 0;
  auto block = std::chrono::microseconds(micro);
  auto time_start = std::chrono::high_resolution_clock::now();

  auto now = std::chrono::high_resolution_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(now - time_start);

  do {
    now = std::chrono::high_resolution_clock::now();
    elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(now - time_start);
  } while (elapsed.count() < micro);
}

void event_synchronize(uintptr_t event_ptr);
void event_wait(uintptr_t event_ptr, uintptr_t stream_ptr);
void stream_synchronize(uintptr_t stream_ptr);

#endif // PARLA_CUDA_UTILITY_H