/*! @file gpu_utility.hpp
 *  @brief Provides architecture independent interface to event and stream
 * creation & synchronization.
 */

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

using namespace std::chrono;


/***
 * @brief Busy sleep on the GPU machine for a given number of cycles
 */
void gpu_busy_sleep(const int device, const unsigned long cycles,
                    uintptr_t stream_ptr);

/***
 * @brief Busy sleep on the host machine for a given number of microseconds
 */
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

/***
 * @brief Synchronize GPU events (host blocking)
 */
void event_synchronize(uintptr_t event_ptr);

/***
 * @brief Submit wait triggers on streams for GPU events (host non-blocking)
 */
void event_wait(uintptr_t event_ptr, uintptr_t stream_ptr);

/***
 * @brief Synchronize gpu streams (host blocking)
 */
void stream_synchronize(uintptr_t stream_ptr);

#endif // PARLA_CUDA_UTILITY_H
