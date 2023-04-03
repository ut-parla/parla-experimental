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

class Event {
public:
  uintptr_t event_ptr;

  Event() = default;
  Event(uintptr_t event) : event_ptr(event) {}

  uintptr_t get_event() { return event_ptr; }
  void set_event(uintptr_t event) { this->event_ptr = event; }
  void synchronize();
  void wait(uintptr_t stream);
};

class Stream {
public:
  uintptr_t stream_ptr;

  Stream() = default;
  Stream(uintptr_t stream) : stream_ptr(stream) {}

  uintptr_t get_stream() { return stream_ptr; }
  void set_stream(uintptr_t stream) { this->stream_ptr = stream; }
  void synchronize();
  void wait(uintptr_t event);
};

#endif // PARLA_CUDA_UTILITY_H