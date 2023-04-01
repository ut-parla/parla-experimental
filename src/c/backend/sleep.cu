#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>

using namespace std;
using namespace chrono;

__device__ void gpu_sleep_1_bak(unsigned long sleep_cycles) {
  unsigned long start = clock64();
  unsigned long cycles_elapsed;
  do {
    cycles_elapsed = clock64() - start;
  } while (cycles_elapsed < sleep_cycles);
}

__global__ void gpu_sleep_1(clock_t clock_count) {
  gpu_sleep_1_bak(clock_count);
}

void gpu_sleeper(const int device, const unsigned long t, intptr_t stream_ptr) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  gpu_sleep_1<<<1, 1, device, stream>>>(t);
}
