#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <gpu_utility.hpp>

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

void set_device(int device) { cudaSetDevice(device); }

int get_device() {
  int device;
  cudaGetDevice(&device);
  return device;
}

int get_num_devices() {
  int num_devices;
  cudaGetDeviceCount(&num_devices);
  return num_devices;
}