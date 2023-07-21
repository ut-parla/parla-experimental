#ifndef CUFFTMG_IMPLE_H_
#define CUFFTMG_IMPLE_H_

#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <stdio.h>
#include <string>

class FFTHandler {
public:
  int device_ids[4] = {0, 0, 0, 0};
  int num_devices = 0;
  int size = 0;
  uintptr_t streams[4] = {0, 0, 0, 0};
  size_t work_sizes[4] = {0, 0, 0, 0};

  cufftDoubleComplex* workspace[4] = {nullptr, nullptr, nullptr, nullptr};


  cufftResult result;
  cufftHandle plan;
  bool is_initialized = false;

  FFTHandler();
  ~FFTHandler();

  void configure(int *device_ids, int size, int num_devices, uint64_t *streams,
                 uint64_t *work_sizes);

  void empty(){};

  void set_workspace(uint64_t* workspace);

  void execute(void *input, void *output,
               int direction);
  void execute(void **input, void **output,
               int direction);
};


#endif // CUFFTMG_IMPLE_H_
