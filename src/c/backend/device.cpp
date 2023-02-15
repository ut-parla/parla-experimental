#include "include/device.hpp"
#include <iostream>

/*
void DeviceManager::RegisterDevices() {
  // Register CPU device.
  for (uint32_t d = 0; d < num_cpus_; ++d) {
    registered_devices_.emplace_back(CPUDevice(d));
  }

  // Register GPU devices.
  for (uint32_t d = 0; d < num_gpus_; ++d) {
    registered_devices_.emplace_back(CUDADevice(d));
  }
}
*/

void DeviceManager::RegisterCudaDevice(DevIDTy dev_id, size_t mem_sz,
                                       size_t num_vcus, void* py_dev) {
  CUDADevice new_cuda_dev = CUDADevice(dev_id, mem_sz, num_vcus, py_dev);
  registered_devices_.emplace_back(new_cuda_dev);
}

void DeviceManager::RegisterCpuDevice(DevIDTy dev_id, size_t mem_sz,
                                      size_t num_vcus, void* py_dev) {
  CPUDevice new_cpu_dev = CPUDevice(dev_id, mem_sz, num_vcus, py_dev);
  registered_devices_.emplace_back(new_cpu_dev);
}
