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

void DeviceManager::RegisterDevice(Device* new_dev) {
  registered_devices_.emplace_back(*new_dev);
}
