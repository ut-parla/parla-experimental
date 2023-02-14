#include "include/device.hpp"
#include <iostream>

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

const std::vector<Device>& DeviceManager::GetAllDevices() {
  // TODO(hc): This is for debugging and will be removed.
  std::cout << "Registered device count:" << registered_devices_.size() << "\n";
  for (Device& d : registered_devices_) {
    std::cout << d.GetName() << "\n";
  }
  return registered_devices_;
}
