#pragma once
#ifndef PARLA_DEVICE_MANAGER_HPP
#define PARLA_DEVICE_MANAGER_HPP

#include "device.hpp"
#include <iostream>
#include <string>
#include <unordered_map>

#include <vector>

using DevIDTy = uint32_t;

/// `DeviceManager` registers/provides devices and their
/// information on the current system to the Parla runtime.
class DeviceManager {
public:
  DeviceManager() {}
  void RegisterDevice(Device *new_dev);

  void PrintRegisteredDevices() {
    // std::cout << "C++ device list:\n";
    for (size_t d = 0; d < registered_devices_.size(); ++d) {
      Device &dev = registered_devices_[d];
      // std::cout << "[" << dev.GetName() << "] mem. sz:" <<
      //   dev.GetMemorySize() << ", num. vcus:" << dev.GetNumVCUs() << "\n";
    }
  }

  std::vector<Device> &GetAllDevices() { return registered_devices_; }

private:
  std::vector<Device> registered_devices_;
};

#endif
