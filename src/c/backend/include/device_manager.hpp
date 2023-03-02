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
    for (size_t d = 0; d < this->registered_devices_.size(); ++d) {
      Device &dev = this->registered_devices_[d];
      // std::cout << "[" << dev.GetName() << "] mem. sz:" <<
      //   dev.GetMemorySize() << ", num. vcus:" << dev.GetNumVCUs() << "\n";
    }
  }

  std::vector<Device> &GetAllDevices() { return registered_devices_; }

#if 0
  ResourceRequirement* CreateResourceRequirement(Device* dev, DeviceResources req) {
    // TODO(hc): This can be known at Python.
    //           Later get this information from Python.
    size_t num_archs = 2;
    DeviceType dev_type;
    std::vector<bool> has_arch_constraint(num_archs);
    if (typeid(*dev) == typeid(CUDADevice)) {
      dev_type = CUDA;
      has_arch_constraint[dev_type] = 1;
    } else if (typeid(*dev) == typeid(CPUDevice)) {
      dev_type = CPU;
      has_arch_constraint[dev_type] = 1;
    }

    std::vector<std::vector<Device*>> dev_ptr_vec(num_archs);
    dev_ptr_vec[dev_type].emplace_back(dev);

    std::vector<DeviceResources> req_vec;
    req_vec.emplace_back(req);
    return new ResourceRequirement(has_arch_constraint,
                                   dev_ptr_vec, req_vec);
  }
#endif

private:
  // TODO(hc): This should use a vector of vector and
  //           the outer vector should be architecture type.
  std::vector<Device> registered_devices_;
};

#endif
