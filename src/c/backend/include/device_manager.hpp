#pragma once
#ifndef PARLA_DEVICE_MANAGER_HPP
#define PARLA_DEVICE_MANAGER_HPP

#include "device.hpp"

#include <iostream>
#include <vector>

using DevID_t = uint32_t;

/// `DeviceManager` registers/provides devices and their
/// information on the current system to the Parla runtime.
class DeviceManager {
public:
  DeviceManager() {}
  DeviceManager(const DeviceManager &) = delete;

  void register_device(Device *new_dev) {
    new_dev->set_global_id(this->last_dev_id_++);
    const int idx = static_cast<int>(new_dev->get_type());
    arch_devices_[idx].emplace_back(new_dev);
    all_devices_.emplace_back(new_dev);
  }

  void print_registered_devices() {
    std::cout << "C++ device list:\n";
    for (DeviceType dev_type : architecture_types) {
      int i = static_cast<int>(dev_type);
      std::cout << "Device type: " << i << "\n";
      auto device_list = arch_devices_[i];

      for (auto j = 0; j < device_list.size(); ++j) {
        auto device = device_list[j];
        std::cout << "Device " << j << ": " << device->get_name()
                  << "\n\t mem. sz:" << device->get_memory_size()
                  << ", num. vcus:" << device->get_num_vcus() << "\n";
      }
    }
  }

  template <DeviceType T> int get_num_devices() {
    if constexpr (T == DeviceType::All) {
      return all_devices_.size();
    } else if constexpr (T == DeviceType::CPU) {
      return arch_devices_[static_cast<int>(DeviceType::CPU)].size();
    } else if constexpr (T == DeviceType::CUDA) {
      return arch_devices_[static_cast<int>(DeviceType::CUDA)].size();
    }
  }

  int get_num_devices(DeviceType dev_type) {
    switch (dev_type) {
    case DeviceType::CPU:
      return get_num_devices<DeviceType::CPU>();
    case DeviceType::CUDA:
      return get_num_devices<DeviceType::CUDA>();
    default:
      return get_num_devices<DeviceType::All>();
    }
  }

  template <DeviceType T> std::vector<Device *> &get_devices() {
    if constexpr (T == DeviceType::CPU) {
      return arch_devices_[static_cast<int>(DeviceType::CPU)];
    } else if constexpr (T == DeviceType::CUDA) {
      return arch_devices_[static_cast<int>(DeviceType::CUDA)];
    } else if constexpr (T == DeviceType::All) {
      return all_devices_;
    }
  }

  std::vector<Device *> &get_devices(DeviceType dev_type) {
    switch (dev_type) {
    case DeviceType::CPU:
      return get_devices<DeviceType::CPU>();
    case DeviceType::CUDA:
      return get_devices<DeviceType::CUDA>();
    default:
      return get_devices<DeviceType::All>();
    }
  }

  size_t get_num_devices() { return all_devices_.size(); }

protected:
  // Global device id counter
  // When used in Scheduler, we assume that only a single device manager holds
  // all devices.
  DevID_t last_dev_id_ = 0;

  // Store devices by architecture type
  std::array<std::vector<Device *>, NUM_DEVICE_TYPES> arch_devices_;
  // Stores all devices in the system
  std::vector<Device *> all_devices_;
};

#endif
