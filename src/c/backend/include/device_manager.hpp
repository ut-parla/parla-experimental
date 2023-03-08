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
    registered_devices_[new_dev->get_type()].emplace_back(new_dev);
  }

  void print_registered_devices() {
    std::cout << "C++ device list:\n";

    for (auto i = 0; i < NUM_DEVICE_TYPES; ++i) {
      std::cout << "Device type: " << i << "\n";
      auto device_list = registered_devices_[i];

      for (auto j = 0; j < device_list.size(); ++j) {
        auto device = device_list[j];
        std::cout << "Device " << j << ": " << device->get_name() << "\n";
      }
    }
  }

  template <DeviceType T> std::vector<Device *> get_devices() {
    std::vector<Device *> returned_devices;

    if constexpr (T == ANY) {
      std::vector<Device *> all_devices;
      for (auto i = 0; i < NUM_DEVICE_TYPES; ++i) {
        returned_devices.insert(returned_devices.end(),
                                registered_devices_[i].begin(),
                                registered_devices_[i].end());
      }
    } else if constexpr (T == CPU) {
      returned_devices = registered_devices_[CPU];
    } else if constexpr (T == CUDA) {
      returned_devices = registered_devices_[CUDA];
    }

    return returned_devices;
  }

protected:
  DevID_t last_dev_id_ = 0;
  std::array<std::vector<Device *>, NUM_DEVICE_TYPES> registered_devices_;
  // TODO(wlr): Add array of devices by global id.
};

#endif
