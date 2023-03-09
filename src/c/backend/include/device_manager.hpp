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
    all_devices_.emplace_back(new_dev);
  }

  void print_registered_devices() {
    std::cout << "C++ device list:\n";

    for (DeviceType i : architecture_types) {
      std::cout << "Device type: " << i << "\n";
      auto device_list = registered_devices_[i];

      for (auto j = 0; j < device_list.size(); ++j) {
        auto device = device_list[j];
        std::cout << "Device " << j << ": " << device->get_name() << "\n";
      }
    }
  }

  // TODO(wlr): err Sorry! I was playing around with these. Feel free to revert.
  //  I had thought a compile time dispatch would be nice, but forgot it would
  //  need a specialized handler class to do runtime inference for each enum
  //  type..
  template <DeviceType T> int get_num_devices() {
    if constexpr (T == ANY) {
      return all_devices_.size();
    } else if constexpr (T == CPU) {
      return registered_devices_[CPU].size();
    } else if constexpr (T == CUDA) {
      return registered_devices_[CUDA].size();
    }
  }

  int get_num_devices(DeviceType dev_type) {
    switch (dev_type) {
    case CPU:
      return get_num_devices<CPU>();
    case CUDA:
      return get_num_devices<CUDA>();
    default:
      return get_num_devices<ANY>();
    }
  }

  template <DeviceType T> std::vector<Device *> &get_devices() {
    if constexpr (T == CPU) {
      return registered_devices_[CPU];
    } else if constexpr (T == CUDA) {
      return registered_devices_[CUDA];
    } else if constexpr (T == ANY) {
      return all_devices_;
    }
  }

  std::vector<Device *> &get_devices(DeviceType dev_type) {
    switch (dev_type) {
    case CPU:
      return get_devices<CPU>();
    case CUDA:
      return get_devices<CUDA>();
    default:
      return get_devices<ANY>();
    }
  }

protected:
  DevID_t last_dev_id_ = 0;
  std::array<std::vector<Device *>, NUM_DEVICE_TYPES> registered_devices_;
  std::vector<Device *> all_devices_;
};

#endif
