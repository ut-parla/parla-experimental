/*! @file device_manager.hpp
 *  @brief Provides interface for device initialization and management.
 */

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
    } else if constexpr (T == DeviceType::GPU) {
      return arch_devices_[static_cast<int>(DeviceType::GPU)].size();
    }
  }

  int get_num_devices(DeviceType dev_type) {
    switch (dev_type) {
    case DeviceType::CPU:
      return get_num_devices<DeviceType::CPU>();
    case DeviceType::GPU:
      return get_num_devices<DeviceType::GPU>();
    default:
      return get_num_devices<DeviceType::All>();
    }
  }

  template <DeviceType T> std::vector<Device *> &get_devices() {
    if constexpr (T == DeviceType::CPU) {
      return arch_devices_[static_cast<int>(DeviceType::CPU)];
    } else if constexpr (T == DeviceType::GPU) {
      return arch_devices_[static_cast<int>(DeviceType::GPU)];
    } else if constexpr (T == DeviceType::All) {
      return all_devices_;
    }
  }

  Device *get_device_by_parray_id(DevID_t parray_dev_id) const {
    DevID_t global_dev_id = this->parrayid_to_globalid(parray_dev_id);
    return all_devices_[global_dev_id];
  }

  Device *get_device_by_global_id(DevID_t global_dev_id) const {
    return all_devices_[global_dev_id];
  }

  std::vector<Device *> &get_devices(DeviceType dev_type) {
    switch (dev_type) {
    case DeviceType::CPU:
      return get_devices<DeviceType::CPU>();
    case DeviceType::GPU:
      return get_devices<DeviceType::GPU>();
    default:
      return get_devices<DeviceType::All>();
    }
  }

  size_t get_num_devices() { return all_devices_.size(); }

  // TODO(hc): use a customized type for device id.

  const int globalid_to_parrayid(unsigned int global_dev_id) const {
    Device *dev = all_devices_[global_dev_id];
    if (dev->get_type() == DeviceType::CPU) {
      return -1;
    } else {
      return dev->get_id();
    }
  }

  const unsigned int parrayid_to_globalid(int parray_dev_id) const {
    if (parray_dev_id == -1) {
      // XXX: This assumes that a CPU device is always single and
      //      is added at first.
      //      Otherwise, we need a loop iterating all devices and
      //      comparing device ids.
      return 0;
    } else {
      return parray_dev_id + 1;
    }
  }

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
