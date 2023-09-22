/*! @file device_manager.hpp
 *  @brief Provides interface for device initialization and management.
 */

#pragma once
#ifndef PARLA_DEVICE_MANAGER_HPP
#define PARLA_DEVICE_MANAGER_HPP

#include "device.hpp"

#include <chrono>
#include "resources.hpp"
#include <iostream>
#include <vector>

using DevID_t = uint32_t;

inline const DevID_t parrayid_to_globalid(int parray_dev_id) {
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

inline const int globalid_to_parrayid(DevID_t global_dev_id) {
  if (global_dev_id == 0) {
    // XXX: This assumes that a CPU device is always single and
    //      is added at first.
    return -1;
  } else {
    return static_cast<int>(global_dev_id) - 1;
  }
}

/// `DeviceManager` registers/provides devices and their
/// information on the current system to the Parla runtime.
class DeviceManager {
public:
  using TimePoint = std::chrono::time_point<std::chrono::system_clock>;

  DeviceManager() {
    this->initial_epoch_ = std::chrono::system_clock::now();
  }
  DeviceManager(const DeviceManager &) = delete;

  void register_device(ParlaDevice *new_dev) {
    new_dev->set_global_id(this->last_dev_id_++);
    new_dev->set_initial_epoch(this->initial_epoch_);
    const int idx = static_cast<int>(new_dev->get_type());
    arch_devices_[idx].emplace_back(new_dev);
    all_devices_.emplace_back(new_dev);
  }

  void reset_device_timers() {
    this->initial_epoch_ = std::chrono::system_clock::now();
    for (ParlaDevice* device : this->all_devices_) {
      device->set_initial_epoch(this->initial_epoch_);
    }
  }

  void print_registered_devices() {
    std::cout << "C++ device list:\n";
    for (ParlaDeviceType dev_type : architecture_types) {
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

  template <ParlaDeviceType T> int get_num_devices() {
    if constexpr (T == ParlaDeviceType::All) {
      return all_devices_.size();
    } else if constexpr (T == ParlaDeviceType::CPU) {
      return arch_devices_[static_cast<int>(ParlaDeviceType::CPU)].size();
    } else if constexpr (T == ParlaDeviceType::CUDA) {
      return arch_devices_[static_cast<int>(ParlaDeviceType::CUDA)].size();
    }
  }

  int get_num_devices(ParlaDeviceType dev_type) {
    switch (dev_type) {
    case ParlaDeviceType::CPU:
      return get_num_devices<ParlaDeviceType::CPU>();
    case ParlaDeviceType::CUDA:
      return get_num_devices<ParlaDeviceType::CUDA>();
    default:
      return get_num_devices<ParlaDeviceType::All>();
    }
  }

  template <ParlaDeviceType T> std::vector<ParlaDevice *> &get_devices() {
    if constexpr (T == ParlaDeviceType::CPU) {
      return arch_devices_[static_cast<int>(ParlaDeviceType::CPU)];
    } else if constexpr (T == ParlaDeviceType::CUDA) {
      return arch_devices_[static_cast<int>(ParlaDeviceType::CUDA)];
    } else if constexpr (T == ParlaDeviceType::All) {
      return all_devices_;
    }
  }

  ParlaDevice *get_device_by_parray_id(DevID_t parray_dev_id) const {
    DevID_t global_dev_id = this->parrayid_to_globalid(parray_dev_id);
    return all_devices_[global_dev_id];
  }

  ParlaDevice *get_device_by_global_id(DevID_t global_dev_id) const {
    return all_devices_[global_dev_id];
  }

  std::vector<ParlaDevice *> &get_devices(ParlaDeviceType dev_type) {
    switch (dev_type) {
    case ParlaDeviceType::CPU:
      return get_devices<ParlaDeviceType::CPU>();
    case ParlaDeviceType::CUDA:
      return get_devices<ParlaDeviceType::CUDA>();
    default:
      return get_devices<ParlaDeviceType::All>();
    }
  }

  size_t get_num_devices() { return all_devices_.size(); }

  /**
   * @brief Free both the mapped and reserved memory on the device by global
   * device id.
   */
  void free_memory(DevID_t global_dev_id, Resource_t memory_size) {

  const DevID_t globalid_to_parrayid(DevID_t global_dev_id) const {
    ParlaDevice *dev = get_device_by_global_id(global_dev_id);
    auto &mapped_memory_pool = dev->get_mapped_pool();
    auto &reserved_memory_pool = dev->get_reserved_pool();

    // Mapped memory counts how much memory is currently mapped to the device.
    // Freeing memory decreases the mapped memory pool.
    mapped_memory_pool.decrease<Resource::Memory>(memory_size);

    // Reserved memory counts how much memory is left on the device.
    // Freeing memory increases the reserved memory pool.
    reserved_memory_pool.increase<Resource::Memory>(memory_size);
  }

  /**
   * @brief Free both the mapped and reserved memory on the device by parray
   * device id. Called by a PArray eviction event.
   */
  void free_memory_by_parray_id(int parray_dev_id, Resource_t memory_size) {
    DevID_t global_dev_id = parrayid_to_globalid(parray_dev_id);
    this->free_memory(global_dev_id, memory_size);
  }

  TimePoint get_initial_epoch() { return this->initial_epoch_; }

  double current_timepoint_count_from_beginning() {
    TimePoint current_time_point = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        current_time_point - this->initial_epoch_).count();
  }

protected:
  // Global device id counter
  // When used in Scheduler, we assume that only a single device manager holds
  // all devices.
  DevID_t last_dev_id_ = 0;

  // Store devices by architecture type
  std::array<std::vector<ParlaDevice *>, NUM_DEVICE_TYPES> arch_devices_;
  // Stores all devices in the system
  std::vector<ParlaDevice *> all_devices_;
  /// System clock time point when this instance created
  TimePoint initial_epoch_;
};

#endif
