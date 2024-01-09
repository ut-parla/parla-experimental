/*! @file device.hpp
 *  @brief Provides interface for abstract device object.
 */

#pragma once
#ifndef PARLA_DEVICE_HPP
#define PARLA_DEVICE_HPP

#include "resources.hpp"
#include <atomic>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

using DevID_t = uint32_t;
using MemorySz_t = Resource_t;
using VCU_t = Resource_t;
// using ResourcePool_t = ResourcePool<std::atomic<Resource_t>>;
using ResourcePool_t = ResourcePool;

class DeviceRequirement;

/**
 * @brief Architecture types for devices.
 */
enum class DeviceType { INVALID = -2, All = -1, CPU = 0, GPU = 1 };

inline const constexpr std::array architecture_types{DeviceType::CPU,
                                                     DeviceType::GPU};
inline const constexpr int NUM_DEVICE_TYPES = architecture_types.size();
inline const std::array<std::string, NUM_DEVICE_TYPES> architecture_names{
    "CPU", "GPU"};

/// Devices can be distinguished from other devices
/// by a class type and its index.
class Device {

public:
  Device() = delete;

  Device(DeviceType arch, DevID_t dev_id, MemorySz_t mem_sz, VCU_t num_vcus,
         void *py_dev, int copy_engines = 2)
      : py_dev_(py_dev), dev_id_(dev_id), dev_type_(arch) {

    res_.set(Resource::VCU, num_vcus);
    res_.set(Resource::Memory, mem_sz);
    res_.set(Resource::Copy, copy_engines);

    reserved_res_.set(Resource::VCU, num_vcus);
    reserved_res_.set(Resource::Memory, mem_sz);
    reserved_res_.set(Resource::Copy, copy_engines);

    mapped_res_.set(Resource::VCU, 0);
    mapped_res_.set(Resource::Memory, 0);
    mapped_res_.set(Resource::Copy, 0);
  }

  /// Return a device id.
  const DevID_t get_id() const { return dev_id_; }

  const std::string get_name() const {
    return architecture_names[static_cast<int>(this->dev_type_)] + ":" +
           std::to_string(dev_id_);
  }

  const Resource_t query_resource(Resource type) const {
    return this->res_.get(type);
  }

  const Resource_t query_reserved_resource(Resource type) const {
    return this->reserved_res_.get(type);
  }

  const Resource_t query_mapped_resource(Resource type) const {
    return this->mapped_res_.get(type);
  }

  const DeviceType get_type() const { return dev_type_; }

  // Comment(wlr): Maybe max resource pool should be const?

  /**
   * @brief Returns the device details (maximum resources available)
   * This is assumed to be constant after device creation.
   */
  const ResourcePool_t &get_resource_pool() const { return res_; }

  /**
   * @brief Returns the currently mapped resources on the device.
   * This starts at 0 and increases as resources are mapped.
   * Decreased when resources are released at the end of a task.
   * This is not runtime necessary, but useful to mapping policy.
   */
  ResourcePool_t &get_mapped_pool() { return mapped_res_; }

  /**
   * @brief Returns the currently reserved resources on the device.
   * This starts at max and decreases as resources are reserved.
   * This represents the resources currently in use by the tasks.
   * This is necessary to determine if tasks can be scheduled without
   * oversubscription or OOM errors.
   */
  ResourcePool_t &get_reserved_pool() { return reserved_res_; }

  /**
   * @brief Returns the pointer to the python device object.
   */
  void *get_py_device() { return py_dev_; }
  void set_global_id(DevID_t global_id) { dev_global_id_ = global_id; }
  const DevID_t get_global_id() const { return dev_global_id_; }

  const MemorySz_t get_memory_size() const {
    return res_.get(Resource::Memory);
  }

  const VCU_t get_num_vcus() const { return res_.get(Resource::VCU); }

  const Resource_t get_max_resource(Resource type) const {
    return this->res_.get(type);
  }

  const Resource_t get_reserved_resource(Resource type) const {
    return this->reserved_res_.get(type);
  }

  const Resource_t get_mapped_resource(Resource type) const {
    return this->mapped_res_.get(type);
  }

  const bool check_resource_availability(DeviceRequirement *dev_req) const;

protected:
  DeviceType dev_type_;
  DevID_t dev_id_;
  DevID_t dev_global_id_;
  ResourcePool_t res_;
  ResourcePool_t mapped_res_;
  ResourcePool_t reserved_res_;
  void *py_dev_;
  std::unordered_map<std::string, size_t> resource_map_;
};

///
class GPUDevice : public Device {
public:
  GPUDevice(DevID_t dev_id, size_t mem_sz, size_t num_vcus, void *py_dev)
      : Device(DeviceType::GPU, dev_id, mem_sz, num_vcus, py_dev, 3) {}

private:
};

///
class CPUDevice : public Device {
public:
  CPUDevice(DevID_t dev_id, size_t mem_sz, size_t num_vcus, void *py_dev)
      : Device(DeviceType::CPU, dev_id, mem_sz, num_vcus, py_dev, 4) {}

private:
};
#endif
