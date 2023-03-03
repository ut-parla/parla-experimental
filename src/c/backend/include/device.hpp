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
using ResourcePool_t = ResourcePool<std::atomic<Resource_t>>;

enum DeviceType { CUDA = 1, CPU = 0 };

/// Devices can be distinguished from other devices
/// by a class type and its index.

class Device {

public:
  Device() = delete;

  Device(std::string dev_type_name, DevID_t dev_id, MemorySz_t mem_sz,
         VCU_t num_vcus, void *py_dev)
      : py_dev_(py_dev), dev_id_(dev_id), dev_type_name_(dev_type_name) {

    res_.set(VCU, num_vcus);
    res_.set(MEMORY, mem_sz);

    reserved_res_.set(VCU, num_vcus);
    reserved_res_.set(MEMORY, num_vcus);

    mapped_res_.set(VCU, 0);
    mapped_res_.set(MEMORY, 0);
  }

  /// Return a device id.
  const DevID_t GetID() const { return dev_id_; }

  const std::string GetName() const {
    return dev_type_name_ + ":" + std::to_string(dev_id_);
  }

  const MemorySz_t GetMemorySize() const { return res_.get(MEMORY); }

  const VCU_t GetNumVCUs() const { return res_.get(VCU); }

  const Resource_t GetMaxResource(Resource type) const {
    return this->res_.get(type);
  }

  const Resource_t GetReservedResource(Resource type) const {
    return this->reserved_res_.get(type);
  }

  const Resource_t GetMappedResource(Resource type) const {
    return this->mapped_res_.get(type);
  }

  const ResourcePool_t &GetResourcePool() const { return res_; }
  const ResourcePool_t &GetMappedResources() const { return mapped_res_; }
  const ResourcePool_t &GetReservedResources() const { return reserved_res_; }

  void *GetPyDevice() { return py_dev_; }

protected:
  std::string dev_type_name_;
  DevID_t dev_id_;
  ResourcePool_t res_;
  ResourcePool_t mapped_res_;
  ResourcePool_t reserved_res_;
  void *py_dev_;
  std::unordered_map<std::string, size_t> resource_map_;
};

///
class CUDADevice : public Device {
public:
  CUDADevice(DevID_t dev_id, size_t mem_sz, size_t num_vcus, void *py_dev)
      : Device("CUDA", dev_id, mem_sz, num_vcus, py_dev) {}

private:
};

///
class CPUDevice : public Device {
public:
  CPUDevice(DevID_t dev_id, size_t mem_sz, size_t num_vcus, void *py_dev)
      : Device("CPU", dev_id, mem_sz, num_vcus, py_dev) {}

private:
};
#endif
