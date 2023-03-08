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

enum DeviceType { ANY = -1, CPU = 0, CUDA = 1 };

inline const constexpr int NUM_DEVICE_TYPES = 2;
inline const std::array<std::string, NUM_DEVICE_TYPES> architecture_names{
    "CPU", "CUDA"};

/// Devices can be distinguished from other devices
/// by a class type and its index.
class Device {

public:
  Device() = delete;

  Device(DeviceType arch, DevID_t dev_id, MemorySz_t mem_sz, VCU_t num_vcus,
         void *py_dev)
      : py_dev_(py_dev), dev_id_(dev_id), dev_type_(arch) {

    res_.set(VCU, num_vcus);
    res_.set(MEMORY, mem_sz);

    reserved_res_.set(VCU, num_vcus);
    reserved_res_.set(MEMORY, num_vcus);

    mapped_res_.set(VCU, 0);
    mapped_res_.set(MEMORY, 0);
  }

  /// Return a device id.
  const DevID_t get_id() const { return dev_id_; }

  const std::string get_name() const {
    return architecture_names[this->dev_type_] + ":" + std::to_string(dev_id_);
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

  const ResourcePool_t &get_resource_pool() const { return res_; }
  const ResourcePool_t &get_mapped_pool() const { return mapped_res_; }
  const ResourcePool_t &get_reserved_pool() const { return reserved_res_; }

  void *get_py_device() { return py_dev_; }

  void set_global_id(DevID_t global_id) { dev_global_id_ = global_id; }
  const DevID_t get_global_id() const { return dev_global_id_; }

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
class CUDADevice : public Device {
public:
  CUDADevice(DevID_t dev_id, size_t mem_sz, size_t num_vcus, void *py_dev)
      : Device(CUDA, dev_id, mem_sz, num_vcus, py_dev) {}

private:
};

///
class CPUDevice : public Device {
public:
  CPUDevice(DevID_t dev_id, size_t mem_sz, size_t num_vcus, void *py_dev)
      : Device(CPU, dev_id, mem_sz, num_vcus, py_dev) {}

private:
};
#endif
