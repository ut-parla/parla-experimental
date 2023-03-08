#pragma once
#ifndef PARLA_DEVICE_HPP
#define PARLA_DEVICE_HPP

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

using DevIDTy = uint32_t;
// `-1` is used to disable resource constraints.
// In this case, a task can be mapped to any compatible
// regardless of its resource states.
using MemorySzTy = int64_t;
using VCUTy = int32_t;

// TODO(hc): This will be a dictionary in the later.
struct DeviceResources {
  /// Supporting device resources.
  MemorySzTy mem_sz; /* Memory size. */
  VCUTy num_vcus;    /* The number of virtual computing units (VCU). */
};

enum DeviceType { CUDA = 1, CPU = 0 };

/// Devices can be distinguished from other devices
/// by a class type and its index.

class Device {

public:
  Device() = delete;

  Device(std::string dev_type_name, DevIDTy dev_id, MemorySzTy mem_sz,
         VCUTy num_vcus, void* py_dev)
      : py_dev_(py_dev), dev_id_(dev_id),
        res_(DeviceResources{mem_sz, num_vcus}), dev_type_name_(dev_type_name) {
  }

  /// Return a device id.
  const DevIDTy GetID() const { return dev_id_; }

  const std::string GetName() const {
    return dev_type_name_ + ":" + std::to_string(dev_id_);
  }

  const MemorySzTy GetMemorySize() const { return res_.mem_sz; }

  const VCUTy GetNumVCUs() const { return res_.num_vcus; }

  void* GetPyDevice() { return py_dev_; }

protected:
  std::string dev_type_name_;
  DevIDTy dev_id_;
  DeviceResources res_;
  void* py_dev_;
  std::unordered_map<std::string, size_t> resource_map_;
};

///
class CUDADevice : public Device {
public:
  CUDADevice(DevIDTy dev_id, size_t mem_sz, size_t num_vcus, void* py_dev)
      : Device("CUDA", dev_id, mem_sz, num_vcus, py_dev) {}

private:
};

///
class CPUDevice : public Device {
public:
  CPUDevice(DevIDTy dev_id, size_t mem_sz, size_t num_vcus, void* py_dev)
      : Device("CPU", dev_id, mem_sz, num_vcus, py_dev) {}

private:
};
#endif
