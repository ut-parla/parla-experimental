#pragma once
#ifndef PARLA_DEVICE_HPP
#define PARLA_DEVICE_HPP

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

using DevIDTy = uint32_t;

/// Devices can be distinguished from other devices
/// by a class type and its index.

class Device {

public:
  Device() = delete;

  Device(std::string dev_type_name, DevIDTy dev_id, size_t mem_sz,
         size_t num_vcus, void* py_dev) :
         py_dev_(py_dev), dev_id_(dev_id), mem_sz_(mem_sz), num_vcus_(num_vcus),
         dev_type_name_(dev_type_name) {}

  /// Return a device id.
  DevIDTy GetID() {
    return dev_id_;
  }

  std::string GetName() {
    return dev_type_name_ + ":" + std::to_string(dev_id_);
  }

  size_t GetMemorySize() {
    return mem_sz_; 
  }

  size_t GetNumVCUs() {
    return num_vcus_;
  }

  void* GetPyDevice() {
    return py_dev_;
  }

protected:
  std::string dev_type_name_;
  DevIDTy dev_id_;
  size_t mem_sz_;
  size_t num_vcus_;
  void* py_dev_;
  std::unordered_map<std::string, size_t> resource_map_;
};

///
class CUDADevice : public Device {
public:
  CUDADevice(DevIDTy dev_id, size_t mem_sz,
             size_t num_vcus, void* py_dev) :
             Device("CUDA", dev_id, mem_sz, num_vcus, py_dev) {}
private:
};

///
class CPUDevice : public Device {
public:
  CPUDevice(DevIDTy dev_id, size_t mem_sz,
            size_t num_vcus, void* py_dev) :
            Device("CPU", dev_id, mem_sz, num_vcus, py_dev) {}
private:
};

class DeviceSet {
public:
  /// This extracts python device objects, and
  /// constructs and returns a vector of them.
  std::vector<void*> GetPyDevices();
private:
  /// A device object in this vector also contains the
  /// correpsonding python device object.
  std::vector<Device> devices_;
};

#endif
