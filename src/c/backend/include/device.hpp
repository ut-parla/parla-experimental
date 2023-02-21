#pragma once
#ifndef PARLA_DEVICE_HPP
#define PARLA_DEVICE_HPP

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

using DevIDTy = uint32_t;

// TODO(hc): This will be a dictionary in the later.
struct DeviceResources {
  /// Supporting device resources.
  size_t mem_sz_; /* Memory size. */
  size_t num_vcus_; /* The number of virtual computing units (VCU). */
};

enum DeviceType {
  CUDA = 1,
  CPU = 0
};

/// Devices can be distinguished from other devices
/// by a class type and its index.

class Device {

public:
  Device() = delete;

  Device(std::string dev_type_name, DevIDTy dev_id, size_t mem_sz,
         size_t num_vcus, void* py_dev) :
         py_dev_(py_dev), dev_id_(dev_id), res_(DeviceResources{mem_sz, num_vcus}),
         dev_type_name_(dev_type_name) {}

  /// Return a device id.
  DevIDTy GetID() {
    return dev_id_;
  }

  std::string GetName() {
    return dev_type_name_ + ":" + std::to_string(dev_id_);
  }

  size_t GetMemorySize() {
    return res_.mem_sz_; 
  }

  size_t GetNumVCUs() {
    return res_.num_vcus_;
  }

  void* GetPyDevice() {
    return py_dev_;
  }

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

/// This class contains a single resource requirement for devices 
/// in the same architecture type for a task.
class ResourceRequirement {
public:
  ResourceRequirement() = delete;
  ResourceRequirement(std::vector<bool> has_arch_constraint,
                      std::vector<std::vector<Device*>> dev_ptr_vec,
                      std::vector<DeviceResources> reqs) :
                      has_arch_constraint_(std::move(has_arch_constraint)),
                      dev_ptr_vec_(std::move(dev_ptr_vec)),
                      reqs_(std::move(reqs)) {}

  // TODO(hc): From the factory function in the device manager,
  //           accumulate (merge) another requirement.
  //           This is for multi-arch or arch requirement.
private:
  std::vector<bool> has_arch_constraint_;
  std::vector<std::vector<Device*>> dev_ptr_vec_;
  std::vector<DeviceResources> reqs_;
};

#if 0
// Base class for device requirement classes.
class ResourceRequirement {};

/// This class contains resource requirements for a single device
/// for a task.
class SingleDevRequirement : ResourceRequirement {
public:
  SingleDevRequirement() = delete;
  SingleDevRequirement(Device* dev_ptr, DeviceResources res_req) :
                       dev_ptr_(dev_ptr), res_req_(res_req) {}

private:
  Device* dev_ptr_;
  DeviceResources res_req_;
};

/// This class contains a single resource requirement for
/// devices in a single and the same architecture for a task.
class SingleArchRequirement : ResourceRequirement {
public:
  SingleArchRequirement(DeviceType dev_type, std::vector<Device*> dev_ptr_vec,
                        DeviceResources res_req) : dev_type_(dev_type),
                        dev_ptr_vec_(std::move(dev_ptr_vec)),
                        res_req_(res_req) {}
private:
  DeviceType dev_type_;
  std::vector<Device*> dev_ptr_vec_;
  DeviceResources res_req_;
};

/// This class contains a single resource requirement for devices 
/// in the same architecture type for a task.
class MultiArchsRequirement : ResourceRequirement {
public:
  MultiArchsRequirement(std::vector<bool> has_arch_constraint,
                        std::vector<std::vector<Device*>> dev_ptr_vec,
                        std::vector<DeviceResources> reqs) :
                        has_arch_constraint_(std::move(has_arch_constraint)),
                        dev_ptr_vec_(std::move(dev_ptr_vec)),
                        reqs_(std::move(reqs)) {}

private:
  std::vector<bool> has_arch_constraint_;
  std::vector<std::vector<Device*>> dev_ptr_vec_;
  std::vector<DeviceResources> reqs_;
};
#endif

#endif
