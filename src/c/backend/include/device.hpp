#pragma once
#ifndef PARLA_DEVICE_HPP
#define PARLA_DEVICE_HPP

#include <string>
#include <vector>

using DevIDTy = uint32_t;

/// Devices can be distinguished from other devices
/// by a class type and its index.

class Device {

public:
  Device() = delete;

  Device(DevIDTy dev_id, std::string dev_name) :
      dev_id_(dev_id), dev_name_(dev_name) {}

  /// Return a device id.
  DevIDTy GetID() {
    return dev_id_;
  }

  std::string GetName() {
    return dev_name_ + ":" + std::to_string(dev_id_);
  }

protected:
  DevIDTy dev_id_;
  std::string dev_name_;
};

///
class CUDADevice : public Device {
public:
  CUDADevice(DevIDTy dev_id) : Device(dev_id, "CUDA") {}
private:
};

///
class CPUDevice : public Device {
public:
  CPUDevice(DevIDTy dev_id) : Device(dev_id, "CPU") {}
private:
};

class DeviceSet {
public:
protected:
  std::vector<Device> devices_;
};

/// `DeviceManager` registers/provides devices and their
/// information on the current system to the Parla runtime.
class DeviceManager {
public:
  DeviceManager(uint32_t num_cpus, uint32_t num_gpus) :
      num_cpus_(num_cpus), num_gpus_(num_gpus) {}
  void RegisterDevices();
  const std::vector<Device>& GetAllDevices();
private:
  std::vector<Device> registered_devices_;
  uint32_t num_cpus_;
  uint32_t num_gpus_;
};

#endif
