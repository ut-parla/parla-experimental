#pragma once
#ifndef PARLA_DEVICE_HPP
#define PARLA_DEVICE_HPP

#include <string>
#include <vector>

using DevIDTy = uint32_t;

/// Devices can be distinguished from other devices
/// by a class type and its index.

// TODO(hc): To support multi-device task,
//           architecture should not be templetized,
//           but be contained into Device.
//           Otherwise, we cannot easily support
//           a vector of device.
//template <typename ArchTy>
class Device {

public:
  Device() = delete;

  Device(DevIDTy dev_id) : dev_id_(dev_id) {}

  /// Return a device id.
  DevIDTy GetID() {
    return dev_id_;
  }

protected:
  DevIDTy dev_id_;
};

class CUDADevice : public Device {
public:
  using Device::Device; // Inherit the parent constructor.
private:
};

class CPUDevice : public Device {
public:
  using Device::Device; // Inherit the parent constructor.
private:
};

class DeviceSet {
public:
protected:
  std::vector<Device> devices_;
};

class EnvironmentManager {
public:
  void InitializeEnvironments();
  std::vector<Device>& GetAllDevices();
private:
  std::vector<Device> managed_devices_;
};

#endif
