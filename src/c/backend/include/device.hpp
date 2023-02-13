#pragma once
#ifndef PARLA_DEVICE_HPP
#define PARLA_DEVICE_HPP

#include <string>
#include <vector>

enum DevID {
  CUDA_GPU = 0,
  CPU = 1,
};

// TODO(hc): To support multi-device task,
//           architecture should not be templetized,
//           but be contained into Device.
//           Otherwise, we cannot easily support
//           a vector of device.
//template <typename ArchTy>
class Device {

public:
  Device() = delete;

  Device(std::string dev_name, DevID dev_id) :
      dev_name_(dev_name), dev_id_(dev_id) {}

  /// Return a device name.
  std::string GetName() {
    return dev_name_;
  }

  /// Return a device id.
  DevID GetID() {
    return dev_id_;
  }

protected:
  std::string dev_name_;
  DevID dev_id_;
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
