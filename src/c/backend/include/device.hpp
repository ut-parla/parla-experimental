#pragma once
#ifndef PARLA_DEVICE_HPP
#define PARLA_DEVICE_HPP

#include <vector>

enum ArchID {
  CUDA_GPU = 0,
  CPU = 1,
};

/// `Architecture` class defines a type and provides information of an
/// architecture.
class Architecture {
public:
  Architecture() = delete;

  Architecture(std::string arch_name, ArchID arch_id) :
      arch_name_(arch_name), arch_id_(arch_id) {}

  /// Return an architecture name.
  std::string GetName() {
    return arch_name_; 
  }

  /// Return an architecture id.
  ArchID GetID() {
    return arch_id_;  
  }

private:
  std::string arch_name_;
  ArchID arch_id_;
};

///
class CUDAArch : Architecture {
public:
  CUDAArch() : Architecture("CUDA GPU", CUDA_GPU) {}
};

///
class CPUArch : Architecture {
public:
 CPUArch() : Architecture("CPU", CPU) {}
};

// TODO(hc): To support multi-device task,
//           architecture should not be templetized,
//           but be contained into Device.
//           Otherwise, we cannot easily support
//           a vector of device.
//template <typename ArchTy>
class Device {
public:
protected:
};

class CUDADevice : public Device {
private:
  CUDAArch arch_;
};

class CPUDevice : public Device {
private:
  CPUArch arch_;
};

class DeviceSet {
public:
protected:
  std::vector<Device> devices_;
};

#endif
