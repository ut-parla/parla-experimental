#include "parray_state.hpp"
#include <unordered_map>
#include <vector>

namespace parray {
PArrayState::PArrayState() {}

bool PArrayState::exists_on_device(uint64_t device_id) {
  if (auto findit = this->_exist_on_device.find(device_id);
      findit != this->_exist_on_device.end()) {
    return findit->second;
  } else {
    return false;
  }
}

bool PArrayState::valid_on_device(uint64_t device_id) {
  if (auto findit = this->_valid_on_device.find(device_id);
      findit != this->_valid_on_device.end()) {
    return findit->second;
  } else {
    return false;
  }
}

void PArrayState::set_exist_on_device(uint64_t device_id, bool exist) {
  this->_exist_on_device[device_id] = exist;
}

void PArrayState::set_valid_on_device(uint64_t device_id, bool valid) {
  this->_valid_on_device[device_id] = valid;
}

std::vector<uint64_t> PArrayState::get_valid_devices() {
  std::vector<uint64_t> valid_devices;
  for (auto &kv : this->_valid_on_device) {
    if (kv.second) {
      valid_devices.push_back(kv.first);
    }
  }
  return valid_devices;
}

} // namespace parray
