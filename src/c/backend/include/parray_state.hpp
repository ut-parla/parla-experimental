#pragma once
#include "device.hpp"
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace parray {
// A Class that keep record of PArray's state
// Multiple PArray may share the same state object
class PArrayState {
public:
  PArrayState();

  // Return True if there is an PArray copy (possibly invalid) on this device
  bool exists_on_device(int device_id);

  // Return True if there is an PArray copy and its coherence state is valid on
  // this device
  bool valid_on_device(int device_id);

  // set the exist status of PArray on a device
  void set_exist_on_device(int device_id, bool exist);

  // set the valid status of PArray on a device
  void set_valid_on_device(int device_id, bool valid);

  // get vector of valid devices
  std::vector<int> get_valid_devices();

private:
  std::unordered_map<int, bool>
      _exist_on_device; // a mapping between device_id and exist status
  std::unordered_map<int, bool>
      _valid_on_device; // a mapping between device_id and valid status
};
} // namespace parray