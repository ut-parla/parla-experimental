#include "include/device_manager.hpp"

void DeviceManager::RegisterDevice(Device* new_dev) {
  registered_devices_.emplace_back(new_dev);
}
