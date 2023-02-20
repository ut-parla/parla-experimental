#include "include/device_manager.hpp"
#include <iostream>

void DeviceManager::RegisterDevice(Device* new_dev) {
  registered_devices_.emplace_back(*new_dev);
}
