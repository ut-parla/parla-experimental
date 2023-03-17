#include "include/device.hpp"
#include "include/resource_requirements.hpp"

const bool Device::check_resource_availability(DeviceRequirement* dev_req) const {
  return get_resource_pool().check_greater<ResourceCategory::All>(dev_req->res_req());
}
