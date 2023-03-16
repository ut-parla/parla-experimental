#include "include/device.hpp"
#include "include/resource_requirements.hpp"

const bool Device::check_resource_availability(DeviceRequirement* dev_req) const {
  return this->get_max_resource(Resource::Memory) >= dev_req->res_req().get(Resource::Memory)
    and this->get_max_resource(Resource::VCU) >= dev_req->res_req().get(Resource::VCU);
}
