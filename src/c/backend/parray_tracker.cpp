#include "include/parray_tracker.hpp"

PArrayTracker::PArrayTracker(DeviceManager* device_manager) :
    device_manager_(device_manager) {
  this->managed_parrays_.resize(
      device_manager->template get_num_devices<DeviceType::All>());
}

void PArrayTracker::track_parray(const InnerPArray& parray, DevID_t dev_id) {
  // This function is called when a PArray is mapped to a device.
  // Since `managed_parrays_` holds information of the PArrays that 
  // have already instantiated or will be moved, set its state to "true". 
  this->managed_parrays_[dev_id].insert({parray.parent_id, true});
}

void PArrayTracker::untrack_parray(const InnerPArray& parray, DevID_t dev_id) {
  // TODO(hc): as we don't have policy regarding this, we are not using
  //           it. we may need this if look-up operation is not cheap
  //           anymore.
  this->managed_parrays_[dev_id].erase(parray.parent_id);
}

void PArrayTracker::reserve_parray(const InnerPArray& parray, Device* device) {
  DevID_t dev_global_id = device->get_global_id();
  if (this->managed_parrays_[dev_global_id].find(parray.parent_id) ==
          this->managed_parrays_[dev_global_id].end()) {
    this->track_parray(parray, dev_global_id);
  }
  if (this->managed_parrays_[dev_global_id][parray.parent_id] == false) {
    this->managed_parrays_[dev_global_id][parray.parent_id] = true;
    // Allocate memory for a PArray to a specified device.
    ResourcePool_t& dev_mapped_pool = device->get_mapped_pool();
    ResourcePool_t parray_resource;
    parray_resource.set(Resource::Memory, parray.get_size());
    dev_mapped_pool.template decrease<ResourceCategory::Persistent>(
        parray_resource);
    std::cout << "[PArrayTracker] PArray ID:" << parray.id << "(parent id:" <<
      parray.parent_id << ") allocates "
      << parray.get_size() << "Bytes in Device " << device->get_name() << "\n";
  }
}

void PArrayTracker::release_parray(const InnerPArray& parray, Device* device) {
  DevID_t dev_global_id = device->get_global_id();
  if (this->managed_parrays_[dev_global_id].find(parray.parent_id) ==
          this->managed_parrays_[dev_global_id].end()) {
    return;
  }
  if (this->managed_parrays_[dev_global_id][parray.parent_id] == true) {
    this->managed_parrays_[dev_global_id][parray.parent_id] = false;
    // Release memory for a PArray to a specified device.
    ResourcePool_t& dev_mapped_pool = device->get_mapped_pool();
    ResourcePool_t parray_resource;
    parray_resource.set(Resource::Memory, parray.get_size());
    dev_mapped_pool.template increase<ResourceCategory::Persistent>(
        parray_resource);
    std::cout << "[PArrayTracker] PArray ID:" << parray.id << "(parent id:" <<
      parray.parent_id << ") releases "
      << parray.get_size() << "Bytes in Device " << device->get_name() << "\n";
  }
}
