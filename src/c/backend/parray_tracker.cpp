#include "include/parray_tracker.hpp"

PArrayTracker::PArrayTracker(DeviceManager *device_manager)
    : device_manager_(device_manager) {
  this->managed_parrays_.resize(
      device_manager->template get_num_devices<DeviceType::All>());
}

void PArrayTracker::track_parray(const InnerPArray &parray, DevID_t dev_id) {
  // This function is called when a PArray is mapped to a device.
  // Since `managed_parrays_` holds information of the PArrays that
  // have already instantiated or will be moved, set its state to "true".
  this->managed_parrays_[dev_id].insert({parray.parent_id, true});
}

void PArrayTracker::untrack_parray(const InnerPArray &parray, DevID_t dev_id) {
  // TODO(hc): as we don't have policy regarding this, we are not using
  //           it. we may need this if look-up operation is not cheap
  //           anymore.
  this->managed_parrays_[dev_id].erase(parray.parent_id);
}

void PArrayTracker::reserve_parray_to_tracker(const InnerPArray &parray, Device *device) {
  DevID_t dev_global_id = device->get_global_id();
  bool first_reservation{false};
  this->mtx.lock();
  if (this->managed_parrays_[dev_global_id].find(parray.parent_id) ==
      this->managed_parrays_[dev_global_id].end()) {
    this->track_parray(parray, dev_global_id);
    first_reservation = true;
  }
  if (this->managed_parrays_[dev_global_id][parray.parent_id] == false or
      first_reservation) {
    this->managed_parrays_[dev_global_id][parray.parent_id] = true;
    // Allocate memory for a PArray to a specified device.
    ResourcePool_t &dev_mapped_pool = device->get_mapped_pool();
    ResourcePool_t parray_resource;
    parray_resource.set(Resource::Memory, parray.get_size());
    dev_mapped_pool.template increase<ResourceCategory::Persistent>(
        parray_resource);
    // std::cout << "[PArrayTracker] PArray ID:" << parray.id << "(parent id:"
    // <<
    //   parray.parent_id << ") allocates "
    //   << parray.get_size() << "Bytes in Device " << device->get_name() <<
    //   "\n";
  }
  this->mtx.unlock();
}

void PArrayTracker::release_parray_from_tracker(const InnerPArray &parray, Device *device) {
  DevID_t dev_global_id = device->get_global_id();
  this->mtx.lock();
  if (this->managed_parrays_[dev_global_id].find(parray.parent_id) ==
      this->managed_parrays_[dev_global_id].end()) {
    return;
  }
  if (this->managed_parrays_[dev_global_id][parray.parent_id] == true) {
    this->managed_parrays_[dev_global_id][parray.parent_id] = false;
    // Release memory for a PArray from a specified device.
    ResourcePool_t &dev_mapped_pool = device->get_mapped_pool();
    ResourcePool_t parray_resource;
    parray_resource.set(Resource::Memory, parray.get_size());
    // XXX It is possible that the memory size of the PArray is bigger than
    // the allocated memory size in the device. This is because in the PArray
    // coherency protocol, a parent PArray of the slice evicts its slices
    // from a device and in this case, the coherency protocol can only view
    // the parent, not its subarrays, so to speak slices.
    // Therefore, it is possible that the tracker allocates the memory size of
    // the slice, as it is set at __init__() of the slice PArray and can be
    // seen, but deallocates the memory byte size of its parent PArray. This
    // makes the mapped memory size counter to a negative number, which we do
    // not prefer. Instead, we set 0 if the counter could be a negative number.
    // This is more accurate than just using parent PArray size for allocation
    // and deallocation.
    // TODO(hc): To resolve this, we may need another slice control layer for
    //           each PArray that tracks the slices.
    if (dev_mapped_pool.template check_greater<ResourceCategory::Persistent>(
            parray_resource)) {
      dev_mapped_pool.template decrease<ResourceCategory::Persistent>(
          parray_resource);
    } else {
      dev_mapped_pool.set(Resource::Memory, 0);
    }
    // std::cout << "[PArrayTracker] PArray ID:" << parray.id << "(parent id:"
    // <<
    //   parray.parent_id << ") releases "
    //   << parray.get_size() << "Bytes in Device " << device->get_name() <<
    //   "\n";
  }
  this->mtx.unlock();
}
