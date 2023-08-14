#include "include/parray_tracker.hpp"
#include "include/resources.hpp"
#include <stdexcept>

PArrayTracker::PArrayTracker(DeviceManager *device_manager)
    : device_manager_(device_manager) {
  this->managed_parrays_.resize(
      device_manager->template get_num_devices<DeviceType::All>());
}

size_t
PArrayTracker::log(const DevID_t dev_id,
                   std::pair<parray::InnerPArray *, AccessMode> parray_access) {

  // Aside from modifying child_parrays_ this lock is overkill.
  // This should be refactored to be per device with this data structure.
  std::lock_guard<std::mutex> guard(this->mtx);

  const auto &[parray, access_mode] = parray_access;

  size_t to_move = parray->get_size();
  size_t self_size = parray->get_size();
  auto self_id = parray->get_id();
  bool is_tracked = this->is_valid_unsafe(parray->id, dev_id);
  bool is_slice = parray->is_subarray();

  if (!is_tracked) {

    // NOTE: This is COMPLETELY BROKEN if the subarray model changes.
    // Its probably broken anyway, but just fyi.
    // If the PArray is a slice, make sure its parent is tracked as well.
    if (is_slice) {

      // Record the parent -> child relation
      // (parray objects do not contain backlinks, only child -> parent)
      // TODO: Maybe they should, and this can be done in the PArray constructor
      // (@yinengy?)
      auto parent_id = parray->get_parent_id();
      this->add_child_parray_unsafe(parent_id, parray);
      bool is_parent_tracked = this->contains_unsafe(parent_id, dev_id);

      if (!is_parent_tracked) {
        this->set_parray_unsafe(dev_id, parent_id, false);
        // this->increase_subarray_size(dev_id, parent_id, self_size);
      } else {
        // If the parent is already logged on this device, we don't need to move
        // the child, but we do need to increase the resident size
        // this->increase_subarray_size(dev_id, parent_id, self_size);
        to_move = 0;
      }

      // If the PArray is not logged on this device, log it
      this->set_parray_unsafe(dev_id, self_id, true);
    }
  } else {
    // If the PArray is already logged on this device, we don't need to move it
    to_move = 0;
  }

  if (access_mode == AccessMode::INOUT) {
    // Invalidate the PArray on all devices except the one it is being moved to
    for (auto i = 0; i < this->managed_parrays_.size(); i++) {
      if (i != dev_id) {
        this->set_parray_unsafe(i, self_id, 0);
      }
    }

    // If the PArray is a slice, invalidate its parent on all devices except the
    // owner of the slice
    // TODO (wlr): I have no idea how parray ownership works or how to fix this.
    // I give up. I think this can't be done in the current model because the
    // owner is only known at runtime and I don't know how to predict it. int
    // owner_id = dev_id;

    if (is_slice) {
      // Invalidating the parent of a slice by a slice is not
      // supported by the Parla runtime. Please contact the
      // PArray developers to fix this.

      /*
      for (auto i = 0; i < this->managed_parrays_.size(); i++) {
        if (i != owner_id) {
          this->log_parray_unsafe(i, parray->get_parent_id(), 0);
        }
      }
      */

    } else {
      // If the PArray is not a slice, invalidate all of its children
      for (auto child : this->child_parrays_[self_id]) {
        for (auto i = 0; i < this->managed_parrays_.size(); i++) {
          if (i != dev_id) {
            this->set_parray_unsafe(i, child->get_id(), 0);
          }
        }
      }
    }
  } else if (access_mode == AccessMode::OUT) {
    throw std::runtime_error(
        "AccessMode::OUT is not supported. Please use INOUT instead.");
  }

  return to_move;
}

void PArrayTracker::track_parray(const InnerPArray &parray, DevID_t dev_id) {
  // This function is called when a PArray is mapped to a device.
  // Since `managed_parrays_` holds information of the PArrays that
  // have already instantiated or will be moved, set its state to "true".
  this->managed_parrays_[dev_id].insert(
      {parray.get_parent_id(), parray.get_size()});
}

void PArrayTracker::untrack_parray(const InnerPArray &parray, DevID_t dev_id) {
  // TODO(hc): as we don't have policy regarding this, we are not using
  //           it. we may need this if look-up operation is not cheap
  //           anymore.
  this->managed_parrays_[dev_id].erase(parray.parent_id);
}

void PArrayTracker::reserve_parray(const InnerPArray &parray, Device *device) {
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
    parray_resource.set<Resource::Memory>(parray.get_size());
    dev_mapped_pool.template increase<Resource::PersistentResources>(
        parray_resource);
    // std::cout << "[PArrayTracker] PArray ID:" << parray.id << "(parent id:"
    // <<
    //   parray.parent_id << ") allocates "
    //   << parray.get_size() << "Bytes in Device " << device->get_name() <<
    //   "\n";
  }
  this->mtx.unlock();
}

void PArrayTracker::release_parray(const InnerPArray &parray, Device *device) {
  DevID_t dev_global_id = device->get_global_id();
  std::lock_guard<std::mutex> lock(this->mtx);

  if (this->managed_parrays_[dev_global_id].find(parray.parent_id) ==
      this->managed_parrays_[dev_global_id].end()) {
    return;
  }
  if (this->managed_parrays_[dev_global_id][parray.parent_id] == true) {
    this->managed_parrays_[dev_global_id][parray.parent_id] = false;
    // Release memory for a PArray to a specified device.
    ResourcePool_t &dev_mapped_pool = device->get_mapped_pool();
    ResourcePool_t parray_resource;
    parray_resource.set<Resource::Memory>(parray.get_size());
    // XXX It is possible that the memory size of the PArray is bigger than
    // the allocated memory size in the device. This is because in the PArray
    // coherency protocol, a parent PArray of the slice evicts its slices
    // from a device and in this case, the coherency protocol can only view
    // the parent, not its subarrays, so to speak slices.
    // Therefore, it is possible that the tracker allocates the memory size of
    // the slice, as it is set at __init__() of the slice PArray and can be
    // seen, but deallocates the memory byte size of its parent PArray. This
    // makes the mapped memory size counter to a negative number, which we do
    // not prefer. Instead, we set 0 if the counter could be a negative
    // number. This is more accurate than just using parent PArray size for
    // allocation and deallocation.
    // TODO(hc): To resolve this, we may need another slice control layer for
    //           each PArray that tracks the slices.
    if (dev_mapped_pool.check_greater<Resource::PersistentResources>(
            parray_resource)) {
      dev_mapped_pool.decrease<Resource::PersistentResources>(parray_resource);
    } else {
      dev_mapped_pool.set<Resource::PersistentResources>({0});
    }
    // std::cout << "[PArrayTracker] PArray ID:" << parray.id << "(parent id:"
    // <<
    //   parray.parent_id << ") releases "
    //   << parray.get_size() << "Bytes in Device " << device->get_name() <<
    //   "\n";
  }
  this->mtx.unlock();
}
