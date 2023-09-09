#include "include/parray_tracker.hpp"
#include "include/device.hpp"
#include "include/parray.hpp"
#include "include/resources.hpp"
#include <atomic>
#include <stdexcept>

size_t PArrayTracker::do_parray_creation_(AccessMode access_mode,
                                          DevID_t dev_id, bool is_tracked,
                                          bool is_slice, InnerPArray *parray) {

  if (access_mode >= AccessMode::OUT || is_tracked) {
    return 0;
  } else {
    size_t to_move = parray->get_size();

    // std::cout << "PArrayTracker::do_parray_creation " << std::endl;

    this->set_parray_unsafe(dev_id, parray->get_id(), true);

    // std::cout << "PArrayTracker::do_parray_creation set state" << std::endl;

    if (is_slice) {

      auto parent_id = parray->get_parent_id();

      if (access_mode == AccessMode::NEW) {
        // Record the parent -> child relation
        // parray objects do not contain backlinks
        // TODO(@yinengy): Maybe they should?
        this->add_child_parray_unsafe(parent_id, parray);
      }

      bool is_parent_tracked = this->is_valid_unsafe(dev_id, parent_id);

      if (is_parent_tracked) {
        // If the parent is on this device, we don't need to move data
        to_move = 0;
      } else {
        // Otherwise, add the parent to the table
        this->set_parray_unsafe(dev_id, parent_id, false);

        // Note (@dialecticDolt):
        // If this is AccessMode::NEW, then this is an error
      }
    }

    return to_move;
  }
}

size_t PArrayTracker::do_parray_removal_(AccessMode access_mode, DevID_t dev_id,
                                         bool is_tracked, bool is_slice,
                                         InnerPArray *parray) {

  // std::cout << "PArrayTracker::do_parray_removal " << std::endl;

  if (access_mode >= AccessMode::INOUT && access_mode != AccessMode::DELETED) {

    auto parray_id = parray->get_id();

    // Invalidate the PArray on all devices except the target
    for (auto i = 0; i < this->managed_parrays_.size(); i++) {
      this->set_parray_unsafe(i, parray_id, false);
    }

    if (access_mode != AccessMode::REMOVED) {
      this->set_parray_unsafe(dev_id, parray_id, false);
    }

    if (is_slice) {

      // If the PArray is a slice, invalidate its parent on all devices except
      // the device that will be the owner of the slice at the time of the
      // removal
      // TODO (@dialecticDolt): I have no idea how parray ownership works or how
      // to fix this. I give up. I think this can't be done in the current model
      // because the owner is only known at runtime and I don't know how to
      // predict it.

      // int owner_id = dev_id;

      // Invalidating the parent of a slice by a slice is not
      // supported by the Parla runtime. Please contact the
      // PArray developers to fix this.

      /*
      for (auto i = 0; i < this->managed_parrays_.size(); i++) {
        if (i != owner_id) {
          this->log_parray_unsafe(i, parray->get_parent_id(), false);
        }
      }
      */
    } else {
      // If the PArray is not a slice, it is a parent
      // invalidate all its children on all devices except the target
      for (auto child : this->child_parrays_[parray_id]) {

        auto child_id = child->get_id();

        for (auto i = 0; i < this->managed_parrays_.size(); i++) {
          this->set_parray_unsafe(i, child_id, false);
        }

        if (access_mode != AccessMode::REMOVED) {
          this->set_parray_unsafe(dev_id, child_id, false);
        }
      }
    }
  } else if (access_mode == AccessMode::DELETED) {
    // If the PArray is deleted, remove it from the table
    this->remove_parray_unsafe(dev_id, parray);
  }

  // Note(@dialecticDolt): I don't have any cost estimates for data movement
  // caused by removals/writes
  return 0;
}

size_t PArrayTracker::do_log(const DevID_t dev_id,
                             PArrayAccess_t parray_access) {

  // Aside from modifying child_parrays_ this lock is overkill.
  // This should be refactored to be per device with this data structure.
  std::lock_guard<std::mutex> guard(this->mtx);

  const auto &[parray, access_mode] = parray_access;

  // std::cout << "PArrayTracker::do_log unpacked" << std::endl;

  bool is_valid = this->is_valid_unsafe(dev_id, parray->get_id());

  // std::cout << "Checked if tracked " << is_valid << std::endl;

  bool is_slice = parray->is_subarray();

  // std::cout << "Checked if slice: " << is_slice << std::endl;

  size_t to_move = this->do_parray_creation_(access_mode, dev_id, is_valid,
                                             is_slice, parray);

  to_move +=
      this->do_parray_removal_(access_mode, dev_id, is_valid, is_slice, parray);

  return to_move;
}

size_t PArrayTracker::check_parray_creation_(AccessMode access_mode,
                                             DevID_t dev_id, bool is_tracked,
                                             bool is_slice,
                                             InnerPArray *parray) {

  if (access_mode >= AccessMode::OUT || is_tracked) {
    return 0;
  } else {

    size_t to_move = parray->get_size();

    if (is_slice) {

      auto parent_id = parray->get_parent_id();
      bool is_parent_tracked = this->is_valid_unsafe(dev_id, parent_id);

      if (is_parent_tracked) {
        // If the parent is on this device, we don't need to move data
        to_move = 0;
      } else {
        // Note (@dialecticDolt):
        // If this is AccessMode::NEW, then this is an error
      }
    }

    return to_move;
  }
}

size_t PArrayTracker::check_parray_removal_(AccessMode access_mode,
                                            DevID_t dev_id, bool is_tracked,
                                            bool is_slice,
                                            InnerPArray *parray) {
  // Note(@dialecticDolt): I don't have any cost estimates for data movement
  // caused by removals/writes
  return 0;
}

size_t PArrayTracker::check_log(const DevID_t dev_id,
                                PArrayAccess_t parray_access) {

  // Aside from modifying child_parrays_ this lock is overkill.
  // This should be refactored to be per device with this data structure.
  std::lock_guard<std::mutex> guard(this->mtx);

  // std::cout << "PArrayTracker::check_log" << std::endl;

  const auto &[parray, access_mode] = parray_access;

  // std::cout << "PArrayTracker::check_log unpacked" << std::endl;

  bool is_tracked = this->is_valid_unsafe(dev_id, parray->get_id());

  // std::cout << "Checked if tracked " << is_tracked << std::endl;

  bool is_slice = parray->is_subarray();

  // std::cout << "Checked if slice: " << is_slice << std::endl;

  size_t to_move = this->check_parray_creation_(access_mode, dev_id, is_tracked,
                                                is_slice, parray);

  // std::cout << "Checked parray creation: " << to_move << std::endl;

  to_move += this->check_parray_removal_(access_mode, dev_id, is_tracked,
                                         is_slice, parray);

  // std::cout << "Checked parray removal: " << to_move << std::endl;

  return to_move;
}
