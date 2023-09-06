#ifndef PARLA_PARRAY_TRACKER_HPP
#define PARLA_PARRAY_TRACKER_HPP

#include "device.hpp"
#include "device_manager.hpp"
#include "parray.hpp"
#include "resources.hpp"
#include <algorithm>

using namespace parray;

class PArrayTracker {
public:
  PArrayTracker(size_t num_devices) : num_devices_(num_devices) {
    this->managed_parrays_.resize(num_devices);
  }

  /**
   * @brief Log the state of the PArray to the tracking table.
   * @param dev_id global ID for target device
   * @param parray_access PArray & access information pair
   * Returns the size (in bytes) to be allocated in the target device.
   */
  size_t do_log(const DevID_t dev_id, PArrayAccess_t parray_access);

  /**
   * @brief Find memory cost of the new state of the PArray to the tracking
   * table without performing the update.
   * @param dev_id global ID for target device
   * @param parray_access PArray & access information pair
   * Returns the size (in bytes) to be allocated in the target device.
   */
  size_t check_log(const DevID_t dev_id, PArrayAccess_t parray_access);

  size_t do_parray_creation_(AccessMode access_mode, DevID_t dev_id,
                             bool is_tracked, bool is_slice,
                             InnerPArray *parray);

  size_t do_parray_removal_(AccessMode access_mode, DevID_t dev_id,
                            bool is_tracked, bool is_slice,
                            InnerPArray *parray);

  size_t check_parray_creation_(AccessMode access_mode, DevID_t dev_id,
                                bool is_tracked, bool is_slice,
                                InnerPArray *parray);

  size_t check_parray_removal_(AccessMode access_mode, DevID_t dev_id,
                               bool is_tracked, bool is_slice,
                               InnerPArray *parray);

  /**
   * @brief Check if a PArray instance is located in a device (without
   * locking the table).
   */
  bool get_parray_state_unsafe(DevID_t global_dev_idx,
                               uint64_t parray_parent_id) {
    return (this->managed_parrays_[global_dev_idx][parray_parent_id] > 0);
  }

  /**
   * @brief Check if a PArray instance is located in a device (with lock).
   */
  bool get_parray_state(DevID_t global_dev_idx, uint64_t parray_parent_id) {
    mtx.lock();
    bool state =
        this->get_parray_state_unsafe(global_dev_idx, parray_parent_id);
    mtx.unlock();
    return state;
  }

  /**
   * @brief Check if a PArray instance is located in a device (without lock).
   */
  bool contains_unsafe(DevID_t global_dev_idx,
                       uint64_t parray_parent_id) const {
    // std::cout << "Checking if " << parray_parent_id << " is in Device"
    //           << global_dev_idx << std::endl;
    // std::cout << "Size of managed_parrays_: " << managed_parrays_.size()
    //           << std::endl;

    return (this->managed_parrays_[global_dev_idx].find(parray_parent_id) !=
            this->managed_parrays_[global_dev_idx].end());
  }

  /**
   * @brief Check if a PArray instance is located in a device (with lock).
   */
  bool contains(DevID_t global_dev_idx, uint64_t parray_id) {
    mtx.lock();
    bool state = this->contains_unsafe(global_dev_idx, parray_id);
    mtx.unlock();
    return state;
  }

  /**
   * @brief Check if the Parray is valid in the device.
   */
  bool is_valid_unsafe(DevID_t global_dev_idx, uint64_t parray_id) {
    if (contains_unsafe(global_dev_idx, parray_id)) {
      return this->managed_parrays_[global_dev_idx][parray_id];
    } else {
      return false;
    }
  }

  /**
   * @brief Record a parent->child relationship between two PArrays.
   */
  void add_child_parray_unsafe(uint64_t parent_id, parray::InnerPArray *child) {
    this->child_parrays_[parent_id].push_back(child);
  }

  /***
   * @brief Remove a parent->child relationship between two PArrays. (SLOW!)
   */
  void remove_child_parray_unsafe(uint64_t parent_id,
                                  parray::InnerPArray *child) {
    std::vector<InnerPArray *> &children = this->child_parrays_[parent_id];
    auto child_id = child->get_id();

    auto comparison = [child_id](InnerPArray *parray) {
      return parray->get_id() == child_id;
    };

    auto position = std::find_if(children.begin(), children.end(), comparison);

    if (position != children.end()) {
      children.erase(position);
    }
  }

  /**
   * @brief Remove a parent parray from the child relation tracking table.
   */
  void remove_parent_parray_unsafe(uint64_t parent_id) {
    this->child_parrays_.erase(parent_id);
  }

  /**
   * @brief Record a parray in the tracking table (without lock)
   */
  void set_parray_unsafe(DevID_t global_dev_idx, uint64_t parray_id,
                         bool exists) {
    auto &active_parrays = this->managed_parrays_[global_dev_idx];
    active_parrays[parray_id] = exists;
  }

  /**
   * @brief Record a parray in the tracking table (with lock)
   */
  void set_parray(DevID_t global_dev_idx, uint64_t parray_id, bool exists) {
    mtx.lock();
    this->set_parray_unsafe(global_dev_idx, parray_id, exists);
    mtx.unlock();
  }

  /**
   * @brief Remove a PArray from the tracking table.
   */
  void remove_parray_unsafe(DevID_t global_dev_idx, InnerPArray *parray) {
    this->managed_parrays_[global_dev_idx].erase(parray->get_id());

    bool is_slice = parray->is_subarray();
    if (is_slice) {
      this->remove_child_parray_unsafe(parray->get_parent_id(), parray);
    } else {
      this->remove_parent_parray_unsafe(parray->get_id());
    }
  }

  /**
   * @brief Remove a PArray from the tracking table.
   */
  void remove_parray(DevID_t global_dev_idx, InnerPArray *parray) {
    mtx.lock();
    this->remove_parray_unsafe(global_dev_idx, parray);
    mtx.unlock();
  }

protected:
  size_t num_devices_ = 0;

  /// Vector index: a device global ID
  /// Map's key: a PArray ID
  ///            (NOTE that this is the ID of the complete array)
  /// Map's value: bool (existance of the PArray in the device)
  ///
  /// Using a device ID as the first key is time efficient since
  /// we don't need to create device keys whenever we add a new PArray
  /// to the table.
  std::vector<std::unordered_map<uint64_t, bool>> managed_parrays_;
  // std::vector<std::unordered_map<uint64_t, size_t>> subarray_sizes_;

  /// A return table for all child PArrays that need to be invalidated
  /// when a parent PArray is invalidated.
  /// Map's key: a PArray ID
  /// Map's value: a vector of child PArray IDs
  std::unordered_map<uint64_t, std::vector<parray::InnerPArray *>>
      child_parrays_;

  /// Any worker thread can update states of PArrays.
  /// Guard operations by this lock.
  /// TODO(hc): This will be replaced with parallel hash map.
  std::mutex mtx;
};

class MappedPArrayTracker final : public PArrayTracker {};
class ReservedPArrayTracker final : public PArrayTracker {};

#endif
