#ifndef PARLA_PARRAY_TRACKER_HPP
#define PARLA_PARRAY_TRACKER_HPP

#include "include/device_manager.hpp"
#include "include/parray.hpp"
#include "include/resources.hpp"
#include "parray.hpp"

using namespace parray;

class PArrayTracker {
public:
  PArrayTracker(DeviceManager *device_manager);

  /**
   * @brief Log the state of the PArray to the tracking table.
   * @param dev_id ID for target device
   * @param parray_access PArray & access information pair
   * Returns the size (in bytes) to be allocated in the target device.
   */
  size_t log(const DevID_t dev_id,
             std::pair<parray::InnerPArray *, AccessMode> parray_access);

  /**
   * @brief It the passed PArray instance, either as a slice or
   * a complete array, is not being tracked but is instantiated or moved to
   * a specific device, register the instance to the PArray tracking table and
   * track its states.
   */
  void track_parray(const InnerPArray &parray, DevID_t dev_id);

  /**
   * @brief Remove a PArray from the PArray tracking table and does not track
   * that until a task attempts to use that.
   * This can improve look-up operation performance.
   */
  void untrack_parray(const InnerPArray &parray, DevID_t dev_id);

  /**
   * @brief Reserve PArray usage in a specified device.
   * If a PArray is reserved to a specific device, it implies that
   * the corresponding PArray instance is planned to be instantiated or is
   * already instantiated in the device.
   */
  void reserve_parray(const InnerPArray &parray, Device *device);

  /**
   * @brief Release a PArray from a specified device.
   * A PArray is released from a device when its instance does not exist
   * and also there is no plan (none of tasks that use the PArray is mapped
   * to the device) to be referenced in the device.
   */
  void release_parray(const InnerPArray &parray, Device *device);

  /**
   * @brief Check if a PArray instance is located in a device (without locking
   * the table).
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
    return (this->managed_parrays_[global_dev_idx].find(parray_parent_id) !=
            this->managed_parrays_[global_dev_idx].end());
  }

  /**
   * @brief Check if a PArray instance is located in a device (with lock).
   */
  bool contains(DevID_t global_dev_idx, uint64_t parray_parent_id) {
    mtx.lock();
    bool state = this->contains_unsafe(global_dev_idx, parray_parent_id);
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

  /**
   * @brief Record a parray in the tracking table.
   */
  void set_parray_unsafe(DevID_t global_dev_idx, uint64_t parray_id,
                         bool exists) {
    auto &active_parrays = this->managed_parrays_[global_dev_idx];
    active_parrays[parray_id] = exists;
  }

  /**
   * @brief Remove a PArray from the tracking table.
   */
  void remove_parray_unsafe(DevID_t global_dev_idx, uint64_t parray_id) {
    this->managed_parrays_[global_dev_idx].erase(parray_id);
  }

protected:
  DeviceManager *device_manager_;

  /// Vector index: a device global ID
  /// Map's key: a PArray ID
  ///            (NOTE that this is the ID of the complete array)
  /// Map's value: bool (existance of the PArray in the device)
  ///              size_t (size of the active subarrays in the device)
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
