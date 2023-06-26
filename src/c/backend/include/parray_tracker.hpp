#ifndef PARLA_PARRAY_TRACKER_HPP
#define PARLA_PARRAY_TRACKER_HPP

#include "include/device_manager.hpp"
#include "include/parray.hpp"

using namespace parray;

/**
 * @brief PArray tracker that tracks PArray mapping state.
 * Note that this does not track slice PArrays, but a complete PArray.
 */
class PArrayTracker {
public:
  PArrayTracker(DeviceManager *deivce_manage);

  /**
   * @brief Start PArray state tracking.
   * @detail If a passed PArray instance, either as a slice or
   * a complete array, is not being tracked but is instantiated or moved to
   * a specific device, register the instance to the PArray tracking table and
   * track its states.
   *
   * @param parray PArray reference to be tracked
   * @param dev_id device id having the parray object
   */
  void track_parray(const InnerPArray &parray, DevID_t dev_id);

  /**
   * @brief Untrack PArray state.
   * @detail Remove a PArray from the PArray tracking table and does not track
   * that until other tasks attempt to use that.
   * As it removes unnecessary information from the table, it could improve
   * look-up operation.
   *
   * @param parray PArray reference to be untracked
   * @param dev_id device id having the parray object
   */
  void untrack_parray(const InnerPArray &parray, DevID_t dev_id);

  /**
   * @brief Start or update PArray state tracking
   * @detail If a PArray is mapped to a specific device, it implies that
   * the corresponding PArray instance will be created or already
   * exists in the device. If a PArray is planned to be created,
   * start its state tracking. If a PArray instance already exists, 
   * update its state.
   *
   * @param parray PArray reference to be tracked or whose state will be updated 
   * @param dev_id device id having the parray object
   */
  void reserve_parray_to_tracker(const InnerPArray &parray, Device *device);

  /**
   * @brief Release a PArray from a specified device.
   * @detail A PArray is released from a device when its instance does not exist
   * or there is no plan to be referenced; none of tasks that will use the
   * PArray is mapped to the device. If it is, decrease memory counter for
   * mapped memory of the device.
   *
   * @param parray PArray reference to be released from the deivce
   * @param dev_id device id having the parray object
   */
  void release_parray_from_tracker(const InnerPArray &parray, Device *device);

  /**
   * @brief Get a state of a PArray instance in a parray tracker.
   *
   * @param global_dev_idx global device index of a device to be accessed
   * @param parent_parray_id PArray parent id to be accessed; note that we
   * do not track slice state.
   */
  bool get_parray_state(DevID_t global_dev_idx, uint64_t parray_parent_id) {
    mtx.lock();
    bool state = this->managed_parrays_[global_dev_idx][parray_parent_id];
    mtx.unlock();
    return state;
  }

private:
  DeviceManager *device_manager_;
  /// Vetor index: a device global id
  /// Map's key: a PArray's parent id
  ///            (NOTE that this is the ID of the complete array)
  /// Map's value: true if the PArray instance exists
  ///              on/will be mapped to the device
  /// Using a device ID as the first key is time efficient since
  /// we don't need to create device keys whenever we add a new PArray
  /// to the table.
  std::vector<std::unordered_map<uint64_t, bool>> managed_parrays_;
  /// Any worker thread can update states of PArrays.
  /// Guard operations by this lock.
  /// TODO(hc): This will be replaced with parallel hash map (phmap).
  std::mutex mtx;
};

#endif
