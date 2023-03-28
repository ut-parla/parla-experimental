#ifndef PARLA_PARRAY_TRACKER_HPP
#define PARLA_PARRAY_TRACKER_HPP

#include "include/device_manager.hpp"
#include "include/parray.hpp"

using namespace parray;

class PArrayTracker {
public:
  PArrayTracker(DeviceManager* deivce_manage);

  /**
   * @brief It the passed PArray instance, either as a slice or
   * a complete array, is not being tracked but is instantiated or moved to
   * a specific device, register the instance to the PArray tracking table and
   * track its states.
   */
  void track_parray(const InnerPArray& parray, DevID_t dev_id);

  /**
   * @brief Remove a PArray from the PArray tracking table and does not track
   * that until a task attempts to use that.
   * This can improve look-up operation performance.
   */
  void untrack_parray(const InnerPArray& parray, DevID_t dev_id);

  /**
   * @brief Reserve PArray usage in a specified device.
   * If a PArray is reserved to a specific device, it implies that
   * the corresponding PArray instance is planned to be instantiated or is
   * already instantiated in the device.
   */
  void reserve_parray(const InnerPArray& parray, Device* device);

  /**
   * @brief Release a PArray from a specified device.
   * A PArray is released from a device when its instance does not exist
   * and also there is no plan (none of tasks that use the PArray is mapped
   * to the device) to be referenced in the device.
   */
  void release_parray(const InnerPArray& parray, Device* device);
private:

  DeviceManager* device_manager_;
  /// Vetor index: a device global ID
  /// Map's key: a PArray ID
  ///            (NOTE that this is the ID of the complete array)
  /// Map's value: true if the PArray instance exists
  ///              on/will be mapped to the device
  /// Using a device ID as the first key is time efficient since
  /// we don't need to create device keys whenever we add a new PArray
  /// to the table.
  std::vector<std::unordered_map<uint64_t, bool>> managed_parrays_;
  /// Any worker thread can update states of PArrays.
  /// Guard operations by this lock.
  /// TODO(hc): This will be replaced with parallel hash map.
  std::mutex mtx;
};

#endif
