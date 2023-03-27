#ifndef PARLA_PARRAY_TRACKER_HPP
#define PARLA_PARRAY_TRACKER_HPP

class PArrayTracker {
public:
  PArrayTracker();

  /**
   * @brief It the passed PArray instance, either as a slice or
   * a complete array, is not being tracked but is instantiated or moved to
   * a specific device, register the instance to the PArray tracking table and
   * track its states.
   */
  void track_parray();

  /**
   * @brief Remove a PArray from the PArray tracking table and does not track
   * that until a task attempts to use that.
   * This can improve look-up operation performance.
   */
  void untrack_parray();

  /**
   * @brief Reserve PArray usage in a specified device.
   * If a PArray is reserved to a specific device, it implies that
   * the corresponding PArray instance is planned to be instantiated or is
   * already instantiated in the device.
   */
  void reserve_parray();

  /**
   * @brief Release a PArray from a specified device.
   * A PArray is released from a device when its instance does not exist
   * and also there is no plan (none of tasks that use the PArray is mapped
   * to the device) to be referenced in the device.
   */
  void release_parray();
protected:

};

#endif
