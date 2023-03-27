from parla.common.parray import core

PArray = core.PArray

class PArrayTracker:

  def __init__(self):
      print("PArray tracker is created.")
      # TODO(hc): add a resource pool or a device manager to update
      #           memory usage.

  def track_parray(self, parray: PArray, dev_id: int):
      """
      It the passed PArray instance, either as a slice or a complete array,
      is not being tracked but is instantiated or moved to a specific device,
      register the instance to the PArray tracking table and track its states.
      """
      print("PArray ID: ", parray.ID, " and its parent ID:",
            parray.parent_ID if parray.ID != parray.parent_ID else None,
            " is tracked.")

  def untracak_parray(self, parray: PArray, dev_id: int):
      """
      Remove a PArray from the PArray tracking table and does not track
      that until a task attempts to use that.
      This can improve look-up operation performance.
      """
      # TODO(hc): as we don't have policy regarding this, we are not using
      #           it. we may need this if look-up operation is not cheap
      #           anymore.
      print("PArray ID: ", parray.ID, " and its parent ID:",
           parray.parent_ID if parray.ID != parray.parent_ID else None,
           " is untracked.")

  def reserve_parray(self, parray: PArray, dev_id: int):
      """
      Reserve PArray usage in a specified device.
      If a PArray is reserved to a specific device, it implies that
      the corresponding PArray instance is planned to be instantiated or is
      already instantiated in the device.
      """
      pass

  def release_parray(self, parray: PArray, dev_id: int):
      """
      Release a PArray from a specified device.
      A PArray is released from a device when its instance does not exist
      and also there is no plan (none of tasks that use the PArray is mapped
      to the device) to be referenced in the device.
      """
      pass

