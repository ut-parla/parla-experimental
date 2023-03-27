from parla.common.parray import core
from parla.cython.device import PyDevice
from parla.cython.device_manager import PyDeviceManager

from typing import Dict

PArray = core.PArray
Device = PyDevice

class PArrayTracker:
  _managed_parray_tbl: Dict[int, Dict[int, bool]]

  def __init__(self, device_manager: PyDeviceManager):
      print("PArray tracker is created.")
      # First key: a device global ID
      # Second key (a value of the first key): a PArray ID
      #            (NOTE that this is the ID of the complete array)
      # Second value: true if the PArray instance exists on the device
      # Using a device ID as the first key is time efficient since
      # we don't need to create device keys whenever we add a new PArray
      # to the table.
      self._managed_parray_tbl = {}
      self._device_manager = device_manager
      self._devices = device_manager.get_all_devices()
      # TODO(hc): add a resource pool or a device manager to update

  def initialize_table(self):
      for d in self._devices:
          self._managed_parray_tbl[d.get_global_id()] = {}

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

  def reserve_parray(self, parray: PArray, device: Device):
      """
      Reserve PArray usage in a specified device.
      If a PArray is reserved to a specific device, it implies that
      the corresponding PArray instance is planned to be instantiated or is
      already instantiated in the device.
      """
      dev_global_id = device.get_global_id()
      if parray.parent_ID not in self._managed_parray_tbl[dev_global_id]:
          self.track_parray(parray, dev_global_id)
      if self._managed_parray_tbl[dev_global_id][dev_global_id] == False:
          self._managed_parray_tbl[dev_global_id][dev_global_id] = True
          # TODO(hc): allocate resources by decreasing mapped resouce metrics
          #           on a device.

  def release_parray(self, parray: PArray, device: Device):
      """
      Release a PArray from a specified device.
      A PArray is released from a device when its instance does not exist
      and also there is no plan (none of tasks that use the PArray is mapped
      to the device) to be referenced in the device.
      """
      pass

