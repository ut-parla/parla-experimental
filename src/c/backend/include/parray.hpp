/*! @file parray.hpp
 *  @brief Provides C++ interface to PArray State and ID.
 *
 * This file contains the inner C++ parray interface. Allows access to coherency
 * and parent/child relations without accessing the Python interpreter.
 */
#ifndef PARLA_PARRAY_HPP
#define PARLA_PARRAY_HPP

#include "atomic_wrapper.hpp"
#include "containers.hpp"
#include "device.hpp"
#include "parray_state.hpp"
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

class InnerTask;

using TaskList = ProtectedVector<InnerTask *>;

/// Access mode to a PArray.
enum AccessMode {
  /// Newly allocated PArray.
  NEW = 0,
  /// Input of a task.
  IN = 1,
  /// Input/Output of a task.
  INOUT = 2,
  /// Output of a task.
  OUT = 3,
  /// Freed PArray (false everywhere).
  FREED = 4,
  /// Deleted PArray (removed from table).
  DELETED = 5,
};

namespace parray {
// PArray C++ interface which provides some information that will be used for
// scheduling task

class InnerPArray;

using PArrayAccess_t = std::pair<InnerPArray *, AccessMode>;
using PArrayLocation_t = std::pair<InnerPArray *, DevID_t>;
using PArrayLocationSize_t = std::tuple<InnerPArray *, DevID_t, size_t>;

class InnerPArray {
public:
  uint64_t id;        // unique ID of the PArray
  uint64_t parent_id; // unique ID of the parent PArray

  /// Track the number of tasks that are using or are planning to use this
  /// PArray.
  /// NOTE that this counter is not necessarily matching to the size of
  /// the `_task_lists`. This is because `_task_lists` does not remove a
  /// task after it is completed (since it is not worth to remove that
  /// compared to restructuring overheads), but this counter is decreased.
  /// This is used to provide more accurate PArray placement information
  /// to the task mapping step.
  std::vector<CopyableAtomic<size_t>> num_referring_tasks;

  InnerPArray() = delete;
  InnerPArray(void *, uint64_t, uint64_t, InnerPArray *, PArrayState *,
              DevID_t);

  /// Get current size (in bytes) of each copy of the PArray
  /// if it is a subarray, return the subarray's size
  const uint64_t get_size() const;

  /// Equality operator
  bool operator==(const InnerPArray &other) const {
    return this->id == other.id;
  };

  /// Comparison operator
  bool operator<(const InnerPArray &other) const {
    return this->id < other.id;
  };

  /// Set the size of the PArray
  void set_size(uint64_t new_size);

  /// Return True if there is an PArray copy (possibly invalid) on this device
  bool exists_on_device(uint64_t device_id);

  /// Return True if there is an PArray copy and its coherence state is valid on
  /// this device
  bool valid_on_device(uint64_t device_id);

  /// Add a pointer of the task that will use this PArray to the task list
  void add_task(InnerTask *task);

  /// Increase the number of the tasks referring to this PArray.
  void incr_num_referring_tasks(DevID_t global_dev_id);

  /// Decrease the number of the tasks referring to this PArray.
  void decr_num_referring_tasks(DevID_t global_dev_id);

  /// Get the number of the tasks referring to this PArray.
  size_t get_num_referring_tasks(DevID_t global_dev_id);

  // TODO(hc): I will replace this list with a concurrent map.
  /// Get a reference to a list of tasks who are using this PArray
  TaskList &get_task_list_ref();

  /// Return the instance of Python PArray.
  void *get_py_parray();

  /// Return the id of the current instance.
  const uint64_t get_id() const;

  /// Return the parent id of the current instance.
  const uint64_t get_parent_id() const;

  /// Return if this PArray is a subarray.
  const bool is_subarray() const;

  /// Return a pointer of the parent PArray.
  InnerPArray *get_parent_parray();

  /// Return a vector of devices that the parray is valid on
  /// These are parray local ids (not global parla ids)!
  std::vector<int> get_valid_devices() { return _state->get_valid_devices(); }

private:
  uint64_t _size; // number of bytes consumed by each copy of the array/subarray
  InnerPArray *_parent_parray;
  PArrayState
      *_state; // state of a PArray (subarray share this object with its parent)
  DevID_t _num_devices;

  // TODO(hc): this should be a concurrent map.
  //           this requires freuqent addition/removal.
  //           I will use this map: https://github.com/greg7mdp/parallel-hashmap
  //           I have used this for a while and it is good.
  TaskList _task_lists;
  /// Python PArray address
  void *_py_parray;
};

} // namespace parray

#endif
