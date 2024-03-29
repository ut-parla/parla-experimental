#include "parray.hpp"
#include <cstdint>
#include <unordered_map>

namespace parray {

InnerPArray::InnerPArray(void *py_parray, uint64_t id, uint64_t parent_id,
                         InnerPArray *parent_parray, PArrayState *state,
                         DevID_t num_devices)
    : _py_parray(py_parray), id(id), parent_id(parent_id),
      _parent_parray(parent_parray), _state(state), _num_devices(num_devices) {
  num_active_tasks.resize(num_devices);
}

const uint64_t InnerPArray::get_size() const { return this->_size; }

void InnerPArray::set_size(uint64_t new_size) { this->_size = new_size; }

bool InnerPArray::exists_on_device(uint64_t device_id) {
  return this->_state->exists_on_device(device_id);
}

bool InnerPArray::valid_on_device(uint64_t device_id) {
  return this->_state->valid_on_device(device_id);
}

void InnerPArray::add_task(InnerTask *task) {
  // This pushing is thread-safe.
  this->_task_lists.push_back(task);
}

void InnerPArray::incr_num_active_tasks(DevID_t global_dev_id) {
  if (this->_parent_parray != nullptr) {
    this->_parent_parray->num_active_tasks[global_dev_id].fetch_add(
        1, std::memory_order_relaxed);
  } else {
    this->num_active_tasks[global_dev_id].fetch_add(1,
                                                    std::memory_order_relaxed);
  }
}

void InnerPArray::decr_num_active_tasks(DevID_t global_dev_id) {
  if (this->_parent_parray != nullptr) {
    this->_parent_parray->num_active_tasks[global_dev_id].fetch_sub(
        1, std::memory_order_relaxed);
  } else {
    this->num_active_tasks[global_dev_id].fetch_sub(1,
                                                    std::memory_order_relaxed);
  }
}

size_t InnerPArray::get_num_active_tasks(DevID_t global_dev_id) {
  if (this->_parent_parray != nullptr) {
    return this->_parent_parray->num_active_tasks[global_dev_id].load(
        std::memory_order_relaxed);
  } else {
    return this->num_active_tasks[global_dev_id].load(
        std::memory_order_relaxed);
  }
}

TaskList &InnerPArray::get_task_list_ref() { return this->_task_lists; }

void *InnerPArray::get_py_parray() { return this->_py_parray; }

uint64_t InnerPArray::get_parray_parentid() { return this->parent_id; }

InnerPArray *InnerPArray::get_parent_parray() {
  if (this->_parent_parray == nullptr) {
    return this;
  }
  return this->_parent_parray;
}

} // namespace parray
