#include "parray.hpp"
#include <cstdint>
#include <unordered_map>

namespace parray {
InnerPArray::InnerPArray() : id(-1), _state(nullptr) {}

InnerPArray::InnerPArray(void *py_parray, uint64_t id, PArrayState *state)
    : _py_parray(py_parray), id(id), _state(state) {}

uint64_t InnerPArray::get_size() { return this->_size; }

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

TaskList &InnerPArray::get_task_list_ref() { return this->_task_lists; }

void *InnerPArray::get_py_parray() { return this->_py_parray; }
} // namespace parray
