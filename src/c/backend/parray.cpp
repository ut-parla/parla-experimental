#include <unordered_map>
#include <cstdint>
#include "parray.hpp"

namespace parray {
    PArray::PArray() : id(-1), _state(nullptr) {}

    PArray::PArray(void *py_parray, uint64_t id, PArrayState* state) :
        _py_parray(py_parray), id(id), _state(state) {}

    uint64_t PArray::get_size() {
        return this->_size;
    }

    void PArray::set_size(uint64_t new_size) {
        this->_size = new_size;
    }

    bool PArray::exists_on_device(uint64_t device_id) {
        return this->_state->exists_on_device(device_id);
    }

    bool PArray::valid_on_device(uint64_t device_id) {
        return this->_state->valid_on_device(device_id);
    }

    void PArray::add_task(InnerTask *task) {
      // This pushing is thread-safe.
      this->_task_lists.push_back(task);
    }

    TaskList& PArray::get_task_list_ref() {
      return this->_task_lists;
    }
}
