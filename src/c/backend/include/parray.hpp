#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>
#include "containers.hpp"
#include "parray_state.hpp"

class InnerTask;

using TaskList = ProtectedVector<InnerTask *>;

namespace parray {
    // PArray C++ interface which provides some information that will be used for scheduling task
    class InnerPArray{
        public:
            uint64_t id;  // unique ID of the PArray

            InnerPArray();
            InnerPArray(void *, uint64_t, PArrayState *);

            // Get current size (in bytes) of each copy of the PArray
            // if it is a subarray, return the subarray's size
            uint64_t get_size();

            // Set the size of the PArray
            void set_size(uint64_t new_size);

            // Return True if there is an PArray copy (possibly invalid) on this device
            bool exists_on_device(uint64_t device_id);

            // Return True if there is an PArray copy and its coherence state is valid on this device
            bool valid_on_device(uint64_t device_id);

            // Add a pointer of the task that will use this PArray to the task list
            void add_task(InnerTask *task);

            // Get a list of tasks who are using this PArray
            TaskList& get_task_list_ref();

            // Return the instance of Python PArray
            void *get_py_parray();

        private:
            uint64_t _size;  // number of bytes consumed by each copy of the array/subarray
            PArrayState* _state;  // state of a PArray (subarray share this object with its parent)
            // TODO(hc): this should be a concurrent map.
            //           this requires freuqent addition/removal.
            //           I will use this map: https://github.com/greg7mdp/parallel-hashmap
            //           I have used this for a while and it is good.
            TaskList _task_lists;
            void *_py_parray;
    };
}
