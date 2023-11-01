#ifndef PARLA_DEVICE_CONTEXTS_HPP
#define PARLA_DEVICE_CONTEXTS_HPP

#include "gpu_utility.hpp"
#include <unistd.h>
#include <unordered_map>
#include <vector>

class InnerStreamPool {
public:
  InnerStreamPool() {}
  InnerStreamPool(const StreamPool &) = delete;

  void register_stream(int device_id, uintptr_t stream) {
    stream_map_[device_id].emplace_back(stream);
  }

  uintptr_t get_stream(int device_id, int stream_id) {
    return stream_map_[device_id][stream_id];
  }

private:
  std::unordered_map<int, std::vector<uintptr_t>> stream_map_;
};

#endif // PARLA_DEVICE_CONTEXTS_HPP
