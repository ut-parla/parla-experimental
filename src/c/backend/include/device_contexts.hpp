#ifndef PARLA_DEVICE_CONTEXTS_HPP
#define PARLA_DEVICE_CONTEXTS_HPP

#include "device.hpp"
#include "gpu_utility.hpp"
#include <unistd.h>
#include <unordered_map>
#include <vector>

class InnerStreamPool {
public:
  InnerStreamPool() {}
  InnerStreamPool(const InnerStreamPool &) = delete;

  void push_stream(int device_id, uintptr_t stream) {
    stream_map_[device_id].emplace_back(stream);
  }

  void push_event(int device_id, uintptr_t event) {
    event_map_[device_id].emplace_back(event);
  }

  uintptr_t pop_stream(int device_id) {
    uintptr_t stream = stream_map_[device_id].back();
    stream_map_[device_id].pop_back();
    return stream;
  }

  uintptr_t pop_event(int device_id) {
    uintptr_t event = event_map_[device_id].back();
    event_map_[device_id].pop_back();
    return event;
  }

  int get_stream_count(int device_id) { return stream_map_[device_id].size(); }

  int get_event_count(int device_id) { return event_map_[device_id].size(); }

private:
  std::unordered_map<int, std::vector<uintptr_t>> stream_map_;
  std::unordered_map<int, std::vector<uintptr_t>> event_map_;
};

#endif // PARLA_DEVICE_CONTEXTS_HPP
