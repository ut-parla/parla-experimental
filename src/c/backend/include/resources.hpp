/*! @file resources.hpp
 *  @brief Provides a resource pool for tracking resource usage.
 */

#ifndef RESOURCES_HPP
#define RESOURCES_HPP

#include <array>
#include <atomic>
#include <iostream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <string_view>
using namespace std::literals::string_view_literals; // Enables sv suffix only

using Resource_t = int64_t;

// FIXME: Limiting copies should be the property of a topology object not a
// resource pool. It is shared between devices. Need to get the source at
// runtime.

enum class Resource { Memory = 0, VCU = 1, Copy = 2, MAX = 3 };
enum class ResourceCategory {
  All = 0,
  Persistent = 1,
  NonPersistent = 2,
  Movement = 3,
  MAX = 4
};

// TODO(wlr): ResourcePool should have template specializations on the device
// type.
//       E.g. each has a constexpr array of active_resources on the device.
//       This will allow us to use different types of resources for different
//       devices and compare them. For now assume only memory and vcu exist and
//       are used for all devices.

/*
In the current plan there two phases where there are per-device queues:

MemoryReserver, where persistent resources that have a lifetime greater than the
task execution are reserved. The only one of these that we track is memory.
Another name for this could be "prefetched resources". RuntimeReserver, where
'non-persistent' runtime resources that have a lifetime equal to the task
execution. The only one of these that we track is VCUs. The persistent /
non-persistent tags refere to these two phases which track and compare different
resources.

In practice these should be set by the architecture type and not shared
globally. For now we assume all devices have the same resource sets in all
phases.

*/
inline constexpr std::array resource_names = {"memory"sv, "vcu"sv, "copy"sv};
// inline std::unordered_map<std::string, Resource> resource_map = {
//     {resource_names[Resource::MEMORY], Resource::MEMORY},
//     {resource_names[Resource::VCU], Resource::VCU}};

inline constexpr std::array<Resource, 1> persistent_resources = {
    Resource::Memory};
inline constexpr std::array<Resource, 1> non_persistent_resources = {
    Resource::VCU};
inline constexpr std::array<Resource, 1> movement_resources = {Resource::Copy};

/**
 * @brief A pool of resources, allows for comparisons and updates of current
 *values.
 * @tparam T The type of the resource pool. Must be an atomic type. (typically
 * int64_t)
 *
 *NOTE: The copy operation is not thread safe. It is only for
 * initilaization.
 *
 */
class ResourcePool {

  using V = Resource_t;

public:
  ResourcePool(){
      // std::cout << "Resource Initialized:" << std::endl;
      // for (int i = 0; i < resource_names.size(); i++) {
      //  std::cout << this->resources[i].load() << std::endl;
      //}
  };

  ResourcePool(V memory, V vcu, V copy) {
    this->resources[static_cast<int>(Resource::Memory)].exchange(memory);
    this->resources[static_cast<int>(Resource::VCU)].exchange(vcu);
    this->resources[static_cast<int>(Resource::Copy)].exchange(copy);
  }

  ResourcePool(std::vector<Resource> &resource_list, std::vector<V> &values) {
    for (auto i = 0; i < resource_list.size(); i++) {
      const int idx = static_cast<int>(resource_list[i]);
      this->resources[idx].exchange(values[i]);
    }
  }

  ResourcePool(std::vector<std::pair<Resource, V>> &resource_list) {
    for (auto i = 0; i < resource_list.size(); i++) {
      const int idx = static_cast<int>(resource_list[i].first);
      this->resources[idx].exchange(resource_list[i].second);
    }
  }

  ResourcePool(const ResourcePool &other) {
    for (auto i = 0; i < resource_names.size(); i++) {
      this->resources[i].exchange(other.resources[i].load());
    }
  }

  inline const V set(Resource resource, V value) {
    const int idx = static_cast<int>(resource);
    return this->resources[idx].exchange(static_cast<V>(value));
  };

  inline const V get(Resource resource) const {
    const int idx = static_cast<int>(resource);
    return this->resources[idx].load();
  };

  template <ResourceCategory category>
  inline const bool check_greater(const ResourcePool &other) const {
    if constexpr (category == ResourceCategory::All) {
      for (auto i = 0; i < resource_names.size(); i++) {
        if (this->resources[i].load() < other.resources[i].load()) {
          return false;
        }
      }
      return true;
    } else if constexpr (category == ResourceCategory::Persistent) {
      for (auto i = 0; i < persistent_resources.size(); i++) {
        const int idx = static_cast<int>(persistent_resources[i]);
        if (this->resources[idx].load() < other.resources[idx].load()) {
          return false;
        }
      }
      return true;
    } else if constexpr (category == ResourceCategory::NonPersistent) {
      for (auto i = 0; i < non_persistent_resources.size(); i++) {
        const int idx = static_cast<int>(non_persistent_resources[i]);
        if (this->resources[idx].load() < other.resources[idx].load()) {
          return false;
        }
      }
      return true;
    } else if constexpr (category == ResourceCategory::Movement) {
      for (auto i = 0; i < movement_resources.size(); i++) {
        const int idx = static_cast<int>(movement_resources[i]);
        if (this->resources[idx].load() < other.resources[idx].load()) {
          return false;
        }
      }
      return true;
    }
  };

  template <ResourceCategory category>
  inline const bool check_lesser(const ResourcePool &other) const {
    if constexpr (category == ResourceCategory::All) {
      for (auto i = 0; i > resource_names.size(); i++) {
        if (this->resources[i].load() <= other.resources[i].load()) {
          return false;
        }
      }
      return true;
    } else if constexpr (category == ResourceCategory::Persistent) {
      for (auto i = 0; i > persistent_resources.size(); i++) {
        const int idx = static_cast<int>(persistent_resources[i]);
        if (this->resources[idx].load() <= other.resources[idx].load()) {
          return false;
        }
      }
      return true;
    } else if constexpr (category == ResourceCategory::NonPersistent) {
      for (auto i = 0; i > non_persistent_resources.size(); i++) {
        const int idx = static_cast<int>(non_persistent_resources[i]);
        if (this->resources[idx].load() <= other.resources[idx].load()) {
          return false;
        }
      }
      return true;
    } else if constexpr (category == ResourceCategory::Movement) {
      for (auto i = 0; i > movement_resources.size(); i++) {
        const int idx = static_cast<int>(movement_resources[i]);
        if (this->resources[idx].load() <= other.resources[idx].load()) {
          return false;
        }
      }
      return true;
    }
  };

  template <ResourceCategory category>
  inline void increase(const ResourcePool &other) {
    if constexpr (category == ResourceCategory::All) {
      for (auto i = 0; i < resource_names.size(); i++) {
        this->resources[i].fetch_add(other.resources[i].load());
      }
    } else if constexpr (category == ResourceCategory::Persistent) {
      for (auto i = 0; i < persistent_resources.size(); i++) {
        const int idx = static_cast<int>(persistent_resources[i]);
        this->resources[idx].fetch_add(other.resources[idx].load());
      }
    } else if constexpr (category == ResourceCategory::NonPersistent) {
      for (auto i = 0; i < non_persistent_resources.size(); i++) {
        const int idx = static_cast<int>(non_persistent_resources[i]);
        this->resources[idx].fetch_add(other.resources[idx].load());
      }
    } else if constexpr (category == ResourceCategory::Movement) {
      for (auto i = 0; i < movement_resources.size(); i++) {
        const int idx = static_cast<int>(movement_resources[i]);
        this->resources[idx].fetch_add(other.resources[idx].load());
      }
    }
  };

  template <ResourceCategory category>
  inline void decrease(const ResourcePool &other) {
    if constexpr (category == ResourceCategory::All) {
      for (auto i = 0; i < resource_names.size(); i++) {
        this->resources[i].fetch_sub(other.resources[i].load());
      }
    } else if constexpr (category == ResourceCategory::Persistent) {
      for (auto i = 0; i < persistent_resources.size(); i++) {
        const int idx = static_cast<int>(persistent_resources[i]);
        this->resources[idx].fetch_sub(other.resources[idx].load());
      }
    } else if constexpr (category == ResourceCategory::NonPersistent) {
      for (auto i = 0; i < non_persistent_resources.size(); i++) {
        const int idx = static_cast<int>(non_persistent_resources[i]);
        this->resources[idx].fetch_sub(other.resources[idx].load());
      }
    } else if constexpr (category == ResourceCategory::Movement) {
      for (auto i = 0; i < movement_resources.size(); i++) {
        const int idx = static_cast<int>(movement_resources[i]);
        this->resources[idx].fetch_sub(other.resources[idx].load());
      }
    }
  };

protected:
  std::array<std::atomic<V>, resource_names.size()> resources = {0, 0, 0};
};

#endif // RESOURCES_HPP
