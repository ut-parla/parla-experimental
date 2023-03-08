#ifndef RESOURCES_HPP
#define RESOURCES_HPP

#include <array>
#include <atomic>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <string_view>
using namespace std::literals::string_view_literals; // Enables sv suffix only

using Resource_t = int64_t;

enum Resource { MEMORY = 0, VCU = 1 };
enum ResourceCategory { ALL = 0, PERSISTENT = 1, NON_PERSISTENT = 2 };

// TODO(wlr): ResourcePool should have template specializations on the device
// type.
//       E.g. each has a constexpr array of active_resources on the device.
//       This will allow us to use different types of resources for different
//       devices and compare them. For now assume only memory and vcu exist and
//       are used for all devices.

inline constexpr std::array resource_names = {"memory"sv, "vcu"sv};
// inline std::unordered_map<std::string, Resource> resource_map = {
//     {resource_names[Resource::MEMORY], Resource::MEMORY},
//     {resource_names[Resource::VCU], Resource::VCU}};

inline constexpr std::array<Resource, 1> persistent_resources = {
    Resource::MEMORY};
inline constexpr std::array<Resource, 1> non_persistent_resources = {
    Resource::VCU};

template <typename T> class ResourcePool {

public:
  using V = typename T::value_type;

  ResourcePool() = default;

  ResourcePool(std::vector<Resource> &resource_list, std::vector<V> &values) {
    for (auto i = 0; i < resource_list.size(); i++) {
      this->resources[resource_list[i]].exchange(values[i]);
    }
  }

  ResourcePool(std::vector<std::pair<Resource, V>> &resource_list) {
    for (auto i = 0; i < resource_list.size(); i++) {
      this->resources[resource_list[i].first].exchange(resource_list[i].second);
    }
  }

  ResourcePool(const ResourcePool &other) {
    for (auto i = 0; i < resource_names.size(); i++) {
      this->resources[i].exchange(other.resources[i].load());
    }
  }

  inline const V set(Resource resource, auto value) {
    return this->resources[resource].exchange(static_cast<T>(value));
  };

  inline const V get(Resource resource) const {
    return this->resources[resource].load();
  };

  template <ResourceCategory category>
  inline const bool check_greater(const ResourcePool &other) const {
    if constexpr (category == ResourceCategory::ALL) {
      for (auto i = 0; i < resource_names.size(); i++) {
        if (this->resources[i].load() >= other.resources[i].load()) {
          return false;
        }
      }
      return true;
    } else if constexpr (category == ResourceCategory::PERSISTENT) {
      for (auto i = 0; i < persistent_resources.size(); i++) {
        if (this->resources[persistent_resources[i]].load() >=
            other.resources[persistent_resources[i]].load()) {
          return false;
        }
      }
      return true;
    } else if constexpr (category == ResourceCategory::NON_PERSISTENT) {
      for (auto i = 0; i < non_persistent_resources.size(); i++) {
        if (this->resources[non_persistent_resources[i]].load() >=
            other.resources[non_persistent_resources[i]].load()) {
          return false;
        }
      }
      return true;
    }
  };

  template <ResourceCategory category>
  inline const bool check_lesser(const ResourcePool &other) const {
    if constexpr (category == ResourceCategory::ALL) {
      for (auto i = 0; i < resource_names.size(); i++) {
        if (this->resources[i].load() <= other.resources[i].load()) {
          return false;
        }
      }
      return true;
    } else if constexpr (category == ResourceCategory::PERSISTENT) {
      for (auto i = 0; i < persistent_resources.size(); i++) {
        if (this->resources[persistent_resources[i]].load() <=
            other.resources[persistent_resources[i]].load()) {
          return false;
        }
      }
      return true;
    } else if constexpr (category == ResourceCategory::NON_PERSISTENT) {
      for (auto i = 0; i < non_persistent_resources.size(); i++) {
        if (this->resources[non_persistent_resources[i]].load() <=
            other.resources[non_persistent_resources[i]].load()) {
          return false;
        }
      }
      return true;
    }
  };

  template <ResourceCategory category>
  inline void increase(const ResourcePool &other) {
    if constexpr (category == ResourceCategory::ALL) {
      for (auto i = 0; i < resource_names.size(); i++) {
        this->resources[i].fetch_add(other.resources[i].load());
      }
    } else if constexpr (category == ResourceCategory::PERSISTENT) {
      for (auto i = 0; i < persistent_resources.size(); i++) {
        this->resources[persistent_resources[i]].fetch_add(
            other.resources[persistent_resources[i]].load());
      }
    } else if constexpr (category == ResourceCategory::NON_PERSISTENT) {
      for (auto i = 0; i < non_persistent_resources.size(); i++) {
        this->resources[non_persistent_resources[i]].fetch_add(
            other.resources[non_persistent_resources[i]].load());
      }
    }
  };

  template <ResourceCategory category>
  inline void decrease(const ResourcePool &other) {
    if constexpr (category == ResourceCategory::ALL) {
      for (auto i = 0; i < resource_names.size(); i++) {
        this->resources[i].fetch_sub(other.resources[i].load());
      }
    } else if constexpr (category == ResourceCategory::PERSISTENT) {
      for (auto i = 0; i < persistent_resources.size(); i++) {
        this->resources[persistent_resources[i]].fetch_sub(
            other.resources[persistent_resources[i]].load());
      }
    } else if constexpr (category == ResourceCategory::NON_PERSISTENT) {
      for (auto i = 0; i < non_persistent_resources.size(); i++) {
        this->resources[non_persistent_resources[i]].fetch_sub(
            other.resources[non_persistent_resources[i]].load());
      }
    }
  };

protected:
  // TODO(wlr): Is there any way to make this compile time initilaization depend
  // on resouce_names.size()?
  std::array<T, resource_names.size()> resources = {0, 0};
};

#endif // RESOURCES_HPP