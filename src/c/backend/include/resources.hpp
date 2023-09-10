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

// Metaprograming Magic (requires C++17)

/// @brief A Container for a list of Resource Types (Memory, VCU, Copy)
template <typename... Types> struct Resources {};

template <typename> class ResourcePool;
template <typename T, typename I, typename R> class ResourcePoolImpl {};

template <typename T, typename... Ts> constexpr bool contains() {
  return std::disjunction_v<std::is_same<T, Ts>...>;
}

template <std::size_t I> using size = std::integral_constant<std::size_t, I>;
template <class...> struct types {
  using type = types;
};

template <class T, class Types> struct index_of {};

/// @brief Partial specialization for the base case of the recursion. (T is the
/// first type in the list)
template <class T, class... Ts>
struct index_of<T, types<T, Ts...>> : size<0> {};

/// @brief Partial specialization for the recursive case. (While T is not the
/// first type in the list we recurse)
template <class T, class T0, class... Ts>
struct index_of<T, types<T0, Ts...>>
    : size<1 + index_of<T, types<Ts...>>::value> {};

///@brief Wrapper for the index_of metafunction in an integral constant
template <class T, class... Ts>
using index_of_t = size<index_of<T, types<Ts...>>::value>;

// FIXME: Limiting memory copies should be the property of a topology object
// not a resource pool. It is shared between devices. Need to get the source
// at runtime.

using namespace std::literals::string_view_literals; // Enables sv suffix only

using Resource_t = int64_t;

namespace Resource {

namespace Category {
inline constexpr std::array names = {"persistent"sv, "non-persistent"sv,
                                     "movement"sv};

/// @brief All the resource types that are managed by the MemoryReserver phase
/// (having interactions greater than the lifetime of a task)
struct Persistent {
  static const int value = 0;
  constexpr static const std::string_view name = names[0];
  explicit operator int() const { return value; }
};

/// @brief All the resource types that are managed by the RuntimeReserver phase
/// (having interactions equal to the lifetime of a task)
struct NonPersistent {
  static const int value = 1;
  constexpr static const std::string_view name = names[1];
  explicit operator int() const { return value; }
};

/// @brief temporary catch all for restricting data copies
struct Movement {
  static const int value = 2;
  constexpr static const std::string_view name = names[2];
  explicit operator int() const { return value; }
};

} // namespace Category

inline constexpr std::array names = {"memory"sv, "vcu"sv, "copy"sv};

/// @brief A resource type specifying the memory (in bytes) used
struct Memory {
  static const int value = 0;
  constexpr static const std::string_view name = names[0];
  explicit operator int() const { return value; }
  using Category = Category::Persistent;
};

/// @brief A resource type specifying the VCU (virtual compute units) aka the
/// fraction of the device used
struct VCU {
  static const int value = 1;
  constexpr static const std::string_view name = names[1];
  explicit operator int() const { return value; }
  using Category = Category::NonPersistent;
};

/// @brief A resource type specifying the number of copy engines used
struct Copy {
  static const int value = 2;
  constexpr static const std::string_view name = names[2];
  explicit operator int() const { return value; }
  using Category = Category::Movement;
};

/// @brief All the resource types that are managed by the MemoryReserver phase
/// (having interactions greater than the lifetime of a task)
using PersistentResources = Resources<Memory>;

/// @brief All the resource types that are managed by the RuntimeReserver phase
/// (having interactions equal to the lifetime of a task)
using NonPersistentResources = Resources<VCU>;

/// @brief temporary catch all for restricting data copies
using MovementResources = Resources<Copy>;

} // namespace Resource

/// @brief The internal implementation of ResourcePool's methods. This enables
/// partial specialization on the different Resources.
template <template <typename...> class InputList, typename... InputTypes,
          template <typename...> class IntermediateList,
          typename... IntermediateTypes,
          template <typename...> class ReferenceList,
          typename... ReferenceTypes>
class ResourcePoolImpl<InputList<InputTypes...>,
                       IntermediateList<IntermediateTypes...>,
                       ReferenceList<ReferenceTypes...>> {

public:
  template <typename T> static constexpr inline bool contained_in_reference() {
    return std::disjunction_v<std::is_same<T, ReferenceTypes>...>;
  };

  template <typename T> static constexpr inline bool contained_in_input() {
    return std::disjunction_v<std::is_same<T, InputTypes>...>;
  };

  template <typename T>
  static constexpr inline bool contained_in_intermediate() {
    return std::disjunction_v<std::is_same<T, IntermediateTypes>...>;
  };

  template <typename T> static constexpr inline int index_in_target() {
    if constexpr (contained_in_reference<T>()) {
      return index_of_t<T, ReferenceTypes...>::value;
    } else {
      // The Resource type is not in the reference list
      return -1;
    }
  }

  template <typename T> static constexpr inline int index_in_source() {
    if constexpr (contained_in_input<T>()) {
      return index_of_t<T, InputTypes...>::value;
    } else {
      // The Resource type is not in the reference list
      return -1;
    }
  }

  static constexpr void raise_error() {
    // static_assert(false, "Resource not found");
    std::cout << "Resource not found" << std::endl;
  }

  template <typename T> static constexpr void print_resource() {
    std::cout << index_in_target<T>() << std::endl;
  }

  static constexpr void print() { (print_resource<InputTypes>(), ...); }

  template <typename T, typename R>
  static constexpr inline void set_resource(R &r, Resource_t value) {
    const int index = index_in_target<T>();
    if constexpr (index != -1) {
      r.resources[index].store(value, std::memory_order_relaxed);
    } else {
      raise_error();
    }
  }

  template <typename T, typename R>
  static constexpr inline Resource_t get_resource(R &r) {
    const int index = index_in_target<T>();
    if constexpr (index != -1) {
      return r.resources[index].load(std::memory_order_relaxed);
    } else {
      raise_error();
      return 0;
    }
  }

  template <typename T, typename R>
  static constexpr inline void increase_resource(R &r, Resource_t value) {
    const int index = index_in_target<T>();
    if constexpr (index != -1) {
      r.resources[index].fetch_add(value, std::memory_order_relaxed);
    } else {
      raise_error();
    }
  }

  template <typename T, typename R>
  static constexpr inline void decrease_resource(R &r, Resource_t value) {
    const int index = index_in_target<T>();
    if constexpr (index != -1) {
      r.resources[index].fetch_sub(value, std::memory_order_relaxed);
    } else {
      raise_error();
    }
  }

  template <typename T, typename R>
  static constexpr inline bool check_lesser_resource(R &r, Resource_t value) {
    const int index = index_in_target<T>();
    if constexpr (index != -1) {
      return r.resources[index].load() <= value;
    } else {
      raise_error();
      return false;
    }
  }

  template <typename T, typename R>
  static constexpr inline bool check_greater_resource(R &r, Resource_t value) {
    const int index = index_in_target<T>();
    if constexpr (index != -1) {
      return r.resources[index].load() >= value;
    } else {
      std::cout << "index is always -1\n" << std::flush;
      raise_error();
      return false;
    }
  }

  template <typename T, typename R1, typename R2>
  static constexpr inline void set_resource(R1 &r1, R2 &r2) {
    const int target_index = index_in_target<T>();
    const int source_index = index_in_source<T>();

    if constexpr (target_index != -1 && source_index != -1) {

      r1.resources[target_index].store(r2.resources[source_index].load(),
                                       std::memory_order_relaxed);
    } else {
      raise_error();
    }
  }

  template <typename T, typename R1, typename R2>
  static constexpr inline void increase_resource(R1 &r1, R2 &r2) {
    const int target_index = index_in_target<T>();
    const int source_index = index_in_source<T>();

    if constexpr (target_index != -1 && source_index != -1) {

      r1.resources[target_index].fetch_add(r2.resources[source_index].load(),
                                           std::memory_order_relaxed);
    } else {
      raise_error();
    }
  }

  template <typename T, typename R1, typename R2>
  static constexpr inline void decrease_resource(R1 &r1, R2 &r2) {
    const int target_index = index_in_target<T>();
    const int source_index = index_in_source<T>();

    if constexpr (target_index != -1 && source_index != -1) {

      r1.resources[target_index].fetch_sub(r2.resources[source_index].load(),
                                           std::memory_order_relaxed);
    } else {
      raise_error();
    }
  }

  template <typename T, typename R1, typename R2>
  static constexpr inline bool check_lesser_resource(R1 &r1, R2 &r2) {
    const int target_index = index_in_target<T>();
    const int source_index = index_in_source<T>();

    if constexpr (target_index != -1 && source_index != -1) {
      return r1.resources[target_index].load() <=
             r2.resources[source_index].load();
    } else {
      raise_error();
      return false;
    }
  }

  template <typename T, typename R1, typename R2>
  static constexpr inline bool check_greater_resource(R1 &r1, R2 &r2) {
    const int target_index = index_in_target<T>();
    const int source_index = index_in_source<T>();

    if constexpr (target_index != -1 && source_index != -1) {
      std::cout << "r1:" << r1.resources[target_index].load() << " vs" <<
        " r2:" << r2.resources[source_index].load() << "\n" << std::flush;
      return r1.resources[target_index].load() >=
             r2.resources[source_index].load();
    } else {
      raise_error();
      return false;
    }
  }

  template <typename R> static constexpr inline void reset(R &r) {
    (set_resource<InputTypes>(r, 0), ...);
  }

  template <typename R>
  static constexpr inline void set(R &r, std::initializer_list<Resource_t> il) {
    size_t i = 0;
    (set_resource<InputTypes>(r, il.begin()[i++]), ...);
  }

  template <typename R>
  static constexpr inline std::array<Resource_t, sizeof...(InputTypes)>
  get(R &r) {
    return {r.resources[index_in_target<InputTypes>()].load()...};
  }

  template <typename R>
  static constexpr inline void increase(R &r,
                                        std::initializer_list<Resource_t> il) {
    size_t i = 0;
    (increase_resource<InputTypes>(r, il.begin()[i++]), ...);
  }

  template <typename R>
  static constexpr inline void decrease(R &r,
                                        std::initializer_list<Resource_t> il) {
    size_t i = 0;
    (decrease_resource<InputTypes>(r, il.begin()[i++]), ...);
  }

  template <typename R>
  static constexpr inline bool
  check_lesser(R &r, std::initializer_list<Resource_t> il) {
    size_t i = 0;
    return (check_lesser_resource<InputTypes>(r, il.begin()[i++]) && ...);
  }

  template <typename R>
  static constexpr inline bool
  check_greater(R &r, std::initializer_list<Resource_t> il) {
    size_t i = 0;
    return (check_greater_resource<InputTypes>(r, il.begin()[i++]) && ...);
  }

  template <typename R1, typename R2>
  static constexpr inline void set(R1 &r1, R2 &r2) {
    (set_resource<InputTypes>(r1, r2), ...);
  }

  template <typename R1, typename R2>
  static constexpr inline void increase(R1 &r1, R2 &r2) {
    (increase_resource<InputTypes>(r1, r2), ...);
  }

  template <typename R1, typename R2>
  static constexpr inline void decrease(R1 &r1, R2 &r2) {
    (decrease_resource<InputTypes>(r1, r2), ...);
  }

  template <typename R1, typename R2>
  static constexpr inline bool check_lesser(R1 &r1, R2 &r2) {
    return (check_lesser_resource<InputTypes>(r1, r2) && ...);
  }

  template <typename R1, typename R2>
  static constexpr inline bool check_greater(R1 &r1, R2 &r2) {
    return (check_greater_resource<InputTypes>(r1, r2) && ...);
  }
};

/***
 *  @brief ResourcePool is a class that holds a set of resources
 *  @tparam TypeList is a list of types that are the resources
 */
template <template <typename... Types> class TypeList, typename... Types>
class ResourcePool<TypeList<Types...>> {

public:
  using Resources = TypeList<Types...>;
  template <typename T> using ordinal = index_of_t<T, Types...>;

  /// @brief Construct a ResourcePool with all resources set to 0
  ResourcePool(){};

  /// @brief Construct a ResourcePool with the given values
  ResourcePool(std::initializer_list<Resource_t> il) {
    set<TypeList<Types...>>(il);
  }

  /// @brief Construct a ResourcePool with another pool of the same type
  ResourcePool(const ResourcePool<TypeList<Types...>> &other) {
    set<TypeList<Types...>>(other);
  }

  /// @brief Construct a ResourcePool with the given values from another pool
  template <typename InputList>
  ResourcePool(const ResourcePool<InputList> &other) {
    set<InputList>(other);
  }

  /// @brief print the current values of all resources in the pool to stdout
  void print() const {
    for (int i = 0; i < sizeof...(Types); ++i) {
      std::cout << this->resources[i].load() << std::endl;
    }
  }

  /// @brief set the resources to the values in the initializer list
  template <typename InputList> void set(std::initializer_list<Resource_t> il) {
    ResourcePoolImpl<InputList, InputList, TypeList<Types...>>::set(*this, il);
  }

  /// @brief Retrieve the current values of the resources in the pool (not
  /// performant)
  template <typename InputList>
  std::array<Resource_t, sizeof...(Types)> get_many() const {
    return ResourcePoolImpl<InputList, InputList, TypeList<Types...>>::get(
        *this);
  }

  /// @brief Increase the resources by the values in the initializer list
  /// @paramt InputList is the list of types of the resources to increase
  /// @param il is the list of values to increase the resources by
  template <typename InputList>
  void increase(std::initializer_list<Resource_t> il) {
    ResourcePoolImpl<InputList, InputList, TypeList<Types...>>::increase(*this,
                                                                         il);
  }

  /// @brief Decrease the resources by the values in the initializer list
  /// @paramt InputList is the list of types of the resources to decrease
  /// @param il is the list of values to decrease the resources by
  template <typename InputList>
  void decrease(std::initializer_list<Resource_t> il) {
    ResourcePoolImpl<InputList, InputList, TypeList<Types...>>::decrease(*this,
                                                                         il);
  }

  /// @brief Check if the resources are lesser or equal than the values in the
  /// initializer
  /// @paramt InputList is the list of types of the resources to check
  /// @param il is the list of values to check the resources against
  template <typename InputList>
  bool check_lesser(std::initializer_list<Resource_t> il) const {
    return ResourcePoolImpl<InputList, InputList,
                            TypeList<Types...>>::check_lesser(*this, il);
  }

  /// @brief Check if the resources are strictly greater than the values in the
  /// initializer
  /// @paramt InputList is the list of types of the resources to check
  /// @param il is the list of values to check the resources against
  template <typename InputList>
  bool check_greater(std::initializer_list<Resource_t> il) const {
    return ResourcePoolImpl<InputList, InputList,
                            TypeList<Types...>>::check_greater(*this, il);
  }

  /// @brief Set the resources to the values in the other resource pool
  /// @paramt InputList is the list of types of the resources to set
  /// @param r is the other resource pool
  template <typename InputList, typename IntermediateList,
            std::enable_if_t<!std::is_same<InputList, IntermediateList>::value,
                             int> = 0>
  void set(const ResourcePool<IntermediateList> &r) {
    ResourcePoolImpl<InputList, IntermediateList, TypeList<Types...>>::set(
        *this, r);
  }

  /// @brief Increase the resources by the values in the other resource pool
  /// @paramt InputList is the list of types of the resources to increase
  /// @param r is the other resource pool
  template <typename InputList, typename IntermediateList,
            std::enable_if_t<!std::is_same<InputList, IntermediateList>::value,
                             int> = 0>
  void increase(ResourcePool<IntermediateList> &r) {
    ResourcePoolImpl<InputList, IntermediateList, TypeList<Types...>>::increase(
        *this, r);
  }

  /// @brief Decrease the resources by the values in the other resource pool
  /// @paramt InputList is the list of types of the resources to decrease
  /// @param r is the other resource pool
  template <typename InputList, typename IntermediateList,
            std::enable_if_t<!std::is_same<InputList, IntermediateList>::value,
                             int> = 0>
  void decrease(ResourcePool<IntermediateList> &r) {
    ResourcePoolImpl<InputList, IntermediateList, TypeList<Types...>>::decrease(
        *this, r);
  }

  /// @brief Check if the resources are lesser or equal than the values in the
  /// other resource pool
  /// @paramt InputList is the list of types of the resources to check
  /// @param r is the other resource pool
  template <typename InputList, typename IntermediateList,
            std::enable_if_t<!std::is_same<InputList, IntermediateList>::value,
                             int> = 0>
  bool check_lesser(ResourcePool<IntermediateList> &r) const {
    return ResourcePoolImpl<InputList, IntermediateList,
                            TypeList<Types...>>::check_lesser(*this, r);
  }

  /// @brief Check if the resources are strictly greater than the values in the
  /// other resource pool
  /// @paramt InputList is the list of types of the resources to check
  /// @param r is the other resource pool
  template <typename InputList, typename IntermediateList,
            std::enable_if_t<!std::is_same<InputList, IntermediateList>::value,
                             int> = 0>
  bool check_greater(ResourcePool<IntermediateList> &r) const {
    return ResourcePoolImpl<InputList, IntermediateList,
                            TypeList<Types...>>::check_greater(*this, r);
  }

  /// @brief Set all resources to values in the other resource pool
  /// @param r is the other resource pool
  template <typename IntermediateList>
  void set(const ResourcePool<IntermediateList> &r) {
    ResourcePoolImpl<TypeList<Types...>, IntermediateList,
                     TypeList<Types...>>::set(*this, r);
  }

  /// @brief Increase all resources by values in the other resource pool
  /// @param r is the other resource pool
  template <typename IntermediateList>
  void increase(ResourcePool<IntermediateList> &r) {
    ResourcePoolImpl<TypeList<Types...>, IntermediateList,
                     TypeList<Types...>>::increase(*this, r);
  }

  /// @brief Increase the all resources by values in the other resource pool
  /// @param r is the other resource pool
  template <typename IntermediateList>
  void decrease(ResourcePool<IntermediateList> &r) {
    ResourcePoolImpl<TypeList<Types...>, IntermediateList,
                     TypeList<Types...>>::decrease(*this, r);
  }

  /// @brief Check if all resources are lesser or equal than the values in the
  /// other resource pool
  /// @param r is the other resource pool
  template <typename IntermediateList>
  bool check_lesser(ResourcePool<IntermediateList> &r) const {
    return ResourcePoolImpl<TypeList<Types...>, IntermediateList,
                            TypeList<Types...>>::check_lesser(*this, r);
  }

  /// @brief Check if all resources are strictly greater than the values in the
  /// other resource pool
  /// @param r is the other resource pool
  template <typename IntermediateList>
  bool check_greater(ResourcePool<IntermediateList> &r) const {
    return ResourcePoolImpl<TypeList<Types...>, IntermediateList,
                            TypeList<Types...>>::check_greater(*this, r);
  }

  /// @brief Get the value of a single resource
  template <typename Resource> Resource_t get() const {
    return this->resources[ordinal<Resource>()].load();
  }

  /// @brief Set the value of a single resource
  template <typename Resource> void set(Resource_t v) {
    this->resources[ordinal<Resource>()].store(v, std::memory_order_relaxed);
  }

  /// @brief Increase the value of a single resource
  template <typename Resource> void increase(Resource_t v) {
    this->resources[ordinal<Resource>()].fetch_add(v,
                                                   std::memory_order_relaxed);
  }

  /// @brief Decrease the value of a single resource
  template <typename Resource> void decrease(Resource_t v) {
    this->resources[ordinal<Resource>()].fetch_sub(v,
                                                   std::memory_order_relaxed);
  }

  /// @brief Check if a single resource is lesser or equal than a value
  template <typename Resource> bool check_lesser(Resource_t v) const {
    return this->resources[ordinal<Resource>()].load() <= v;
  }

  /// @brief Check if a single resource is greater or equal to the value
  template <typename Resource> bool check_greater(Resource_t v) const {
    return this->resources[ordinal<Resource>()].load() >= v;
  }

  /// @brief Get the values of multiple resources by index
  void set(std::initializer_list<size_t> target,
           std::initializer_list<Resource_t> il) {
    size_t i = 0;
    for (auto t : target) {
      this->resources[t].store(il.begin()[i++], std::memory_order_relaxed);
    }
  }

  /// @brief Get the values of multiple resources by index
  std::array<Resource_t, sizeof...(Types)>
  get(std::initializer_list<size_t> target) const {
    std::array<Resource_t, sizeof...(Types)> arr;
    size_t i = 0;
    for (auto t : target) {
      arr[i++] = this->resources[t].load();
    }
    return arr;
  }

  /// @brief Increase the values of multiple resources by index
  void increase(const std::vector<size_t> &target,
                const std::vector<Resource_t> &il) {
    for (size_t i = 0; i < target.size(); i++) {
      this->resources[target[i]].fetch_add(il[i], std::memory_order_relaxed);
    }
  }

  /// @brief Decrease the values of multiple resources by index
  void decrease(const std::vector<size_t> &target,
                const std::vector<Resource_t> &il) {
    for (size_t i = 0; i < target.size(); i++) {
      this->resources[target[i]].fetch_sub(il[i], std::memory_order_relaxed);
    }
  }

  /// @brief Check if the values of multiple resources by index are lesser or
  /// equal than the values in the other resource pool
  bool check_lesser(const std::vector<size_t> target,
                    const std::vector<Resource_t> &il) const {
    for (size_t i = 0; i < target.size(); i++) {
      if (this->resources[i].load() <= il[i]) {
        return false;
      }
    }
    return true;
  }

  /// @brief Check if the values of multiple resources by index are
  /// greater or equal than the values in the other resource pool
  bool check_greater(const std::vector<size_t> target,
                     const std::vector<Resource_t> &il) const {
    for (size_t i = 0; i < target.size(); i++) {
      if (this->resources[i].load() >= il[i]) {
        std::cout << this->resources[i].load() << " vs " << il[i] << "\n";
        return false;
      }
    }
    return true;
  }

  /// @brief Get the index of a single resource by type (compile-time) in the
  /// pool
  template <typename Resource> size_t get_index() const {
    return ordinal<Resource>();
  }

private:
  template <typename T, typename I, typename R> friend class ResourcePoolImpl;

  /// @brief the atomic resource array
  std::array<std::atomic<Resource_t>, sizeof...(Types)> resources = {};
};

#endif // RESOURCES_HPP
