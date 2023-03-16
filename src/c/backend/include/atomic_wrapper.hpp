#ifndef PARLA_ATOMIC_WRAPPER_HPP
#define PARLA_ATOMIC_WRAPPER_HPP

#include <atomic>

/// A copyable atomic class inherited from std::atomic
template <typename T> class CopyableAtomic : public std::atomic<T> {

public:
  CopyableAtomic() : std::atomic<T>(T{}) {}
  constexpr CopyableAtomic(T base) : std::atomic<T>(base) {}

  /// Copy constructor
  constexpr CopyableAtomic(const CopyableAtomic<T> &other)
      : CopyableAtomic(other.load(std::memory_order_relaxed)) {}

  /// Copy assignment operator
  CopyableAtomic &operator=(const CopyableAtomic<T> &other) {
    this->store(other.load(std::memory_order_relaxed),
                std::memory_order_relaxed);
    return *this;
  }
};

#endif
