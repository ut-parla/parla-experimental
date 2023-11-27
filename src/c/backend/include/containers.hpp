/*! @file containers.hpp
 *  @brief Provides interface for thread-safe containers.
 *
 *
 */
#ifndef PARLA_CONTAINERS_HPP
#define PARLA_CONTAINERS_HPP

#include <iostream>

#include <deque>
#include <list>
#include <map>
#include <vector>

#include <queue>
#include <stack>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>

#include <assert.h>
#include <chrono>

/* Header only implementations of thread-safe data structures and helpers */
/* This will probably grow and change signifigantly as development continues and
 * needs and bottlenecks change.*/

// By default, types are not atomic,
// template <typename T> auto constexpr is_atomic = false;
// but std::atomic<T> types are,
// template <typename T> auto constexpr is_atomic<std::atomic<T>> = true;
// as well as std::atomic_flag.
// template <> auto constexpr is_atomic<std::atomic_flag> = true;

template <typename T> class ProtectedVector {

private:
  std::vector<T> vec = std::vector<T>();
  std::atomic<int> length{0};
  std::mutex mtx;
  std::string name;

public:
  ProtectedVector() { this->name = "default"; };

  ProtectedVector(const ProtectedVector<T> &other) {
    this->name = other.name;
    this->vec = other.vec;
    this->length.exchange(other.length);
  }

  ProtectedVector(std::string name) { this->name = name; }

  ProtectedVector(std::string name, std::vector<T> vec) {
    this->name = name;
    this->vec = vec;
  }

  ProtectedVector(std::string name, size_t size) {
    this->name = name;
    this->vec.reserve(size);
  }

  // ProtectedVector &operator=(const ProtectedVector<T> &&other) {
  //   this->length.exchange(other.length);
  //   this->vec = std::move(other.vec);
  //   // The string should be small
  //   this->name = std::move(other.name);
  //   return *this;
  // }

  void lock() { this->mtx.lock(); }

  void unlock() { this->mtx.unlock(); }

  void push_back(T a) {
    this->mtx.lock();
    this->vec.push_back(a);
    this->mtx.unlock();
    this->length++;
  }

  void push_back(std::vector<T> &a) {
    this->mtx.lock();
    this->vec.insert(this->vec.end(), a.begin(), a.end());
    this->mtx.unlock();
    this->length += a.size();
  }

  void push_back_unsafe(T a) {
    this->vec.push_back(a);
    this->length++;
  }

  void push_back_unsafe(std::vector<T> &a) {
    this->vec.insert(this->vec.end(), a.begin(), a.end());
    this->length += a.size();
  }

  void pop_back() {
    this->mtx.lock();
    this->vec.pop_back();
    this->mtx.unlock();
    this->length--;
  }

  void pop_back_unsafe() {
    this->vec.pop_back();
    this->length--;
  }

  int atomic_size() { return this->length.load(); }

  size_t size() {
    this->mtx.lock();
    int size = this->vec.size();
    this->mtx.unlock();
    return size;
  }

  size_t size_unsafe() { return this->vec.size(); }

  T operator[](size_t i) {
    this->mtx.lock();
    auto val = this->vec[i];
    this->mtx.unlock();
    return val;
  }

  T at(size_t i) {
    this->mtx.lock();
    T val = this->vec.at(i);
    this->mtx.unlock();
    return val;
  }

  T at_unsafe(size_t i) { return this->vec.at(i); }

  T back() {
    this->mtx.lock();
    T val = this->vec.back();
    this->mtx.unlock();
    return val;
  }

  inline T back_unsafe() { return this->vec.back(); }

  T back_and_pop() {
    this->mtx.lock();
    T val = this->back_unsafe();
    this->pop_back_unsafe();
    this->mtx.unlock();
    return val;
  }

  inline T back_and_pop_unsafe() {
    T val = this->back_unsafe();
    this->pop_back_unsafe();
    return val;
  }

  T front() {
    this->mtx.lock();
    T val = this->vec.front();
    this->mtx.unlock();
    return val;
  }

  inline T front_unsafe() { return this->vec.front(); }

  // TODO(hc): I think this can be just called "pop"
  // TODO(wlr): I wasn't sure since STL container `pop` doesn't return the old
  // head.
  T front_and_pop() {
    this->mtx.lock();
    T val = this->front_unsafe();
    this->pop_front_unsafe();
    this->mtx.unlock();
    return val;
  }

  T front_and_pop_unsafe() {
    T val = this->front_unsafe();
    this->pop_front_unsafe();
    return val;
  }

  void clear() {
    this->mtx.lock();
    this->vec.clear();
    this->mtx.unlock();
    this->length = 0;
  }

  inline void clear_unsafe() {
    this->vec.clear();
    this->length = 0;
  }

  void reserve(size_t size) {
    this->mtx.lock();
    this->vec.reserve(size);
    this->mtx.unlock();
  }

  inline void reserve_unsafe(size_t size) { this->vec.reserve(size); }

  void resize(size_t size) {
    this->mtx.lock();
    this->vec.resize(size);
    this->mtx.unlock();
  }

  void resize_unsafe(size_t size) { this->vec.resize(size); }

  T get(size_t i) {
    this->mtx.lock();
    T val = this->vec[i];
    this->mtx.unlock();
    return val;
  }

  inline T get_unsafe(size_t i) { return this->vec[i]; }

  void set(size_t i, T val) {
    this->mtx.lock();
    this->vec[i] = val;
    this->mtx.unlock();
  }

  inline void set_unsafe(size_t i, T val) { this->vec[i] = val; }

  std::vector<T> get_vector_copy() {
    this->mtx.lock();
    std::vector<T> vec = this->vec;
    this->mtx.unlock();
    return vec;
  }

  inline std::vector<T> get_vector_copy_unsafe() {
    std::vector<T> vec = this->vec;
    return vec;
  }

  std::vector<T> &get_vector() {
    this->mtx.lock();
    std::vector<T> &vec = this->vec;
    this->mtx.unlock();
    return vec;
  }

  std::vector<T> &get_vector_unsafe() {
    this->mtx.lock();
    std::vector<T> &vec = this->vec;
    this->mtx.unlock();
    return vec;
  }

  bool empty() {
    this->mtx.lock();
    bool empty = this->vec.empty();
    this->mtx.unlock();
    return empty;
  }

  inline bool empty_unsafe() { return this->vec.empty(); }
};

template <typename T> class ProtectedQueue {

private:
  std::deque<T> q = std::deque<T>();
  std::atomic<int> length{0};
  std::mutex mtx;
  std::string name;

public:
  ProtectedQueue() = default;

  ProtectedQueue(std::string name) {
    this->mtx.lock();
    this->name = name;
    this->mtx.unlock();
  }

  ProtectedQueue(std::string name, std::deque<T> q) {
    this->mtx.lock();
    this->name = name;
    this->q = q;
    this->mtx.unlock();
  }

  ProtectedQueue(std::string name, size_t size) {
    this->mtx.lock();
    this->name = name;
    this->q.reserve(size);
    this->mtx.unlock();
  }

  void lock() { this->mtx.lock(); }

  void unlock() { this->mtx.unlock(); }

  void push_back(T a) {
    this->mtx.lock();
    this->q.push_back(a);
    this->mtx.unlock();
    this->length++;
  }

  inline void push_back_unsafe(T a) {
    this->q.push_back(a);
    this->length++;
  }

  void push_back(std::vector<T> &a) {
    this->mtx.lock();
    for (auto val : a) {
      this->push_back_unsafe(val);
    }
    this->mtx.unlock();
  }

  inline void push_back_unsafe(std::vector<T> &a) {
    for (auto val : a) {
      this->push_back_unsafe(val);
    }
  }

  void push_front(T a) {
    this->mtx.lock();
    this->q.push_back(a);
    this->mtx.unlock();
    this->length++;
  }

  inline void push_front_unsafe(T a) {
    this->q.push_back(a);
    this->length++;
  }

  void push_front(std::vector<T> &a) {
    this->mtx.lock();
    for (auto val : a) {
      this->push_back_unsafe(val);
    }
    this->mtx.unlock();
  }

  inline void push_front_unsafe(std::vector<T> &a) {
    for (auto val : a) {
      this->push_back_unsafe(val);
    }
  }

  void pop_back() {
    this->mtx.lock();
    this->q.pop_back();
    this->mtx.unlock();
    this->length--;
  }

  inline void pop_back_unsafe() {
    this->q.pop_back();
    this->length--;
  }

  void pop_front() {
    this->mtx.lock();
    this->q.pop_front();
    this->mtx.unlock();
    this->length--;
  }

  inline void pop_front_unsafe() {
    this->q.pop_front();
    this->length--;
  }

  size_t atomic_size() { return this->length.load(); }

  size_t size() {
    this->mtx.lock();
    int size = this->q.size();
    this->mtx.unlock();
    return size;
  }

  size_t size_unsafe() { return this->q.size(); }

  T operator[](size_t i) {
    this->mtx.lock();
    auto val = this->q[i];
    this->mtx.unlock();
    return val;
  }

  T at(size_t i) {
    this->mtx.lock();
    T val = this->q.at(i);
    this->mtx.unlock();
    return val;
  }

  inline T at_unsafe(size_t i) { return this->q.at(i); }

  T back() {
    this->mtx.lock();
    T val = this->q.back();
    this->mtx.unlock();
    return val;
  }

  inline T back_unsafe() { return this->q.back(); }

  T back_and_pop() {
    this->mtx.lock();
    T val = this->back_unsafe();
    this->pop_back_unsafe();
    this->mtx.unlock();
    return val;
  }

  inline T back_and_pop_unsafe() {
    T val = this->back_unsafe();
    this->pop_back_unsafe();
    return val;
  }

  T front() {
    this->mtx.lock();
    T val = this->q.front();
    this->mtx.unlock();
    return val;
  }

  inline T front_unsafe() { return this->q.front(); }

  T front_and_pop() {
    this->mtx.lock();
    T val = this->front_unsafe();
    this->pop_front_unsafe();
    this->mtx.unlock();
    return val;
  }

  inline T front_and_pop_unsafe() {
    T val = this->front_unsafe();
    this->pop_front_unsafe();
    return val;
  }

  void clear() {
    this->mtx.lock();
    this->q.clear();
    this->mtx.unlock();
    this->length = 0;
  }

  inline void clear_unsafe() {
    this->q.clear();
    this->length = 0;
  }

  bool empty() {
    this->mtx.lock();
    bool empty = this->q.empty();
    this->mtx.unlock();
    return empty;
  }

  inline bool empty_unsafe() { return this->q.empty(); }
};

template <typename T> class ComparePriority {
    public:
       bool operator()(T a, T b){
           if(a->priority < b->priority) { 
               return true; // the order is correct and NO swapping of elements takes place
           }
           return false; // the order is NOT correct and swapping of elements takes place
      }
};

template <typename T> class ProtectedPriorityQueue {

private:
  std::priority_queue<T, std::vector<T>, ComparePriority<T>> pq;
  std::atomic<int> length{0};
  std::mutex mtx;
  std::string name;

public:
  ProtectedPriorityQueue() = default;

  ProtectedPriorityQueue(std::string name) {
    this->mtx.lock();
    this->name = name;
    this->mtx.unlock();
  }

  ProtectedPriorityQueue(std::string name, std::priority_queue<T> pq) {
    this->mtx.lock();
    this->name = name;
    this->pq = pq;
    this->mtx.unlock();
  }

  ProtectedPriorityQueue(std::string name, size_t size) {
    this->mtx.lock();
    this->name = name;
    this->pq.reserve(size);
    this->mtx.unlock();
  }

  void lock() { this->mtx.lock(); }

  void unlock() { this->mtx.unlock(); }

  void push(T a) {
    this->mtx.lock();
    this->pq.push(a);
    this->mtx.unlock();
    this->length++;
  }

  inline void push_unsafe(T a) {
    this->pq.push(a);
    this->length++;
  }

  void push(std::vector<T> &a) {
    this->mtx.lock();
    for (auto val : a) {
      this->push_unsafe(val);
    }
    this->mtx.unlock();
  }

  inline void push_unsafe(std::vector<T> &a) {
    for (auto val : a) {
      this->push_unsafe(val);
    }
  }

  void pop() {
    this->mtx.lock();
    this->pq.pop();
    this->mtx.unlock();
    this->length--;
  }

  inline void pop_unsafe() {
    this->pq.pop();
    this->length--;
  }

  size_t atomic_size() { return this->length.load(); }

  size_t size() {
    this->mtx.lock();
    int size = this->pq.size();
    this->mtx.unlock();
    return size;
  }

  size_t size_unsafe() { return this->q.size(); }

  T operator[](size_t i) {
    this->mtx.lock();
    auto val = this->pq[i];
    this->mtx.unlock();
    return val;
  }

  T at(size_t i) {
    this->mtx.lock();
    T val = this->pq.at(i);
    this->mtx.unlock();
    return val;
  }

  inline T at_unsafe(size_t i) { return this->pq.at(i); }

  T front() {
    this->mtx.lock();
    T val = this->pq.top();
    this->mtx.unlock();
    return val;
  }

  inline T front_unsafe() { return this->pq.top(); }

  T front_and_pop() {
    this->mtx.lock();
    T val = this->front_unsafe();
    this->pop_unsafe();
    this->mtx.unlock();
    return val;
  }

  inline T front_and_pop_unsafe() {
    T val = this->front_unsafe();
    this->pop_unsafe();
    return val;
  }

  // Add implementation of clear()
  // void clear() {
  //   this->mtx.lock();
  //   this->pq.clear();
  //   this->mtx.unlock();
  //   this->length = 0;
  // }

  // inline void clear_unsafe() {
  //   this->pq.clear();
  //   this->length = 0;
  // }

  bool empty() {
    this->mtx.lock();
    bool empty = this->pq.empty();
    this->mtx.unlock();
    return empty;
  }

  inline bool empty_unsafe() { return this->pq.empty(); }
};

#endif // PARLA_CONTAINERS_HPP
