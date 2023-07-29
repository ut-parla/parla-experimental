#pragma once
#ifndef PARLA_DEVICE_HPP
#define PARLA_DEVICE_HPP

#include "resources.hpp"
#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

using DevID_t = uint32_t;
using MemorySz_t = Resource_t;
using VCU_t = Resource_t;
// using ResourcePool_t = ResourcePool<std::atomic<Resource_t>>;
using ResourcePool_t = ResourcePool;

class DeviceRequirement;

/**
 * @brief Architecture types for devices.
 */
enum class ParlaDeviceType { All = -1, CPU = 0, CUDA = 1 };

inline const constexpr std::array architecture_types{ParlaDeviceType::CPU,
                                                     ParlaDeviceType::CUDA};
inline const constexpr int NUM_DEVICE_TYPES = architecture_types.size();
inline const std::array<std::string, NUM_DEVICE_TYPES> architecture_names{
    "CPU", "CUDA"};

/// Devices can be distinguished from other devices
/// by a class type and its index.
class ParlaDevice {
  using TimePoint = std::chrono::time_point<std::chrono::system_clock>;

public:
  ParlaDevice() = delete;

  ParlaDevice(ParlaDeviceType arch, DevID_t dev_id, MemorySz_t mem_sz,
              VCU_t num_vcus, void *py_dev, int copy_engines = 2)
      : py_dev_(py_dev), dev_id_(dev_id), dev_type_(arch) {

    res_.set(Resource::VCU, num_vcus);
    res_.set(Resource::Memory, mem_sz);
    res_.set(Resource::Copy, copy_engines);

    reserved_res_.set(Resource::VCU, num_vcus);
    reserved_res_.set(Resource::Memory, mem_sz);
    reserved_res_.set(Resource::Copy, copy_engines);

    mapped_res_.set(Resource::VCU, 0);
    mapped_res_.set(Resource::Memory, 0);
    mapped_res_.set(Resource::Copy, 0);

    this->idle_begin_time_ = {};
    this->idle_end_time_ = {};
    this->accumulated_idle_time_ = 0;
  }

  /// Return a device id.
  const DevID_t get_id() const { return dev_id_; }

  const std::string get_name() const {
    return architecture_names[static_cast<int>(this->dev_type_)] + ":" +
           std::to_string(dev_id_);
  }

  const Resource_t query_resource(Resource type) const {
    return this->res_.get(type);
  }

  const Resource_t query_reserved_resource(Resource type) const {
    return this->reserved_res_.get(type);
  }

  const Resource_t query_mapped_resource(Resource type) const {
    return this->mapped_res_.get(type);
  }

  const ParlaDeviceType get_type() const { return dev_type_; }

  // Comment(wlr): Maybe max resource pool should be const?

  /**
   * @brief Returns the device details (maximum resources available)
   * This is assumed to be constant after device creation.
   */
  const ResourcePool_t &get_resource_pool() const { return res_; }

  /**
   * @brief Returns the currently mapped resources on the device.
   * This starts at 0 and increases as resources are mapped.
   * Decreased when resources are released at the end of a task.
   * This is not runtime necessary, but useful to mapping policy.
   */
  ResourcePool_t &get_mapped_pool() { return mapped_res_; }

  /**
   * @brief Returns the currently reserved resources on the device.
   * This starts at max and decreases as resources are reserved.
   * This represents the resources currently in use by the tasks.
   * This is necessary to determine if tasks can be scheduled without
   * oversubscription or OOM errors.
   */
  ResourcePool_t &get_reserved_pool() { return reserved_res_; }

  /**
   * @brief Returns the pointer to the python device object.
   */
  void *get_py_device() { return py_dev_; }
  void set_global_id(DevID_t global_id) { dev_global_id_ = global_id; }
  const DevID_t get_global_id() const { return dev_global_id_; }

  const MemorySz_t get_memory_size() const {
    return res_.get(Resource::Memory);
  }

  const VCU_t get_num_vcus() const { return res_.get(Resource::VCU); }

  const Resource_t get_max_resource(Resource type) const {
    return this->res_.get(type);
  }

  const Resource_t get_reserved_resource(Resource type) const {
    return this->reserved_res_.get(type);
  }

  const Resource_t get_mapped_resource(Resource type) const {
    return this->mapped_res_.get(type);
  }

  const bool check_resource_availability(DeviceRequirement *dev_req) const;

  /**
   * @brief Get the current system clock as the beginning of a
   * device idle state.
   */
  void begin_device_idle() {
    this->idle_timer_mtx_.lock();
    this->is_idle = true;
    // Get the current time point as an idle time begin.
    this->idle_begin_time_ = std::chrono::system_clock::now();
    this->idle_timer_mtx_.unlock();
  }

  /**
   * @brief Get the current system clock as the end of a device
   * idle state, and accumulate the duration.
   */
  void end_device_idle() {
    this->idle_timer_mtx_.lock();
    this->is_idle = false;
    this->end_device_idle_unsafe();
    this->idle_timer_mtx_.unlock();
  }

  void end_device_idle_unsafe() {
    // Get the current time point as an idle time end.
    this->idle_end_time_ = std::chrono::system_clock::now();
    // Calculate and accumulate duration between two points.
    this->accumulated_idle_time_ += std::chrono::duration_cast<
        std::chrono::milliseconds>(
            this->idle_end_time_ - this->idle_begin_time_).count();
    #if 0
    std::cout << "Idle end time count:" <<
      std::chrono::duration_cast<std::chrono::milliseconds>(
          this->idle_end_time_.time_since_epoch()).count() <<
          ", begin time count:" <<
      std::chrono::duration_cast<std::chrono::milliseconds>(
          this->idle_begin_time_.time_since_epoch()).count() <<
          "\n";
    std::cout << "Initial time count:" <<
      std::chrono::duration_cast<std::chrono::milliseconds>(
          this->initial_epoch_.time_since_epoch()).count() <<
          "\n";
    #endif
    // Reset time points; New duration will be accumulated.
    this->idle_begin_time_ = this->idle_end_time_;
    this->idle_end_time_ = {};
  }

  /**
   * @brief Get the duration of the idle time of a device.
   *
   * @return Pair of idle and non-idle times.
   */
  std::pair<double, double> get_total_idle_time() {
    double old_total_idle_time{0}, total_time{0};
    this->idle_timer_mtx_.lock();
    if (this->is_idle) {
      this->end_device_idle_unsafe();
    }
    old_total_idle_time = this->accumulated_idle_time_;
    TimePoint current_time_point = std::chrono::system_clock::now();
    total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        current_time_point - this->initial_epoch_).count();
    this->num_get_idle_time_++;
    #if 0
    std::cout << "Total time:" << total_time << "\n";
    std::cout << "accumulated idle time:" << this->accumulated_idle_time_ << ", " <<
      this->num_get_idle_time_ << "\n";
    #endif
    this->idle_timer_mtx_.unlock();
    return std::make_pair(old_total_idle_time, total_time - old_total_idle_time);
  }

  double current_timepoint_count_from_beginning() {
    TimePoint current_time_point = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        current_time_point - this->initial_epoch_).count();
  }

  void set_initial_epoch(TimePoint initial_epoch) {
    this->idle_timer_mtx_.lock();
    this->initial_epoch_ = initial_epoch;
    this->accumulated_idle_time_ = 0;
    this->idle_begin_time_ = initial_epoch;
    this->idle_end_time_ = {};
    this->num_get_idle_time_ = 0;
    this->idle_timer_mtx_.unlock();
  }

protected:
  ParlaDeviceType dev_type_;
  DevID_t dev_id_;
  DevID_t dev_global_id_;
  ResourcePool_t res_;
  ResourcePool_t mapped_res_;
  ResourcePool_t reserved_res_;
  void *py_dev_;
  std::unordered_map<std::string, size_t> resource_map_;
  /// System clock time point when the device manager created
  TimePoint initial_epoch_;
  /// System clock time point of the beginning of the device idle
  TimePoint idle_begin_time_;
  /// System clock time point of the end of the device idle
  TimePoint idle_end_time_;
  /// Accumulated system clock counts during the device idle
  double accumulated_idle_time_;
  std::mutex idle_timer_mtx_;
  size_t num_get_idle_time_{0};
  bool is_idle{true};
};

///
class CUDADevice : public ParlaDevice {
public:
  CUDADevice(DevID_t dev_id, size_t mem_sz, size_t num_vcus, void *py_dev)
      : ParlaDevice(ParlaDeviceType::CUDA, dev_id, mem_sz, num_vcus, py_dev,
                    3) {}

private:
};

///
class CPUDevice : public ParlaDevice {
public:
  CPUDevice(DevID_t dev_id, size_t mem_sz, size_t num_vcus, void *py_dev)
      : ParlaDevice(ParlaDeviceType::CPU, dev_id, mem_sz, num_vcus, py_dev, 4) {
  }

private:
};

#endif
