#pragma once
#ifndef PARLA_PHASES_HPP
#define PARLA_PHASES_HPP

#include "containers.hpp"
#include "device.hpp"
#include "device_manager.hpp"
#include "runtime.hpp"

#include <string.h>

class PhaseStatus {
protected:
  const static int size = 3;
  std::string name = "Status";

public:
  int status[size];

  void reset() {
    for (int i = 0; i < size; ++i) {
      this->status[i] = 0;
    }
  }

  void set(int index, int value) { this->status[index] = value; }
  int get(int index) { return this->status[index]; }
  void increase(int state) { this->status[state]++; }

  void print() {
    std::cout << this->name + "(";
    for (int i = 0; i < size; ++i) {
      std::cout << this->status[i];
    }
    std::cout << ")\n";
  }
};

class SchedulerPhase {
public:
  SchedulerPhase() = default;
  SchedulerPhase(InnerScheduler* scheduler, DeviceManager* devices)
      : scheduler(scheduler), device_manager(devices) {}

  virtual void enqueue(InnerTask* task) = 0;
  virtual void enqueue(std::vector<InnerTask*>& tasks) = 0;
  virtual void run(SchedulerPhase* next_phase) = 0;
  virtual size_t get_count() = 0;

  PhaseStatus status;

protected:
  std::string name = "Phase";
  std::mutex mtx;
  InnerScheduler* scheduler;
  DeviceManager* device_manager;
  TaskStateList enqueue_buffer;
};

namespace Map {

enum State { failure, success };
class Status : public PhaseStatus {
protected:
  const static int size = 2;
  std::string name = "Mapper";
};

} // namespace Map

class Mapper : virtual public SchedulerPhase {
public:
  Map::Status status;
  Mapper() : SchedulerPhase(), dummy_dev_idx_{0} {}

  Mapper(InnerScheduler* scheduler, DeviceManager* devices)
      : SchedulerPhase(scheduler, devices), dummy_dev_idx_{0} {}

  void enqueue(InnerTask* task);
  void enqueue(std::vector<InnerTask*>& tasks);
  void run(SchedulerPhase* next_phase);
  size_t get_count();

protected:
  std::string name = "Mapper";
  TaskQueue mappable_tasks;
  std::vector<InnerTask*> mapped_tasks_buffer;
  uint64_t dummy_dev_idx_;
};

namespace Reserved {

enum State { failure, success };

class Status : public PhaseStatus {
protected:
  const static int size = 2;
  std::string name = "MemoryReserver";
};
} // namespace Reserved

class MemoryReserver : virtual public SchedulerPhase {
public:
  Reserved::Status status;

  MemoryReserver(InnerScheduler* scheduler, DeviceManager* devices)
      : SchedulerPhase(scheduler, devices) {}

  void enqueue(InnerTask* task);
  void enqueue(std::vector<InnerTask*>& tasks);
  void run(SchedulerPhase* next_phase);
  size_t get_count();

protected:
  std::string name = "Memory Reserver";
  TaskQueue reservable_tasks;
  std::vector<InnerTask*> reserved_tasks_buffer;
};

namespace Ready {

enum State { entered, task_miss, resource_miss, worker_miss, success };

class Status : virtual public PhaseStatus {
protected:
  const static int size = 5;
  std::string name = "RuntimeReserver";
};
} // namespace Ready

#ifdef PARLA_ENABLE_LOGGING
LOG_ADAPT_STRUCT(Ready::Status, status)
#endif

class RuntimeReserver : virtual public SchedulerPhase {

public:
  Ready::Status status;

  RuntimeReserver(InnerScheduler* scheduler, DeviceManager* devices)
      : SchedulerPhase(scheduler, devices) {}

  void enqueue(InnerTask* task);
  void enqueue(std::vector<InnerTask*>& tasks);
  void run(SchedulerPhase* next_phase);
  size_t get_count();

protected:
  std::string name = "Runtime Reserver";
  TaskQueue runnable_tasks;
  std::vector<InnerTask*> launchable_tasks_buffer;
};

#ifdef PARLA_ENABLE_LOGGING
LOG_ADAPT_STRUCT(RuntimeReserver, status)
#endif

namespace Launch {

enum State { failure, success };

class Status : public PhaseStatus {
protected:
  const static int size = 2;
  std::string name = "Launcher";
};
} // namespace Launch

class Launcher : virtual public SchedulerPhase {
public:
  Launch::Status status;

  /*Number of running tasks. A task is running if it has been assigned to a
   * worker and is not complete*/
  std::atomic<size_t> num_running_tasks;

  Launcher(InnerScheduler* scheduler, DeviceManager* devices)
      : SchedulerPhase(scheduler, devices) {}

  /*Add a task to the launcher. Currently this acquires the GIL and dispatches
   * the work to a Python Worker for each task */
  void enqueue(InnerTask* task){};
  void enqueue(InnerTask* task, InnerWorker* worker);
  void enqueue(std::vector<InnerTask*>& tasks){};

  /* A placeholder function in case work needs to be done at this stage. For
   * example, dispatching a whole buffer of tasks*/
  void run();
  void run(SchedulerPhase* next_phase) { this->run(); };

  /* Number of running tasks. A task is running if it has been assigned to a
   * worker and is not complete */
  size_t get_count() { return this->num_running_tasks.load(); }

protected:
  std::string name = "Launcher";
  /*Buffer to store not yet launched tasks. Currently unused. Placeholder in
   * case it becomes useful.*/
  TaskList task_buffer;
  WorkerList worker_buffer;
};

#endif // PARLA_PHASES_HPP
