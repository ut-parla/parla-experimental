#include "include/phases.hpp"
#include "include/device.hpp"
#include "include/policy.hpp"
#include "include/profiling.hpp"
#include "include/resources.hpp"
#include "include/runtime.hpp"
#include <algorithm>

/**************************/
// Mapper Implementation

void Mapper::enqueue(InnerTask *task) { this->mappable_tasks.push_back(task); }

void Mapper::enqueue(std::vector<InnerTask *> &tasks) {
  this->mappable_tasks.push_back(tasks);
}

size_t Mapper::get_count() {
  size_t count = this->mappable_tasks.atomic_size();
  return count;
}

void Mapper::run(SchedulerPhase *next_phase) {
  NVTX_RANGE("Mapper::run", NVTX_COLOR_LIGHT_GREEN)

  MemoryReserver *memory_reserver = dynamic_cast<MemoryReserver *>(next_phase);
  std::cout << "Mapper::run" << std::endl;

  // TODO: Refactor this so its readable without as many nested conditionals

  // This is a non-critical region
  // Comment(wlr): Why is this a noncritical region?

  // Assumptions:
  // Scheduler maps a task to a device.
  // Scheduler does not reserve any resource at this phase.

  // TODO(hc): for now, I'm planning task mapping without policy.

  this->status.reset();

  bool has_task = true;

  has_task = this->get_count() > 0;
  while (has_task) {
    InnerTask *task = this->mappable_tasks.front_and_pop();

    // TODO(wlr): Testing
    // Assign two random devices to each task
    // TODO(wlr): This is just used for random testing policy
    //            Remove this when we implement a policy.

    std::vector<Device *> devices;
    devices.insert(devices.end(),
                   this->device_manager->get_devices(DeviceType::ANY).begin(),
                   this->device_manager->get_devices(DeviceType::ANY).end());
    std::random_shuffle(devices.begin(), devices.end());

    std::cout << "Mapping task " << task->get_name() << " to devices "
              << devices[0]->get_name() << " and " << devices[1]->get_name()
              << std::endl;

    ResourcePool_t sample;
    sample.set(MEMORY, 0);
    sample.set(VCU, 500);

    task->assigned_devices.push_back(devices[0]);
    task->assigned_devices.push_back(devices[1]);

    // TODO(wlr): Maybe use shared_ptr<ResourcePool_t> to pass from existing
    // res_req? Cannot be shared pools between devices. These are copies here.
    task->device_constraints.insert({devices[0]->get_global_id(), sample});
    task->device_constraints.insert({devices[1]->get_global_id(), sample});

    this->mapped_tasks_buffer.push_back(task);
    has_task = this->get_count() > 0;
  } // while there are mappable tasks

  for (InnerTask *mapped_task : this->mapped_tasks_buffer) {
    mapped_task->notify_dependents(this->enqueue_buffer, Task::MAPPED);
    this->scheduler->enqueue_tasks(this->enqueue_buffer);
    this->enqueue_buffer.clear();

    bool enqueue_flag =
        (mapped_task->num_unreserved_dependencies.fetch_sub(1) == 1);

    if (enqueue_flag) {
      mapped_task->set_status(Task::RESERVABLE);
      memory_reserver->enqueue(mapped_task);
    }
  }

  this->mapped_tasks_buffer.clear();
  std::cout << "Mapper::run done" << std::endl;
}

/**************************/
// Reserved Phase implementation

void MemoryReserver::enqueue(InnerTask *task) {
  std::cout << "MemoryReserver::enqueue: " << task->get_name() << std::endl;
  this->reservable_tasks->enqueue(task);
}

void MemoryReserver::enqueue(std::vector<InnerTask *> &tasks) {
  for (InnerTask *task : tasks) {
    this->enqueue(task);
  }
}

size_t MemoryReserver::get_count() {
  size_t count = this->reservable_tasks->size();
  return count;
}

bool MemoryReserver::check_resources(InnerTask *task) {
  bool status = true;
  for (Device *device : task->assigned_devices) {
    ResourcePool_t &task_pool =
        task->device_constraints[device->get_global_id()];
    ResourcePool_t &device_pool = device->get_reserved_pool();

    status = device_pool.check_greater<ResourceCategory::PERSISTENT>(task_pool);

    if (!status) {
      break;
    }
  }
  return status;
}

void MemoryReserver::reserve_resources(InnerTask *task) {
  // TODO(wlr): Add runtime error check if resource failure

  for (Device *device : task->assigned_devices) {
    ResourcePool_t &task_pool =
        task->device_constraints[device->get_global_id()];
    ResourcePool_t &device_pool = device->get_reserved_pool();

    device_pool.decrease<ResourceCategory::PERSISTENT>(task_pool);
  }
}

void MemoryReserver::run(SchedulerPhase *next_phase) {
  NVTX_RANGE("MemoryReserver::run", NVTX_COLOR_LIGHT_GREEN)

  RuntimeReserver *runtime_reserver =
      dynamic_cast<RuntimeReserver *>(next_phase);
  std::cout << "MemoryReserver::run" << std::endl;
  std::cout << "runtime_reserver pointer: "
            << reinterpret_cast<void *>(runtime_reserver->get_runnable_tasks())
            << std::endl;
  this->status.reset();

  // Only one thread can reserve memory at a time.
  // Useful for a multi-threaded scheduler. Not needed for a single-threaded.
  // std::unique_lock<std::mutex> lock(this->mtx);

  // TODO:: Dummy implementation that just passes tasks through
  bool has_task = this->get_count() > 0;
  while (has_task) {
    InnerTask *task = this->reservable_tasks->front();

    if (task == nullptr) {
      throw std::runtime_error("MemoryReserver::run: task is nullptr");
    }

    // Is there enough memory on the devices to schedule this task?
    bool can_reserve = this->check_resources(task);
    if (can_reserve) {
      this->reserve_resources(task);
      this->reservable_tasks->pop();
      this->reserved_tasks_buffer.push_back(task);
    } else {
      // TODO:(wlr) we need some break condition to allow the scheduler to
      // continue if not enough resources are available Hochan, do you
      // have any ideas? One failure per scheduler loop (written here) is
      // bad. Is one failure per device per scheduler loop better?
      break;
    }

    has_task = this->get_count() > 0;
  }

  for (InnerTask *reserved_task : this->reserved_tasks_buffer) {
    reserved_task->notify_dependents(this->enqueue_buffer, Task::RESERVED);
    this->scheduler->enqueue_tasks(this->enqueue_buffer);
    this->enqueue_buffer.clear();

    // TODO:(wlr) Create and possibly enqueue data movement tasks

    // Possibly enqueue this task
    bool enqueue_flag =
        (reserved_task->num_blocking_dependencies.fetch_sub(1) == 1);

    if (enqueue_flag) {
      reserved_task->set_status(Task::RUNNABLE);
      std::cout << "ENQUEUE FROM PHASE: " << std::endl;
      runtime_reserver->enqueue(reserved_task);
    }
  }

  this->reserved_tasks_buffer.clear();
  std::cout << "MemoryReserver::run done" << std::endl;
  std::cout << "runtime_reserver pointer: "
            << reinterpret_cast<void *>(runtime_reserver->get_runnable_tasks())
            << std::endl;
}

/**************************/
// Ready Phase implementation

void RuntimeReserver::enqueue(InnerTask *task) {
  std::cout << "RuntimeReserver::enqueue: " << task->get_name() << std::endl;
  this->runnable_tasks->enqueue(task);
}

void RuntimeReserver::enqueue(std::vector<InnerTask *> &tasks) {
  for (InnerTask *task : tasks) {
    this->enqueue(task);
  }
}

size_t RuntimeReserver::get_count() {
  size_t count = this->runnable_tasks->size();
  return count;
}

bool RuntimeReserver::check_resources(InnerTask *task) {
  bool status = true;
  for (Device *device : task->assigned_devices) {
    ResourcePool_t &task_pool =
        task->device_constraints[device->get_global_id()];
    ResourcePool_t &device_pool = device->get_reserved_pool();

    status =
        device_pool.check_greater<ResourceCategory::NON_PERSISTENT>(task_pool);

    if (!status) {
      break;
    }
  }
  return status;
}

void RuntimeReserver::reserve_resources(InnerTask *task) {
  // TODO(wlr): Add runtime error check if resource failure

  for (Device *device : task->assigned_devices) {
    ResourcePool_t &task_pool =
        task->device_constraints[device->get_global_id()];
    ResourcePool_t &device_pool = device->get_reserved_pool();

    device_pool.decrease<ResourceCategory::NON_PERSISTENT>(task_pool);
  }
}

void RuntimeReserver::run(SchedulerPhase *next_phase) {
  NVTX_RANGE("RuntimeReserver::run", NVTX_COLOR_LIGHT_GREEN)

  std::cout << "RuntimeReserver::run" << std::endl;
  std::cout << "runtime_reserver pointer: "
            << reinterpret_cast<void *>(this->get_runnable_tasks())
            << std::endl;

  Launcher *launcher = dynamic_cast<Launcher *>(next_phase);
  this->status.reset();

  // Only one thread can reserve runtime resources at a time.
  // Useful for a multi-threaded scheduler. Not needed for a single-threaded.
  // std::unique_lock<std::mutex> lock(this->mtx);

  std::cout << "NDevices: " << this->runnable_tasks->get_num_devices()
            << std::endl;
  std::cout << "NQueues: " << this->runnable_tasks->get_num_device_queues()
            << std::endl;

  bool has_task = true;

  while (has_task) {
    has_task = this->get_count() > 0;

    if (has_task) {
      std::cout << "bfront runtime_reserver pointer: "
                << reinterpret_cast<void *>(this->get_runnable_tasks())
                << std::endl;
      InnerTask *task = this->runnable_tasks->front();
      std::cout << "afront runtime_reserver pointer: "
                << reinterpret_cast<void *>(this->get_runnable_tasks())
                << std::endl;

      if (task == nullptr) {
        throw std::runtime_error("RuntimeReserver::run: task is nullptr");
      }

      bool has_resources = check_resources(task);

      if (has_resources) {
        bool has_thread = scheduler->workers.get_num_available_workers() > 0;

        if (has_thread) {
          std::cout << "bpop runtime_reserver pointer: "
                    << reinterpret_cast<void *>(this->get_runnable_tasks())
                    << std::endl;
          InnerTask *task = this->runnable_tasks->pop();
          std::cout << "apop runtime_reserver pointer: "
                    << reinterpret_cast<void *>(this->get_runnable_tasks())
                    << std::endl;
          InnerWorker *worker = scheduler->workers.dequeue_worker();

          // Decrease Resources
          this->reserve_resources(task);
          std::cout << "areserve runtime_reserver pointer: "
                    << reinterpret_cast<void *>(this->get_runnable_tasks())
                    << std::endl;

          launcher->enqueue(task, worker);
          std::cout << "alaunch runtime_reserver pointer: "
                    << reinterpret_cast<void *>(this->get_runnable_tasks())
                    << std::endl;

          this->status.increase(Ready::success);
          std::cout << "HERE: astatus runtime_reserver pointer: "
                    << reinterpret_cast<void *>(this->get_runnable_tasks())
                    << std::endl;
        } else {
          this->status.increase(Ready::worker_miss);
          break; // No more workers available
        }
      } else {
        this->status.increase(Ready::resource_miss);
        break; // No more resources available
      }
    } else {
      this->status.increase(Ready::task_miss);
      break; // No more tasks available
    }
  }

  std::cout << "RuntimeReserver::run done" << std::endl;
  std::cout << "runtime_reserver pointer: "
            << reinterpret_cast<void *>(this->get_runnable_tasks())
            << std::endl;
}

/**************************/
// Launcher Phase implementation

void Launcher::enqueue(InnerTask *task, InnerWorker *worker) {
  NVTX_RANGE("Launcher::enqueue", NVTX_COLOR_LIGHT_GREEN)

  // Immediately launch task
  task->set_state(Task::RUNNING);
  this->num_running_tasks++;

  // Assign task to thread and notify via c++ condition variable.
  // No GIL needed until worker wakes.
  worker->assign_task(task);

  // std::cout << "Assigned " << task->name << " to " << worker->thread_idx
  //           << std::endl;
  LOG_INFO(WORKER, "Assigned {} to {}", task, worker);
}

void Launcher::run() {
  throw std::runtime_error("Launcher::run() not implemented.");
}
