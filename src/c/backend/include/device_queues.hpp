/*! @file device_queues.hpp
 *  @brief Provides interface for task queues and collections of task queues for
 * multidevice tasks.
 */
#ifndef PARLA_DEVICE_QUEUES_HPP
#define PARLA_DEVICE_QUEUES_HPP

#include "device_manager.hpp"
#include "runtime.hpp"
// #include <chrono>
#include <ctime>

struct taskStructure {             // Structure declaration
  int priority;         // Member (int variable)
  InnerTask *task;   // Member (string variable)
}; 
// TODO(wlr): FIXME Change these back to smart pointers. I'm leaking memory
// here...

/**
 * Per-device container for tasks that are waiting to be dequeued.
 * Supports both single and multi-device tasks.
 * Multi-device tasks are shared between DeviceQueues.
 * Should ONLY be used through PhaseManager.
 *
 * @tparam category the resource category (persistent/non-persistent) that this
 * queue supervises the phase of.
 */
template <typename ResourceCategory> class DeviceQueue {
  using MixedQueue_t = PriorityTaskQueue;
  using MDQueue_t = TaskQueue;

public:
  DeviceQueue() = default;
  DeviceQueue(Device *device) : device(device) {}

  /**
   * @return the device that this queue is associated with
   */
  Device *get_device() { return device; }

  /**
   * Enqueues a task on this device.
   * @param task the task to enqueue
   */

  int determine_priority(InnerTask *task) {
    int num_dependents = task->dependents.size(); // directly propotional
    int num_gpus_required = task->assigned_devices.size(); // inveresly propotional
    int start_time =  time(NULL); // directly propotional
    priority = (num_dependents * start_time) / num_gpus_required; // normalize and change this
    return priority;
    // critical path length to most recently spawned task
    // estimated completion time

  }
  void enqueue(InnerTask *task) {
    taskStructure new_task;
    new_task.priority = task->determine_priority(task);
    new_task.task = task;
    // std::cout << "DeviceQueue::enqueue() - " << task->get_name() <<
    // std::endl;

    // std::cout << "Mixed Queue size: " << mixed_queue.size() << std::endl;
    this->mixed_queue.push(new_task);
    num_tasks++;
  };

  /**
  * @return the next task that can be dequeued on this device.
  * If there are no tasks that can dequeued, returns nullptr.
  *
  * A task is can be dequeued if:
  * 1. It is a single-device task
  * 2. It is a multi-device task that is no longer waiting for its other
  * instances

  * We do not block on multi-device tasks that are still waiting for their other
  * instances. This allows progress on single-device tasks and avoids
  * deadlock if multi-device tasks are enqueued out of order.

  * This does not check resources, it only checks if the task is ready to
  * dequeue. It does not remove the returned task from the queue.
  */
  InnerTask *front() {

    // std::cout << "DeviceQueue::front()" << std::endl;
    // std::cout << "Waiting Queue size: " << waiting_queue.size() << std::endl;
    // std::cout << "Mixed Queue size: " << mixed_queue.size() << std::endl;
    // std::cout << "Num Tasks: " << num_tasks << std::endl;

    // First, check any waiting multi-device tasks
    if (!waiting_queue.empty()) {
      InnerTask *head = waiting_queue.front();
      int waiting_count = head->get_num_instances<ResourceCategory>();

      /*
      std::cout << "MD Head: " << head->get_name()
                << " Instances: " << waiting_count
                << " Removed: " << head->get_removed<ResourceCategory>()
                << std::endl;
       */

      // Any MD task that is no longer waiting should be blocking
      if (waiting_count < 1) {
        // Remove from waiting queue if dequeued by last instance
        if (head->get_removed<ResourceCategory>()) {
          // TODO(wlr): Should I remove this here?
          waiting_queue.pop_front();

          // TODO(wlr): Should num_tasks include waiting tasks?
          // (1)
          // this->num_tasks--;
        }
        return nullptr;
      }
    }

    if (!mixed_queue.empty()) {
      InnerTask *head = mixed_queue.front();
      int prev_waiting_count =
          head->decrement_num_instances<ResourceCategory>();

      // std::cout << "Mixed Head: " << head->get_name()
      //           << " Instances: " << prev_waiting_count
      //           << " Removed: " << head->get_removed<ResourceCategory>()
      //           << std::endl;

      // Check if the task is waiting for other instances
      if (prev_waiting_count <= 1) {
        // std::cout << "Return task: " << head->get_name() << std::endl;
        return head;
      } else {
        // If the task is still waiting, move it to the waiting queue
        std::cout << "Moving task to waiting queue: " << head->get_name()
                  << std::endl;
        waiting_queue.push_back(head);
        mixed_queue.pop();

        // TODO(wlr): Should num_tasks include waiting tasks?
        // (2)
        this->num_tasks--;
      }
    }

    return nullptr;
  }

  /**
   * Grabs the next task that can be dequeued on this device.
   * If there are no tasks that can dequeued, returns nullptr.
   * Removes the returned task from the queue.
   *
   * Should be called only after front() returns a non-null task.
   * Otherwise, the internal state may be modified (may push multi-device to
   * waiting queue).
   *
   * @return the previous task at HEAD of the queue.
   */
  InnerTask *pop() {
    // std::cout << "DeviceQueue::pop()" << std::endl;
    InnerTask *task = front();
    if (task != nullptr) {
      // std::cout << "Popping task: " << task->get_name() << std::endl;
      mixed_queue.pop();
      // Set removed status so task can be pruned from other queues
      task->set_removed<ResourceCategory>(true);
      num_tasks--;
    }
    return task;
  }

  inline size_t size() { return num_tasks.load(); }
  inline bool empty() { return mixed_queue.empty() && waiting_queue.empty(); }

protected:
  Device *device;
  MixedQueue_t mixed_queue;
  MDQueue_t waiting_queue;
  std::atomic<int> num_tasks{0};
};

// TODO(wlr): I don't know what to name this.
/**
 * Manages a group of DeviceQueues.
 * Supports both single and multi-device tasks.
 * Multi-device tasks are shared between DeviceQueues.
 *
 * @tparam category the resource category (persistent/non-persistent) that this
 * manager supervises
 */
template <typename ResourceCategory> class PhaseManager {
public:
  /**
   * Initializes a DeviceQueue for each device in the DeviceManager.
   * @param device_manager the DeviceManager to initialize from
   **/
  PhaseManager(DeviceManager *device_manager) {
    // std::cout << "Initializing PhaseManager" << std::endl;

    for (const DeviceType dev_type : architecture_types) {
      this->ndevices += device_manager->get_num_devices(dev_type);

      for (Device *device : device_manager->get_devices(dev_type)) {
        this->device_queues.emplace_back(
            new DeviceQueue<ResourceCategory>(device));
        // std::cout << "Initialized DeviceQueue for Device: "
        //           << device->get_name() << std::endl;
      }

      this->last_device_idx = 0;
    }

    // std::cout << "Initialized PhaseManager with " << this->ndevices
    //           << " devices" << std::endl;
    // std::cout << "Initialized PhaseManager with " <<
    // this->device_queues.size()
    //           << " queues" << std::endl;
  }

  ~PhaseManager() {
    for (auto &queue : this->device_queues) {
      delete queue;
    }
  }

  /**
   * Enqueues a task to the appropriate DeviceQueue(s).
   * @param task the task to enqueue. May be single or multi-device.
   **/
  void enqueue(InnerTask *task) {
    // std::cout << "pointer: " << reinterpret_cast<void *>(this) << std::endl;
    // std::cout << "ndevices: " << this->ndevices << std::endl;
    // std::cout << "nqueues: " << this->device_queues.size() << std::endl;
    // std::cout << "Enqueuing task to phase manager: " << task->get_name()
    //           << std::endl;
    task->set_num_instances<ResourceCategory>();
    for (auto device : task->assigned_devices) {
      this->device_queues[device->get_global_id()]->enqueue(task);
    }
    this->num_tasks++;
  }

  /**
   * @return the next task that can be dequeued on any device.
   * @see DeviceQueue::front
   *
   * Loops over all DeviceQueues in round-robin order to find the next dequeable
   * task. The search is restarted from the next DeviceQueue after the last
   * successful dequeue. A success increases the last_device_idx by 1.
   *
   * If there are no tasks remaining, returns nullptr.
   * Will infinitely loop if there are tasks remaining but none can be dequeued
   * on any device. (invalid state).
   *
   * Note that each call to DeviceQueue::front pushes the HEAD task to a waiting
   * queue if it is a multi-device task that hasn't reached HEAD on all
   * instances. This modifies the internal state of the DeviceQueue.
   */
  InnerTask *front() {
    // TODO(wlr): Hochan, can you check this?
    // I'm not sure if this is the right way to loop over dequeable tasks
    // Should we drain each device first, or try each device in
    // turn?

    // std::cout << "PhaseManager::front" << std::endl;

    int start_idx = last_device_idx;
    int end_idx = start_idx + ndevices;
    int current_idx = start_idx;

    bool has_task = this->size() > 0;
    // std::cout << "Size of PhaseManager: " << this->size() << std::endl;
    while (has_task) {

      // std::cout << "Size of PhaseManager: " << this->size() << std::endl;

      // Loop over all devices starting from after last success location
      for (int i = start_idx; i < end_idx; ++i) {
        current_idx = i % ndevices;

        // Try to get a non-waiting task
        // std::cout << "Trying DeviceQueue " << current_idx << " Device: "
        //          <<
        //          this->device_queues[current_idx]->get_device()->get_name()
        //          << std::endl;

        InnerTask *task = this->device_queues[current_idx]->front();
        if (task != nullptr) {
          // std::cout << "Not null." << std::endl;
          // std::cout << "Found task: " << task->get_name() << std::endl;
          this->last_device_idx = ++current_idx;
          return task;
        }
      }

      has_task = this->size() > 0;
    }

    // If we get here, there are no tasks that can be dequeued
    // This should only happen if called on an empty phase
    // std::cout << "No tasks found" << std::endl;
    return nullptr;
  }

  /**
   * Removed the previous head task from the queue.
   * @return the removed task
   * @see DeviceQueue::pop
   **/
  InnerTask *pop() {
    // std::cout << "PhaseManager::pop" << std::endl;
    int idx = (this->last_device_idx - 1) % this->ndevices;
    // std::cout << "Popping from DeviceQueue " << idx << std::endl;
    InnerTask *task = this->device_queues[idx]->pop();
    // std::cout << "Popped task: " << task->get_name() << std::endl;
    this->num_tasks--;
    return task;
  }

  inline size_t size() { return this->num_tasks.load(); }
  inline size_t get_num_devices() { return this->ndevices; }
  inline size_t get_num_device_queues() { return this->device_queues.size(); }

protected:
  std::vector<DeviceQueue<ResourceCategory> *> device_queues;

  int last_device_idx{0};
  // DeviceType last_device_type{CPU};

  int ndevices{0};
  std::atomic<int> num_tasks{0};
};

#endif // PARLA_DEVICE_QUEUE_HPP
