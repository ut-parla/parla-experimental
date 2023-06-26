#ifndef PARLA_MEMORY_MANNGER_HPP
#define PARLA_MEMORY_MANNGER_HPP

#include "device_manager.hpp"
#include "parray.hpp"


/**
 * @brief Node type of a PArray eviction double-linked list.
 */
class PArrayNode {
public:
  PArrayNode(parray::InnerPArray *parr, size_t prior = 0) :
      parray(parr), priority(prior), next(nullptr), prev(nullptr)
  {}

  /// Pointer of a PArray instance
  parray::InnerPArray *parray;
  /// Priority of the node
  /// TODO(hc): This is not used, but keep it for the future
  size_t priority;
  /// Pointers to the next and the previous PArrayNodes
  PArrayNode *next;
  PArrayNode *prev;
};

/**
 * @brief Double-linked list of candidate PArrays for eviction.
 * @details PArray eviction manager selects and evicts PArray instances
 * in this list depending on an eviction policy.
 * Note that an eviction manager manages this list for each device.
 */
class DoubleLinkedList {
public:

  /**
   * @brief Print the current list.
   */
  void print() {
    PArrayNode *node = this->head_;
    std::cout << "\n";
    while (node != nullptr) {
      std::cout << node->parray->id << " -> \n"; 
      node = node->next;
    }
    std::cout << "\n";
  }

  /**
   * @brief Append a PArray node to the list.
   * @detail The first PArray of the list is set to both head and tail, and
   * the last added PArray is set to tail.
   *
   * @param node PArray node to be appended
   */
  void append(PArrayNode *node) {
    this->mtx_.lock();
    if (this->list_size_ == 0) {
      this->head_ = node;
      this->tail_ = node;
    } else {
      this->tail_->next = node;
      node->prev = this->tail_;
      this->tail_ = node;
    }
    this->list_size_ += 1;
    this->mtx_.unlock();
  }

  /**
   * @brief Insert a PArray node between `node` and `node->next`.
   *
   * @param node existing PArray node where `new_node` is being linked
   * @param new_node PArray node to be appended after `node`
   */
  void insert_after(PArrayNode *node, PArrayNode *new_node) {
    this->mtx_.lock();
    if (node->next != nullptr) {
      node->next->prev = new_node;
      new_node->next = node->next;
    } else {
      this->tail_ = new_node;
    }
    node->next = new_node;
    new_node->prev = node;
    this->mtx_.unlock();
  }

  /**
   * @brief Insert a PArray node between `node` and `node->prev`.
   *
   * @param node existing PArray node where `new_node` is being linked
   * @param new_node PArray node to be appended before `node`
   */
  void insert_before(PArrayNode *node, PArrayNode *new_node) {
    this->mtx_.lock();
    if (node->prev != nullptr) {
      node->prev->next = new_node;
      new_node->prev = node->prev;
    } else {
      this->head_ = new_node;
    }
    node->prev = new_node;
    new_node->next = node;
    this->mtx_.unlock();
  }

  /**
   * @brief Remove and return the current head PArray node from a list.
   */
  PArrayNode *remove_head() {
    this->mtx_.lock();
    PArrayNode *old_head = this->head_;
    if (old_head != nullptr) {
      this->remove_unsafe(old_head); 
    }
    this->mtx_.unlock();
    return old_head;
  }

  /**
   * @brief Remove a node and return true if it is removed false otherwise.
   *
   * @param node PArray node to be removed from a list
   */
  bool remove(PArrayNode *node) {
    this->mtx_.lock();
    bool rv = this->remove_unsafe(node);
    this->mtx_.unlock();
    return rv;
  }

  /**
   * @brief Remove a node and return true if it is removed false otherwise.
   * This function is not thread safe.
   *
   * @param node PArray node to be removed from a list
   */
  bool remove_unsafe(PArrayNode *node) {
    if (node->prev == nullptr && node->next == nullptr &&
        node != this->head_ && node != this->tail_) {
      // If a node is not in a list, do nothing and return false.
      return false;
    }

    if (this->list_size_ == 1) {
      // A node is a single node in a list.
      this->head_ = this->tail_ = nullptr;
    } else {
      if (this->head_ == node) {
        // A node is a head, and so break link of node->next->prev.
        this->head_ = node->next;
        node->next->prev = nullptr;
      } else if (this->tail_ == node) {
        // A node is a tail, and so break link of node->prev->next.
        this->tail_ = node->prev;
        node->prev->next = nullptr;
      } else {
        // A node is in the middle of a list, and so break two links.
        node->prev->next = node->next;
        node->next->prev = node->prev;
      }
    }
    node->prev = node->next = nullptr;
    this->list_size_ -= 1;
    return true;
  }

  /**
   * @brief Return a size of a list.
   */
  size_t size() {
    this->mtx_.lock();
    size_t list_size = this->list_size_;
    this->mtx_.unlock();
    return list_size;
  }

  /**
   * @brief Return the current head.
   * This function is not thread safe.
   */
  PArrayNode *get_head() {
    return this->head_;
  }

  /**
   * @brief Return the current tail.
   * This function is not thread safe.
   */
  PArrayNode *get_tail() {
    return this->tail_;
  }

private:
  PArrayNode *head_{nullptr};
  PArrayNode *tail_{nullptr};
  std::mutex mtx_;
  size_t list_size_{0};
};


/**
 * @brief Least-recently-used policy based eviction manager for a device.
 * @details It holds PArrays which are not referenced by tasks which are
 * between task mapping and termination phases.
 */
class LRUDeviceEvictionManager {
public:
  struct PArrayMetaInfo {
    // Points to a PArray node if it exists
    PArrayNode *parray_node_ptr;
    // The number of references to a PArray
    size_t ref_count;
  };

  LRUDeviceEvictionManager(DevID_t dev_id) : dev_id_(dev_id) {}

  /**
   * @brief A task refers `parray` in the device.
   * @detail This function is called when a task being mapped
   * refers `parray`. This increases a reference count of the PArray
   * and removes it from a zero-referenced list if it exists.
   *
   * @param parray pointer to a parray to be referred by a task
   */
  void acquire_data(parray::InnerPArray *parray) {
    this->mtx_.lock();
    uint64_t parray_id = parray->id;
    auto found = this->parray_reference_counts_.find(parray_id);
    if (found == this->parray_reference_counts_.end()) {
      // Add `parray` to a zr list if it does not exist.
      PArrayNode *parray_node = new PArrayNode(parray);
      this->parray_reference_counts_[parray_id] =
          PArrayMetaInfo{parray_node, 1};
    } else {
      // If `parray` is already in a zr list, removes it
      // from the list and decreases its reference count.
      found->second.ref_count++; 
      this->zr_parray_list_.remove(found->second.parray_node_ptr);
    }
    this->mtx_.unlock();
  }

  /**
   * @brief A task is finished and releases `parray` in the device.
   * @detail This function is called by a worker thread when a task
   * assigned to that completes. So, the thread also releases a
   * `parray`, and decreases a reference count of that in the device.
   * If the reference count becomes 0, the `parray` is added to
   * its zero-referenced list.
   *
   * @param parray pointer to a parray to be released by a task
   */
  void release_data(parray::InnerPArray *parray) {
    this->mtx_.lock();
    uint64_t parray_id = parray->id;
    auto found = this->parray_reference_counts_.find(parray_id);
    if (found != this->parray_reference_counts_.end()) {
      found->second.ref_count--; 
      if (found->second.ref_count == 0) {
        // If none of task referes to `parray`, add it to
        // a zr list.
        this->zr_parray_list_.append(found->second.parray_node_ptr);
      }
    }
    this->mtx_.unlock();
  }


  /**
   * @brief Return a size of a list.
   */
  size_t size() {
    size_t zr_parray_list_size{0};
    this->mtx_.lock();
    zr_parray_list_size = zr_parray_list_.size();
    this->mtx_.unlock();
    return zr_parray_list_size;
  }

  /**
   * @brief Remove and return a head of the zero-referenced list.
   * @detail This function is not thread safe since it assumes that only
   * the scheduler thread calls into this function during eviction. 
   */
  PArrayNode *remove_and_return_head_from_zrlist() {
    PArrayNode* old_head{nullptr};
    this->mtx_.lock();
    old_head = this->zr_parray_list_.remove_head();
    this->mtx_.unlock();
    return old_head;
  }

  /**
   * @brief This function clears all existing PArrays in the
   * zero-referenced list.
   * @detail This function has two purposes.
   * First, it is used to fix unlinked Python and C++ PArray
   * instances. It is possible that Python PArrays are destroyed
   * due to, for example, out-of-scope. Then, C++ PArrays
   * start to hold invalid Python PArray pointers.
   * When a scheduler starts PArray eviction, it is possible that
   * the C++ PArrays holding invalid Python PArrays are chosen 
   * as evictable PArrays and causes segmentation fault.
   * This function removes those PArrays in advance to avoid
   * this issue (But users should be aware of and take care of this scenario).
   * The second purpose is to allow users to clear all memory
   * related states managed by the Parla runtime.
   */
  // TODO(hc): This bulk flushing is not ideal IMO. The Parla runtime
  //           should provide a function that flushes only a single PArray.
  //           I am postponing this work since we need to take care of
  //           the zero-referenced list, but I have higher priorities.
  void clear_all_instances() {
    this->mtx_.lock();
    PArrayNode* head{nullptr};
    do {
      head = this->zr_parray_list_.remove_head(); 
    } while (head != nullptr);
    this->mtx_.unlock();
  }

private:
  /// This eviction manager manages PArray instances in this device
  DevID_t dev_id_;
  std::mutex mtx_;
  /// Key: PArray ID, Value: Meta information including reference
  /// count of a PArray
  std::unordered_map<uint64_t, PArrayMetaInfo> parray_reference_counts_;
  /// A list of zero-referenced PArrays.
  DoubleLinkedList zr_parray_list_;
};


/**
 * @brief Least-recently-used policy based global eviction manager.
 * @details External components access and manipulate PArray instances in any
 * device through this manager.
 */
class LRUGlobalEvictionManager {
public:
  LRUGlobalEvictionManager(DeviceManager *device_manager) :
    device_manager_(device_manager) {
    this->device_mm_.resize(
        device_manager->template get_num_devices<DeviceType::All>());
    for (size_t i = 0; i < this->device_mm_.size(); ++i) {
      this->device_mm_[i] = new LRUDeviceEvictionManager(i);
    }
  }

  /**
   * @brief A task refers `parray` in `dev_id` device.
   * @detail This function is called when a task being mapped
   * refers `parray`. This increases a reference count of the PArray
   * and removes it from a zero-referenced list if it exists.
   *
   * @param parray pointer to a parray to be referred by a task
   * @param dev_id device id of a device to access its information
   */
  void acquire_data(parray::InnerPArray *parray, DevID_t dev_id) {
    this->device_mm_[dev_id]->acquire_data(parray);
  }

  /**
   * @brief A task is finished and releases `parray` in `dev_id` device.
   * @detail This function is called by a worker thread when a task
   * assigned to that completes. So, the thread also releases a
   * `parray`, and decreases a reference count of that in the device.
   * If the reference count becomes 0, the `parray` is added to
   * its zero-referenced list.
   *
   * @param parray pointer to a parray to be released by a task
   * @param dev_id device id of a device to access its information
   */
  void release_data(parray::InnerPArray *parray, DevID_t dev_id) {
    this->device_mm_[dev_id]->release_data(parray);
  }

  /**
   * @brief Return a size of a list.
   *
   * @param dev_id device id of a device to access its information
   */
  size_t size(DevID_t dev_id) {
    return this->device_mm_[dev_id]->size();
  }

  /**
   * @brief Remove and return a head of the zero-referenced list.
   * @detail This function is not thread safe since it assumes that only
   * the scheduler thread calls into this function during eviction. 
   *
   * @param dev_id device id of a device to access its information
   */
  void *remove_and_return_head_from_zrlist(DevID_t dev_id) {
    PArrayNode *old_head =
        this->device_mm_[dev_id]->remove_and_return_head_from_zrlist();
    void *py_parray{nullptr};
    if (old_head != nullptr) {
      parray::InnerPArray *c_parray = old_head->parray;
      py_parray = c_parray->get_py_parray();
    }
    return py_parray;
  }

  void clear_all_instances() {
    for (size_t i = 0; i < device_mm_.size(); ++i) {
      device_mm_[i]->clear_all_instances();
    }
  }

private:
  /// Device manager managing system environment
  DeviceManager *device_manager_;
  /// A list of LRU-based eviction manager for each device
  std::vector<LRUDeviceEvictionManager *> device_mm_;
};

#endif
