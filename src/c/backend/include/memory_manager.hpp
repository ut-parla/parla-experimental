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

  /// Pointer to the PArray instance
  parray::InnerPArray *parray;
  /// Priority of the node
  /// TODO(hc): This is not used, but keep it for the future
  size_t priority;
  /// Pointers to the next and the previous PArrayNodes
  PArrayNode *next;
  PArrayNode *prev;
};

/**
 * @brief Double-linked list of evicatble PArrays.
 * @detail List of PArrays that none of tasks refer to.
 * The PArray eviction manager selects and evicts PArray objects
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
    std::cout << "[Evictable PArray List]\n";
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
    std::lock_guard guard(this->mtx_);
    if (this->list_size_ == 0) {
      this->head_ = node;
      this->tail_ = node;
    } else {
      this->tail_->next = node;
      node->prev = this->tail_;
      this->tail_ = node;
    }
    this->list_size_ += 1;
  }

  /**
   * @brief Insert a PArray node between `node` and `node->next`.
   *
   * @param node existing PArray node where `new_node` is being linked
   * @param new_node PArray node to be appended after `node`
   */
  void insert_after(PArrayNode *node, PArrayNode *new_node) {
    std::lock_guard guard(this->mtx_);
    if (node->next != nullptr) {
      // * -> x, and want to add n.
      // * -> n -> x
      node->next->prev = new_node;
      new_node->next = node->next;
    } else {
      // * -> NULL, and want to add n.
      // * -> n -> NULL
      this->tail_ = new_node;
    }
    node->next = new_node;
    new_node->prev = node;
  }

  /**
   * @brief Insert a PArray node between `node` and `node->prev`.
   *
   * @param node existing PArray node where `new_node` is being linked
   * @param new_node PArray node to be appended before `node`
   */
  void insert_before(PArrayNode *node, PArrayNode *new_node) {
    std::lock_guard guard(this->mtx_);
    if (node->prev != nullptr) {
      // p -> *, and want to add n.
      // p -> n -> *
      node->prev->next = new_node;
      new_node->prev = node->prev;
    } else {
      // NULL -> *, and want to add n.
      // n -> *
      this->head_ = new_node;
    }
    node->prev = new_node;
    new_node->next = node;
  }

  /**
   * @brief Remove and return the current head PArray node from this list.
   */
  PArrayNode *remove_head() {
    std::lock_guard guard(this->mtx_);
    PArrayNode *old_head = this->head_;
    if (old_head != nullptr) {
      this->remove_unsafe(old_head); 
    }
    return old_head;
  }

  /**
   * @brief Remove a node and return true if it is removed, false otherwise.
   *
   * @param node PArray node to be removed from this list
   */
  bool remove(PArrayNode *node) {
    std::lock_guard guard(this->mtx_);
    return this->remove_unsafe(node);
  }

  /**
   * @brief Remove a node and return true if it is removed, false otherwise.
   * This function is not thread safe.
   *
   * @param node PArray node to be removed from this list
   */
  bool remove_unsafe(PArrayNode *node) {
    if (node->prev == nullptr && node->next == nullptr &&
        node != this->head_ && node != this->tail_) {
      // If a node is not in this list, do nothing and return false.
      return false;
    }

    if (this->list_size_ == 1) {
      // A node is a single node in this list.
      this->head_ = this->tail_ = nullptr;
    } else {
      if (this->head_ == node) {
        // If a node is the head of this list, set the next node as the head.
        this->head_ = node->next;
        node->next->prev = nullptr;
      } else if (this->tail_ == node) {
        // If a node is the tail of this list, set the previous node as
        // the tail.
        this->tail_ = node->prev;
        node->prev->next = nullptr;
      } else {
        // If a node is in the middle of this list, not as the head or the
        // tail, break its prev/next links.
        node->prev->next = node->next;
        node->next->prev = node->prev;
      }
    }
    // Detach a node from this list.
    node->prev = node->next = nullptr;
    this->list_size_ -= 1;
    return true;
  }

  /**
   * @brief Return the size of this list.
   */
  size_t size() {
    std::lock_guard guard(this->mtx_);
    return this->list_size_;
  }

  /**
   * @brief Return the current head of this list.
   * This function is not thread safe.
   */
  PArrayNode *get_head() {
    return this->head_;
  }

  /**
   * @brief Return the current tail of this list.
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
 * @brief Least-recently-used (LRU) policy based eviction manager for
 * a single device.
 * @detail It holds PArrays which are not referenced to by tasks which are
 * between task mapping and runahead states.
 */
class LRUDeviceEvictionManager {
public:
  struct ParrayRefInfo {
    // Points to a PArray node if it exists
    PArrayNode *parray_node_ptr;
    // Reference count of a PArray
    size_t ref_count;
  };

  LRUDeviceEvictionManager(DevID_t dev_id) : dev_id_(dev_id) {}

  void print() {
    this->zr_parray_list_.print();
  }

  /**
   * @brief A task started to refer to `parray` in the device.
   * @detail This function is called when a task being mapped
   * starts to refer to `parray`. This increases the reference count
   * of the PArray, and removes it from a zero-referenced list if it existed.
   *
   * @param parray pointer to a parray to be referred by a task
   */
  void grab_parray_reference(parray::InnerPArray *parray,
      size_t& accumulated_used_parray_bytes) {
    std::lock_guard guard(this->mtx_);
    uint64_t parray_id = parray->id;
    auto found = this->parray_reference_counts_.find(parray_id);
    if (found == this->parray_reference_counts_.end()) {
      // Add `parray` to the zr list if it does not exist.
      PArrayNode *parray_node = new PArrayNode(parray);
      this->parray_reference_counts_[parray_id] =
          ParrayRefInfo{parray_node, 1};
      // This PArray would be used for the first time, so accumulate
      // its bytes.
      accumulated_used_parray_bytes += parray->get_size();
    } else {
      // If `parray` is already in the zr list, removes it
      // from the list and increases its reference count.
      found->second.ref_count++; 
      if (this->zr_parray_list_.remove(found->second.parray_node_ptr)) {
        // If this PArray was in the zero-referenced list, it implies
        // new allocation will be performed.
        accumulated_used_parray_bytes += parray->get_size();
      }
    }
  }

  /**
   * @brief A task is finished and releases `parray` in the device.
   * @detail This function is called by a worker thread when a task
   * assigned to that is completed. The thread releases the
   * `parray` instance in that device, and decreases its reference count.
   * If the reference count becomes 0, the `parray` is added to
   * the zero-referenced list.
   *
   * @param parray pointer to a parray to be released by a task
   */
  void release_parray_reference(parray::InnerPArray *parray,
      size_t& evictable_parray_bytes) {
    std::lock_guard guard(this->mtx_);
    uint64_t parray_id = parray->id;
    auto found = this->parray_reference_counts_.find(parray_id);
    if (found != this->parray_reference_counts_.end()) {
      found->second.ref_count--; 
      if (found->second.ref_count == 0) {
        // If none of the tasks referes to `parray`, add it to
        // the zr list.
        this->zr_parray_list_.append(found->second.parray_node_ptr);
        // PArrays in the zero-referenced list are evictable, and so track
        // its bytes.
        evictable_parray_bytes += parray->get_size();
      }
    }
  }

  /**
   * @brief Return a size of a list.
   */
  size_t size() {
    std::lock_guard guard(this->mtx_);
    return zr_parray_list_.size();
  }

  /**
   * @brief Remove and return the head of the zero-referenced list.
   * @detail This removes and returns the head of the zero-referenced list
   * to be evicted from this device.
   * Note that this function is not thread safe since it assumes that only
   * the scheduler thread calls into this function during eviction. 
   */
  PArrayNode *remove_and_return_head_from_zrlist(
      size_t& accumulated_released_parray_bytes, size_t& evictable_parray_bytes) {
    std::lock_guard guard(this->mtx_);
    PArrayNode* head = this->zr_parray_list_.remove_head();
    // This function is called when this head is evicted.
    accumulated_released_parray_bytes += head->parray->get_size();
    evictable_parray_bytes -= head->parray->get_size();
    return head;
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
  /// Key: PArray ID, Value: Reference count information of a PArray
  std::unordered_map<uint64_t, ParrayRefInfo> parray_reference_counts_;
  /// A list of zero-referenced PArrays.
  DoubleLinkedList zr_parray_list_;
};


/**
 * @brief Least-recently-used policy (LRU) based global eviction manager.
 * @detail This manages and evicts PArrays if necessary.
 */
class LRUGlobalEvictionManager {
public:
  LRUGlobalEvictionManager(DeviceManager *device_manager) :
    device_manager_(device_manager) {
    DevID_t num_devices =
        device_manager->template get_num_devices<ParlaDeviceType::All>();
    this->device_mm_.resize(num_devices);
    for (size_t i = 0; i < this->device_mm_.size(); ++i) {
      this->device_mm_[i] = new LRUDeviceEvictionManager(i);
    }
    this->evictable_parray_bytes_.resize(num_devices);
  }

  void print_stats() {
    size_t total_evictable_bytes{0};
    for (DevID_t d = 0; d < this->evictable_parray_bytes_.size(); ++d) {
      total_evictable_bytes += this->evictable_parray_bytes_[d];
    }
    std::cout << "Total accumulated used parray bytes:" <<
      this->accumulated_used_parray_bytes_ << ", total accumulated" <<
      " released parray bytes:" << this->accumulated_released_parray_bytes_ <<
      ", evictable parray bytes:" << total_evictable_bytes << "\n";
  }

  /**
   * @brief A task refers to `parray` in `dev_id` device.
   * @detail This function is called when a task being mapped
   * refers `parray`. This increases a reference count of the PArray
   * and removes it from the zero-referenced list if it existed.
   *
   * @param parray pointer to a parray to be referred to by a task
   * @param dev_id device id of a device to access its information
   */
  void grab_parray_reference(parray::InnerPArray *parray, DevID_t dev_id) {
    this->device_mm_[dev_id]->grab_parray_reference(
        parray, this->accumulated_used_parray_bytes_);
  }

  /**
   * @brief A task is finished and releases `parray` in `dev_id` device.
   * @detail This function is called by a worker thread when a task
   * assigned to that completes. So, the thread releases `parray`,
   * and decreases its reference count in the device.
   * If the reference count becomes 0, the `parray` is added to
   * the zero-referenced list.
   *
   * @param parray pointer to a parray to be released by a task
   * @param dev_id device id of a device to access its information
   */
  void release_parray_reference(parray::InnerPArray *parray, DevID_t dev_id) {
    this->device_mm_[dev_id]->release_parray_reference(
        parray, this->evictable_parray_bytes_[dev_id]);
  }

  /**
   * @brief Return the size of the zero-referenced list.
   *
   * @param dev_id device id of a device to access its information
   */
  size_t size(DevID_t dev_id) {
    return this->device_mm_[dev_id]->size();
  }

  /**
   * @brief Remove and return the head of the zero-referenced list.
   * @detail This removes and returns the head of the zero-referenced list
   * to be evicted from this device. Note that this function is not thread safe
   * since only the scheduler calls this function.
   *
   * @param dev_id device id of a device to access its information
   */
  void *remove_and_return_head_from_zrlist(DevID_t dev_id) {
    PArrayNode *old_head =
        this->device_mm_[dev_id]->remove_and_return_head_from_zrlist(
            this->accumulated_released_parray_bytes_,
            this->evictable_parray_bytes_[dev_id]);
    void *py_parray{nullptr};
    if (old_head != nullptr) {
      // TODO(hc): check if this case can happen.
      parray::InnerPArray *c_parray = old_head->parray;
      py_parray = c_parray->get_py_parray();
    }
    return py_parray;
  }

  // TODO(hc): This is not used.
  void clear_all_instances() {
    for (size_t i = 0; i < device_mm_.size(); ++i) {
      device_mm_[i]->clear_all_instances();
    }
  }

  size_t get_evictable_bytes(DevID_t dev_id) {
    return this->evictable_parray_bytes_[dev_id];
  }

private:
  /// Device manager managing system environment
  DeviceManager *device_manager_;
  /// A list of LRU-based eviction manager for each device
  std::vector<LRUDeviceEvictionManager *> device_mm_;
  /// Accumulation of the total bytes to allocate PArrays
  size_t accumulated_used_parray_bytes_{0};
  /// Accumulation of the total bytes released by the eviction manager 
  size_t accumulated_released_parray_bytes_{0};
  /// The total bytes of the evictable PArrays in the zr list
  std::vector<size_t> evictable_parray_bytes_{0};
};

#endif
