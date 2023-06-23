#ifndef PARLA_MEMORY_MANNGER_HPP
#define PARLA_MEMORY_MANNGER_HPP

#include "device_manager.hpp"
#include "parray.hpp"

enum PArrayInstanceState {
  // A PArray is being prefetched (moved).
  PREFETCHING = 0,
  // A PArray's reference count is more than 1,
  // but it is not acquired yet.
  RESERVED = 1,
  // A task acuires and is using a PArray.
  ACQUIRED = 2,
  // None of mapped/running tasks is using this PArray.
  FREE = 3
};


class PArrayNode {
public:
  PArrayNode(parray::InnerPArray *parr, DevID_t dev, size_t prior = 0) :
      parray(parr), device(dev), priority(prior), next(nullptr), prev(nullptr)
  {}

  parray::InnerPArray *parray;
  DevID_t device;
  size_t priority;
  PArrayNode *next;
  PArrayNode *prev;
};


class DoubleLinkedList {
public:
  void print() {
    PArrayNode *node = this->head_;
    std::cout << "\n";
    while (node != nullptr) {
      std::cout << node->parray->id << " -> \n"; 
      node = node->next;
    }

    if (this->tail_ != nullptr) {
      std::cout << "Final tail:" << this->tail_->parray->id << "\n\n";
    }
    std::cout << "\n";
  }

  /// Append a node to the tail.
  void append(PArrayNode *node) {
    this->mtx_.lock();
    if (this->list_size_ == 0) {
      //std::cout << node->parray->id << " is set as head\n";
      this->head_ = node;
      this->tail_ = node;
    } else {
      //std::cout << node->parray->id << " is attached to tail\n";
      this->tail_->next = node;
      node->prev = this->tail_;
      this->tail_ = node;
    }
    //std::cout << "Append :" << node->parray->id << "\n";
    //this->print();
    this->list_size_ += 1;
    this->mtx_.unlock();
  }

  /// Insert the new_node to the next to the node.
  /// node -> [new node] -> ..
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

  /// Insert the new_node to the before the node.
  /// .. -> [new node] -> node
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
    //this->print();

    this->mtx_.unlock();
  }

  /// Remove the current head node from the list.
  PArrayNode *remove_head() {
    this->mtx_.lock();
    PArrayNode *old_head = this->head_;
    //std::cout << " zr list size:" << this->list_size_ << "\n";
    if (old_head != nullptr) {
      this->remove_unsafe(old_head); 
    }
    //this->print();

    this->mtx_.unlock();
    return old_head;
  }

  bool remove(PArrayNode *node) {
    this->mtx_.lock();
    bool rv = this->remove_unsafe(node);
    //this->print();
    this->mtx_.unlock();
    return rv;
  }

  /// Remove the node from the list.
  bool remove_unsafe(PArrayNode *node) {
    if (node->prev == nullptr && node->next == nullptr &&
        node != this->head_ && node != this->tail_) {
      return false;
    }

    if (this->list_size_ == 1) {
      if (node == this->head_ || node == this->tail_) {
        this->head_ = this->tail_ = nullptr;
      }
    } else {
      if (this->head_ == node) {
        this->head_ = node->next;
        node->next->prev = nullptr;
      } else if (this->tail_ == node) {
        this->tail_ = node->prev;
        node->prev->next = nullptr;
      } else {
        // TODO(hc):check it again
        node->prev->next = node->next;
        node->next->prev = node->prev;
      }
    }
    node->prev = node->next = nullptr;
    this->list_size_ -= 1;
    //std::cout << node->parray->id << " was removed from list: " <<
    //  this->list_size_ << "\n";
    return true;
  }

  size_t size() {
    this->mtx_.lock();
    size_t list_size = this->list_size_;
    this->mtx_.unlock();
    return list_size;
  }

  PArrayNode *get_head() {
    return this->head_;
  }

  PArrayNode *get_tail() {
    return this->tail_;
  }

private:
  PArrayNode *head_{nullptr};
  PArrayNode *tail_{nullptr};

  std::mutex mtx_;
  size_t list_size_{0};
};


class LRUDeviceMemoryManager {
public:
  struct PArrayMetaInfo {
    // Points to a PArray node if it exists
    PArrayNode *parray_node_ptr;
    // The number of references to a PArray
    size_t ref_count;
  };

  LRUDeviceMemoryManager(DevID_t dev_id) : dev_id_(dev_id) {}

  /**
   * @brief This function is called when a task acquires
   * a PArray. It increases a reference count of the PArray
   * and removes it from a zero-referenced list if it exists.
   */
  void acquire_data(parray::InnerPArray *parray) {
    this->mtx_.lock();
    uint64_t parray_id = parray->id;
    auto found = this->parray_reference_counts_.find(parray_id);
    if (found == this->parray_reference_counts_.end()) {
      //std::cout << "Parray:" << parray_id << "," <<
      //  " was not found\n";
      PArrayNode *parray_node = new PArrayNode(parray, this->dev_id_);
      this->parray_reference_counts_[parray_id] =
          PArrayMetaInfo{parray_node, 1};
      //std::cout << "Parray:" << parray->id << "," <<
      //  " size:" << parray->get_size() << " was ceated, "
      //  << " reference count: 1 \n";
    } else {
      found->second.ref_count++; 
      //std::cout << "Parray:" << parray->id << "," <<
      //  " size:" << parray->get_size() << " was referenced, "
      //  << " reference count: " << found->second.ref_count << 
      //  ", " << &this->zr_parray_list_ << "\n";
      this->zr_parray_list_.remove(found->second.parray_node_ptr);
    }
    this->mtx_.unlock();
  }

  /**
   * @brief This function is called when a task releases 
   * a PArray. It decreases a reference count of the PArray
   * and adds it to a zero-referenced list if its reference 
   * count became 0.
   */
  void release_data(parray::InnerPArray *parray) {
    this->mtx_.lock();
    uint64_t parray_id = parray->id;
    auto found = this->parray_reference_counts_.find(parray_id);
    if (found != this->parray_reference_counts_.end()) {
      //std::cout << "Parray:" << parray->id << "," << " device id:" << this->dev_id_
      //  << " size:" << parray->get_size() << " was released, "
      //  << " reference count:" << found->second.ref_count << 
      //  ", " << &this->zr_parray_list_ << " \n";
      found->second.ref_count--; 
      if (found->second.ref_count == 0) {
        this->zr_parray_list_.append(found->second.parray_node_ptr);
      }
      /*
      std::cout << "Parray:" << parray->id << "," << " device id:" << this->dev_id_
        << " size:" << parray->get_size() << " was released, "
        << " reference count:" << found->second.ref_count << 
        ", " << &this->zr_parray_list_ << " [done] \n";
      */
    }
    this->mtx_.unlock();
  }

  size_t size() {
    size_t zr_parray_list_size{0};
    this->mtx_.lock();
    zr_parray_list_size = zr_parray_list_.size();
    this->mtx_.unlock();
    return zr_parray_list_size;
  }

  /**
   * @brief Remove and return a head of the zero-referenced list.
   *        This function is not thread safe since it assumes that only
   *        the memory manager calls into this function during eviction. 
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
   *        zero-referenced list. This function has two purposes.
   *        First, it is used to fix unlinked Python and C++ PArray
   *        instances. It is possible that Python PArrays are destroyed
   *        due to, for example, out-of-scope. Then, C++ PArrays
   *        start to hold invalid Python PArray pointers.
   *        When a scheduler starts PArray eviction, it is possible that
   *        the C++ PArrays holding invalid Python PArrays are chosen 
   *        as evictable PArrays and causes segmentation fault.
   *        This function removes those PArrays in advance to avoid
   *        this issue.
   *        The second purpose is to allow users to clear all memory
   *        related states managed by the Parla runtime.
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
  DevID_t dev_id_;
  std::mutex mtx_;
  /// Key: PArray ID, Value: Meta information including reference
  /// count of a PArray
  std::unordered_map<uint64_t, PArrayMetaInfo> parray_reference_counts_;
  /// A list of zero-referenced PArrays.
  DoubleLinkedList zr_parray_list_;
};


class LRUGlobalMemoryManager {
public:
  LRUGlobalMemoryManager(DeviceManager *device_manager) :
    device_manager_(device_manager) {
    this->device_mm_.resize(
        device_manager->template get_num_devices<DeviceType::All>());
    for (size_t i = 0; i < this->device_mm_.size(); ++i) {
      this->device_mm_[i] = new LRUDeviceMemoryManager(i);
    }
  }

  void acquire_data(parray::InnerPArray *parray, DevID_t dev_id) {
    this->device_mm_[dev_id]->acquire_data(parray);
  }

  void release_data(parray::InnerPArray *parray, DevID_t dev_id) {
    this->device_mm_[dev_id]->release_data(parray);
  }

  size_t size(DevID_t dev_id) {
    return this->device_mm_[dev_id]->size();
  }

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
  DeviceManager *device_manager_;
  std::vector<LRUDeviceMemoryManager *> device_mm_;
};

#endif
