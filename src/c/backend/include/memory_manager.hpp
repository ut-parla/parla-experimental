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
    this->mtx_.unlock();
  }

  /// Remove the current head node from the list.
  PArrayNode *remove_head() {
    this->mtx_.lock();
    PArrayNode *old_head = this->head_;
    //std::cout << " zr list size:" << this->list_size_ << "\n";
    if (old_head != nullptr) {
      std::cout << "old head is not NULL so try to remove this\n";
      this->remove_unsafe(old_head); 
    } else {
      std::cout << "old head is NULL\n";
    }
    this->mtx_.unlock();
    return old_head;
  }

  bool remove(PArrayNode *node) {
    this->mtx_.lock();
    bool rv = this->remove_unsafe(node);
    this->mtx_.unlock();
    return rv;
  }

  /// Remove the node from the list.
  bool remove_unsafe(PArrayNode *node) {
    bool is_removed{false};
    if (node->prev == nullptr && node->next == nullptr) {
      return is_removed;
    }

    if (this->list_size_ == 1 and (node == this->head_ or node == this->tail_)) {
      is_removed = true;
      //std::cout << node->parray->id << " was emptified from list: " <<
      //  this->list_size_ << " [pre] \n";

      this->head_ = this->tail_ = node->next = node->prev = nullptr;
      //std::cout << node->parray->id << " was emptified from list: " <<
      //  this->list_size_ << "\n";
    }

    if (this->head_ == node) {
      this->head_ = node->next;
      node->next->prev = nullptr;
    } else if ( this->tail_ == node) {
      this->tail_ = node->prev;
      node->prev->next = nullptr;
    } else {
      // TODO(hc):check it again
      if (node->prev != nullptr) {
        node->prev->next = node->next;
      }
      if (node->next != nullptr) {
        node->next->prev = node->prev;
      }
    }
    is_removed = true;

    node->prev = node->next = nullptr;
    if (is_removed) {
      this->list_size_ -= 1;
      //std::cout << node->parray->id << " was removed from list: " <<
      //  this->list_size_ << "\n";
    }
    return is_removed;
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

  void acquire_data(parray::InnerPArray *parray) {
    this->mtx_.lock();
    //std::cout << "acquire data: " << &this->zr_parray_list_ << "\n";
    //std::cout << "Parray:" << parray->id << "," <<
    //  " size:" << parray->get_size() << " was acquired\n";
    uint64_t parray_id = parray->id;
    auto found = this->parray_reference_counts_.find(parray_id);
    //std::cout << "Parray:" << parray_id << "," <<
    //  " was found\n";
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
      //std::cout << "Parray:" << parray_id << "," <<
      //  " increase ! \n";
      found->second.ref_count++; 
      //std::cout << "Parray:" << parray_id << "," <<
      //  " increase to " << found->second.ref_count << "! \n";
      this->zr_parray_list_.remove(found->second.parray_node_ptr);
      //std::cout << "Parray:" << parray->id << "," <<
      //  " size:" << parray->get_size() << " was referenced, "
      //  << " reference count: " << found->second.ref_count << 
      //  ", " << &this->zr_parray_list_ << "\n";
    }
    this->mtx_.unlock();
  }

  void release_data(parray::InnerPArray *parray) {
    this->mtx_.lock();
    //std::cout << "release data: " << &this->zr_parray_list_ << "\n";
    uint64_t parray_id = parray->id;
    auto found = this->parray_reference_counts_.find(parray_id);
    if (found == this->parray_reference_counts_.end()) {
      std::cout << "This should not happen\n";
    } else {
      found->second.ref_count--; 
      if (found->second.ref_count == 0) {
        this->zr_parray_list_.append(found->second.parray_node_ptr);
      }
      std::cout << "Parray:" << parray->id << "," <<
        " size:" << parray->get_size() << " was released, "
        << " reference count:" << found->second.ref_count << 
        ", " << &this->zr_parray_list_ << " \n";
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

  PArrayNode *remove_and_return_head_from_zrlist() {
    //std::cout << "remove and return head:" << &this->zr_parray_list_ << "\n";
    //std::cout << "call remove head:" << &this->zr_parray_list_ << " \n";
    return this->zr_parray_list_.remove_head();
  }

  /*
  void evict_data(parray:::InnerPArray *target_parray) {}
  void run_eviction() {}
  */

private:
  DevID_t dev_id_;
  std::mutex mtx_;
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
    //std::cout << "LRUDeviceMemoryManager was resized:" << this->device_mm_.size() << "\n";
    //std::cout << "vector addr:" << &this->device_mm_ << "\n";
    for (size_t i = 0; i < this->device_mm_.size(); ++i) {
      this->device_mm_[i] = new LRUDeviceMemoryManager(i);
    }
  }

  void acquire_data(parray::InnerPArray *parray, DevID_t dev_id) {
    //std::cout << dev_id << " starts acquiring zrlist head\n";
    //std::cout << "Parray:" << parray->id << "\n";
    this->device_mm_[dev_id]->acquire_data(parray);
  }

  void release_data(parray::InnerPArray *parray, DevID_t dev_id) {
    //std::cout << dev_id << " starts releasing zrlist head\n";
    //std::cout << "Parray:" << parray->id << "\n";
    this->device_mm_[dev_id]->release_data(parray);
  }

  size_t size(DevID_t dev_id) {
    return this->device_mm_[dev_id]->size();
  }

  void *remove_and_return_head_from_zrlist(DevID_t dev_id) {
    //std::cout << dev_id << " starts removing and returning zrlist head\n";
    //std::cout << " device mm size:" <<
    //  this->device_mm_.size() << "\n" << std::flush;
    PArrayNode *old_head =
        this->device_mm_[dev_id]->remove_and_return_head_from_zrlist();
    void *py_parray{nullptr};
    if (old_head != nullptr) {
      parray::InnerPArray *c_parray = old_head->parray;
      py_parray = c_parray->get_py_parray();
      std::cout << "Return parray:" << c_parray->id << "\n";
    } else {
      std::cout << "Return parray is NULL on C\n";
    }
    return py_parray;
  }

private:
  DeviceManager *device_manager_;
  std::vector<LRUDeviceMemoryManager *> device_mm_;
};

#endif
