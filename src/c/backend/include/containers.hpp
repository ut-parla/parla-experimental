#pragma once
#ifndef PARLA_CONTAINERS_HPP

#include <iostream>

#include <vector>
#include <list>
#include <map>
#include <deque>

#include <queue>
#include <stack>

#include <string>
#include <mutex>
#include <atomic>
#include <condition_variable>

#include <chrono>
#include <assert.h>

/* Header only implementations of thread-safe data structures and helpers */
/* This will probably grow and change signifigantly as development continues and needs & bottlenecks change.*/

// By default, types are not atomic,
template<typename T> auto constexpr is_atomic = false;
// but std::atomic<T> types are,
template<typename T> auto constexpr is_atomic<std::atomic<T>> = true;
// as well as std::atomic_flag.
template<> auto constexpr is_atomic<std::atomic_flag> = true;

template<typename T>
class ProtectedVector {

    private:
        std::vector<T> vec = std::vector<T>();
        std::atomic<int> length;
        std::mutex mtx;
        std::string name;

    public:

        ProtectedVector() = default;

        ProtectedVector(std::string name){
            this->mtx.lock();
            this->name = name;
            this->mtx.unlock();
        }

        ProtectedVector(std::string name, std::vector<T> vec){
            this->mtx.lock();
            this->name = name;
            this->vec = vec;
            this->mtx.unlock();
        }

        ProtectedVector(std::string name, size_t size){
            this->mtx.lock();
            this->name = name;
            this->vec.reserve(size);
            this->mtx.unlock();
        }

        void lock(){
            this->mtx.lock();
        }

        void unlock(){
            this->mtx.unlock();
        }

        void push_back(T a){
            this->mtx.lock();
            this->vec.push_back(a);
            this->mtx.unlock();
            this->length++;
        }

        void push_back_unsafe(T a){
            this->vec.push_back(a);
            this->length++;
        }

        void pop_back(){
            this->mtx.lock();
            this->vec.pop_back();
            this->mtx.unlock();
            this->length--;
        }

        void pop_back_unsafe(){
            this->vec.pop_back();
            this->length--;
        }

        int atomic_size(){
            return this->length.load();
        }

        size_t size(){
            this->mtx.lock();
            int size = this->vec.size();
            this->mtx.unlock();
            return size;
        }

        size_t size_unsafe(){
            return this->vec.size();
        }

        T& operator[](size_t i){
            this->mtx.lock();
            auto val = this->vec[i];
            this->mtx.unlock();
            return val;
        }

        T& at(size_t i){
            this->mtx.lock();
            T& val = this->vec.at(i);
            this->mtx.unlock();
            return val;
        }

        T& at_unsafe(size_t i){
            return this->vec.at(i);
        }

        T& back(){
            this->mtx.lock();
            T& val = this->vec.back();
            this->mtx.unlock();
            return val;
        }

        T& back_unsafe(){
            return this->vec.back();
        }

        T& back_and_pop(){
            this->mtx.lock();
            T& val = this->back_unsafe();
            this->pop_back_unsafe();
            this->mtx.unlock();
            return val;
        }

        T& back_and_pop_unsafe(){
            T& val = this->back_unsafe();
            this->pop_back_unsafe();
            return val;
        }

        T& front(){
            this->mtx.lock();
            T& val = this->vec.front();
            this->mtx.unlock();
            return val;
        }

        T& front_unsafe(){
            return this->vec.front();
        }

        T& front_and_pop(){
            this->mtx.lock();
            T& val = this->front_unsafe();
            this->pop_front_unsafe();
            this->mtx.unlock();
            return val;
        }

        T& front_and_pop_unsafe(){
            T& val = this->front_unsafe();
            this->pop_front_unsafe();
            return val;
        }

        void clear(){
            this->mtx.lock();
            this->vec.clear();
            this->mtx.unlock();
            this->length = 0;
        }

        void clear_unsafe(){
            this->vec.clear();
            this->length = 0;
        }

        void reserve(size_t size){
            this->mtx.lock();
            this->vec.reserve(size);
            this->mtx.unlock();
        }

        void reserve_unsafe(size_t size){
            this->vec.reserve(size);
        }

        void resize(size_t size){
            this->mtx.lock();
            this->vec.resize(size);
            this->mtx.unlock();
        }

        void resize_unsafe(size_t size){
            this->vec.resize(size);
        }

        T& get(size_t i){
            this->mtx.lock();
            T& val = this->vec[i];
            this->mtx.unlock();
            return val;
        }

        T& get_unsafe(size_t i){
            return this->vec[i];
        }

        void set(size_t i, T val){
            this->mtx.lock();
            this->vec[i] = val;
            this->mtx.unlock();
        }

        void set_unsafe(size_t i, T val){
            this->vec[i] = val;
        }

        std::vector<T> get_vector_copy(){
            this->mtx.lock();
            std::vector<T> vec = this->vec;
            this->mtx.unlock();
            return vec;
        }

        std::vector<T> get_vector_copy_unsafe(){
            std::vector<T> vec = this->vec;
            return vec;
        }

        std::vector<T>& get_vector(){
            this->mtx.lock();
            std::vector<T>& vec = this->vec;
            this->mtx.unlock();
            return vec;
        }

        std::vector<T>& get_vector_unsafe(){
            this->mtx.lock();
            std::vector<T>& vec = this->vec;
            this->mtx.unlock();
            return vec;
        }

};

template<typename T>
class ProtectedQueue {

    private:
        std::deque<T> q = std::deque<T>();
        std::atomic<int> length;
        std::mutex mtx;
        std::string name;

    public:

        ProtectedQueue() = default;

        ProtectedQueue(std::string name){
            this->mtx.lock();
            this->name = name;
            this->mtx.unlock();
        }

        ProtectedQueue(std::string name, std::deque<T> q){
            this->mtx.lock();
            this->name = name;
            this->q = q;
            this->mtx.unlock();
        }

        ProtectedQueue(std::string name, size_t size){
            this->mtx.lock();
            this->name = name;
            this->q.reserve(size);
            this->mtx.unlock();
        }

        void lock(){
            this->mtx.lock();
        }

        void unlock(){
            this->mtx.unlock();
        }

        void push_back(T a){
            this->mtx.lock();
            this->q.push_back(a);
            this->mtx.unlock();
            this->length++;
        }

        void push_back_unsafe(T a){
            this->q.push_back(a);
            this->length++;
        }

        void push_back(std::vector<T>& a){
            this->mtx.lock();
            for(auto& val : a){
                this->push_back_unsafe(val);
            }
            this->mtx.unlock();
        }

        void push_back_unsafe(std::vector<T>& a){
            for(auto& val : a){
                this->push_back_unsafe(val);
            }
        }

        void push_front(T a){
            this->mtx.lock();
            this->q.push_back(a);
            this->mtx.unlock();
            this->length++;
        }

        void push_front_unsafe(T a){
            this->q.push_back(a);
            this->length++;
        }

        void push_front(std::vector<T>& a){
            this->mtx.lock();
            for(auto& val : a){
                this->push_back_unsafe(val);
            }
            this->mtx.unlock();
        }

        void push_front_unsafe(std::vector<T>& a){
            for(auto& val : a){
                this->push_back_unsafe(val);
            }
        }

        void pop_back(){
            this->mtx.lock();
            this->q.pop_back();
            this->mtx.unlock();
            this->length--;
        }

        void pop_back_unsafe(){
            this->q.pop_back();
            this->length--;
        }

        void pop_front(){
            this->mtx.lock();
            this->q.pop_front();
            this->mtx.unlock();
            this->length--;
        }

        void pop_front_unsafe(){
            this->q.pop_front();
            this->length--;
        }

        size_t atomic_size(){
            return this->length.load();
        }

        size_t size(){
            this->mtx.lock();
            int size = this->q.size();
            this->mtx.unlock();
            return size;
        }

        size_t size_unsafe(){
            return this->q.size();
        }

        T& operator[](size_t i){
            this->mtx.lock();
            auto val = this->q[i];
            this->mtx.unlock();
            return val;
        }

        T& at(size_t i){
            this->mtx.lock();
            T& val = this->q.at(i);
            this->mtx.unlock();
            return val;
        }

        T& at_unsafe(size_t i){
            return this->q.at(i);
        }

        T& back(){
            this->mtx.lock();
            T& val = this->q.back();
            this->mtx.unlock();
            return val;
        }

        T& back_unsafe(){
            return this->q.back();
        }

        T& back_and_pop(){
            this->mtx.lock();
            T& val = this->back_unsafe();
            this->pop_back_unsafe();
            this->mtx.unlock();
            return val;
        }

        T& back_and_pop_unsafe(){
            T& val = this->back_unsafe();
            this->pop_back_unsafe();
            return val;
        }

        T& front(){
            this->mtx.lock();
            T& val = this->q.front();
            this->mtx.unlock();
            return val;
        }

        T& front_unsafe(){
            return this->q.front();
        }

        T& front_and_pop(){
            this->mtx.lock();
            T& val = this->front_unsafe();
            this->pop_front_unsafe();
            this->mtx.unlock();
            return val;
        }

        T& front_and_pop_unsafe(){
            T& val = this->front_unsafe();
            this->pop_front_unsafe();
            return val;
        }

        void clear(){
            this->mtx.lock();
            this->q.clear();
            this->mtx.unlock();
            this->length = 0;
        }

        void clear_unsafe(){
            this->q.clear();
            this->length = 0;
        }
};


#endif //PARLA_CONTAINERS_HPP