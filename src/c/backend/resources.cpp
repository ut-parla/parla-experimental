#include "include/containers.hpp"
#include "include/runtime.hpp"

//ResourcePool Implementation

//TODO(will): This is a temporary mock implementation.
//            It hardcodes vcus and only supports vcus.

//I don't think other would EVER need to be atomic I'm really just doing this for convienence in the mockup.


template<typename T>
void InnerResourcePool<T>::set(std::string, T value){
        this->vcus.store(value);
}

template<typename T>
T InnerResourcePool<T>::get(std::string){
        return this->vcus.load();
}

template<typename T>
template<typename J>
bool InnerResourcePool<T>::check_greater(InnerResourcePool<J>& other){
    /*
    if constexpr (is_atomic<J> && is_atomic<T>){
        return this->vcus.load() >= other.vcus.load();
    }
    else if constexpr (!is_atomic<J> && is_atomic<T>) {
        return this->vcus >= other.vcus.load();
    }
    else if constexpr (is_atomic<J> && !is_atomic<T>) {
        return this->vcus >= other.vcus.load();
    }
    else{
        return this->vcus >= other.vcus;
    }*/
    return this->vcus.load() >= other.vcus.load();
}

template<typename T>
template<typename J>
bool InnerResourcePool<T>::check_lesser(InnerResourcePool<J>& other){
    /*
    if constexpr (is_atomic<J> && is_atomic<T>){
        return this->vcus.load() <= other.vcus.load();
    }
    else if constexpr (!is_atomic<J> && is_atomic<T>) {
        return this->vcus ><= other.vcus.load();
    }
    else if constexpr (is_atomic<J> && !is_atomic<T>) {
        return this->vcus <= other.vcus.load();
    }
    else{
        return this->vcus ><= other.vcus;
    }*/
    return this->vcus.load() <= other.vcus.load();
}

template<typename T>
template<typename J>
T InnerResourcePool<T>::increase(InnerResourcePool<J>& other){
    /*
    if constexpr (is_atomic<J> && is_atomic<T>){
        return this->vcus.fetch_add(other.vcus.load());
    }
    else if constexpr (!is_atomic<J> && is_atomic<T>) {
        return this->vcus.fetch_add(other.vcus);
    }
    else if constexpr (is_atomic<J> && !is_atomic<T>) {
        this->vcus += other.vcus.load();
        return this->vcus;
    }
    else{
        this->vcus += other.vcus;
        return this->vcus;
    }*/
    return this->vcus.fetch_add(other.vcus.load());
}

template<typename T>
template<typename J>
T InnerResourcePool<T>::decrease(InnerResourcePool<J>& other){
    /*
    if constexpr (is_atomic<J> && is_atomic<T>){
        return this->vcus.fetch_sub(other.vcus.load());
    }
    else if constexpr (!is_atomic<J> && is_atomic<T>) {
        return this->vcus.fetch_sub(other.vcus);
    }
    else if constexpr (is_atomic<J> && !is_atomic<T>) {
        this->vcus -= other.vcus.load();
        return this->vcus;
    }
    else{
        this->vcus -= other.vcus;
        return this->vcus;
    }*/
    return this->vcus.fetch_sub(other.vcus.load());
}

//template class InnerResourcePool<int>;

template class InnerResourcePool<float>;
template float InnerResourcePool<float>::decrease(InnerResourcePool<float>&);
template float InnerResourcePool<float>::increase(InnerResourcePool<float>&);
template bool InnerResourcePool<float>::check_lesser(InnerResourcePool<float>&);
template bool InnerResourcePool<float>::check_greater(InnerResourcePool<float>&);