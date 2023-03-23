#ifndef PARLA_EVENTS_HPP
#define PARLA_EVENTS_HPP

#include <cuda_runtime.h>
#include <stdexcept>

class Event {
public:
  void *event = nullptr;

  Event() = default;

  void *get_event() { return event; }
  void set_event(void *event) { this->event = event; }
  void synchronize() { throw std::runtime_error("Not implemented"); }
  void wait(void *stream) { throw std::runtime_error("Not implemented"); }
};

class CUDAEvent : public Event {

public:
  CUDAEvent() = default;

  void synchronize() {
    cudaEvent_t l_event = dynamic_cast<cudaEvent_t>(event);
    cudaEventSynchronize(l_event);
  }

  void wait(void *stream) {
    // The 0 is for the flags.
    // 0 means that the event will be waited on in the default manner.
    // 1 has to do with CUDA graphs.

    cudaStream_t l_stream = dynamic_cast<cudaStream_t>(stream);
    cudaEvent_t l_event = dynamic_cast<cudaEvent_t>(event);
    cudaStreamWaitEvent(l_stream, l_event, 0);
  }
};

/*
class HIPEvent : public Event {

public:
  hipEvent_t event;

  HIPEvent() = default;

  void set_event(hipEvent_t event) { this->event = event; }

  hipEvent_t get_event() { return event; }

  void synchronize() { hipEventSynchronize(event); }

  void wait(hipStream_t stream) {
    // The 0 is for the flags.
    // 0 means that the event will be waited on in the default manner.
    // 1 is not supported by HIP (interface provided for CUDA compatibility)
    hipStreamWaitEvent(stream, event, 0);
  }
};
*/

class Stream {
public:
  void *stream = nullptr;

  Stream() = default;

  void *get_stream() { return stream; }
  void set_stream(void *stream) { this->stream = stream; }

  void synchronize() { throw std::runtime_error("Not implemented"); }
  void wait(void *event) { throw std::runtime_error("Not implemented"); }
};

class CUDAStream : public Stream {

public:
  CUDAStream() = default;

  void synchronize() {
    cudaStream_t l_stream = dynamic_cast<cudaStream_t>(stream);
    cudaStreamSynchronize(l_stream);
  }

  void wait(void *event) {
    // The 0 is for the flags.
    // 0 means that the event will be waited on in the default manner.
    // 1 has to do with CUDA graphs.
    cudaStream_t l_stream = dynamic_cast<cudaStream_t>(stream);
    cudaEvent_t l_event = dynamic_cast<cudaEvent_t>(event);
    cudaStreamWaitEvent(stream, event, 0);
  }
};

/*
class HIPStream : public Stream {

public:
  hipStream_t stream;

  HIPStream() = default;

  void set_stream(hipStream_t stream) { this->stream = stream; }

  hipStream_t get_stream() { return stream; }

  void synchronize() { hipStreamSynchronize(stream); }

  void wait(hipEvent_t event) {
    // The 0 is for the flags.
    // 0 means that the event will be waited on in the default manner.
    // 1 has to do with CUDA graphs.
    hipStreamWaitEvent(stream, event, 0);
  }
};
*/

#endif // PARLA_EVENTS_HPP