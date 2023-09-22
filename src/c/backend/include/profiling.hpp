/*! @file profiling.hpp
 *  @brief Provides macros for NVTX profiling and BINLOG tracing.
 */

// #pragma once
#ifndef PARLA_PROFILING_HPP
#define PARLA_PROFILING_HPP

#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>

#if defined(PARLA_ENABLE_NVTX)

#include <nvtx3/nvtx3.hpp>

struct my_domain {
  static constexpr char const *name{"Parla Runtime"};
};

using my_scoped_range = nvtx3::scoped_range_in<my_domain>;
using my_registered_string = nvtx3::registered_string_in<my_domain>;

/*
// Statically configure the message strings so they are not reiniailized on
// every tracing call
// This decreases the tracing overhead
struct add_dependent_msg{ static constexpr char const*
message{"add_dependency"}; }; struct add_dependency_msg{ static constexpr char
const* message{"add_dependent"}; }; struct notify_dependents_msg{ static
constexpr char const* message{"notify_dependents"}; }; struct run_launcher_msg{
static constexpr char const* message{"run_launcher"}; };
*/

#define NVTX_COLOR(c1, c2, c3)                                                 \
  nvtx3::rgb { c1, c2, c3 }
#define NVTX_RANGE(name, color) my_scoped_range r(name, color);

#else

#define NVTX_COLOR(c1, c2, c3)

#define NVTX_RANGE(name, color)

#endif // PARLA_ENABLE_NVTX

#define NVTX_COLOR_RED NVTX_COLOR(255, 0, 0)
#define NVTX_COLOR_GREEN NVTX_COLOR(0, 255, 0)
#define NVTX_COLOR_BLUE NVTX_COLOR(0, 0, 255)

#define NVTX_COLOR_YELLOW NVTX_COLOR(255, 255, 0)
#define NVTX_COLOR_MAGENTA NVTX_COLOR(255, 0, 255)
#define NVTX_COLOR_CYAN NVTX_COLOR(0, 255, 255)
#define NVTX_COLOR_LIGHT_GREEN NVTX_COLOR(127, 255, 0)

#define NVTX_COLOR_ORANGE NVTX_COLOR(255, 127, 0)
#define NVTX_COLOR_PURPLE NVTX_COLOR(127, 0, 255)
#define NVTX_COLOR_TEAL NVTX_COLOR(0, 168, 255)

#define NVTX_COLOR_WHITE NVTX_COLOR(255, 255, 255)
#define NVTX_COLOR_BLACK NVTX_COLOR(0, 0, 0)

#define NVTX_COLOR_GRAY NVTX_COLOR(127, 127, 127)

#ifdef PARLA_ENABLE_LOGGING
#include <binlog/Session.hpp>
#include <binlog/SessionWriter.hpp>
#include <binlog/advanced_log_macros.hpp>
#include <binlog/binlog.hpp>

namespace binlog {

extern int global_reset_count;

inline Session &parla_session() {
  static Session parla_session;
  static unsigned int reset_count;
  // if (reset_count != global_reset_count) {
  //   parla_session = Session();
  //   reset_count = binlog::global_reset_count;
  // }
  return parla_session;
}

inline SessionWriter &parla_writer() {
  static thread_local SessionWriter s_writer(parla_session(), 1 << 20, 0,
                                             detail::this_thread_id_string());

  static thread_local unsigned int reset_count;

  if (reset_count != binlog::global_reset_count) {
    /*
    std::cout << "Resetting binlog session writer" << std::endl;
    s_writer = SessionWriter(parla_session(), 1 << 20, 0,
                             detail::this_thread_id_string());
    reset_count = binlog::global_reset_count;
    */
  }
  return s_writer;
}

} // namespace binlog

#define LOG_TRACE(args...) BINLOG_TRACE_WC(binlog::parla_writer(), args)
#define LOG_DEBUG(args...) BINLOG_DEBUG_WC(binlog::parla_writer(), args)
#define LOG_INFO(args...) BINLOG_INFO_WC(binlog::parla_writer(), args)
#define LOG_WARN(args...) BINLOG_WARN_WC(binlog::parla_writer(), args)
#define LOG_ERROR(args...) BINLOG_ERROR_WC(binlog::parla_writer(), args)
#define LOG_FATAL(args...) BINLOG_CRITICAL_WC(binlog::parla_writer(), args)

#define LOG_ADAPT_STRUCT(args...) BINLOG_ADAPT_STRUCT(args)
#define LOG_ADAPT_DERIVED(args...) BINLOG_ADAPT_DERIVED(args)
#define LOG_ADAPT_ENUM(args...) BINLOG_ADAPT_ENUM(args)
#define LOG_ADAPT_DERIVED(args...) BINLOG_ADAPT_DERIVED(args)

inline int initialize_log(std::string filename) {
  std::ofstream logfile(filename.c_str(),
                        std::ofstream::out | std::ofstream::binary);
  binlog::global_reset_count++;
  binlog::parla_session().reconsumeMetadata(logfile);

  // std::cout << "Metadata written to: " << filename
  //           << binlog::detail::this_thread_id_string() << std::endl;
  logfile.close();
  return 0;
}

inline int write_log(std::string filename) {
  std::ofstream logfile(filename.c_str(),
                        std::ofstream::app | std::ofstream::binary);

  binlog::parla_session().consume(logfile);
  logfile.close();

  if (!logfile) {
    std::cerr << "Failed to write logfile!\n";
    return 1;
  }

  std::cout << "Log file written to: " << filename << std::endl;
  return 0;
}

#else
#define LOG_TRACE(args...)
#define LOG_DEBUG(args...)
#define LOG_INFO(args...)
#define LOG_WARN(args...)
#define LOG_ERROR(args...)
#define LOG_FATAL(args...)

#define LOG_ADAPT_STRUCT(args...)
#define LOG_ADAPT_DERIVED(args...)
#define LOG_ADAPT_ENUM(args...)

inline int initialize_log(std::string filename) { return 0; }

inline int write_log(std::string filename) { return 0; }

#endif

#define WORKER 'Worker'
#define SCHEDULER 'Scheduler'
#define TASK 'Task'
#define RL 'RL'

inline void log_rl_msg(const int type, std::string msg) {
  // const char* _msg = msg.c_str();
  switch (type) {
  case 0:
    LOG_TRACE(RL, "{}", msg);
    break;
  case 1:
    LOG_DEBUG(RL, "{}", msg);
    break;
  case 2:
    LOG_INFO(RL, "{}", msg);
    break;
  case 3:
    LOG_WARN(RL, "{}", msg);
    break;
  case 4:
    LOG_ERROR(RL, "{}", msg);
    break;
  case 5:
    LOG_FATAL(RL, "{}", msg);
    break;
  }
}

/*!
 * @brief Python interface to log a string in the 'task' message category
 * @param type The log level.
 * @param msg The message to log.
 */
inline void log_task_msg(const int type, std::string msg) {
  // const char* _msg = msg.c_str();
  switch (type) {
  case 0:
    LOG_TRACE(TASK, "{}", msg);
    break;
  case 1:
    LOG_DEBUG(TASK, "{}", msg);
    break;
  case 2:
    LOG_INFO(TASK, "{}", msg);
    break;
  case 3:
    LOG_WARN(TASK, "{}", msg);
    break;
  case 4:
    LOG_ERROR(TASK, "{}", msg);
    break;
  case 5:
    LOG_FATAL(TASK, "{}", msg);
    break;
  }
}

/*!
 * @brief Python interface to log a string in the 'worker' message category
 * @param type The log level.
 * @param msg The message to log.
 */
inline void log_worker_msg(const int type, std::string msg) {
  // const char* _msg = msg.c_str();
  switch (type) {
  case 0:
    LOG_TRACE(WORKER, "{}", msg);
    break;
  case 1:
    LOG_DEBUG(WORKER, "{}", msg);
    break;
  case 2:
    LOG_INFO(WORKER, "{}", msg);
    break;
  case 3:
    LOG_WARN(WORKER, "{}", msg);
    break;
  case 4:
    LOG_ERROR(WORKER, "{}", msg);
    break;
  case 5:
    LOG_FATAL(WORKER, "{}", msg);
    break;
  }
}

/*!
 * @brief Python interface to log a string in the 'scheduler' message category.
 * @param type The log level.
 * @param msg The message to log.
 */
inline void log_scheduler_msg(const int type, std::string msg) {
  // const char* _msg = msg.c_str();
  switch (type) {
  case 0:
    LOG_TRACE(SCHEDULER, "{}", msg);
    break;
  case 1:
    LOG_DEBUG(SCHEDULER, "{}", msg);
    break;
  case 2:
    LOG_INFO(SCHEDULER, "{}", msg);
    break;
  case 3:
    LOG_WARN(SCHEDULER, "{}", msg);
    break;
  case 4:
    LOG_ERROR(SCHEDULER, "{}", msg);
    break;
  case 5:
    LOG_FATAL(SCHEDULER, "{}", msg);
    break;
  }
}

/*!
 * @brief Python interface to log class instance and message in the 'task'
 * message category.
 * @details The class instance must be configured to be printable with binlog.
 * @tparam T Class type
 * @param type Log level
 * @param msg String message to log
 * @param class_ptr Pointer to class instance
 */
template <typename T>
inline void log_task_1(const int type, std::string msg, T *class_ptr) {
  // const char* _msg = msg.c_str();
  switch (type) {
  case 0:
    LOG_TRACE(TASK, "{} : {}", msg, class_ptr);
    break;
  case 1:
    LOG_DEBUG(TASK, "{} : {}", msg, class_ptr);
    break;
  case 2:
    LOG_INFO(TASK, "{} : {}", msg, class_ptr);
    break;
  case 3:
    LOG_WARN(TASK, "{} : {}", msg, class_ptr);
    break;
  case 4:
    LOG_ERROR(TASK, "{} : {}", msg, class_ptr);
    break;
  case 5:
    LOG_FATAL(TASK, "{} : {}", msg, class_ptr);
    break;
  }
}

/*!
 * @brief Python interface to log class instance and message in the 'worker'
 * category
 * @details The class instance must be configured to be printable with binlog.
 * @tparam T Class type
 * @param type Log level
 * @param msg String message to log
 * @param class_ptr Pointer to class instance
 */
template <typename T>
inline void log_worker_1(const int type, std::string msg, T *class_ptr) {
  // const char* _msg = msg.c_str();
  switch (type) {
  case 0:
    LOG_TRACE(WORKER, "{} : {}", msg, class_ptr);
    break;
  case 1:
    LOG_DEBUG(WORKER, "{} : {}", msg, class_ptr);
    break;
  case 2:
    LOG_INFO(WORKER, "{} : {}", msg, class_ptr);
    break;
  case 3:
    LOG_WARN(WORKER, "{} : {}", msg, class_ptr);
    break;
  case 4:
    LOG_ERROR(WORKER, "{} : {}", msg, class_ptr);
    break;
  case 5:
    LOG_FATAL(WORKER, "{} : {}", msg, class_ptr);
    break;
  }
}

/*!
 * @brief Python interface to log class instance and message in the 'scheduler'
 * category
 * @details The class instance must be configured to be printable with binlog.
 * @tparam T Class type
 * @param type Log level
 * @param msg String message to log
 * @param class_ptr Pointer to class instance
 */
template <typename T>
inline void log_scheduler_1(const int type, std::string msg, T *class_ptr) {
  // const char* _msg = msg.c_str();
  switch (type) {
  case 0:
    LOG_TRACE(SCHEDULER, "{} : {}", msg, class_ptr);
    break;
  case 1:
    LOG_DEBUG(SCHEDULER, "{} : {}", msg, class_ptr);
    break;
  case 2:
    LOG_INFO(SCHEDULER, "{} : {}", msg, class_ptr);
    break;
  case 3:
    LOG_WARN(SCHEDULER, "{} : {}", msg, class_ptr);
    break;
  case 4:
    LOG_ERROR(SCHEDULER, "{} : {}", msg, class_ptr);
    break;
  case 5:
    LOG_FATAL(SCHEDULER, "{} : {}", msg, class_ptr);
    break;
  }
}

/*!
 * @brief Python interface to log 2 class instances and message in the 'task'
 * category. Prints as msg1 + class1 + msg2 + class2.
 * @details The class instance must be configured to be printable with binlog.
 * @tparam T The first class type
 * @tparam G The second class type
 * @param type Log level
 * @param msg1 First string message to log
 * @param class_ptr1 Pointer to first class instance
 * @param msg2 Second string message to log
 * @param class_ptr2 Pointer to second class instance
 */
template <typename T, typename G>
inline void log_task_2(const int type, std::string msg1, T *class_ptr1,
                       std::string msg2, G *class_ptr2) {
  // const char* _msg = msg.c_str();
  switch (type) {
  case 0:
    LOG_TRACE(TASK, "{} : {} {} {}", msg1, class_ptr1, msg2, class_ptr2);
    break;
  case 1:
    LOG_DEBUG(TASK, "{} : {} {} {}", msg1, class_ptr1, msg2, class_ptr2);
    break;
  case 2:
    LOG_INFO(TASK, "{} : {} {} {}", msg1, class_ptr1, msg2, class_ptr2);
    break;
  case 3:
    LOG_WARN(TASK, "{} : {} {} {}", msg1, class_ptr1, msg2, class_ptr2);
    break;
  case 4:
    LOG_ERROR(TASK, "{} : {} {} {}", msg1, class_ptr1, msg2, class_ptr2);
    break;
  case 5:
    LOG_FATAL(TASK, "{} : {} {} {}", msg1, class_ptr1, msg2, class_ptr2);
    break;
  }
}

/*!
 * @brief Python interface to log 2 class instances and message in the 'worker'
 * category. Prints as msg1 + class1 + msg2 + class2.
 * @details The class instance must be configured to be printable with binlog.
 * @tparam T The first class type
 * @tparam G The second class type
 * @param type Log level
 * @param msg1 First string message to log
 * @param class_ptr1 Pointer to first class instance
 * @param msg2 Second string message to log
 * @param class_ptr2 Pointer to second class instance
 */
template <typename T, typename G>
inline void log_worker_2(const int type, std::string msg1, T *class_ptr1,
                         std::string msg2, G *class_ptr2) {
  // const char* _msg = msg.c_str();
  switch (type) {
  case 0:
    LOG_TRACE(WORKER, "{} : {} {} {}", msg1, class_ptr1, msg2, class_ptr2);
    break;
  case 1:
    LOG_DEBUG(WORKER, "{} : {} {} {}", msg1, class_ptr1, msg2, class_ptr2);
    break;
  case 2:
    LOG_INFO(WORKER, "{} : {} {} {}", msg1, class_ptr1, msg2, class_ptr2);
    break;
  case 3:
    LOG_WARN(WORKER, "{} : {} {} {}", msg1, class_ptr1, msg2, class_ptr2);
    break;
  case 4:
    LOG_ERROR(WORKER, "{} : {} {} {}", msg1, class_ptr1, msg2, class_ptr2);
    break;
  case 5:
    LOG_FATAL(WORKER, "{} : {} {} {}", msg1, class_ptr1, msg2, class_ptr2);
    break;
  }
}

/*!
 * @brief Python interface to log 2 class instances and message in the
 * 'scheduler' category. Prints as msg1 + class1 + msg2 + class2.
 * @details The class instance must be configured to be printable with binlog.
 * @tparam T The first class type
 * @tparam G The second class type
 * @param type Log level
 * @param msg1 First string message to log
 * @param class_ptr1 Pointer to first class instance
 * @param msg2 Second string message to log
 * @param class_ptr2 Pointer to second class instance
 */
template <typename T, typename G>
inline void log_scheduler_2(const int type, std::string msg1, T *class_ptr1,
                            std::string msg2, G *class_ptr2) {
  // const char* _msg = msg.c_str();
  switch (type) {
  case 0:
    LOG_TRACE(WORKER, "{} : {} {} {}", msg1, class_ptr1, msg2, class_ptr2);
    break;
  case 1:
    LOG_DEBUG(WORKER, "{} : {} {} {}", msg1, class_ptr1, msg2, class_ptr2);
    break;
  case 2:
    LOG_INFO(WORKER, "{} : {} {} {}", msg1, class_ptr1, msg2, class_ptr2);
    break;
  case 3:
    LOG_WARN(WORKER, "{} : {} {} {}", msg1, class_ptr1, msg2, class_ptr2);
    break;
  case 4:
    LOG_ERROR(WORKER, "{} : {} {} {}", msg1, class_ptr1, msg2, class_ptr2);
    break;
  case 5:
    LOG_FATAL(WORKER, "{} : {} {} {}", msg1, class_ptr1, msg2, class_ptr2);
    break;
  }
}

#endif // PARLA_PROFILING_HPP
