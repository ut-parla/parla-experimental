#pragma once
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
#include <binlog/binlog.hpp>

#define LOG_TRACE(args...) BINLOG_TRACE(args)
#define LOG_DEBUG(args...) BINLOG_DEBUG(args)
#define LOG_INFO(args...) BINLOG_INFO(args)
#define LOG_WARN(args...) BINLOG_WARNING(args)
#define LOG_ERROR(args...) BINLOG_ERROR(args)
#define LOG_FATAL(args...) BINLOG_CRITICAL(args)

#define LOG_ADAPT_STRUCT(args...) BINLOG_ADAPT_STRUCT(args)
#define LOG_ADAPT_DERIVED(args...) BINLOG_ADAPT_DERIVED(args)

inline int write_log(std::string filename){
  std::ofstream logfile(filename.c_str(),
                        std::ofstream::out | std::ofstream::binary);
  binlog::consume(logfile);

  if (!logfile) {
    std::cerr << "Failed to write logfile!\n";
    return 1;
  }

  std::cout << "Log file written to: " << filename << std::endl;
  return 0;
}
#endif

#endif // PARLA_PROFILING_HPP