#pragma once
#ifndef PARLA_PROFILING_HPP

#include<iostream>
#include<fstream>
#include<chrono>
#include<thread>

#if defined (ENABLE_NVTX)

    #include <nvtx3/nvtx3.hpp>

    struct my_domain{ static constexpr char const* name{"Parla Runtime"}; };

    using my_scoped_range = nvtx3::scoped_range_in<my_domain>;
    using my_registered_string = nvtx3::registered_string_in<my_domain>;

    struct add_dependent_msg{ static constexpr char const* message{"add_dependency"}; };
    struct add_dependency_msg{ static constexpr char const* message{"add_dependent"}; };
    struct notify_dependents_msg{ static constexpr char const* message{"notify_dependents"}; };
    struct run_launcher_msg{ static constexpr char const* message{"run_launcher"}; };

#endif // ENABLE_NVTX


#ifdef ENABLE_LOGGING
    #include <binlog/binlog.hpp>

    #define LOG(args...) BINLOG_INFO(args)

    int log_write(std::string filename) {
        std::ofstream logfile(filename.c_str(),
                                std::ofstream::out | std::ofstream::binary);
        binlog::consume(logfile);

        if (!logfile) {
            std::cerr << "Failed to write logfile!\n";
            return 1;
        }

        std::cout << "Log file written to " << filename << std::endl;
        return 0;
    }

#else
    #define LOG(args...)

    int log_write(std::string filename) { return 0; }

#endif // ENABLE_LOGGING





#endif // PARLA_PROFILING_HPP