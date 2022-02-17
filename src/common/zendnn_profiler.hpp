/*******************************************************************************
* Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef COMMON_ZENDNN_PROFILER_HPP
#define COMMON_ZENDNN_PROFILER_HPP

#include <ctime>
#include <ratio>
#include <chrono>
#include <cstdint>

#define  CLK_FREQ   (2800000000)         /* Hz */

namespace zendnn {
namespace impl {
namespace profiler {

// a high resolution clock based on rdtsc
struct rdtsc_clk_t {
    using rep        = uint64_t;
    using period     = std::ratio<1,CLK_FREQ>;
    using duration   = std::chrono::duration<rep, period>;
    using time_point = std::chrono::time_point<rdtsc_clk_t>;

    static time_point start() {

        uint32_t hi, lo;
        asm volatile("CPUID \n\t"
                     "RDTSC \n\t"
                     "mov %%edx, %0 \n\t"
                     "mov %%eax, %1 \n\t"
                     :"=r"(hi), "=r"(lo)
                     :
                     :"%rax", "%rbx", "%rcx", "%rdx");

        return time_point(duration(static_cast<rep>(hi) << 32 | lo));
    }

    static time_point stop() {

        uint32_t hi, lo;
        asm volatile("RDTSCP \n\t"
                     "mov %%edx, %0 \n\t"
                     "mov %%eax, %1 \n\t"
                     "CPUID \n\t"
                     :"=r"(hi), "=r"(lo)
                     :
                     :"%rax", "%rbx", "%rcx", "%rdx");

        return time_point(duration(static_cast<rep>(hi) << 32 | lo));
    }

};

struct high_res_clk_t {
    using rep        = std::chrono::high_resolution_clock::rep;
    using period     = std::chrono::high_resolution_clock::period;
    using duration   = std::chrono::high_resolution_clock::duration;
    using time_point = std::chrono::high_resolution_clock::time_point;

    static time_point start() {
        return std::chrono::high_resolution_clock::now();
    }

    static time_point stop() {
        return std::chrono::high_resolution_clock::now();
    }

};

template<typename clk>
struct cpu_timer_t {
public:
    using    time_point = typename clk::time_point;
    using    duration   = typename clk::duration;

    using    ms_t       = std::chrono::milliseconds;
    using    us_t       = std::chrono::microseconds;
    using    ns_t       = std::chrono::nanoseconds;

    double       start_timer();
    double       stop_timer();
    uint64_t     elapsed_counts();
    double       elapsed_time_ms();
    double       elapsed_time_us();
    double       elapsed_time_ns();

private:
    time_point  start_time;
    time_point  stop_time;
    duration    elapsed_count;
};

template<typename clk>
double cpu_timer_t<clk>::start_timer() {
    start_time = clk::start();
    return start_time.time_since_epoch().count();
}

template<typename clk>
double cpu_timer_t<clk>::stop_timer() {
    stop_time = clk::stop();
    elapsed_count = static_cast<duration>(stop_time - start_time);

    return stop_time.time_since_epoch().count();
}

template<typename clk>
uint64_t cpu_timer_t<clk>::elapsed_counts() {
    return static_cast<uint64_t>(elapsed_count.count());
}

template<typename clk>
double cpu_timer_t<clk>::elapsed_time_ms() {
    using duration_milli = std::chrono::duration<double, std::milli>;
    duration_milli  duration_ms
        = std::chrono::duration_cast<duration_milli>(elapsed_count);
    return static_cast<double>(duration_ms.count());
}

template<typename clk>
double cpu_timer_t<clk>::elapsed_time_us() {
    using duration_micro = std::chrono::duration<double, std::micro>;
    duration_micro  duration_us
        = std::chrono::duration_cast<duration_micro>(elapsed_count);
    return static_cast<double>(duration_us.count());
}

template<typename clk>
double cpu_timer_t<clk>::elapsed_time_ns() {
    using duration_nano = std::chrono::duration<double, std::nano>;
    duration_nano  duration_ns
        = std::chrono::duration_cast<duration_nano>(elapsed_count);
    return static_cast<double>(duration_ns.count());
}

// typedefs
using  rdtsc_cpu_timer_t = cpu_timer_t<rdtsc_clk_t>;
using  hires_cpu_timer_t = cpu_timer_t<high_res_clk_t>;

} //namespace profiler
} //namespace impl
} //namespace zendnn

#endif
