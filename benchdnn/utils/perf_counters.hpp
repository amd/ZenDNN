/*******************************************************************************
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#ifndef BENCHDNN_PERF_COUNTERS_HPP
#define BENCHDNN_PERF_COUNTERS_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace zendnnl {
namespace benchdnn {

struct PerfCounterResult {
    std::string name;
    uint64_t raw_event;
    int64_t value;
};

struct PerfCounterDerived {
    double l1_miss_pct;
    double l2_miss_pct;
    double l3_miss_pct;
    double pf_l2_pct;
    double pf_l3_pct;
    double pf_dram_pct;
    double l2_bw_meas_gbs;
    double l2_bw_meas_pct;
};

class PerfCounterGroup {
public:
    PerfCounterGroup();
    ~PerfCounterGroup();
    PerfCounterGroup(const PerfCounterGroup &) = delete;
    PerfCounterGroup &operator=(const PerfCounterGroup &) = delete;

    bool open();
    void close();
    void reset();
    void enable();
    void disable();
    bool read();

    bool is_available() const { return available_; }

    const std::vector<PerfCounterResult> &results() const { return results_; }

    PerfCounterDerived derive(double elapsed_sec, int num_threads = 1) const;

    static void print_header(bool tab_separated = false);
    void print_values(const PerfCounterDerived &d, bool tab_separated = false) const;
    void print_raw_counters() const;

private:
    struct Counter {
        std::string name;
        uint32_t type;
        uint64_t config;
        int fd;
    };

    std::vector<Counter> counters_;
    std::vector<PerfCounterResult> results_;
    bool available_;
    bool opened_;
};

} // namespace benchdnn
} // namespace zendnnl

#endif // BENCHDNN_PERF_COUNTERS_HPP
