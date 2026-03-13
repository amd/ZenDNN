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

// Supported AMD Zen microarchitectures
enum class ZenArch : int {
    UNKNOWN = 0,
    ZEN4    = 4,   // Family 19h (EPYC 9004 Genoa / 97x4 Bergamo)
    ZEN5    = 5,   // Family 1Ah (EPYC 9005 Turin)
};

// Per-architecture constants for derived metric calculations
struct ArchConstants {
    ZenArch arch;
    const char *name;
    int    family;            // CPUID family (0x19 = Zen4, 0x1A = Zen5)
    int    l1d_bytes;         // L1 data cache size per core
    int    l2_bytes;          // L2 cache size per core
    int    l3_bytes_per_ccd;  // L3 size per CCD
    int    l2_bw_bytes_per_cycle;  // L2 fill bandwidth (bytes/cycle)
    double cpu_freq_ghz;      // nominal boost frequency
    int    dispatch_width;    // dispatch slots per cycle
};

// Detect the running CPU's Zen generation via CPUID
ZenArch detect_zen_arch();

// Get architecture constants for the detected (or specified) arch
const ArchConstants &get_arch_constants(ZenArch arch = ZenArch::UNKNOWN);

// Counter profiles: each collects a different set of events
// optimized for specific analysis (within the 6-PMC hardware limit).
enum class PerfProfile : int {
    CACHE  = 0,   // L1/L2 hit-miss + L2 prefetcher (current default)
    TLB    = 1,   // DTLB misses + IPC (retired insn / cycles)
    STALLS = 2,   // Backend dispatch stalls + IPC
};

PerfProfile parse_perf_profile(const char *str);
const char *perf_profile_name(PerfProfile p);

struct PerfCounterResult {
    std::string name;
    uint64_t raw_event;
    int64_t value;
};

struct PerfCounterDerived {
    PerfProfile profile;

    // ── cache profile ──
    double l1_miss_pct;
    double l2_miss_pct;
    double l3_miss_pct;
    double pf_l2_pct;
    double pf_l3_pct;
    double pf_dram_pct;
    double l2_bw_meas_gbs;
    double l2_bw_meas_pct;

    // ── tlb profile ──
    double dtlb_l2_hit_pct;   // L1 DTLB miss → L2 DTLB hit %
    double dtlb_l2_miss_pct;  // L1+L2 DTLB miss → page walk %
    double ipc;               // retired instructions / unhalted cycles

    // ── stalls profile ──
    double fp_reg_stall_pct;  // FP register file full stall cycles %
    double fp_sched_stall_pct;// FP scheduler full stall cycles %
    double lq_stall_pct;      // load queue full stall cycles %
    double retire_stall_pct;  // retire token stall cycles %
};

class PerfCounterGroup {
public:
    explicit PerfCounterGroup(PerfProfile profile = PerfProfile::CACHE);
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
    PerfProfile profile() const { return profile_; }
    ZenArch arch() const { return arch_; }

    const std::vector<PerfCounterResult> &results() const { return results_; }

    PerfCounterDerived derive(double elapsed_sec, int num_threads = 1) const;

    void print_header(bool tab_separated = false) const;
    void print_values(const PerfCounterDerived &d, bool tab_separated = false) const;
    void print_raw_counters() const;

private:
    struct Counter {
        std::string name;
        uint32_t type;
        uint64_t config;
        int fd;
    };

    PerfProfile profile_;
    ZenArch arch_;
    std::vector<Counter> counters_;
    std::vector<PerfCounterResult> results_;
    bool available_;
    bool opened_;
};

} // namespace benchdnn
} // namespace zendnnl

#endif // BENCHDNN_PERF_COUNTERS_HPP
