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

#include "perf_counters.hpp"

#ifdef __linux__
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <cstring>
#include <cerrno>
#endif

#include <cstdio>
#include <algorithm>

namespace zendnnl {
namespace benchdnn {

static constexpr double CPU_FREQ_GHZ = 4.121;
static constexpr int L2_BW_BYTES_PER_CYCLE = 64;
static constexpr double L2_PEAK_GBS = L2_BW_BYTES_PER_CYCLE * CPU_FREQ_GHZ;

#ifdef __linux__
// AMD Zen 5 PMU event definitions (verified against AMD PPR / illumos docs)
//
// PMCx064 L2CacheReqStat — "Core to L2 Cacheable Request Access Status"
//   Does NOT include L2 Prefetcher requests.
//   rF064 = umask 0xF0 = all DC hits (0x80 shared + 0x40 mod + 0x20 non-mod + 0x10 store)
//   r0864 = umask 0x08 = DC miss
//
// PMCx070-072 — L2 Prefetcher events (separate from demand)
//   rFF70 = L2PfHitL2: PF hit in L2
//   rFF71 = L2PfMissL2HitL3: PF miss L2, hit L3
//   rFF72 = L2PfMissL2L3: PF miss both L2 & L3 (DRAM)
//
// L1-dcache-loads/misses use PERF_TYPE_HW_CACHE (generic kernel events)

struct EventDef {
    const char *name;
    uint32_t type;
    uint64_t config;
};

static const EventDef AMD_ZEN5_EVENTS[] = {
    {"L1_loads",         PERF_TYPE_HW_CACHE,
        (PERF_COUNT_HW_CACHE_L1D) |
        (PERF_COUNT_HW_CACHE_OP_READ << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16)},
    {"L1_misses",        PERF_TYPE_HW_CACHE,
        (PERF_COUNT_HW_CACHE_L1D) |
        (PERF_COUNT_HW_CACHE_OP_READ << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)},
    {"L2_DC_hit",        PERF_TYPE_RAW, 0xF064},   // PMCx064 umask=0xF0
    {"L2_DC_miss",       PERF_TYPE_RAW, 0x0864},   // PMCx064 umask=0x08
    {"L2PF_hit_L2",      PERF_TYPE_RAW, 0xFF70},   // PMCx070
    {"L2PF_miss_L2_L3",  PERF_TYPE_RAW, 0xFF71},   // PMCx071
    {"L2PF_miss_all",    PERF_TYPE_RAW, 0xFF72},   // PMCx072
};

static constexpr int NUM_EVENTS = sizeof(AMD_ZEN5_EVENTS) / sizeof(AMD_ZEN5_EVENTS[0]);

// Event names matching perf stat output format for analyze_benchmark.py
static const char *PERF_STAT_NAMES[] = {
    "L1-dcache-loads",
    "L1-dcache-load-misses",
    "rF064",
    "r0864",
    "rFF70",
    "rFF71",
    "rFF72",
};
#endif // __linux__

#ifdef __linux__
static long perf_event_open_syscall(struct perf_event_attr *attr,
                                     pid_t pid, int cpu, int group_fd,
                                     unsigned long flags) {
    return syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags);
}
#endif

PerfCounterGroup::PerfCounterGroup()
    : available_(false), opened_(false) {
#ifdef __linux__
    available_ = true;
#endif
}

PerfCounterGroup::~PerfCounterGroup() {
    close();
}

bool PerfCounterGroup::open() {
#ifdef __linux__
    if (opened_) return true;

    counters_.clear();
    counters_.reserve(NUM_EVENTS);

    for (int i = 0; i < NUM_EVENTS; ++i) {
        struct perf_event_attr attr;
        std::memset(&attr, 0, sizeof(attr));
        attr.size = sizeof(attr);
        attr.type = AMD_ZEN5_EVENTS[i].type;
        attr.config = AMD_ZEN5_EVENTS[i].config;
        attr.disabled = 1;
        attr.exclude_kernel = 1;
        attr.exclude_hv = 1;
        attr.inherit = 1;

        int fd = static_cast<int>(
            perf_event_open_syscall(&attr, getpid(), -1, -1, 0));
        if (fd < 0) {
            fprintf(stderr, "[PERF] Warning: failed to open counter '%s' "
                    "(errno=%d). Run with sudo or set "
                    "perf_event_paranoid<=1.\n",
                    AMD_ZEN5_EVENTS[i].name, errno);
            for (auto &c : counters_) ::close(c.fd);
            counters_.clear();
            available_ = false;
            return false;
        }

        counters_.push_back({AMD_ZEN5_EVENTS[i].name,
                             AMD_ZEN5_EVENTS[i].type,
                             AMD_ZEN5_EVENTS[i].config, fd});
    }

    results_.resize(NUM_EVENTS);
    opened_ = true;
    return true;
#else
    return false;
#endif
}

void PerfCounterGroup::close() {
#ifdef __linux__
    for (auto &c : counters_)
        if (c.fd >= 0) ::close(c.fd);
    counters_.clear();
    opened_ = false;
#endif
}

void PerfCounterGroup::reset() {
#ifdef __linux__
    for (auto &c : counters_)
        ioctl(c.fd, PERF_EVENT_IOC_RESET, 0);
#endif
}

void PerfCounterGroup::enable() {
#ifdef __linux__
    for (auto &c : counters_)
        ioctl(c.fd, PERF_EVENT_IOC_ENABLE, 0);
#endif
}

void PerfCounterGroup::disable() {
#ifdef __linux__
    for (auto &c : counters_)
        ioctl(c.fd, PERF_EVENT_IOC_DISABLE, 0);
#endif
}

bool PerfCounterGroup::read() {
#ifdef __linux__
    if (!opened_) return false;
    for (size_t i = 0; i < counters_.size(); ++i) {
        int64_t val = 0;
        ssize_t n = ::read(counters_[i].fd, &val, sizeof(val));
        if (n != sizeof(val)) val = -1;
        results_[i] = {counters_[i].name, counters_[i].config, val};
    }
    return true;
#else
    return false;
#endif
}

static int64_t find_val(const std::vector<PerfCounterResult> &r,
                        const char *name) {
    for (auto &e : r)
        if (e.name == name) return (e.value >= 0) ? e.value : 0;
    return 0;
}

PerfCounterDerived PerfCounterGroup::derive(double elapsed_sec,
                                             int num_threads) const {
    PerfCounterDerived d = {};
    if (results_.empty()) return d;

    int64_t l1_ld   = find_val(results_, "L1_loads");
    int64_t l1_miss = find_val(results_, "L1_misses");
    int64_t l2_hit  = find_val(results_, "L2_DC_hit");
    int64_t l2_miss = find_val(results_, "L2_DC_miss");
    int64_t pf_l2   = find_val(results_, "L2PF_hit_L2");
    int64_t pf_l3   = find_val(results_, "L2PF_miss_L2_L3");
    int64_t pf_dram = find_val(results_, "L2PF_miss_all");

    int nt = std::max(num_threads, 1);

    if (l1_ld > 0)
        d.l1_miss_pct = 100.0 * l1_miss / l1_ld;

    int64_t l2_all_hit  = l2_hit + pf_l2;
    int64_t l2_all_miss = l2_miss + pf_l3 + pf_dram;
    int64_t l2_total    = l2_all_hit + l2_all_miss;
    if (l2_total > 0)
        d.l2_miss_pct = 100.0 * l2_all_miss / l2_total;

    int64_t l3_access = l2_miss + pf_l3 + pf_dram;
    if (l3_access > 0)
        d.l3_miss_pct = 100.0 * pf_dram / l3_access;

    int64_t pf_total = pf_l2 + pf_l3 + pf_dram;
    if (pf_total > 0) {
        d.pf_l2_pct   = 100.0 * pf_l2 / pf_total;
        d.pf_l3_pct   = 100.0 * pf_l3 / pf_total;
        d.pf_dram_pct = 100.0 * pf_dram / pf_total;
    }

    if (elapsed_sec > 0 && l1_miss > 0) {
        d.l2_bw_meas_gbs = (static_cast<double>(l1_miss) / nt * 64.0)
                           / elapsed_sec / 1e9;
        d.l2_bw_meas_pct = d.l2_bw_meas_gbs / L2_PEAK_GBS * 100.0;
    }

    return d;
}

void PerfCounterGroup::print_header(bool tab) {
    const char *sep = tab ? "\t" : "  ";
    printf("%sL1miss%%%sL2miss%%%sL3miss%%%sPF_L2%%%sPF_L3%%%sPF_DR%%%sL2BW%%m",
           sep, sep, sep, sep, sep, sep, sep);
}

void PerfCounterGroup::print_values(const PerfCounterDerived &d,
                                     bool tab) const {
    const char *sep = tab ? "\t" : "  ";
    printf("%s%5.1f%%%s%5.1f%%%s%5.1f%%%s%5.1f%%%s%5.1f%%%s%5.1f%%%s%5.1f%%",
           sep, d.l1_miss_pct,
           sep, d.l2_miss_pct,
           sep, d.l3_miss_pct,
           sep, d.pf_l2_pct,
           sep, d.pf_l3_pct,
           sep, d.pf_dram_pct,
           sep, d.l2_bw_meas_pct);
}

void PerfCounterGroup::print_raw_counters() const {
#ifdef __linux__
    if (results_.size() < NUM_EVENTS) return;
    for (size_t i = 0; i < results_.size() && i < NUM_EVENTS; ++i) {
        printf("  %16lld      %s\n",
               static_cast<long long>(results_[i].value),
               PERF_STAT_NAMES[i]);
    }
#endif
}

} // namespace benchdnn
} // namespace zendnnl
