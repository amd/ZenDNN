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

#ifdef __x86_64__
#include <cpuid.h>
#endif

#include <cstdio>
#include <cstring>
#include <algorithm>

namespace zendnnl {
namespace benchdnn {

// ════════════════════════════════════════════════════════════════════════════
// Architecture detection and constants
// ════════════════════════════════════════════════════════════════════════════

static const ArchConstants ARCH_ZEN4 = {
    ZenArch::ZEN4, "Zen4", 0x19,
    32 * 1024,           // L1d = 32 KB
    1024 * 1024,         // L2  = 1 MB
    32 * 1024 * 1024,    // L3  = 32 MB per CCD
    32,                  // L2 fill BW = 32 B/cycle
    3.7,                 // nominal boost (varies by SKU)
    6,                   // 6-wide dispatch
};

static const ArchConstants ARCH_ZEN5 = {
    ZenArch::ZEN5, "Zen5", 0x1A,
    48 * 1024,           // L1d = 48 KB
    1024 * 1024,         // L2  = 1 MB
    32 * 1024 * 1024,    // L3  = 32 MB per CCD
    64,                  // L2 fill BW = 64 B/cycle (2x Zen4)
    4.121,               // EPYC 9B45 boost
    8,                   // 8-wide dispatch
};

static const ArchConstants ARCH_UNKNOWN = {
    ZenArch::UNKNOWN, "Unknown", 0,
    32 * 1024, 1024 * 1024, 32 * 1024 * 1024,
    32, 3.5, 6,
};

ZenArch detect_zen_arch() {
#ifdef __x86_64__
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (!__get_cpuid(0, &eax, &ebx, &ecx, &edx))
        return ZenArch::UNKNOWN;

    // Check for "AuthenticAMD"
    if (ebx != 0x68747541 || edx != 0x69746e65 || ecx != 0x444d4163)
        return ZenArch::UNKNOWN;

    __get_cpuid(1, &eax, &ebx, &ecx, &edx);
    unsigned int base_family = (eax >> 8) & 0xF;
    unsigned int ext_family  = (eax >> 20) & 0xFF;
    unsigned int family = base_family + ((base_family >= 0xF) ? ext_family : 0);

    if (family == 0x19) return ZenArch::ZEN4;
    if (family == 0x1A) return ZenArch::ZEN5;
#endif
    return ZenArch::UNKNOWN;
}

const ArchConstants &get_arch_constants(ZenArch arch) {
    if (arch == ZenArch::UNKNOWN)
        arch = detect_zen_arch();
    switch (arch) {
        case ZenArch::ZEN4: return ARCH_ZEN4;
        case ZenArch::ZEN5: return ARCH_ZEN5;
        default:            return ARCH_UNKNOWN;
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Profile name parsing
// ════════════════════════════════════════════════════════════════════════════

PerfProfile parse_perf_profile(const char *str) {
    if (!str || !*str || std::strcmp(str, "cache") == 0)
        return PerfProfile::CACHE;
    if (std::strcmp(str, "tlb") == 0)
        return PerfProfile::TLB;
    if (std::strcmp(str, "stalls") == 0)
        return PerfProfile::STALLS;
    static bool warned = false;
    if (!warned) {
        fprintf(stderr, "[PERF] Unknown profile '%s', using 'cache'\n", str);
        warned = true;
    }
    return PerfProfile::CACHE;
}

const char *perf_profile_name(PerfProfile p) {
    switch (p) {
        case PerfProfile::CACHE:  return "cache";
        case PerfProfile::TLB:    return "tlb";
        case PerfProfile::STALLS: return "stalls";
    }
    return "cache";
}

// ════════════════════════════════════════════════════════════════════════════
// PMU event definitions — uniform across Zen4 (Family 19h) and Zen5 (1Ah)
//
// Verified via LIKWID perfmon_zen4_events.txt and AMD PPR for Family 1Ah.
// All core PMC event codes (PMCx0xx) are identical between Zen4 and Zen5.
// ════════════════════════════════════════════════════════════════════════════

#ifdef __linux__

struct EventDef {
    const char *name;      // internal name for find_val()
    const char *perf_name; // name matching `perf stat` output for analyze_benchmark.py
    uint32_t type;
    uint64_t config;
};

// ── CACHE profile (7 events) ────────────────────────────────────────────────
// PMCx064 L2CacheReqStat — demand requests from L1 to L2 (excludes PF)
// PMCx070-072 — L2 prefetcher outcome counters
static const EventDef EVENTS_CACHE[] = {
    {"L1_loads",   "L1-dcache-loads",       PERF_TYPE_HW_CACHE,
        (PERF_COUNT_HW_CACHE_L1D) |
        (PERF_COUNT_HW_CACHE_OP_READ << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16)},
    {"L1_misses",  "L1-dcache-load-misses", PERF_TYPE_HW_CACHE,
        (PERF_COUNT_HW_CACHE_L1D) |
        (PERF_COUNT_HW_CACHE_OP_READ << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)},
    {"L2_DC_hit",       "rF064", PERF_TYPE_RAW, 0xF064},  // PMCx064 umask=0xF0
    {"L2_DC_miss",      "r0864", PERF_TYPE_RAW, 0x0864},  // PMCx064 umask=0x08
    {"L2PF_hit_L2",     "rFF70", PERF_TYPE_RAW, 0xFF70},  // PMCx070
    {"L2PF_miss_L2_L3", "rFF71", PERF_TYPE_RAW, 0xFF71},  // PMCx071
    {"L2PF_miss_all",   "rFF72", PERF_TYPE_RAW, 0xFF72},  // PMCx072
};
static constexpr int NUM_EVENTS_CACHE = sizeof(EVENTS_CACHE) / sizeof(EVENTS_CACHE[0]);

// ── TLB profile (6 events) ─────────────────────────────────────────────────
// PMCx045 L1DtlbMiss — L1 DTLB miss outcomes
// PMCx0C0 retired instructions, PMCx076 unhalted cycles → IPC
static const EventDef EVENTS_TLB[] = {
    {"L1_loads",   "L1-dcache-loads",       PERF_TYPE_HW_CACHE,
        (PERF_COUNT_HW_CACHE_L1D) |
        (PERF_COUNT_HW_CACHE_OP_READ << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16)},
    {"L1_misses",  "L1-dcache-load-misses", PERF_TYPE_HW_CACHE,
        (PERF_COUNT_HW_CACHE_L1D) |
        (PERF_COUNT_HW_CACHE_OP_READ << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)},
    {"DTLB_L2_hit",  "r0F45", PERF_TYPE_RAW, 0x0F45},  // PMCx045 umask=0x0F (any L2 DTLB hit)
    {"DTLB_L2_miss", "rF045", PERF_TYPE_RAW, 0xF045},  // PMCx045 umask=0xF0 (any L2 DTLB miss → page walk)
    {"ret_insn",     "r00C0", PERF_TYPE_RAW, 0x00C0},  // PMCx0C0 retired instructions
    {"cycles",       "r0076", PERF_TYPE_RAW, 0x0076},  // PMCx076 unhalted cycles
};
static constexpr int NUM_EVENTS_TLB = sizeof(EVENTS_TLB) / sizeof(EVENTS_TLB[0]);

// ── STALLS profile (6 events) ──────────────────────────────────────────────
// PMCx0AE dispatch resource stalls, PMCx0AF retire stalls
static const EventDef EVENTS_STALLS[] = {
    {"ret_insn",       "r00C0", PERF_TYPE_RAW, 0x00C0},  // PMCx0C0 retired instructions
    {"cycles",         "r0076", PERF_TYPE_RAW, 0x0076},  // PMCx076 unhalted cycles
    {"fp_reg_stall",   "r20AE", PERF_TYPE_RAW, 0x20AE},  // PMCx0AE umask=0x20 FP regfile full
    {"fp_sched_stall", "r40AE", PERF_TYPE_RAW, 0x40AE},  // PMCx0AE umask=0x40 FP scheduler full
    {"lq_stall",       "r02AE", PERF_TYPE_RAW, 0x02AE},  // PMCx0AE umask=0x02 load queue full
    {"retire_stall",   "r20AF", PERF_TYPE_RAW, 0x20AF},  // PMCx0AF umask=0x20 retire token stall
};
static constexpr int NUM_EVENTS_STALLS = sizeof(EVENTS_STALLS) / sizeof(EVENTS_STALLS[0]);

static const EventDef *get_events(PerfProfile p, int &count) {
    switch (p) {
        case PerfProfile::CACHE:
            count = NUM_EVENTS_CACHE;
            return EVENTS_CACHE;
        case PerfProfile::TLB:
            count = NUM_EVENTS_TLB;
            return EVENTS_TLB;
        case PerfProfile::STALLS:
            count = NUM_EVENTS_STALLS;
            return EVENTS_STALLS;
    }
    count = NUM_EVENTS_CACHE;
    return EVENTS_CACHE;
}

static long perf_event_open_syscall(struct perf_event_attr *attr,
                                     pid_t pid, int cpu, int group_fd,
                                     unsigned long flags) {
    return syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags);
}
#endif // __linux__

// ════════════════════════════════════════════════════════════════════════════
// PerfCounterGroup implementation
// ════════════════════════════════════════════════════════════════════════════

PerfCounterGroup::PerfCounterGroup(PerfProfile profile)
    : profile_(profile), arch_(ZenArch::UNKNOWN),
      available_(false), opened_(false) {
#ifdef __linux__
    arch_ = detect_zen_arch();
    available_ = true;
#endif
}

PerfCounterGroup::~PerfCounterGroup() {
    close();
}

bool PerfCounterGroup::open() {
#ifdef __linux__
    if (opened_) return true;

    int num_events = 0;
    const EventDef *events = get_events(profile_, num_events);

    const auto &ac = get_arch_constants(arch_);
    static bool printed_banner = false;
    if (!printed_banner) {
        fprintf(stderr, "[PERF] Detected %s (Family 0x%02X), profile=%s, "
                "L2 fill BW=%d B/cycle\n",
                ac.name, ac.family, perf_profile_name(profile_),
                ac.l2_bw_bytes_per_cycle);
        printed_banner = true;
    }

    counters_.clear();
    counters_.reserve(num_events);

    for (int i = 0; i < num_events; ++i) {
        struct perf_event_attr attr;
        std::memset(&attr, 0, sizeof(attr));
        attr.size = sizeof(attr);
        attr.type = events[i].type;
        attr.config = events[i].config;
        attr.disabled = 1;
        attr.exclude_kernel = 1;
        attr.exclude_hv = 1;
        attr.inherit = 1;

        int fd = static_cast<int>(
            perf_event_open_syscall(&attr, getpid(), -1, -1, 0));
        if (fd < 0) {
            fprintf(stderr, "[PERF] Warning: failed to open counter '%s' "
                    "(errno=%d: %s). Run with sudo or set "
                    "perf_event_paranoid<=1.\n",
                    events[i].name, errno, strerror(errno));
            for (auto &c : counters_) ::close(c.fd);
            counters_.clear();
            available_ = false;
            return false;
        }

        counters_.push_back({events[i].name,
                             events[i].type,
                             events[i].config, fd});
    }

    results_.resize(num_events);
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

// ════════════════════════════════════════════════════════════════════════════
// Derived metrics — uniform output regardless of Zen4 or Zen5
// ════════════════════════════════════════════════════════════════════════════

static int64_t find_val(const std::vector<PerfCounterResult> &r,
                        const char *name) {
    for (auto &e : r)
        if (e.name == name) return (e.value >= 0) ? e.value : 0;
    return 0;
}

PerfCounterDerived PerfCounterGroup::derive(double elapsed_sec,
                                             int num_threads) const {
    PerfCounterDerived d = {};
    d.profile = profile_;
    if (results_.empty()) return d;

    const auto &ac = get_arch_constants(arch_);
    const double l2_peak_gbs = ac.l2_bw_bytes_per_cycle * ac.cpu_freq_ghz;
    int nt = std::max(num_threads, 1);

    switch (profile_) {
    case PerfProfile::CACHE: {
        int64_t l1_ld   = find_val(results_, "L1_loads");
        int64_t l1_miss = find_val(results_, "L1_misses");
        // HW_CACHE L1 events can undercount on AMD Zen (streaming stores
        // not tracked). Use max(ld, miss) as denominator to avoid > 100%.
        int64_t l1_denom = std::max(l1_ld, l1_miss);
        int64_t l2_hit  = find_val(results_, "L2_DC_hit");
        int64_t l2_miss = find_val(results_, "L2_DC_miss");
        int64_t pf_l2   = find_val(results_, "L2PF_hit_L2");
        int64_t pf_l3   = find_val(results_, "L2PF_miss_L2_L3");
        int64_t pf_dram = find_val(results_, "L2PF_miss_all");

        if (l1_denom > 0)
            d.l1_miss_pct = std::min(100.0, 100.0 * l1_miss / l1_denom);

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
            d.l2_bw_meas_pct = d.l2_bw_meas_gbs / l2_peak_gbs * 100.0;
        }
        break;
    }
    case PerfProfile::TLB: {
        int64_t l1_ld     = find_val(results_, "L1_loads");
        int64_t l1_miss   = find_val(results_, "L1_misses");
        int64_t l1_denom  = std::max(l1_ld, l1_miss);
        int64_t dtlb_hit  = find_val(results_, "DTLB_L2_hit");
        int64_t dtlb_miss = find_val(results_, "DTLB_L2_miss");
        int64_t ret_insn  = find_val(results_, "ret_insn");
        int64_t cycles    = find_val(results_, "cycles");

        if (l1_denom > 0)
            d.l1_miss_pct = std::min(100.0, 100.0 * l1_miss / l1_denom);

        if (l1_ld > 0) {
            d.dtlb_l2_hit_pct  = 100.0 * dtlb_hit / l1_ld;
            d.dtlb_l2_miss_pct = 100.0 * dtlb_miss / l1_ld;
        }

        if (cycles > 0)
            d.ipc = static_cast<double>(ret_insn) / cycles;

        if (elapsed_sec > 0 && l1_miss > 0) {
            d.l2_bw_meas_gbs = (static_cast<double>(l1_miss) / nt * 64.0)
                               / elapsed_sec / 1e9;
            d.l2_bw_meas_pct = d.l2_bw_meas_gbs / l2_peak_gbs * 100.0;
        }
        break;
    }
    case PerfProfile::STALLS: {
        int64_t ret_insn    = find_val(results_, "ret_insn");
        int64_t cycles      = find_val(results_, "cycles");
        int64_t fp_reg      = find_val(results_, "fp_reg_stall");
        int64_t fp_sched    = find_val(results_, "fp_sched_stall");
        int64_t lq          = find_val(results_, "lq_stall");
        int64_t retire      = find_val(results_, "retire_stall");

        if (cycles > 0) {
            d.ipc = static_cast<double>(ret_insn) / cycles;
            d.fp_reg_stall_pct   = 100.0 * fp_reg / cycles;
            d.fp_sched_stall_pct = 100.0 * fp_sched / cycles;
            d.lq_stall_pct       = 100.0 * lq / cycles;
            d.retire_stall_pct   = 100.0 * retire / cycles;
        }
        break;
    }
    }
    return d;
}

// ════════════════════════════════════════════════════════════════════════════
// Printing — uniform format across architectures
// ════════════════════════════════════════════════════════════════════════════

void PerfCounterGroup::print_header(bool tab) const {
    const char *sep = tab ? "\t" : "  ";
    switch (profile_) {
    case PerfProfile::CACHE:
        printf("%sL1miss%%%sL2miss%%%sL3miss%%%sPF_L2%%%sPF_L3%%%sPF_DR%%%sL2BW%%m",
               sep, sep, sep, sep, sep, sep, sep);
        break;
    case PerfProfile::TLB:
        printf("%sL1miss%%%sDTLB_hit%%%sDTLB_walk%%%sIPC%sL2BW%%m",
               sep, sep, sep, sep, sep);
        break;
    case PerfProfile::STALLS:
        printf("%sIPC%sFP_reg%%%sFP_sch%%%sLQ%%%sRetire%%",
               sep, sep, sep, sep, sep);
        break;
    }
}

void PerfCounterGroup::print_values(const PerfCounterDerived &d,
                                     bool tab) const {
    const char *sep = tab ? "\t" : "  ";
    switch (profile_) {
    case PerfProfile::CACHE:
        printf("%s%5.1f%%%s%5.1f%%%s%5.1f%%%s%5.1f%%%s%5.1f%%%s%5.1f%%%s%5.1f%%",
               sep, d.l1_miss_pct,
               sep, d.l2_miss_pct,
               sep, d.l3_miss_pct,
               sep, d.pf_l2_pct,
               sep, d.pf_l3_pct,
               sep, d.pf_dram_pct,
               sep, d.l2_bw_meas_pct);
        break;
    case PerfProfile::TLB:
        printf("%s%5.1f%%%s%5.2f%%%s%5.3f%%%s%5.2f%s%5.1f%%",
               sep, d.l1_miss_pct,
               sep, d.dtlb_l2_hit_pct,
               sep, d.dtlb_l2_miss_pct,
               sep, d.ipc,
               sep, d.l2_bw_meas_pct);
        break;
    case PerfProfile::STALLS:
        printf("%s%5.2f%s%5.1f%%%s%5.1f%%%s%5.1f%%%s%5.1f%%",
               sep, d.ipc,
               sep, d.fp_reg_stall_pct,
               sep, d.fp_sched_stall_pct,
               sep, d.lq_stall_pct,
               sep, d.retire_stall_pct);
        break;
    }
}

void PerfCounterGroup::print_raw_counters() const {
#ifdef __linux__
    int num_events = 0;
    const EventDef *events = get_events(profile_, num_events);
    if (static_cast<int>(results_.size()) < num_events) return;

    const auto &ac = get_arch_constants(arch_);
    printf("  [ARCH] %s (Family 0x%02X) profile=%s\n",
           ac.name, ac.family, perf_profile_name(profile_));

    for (int i = 0; i < num_events; ++i) {
        printf("  %16lld      %s\n",
               static_cast<long long>(results_[i].value),
               events[i].perf_name);
    }
#endif
}

} // namespace benchdnn
} // namespace zendnnl
