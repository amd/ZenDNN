/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lowoha_operators/matmul/matmul_native/common/native_hw.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include <string_view>

#include "common/zendnnl_cpuid_compat.hpp"

// After every library header: <windows.h> defines macros (interface, min,
// max) those headers must not see.
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

namespace {

/// Logical CPUs sharing the level-3 cache: EAX[25:14] + 1 of the level-3
/// sub-leaf of a cache-properties leaf; 0 when no sub-leaf reports one.
int cpuid_l3_threads(unsigned leaf) {
    for (unsigned sub = 0; sub < 16; ++sub) {
        unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
        zendnnl_cpuid_count(leaf, sub, eax, ebx, ecx, edx);
        const unsigned type = eax & 0x1Fu;
        if (type == 0) break;
        if (((eax >> 5) & 0x7u) == 3 && (type == 1 || type == 3))
            return static_cast<int>(((eax >> 14) & 0xFFFu) + 1);
    }
    return 0;
}

struct cpuid_topology_t {
    int smt = 0;
    int l3_threads = 0;
};

/// SMT width and L3 sharing from CPUID: the extended topology leaves when
/// CPUID 0x80000001 ECX[22] (topology extensions) is set, else leaves 0xB
/// and 4.
cpuid_topology_t cpuid_topology() {
    cpuid_topology_t t;
    unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
    zendnnl_cpuid(0u, eax, ebx, ecx, edx);
    const unsigned max_leaf = eax;
    zendnnl_cpuid(0x80000000u, eax, ebx, ecx, edx);
    const unsigned max_ext = eax;
    bool topo_ext = false;
    if (max_ext >= 0x80000001u) {
        zendnnl_cpuid(0x80000001u, eax, ebx, ecx, edx);
        topo_ext = (ecx & (1u << 22)) != 0;
    }
    if (topo_ext && max_ext >= 0x8000001Eu) {
        zendnnl_cpuid(0x8000001Eu, eax, ebx, ecx, edx);
        t.smt = static_cast<int>(((ebx >> 8) & 0xFFu) + 1);
    } else if (max_leaf >= 0xBu) {
        zendnnl_cpuid_count(0xBu, 0u, eax, ebx, ecx, edx);
        if (((ecx >> 8) & 0xFFu) == 1) t.smt = static_cast<int>(ebx & 0xFFFFu);
    }
    if (topo_ext && max_ext >= 0x8000001Du)
        t.l3_threads = cpuid_l3_threads(0x8000001Du);
    else if (max_leaf >= 4)
        t.l3_threads = cpuid_l3_threads(4u);
    return t;
}

/// AVX-512 F (detect_uarch()'s, which includes the OS support check) with
/// BW, VL and DQ (leaf 7 EBX) and FMA (leaf 1 ECX).
bool cpuid_base_isa(const UarchParams &uarch) {
    unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
    zendnnl_cpuid(1, eax, ebx, ecx, edx);
    const bool fma = (ecx >> 12) & 1u;
    zendnnl_cpuid_count(7, 0, eax, ebx, ecx, edx);
    const bool dq = (ebx >> 17) & 1u;
    const bool bw = (ebx >> 30) & 1u;
    const bool vl = (ebx >> 31) & 1u;
    return uarch.avx512f && fma && dq && bw && vl;
}

/// Leaf 0's vendor string and leaf 1's family, model and stepping, with the
/// extended fields folded in as the processor manuals define.  A vendor
/// string with a byte that is not printable ASCII is dropped: logs print
/// it.
void cpuid_identity(native_hw_t &hw) {
    unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
    zendnnl_cpuid(0u, eax, ebx, ecx, edx);
    const unsigned max_leaf = eax;
    const unsigned words[3] = {ebx, edx, ecx};
    std::string vendor;
    for (int i = 0; i < 12; ++i) {
        const unsigned c = (words[i / 4] >> (8 * (i % 4))) & 0xFFu;
        if (c < 0x20u || c > 0x7Eu) {
            vendor.clear();
            break;
        }
        vendor += static_cast<char>(c);
    }
    hw.vendor = vendor;
    if (max_leaf < 1) return;
    const native_cpuid_identity_t id = native_cpuid_identity();
    hw.family = id.family;
    hw.model = id.model;
    hw.stepping = id.stepping;
}

#if defined(_WIN32)

int popcount(unsigned long long m) {
    int n = 0;
    for (; m != 0; m &= m - 1)
        ++n;
    return n;
}

struct os_topology_t {
    int logical = 0;
    int smt = 0;
    int cores = 0;
    int l3_threads = 0;
};

os_topology_t os_topology() {
    os_topology_t t;
    t.logical = static_cast<int>(GetActiveProcessorCount(ALL_PROCESSOR_GROUPS));
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationAll, nullptr, &len);
    if (len == 0) return t;
    std::vector<unsigned char> buf(len);
    if (!GetLogicalProcessorInformationEx(RelationAll,
                reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(
                        buf.data()),
                &len))
        return t;
    for (DWORD off = 0; off < len;) {
        const auto *rec = reinterpret_cast<
                const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(
                buf.data() + off);
        if (rec->Relationship == RelationProcessorCore) {
            if (t.cores++ == 0)
                t.smt = popcount(static_cast<unsigned long long>(
                        rec->Processor.GroupMask[0].Mask));
        } else if (rec->Relationship == RelationCache && rec->Cache.Level == 3
                && t.l3_threads == 0) {
            t.l3_threads = popcount(
                    static_cast<unsigned long long>(rec->Cache.GroupMask.Mask));
        }
        if (rec->Size == 0) break;
        off += rec->Size;
    }
    return t;
}

#else

/// CPUs in a sysfs CPU list such as "0-3,8,10-11"; 0 when unreadable.
int count_cpu_list(const std::string &path) {
    std::ifstream f(path);
    std::string s;
    if (!f || !std::getline(f, s)) return 0;
    int n = 0;
    const char *p = s.c_str();
    while (*p != '\0') {
        char *end = nullptr;
        const long first = std::strtol(p, &end, 10);
        if (end == p || first < 0) return 0;
        long last = first;
        p = end;
        if (*p == '-') {
            last = std::strtol(p + 1, &end, 10);
            if (end == p + 1 || last < first) return 0;
            p = end;
        }
        n += static_cast<int>(last - first + 1);
        if (*p == ',')
            ++p;
        else if (*p != '\0')
            return 0;
    }
    return n;
}

/// Logical CPUs sharing the level-3 cache of CPU 0; 0 when unknown.
int sysfs_l3_threads() {
    const std::string base = "/sys/devices/system/cpu/cpu0/cache/index";
    for (int i = 0; i < 16; ++i) {
        std::ifstream f(base + std::to_string(i) + "/level");
        int level = 0;
        if (!(f >> level)) break;
        if (level == 3)
            return count_cpu_list(
                    base + std::to_string(i) + "/shared_cpu_list");
    }
    return 0;
}

#endif

native_hw_t detect_hw() {
    native_hw_t hw;
    hw.uarch = detect_uarch();
    hw.base_isa = cpuid_base_isa(hw.uarch);
    cpuid_identity(hw);
#if defined(_WIN32)
    const os_topology_t os = os_topology();
    hw.logical_cpus = os.logical;
    hw.smt = os.smt;
    hw.physical_cores = os.cores;
    hw.l3_domain_threads = os.l3_threads;
#else
    const long online = sysconf(_SC_NPROCESSORS_ONLN);
    hw.logical_cpus = online > 0 ? static_cast<int>(online) : 0;
    hw.smt = count_cpu_list(
            "/sys/devices/system/cpu/cpu0/topology/thread_siblings_list");
    hw.l3_domain_threads = sysfs_l3_threads();
#endif
    if (hw.smt <= 0 || hw.l3_domain_threads <= 0) {
        const cpuid_topology_t c = cpuid_topology();
        if (hw.smt <= 0) hw.smt = c.smt;
        if (hw.l3_domain_threads <= 0) hw.l3_domain_threads = c.l3_threads;
    }
    if (hw.physical_cores <= 0 && hw.smt > 0)
        hw.physical_cores = hw.logical_cpus / hw.smt;
    return hw;
}

} // namespace

const native_hw_t &native_hw() {
    // Never destroyed: calls may still plan while static objects are
    // destroyed.
    static const native_hw_t *const hw = new native_hw_t(detect_hw());
    return *hw;
}

UarchParams native_plan_uarch(const native_hw_t &hw) {
    UarchParams u = hw.uarch;
    if (!u.l3_detected) u.l3_domain_bytes = 0;
    u.l3_domain_cores = 0;
    if (hw.l3_domain_threads > 0) {
        const int cores = hw.l3_domain_threads / (hw.smt > 0 ? hw.smt : 1);
        u.l3_domain_cores = cores > 0 ? cores : 1;
    }
    return u;
}

namespace {

std::string_view trim(std::string_view s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t'))
        s.remove_prefix(1);
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t'))
        s.remove_suffix(1);
    return s;
}

/// One item of a ZENDNNL_MATMUL_NATIVE_UARCH value applied to out; false
/// for a bad item.  A value is a finite decimal number without spaces, as
/// the tuner's --uarch reads it (schema.Uarch.with_overrides).
bool apply_uarch_item(UarchParams &out, std::string_view item) {
    const size_t eq = item.find('=');
    const size_t us = item.rfind('_', eq);
    if (eq == std::string_view::npos || us == std::string_view::npos)
        return false;
    const std::string_view key = item.substr(0, eq);
    int path = -1;
    for (int i = 0; i < kNativeDotPaths; ++i)
        if (item.substr(0, us) == kNativeDotPathNames[i]) path = i;
    const std::string_view field = item.substr(us + 1, eq - us - 1);
    const std::string_view value = item.substr(eq + 1);
    char text[32];
    if (value.empty() || value.size() >= sizeof(text)
            || value.find_first_not_of("0123456789.eE+-")
                    != std::string_view::npos)
        return false;
    value.copy(text, value.size());
    text[value.size()] = '\0';
    char *end = nullptr;
    const double v = std::strtod(text, &end);
    if (end != text + value.size() || !std::isfinite(v)) return false;
    const bool whole = v == std::floor(v);
    if ((key == "mem_cycles" || key == "stream_cycles") && whole && v >= 1
            && v <= (1 << 20))
        (key == "mem_cycles" ? out.mem_cycles : out.stream_cycles)
                = static_cast<int>(v);
    else if (path >= 0 && field == "ipc" && v > 0 && v <= 64)
        out.dot[path].ipc = v;
    else if (path >= 0 && field == "lat" && whole && v >= 1 && v <= 64)
        out.dot[path].latency = static_cast<int>(v);
    else
        return false;
    out.dot_source = native_dot_source_t::set;
    return true;
}

/// Why the process's ZENDNNL_MATMUL_NATIVE_UARCH was ignored.
std::string &uarch_refusal() {
    static std::string why;
    return why;
}

} // namespace

bool apply_native_uarch(
        UarchParams &u, std::string_view text, std::string &why) {
    UarchParams out = u;
    for (size_t at = 0; at <= text.size();) {
        size_t end = text.find(',', at);
        if (end == std::string_view::npos) end = text.size();
        const std::string_view item = trim(text.substr(at, end - at));
        at = end + 1;
        if (!apply_uarch_item(out, item)) {
            char text[192];
            std::snprintf(text, sizeof(text),
                    "bad item '%.*s' (<path>_ipc=(0..64], <path>_lat=1..64, "
                    "mem_cycles=1..2^20 or stream_cycles=1..2^20)",
                    static_cast<int>(std::min<size_t>(item.size(), 64)),
                    item.data());
            why = text;
            return false;
        }
    }
    out.fma_ports = std::max(1,
            static_cast<int>(std::lround(
                    out.dot[static_cast<int>(native_dot_path_t::bf16)].ipc)));
    u = out;
    return true;
}

const UarchParams &native_plan_uarch() {
    static const UarchParams u = [] {
        UarchParams p = native_plan_uarch(native_hw());
        const char *text = std::getenv("ZENDNNL_MATMUL_NATIVE_UARCH");
        if (text != nullptr && *text != '\0')
            apply_native_uarch(p, text, uarch_refusal());
        return p;
    }();
    return u;
}

const std::string &native_uarch_refusal() {
    native_plan_uarch();
    return uarch_refusal();
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
