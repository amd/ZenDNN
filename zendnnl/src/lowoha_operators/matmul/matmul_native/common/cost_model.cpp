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

#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"
#include "common/platform_info.hpp"
#include "common/zendnnl_global.hpp"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <omp.h>
#include "common/zendnnl_cpuid_compat.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

namespace {

/// CPUID leaf 0's EBX of the vendor whose CPUs report their caches in leaf
/// 0x8000001D (the others in leaf 4).
constexpr uint32_t kExtCacheVendor = 0x68747541u;

/// Bytes of the level-level data or unified cache the cache-properties leaf
/// (0x8000001D or 4) reports; 0 when no sub-leaf reports one.
int cpuid_cache_size(unsigned leaf, int level) {
    for (unsigned sub = 0; sub < 16; ++sub) {
        unsigned eax = 0, ebx = 0, ecx_out = 0, edx = 0;
        zendnnl_cpuid_count(leaf, sub, eax, ebx, ecx_out, edx);
        int cache_type = eax & 0x1F;
        if (cache_type == 0) break;
        int cache_level = (eax >> 5) & 0x7;
        if (cache_level == level && (cache_type == 1 || cache_type == 3)) {
            int line_size = (ebx & 0xFFF) + 1;
            int partitions = ((ebx >> 12) & 0x3FF) + 1;
            int ways = ((ebx >> 22) & 0x3FF) + 1;
            int sets = ecx_out + 1;
            return static_cast<int>(
                    static_cast<int64_t>(line_size) * partitions * ways * sets);
        }
    }
    return 0;
}

} // namespace

int native_dot_chains(const native_dot_t &d) {
    return std::max(1, static_cast<int>(std::ceil(d.ipc * d.latency - 1e-9)));
}

bool native_dot_prices(const UarchParams &u) {
    return u.dot_source == native_dot_source_t::set;
}

native_cpuid_identity_t native_cpuid_identity() {
    native_cpuid_identity_t id;
    unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
    zendnnl_cpuid(0u, eax, ebx, ecx, edx);
    id.vendor_ebx = ebx;
    if (eax < 1) return id;
    zendnnl_cpuid(1u, eax, ebx, ecx, edx);
    const int family = static_cast<int>((eax >> 8) & 0xFu);
    const int model = static_cast<int>((eax >> 4) & 0xFu);
    id.stepping = static_cast<int>(eax & 0xFu);
    id.family = family == 0xF ? family + static_cast<int>((eax >> 20) & 0xFFu)
                              : family;
    id.model = family == 0x6 || family == 0xF
            ? model + static_cast<int>(((eax >> 16) & 0xFu) << 4)
            : model;
    return id;
}

void set_cache_sizes(UarchParams &p, int l1d, int l2, int l3) {
    const UarchParams d;
    p.l1d_bytes = l1d > 0 ? l1d : d.l1d_bytes;
    p.l2_bytes = l2 > 0 ? l2 : d.l2_bytes;
    p.l3_domain_bytes = l3 > 0 ? l3 : p.l2_bytes;
    p.l1d_detected = l1d > 0;
    p.l2_detected = l2 > 0;
    p.l3_detected = l3 > 0;
}

static UarchParams do_detect() {
    UarchParams p;
    auto &pinfo = zendnnl::common::zendnnl_platform_info();
    // avx512f status includes OS XSAVE/OSXSAVE + XCR0 validation
    // (verified inside platform_info), so sub-feature checks below
    // only need to confirm the CPUID bit + avx512f gate.
    p.avx512f = pinfo.get_avx512f_status();
    // Note: num_cores is the OMP thread count (logical CPUs), not the
    // physical core count. On SMT systems this may be 2x the core count.
    // Planner code that computes threads_sharing_l3 should account for this.
    p.num_cores = omp_get_max_threads();

    uint32_t cpu_family = pinfo.get_cpu_family();

    const native_cpuid_identity_t id = native_cpuid_identity();
    const bool ext_vendor = id.vendor_ebx == kExtCacheVendor;
    const unsigned cache_leaf = ext_vendor ? 0x8000001Du : 0x04u;
    set_cache_sizes(p, cpuid_cache_size(cache_leaf, 1),
            cpuid_cache_size(cache_leaf, 2), cpuid_cache_size(cache_leaf, 3));

    if (ext_vendor && cpu_family == 0x1A) {
        p.vec_width_bits = 512;
        if (p.l1d_bytes < 49152) p.l1d_bytes = 49152;
    } else if (ext_vendor && cpu_family == 0x19) {
        p.vec_width_bits = 256;
    } else {
        p.vec_width_bits = p.avx512f ? 512 : 256;
    }

    unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
    zendnnl_cpuid_count(7, 0, eax, ebx, ecx, edx);
    p.avx512vnni = p.avx512f && (ecx & (1u << 11));
    p.avx512fp16 = p.avx512f && (edx & (1u << 23));
    zendnnl_cpuid_count(7, 1, eax, ebx, ecx, edx);
    p.avx512bf16 = p.avx512f && (eax & (1u << 5));

    // Cores per L3 domain of the CPU families recognized here.
    if (ext_vendor && (cpu_family == 0x19 || cpu_family == 0x1A))
        p.family_l3_cores = 8;
    else
        p.family_l3_cores = 1;

    return p;
}

const UarchParams &detect_uarch() {
    static UarchParams cached = do_detect();
    return cached;
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
