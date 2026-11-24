/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lowoha_operators/matmul/group_matmul/common/grp_matmul_machine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace grp_machine {

namespace {

/// Where the u8-source kernels were measured to pay before the rates were:
/// AMD family 1Ah models 50h-5Fh.
bool u8_src_paid_on(const native::native_hw_t &hw) {
    return hw.vendor == "AuthenticAMD" && hw.family == 0x1A && hw.model >= 0x50
            && hw.model <= 0x5F;
}

} // namespace

int dot_rates_t::chains(const dot_path p) const {
    const dot_rate &r = rate(p);
    return std::max(1, static_cast<int>(std::ceil(r.ipc * r.latency - 1e-9)));
}

machine_t make_machine(const native::native_hw_t &hw) {
    machine_t m;
    m.l1d_bytes = hw.uarch.l1d_bytes;
    m.l2_bytes = hw.uarch.l2_bytes;
    m.l3_domain_bytes = hw.uarch.l3_domain_bytes;
    m.smt = std::max(hw.smt, 1);
    m.l3_domain_cores
            = hw.l3_domain_threads > 0 ? hw.l3_domain_threads / m.smt : 0;
    m.l2_plan_bytes = m.l2_bytes >= 64 * 1024 ? m.l2_bytes : 256 * 1024;
    m.l2_budget_bytes = m.l2_plan_bytes * 4 / 5;
    return m;
}

dot_rates_t make_dot_rates(const native::native_hw_t &hw,
        const std::array<dot_rate, kDotPaths> &dot, const bool measured) {
    dot_rates_t d;
    d.dot = dot;
    d.measured = measured;
    const double others
            = std::max(d.rate(dot_path::bf16).ipc, d.rate(dot_path::fp32).ipc);
    d.u8_src_pays = measured
            ? others > 0 && d.rate(dot_path::int8).ipc > 1.5 * others
            : u8_src_paid_on(hw);
    return d;
}

const machine_t &machine() {
    // Never destroyed: calls may still plan while static objects are
    // destroyed.
    static const machine_t *const m
            = new machine_t(make_machine(native::native_hw()));
    return *m;
}

const dot_rates_t &dot_rates() {
    static const dot_rates_t *const d = [] {
        const native::native_hw_t &hw = native::native_hw();
        std::array<dot_rate, kDotPaths> dot {};
        const auto t0 = std::chrono::steady_clock::now();
        const bool measured = measure_dot_rates(hw.uarch, dot);
        const auto t1 = std::chrono::steady_clock::now();
        auto *made = new dot_rates_t(make_dot_rates(hw, dot, measured));
        made->measure_ms
                = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return made;
    }();
    return *d;
}

std::string describe(const machine_t &m, const dot_rates_t &d) {
    std::string out;
    char buf[160];
    std::snprintf(buf, sizeof(buf),
            "l1d=%lldK l2=%lldK l3=%lldK/%d cores smt=%d",
            static_cast<long long>(m.l1d_bytes / 1024),
            static_cast<long long>(m.l2_bytes / 1024),
            static_cast<long long>(m.l3_domain_bytes / 1024), m.l3_domain_cores,
            m.smt);
    out += buf;
    if (d.measured) {
        std::snprintf(
                buf, sizeof(buf), " dot(measured, %.1f ms):", d.measure_ms);
        out += buf;
        for (int p = 0; p < kDotPaths; ++p) {
            std::snprintf(buf, sizeof(buf), " %s=%.2f/%d", kDotPathNames[p],
                    d.dot[p].ipc, d.dot[p].latency);
            out += buf;
        }
    } else {
        out += " dot(not measured)";
    }
    std::snprintf(buf, sizeof(buf), " l2_budget=%lldK u8_src=%d",
            static_cast<long long>(m.l2_budget_bytes / 1024),
            d.u8_src_pays ? 1 : 0);
    out += buf;
    return out;
}

} // namespace grp_machine
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
