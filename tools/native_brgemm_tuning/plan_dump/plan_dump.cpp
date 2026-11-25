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

// The library's plan of each case of tests/plan_cases.py, for the tuner's
// plan fixture.
//
// Reads one case per line, "cfg l1d l2 l3 family_l3 dom wei scheme N K M
// threads pack_nr group scale_bytes" (cfg: an override setting list, or -; l3
// 0: a machine without an L3; wei: a format's name, bf16, s8, s4, f16 or f32;
// scheme: per_channel or per_group, read for a quantized weight of other
// than group scales only (s8); group and scale_bytes: the group size along K
// and the bytes of one weight scale of a weight of group scales only (s4), 0
// for other weights). make_native_problem builds the planner's problem of a
// call with a bf16 source and dst (a float call: of the weight's type); a
// call over a quantized weight quantizes its source. Prints per line the problem's N
// and every plan field with validate_native_plan's verdict (valid=0|1),
// "BADCFG <why>" for a config the parser rejects, or "BADLINE". A config that
// names a key of an M = 1 call adds the GEMV kernels of the call (none on the
// tiles), whether it runs on the calling thread, and on the GEMV kernels its
// threads and blocks per kernel call over the 32-column blocks of its N
// (native_gemv_threads, native_gemv_call_blocks). The kernel registry, and
// with it the kernels of an f16 call, is this CPU's. The compute model of
// every case is the class default (UarchParams), with
// ZENDNNL_MATMUL_NATIVE_UARCH's values applied (apply_native_uarch).
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/native_kernel_registry.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_config.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_planner.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_format.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_hw.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_pack.hpp"

namespace mm = zendnnl::lowoha::matmul;
namespace nat = zendnnl::lowoha::matmul::native;
using zendnnl::memory::data_type_t;

namespace {

/// The format named wei; nullptr for no format of that name.
const nat::native_format_desc_t *format_named(const std::string &wei) {
    for (const nat::native_format_desc_t &f : nat::kNativeFormats)
        if (wei == f.name) return &f;
    return nullptr;
}

/// The first scale type of f's contract, in native_scale_t order, whose
/// scales take bytes bytes each: in dt, the data type a call gives them as;
/// false for none.
bool scale_type(
        const nat::native_format_desc_t &f, int bytes, data_type_t &dt) {
    for (unsigned i = 0; i < 8 * sizeof(f.contract.scale_types); ++i) {
        if ((f.contract.scale_types & (1u << i)) == 0) continue;
        const auto s = static_cast<nat::native_scale_t>(i);
        if (static_cast<int>(nat::native_scale_bytes(s)) == bytes) {
            dt = nat::native_scale_dt(s);
            return true;
        }
    }
    return false;
}

} // namespace

int main() {
    nat::UarchParams model;
    const char *uarch = std::getenv("ZENDNNL_MATMUL_NATIVE_UARCH");
    std::string bad;
    if (uarch != nullptr && !nat::apply_native_uarch(model, uarch, bad)) {
        std::fprintf(stderr, "ZENDNNL_MATMUL_NATIVE_UARCH: %s\n", bad.c_str());
        return 2;
    }
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream is(line);
        std::string cfg, wei, scheme;
        int l1d, l2, l3, family_l3, dom, N, K, M, threads, pack_nr, group,
                scale_bytes;
        const nat::native_format_desc_t *f = nullptr;
        data_type_t scale_dt = data_type_t::none;
        bool grouped = false;
        if (!(is >> cfg >> l1d >> l2 >> l3 >> family_l3 >> dom >> wei >> scheme
                    >> N >> K >> M >> threads >> pack_nr >> group
                    >> scale_bytes)
                || (f = format_named(wei)) == nullptr
                || (scheme != "per_channel" && scheme != "per_group")
                || ((grouped = (nat::native_type_bit(f->wei_dt)
                                       & nat::kNativeGroupedTypes)
                                    != 0)
                        && (group < 1 || K % group != 0
                                || !scale_type(*f, scale_bytes, scale_dt)))) {
            std::printf("BADLINE\n");
            continue;
        }
        const data_type_t wei_dt = f->wei_dt;
        nat::native_config_t c;
        std::string why;
        if (cfg != "-" && !nat::parse_native_config(cfg, c, why)) {
            std::printf("BADCFG %s\n", why.c_str());
            continue;
        }
        const nat::heuristic_params_t hp
                = nat::apply_native_config(nat::heuristic_params_t {}, c);
        nat::UarchParams u = model;
        u.l1d_bytes = l1d;
        u.l2_bytes = l2;
        u.l3_domain_bytes = l3;
        u.l3_detected = l3 > 0;
        u.family_l3_cores = family_l3;
        u.l3_domain_cores = dom;
        u.avx512f = u.avx512bf16 = u.avx512vnni = true;
        const bool int8 = f->quantized();
        nat::GemmDescriptor desc;
        desc.M = M;
        desc.N = N;
        desc.K = K;
        desc.src_dt = int8 ? data_type_t::bf16 : wei_dt;
        desc.wei_dt = wei_dt;
        desc.dst_dt = int8 ? data_type_t::bf16 : wei_dt;
        desc.num_threads = threads;
        mm::matmul_params params;
        params.dynamic_quant = int8;
        nat::native_int8_decision_t decision;
        decision.scheme = scheme == "per_group" || grouped
                ? nat::native_int8_scheme_t::per_group
                : nat::native_int8_scheme_t::per_channel;
        if (grouped) {
            decision.groups = K / group;
            params.quant_params.wei_scale.dt = scale_dt;
        }
        const nat::native_problem_t p = nat::make_native_problem(
                desc, params, nullptr, pack_nr, decision);
        const nat::native_plan_t q = nat::heuristic_plan(p, u, hp);
        std::string m1;
        if (c.gemv || c.gemv_blocks || c.gemv_one_thread_pct) {
            const size_t packed = nat::native_packed_bytes(
                    wei_dt, K, N, nat::kNativeLayoutPackNr);
            const size_t l2 = static_cast<size_t>(u.l2_bytes);
            m1 = std::string(" gemv=")
                    + (q.gemv ? nat::native_route(q.route).gemv : "none")
                    + " one_thread="
                    + (nat::native_one_thread(q, packed, l2) ? "1" : "0");
            if (q.gemv) {
                const int units = (N + nat::kNativeLayoutPackNr - 1)
                        / nat::kNativeLayoutPackNr;
                const int t = nat::native_gemv_threads(q, packed, l2, units);
                m1 += " gemv_threads=" + std::to_string(t) + " gemv_blocks="
                        + std::to_string(nat::native_gemv_call_blocks(
                                q, packed, l2, t, 1));
            }
        }
        std::printf(
                "n=%d nr=%d mr=%d mb=%d m_blocks=%d n_tiles=%d kc=%d kind=%s "
                "threads=%d col_groups=%d unit_tiles=%d domain_threads=%d "
                "row_unit=%d traversal=%s scratch=%zu rows=%d/%d cw=%d "
                "budget=%lld quads=%d direct=%d dequant_bias=%d scheme=%s "
                "copy_block=%d copy_par_min=%lld valid=%d%s\n",
                p.N, q.nr, q.mr, q.mb, q.m_blocks, q.n_tiles, q.kc,
                nat::native_family(q.kind).name, q.threads, q.col_groups,
                q.unit_tiles, q.domain_threads, q.row_unit,
                q.traversal == nat::native_traversal_t::m_inner ? "m_inner"
                                                                : "n_inner",
                q.scratch_floats, q.int8_rows_nv1, q.int8_rows_nv2,
                q.int8_cw_mask, static_cast<long long>(q.int8_pass_budget),
                q.int8_pass_quads, q.int8_direct_store ? 1 : 0,
                q.int8_bias_in_dequant ? 1 : 0,
                nat::native_int8_scheme_name(q.int8_scheme), q.copy_block,
                static_cast<long long>(q.copy_par_min),
                nat::validate_native_plan(q, p) == nullptr ? 1 : 0, m1.c_str());
    }
    return 0;
}
