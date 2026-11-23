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
// 0: a machine without an L3; wei: bf16, s8, s4, f16 or f32; scheme:
// per_channel or per_group, read for an s8 weight; group and scale_bytes:
// an s4 weight's group size along K and the bytes of one weight scale, 0
// for other weights). make_native_problem builds the planner's problem of a
// call with a bf16 source and dst (an f16 or f32 call: of the weight's
// type); an INT8 call quantizes its source. Prints per line the problem's N
// and every plan field with validate_native_plan's verdict (valid=0|1),
// "BADCFG <why>" for a config the parser rejects, or "BADLINE". The kernel
// registry, and with it the kernels of an f16 call, is this CPU's.
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>

#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/native_kernel_registry.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_config.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_planner.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_format.hpp"

namespace mm = zendnnl::lowoha::matmul;
namespace nat = zendnnl::lowoha::matmul::native;
using zendnnl::memory::data_type_t;

namespace {

bool weight_type(const std::string &wei, data_type_t &dt) {
    if (wei == "bf16")
        dt = data_type_t::bf16;
    else if (wei == "s8")
        dt = data_type_t::s8;
    else if (wei == "s4")
        dt = data_type_t::s4;
    else if (wei == "f16")
        dt = data_type_t::f16;
    else if (wei == "f32")
        dt = data_type_t::f32;
    else
        return false;
    return true;
}

} // namespace

int main() {
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream is(line);
        std::string cfg, wei, scheme;
        int l1d, l2, l3, family_l3, dom, N, K, M, threads, pack_nr, group,
                scale_bytes;
        data_type_t wei_dt = data_type_t::none;
        if (!(is >> cfg >> l1d >> l2 >> l3 >> family_l3 >> dom >> wei >> scheme
                    >> N >> K >> M >> threads >> pack_nr >> group
                    >> scale_bytes)
                || !weight_type(wei, wei_dt)
                || (scheme != "per_channel" && scheme != "per_group")
                || (wei_dt == data_type_t::s4
                        && (group < 1 || K % group != 0
                                || (scale_bytes != 2 && scale_bytes != 4)))) {
            std::printf("BADLINE\n");
            continue;
        }
        nat::native_config_t c;
        std::string why;
        if (cfg != "-" && !nat::parse_native_config(cfg, c, why)) {
            std::printf("BADCFG %s\n", why.c_str());
            continue;
        }
        const nat::heuristic_params_t hp
                = nat::apply_native_config(nat::heuristic_params_t {}, c);
        nat::UarchParams u;
        u.l1d_bytes = l1d;
        u.l2_bytes = l2;
        u.l3_domain_bytes = l3;
        u.l3_detected = l3 > 0;
        u.family_l3_cores = family_l3;
        u.l3_domain_cores = dom;
        u.avx512f = u.avx512bf16 = u.avx512vnni = true;
        const bool int8
                = wei_dt == data_type_t::s8 || wei_dt == data_type_t::s4;
        const bool own_type
                = wei_dt == data_type_t::f16 || wei_dt == data_type_t::f32;
        nat::GemmDescriptor desc;
        desc.M = M;
        desc.N = N;
        desc.K = K;
        desc.src_dt = own_type ? wei_dt : data_type_t::bf16;
        desc.wei_dt = wei_dt;
        desc.dst_dt = own_type ? wei_dt : data_type_t::bf16;
        desc.num_threads = threads;
        mm::matmul_params params;
        params.dynamic_quant = int8;
        nat::native_int8_decision_t decision;
        decision.scheme = scheme == "per_group" || wei_dt == data_type_t::s4
                ? nat::native_int8_scheme_t::per_group
                : nat::native_int8_scheme_t::per_channel;
        if (wei_dt == data_type_t::s4) {
            decision.groups = K / group;
            params.quant_params.wei_scale.dt
                    = scale_bytes == 4 ? data_type_t::f32 : data_type_t::bf16;
        }
        const nat::native_problem_t p = nat::make_native_problem(
                desc, params, nullptr, pack_nr, decision);
        const nat::native_plan_t q = nat::heuristic_plan(p, u, hp);
        std::printf(
                "n=%d nr=%d mr=%d mb=%d m_blocks=%d n_tiles=%d kc=%d kind=%s "
                "threads=%d col_groups=%d unit_tiles=%d domain_threads=%d "
                "row_unit=%d traversal=%s scratch=%zu rows=%d/%d cw=%d "
                "budget=%lld quads=%d direct=%d dequant_bias=%d scheme=%s "
                "copy_block=%d copy_par_min=%lld valid=%d\n",
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
                nat::validate_native_plan(q, p) == nullptr ? 1 : 0);
    }
    return 0;
}
