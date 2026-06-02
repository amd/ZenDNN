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

#include "grp_matmul_utils.hpp"
#include "utils/benchdnn_utils.hpp"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>

namespace zendnnl {
namespace benchdnn {
namespace grp_matmul {

std::vector<int> parse_M(const std::string &s, int num_ops) {
    std::vector<int> result;
    if (s.find(':') != std::string::npos) {
        std::istringstream ms(s);
        std::string tok;
        while (std::getline(ms, tok, ':'))
            result.push_back(std::stoi(tok));
    } else {
        int m = std::stoi(s);
        result.assign(num_ops, m);
    }
    if (static_cast<int>(result.size()) < num_ops)
        result.resize(num_ops, result.empty() ? 1 : result.back());
    return result;
}

bool parse_config(const std::string &line, GrpMatmulConfig &cfg) {
    if (line.empty() || line[0] == '#') return false;
    std::istringstream ss(line);
    std::string tok;
    bool has_more = true;
    auto next = [&]() -> std::string {
        if (!has_more) return "";
        if (!std::getline(ss, tok, ',')) { has_more = false; return ""; }
        size_t s = tok.find_first_not_of(" \t");
        size_t e = tok.find_last_not_of(" \t");
        return (s == std::string::npos) ? "" : tok.substr(s, e - s + 1);
    };
    try {
        cfg.num_ops = std::stoi(next());
        std::string m_str = next();
        cfg.K = std::stoi(next());
        cfg.N = std::stoi(next());
        cfg.iters = std::stoi(next());
        std::string dtypes = next();
        auto p1 = dtypes.find(':');
        auto p2 = dtypes.find(':', p1 + 1);
        cfg.src_dt = strToDatatype(dtypes.substr(0, p1));
        cfg.wei_dt = strToDatatype(dtypes.substr(p1 + 1, p2 - p1 - 1));
        cfg.dst_dt = strToDatatype(dtypes.substr(p2 + 1));
        std::string wc = next();
        cfg.is_weights_const = (wc != "false" && wc != "0");
        std::string wp = next();
        if (!wp.empty()) cfg.warmup = std::stoi(wp);
        std::string moe_str = next();
        if (!moe_str.empty()) cfg.moe_topk = std::stoi(moe_str);
        std::string act_str = next();
        if (!act_str.empty()) cfg.gated_act = std::stoi(act_str);
        std::string ndown_str = next();
        if (!ndown_str.empty()) cfg.N_down = std::stoi(ndown_str);
        // 12th column (optional): use_internal_alloc.  Accept "1" / "true"
        // / "internal_alloc" as ON; anything else (including empty) is OFF.
        std::string ialloc_str = next();
        if (!ialloc_str.empty()) {
            cfg.use_internal_alloc =
                (ialloc_str == "1" || ialloc_str == "true"
                 || ialloc_str == "internal_alloc") ? 1 : 0;
        }
        // 13th column (optional): total_experts.  Drives the framework
        // prepack-extras contract: when > num_ops the dispatcher receives
        // weight buffers for all `total_experts` slots (first `num_ops` =
        // firing experts, rest = extras the prepack module pre-warms).
        // When 0 / absent, treated as `total_experts = num_ops` (legacy
        // contract: active == total).  See `grp_matmul_utils.hpp` for the
        // full semantic and `params[i].active_matmul / total_matmul` in
        // `lowoha_common.hpp` for the library-side contract.
        std::string total_str = next();
        if (!total_str.empty()) cfg.total_experts = std::stoi(total_str);
        // 14th column (optional): dynamic_quant toggle for the DQ-INT8
        // family.  Accepted forms:
        //   "0" / "false" / "off" / empty   → bf16 path (default)
        //   "1" / "true" / "on" / "dq_int8" → DQ-INT8 path
        // Validation against the (src, wei, dst) tuple and the K%4
        // microkernel constraint happens below, after total_experts
        // is settled, so a single message can refer to all settled
        // fields.
        std::string dq_str = next();
        if (!dq_str.empty()) {
            if (dq_str == "0" || dq_str == "false" || dq_str == "off") {
                cfg.dynamic_quant = 0;
            } else if (dq_str == "1" || dq_str == "true" || dq_str == "on"
                       || dq_str == "dq_int8" || dq_str == "dynamic_quant") {
                cfg.dynamic_quant = 1;
            } else {
                std::cerr << "parse_config: dynamic_quant='" << dq_str
                          << "' must be one of {0,1,false,true,off,on,"
                             "dq_int8,dynamic_quant}.\n";
                return false;
            }
        }
        // 15th column (optional): compute_dt for the DQ-INT8 family.
        // Drives the sym vs asym discriminator on
        // `params[i].dtypes.compute`:
        //   "s8" (default)        → kS8_S8_BF16_SYM
        //   "u8" / "asym"         → kU8_S8_BF16_ASYM
        // Ignored when dynamic_quant == 0 so legacy bf16 input lines
        // (which never carry this column) keep the same defaults.
        std::string cdt_str = next();
        if (!cdt_str.empty()) {
            if (cdt_str == "u8" || cdt_str == "asym")
                cfg.compute_dt = data_type_t::u8;
            else if (cdt_str == "s8" || cdt_str == "sym")
                cfg.compute_dt = data_type_t::s8;
            else {
                std::cerr << "parse_config: compute_dt='" << cdt_str
                          << "' must be one of {s8, u8, sym, asym}.\n";
                return false;
            }
        }
        cfg.M_per_op = parse_M(m_str, cfg.num_ops);
    } catch (...) {
        return false;
    }
    // Internal-alloc requires fused down_proj: the library only owns
    // Op1's scratch and reuses src for Op2 output, which is meaningful
    // only when there IS an Op2.
    if (cfg.use_internal_alloc && cfg.N_down <= 0) {
        std::cerr << "parse_config: use_internal_alloc=1 requires N_down>0 "
                     "(library-managed scratch is only meaningful for "
                     "fused Op1+Act+Op2). Skipping.\n";
        return false;
    }
    // Reject zero or negative M values.  group_matmul_direct treats M<=0 as
    // invalid, and the fused-moe buffer allocator would otherwise call
    // aligned_alloc(64, 0).  Inactive experts must be expressed as fewer
    // num_ops, not as zero-M entries.
    if (cfg.num_ops <= 0 || cfg.K <= 0 || cfg.N <= 0) return false;
    for (int m : cfg.M_per_op) {
        if (m <= 0) {
            std::cerr << "parse_config: M[?]=" << m
                      << " is not positive (reduce num_ops instead)\n";
            return false;
        }
    }
    // total_experts: defaulted to num_ops when absent; reject the
    // logically-impossible "fewer total than firing".
    if (cfg.total_experts == 0) {
        cfg.total_experts = cfg.num_ops;
    } else if (cfg.total_experts < cfg.num_ops) {
        std::cerr << "parse_config: total_experts=" << cfg.total_experts
                  << " < num_ops=" << cfg.num_ops
                  << " is invalid (total must include all firing experts)\n";
        return false;
    }

    // DQ-INT8 contract — the resolve_variant() truth table in
    // custom_kernel/dispatch.cpp accepts dynamic_quant=true ONLY for
    // (src=bf16, wei=s8, dst=bf16) with compute ∈ {s8, u8}.  Refuse
    // here so the driver fails fast with a precise message instead
    // of dispatching a config the library will silently fall back
    // off the CK path for.  Also enforce the K%4 microkernel
    // alignment up front (mirrors the `sym_k = (k/4)*4` clamp in
    // test_quant.cpp::INT8_DYNAMIC_GEMM_BF16).
    if (cfg.dynamic_quant) {
        if (cfg.src_dt != data_type_t::bf16
            || cfg.wei_dt != data_type_t::s8
            || cfg.dst_dt != data_type_t::bf16) {
            std::cerr << "parse_config: dynamic_quant=1 requires "
                         "src:wei:dst = bf16:s8:bf16 (CK DQ-INT8 truth "
                         "table); got "
                      << datatypeToStr(cfg.src_dt) << ":"
                      << datatypeToStr(cfg.wei_dt) << ":"
                      << datatypeToStr(cfg.dst_dt) << "\n";
            return false;
        }
        if (cfg.compute_dt != data_type_t::s8
            && cfg.compute_dt != data_type_t::u8) {
            std::cerr << "parse_config: dynamic_quant=1 compute_dt must "
                         "be s8 (sym) or u8 (asym); got "
                      << datatypeToStr(cfg.compute_dt) << "\n";
            return false;
        }
        if ((cfg.K & 3) != 0) {
            std::cerr << "parse_config: dynamic_quant=1 requires K%4==0 "
                         "(int8 microkernel reduces in 4-byte VPDPBUSD "
                         "lanes); got K="
                      << cfg.K << "\n";
            return false;
        }
    }
    return true;
}

std::string format_M(const GrpMatmulConfig &cfg) {
    if (cfg.is_uniform_M())
        return std::to_string(cfg.M_per_op[0]);
    int mn = *std::min_element(cfg.M_per_op.begin(), cfg.M_per_op.end());
    int mx = cfg.max_M();
    int avg = cfg.total_M() / cfg.num_ops;
    return std::to_string(mn) + "-" + std::to_string(mx)
           + "(" + std::to_string(avg) + ")";
}

void fill_bf16_random(void *buf, size_t elems, uint32_t seed) {
    auto *p = static_cast<uint16_t *>(buf);
    uint32_t state = seed;
    for (size_t i = 0; i < elems; ++i) {
        state = state * 1103515245u + 12345u;
        float val = static_cast<float>(static_cast<int>(state >> 16) % 200 - 100)
                    * 0.01f;
        uint32_t bits;
        std::memcpy(&bits, &val, sizeof(bits));
        p[i] = static_cast<uint16_t>(bits >> 16);
    }
}

void fill_buffer(void *buf, size_t elems, data_type_t dt, uint32_t seed) {
    if (dt == data_type_t::bf16) {
        fill_bf16_random(buf, elems, seed);
    } else if (dt == data_type_t::f32) {
        auto *p = static_cast<float *>(buf);
        uint32_t state = seed;
        for (size_t i = 0; i < elems; ++i) {
            state = state * 1103515245u + 12345u;
            p[i] = static_cast<float>(static_cast<int>(state >> 16) % 200 - 100)
                    * 0.01f;
        }
    } else if (dt == data_type_t::s8) {
        // Symmetric int8 weight — uniform in [-127, 127].  -128 is
        // intentionally avoided so a future asym-corrupt sanity
        // pass cannot detect the [-128] all-bits-set sentinel as an
        // accidental wraparound.
        auto *p = static_cast<int8_t *>(buf);
        uint32_t state = seed;
        for (size_t i = 0; i < elems; ++i) {
            state = state * 1103515245u + 12345u;
            int v = static_cast<int>(state >> 16) % 255 - 127;
            p[i] = static_cast<int8_t>(v);
        }
    } else if (dt == data_type_t::u8) {
        auto *p = static_cast<uint8_t *>(buf);
        uint32_t state = seed;
        for (size_t i = 0; i < elems; ++i) {
            state = state * 1103515245u + 12345u;
            p[i] = static_cast<uint8_t>((state >> 16) & 0xFF);
        }
    } else {
        std::memset(buf, 0, elems * size_of(dt));
    }
}

void fill_quant_scale_f32(float *buf, size_t elems, uint32_t seed) {
    // Real per-channel / per-token scales for bf16 inputs in
    // [-1, 1] are roughly `max(|x|) / 127 ≈ 7.87e-3`.  We sample in
    // [4e-3, 1.2e-2] to keep the signed accumulator within ±2^23
    // for K up to ~16k (avoids spurious saturation on the int8
    // matmul output that would otherwise mask perf regressions
    // behind the dst clamp).
    uint32_t state = seed;
    for (size_t i = 0; i < elems; ++i) {
        state = state * 1103515245u + 12345u;
        const float u = static_cast<float>((state >> 16) & 0xFFFF) / 65535.0f;
        buf[i] = 4.0e-3f + u * 8.0e-3f;  // [4e-3, 12e-3]
    }
}

} // namespace grp_matmul
} // namespace benchdnn
} // namespace zendnnl
