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
        cfg.M_per_op = parse_M(m_str, cfg.num_ops);
    } catch (...) {
        return false;
    }
    return cfg.num_ops > 0 && cfg.K > 0 && cfg.N > 0 && cfg.max_M() > 0;
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
    } else {
        std::memset(buf, 0, elems * size_of(dt));
    }
}

} // namespace grp_matmul
} // namespace benchdnn
} // namespace zendnnl
