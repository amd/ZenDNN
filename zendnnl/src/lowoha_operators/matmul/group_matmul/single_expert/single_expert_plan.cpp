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

#include "single_expert_plan.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <sstream>

#include "../common/group_matmul_parallel_common.hpp"
#include "common/zendnnl_global.hpp"
#include "single_expert_policy.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

namespace {

bool parse_positive(const std::string &v, int *out) {
    if (v.empty()) return false;
    char *end = nullptr;
    const long n = std::strtol(v.c_str(), &end, 10);
    if (*end != '\0' || n < 0 || n > (1 << 20)) return false;
    *out = static_cast<int>(n);
    return true;
}

} // namespace

bool parse_single_expert_block(
        const char *value, se_block_knob_t *knob, std::string *error) {
    se_block_knob_t k;
    std::string s(value != nullptr ? value : "");
    for (char &c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (c == ',') c = ' ';
    }
    std::istringstream in(s);
    std::string tok;
    bool any = false;
    while (in >> tok) {
        any = true;
        const size_t eq = tok.find('=');
        const std::string key = tok.substr(0, eq);
        const std::string val
                = eq == std::string::npos ? "" : tok.substr(eq + 1);
        bool ok = true;
        if (key == "split") {
            if (val == "legacy")
                k.split = se_split_t::legacy;
            else if (val == "remainder")
                k.split = se_split_t::remainder;
            else if (val == "grid")
                k.split = se_split_t::grid;
            else
                ok = false;
        } else if (key == "mparts") {
            ok = parse_positive(val, &k.m_parts) && k.m_parts >= 1;
        } else if (key == "kparts") {
            ok = val == "auto"
                    ? (k.k_parts = 0, true)
                    : parse_positive(val, &k.k_parts) && k.k_parts >= 1;
        } else if (key == "kc") {
            ok = parse_positive(val, &k.kc);
        } else if (key == "mr") {
            ok = parse_positive(val, &k.mr) && k.mr <= 12;
        } else if (key == "order") {
            if (val == "n")
                k.order = se_order_t::n_outer;
            else if (val == "k")
                k.order = se_order_t::k_outer;
            else
                ok = false;
        } else {
            ok = false;
        }
        if (!ok) {
            if (error != nullptr) *error = tok;
            return false;
        }
    }
    k.given = any;
    *knob = k;
    return true;
}

const se_block_knob_t &get_single_expert_block_knob() {
    const se_block_knob_t *ovr = test_api::s_single_expert_block_override.load(
            std::memory_order_relaxed);
    if (ovr != nullptr) return *ovr;
    static const se_block_knob_t v = []() {
        se_block_knob_t k;
        const char *e = std::getenv("ZENDNNL_GRP_MATMUL_SINGLE_EXPERT_BLOCK");
        if (e == nullptr) return k;
        std::string error;
        if (!parse_single_expert_block(e, &k, &error)) {
            apilog_warning("ZENDNNL_GRP_MATMUL_SINGLE_EXPERT_BLOCK: ignoring '",
                    e, "' (bad token '", error, "'); keeping the legacy plan");
            return se_block_knob_t {};
        }
        return k;
    }();
    return v;
}

bool get_grp_matmul_single_expert_ksplit() {
    const int ovr = test_api::s_grp_matmul_single_expert_ksplit_override.load(
            std::memory_order_relaxed);
    if (ovr >= 0) return ovr != 0;
    static const bool v = []() {
        const char *e = std::getenv("ZENDNNL_GRP_MATMUL_SINGLE_EXPERT_KSPLIT");
        int parsed = 0;
        if (!parse_env_int_strict(e, parsed)) return false;
        return parsed != 0;
    }();
    return v;
}

bool se_block_plan_enabled() {
    const se_block_knob_t &knob = get_single_expert_block_knob();
    return knob.given ? knob.split != se_split_t::legacy
                      : get_grp_matmul_single_expert_ksplit();
}

se_block_plan_t se_block_plan(int num_ops, int M, int N, int K, int pack_nr,
        int k_step, int num_threads) {
    se_block_plan_t p;
    if (!is_dense_ffn_decode(num_ops, M) || M <= 0 || pack_nr <= 0
            || k_step <= 0 || num_threads < 2 || N % pack_nr != 0
            || K % k_step != 0)
        return p;
    const se_block_knob_t &knob = get_single_expert_block_knob();
    if (!se_block_plan_enabled()) return p;
    const bool explicit_plan = knob.given;
    const se_split_t split = explicit_plan ? knob.split : se_split_t::remainder;
    const int blocks = N / pack_nr;

    se_block_plan_t q;
    q.split = split;
    q.kc = knob.kc > 0 ? std::max(k_step, knob.kc / k_step * k_step) : 0;
    q.mr = knob.mr;
    q.order = knob.order;
    if (split == se_split_t::remainder) {
        const int rem = blocks % num_threads;
        if (rem == 0) return p;
        int parts = num_threads / rem;
        if (knob.k_parts > 0) parts = std::min(parts, knob.k_parts);
        const int min_k = explicit_plan ? k_step : kSeKsplitMinK;
        if (parts < 2 || K / parts < min_k) return p;
        q.whole_blocks = blocks / num_threads;
        q.split_blocks = rem;
        q.k_parts = parts;
        q.n_parts = num_threads;
        return q;
    }
    q.m_parts = std::min(knob.m_parts, M);
    q.k_parts = std::max(1, knob.k_parts);
    if (num_threads % (q.m_parts * q.k_parts) != 0) return p;
    if (K / q.k_parts < k_step) return p;
    q.n_parts = num_threads / (q.m_parts * q.k_parts);
    return q;
}

std::string se_block_plan_label(const se_block_plan_t &plan) {
    std::ostringstream ss;
    switch (plan.split) {
        case se_split_t::legacy: ss << "legacy"; break;
        case se_split_t::remainder:
            ss << "remainder whole" << plan.whole_blocks << " split"
               << plan.split_blocks << " k" << plan.k_parts;
            break;
        case se_split_t::grid:
            ss << "grid m" << plan.m_parts << " n" << plan.n_parts << " k"
               << plan.k_parts;
            break;
    }
    ss << " kc" << plan.kc << " mr" << plan.mr
       << " order=" << (plan.order == se_order_t::k_outer ? 'k' : 'n');
    return ss.str();
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
