/********************************************************************************
# * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# *******************************************************************************/

/// Kernel selection of hstu_attention_direct, as pure functions of the
/// request, the environment and the CPU, and the parsers of its environment
/// variables (internal; unit-tested).

#ifndef LOWOHA_HSTU_SELECT_HPP
#define LOWOHA_HSTU_SELECT_HPP

#include "common/config_params.hpp"
#include "lowoha_operators/hstu_attention/lowoha_hstu_attention_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace hstu_attention {

/// A boolean environment variable (ZENDNNL_HSTU_ATTENTION_INT8_QK / _PV,
/// _FAST_SILU, _VARLEN): -1 when `value` is unset, empty or not a boolean
/// (0/false/off/no, 1/true/on/yes), else 0 or 1.
inline int parse_bool_env(const char *value) {
    bool on = false;
    return zendnnl::common::parse_cache_bool(value, on) ? int(on) : -1;
}

/// A quant flag: the override when set, the params field otherwise.
inline bool resolve_int8_flag(int env, bool param) {
    return env < 0 ? param : env == 1;
}

/// The largest qk_dim INT8 Q*K^T supports: its int32 dot products reach
/// 127^2 * qk_dim.
constexpr int64_t kInt8MaxQkDim = 2147483647 / (127 * 127);
static_assert(127LL * 127 * kInt8MaxQkDim <= 2147483647
                && 127LL * 127 * (kInt8MaxQkDim + 1) > 2147483647,
        "kInt8MaxQkDim must be the largest qk_dim whose INT8 dot products "
        "fit int32");

/// The quant flags a call runs with, and why requested ones were dropped.
struct quant_flags_t {
    bool qk, pv;
    bool dropped_dtype; ///< q/k/v not bf16: both flags dropped
    bool dropped_qk_dim; ///< qk_dim > kInt8MaxQkDim: is_qk_quant dropped
};

/// is_qk_quant / is_pv_quant with the environment overrides `env_qk` /
/// `env_pv` (parse_bool_env), for q/k/v of `qkv_dt` and head size `qk_dim`.
inline quant_flags_t resolve_quant_flags(bool qk, bool pv, data_type_t qkv_dt,
        int64_t qk_dim, int env_qk, int env_pv) {
    quant_flags_t f {resolve_int8_flag(env_qk, qk),
            resolve_int8_flag(env_pv, pv), false, false};
    if ((f.qk || f.pv) && qkv_dt != data_type_t::bf16) {
        f.qk = f.pv = false;
        f.dropped_dtype = true;
    }
    if (f.qk && qk_dim > kInt8MaxQkDim) {
        f.qk = false;
        f.dropped_qk_dim = true;
    }
    return f;
}

/// The kernel a call runs, and whether the tiled kernel runs on INT8.
struct kernel_choice_t {
    hstu_attention_kernel_t kernel;
    bool int8;
};

/**
 * @brief The kernel of a call.
 *
 * `requested` (params or ZENDNNL_HSTU_ATTENTION_KERNEL) wins; `none` picks
 * `tiled` when it supports the call (`tiled_ok`) and `reference` otherwise.
 * The tiled kernel runs on INT8 when a quant flag is set (`quant`) and the
 * CPU has AVX512-VNNI (`vnni`), else on BF16. The reference kernel ignores
 * the flags.
 */
inline kernel_choice_t choose_kernel(hstu_attention_kernel_t requested,
        bool tiled_ok, bool quant, bool vnni) {
    kernel_choice_t c {requested, false};
    if (c.kernel == hstu_attention_kernel_t::none)
        c.kernel = tiled_ok ? hstu_attention_kernel_t::tiled
                            : hstu_attention_kernel_t::reference;
    c.int8 = c.kernel == hstu_attention_kernel_t::tiled && tiled_ok && quant
            && vnni;
    return c;
}

} // namespace hstu_attention
} // namespace lowoha
} // namespace zendnnl

#endif // LOWOHA_HSTU_SELECT_HPP
