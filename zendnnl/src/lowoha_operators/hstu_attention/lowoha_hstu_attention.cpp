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

#include "lowoha_operators/hstu_attention/lowoha_hstu_attention.hpp"

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <sstream>
#include "common/logging.hpp"
#include "common/zendnnl_global.hpp"
#include "lowoha_operators/hstu_attention/common/hstu_attention_pad.hpp"
#include "lowoha_operators/hstu_attention/reference/hstu_attention_reference.hpp"
#include "lowoha_operators/hstu_attention/tiled/hstu_attention_tiled.hpp"

namespace zendnnl {
namespace lowoha {
namespace hstu_attention {

using namespace zendnnl::common;
using namespace zendnnl::error_handling;

namespace {

bool is_float_dt(data_type_t dt) {
    return dt == data_type_t::f32 || dt == data_type_t::bf16;
}

status_t invalid(const char *what) {
    log_error("hstu_attention_direct: ", what);
    return status_t::failure;
}

/// Whether rows of `dim` elements at t * st + h * sh (t < rows, h < heads)
/// overlap; the smaller stride is the inner one.
bool rows_overlap(
        int64_t st, int64_t sh, int64_t rows, int64_t heads, int64_t dim) {
    const bool token_inner = st < sh;
    const int64_t inner = token_inner ? st : sh;
    const int64_t n_inner = token_inner ? rows : heads;
    const int64_t outer = token_inner ? sh : st;
    const int64_t n_outer = token_inner ? heads : rows;
    if (n_inner > 1 && inner < dim) return true;
    return n_outer > 1 && outer < (n_inner - 1) * inner + dim;
}

/// O(num_seqs) argument checks. They guard every pointer the kernels form, so
/// they run unconditionally (not behind ZENDNNL_DIAGNOSTICS_ENABLE).
status_t validate(const void *q, const void *k, const void *v, const void *out,
        const hstu_attention_params &p) {
    if (!q || !k || !v || !out || !p.seq_offsets)
        return invalid("null tensor or seq_offsets pointer");
    if (p.num_seqs < 0) return invalid("num_seqs must be >= 0");
    if (p.num_heads <= 0 || p.qk_dim <= 0 || p.v_dim <= 0 || p.max_seq_len <= 0)
        return invalid("num_heads, qk_dim, v_dim and max_seq_len must be > 0");
    if (!is_float_dt(p.qkv_dt) || !is_float_dt(p.out_dt))
        return invalid("q/k/v and output must be f32 or bf16");
    if (p.q_stride_t <= 0 || p.q_stride_h <= 0 || p.k_stride_t <= 0
            || p.k_stride_h <= 0 || p.v_stride_t <= 0 || p.v_stride_h <= 0
            || p.o_stride_t <= 0 || p.o_stride_h <= 0)
        return invalid("strides must be > 0");
    if (p.delta_size < 0 || p.max_attn_len < 0 || p.contextual_seq_len < 0
            || p.min_full_attn_seq_len < 0 || p.num_threads < 0)
        return invalid(
                "delta_size, max_attn_len, contextual_seq_len, "
                "min_full_attn_seq_len and num_threads must be >= 0");
    if (p.delta_size > 0 && p.attn_scale_per_token)
        return invalid("attn_scale_per_token is supported in full mode only");
    if (p.seq_offsets[0] < 0) return invalid("seq_offsets[0] must be >= 0");
    for (int64_t b = 0; b < p.num_seqs; ++b) {
        const int64_t len = p.seq_offsets[b + 1] - p.seq_offsets[b];
        if (len < 0 || len > p.max_seq_len)
            return invalid("sequence lengths must be in [0, max_seq_len]");
        if (len < p.delta_size)
            return invalid("sequences must be at least delta_size long");
        if (p.num_targets && (p.num_targets[b] < 0 || p.num_targets[b] > len))
            return invalid("num_targets must be in [0, sequence length]");
    }
    const int64_t out_rows = p.delta_size > 0 ? p.num_seqs * p.delta_size
                                              : p.seq_offsets[p.num_seqs];
    if (rows_overlap(
                p.o_stride_t, p.o_stride_h, out_rows, p.num_heads, p.v_dim))
        return invalid("output rows overlap (check o_stride_t / o_stride_h)");
    return status_t::success;
}

hstu_attention_kernel_t requested_kernel(const hstu_attention_params &p) {
    static const hstu_attention_kernel_t env = [] {
        const char *s = std::getenv("ZENDNNL_HSTU_ATTENTION_KERNEL");
        if (!s || !*s) return hstu_attention_kernel_t::none;
        if (std::strcmp(s, "tiled") == 0) return hstu_attention_kernel_t::tiled;
        if (std::strcmp(s, "reference") == 0)
            return hstu_attention_kernel_t::reference;
        log_warning(
                "hstu_attention_direct: ignoring "
                "ZENDNNL_HSTU_ATTENTION_KERNEL=",
                s, " (expected tiled or reference)");
        return hstu_attention_kernel_t::none;
    }();
    return env != hstu_attention_kernel_t::none ? env : p.kernel;
}

/// ZENDNNL_HSTU_ATTENTION_FAST_SILU (off by default), read once.
bool fast_silu_requested() {
    static const bool on = [] {
        const char *s = std::getenv("ZENDNNL_HSTU_ATTENTION_FAST_SILU");
        const int v = parse_fast_silu_env(s);
        if (v < 0 && s && *s)
            log_warning(
                    "hstu_attention_direct: ignoring "
                    "ZENDNNL_HSTU_ATTENTION_FAST_SILU=",
                    s, " (expected 0/false/off/no or 1/true/on/yes)");
        return v == 1;
    }();
    return on;
}

/// is_varlen in effect: ZENDNNL_HSTU_ATTENTION_VARLEN (read once) overrides
/// the params field; padding needs full mode.
bool varlen_in_effect(const hstu_attention_params &p) {
    static const int env = [] {
        const char *s = std::getenv("ZENDNNL_HSTU_ATTENTION_VARLEN");
        const int v = parse_varlen_env(s);
        if (v < 0 && s && *s)
            log_warning(
                    "hstu_attention_direct: ignoring "
                    "ZENDNNL_HSTU_ATTENTION_VARLEN=",
                    s, " (expected 0/false/off/no or 1/true/on/yes)");
        return v;
    }();
    const bool varlen = env < 0 ? p.is_varlen : env == 1;
    if (!varlen && !hstu_attention_pad_supported(p)) {
        static std::once_flag warned;
        std::call_once(warned, [] {
            log_warning(
                    "hstu_attention_direct: is_varlen = false is ignored in "
                    "delta mode");
        });
        return true;
    }
    return varlen;
}

} // namespace

status_t hstu_attention_direct(const void *query, const void *key,
        const void *value, void *output, const hstu_attention_params &params) {
    zendnnl::profile::profiler_t profiler;
    const bool is_profile = is_profile_enabled();
    if (is_profile) profiler.tbp_start();

    status_t st = validate(query, key, value, output, params);
    if (st != status_t::success) return st;
    if (params.num_seqs == 0) return status_t::success;

    hstu_attention_kernel_t kernel = requested_kernel(params);
    const bool tiled_ok = hstu_attention_tiled_supported(params);
    if (kernel == hstu_attention_kernel_t::none) {
        kernel = tiled_ok ? hstu_attention_kernel_t::tiled
                          : hstu_attention_kernel_t::reference;
        if (!tiled_ok) {
            static std::once_flag warned;
            std::call_once(warned, [] {
                log_warning(
                        "hstu_attention_direct: using the scalar reference "
                        "kernel (the tiled kernel needs a causal mask, "
                        "bf16 q/k/v and AVX512-BF16); it is 1-2 orders of "
                        "magnitude slower");
            });
        }
    }

    hstu_attention_run_t run;
    switch (kernel) {
        case hstu_attention_kernel_t::tiled:
            if (!tiled_ok) {
                log_error(
                        "hstu_attention_direct: the tiled kernel needs a "
                        "causal mask, bf16 q/k/v and AVX512-BF16");
                return status_t::unimplemented;
            }
            run = [](const void *q, const void *k, const void *v, void *o,
                          const hstu_attention_params &p) {
                return hstu_attention_tiled(
                        q, k, v, o, p, fast_silu_requested());
            };
            break;
        case hstu_attention_kernel_t::reference:
            run = hstu_attention_reference;
            break;
        default:
            log_error("hstu_attention_direct: unsupported kernel ",
                    kernel_to_string(kernel));
            return status_t::failure;
    }
    const bool varlen = varlen_in_effect(params);
    try {
        st = varlen
                ? run(query, key, value, output, params)
                : hstu_attention_padded(query, key, value, output, params, run);
    } catch (const std::bad_alloc &) {
        log_error("hstu_attention_direct: out of memory for scratch buffers");
        return status_t::failure;
    }
    if (st != status_t::success) return st;

    if (is_profile) profiler.tbp_stop();
    if (apilog_info_enabled() || is_profile) {
        std::ostringstream ss;
        ss << "LOWOHA hstu_attention_direct: num_seqs=" << params.num_seqs
           << ", total_len="
           << params.seq_offsets[params.num_seqs] - params.seq_offsets[0]
           << ", num_heads=" << params.num_heads << ", qk_dim=" << params.qk_dim
           << ", v_dim=" << params.v_dim
           << ", max_seq_len=" << params.max_seq_len
           << ", delta_size=" << params.delta_size
           << ", causal=" << (params.causal ? "true" : "false")
           << ", has_targets=" << (params.num_targets ? "true" : "false")
           << ", max_attn_len=" << params.max_attn_len
           << ", contextual_seq_len=" << params.contextual_seq_len
           << ", min_full_attn_seq_len=" << params.min_full_attn_seq_len
           << ", alpha=" << params.alpha
           << ", qkv_dt=" << dtype_info(params.qkv_dt)
           << ", out_dt=" << dtype_info(params.out_dt)
           << ", kernel=" << kernel_to_string(kernel);
        if (kernel == hstu_attention_kernel_t::tiled && fast_silu_requested())
            ss << ", fast_silu=true";
        if (!varlen) ss << ", varlen=false";
        apilog_info(ss.str());
        if (is_profile)
            profilelog_verbose(ss.str(), ", time=", profiler.tbp_elapsedtime(),
                    profiler.get_res_str());
    }
    return status_t::success;
}

} // namespace hstu_attention
} // namespace lowoha
} // namespace zendnnl
