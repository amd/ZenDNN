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

#include "lowoha_flash_sdpa.hpp"
#include "lowoha_operators/common/operator_instrumentation.hpp"

#include <algorithm>

namespace zendnnl {
namespace lowoha {
namespace sdpa {

namespace {

sdpa_flash_cpu_tensor_view build_tensor_view(const void *data,
        const sdpa_params &p, int64_t sb, int64_t sh, int64_t ss, int64_t sd,
        int64_t num_heads, int64_t seq_len) {
    sdpa_flash_cpu_tensor_view v {};
    v.data = data;
    v.size_b = p.batch;
    v.size_h = num_heads;
    v.size_s = seq_len;
    v.size_d = p.head_dim;
    v.stride_b = sb;
    v.stride_h = sh;
    v.stride_s = ss;
    v.stride_d = sd;
    return v;
}

} // namespace

status_t flash_sdpa(const void *query, const void *key, const void *value,
        const void *attn_mask, void *output, const sdpa_params &params) {

    // Validate inputs unless ZENDNNL_DIAGNOSTICS_ENABLE=0 is set (the gate
    // defaults to enabled). In production hot paths where the variable is
    // explicitly disabled, this resolves to a single predicted-taken branch,
    // skipping the full validation path (null-pointer checks, dimension
    // checks, and quantization-parameter validation).
    status_t status = zendnnl::common::op_instrumentation::validate([&]() {
        return validate_flash_sdpa_inputs(
                query, key, value, output, attn_mask, params);
    });
    if (status != status_t::success) { return status; }

    const int64_t eff_kv_seq_len
            = (params.kv_seq_len > 0) ? params.kv_seq_len : params.seq_len;
    const int64_t eff_kv_num_heads = (params.kv_num_heads > 0)
            ? params.kv_num_heads
            : params.num_heads;

    sdpa_flash_cpu_tensor_view qv = build_tensor_view(query, params,
            params.q_stride_b, params.q_stride_h, params.q_stride_s,
            params.q_stride_d, params.num_heads, params.seq_len);
    sdpa_flash_cpu_tensor_view kv = build_tensor_view(key, params,
            params.k_stride_b, params.k_stride_h, params.k_stride_s,
            params.k_stride_d, eff_kv_num_heads, eff_kv_seq_len);
    sdpa_flash_cpu_tensor_view vv = build_tensor_view(value, params,
            params.v_stride_b, params.v_stride_h, params.v_stride_s,
            params.v_stride_d, eff_kv_num_heads, eff_kv_seq_len);
    sdpa_flash_cpu_tensor_view ov = build_tensor_view(output, params,
            params.o_stride_b, params.o_stride_h, params.o_stride_s,
            params.o_stride_d, params.num_heads, params.seq_len);

    const sdpa_flash_cpu_mask_view *mask_ptr = nullptr;
    sdpa_flash_cpu_mask_view mask_view {};
    if (attn_mask != nullptr && params.mask_ndims > 0) {
        mask_view.data = attn_mask;
        mask_view.ndim = params.mask_ndims;
        for (int i = 0; i < 4; ++i) {
            mask_view.sizes[i] = params.mask_sizes[i];
            mask_view.strides[i] = params.mask_strides[i];
        }
        mask_ptr = &mask_view;
    }

    double scale_val = params.scale;
    const double *scale_opt = (scale_val != 0.0) ? &scale_val : nullptr;

    return sdpa_flash_cpu_run_internal(ov, qv, kv, vv, params.dropout_p,
            params.is_causal, params.sliding_window, params.sliding_window_size,
            mask_ptr, scale_opt, params.qkv_dt, params.mask_dt,
            params.is_qk_quant, params.is_pv_quant, params.num_threads);
}

status_t flash_sdpa_varlen(const void *query, const void *key,
        const void *value, const void *attn_mask, void *output,
        const sdpa_params &params) {
    status_t status = zendnnl::common::op_instrumentation::validate([&]() {
        return validate_flash_sdpa_varlen_inputs(
                query, key, value, output, attn_mask, params);
    });
    if (status != status_t::success) { return status; }

    const int32_t *cu_q = params.cu_seqlens_q;
    const int32_t *cu_k
            = (params.cu_seqlens_k != nullptr) ? params.cu_seqlens_k : cu_q;
    const int64_t num_seqs = params.batch;

    // Tiling and scratch follow the longest sequence actually present, so a
    // loose seq_len / kv_seq_len bound costs nothing.
    int64_t max_q = 0;
    int64_t max_kv = 0;
    for (int64_t b = 0; b < num_seqs; ++b) {
        max_q = std::max<int64_t>(max_q, cu_q[b + 1] - cu_q[b]);
        max_kv = std::max<int64_t>(max_kv, cu_k[b + 1] - cu_k[b]);
    }

    const int64_t eff_kv_num_heads = (params.kv_num_heads > 0)
            ? params.kv_num_heads
            : params.num_heads;

    sdpa_flash_cpu_tensor_view qv = build_tensor_view(query, params, 0,
            params.q_stride_h, params.q_stride_s, params.q_stride_d,
            params.num_heads, max_q);
    sdpa_flash_cpu_tensor_view kv = build_tensor_view(key, params, 0,
            params.k_stride_h, params.k_stride_s, params.k_stride_d,
            eff_kv_num_heads, max_kv);
    sdpa_flash_cpu_tensor_view vv = build_tensor_view(value, params, 0,
            params.v_stride_h, params.v_stride_s, params.v_stride_d,
            eff_kv_num_heads, max_kv);
    sdpa_flash_cpu_tensor_view ov = build_tensor_view(output, params, 0,
            params.o_stride_h, params.o_stride_s, params.o_stride_d,
            params.num_heads, max_q);

    double scale_val = params.scale;
    const double *scale_opt = (scale_val != 0.0) ? &scale_val : nullptr;

    return sdpa_flash_cpu_varlen_run_internal(ov, qv, kv, vv, cu_q, cu_k,
            params.is_causal, params.sliding_window, params.sliding_window_size,
            scale_opt, params.qkv_dt, params.is_qk_quant, params.is_pv_quant,
            params.num_threads);
}

} // namespace sdpa
} // namespace lowoha
} // namespace zendnnl
