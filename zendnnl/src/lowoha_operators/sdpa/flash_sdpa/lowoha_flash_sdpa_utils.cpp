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

#include "lowoha_flash_sdpa_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace sdpa {

status_t validate_flash_sdpa_inputs(const void *query, const void *key,
        const void *value, void *output, const void *attn_mask,
        const sdpa_params &params) {

    // --- pointer checks ---
    if (query == nullptr) {
        log_error("sdpa_flash_cpu: query pointer is null");
        return status_t::failure;
    }
    if (key == nullptr) {
        log_error("sdpa_flash_cpu: key pointer is null");
        return status_t::failure;
    }
    if (value == nullptr) {
        log_error("sdpa_flash_cpu: value pointer is null");
        return status_t::failure;
    }
    if (output == nullptr) {
        log_error("sdpa_flash_cpu: output pointer is null");
        return status_t::failure;
    }

    // --- dimension checks ---
    if (params.batch <= 0) {
        log_error("sdpa_flash_cpu: batch must be > 0");
        return status_t::failure;
    }
    if (params.num_heads <= 0) {
        log_error("sdpa_flash_cpu: num_heads must be > 0");
        return status_t::failure;
    }
    const int64_t eff_kv_num_heads = (params.kv_num_heads > 0)
            ? params.kv_num_heads
            : params.num_heads;
    if (params.kv_num_heads < 0) {
        log_error(
                "sdpa_flash_cpu: kv_num_heads must be >= 0 "
                "(0 means same as num_heads)");
        return status_t::failure;
    }
    if (eff_kv_num_heads <= 0) {
        log_error("sdpa_flash_cpu: effective kv_num_heads must be > 0");
        return status_t::failure;
    }
    if (eff_kv_num_heads > params.num_heads) {
        log_error("sdpa_flash_cpu: kv_num_heads must be <= num_heads");
        return status_t::failure;
    }
    if (params.num_heads % eff_kv_num_heads != 0) {
        log_error(
                "sdpa_flash_cpu: num_heads must be divisible by kv_num_heads");
        return status_t::failure;
    }
    if (params.seq_len <= 0) {
        log_error("sdpa_flash_cpu: seq_len must be > 0");
        return status_t::failure;
    }
    if (params.kv_seq_len < 0) {
        log_error(
                "sdpa_flash_cpu: kv_seq_len must be >= 0 "
                "(0 means same as seq_len)");
        return status_t::failure;
    }
    if (params.head_dim <= 0) {
        log_error("sdpa_flash_cpu: head_dim must be > 0");
        return status_t::failure;
    }

    // --- stride_d checks: Q/K/V must be contiguous along head_dim (GEMM requirement) ---
    if (params.q_stride_d != 1) {
        log_error("sdpa_flash_cpu: q_stride_d must be 1 (contiguous head_dim)");
        return status_t::failure;
    }
    if (params.k_stride_d != 1) {
        log_error("sdpa_flash_cpu: k_stride_d must be 1 (contiguous head_dim)");
        return status_t::failure;
    }
    if (params.v_stride_d != 1) {
        log_error("sdpa_flash_cpu: v_stride_d must be 1 (contiguous head_dim)");
        return status_t::failure;
    }

    // --- stride_s checks: seq stride is the GEMM leading dimension, must be positive ---
    if (params.q_stride_s <= 0) {
        log_error("sdpa_flash_cpu: q_stride_s must be > 0");
        return status_t::failure;
    }
    if (params.k_stride_s <= 0) {
        log_error("sdpa_flash_cpu: k_stride_s must be > 0");
        return status_t::failure;
    }
    if (params.v_stride_s <= 0) {
        log_error("sdpa_flash_cpu: v_stride_s must be > 0");
        return status_t::failure;
    }
    if (params.o_stride_s <= 0) {
        log_error("sdpa_flash_cpu: o_stride_s must be > 0");
        return status_t::failure;
    }
    if (params.o_stride_d <= 0) {
        log_error("sdpa_flash_cpu: o_stride_d must be > 0");
        return status_t::failure;
    }

    // --- batch/head stride checks for output (written in parallel, must not alias) ---
    if (params.batch > 1 && params.o_stride_b <= 0) {
        log_error("sdpa_flash_cpu: o_stride_b must be > 0 when batch > 1");
        return status_t::failure;
    }
    if (params.num_heads > 1 && params.o_stride_h <= 0) {
        log_error("sdpa_flash_cpu: o_stride_h must be > 0 when num_heads > 1");
        return status_t::failure;
    }

    // --- data type checks ---
    if (params.qkv_dt != data_type_t::f32 && params.qkv_dt != data_type_t::bf16
            && params.qkv_dt != data_type_t::f16) {
        log_error("sdpa_flash_cpu: qkv_dt must be f32, bf16, or f16");
        return status_t::failure;
    }
    const bool any_int8_quant = params.is_qk_quant || params.is_pv_quant;
    if (any_int8_quant && params.qkv_dt != data_type_t::bf16) {
        log_error(
                "sdpa_flash_cpu: dynamic INT8 quantization currently requires "
                "BF16 Q/K/V");
        return status_t::unimplemented;
    }
#if !ZENDNNL_DEPENDS_AOCLDLP
    if (any_int8_quant) {
        log_error(
                "sdpa_flash_cpu: dynamic INT8 quantization requires an "
                "AOCL-DLP enabled build");
        return status_t::unimplemented;
    }
#endif
    // VNNI is required by the s8xs8 / u8xs8 matmuls; F, BW and VL are
    // required by group_dynamic_quant (lowoha_reorder.cpp).
    if (any_int8_quant
            && (!zendnnl_platform_info().get_avx512f_status()
                    || !zendnnl_platform_info().get_avx512_bw_vl_status()
                    || !zendnnl_platform_info().get_avx512_vnni_status())) {
        log_error(
                "sdpa_flash_cpu: dynamic INT8 quantization requires "
                "AVX512-F, AVX512-BW, AVX512-VL, and AVX512-VNNI");
        return status_t::isa_unsupported;
    }

    // --- dropout (only 0 supported) ---
    if (params.dropout_p != 0.0) {
        log_error("sdpa_flash_cpu: dropout_p must be 0");
        return status_t::failure;
    }

    // --- sliding window ---
    if (params.sliding_window && params.sliding_window_size <= 0) {
        log_error(
                "sdpa_flash_cpu: sliding_window_size must be > 0 when "
                "sliding_window is enabled");
        return status_t::failure;
    }

    // --- mask checks ---
    if (params.mask_ndims != 0 && params.mask_ndims != 2
            && params.mask_ndims != 4) {
        log_error("sdpa_flash_cpu: mask_ndims must be 0, 2, or 4");
        return status_t::failure;
    }
    if (params.mask_ndims > 0 || attn_mask != nullptr) {
        if (params.mask_ndims == 0 && attn_mask != nullptr) {
            log_error(
                    "sdpa_flash_cpu: attn_mask is non-null but mask_ndims is "
                    "0; "
                    "set mask_ndims to 2 or 4");
            return status_t::failure;
        }
        if (params.mask_ndims > 0 && attn_mask == nullptr) {
            log_error("sdpa_flash_cpu: attn_mask is null but mask_ndims is ",
                    params.mask_ndims);
            return status_t::failure;
        }
        if (params.qkv_dt == data_type_t::bf16
                && params.mask_dt != data_type_t::f32
                && params.mask_dt != data_type_t::bf16) {
            log_error(
                    "sdpa_flash_cpu: mask_dt must be f32 or bf16 when qkv_dt "
                    "is bf16 and mask is "
                    "provided");
            return status_t::failure;
        } else if (params.qkv_dt == data_type_t::f16
                && params.mask_dt != data_type_t::f32
                && params.mask_dt != data_type_t::f16) {
            log_error(
                    "sdpa_flash_cpu: mask_dt must be f32 or f16 when qkv_dt is "
                    "f16 and mask is "
                    "provided");
            return status_t::failure;
        } else if (params.qkv_dt == data_type_t::f32
                && params.mask_dt != data_type_t::f32) {
            log_error(
                    "sdpa_flash_cpu: mask_dt must be f32 when qkv_dt is f32 "
                    "and mask is "
                    "provided");
            return status_t::failure;
        }
    }
    if (params.out_dt != data_type_t::none && params.out_dt != params.qkv_dt) {
        log_error("sdpa_flash_cpu: out_dt must be the same as qkv_dt");
        return status_t::failure;
    }

    // --- num_threads ---
    if (params.num_threads < 0) {
        log_error("sdpa_flash_cpu: num_threads must be >= 0");
        return status_t::failure;
    }

    return status_t::success;
}

namespace {

// Offsets must start at 0, never decrease, and keep every sequence within
// the declared maximum length.
bool check_varlen_offsets(const int32_t *cu_seqlens, int64_t num_seqs,
        int64_t max_len, const char *name) {
    if (cu_seqlens[0] != 0) {
        log_error("sdpa_flash_cpu_varlen: ", name, "[0] must be 0, got ",
                cu_seqlens[0]);
        return false;
    }
    for (int64_t b = 0; b < num_seqs; ++b) {
        const int64_t len = static_cast<int64_t>(cu_seqlens[b + 1])
                - static_cast<int64_t>(cu_seqlens[b]);
        if (len < 0) {
            log_error("sdpa_flash_cpu_varlen: ", name,
                    " must be non-decreasing (sequence ", b, ")");
            return false;
        }
        if (len > max_len) {
            log_error("sdpa_flash_cpu_varlen: sequence ", b, " of ", name,
                    " has ", len, " tokens, above the declared maximum ",
                    max_len);
            return false;
        }
    }
    return true;
}

} // namespace

status_t validate_flash_sdpa_varlen_inputs(const void *query, const void *key,
        const void *value, void *output, const void *attn_mask,
        const sdpa_params &params) {
    if (query == nullptr || key == nullptr || value == nullptr
            || output == nullptr) {
        log_error(
                "sdpa_flash_cpu_varlen: query, key, value and output "
                "pointers must be non-null");
        return status_t::failure;
    }
    if (params.cu_seqlens_q == nullptr) {
        log_error("sdpa_flash_cpu_varlen: cu_seqlens_q is null");
        return status_t::failure;
    }

    if (params.batch <= 0) {
        log_error(
                "sdpa_flash_cpu_varlen: batch (number of sequences) must be "
                "> 0");
        return status_t::failure;
    }
    if (params.num_heads <= 0) {
        log_error("sdpa_flash_cpu_varlen: num_heads must be > 0");
        return status_t::failure;
    }
    if (params.kv_num_heads < 0) {
        log_error(
                "sdpa_flash_cpu_varlen: kv_num_heads must be >= 0 "
                "(0 means same as num_heads)");
        return status_t::failure;
    }
    const int64_t eff_kv_num_heads = (params.kv_num_heads > 0)
            ? params.kv_num_heads
            : params.num_heads;
    if (eff_kv_num_heads > params.num_heads
            || params.num_heads % eff_kv_num_heads != 0) {
        log_error(
                "sdpa_flash_cpu_varlen: num_heads must be a multiple of "
                "kv_num_heads");
        return status_t::failure;
    }
    if (params.seq_len <= 0) {
        log_error(
                "sdpa_flash_cpu_varlen: seq_len (longest query sequence) "
                "must be > 0");
        return status_t::failure;
    }
    if (params.kv_seq_len < 0) {
        log_error(
                "sdpa_flash_cpu_varlen: kv_seq_len must be >= 0 "
                "(0 means same as seq_len)");
        return status_t::failure;
    }
    if (params.head_dim <= 0) {
        log_error("sdpa_flash_cpu_varlen: head_dim must be > 0");
        return status_t::failure;
    }

    // Head dim is the GEMM's contiguous axis and the token stride its
    // leading dimension; output rows are written in parallel.
    if (params.q_stride_d != 1 || params.k_stride_d != 1
            || params.v_stride_d != 1) {
        log_error(
                "sdpa_flash_cpu_varlen: q/k/v_stride_d must be 1 "
                "(contiguous head_dim)");
        return status_t::failure;
    }
    if (params.q_stride_s <= 0 || params.k_stride_s <= 0
            || params.v_stride_s <= 0 || params.o_stride_s <= 0) {
        log_error(
                "sdpa_flash_cpu_varlen: q/k/v/o_stride_s (token stride) "
                "must be > 0");
        return status_t::failure;
    }
    if (params.o_stride_d <= 0) {
        log_error("sdpa_flash_cpu_varlen: o_stride_d must be > 0");
        return status_t::failure;
    }
    if (params.num_heads > 1 && params.o_stride_h <= 0) {
        log_error(
                "sdpa_flash_cpu_varlen: o_stride_h must be > 0 when "
                "num_heads > 1");
        return status_t::failure;
    }

    if (attn_mask != nullptr || params.mask_ndims != 0) {
        log_error(
                "sdpa_flash_cpu_varlen: attn_mask is not supported in "
                "packed variable-length mode");
        return status_t::unimplemented;
    }
    if (params.dropout_p != 0.0) {
        log_error("sdpa_flash_cpu_varlen: dropout_p must be 0");
        return status_t::failure;
    }
    if (params.sliding_window && params.sliding_window_size <= 0) {
        log_error(
                "sdpa_flash_cpu_varlen: sliding_window_size must be > 0 "
                "when sliding_window is enabled");
        return status_t::failure;
    }

    if (params.qkv_dt != data_type_t::f32 && params.qkv_dt != data_type_t::bf16
            && params.qkv_dt != data_type_t::f16) {
        log_error("sdpa_flash_cpu_varlen: qkv_dt must be f32, bf16, or f16");
        return status_t::failure;
    }
    if (params.out_dt != data_type_t::none && params.out_dt != params.qkv_dt) {
        log_error("sdpa_flash_cpu_varlen: out_dt must be the same as qkv_dt");
        return status_t::failure;
    }
    const bool any_int8_quant = params.is_qk_quant || params.is_pv_quant;
    if (any_int8_quant && params.qkv_dt != data_type_t::bf16) {
        log_error(
                "sdpa_flash_cpu_varlen: dynamic INT8 quantization requires "
                "BF16 Q/K/V");
        return status_t::unimplemented;
    }
#if !ZENDNNL_DEPENDS_AOCLDLP
    if (any_int8_quant) {
        log_error(
                "sdpa_flash_cpu_varlen: dynamic INT8 quantization requires "
                "an AOCL-DLP enabled build");
        return status_t::unimplemented;
    }
#endif
    if (any_int8_quant
            && (!zendnnl_platform_info().get_avx512f_status()
                    || !zendnnl_platform_info().get_avx512_bw_vl_status()
                    || !zendnnl_platform_info().get_avx512_vnni_status())) {
        log_error(
                "sdpa_flash_cpu_varlen: dynamic INT8 quantization requires "
                "AVX512-F, AVX512-BW, AVX512-VL, and AVX512-VNNI");
        return status_t::isa_unsupported;
    }
    if (params.num_threads < 0) {
        log_error("sdpa_flash_cpu_varlen: num_threads must be >= 0");
        return status_t::failure;
    }

    const int64_t eff_kv_seq_len
            = (params.kv_seq_len > 0) ? params.kv_seq_len : params.seq_len;
    if (!check_varlen_offsets(params.cu_seqlens_q, params.batch, params.seq_len,
                "cu_seqlens_q")) {
        return status_t::failure;
    }
    // A null K-offset array means self-attention and resolves to Q offsets.
    // Validate those effective K lengths against kv_seq_len as well, so a
    // nonzero K/V bound cannot contradict the shared offsets.
    const int32_t *effective_cu_seqlens_k = params.cu_seqlens_k != nullptr
            ? params.cu_seqlens_k
            : params.cu_seqlens_q;
    if (!check_varlen_offsets(effective_cu_seqlens_k, params.batch,
                eff_kv_seq_len, "cu_seqlens_k")) {
        return status_t::failure;
    }

    return status_t::success;
}

} // namespace sdpa
} // namespace lowoha
} // namespace zendnnl
