/********************************************************************************
# * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# *******************************************************************************/

#ifndef LOWOHA_SDPA_FLASH_CPU_HPP
#define LOWOHA_SDPA_FLASH_CPU_HPP

#include <cstdint>

#include "lowoha_operators/sdpa/lowoha_sdpa_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace sdpa {

/// Layout [Batch, Head, Seq, Dim] (PyTorch-style BHSD), or a packed
/// variable-length tensor (see sdpa_flash_cpu_varlen_run_internal).
/// Q/output use query heads; K/V may use fewer heads for GQA/MQA.
struct sdpa_flash_cpu_tensor_view {
    const void *data;
    int64_t stride_b;
    int64_t stride_h;
    int64_t stride_s;
    int64_t stride_d;
    int64_t size_b;
    int64_t size_h;
    int64_t size_s;
    int64_t size_d;
};

/// Optional mask: 2D [S_q, S_kv] or 4D [B, H, S_q, S_kv] with broadcast strides.
struct sdpa_flash_cpu_mask_view {
    const void *data;
    int ndim;
    int64_t sizes[4];
    int64_t strides[4];
};

/// Eagerly free the thread-local floating-point and INT8 flash scratch buffers.
/// The buffers are also freed automatically when the thread exits (RAII). The
/// custom-kernel GEMMs keep per-worker buffers until their worker thread exits:
/// the BF16 packed B block (`ceil(K/2) * 128` bytes, 32 KB for 512-key tiles),
/// and the INT8 packed B block (`ceil(K/4) * 128` bytes, 16 KB for 512-key
/// tiles) and U8 copy of the Q tile (rows * head_dim bytes, 128 KB for 512 x
/// 256).
void sdpa_flash_cpu_free_scratch();

status_t sdpa_flash_cpu_run_internal(const sdpa_flash_cpu_tensor_view &output,
        const sdpa_flash_cpu_tensor_view &query,
        const sdpa_flash_cpu_tensor_view &key,
        const sdpa_flash_cpu_tensor_view &value, double dropout_p,
        bool is_causal, bool sliding_window, int64_t sliding_window_size,
        const sdpa_flash_cpu_mask_view *mask, const double *scale_opt,
        data_type_t qkv_dt, data_type_t mask_dtype, bool is_qk_quant,
        bool is_pv_quant, int num_threads = 0);

/// Packed variable-length flash attention. Each view describes a packed
/// [total_tokens, heads, head_dim] tensor: size_b is the number of sequences,
/// size_s the longest sequence, stride_s / stride_h the token and head
/// strides; stride_b is unused. Sequence b spans rows
/// [cu_seqlens_q[b], cu_seqlens_q[b + 1]) of query / output and
/// [cu_seqlens_k[b], cu_seqlens_k[b + 1]) of key / value. Queries are aligned
/// to the end of their keys for the causal and window bands.
status_t sdpa_flash_cpu_varlen_run_internal(
        const sdpa_flash_cpu_tensor_view &output,
        const sdpa_flash_cpu_tensor_view &query,
        const sdpa_flash_cpu_tensor_view &key,
        const sdpa_flash_cpu_tensor_view &value, const int32_t *cu_seqlens_q,
        const int32_t *cu_seqlens_k, bool is_causal, bool sliding_window,
        int64_t sliding_window_size, const double *scale_opt,
        data_type_t qkv_dt, bool is_qk_quant, bool is_pv_quant,
        int num_threads = 0);

} // namespace sdpa
} // namespace lowoha
} // namespace zendnnl

#endif // LOWOHA_SDPA_FLASH_CPU_HPP
