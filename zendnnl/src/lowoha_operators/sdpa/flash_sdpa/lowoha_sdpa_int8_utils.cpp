/********************************************************************************
# * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# *******************************************************************************/

#include "lowoha_sdpa_int8_utils.hpp"

#include <vector>

#include "lowoha_operators/matmul/lowoha_matmul.hpp"
#include "lowoha_operators/reorder/lowoha_reorder.hpp"

namespace zendnnl {
namespace lowoha {
namespace sdpa {

namespace {

matmul::matmul_params base_int8_matmul_params() {
    matmul::matmul_params params;
    params.dtypes.bias = data_type_t::none;
    params.dtypes.compute = data_type_t::none;
    // QK is promoted to AOCL-DLP's symmetric-quant blocked variant by
    // kernel_select.
    params.lowoha_algo = zendnnl::ops::matmul_algo_t::aocl_dlp;
    params.num_threads = 1;
    // Q/K buffers are call-local and their addresses may be reused with new
    // contents on the next SDPA invocation. Never retain a weight-cache entry.
    params.weight_cache_type = 0;
    return params;
}

// Shared body for the bf16 -> s8 grouped quantizers. Only the scale
// granularity differs between the Q/K and V call sites; keeping the
// reorder types confined here means the SDPA headers stay free of them.
status_t group_quantize_bf16_s8(const std::vector<const void *> &src,
        const std::vector<int> &rows, const std::vector<int> &cols,
        const std::vector<std::vector<int64_t>> &src_strides,
        const std::vector<void *> &dst, const std::vector<void *> &scales,
        int num_threads,
        reorder::group_dynamic_quant_granularity_t granularity) {
    reorder::group_dynamic_quant_params_t params;
    params.src_dtype = data_type_t::bf16;
    params.dst_dtype = data_type_t::s8;
    params.scale_dtype = data_type_t::f32;
    params.num_threads = num_threads;
    params.num_groups = 0;
    params.granularity = granularity;

    const std::vector<std::vector<int64_t>> dst_strides;
    return reorder::group_dynamic_quant(
            src, rows, cols, src_strides, dst, dst_strides, scales, params);
}

} // namespace

status_t sdpa_group_quantize_bf16_s8_per_token(
        const std::vector<const void *> &src, const std::vector<int> &rows,
        const std::vector<int> &cols,
        const std::vector<std::vector<int64_t>> &src_strides,
        const std::vector<void *> &dst, const std::vector<void *> &scales,
        int num_threads) {
    return group_quantize_bf16_s8(src, rows, cols, src_strides, dst, scales,
            num_threads, reorder::group_dynamic_quant_granularity_t::per_token);
}

status_t sdpa_group_quantize_bf16_s8_per_channel(
        const std::vector<const void *> &src, const std::vector<int> &rows,
        const std::vector<int> &cols,
        const std::vector<std::vector<int64_t>> &src_strides,
        const std::vector<void *> &dst, const std::vector<void *> &scales,
        int num_threads) {
    return group_quantize_bf16_s8(src, rows, cols, src_strides, dst, scales,
            num_threads,
            reorder::group_dynamic_quant_granularity_t::per_channel);
}

status_t sdpa_qk_int8_matmul(const int8_t *query, const int8_t *key,
        float *scores, int M, int N, int K, const float *query_scales,
        const float *key_scales) {
    matmul::matmul_params params = base_int8_matmul_params();
    params.dtypes.src = data_type_t::s8;
    params.dtypes.wei = data_type_t::s8;
    params.dtypes.dst = data_type_t::f32;
    params.quant_params.src_scale.buff = query_scales;
    params.quant_params.src_scale.dt = data_type_t::f32;
    params.quant_params.src_scale.dims = {M, 1};
    params.quant_params.wei_scale.buff = key_scales;
    params.quant_params.wei_scale.dt = data_type_t::f32;
    params.quant_params.wei_scale.dims = {1, N};

    matmul::matmul_batch_params_t batch_params;
    return matmul::matmul_direct('r', false, true, M, N, K, 1.0f, query, K, key,
            K, nullptr, 0.0f, scores, N,
            /*is_weights_const=*/false, batch_params, params);
}

status_t sdpa_pv_int8_matmul(const uint8_t *probability, const int8_t *value,
        float *output, int M, int N, int K, const float *probability_scale,
        const float *value_scales) {
    static const int32_t probability_zero_point = 0;

    matmul::matmul_params params = base_int8_matmul_params();
    params.dtypes.src = data_type_t::u8;
    params.dtypes.wei = data_type_t::s8;
    params.dtypes.dst = data_type_t::f32;
    params.quant_params.src_scale.buff = probability_scale;
    params.quant_params.src_scale.dt = data_type_t::f32;
    params.quant_params.src_scale.dims = {1, 1};
    params.quant_params.src_zp.buff = &probability_zero_point;
    params.quant_params.src_zp.dt = data_type_t::s32;
    params.quant_params.src_zp.dims = {1, 1};
    params.quant_params.wei_scale.buff = value_scales;
    params.quant_params.wei_scale.dt = data_type_t::f32;
    params.quant_params.wei_scale.dims = {1, N};

    matmul::matmul_batch_params_t batch_params;
    return matmul::matmul_direct('r', false, false, M, N, K, 1.0f, probability,
            K, value, N, nullptr, 0.0f, output, N,
            /*is_weights_const=*/false, batch_params, params);
}

} // namespace sdpa
} // namespace lowoha
} // namespace zendnnl
