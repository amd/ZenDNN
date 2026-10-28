/********************************************************************************
# * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef _LOWOHA_DISPATCH_KERNEL_HPP
#define _LOWOHA_DISPATCH_KERNEL_HPP

#include "common/op_config.hpp"
#include "lowoha_embag_common.hpp"
#include "lowoha_embag_ref_kernel.hpp"
#include "native_kernels/embag_avx2_kernels.hpp"
#include "native_kernels/embag_avx512_kernels.hpp"
#if ZENDNNL_DEPENDS_FBGEMM
#include "fbgemm_kernel.hpp"
#endif

namespace zendnnl {
namespace lowoha {
namespace embag {

using zendnnl::common::float16_t;

/**
 * @brief Dispatch to native AVX512 embedding bag kernel
 *
 * Dispatches to the appropriate native AVX512 kernel instantiation based on
 * indices, offsets, table, and output data types.
 *
 * @return status_t::success when a kernel runs, or status_t::unimplemented
 *         when the table/output or indices/offsets combination is unsupported.
 */
static status_t embag_native_kernel(const void *table, const void *indices,
        const void *offsets, const float *weights, void *dst,
        const embag_params_t &params) {

    const uint64_t embedding_dim = params.embedding_dim;
    const uint64_t num_indices = params.num_indices;
    const uint64_t num_bags = params.num_bags;
    const int64_t padding_idx = params.padding_idx;
    const bool include_last_offset = params.include_last_offset;
    const data_type_t table_dtype = params.dtypes.table;
    const bool fp16_scale_bias = params.fp16_scale_bias;

    // Use algo directly since lowoha::embag_algo_t is aliased to common::embag_algo_t
    const embag_algo_t algo = params.algo;

    const bool is_weights = params.is_weights;
    const int64_t dst_stride = params.dst_stride;

    // Dispatch based on indices/offsets types
    // For embedding lookup (algo == none), offsets is nullptr
    const bool is_offsets = (offsets != nullptr);

    if (params.dtypes.indices == data_type_t::s64
            && (!is_offsets || params.dtypes.offsets == data_type_t::s64)) {
        if (params.dtypes.table == data_type_t::f32
                && params.dtypes.output == data_type_t::f32) {
            embag_avx512_kernel<float, int64_t, int64_t, float>(
                    static_cast<const float *>(table), weights,
                    static_cast<const int64_t *>(indices),
                    static_cast<const int64_t *>(offsets),
                    static_cast<float *>(dst), embedding_dim, num_indices,
                    num_bags, padding_idx, is_weights, algo, dst_stride,
                    include_last_offset);
        } else if (params.dtypes.table == data_type_t::bf16
                && params.dtypes.output == data_type_t::bf16) {
#if __GNUC__ >= 12
            embag_avx512_kernel<uint16_t, int64_t, int64_t, uint16_t>(
                    static_cast<const uint16_t *>(table), weights,
                    static_cast<const int64_t *>(indices),
                    static_cast<const int64_t *>(offsets),
                    static_cast<uint16_t *>(dst), embedding_dim, num_indices,
                    num_bags, padding_idx, is_weights, algo, dst_stride,
                    include_last_offset);
#else
            embag_avx2_kernel<uint16_t, int64_t, int64_t, uint16_t>(
                    static_cast<const uint16_t *>(table), weights,
                    static_cast<const int64_t *>(indices),
                    static_cast<const int64_t *>(offsets),
                    static_cast<uint16_t *>(dst), embedding_dim, num_indices,
                    num_bags, padding_idx, is_weights, algo, dst_stride,
                    include_last_offset);
#endif
        } else if (params.dtypes.table == data_type_t::bf16
                && params.dtypes.output == data_type_t::f32) {
#if __GNUC__ >= 12
            embag_avx512_kernel<uint16_t, int64_t, int64_t, float>(
                    static_cast<const uint16_t *>(table), weights,
                    static_cast<const int64_t *>(indices),
                    static_cast<const int64_t *>(offsets),
                    static_cast<float *>(dst), embedding_dim, num_indices,
                    num_bags, padding_idx, is_weights, algo, dst_stride,
                    include_last_offset);
#else
            embag_avx2_kernel<uint16_t, int64_t, int64_t, float>(
                    static_cast<const uint16_t *>(table), weights,
                    static_cast<const int64_t *>(indices),
                    static_cast<const int64_t *>(offsets),
                    static_cast<float *>(dst), embedding_dim, num_indices,
                    num_bags, padding_idx, is_weights, algo, dst_stride,
                    include_last_offset);
#endif
        } else if (params.dtypes.table == data_type_t::f32
                && params.dtypes.output == data_type_t::bf16) {
            embag_avx512_kernel<float, int64_t, int64_t, uint16_t>(
                    static_cast<const float *>(table), weights,
                    static_cast<const int64_t *>(indices),
                    static_cast<const int64_t *>(offsets),
                    static_cast<uint16_t *>(dst), embedding_dim, num_indices,
                    num_bags, padding_idx, is_weights, algo, dst_stride,
                    include_last_offset);
        } else if (params.dtypes.table == data_type_t::f16
                && params.dtypes.output == data_type_t::f16) {
#if __GNUC__ >= 12
            if (can_use_f16_fma_kernel()) {
                embag_avx512_f16_fma_kernel<float16_t, int64_t, int64_t,
                        float16_t>(static_cast<const float16_t *>(table),
                        weights, static_cast<const int64_t *>(indices),
                        static_cast<const int64_t *>(offsets),
                        static_cast<float16_t *>(dst), embedding_dim,
                        num_indices, num_bags, padding_idx, is_weights, algo,
                        dst_stride, include_last_offset);
            } else
#endif
            {
                embag_avx512_kernel<float16_t, int64_t, int64_t, float16_t>(
                        static_cast<const float16_t *>(table), weights,
                        static_cast<const int64_t *>(indices),
                        static_cast<const int64_t *>(offsets),
                        static_cast<float16_t *>(dst), embedding_dim,
                        num_indices, num_bags, padding_idx, is_weights, algo,
                        dst_stride, include_last_offset);
            }
        } else if (params.dtypes.table == data_type_t::f16
                && params.dtypes.output == data_type_t::f32) {
#if __GNUC__ >= 12
            if (can_use_f16_fma_kernel()) {
                embag_avx512_f16_fma_kernel<float16_t, int64_t, int64_t, float>(
                        static_cast<const float16_t *>(table), weights,
                        static_cast<const int64_t *>(indices),
                        static_cast<const int64_t *>(offsets),
                        static_cast<float *>(dst), embedding_dim, num_indices,
                        num_bags, padding_idx, is_weights, algo, dst_stride,
                        include_last_offset);
            } else
#endif
            {
                embag_avx512_kernel<float16_t, int64_t, int64_t, float>(
                        static_cast<const float16_t *>(table), weights,
                        static_cast<const int64_t *>(indices),
                        static_cast<const int64_t *>(offsets),
                        static_cast<float *>(dst), embedding_dim, num_indices,
                        num_bags, padding_idx, is_weights, algo, dst_stride,
                        include_last_offset);
            }
        } else if (params.dtypes.table == data_type_t::f32
                && params.dtypes.output == data_type_t::f16) {
#if __GNUC__ >= 12
            if (can_use_f16_fma_kernel()) {
                embag_avx512_f16_fma_kernel<float, int64_t, int64_t, float16_t>(
                        static_cast<const float *>(table), weights,
                        static_cast<const int64_t *>(indices),
                        static_cast<const int64_t *>(offsets),
                        static_cast<float16_t *>(dst), embedding_dim,
                        num_indices, num_bags, padding_idx, is_weights, algo,
                        dst_stride, include_last_offset);
            } else
#endif
            {
                embag_avx512_kernel<float, int64_t, int64_t, float16_t>(
                        static_cast<const float *>(table), weights,
                        static_cast<const int64_t *>(indices),
                        static_cast<const int64_t *>(offsets),
                        static_cast<float16_t *>(dst), embedding_dim,
                        num_indices, num_bags, padding_idx, is_weights, algo,
                        dst_stride, include_last_offset);
            }
        } else if (params.dtypes.table == data_type_t::s8
                && params.dtypes.output == data_type_t::f32) {
            embag_avx512_int8_int4_kernel<false, int8_t, int64_t, int64_t,
                    float>(static_cast<const int8_t *>(table), weights,
                    static_cast<const int64_t *>(indices),
                    static_cast<const int64_t *>(offsets),
                    static_cast<float *>(dst), embedding_dim, num_indices,
                    num_bags, padding_idx, is_weights, algo, dst_stride,
                    include_last_offset, table_dtype, fp16_scale_bias);
        } else if (params.dtypes.table == data_type_t::s8
                && params.dtypes.output == data_type_t::bf16) {
            embag_avx512_int8_int4_kernel<false, int8_t, int64_t, int64_t,
                    uint16_t>(static_cast<const int8_t *>(table), weights,
                    static_cast<const int64_t *>(indices),
                    static_cast<const int64_t *>(offsets),
                    static_cast<uint16_t *>(dst), embedding_dim, num_indices,
                    num_bags, padding_idx, is_weights, algo, dst_stride,
                    include_last_offset, table_dtype, fp16_scale_bias);
        } else if ((params.dtypes.table == data_type_t::s4
                           || params.dtypes.table == data_type_t::u4)
                && params.dtypes.output == data_type_t::f32) {
            embag_avx512_int8_int4_kernel<true, uint8_t, int64_t, int64_t,
                    float>(static_cast<const uint8_t *>(table), weights,
                    static_cast<const int64_t *>(indices),
                    static_cast<const int64_t *>(offsets),
                    static_cast<float *>(dst), embedding_dim, num_indices,
                    num_bags, padding_idx, is_weights, algo, dst_stride,
                    include_last_offset, table_dtype, fp16_scale_bias);
        } else if ((params.dtypes.table == data_type_t::s4
                           || params.dtypes.table == data_type_t::u4)
                && params.dtypes.output == data_type_t::bf16) {
            embag_avx512_int8_int4_kernel<true, uint8_t, int64_t, int64_t,
                    uint16_t>(static_cast<const uint8_t *>(table), weights,
                    static_cast<const int64_t *>(indices),
                    static_cast<const int64_t *>(offsets),
                    static_cast<uint16_t *>(dst), embedding_dim, num_indices,
                    num_bags, padding_idx, is_weights, algo, dst_stride,
                    include_last_offset, table_dtype, fp16_scale_bias);
        } else if (params.dtypes.table == data_type_t::s8
                && params.dtypes.output == data_type_t::f16) {
#if __GNUC__ >= 12
            if (can_use_f16_fma_kernel()) {
                embag_avx512_int8_int4_f16_fma_kernel<false, int8_t, int64_t,
                        int64_t, float16_t>(static_cast<const int8_t *>(table),
                        weights, static_cast<const int64_t *>(indices),
                        static_cast<const int64_t *>(offsets),
                        static_cast<float16_t *>(dst), embedding_dim,
                        num_indices, num_bags, padding_idx, is_weights, algo,
                        dst_stride, include_last_offset, table_dtype,
                        fp16_scale_bias);
            } else
#endif
            {
                embag_avx512_int8_int4_kernel<false, int8_t, int64_t, int64_t,
                        float16_t>(static_cast<const int8_t *>(table), weights,
                        static_cast<const int64_t *>(indices),
                        static_cast<const int64_t *>(offsets),
                        static_cast<float16_t *>(dst), embedding_dim,
                        num_indices, num_bags, padding_idx, is_weights, algo,
                        dst_stride, include_last_offset, table_dtype,
                        fp16_scale_bias);
            }
        } else if ((params.dtypes.table == data_type_t::s4
                           || params.dtypes.table == data_type_t::u4)
                && params.dtypes.output == data_type_t::f16) {
#if __GNUC__ >= 12
            if (can_use_f16_fma_kernel()) {
                embag_avx512_int8_int4_f16_fma_kernel<true, uint8_t, int64_t,
                        int64_t, float16_t>(static_cast<const uint8_t *>(table),
                        weights, static_cast<const int64_t *>(indices),
                        static_cast<const int64_t *>(offsets),
                        static_cast<float16_t *>(dst), embedding_dim,
                        num_indices, num_bags, padding_idx, is_weights, algo,
                        dst_stride, include_last_offset, table_dtype,
                        fp16_scale_bias);
            } else
#endif
            {
                embag_avx512_int8_int4_kernel<true, uint8_t, int64_t, int64_t,
                        float16_t>(static_cast<const uint8_t *>(table), weights,
                        static_cast<const int64_t *>(indices),
                        static_cast<const int64_t *>(offsets),
                        static_cast<float16_t *>(dst), embedding_dim,
                        num_indices, num_bags, padding_idx, is_weights, algo,
                        dst_stride, include_last_offset, table_dtype,
                        fp16_scale_bias);
            }
        } else {
            log_error(
                    "embedding_bag_direct: unsupported table and output data "
                    "types");
            return status_t::unimplemented;
        }
    } else if (params.dtypes.indices == data_type_t::s32
            && (!is_offsets || params.dtypes.offsets == data_type_t::s32)) {
        if (params.dtypes.table == data_type_t::f32
                && params.dtypes.output == data_type_t::f32) {
            embag_avx512_kernel<float, int32_t, int32_t, float>(
                    static_cast<const float *>(table), weights,
                    static_cast<const int32_t *>(indices),
                    static_cast<const int32_t *>(offsets),
                    static_cast<float *>(dst), embedding_dim, num_indices,
                    num_bags, padding_idx, is_weights, algo, dst_stride,
                    include_last_offset);
        } else if (params.dtypes.table == data_type_t::bf16
                && params.dtypes.output == data_type_t::bf16) {
#if __GNUC__ >= 12
            embag_avx512_kernel<uint16_t, int32_t, int32_t, uint16_t>(
                    static_cast<const uint16_t *>(table), weights,
                    static_cast<const int32_t *>(indices),
                    static_cast<const int32_t *>(offsets),
                    static_cast<uint16_t *>(dst), embedding_dim, num_indices,
                    num_bags, padding_idx, is_weights, algo, dst_stride,
                    include_last_offset);
#else
            embag_avx2_kernel<uint16_t, int32_t, int32_t, uint16_t>(
                    static_cast<const uint16_t *>(table), weights,
                    static_cast<const int32_t *>(indices),
                    static_cast<const int32_t *>(offsets),
                    static_cast<uint16_t *>(dst), embedding_dim, num_indices,
                    num_bags, padding_idx, is_weights, algo, dst_stride,
                    include_last_offset);
#endif
        } else if (params.dtypes.table == data_type_t::bf16
                && params.dtypes.output == data_type_t::f32) {
#if __GNUC__ >= 12
            embag_avx512_kernel<uint16_t, int32_t, int32_t, float>(
                    static_cast<const uint16_t *>(table), weights,
                    static_cast<const int32_t *>(indices),
                    static_cast<const int32_t *>(offsets),
                    static_cast<float *>(dst), embedding_dim, num_indices,
                    num_bags, padding_idx, is_weights, algo, dst_stride,
                    include_last_offset);
#else
            embag_avx2_kernel<uint16_t, int32_t, int32_t, float>(
                    static_cast<const uint16_t *>(table), weights,
                    static_cast<const int32_t *>(indices),
                    static_cast<const int32_t *>(offsets),
                    static_cast<float *>(dst), embedding_dim, num_indices,
                    num_bags, padding_idx, is_weights, algo, dst_stride,
                    include_last_offset);
#endif
        } else if (params.dtypes.table == data_type_t::f32
                && params.dtypes.output == data_type_t::bf16) {
            embag_avx512_kernel<float, int32_t, int32_t, uint16_t>(
                    static_cast<const float *>(table), weights,
                    static_cast<const int32_t *>(indices),
                    static_cast<const int32_t *>(offsets),
                    static_cast<uint16_t *>(dst), embedding_dim, num_indices,
                    num_bags, padding_idx, is_weights, algo, dst_stride,
                    include_last_offset);
        } else if (params.dtypes.table == data_type_t::f16
                && params.dtypes.output == data_type_t::f16) {
#if __GNUC__ >= 12
            if (can_use_f16_fma_kernel()) {
                embag_avx512_f16_fma_kernel<float16_t, int32_t, int32_t,
                        float16_t>(static_cast<const float16_t *>(table),
                        weights, static_cast<const int32_t *>(indices),
                        static_cast<const int32_t *>(offsets),
                        static_cast<float16_t *>(dst), embedding_dim,
                        num_indices, num_bags, padding_idx, is_weights, algo,
                        dst_stride, include_last_offset);
            } else
#endif
            {
                embag_avx512_kernel<float16_t, int32_t, int32_t, float16_t>(
                        static_cast<const float16_t *>(table), weights,
                        static_cast<const int32_t *>(indices),
                        static_cast<const int32_t *>(offsets),
                        static_cast<float16_t *>(dst), embedding_dim,
                        num_indices, num_bags, padding_idx, is_weights, algo,
                        dst_stride, include_last_offset);
            }
        } else if (params.dtypes.table == data_type_t::f16
                && params.dtypes.output == data_type_t::f32) {
#if __GNUC__ >= 12
            if (can_use_f16_fma_kernel()) {
                embag_avx512_f16_fma_kernel<float16_t, int32_t, int32_t, float>(
                        static_cast<const float16_t *>(table), weights,
                        static_cast<const int32_t *>(indices),
                        static_cast<const int32_t *>(offsets),
                        static_cast<float *>(dst), embedding_dim, num_indices,
                        num_bags, padding_idx, is_weights, algo, dst_stride,
                        include_last_offset);
            } else
#endif
            {
                embag_avx512_kernel<float16_t, int32_t, int32_t, float>(
                        static_cast<const float16_t *>(table), weights,
                        static_cast<const int32_t *>(indices),
                        static_cast<const int32_t *>(offsets),
                        static_cast<float *>(dst), embedding_dim, num_indices,
                        num_bags, padding_idx, is_weights, algo, dst_stride,
                        include_last_offset);
            }
        } else if (params.dtypes.table == data_type_t::f32
                && params.dtypes.output == data_type_t::f16) {
#if __GNUC__ >= 12
            if (can_use_f16_fma_kernel()) {
                embag_avx512_f16_fma_kernel<float, int32_t, int32_t, float16_t>(
                        static_cast<const float *>(table), weights,
                        static_cast<const int32_t *>(indices),
                        static_cast<const int32_t *>(offsets),
                        static_cast<float16_t *>(dst), embedding_dim,
                        num_indices, num_bags, padding_idx, is_weights, algo,
                        dst_stride, include_last_offset);
            } else
#endif
            {
                embag_avx512_kernel<float, int32_t, int32_t, float16_t>(
                        static_cast<const float *>(table), weights,
                        static_cast<const int32_t *>(indices),
                        static_cast<const int32_t *>(offsets),
                        static_cast<float16_t *>(dst), embedding_dim,
                        num_indices, num_bags, padding_idx, is_weights, algo,
                        dst_stride, include_last_offset);
            }
        } else if (params.dtypes.table == data_type_t::s8
                && params.dtypes.output == data_type_t::f32) {
            embag_avx512_int8_int4_kernel<false, int8_t, int32_t, int32_t,
                    float>(static_cast<const int8_t *>(table), weights,
                    static_cast<const int32_t *>(indices),
                    static_cast<const int32_t *>(offsets),
                    static_cast<float *>(dst), embedding_dim, num_indices,
                    num_bags, padding_idx, is_weights, algo, dst_stride,
                    include_last_offset, table_dtype, fp16_scale_bias);
        } else if (params.dtypes.table == data_type_t::s8
                && params.dtypes.output == data_type_t::bf16) {
            embag_avx512_int8_int4_kernel<false, int8_t, int32_t, int32_t,
                    uint16_t>(static_cast<const int8_t *>(table), weights,
                    static_cast<const int32_t *>(indices),
                    static_cast<const int32_t *>(offsets),
                    static_cast<uint16_t *>(dst), embedding_dim, num_indices,
                    num_bags, padding_idx, is_weights, algo, dst_stride,
                    include_last_offset, table_dtype, fp16_scale_bias);
        } else if ((params.dtypes.table == data_type_t::s4
                           || params.dtypes.table == data_type_t::u4)
                && params.dtypes.output == data_type_t::f32) {
            embag_avx512_int8_int4_kernel<true, uint8_t, int32_t, int32_t,
                    float>(static_cast<const uint8_t *>(table), weights,
                    static_cast<const int32_t *>(indices),
                    static_cast<const int32_t *>(offsets),
                    static_cast<float *>(dst), embedding_dim, num_indices,
                    num_bags, padding_idx, is_weights, algo, dst_stride,
                    include_last_offset, table_dtype, fp16_scale_bias);
        } else if ((params.dtypes.table == data_type_t::s4
                           || params.dtypes.table == data_type_t::u4)
                && params.dtypes.output == data_type_t::bf16) {
            embag_avx512_int8_int4_kernel<true, uint8_t, int32_t, int32_t,
                    uint16_t>(static_cast<const uint8_t *>(table), weights,
                    static_cast<const int32_t *>(indices),
                    static_cast<const int32_t *>(offsets),
                    static_cast<uint16_t *>(dst), embedding_dim, num_indices,
                    num_bags, padding_idx, is_weights, algo, dst_stride,
                    include_last_offset, table_dtype, fp16_scale_bias);
        } else if (params.dtypes.table == data_type_t::s8
                && params.dtypes.output == data_type_t::f16) {
#if __GNUC__ >= 12
            if (can_use_f16_fma_kernel()) {
                embag_avx512_int8_int4_f16_fma_kernel<false, int8_t, int32_t,
                        int32_t, float16_t>(static_cast<const int8_t *>(table),
                        weights, static_cast<const int32_t *>(indices),
                        static_cast<const int32_t *>(offsets),
                        static_cast<float16_t *>(dst), embedding_dim,
                        num_indices, num_bags, padding_idx, is_weights, algo,
                        dst_stride, include_last_offset, table_dtype,
                        fp16_scale_bias);
            } else
#endif
            {
                embag_avx512_int8_int4_kernel<false, int8_t, int32_t, int32_t,
                        float16_t>(static_cast<const int8_t *>(table), weights,
                        static_cast<const int32_t *>(indices),
                        static_cast<const int32_t *>(offsets),
                        static_cast<float16_t *>(dst), embedding_dim,
                        num_indices, num_bags, padding_idx, is_weights, algo,
                        dst_stride, include_last_offset, table_dtype,
                        fp16_scale_bias);
            }
        } else if ((params.dtypes.table == data_type_t::s4
                           || params.dtypes.table == data_type_t::u4)
                && params.dtypes.output == data_type_t::f16) {
#if __GNUC__ >= 12
            if (can_use_f16_fma_kernel()) {
                embag_avx512_int8_int4_f16_fma_kernel<true, uint8_t, int32_t,
                        int32_t, float16_t>(static_cast<const uint8_t *>(table),
                        weights, static_cast<const int32_t *>(indices),
                        static_cast<const int32_t *>(offsets),
                        static_cast<float16_t *>(dst), embedding_dim,
                        num_indices, num_bags, padding_idx, is_weights, algo,
                        dst_stride, include_last_offset, table_dtype,
                        fp16_scale_bias);
            } else
#endif
            {
                embag_avx512_int8_int4_kernel<true, uint8_t, int32_t, int32_t,
                        float16_t>(static_cast<const uint8_t *>(table), weights,
                        static_cast<const int32_t *>(indices),
                        static_cast<const int32_t *>(offsets),
                        static_cast<float16_t *>(dst), embedding_dim,
                        num_indices, num_bags, padding_idx, is_weights, algo,
                        dst_stride, include_last_offset, table_dtype,
                        fp16_scale_bias);
            }
        } else {
            log_error(
                    "embedding_bag_direct: unsupported table and output data "
                    "types");
            return status_t::unimplemented;
        }
    } else {
        log_error(
                "embedding_bag_direct: unsupported indices/offsets data types");
        return status_t::unimplemented;
    }
    return status_t::success;
}

#if ZENDNNL_DEPENDS_FBGEMM
/**
 * @brief Dispatch to FBGEMM embedding bag kernel
 *
 * Dispatches to the appropriate FBGEMM kernel instantiation based on
 * indices, offsets, table, and output data types.
 *
 * @return The kernel's status (status_t::failure when FBGEMM rejects the
 *         indices or offsets), or status_t::unimplemented when the
 *         table/output or indices/offsets combination is unsupported.
 */
static status_t embag_fbgemm_kernel(const void *table, const void *indices,
        const void *offsets, const float *weights, void *dst,
        const embag_params_t &params) {

    const data_type_t table_dtype = params.dtypes.table;
    const data_type_t output_dtype = params.dtypes.output;

    // Dispatch based on indices/offsets types (s64 or s32)
    if (params.dtypes.indices == data_type_t::s64
            && params.dtypes.offsets == data_type_t::s64) {
        if (table_dtype == data_type_t::f32
                && output_dtype == data_type_t::f32) {
            return invoke_fbgemm_kernel<false, float, int64_t, int64_t, float>(
                    table, indices, offsets, weights, dst, params,
                    /*bit_rate=*/0, /*is_bf16_in=*/false,
                    /*is_bf16_out=*/false);
        } else if (table_dtype == data_type_t::bf16
                && output_dtype == data_type_t::bf16) {
            return invoke_fbgemm_kernel<false, uint16_t, int64_t, int64_t,
                    uint16_t>(table, indices, offsets, weights, dst, params,
                    /*bit_rate=*/0, /*is_bf16_in=*/true, /*is_bf16_out=*/true);
        } else if (table_dtype == data_type_t::bf16
                && output_dtype == data_type_t::f32) {
            return invoke_fbgemm_kernel<false, uint16_t, int64_t, int64_t,
                    float>(table, indices, offsets, weights, dst, params,
                    /*bit_rate=*/0, /*is_bf16_in=*/true, /*is_bf16_out=*/false);
        } else if (table_dtype == data_type_t::f32
                && output_dtype == data_type_t::bf16) {
            return invoke_fbgemm_kernel<false, float, int64_t, int64_t,
                    uint16_t>(table, indices, offsets, weights, dst, params,
                    /*bit_rate=*/0, /*is_bf16_in=*/false, /*is_bf16_out=*/true);
        } else if (table_dtype == data_type_t::f16
                && output_dtype == data_type_t::f16) {
            return invoke_fbgemm_kernel<false, uint16_t, int64_t, int64_t,
                    uint16_t>(table, indices, offsets, weights, dst, params,
                    /*bit_rate=*/0, /*is_bf16_in=*/false,
                    /*is_bf16_out=*/false);
        } else if (table_dtype == data_type_t::f16
                && output_dtype == data_type_t::f32) {
            return invoke_fbgemm_kernel<false, uint16_t, int64_t, int64_t,
                    float>(table, indices, offsets, weights, dst, params,
                    /*bit_rate=*/0, /*is_bf16_in=*/false,
                    /*is_bf16_out=*/false);
        } else if (table_dtype == data_type_t::f32
                && output_dtype == data_type_t::f16) {
            return invoke_fbgemm_kernel<false, float, int64_t, int64_t,
                    uint16_t>(table, indices, offsets, weights, dst, params,
                    /*bit_rate=*/0, /*is_bf16_in=*/false,
                    /*is_bf16_out=*/false);
        }
        // TODO: Explore the feasibility of using FBGEMM for s4/s8 quantized types.
        // s4 and s8 are handled by the native kernel.
        else if (table_dtype == data_type_t::u4
                && output_dtype == data_type_t::f32) {
            return invoke_fbgemm_kernel<true, uint8_t, int64_t, int64_t, float>(
                    table, indices, offsets, weights, dst, params,
                    /*bit_rate=*/4, /*is_bf16_in=*/false,
                    /*is_bf16_out=*/false);
        } else if (table_dtype == data_type_t::u4
                && output_dtype == data_type_t::bf16) {
            return invoke_fbgemm_kernel<true, uint8_t, int64_t, int64_t,
                    uint16_t>(table, indices, offsets, weights, dst, params,
                    /*bit_rate=*/4, /*is_bf16_in=*/false, /*is_bf16_out=*/true);
        } else if (table_dtype == data_type_t::u4
                && output_dtype == data_type_t::f16) {
            return invoke_fbgemm_kernel<true, uint8_t, int64_t, int64_t,
                    uint16_t>(table, indices, offsets, weights, dst, params,
                    /*bit_rate=*/4, /*is_bf16_in=*/false,
                    /*is_bf16_out=*/false);
        } else {
            log_error(
                    "embedding_bag_direct: unsupported table/output data types "
                    "for FBGEMM backend");
            return status_t::unimplemented;
        }
    } else if (params.dtypes.indices == data_type_t::s32
            && params.dtypes.offsets == data_type_t::s32) {
        if (table_dtype == data_type_t::f32
                && output_dtype == data_type_t::f32) {
            return invoke_fbgemm_kernel<false, float, int32_t, int32_t, float>(
                    table, indices, offsets, weights, dst, params,
                    /*bit_rate=*/0, /*is_bf16_in=*/false,
                    /*is_bf16_out=*/false);
        } else if (table_dtype == data_type_t::bf16
                && output_dtype == data_type_t::bf16) {
            return invoke_fbgemm_kernel<false, uint16_t, int32_t, int32_t,
                    uint16_t>(table, indices, offsets, weights, dst, params,
                    /*bit_rate=*/0, /*is_bf16_in=*/true, /*is_bf16_out=*/true);
        } else if (table_dtype == data_type_t::bf16
                && output_dtype == data_type_t::f32) {
            return invoke_fbgemm_kernel<false, uint16_t, int32_t, int32_t,
                    float>(table, indices, offsets, weights, dst, params,
                    /*bit_rate=*/0, /*is_bf16_in=*/true, /*is_bf16_out=*/false);
        } else if (table_dtype == data_type_t::f32
                && output_dtype == data_type_t::bf16) {
            return invoke_fbgemm_kernel<false, float, int32_t, int32_t,
                    uint16_t>(table, indices, offsets, weights, dst, params,
                    /*bit_rate=*/0, /*is_bf16_in=*/false, /*is_bf16_out=*/true);
        } else if (table_dtype == data_type_t::f16
                && output_dtype == data_type_t::f16) {
            return invoke_fbgemm_kernel<false, uint16_t, int32_t, int32_t,
                    uint16_t>(table, indices, offsets, weights, dst, params,
                    /*bit_rate=*/0, /*is_bf16_in=*/false,
                    /*is_bf16_out=*/false);
        } else if (table_dtype == data_type_t::f16
                && output_dtype == data_type_t::f32) {
            return invoke_fbgemm_kernel<false, uint16_t, int32_t, int32_t,
                    float>(table, indices, offsets, weights, dst, params,
                    /*bit_rate=*/0, /*is_bf16_in=*/false,
                    /*is_bf16_out=*/false);
        } else if (table_dtype == data_type_t::f32
                && output_dtype == data_type_t::f16) {
            return invoke_fbgemm_kernel<false, float, int32_t, int32_t,
                    uint16_t>(table, indices, offsets, weights, dst, params,
                    /*bit_rate=*/0, /*is_bf16_in=*/false,
                    /*is_bf16_out=*/false);
        } else if (table_dtype == data_type_t::u4
                && output_dtype == data_type_t::f32) {
            return invoke_fbgemm_kernel<true, uint8_t, int32_t, int32_t, float>(
                    table, indices, offsets, weights, dst, params,
                    /*bit_rate=*/4, /*is_bf16_in=*/false,
                    /*is_bf16_out=*/false);
        } else if (table_dtype == data_type_t::u4
                && output_dtype == data_type_t::bf16) {
            return invoke_fbgemm_kernel<true, uint8_t, int32_t, int32_t,
                    uint16_t>(table, indices, offsets, weights, dst, params,
                    /*bit_rate=*/4, /*is_bf16_in=*/false, /*is_bf16_out=*/true);
        } else if (table_dtype == data_type_t::u4
                && output_dtype == data_type_t::f16) {
            return invoke_fbgemm_kernel<true, uint8_t, int32_t, int32_t,
                    uint16_t>(table, indices, offsets, weights, dst, params,
                    /*bit_rate=*/4, /*is_bf16_in=*/false,
                    /*is_bf16_out=*/false);
        } else {
            log_error(
                    "embedding_bag_direct: unsupported table/output data types "
                    "for FBGEMM backend");
            return status_t::unimplemented;
        }
    } else {
        log_error(
                "embedding_bag_direct: unsupported indices/offsets data types "
                "for FBGEMM backend");
        return status_t::unimplemented;
    }
}
#endif

/**
 * @brief Dispatch to the AVX2 embedding bag kernel
 *
 * select_embag_backend picks this kernel on CPUs that cannot run the native
 * AVX-512 kernels (see can_use_avx512_embag_kernel).
 *
 * @return status_t::success when the kernel runs, status_t::failure when it
 *         cannot allocate its scratch, or status_t::unimplemented for types
 *         the AVX2 kernel does not cover.
 */
static status_t embag_avx2_native_kernel(const void *table, const void *indices,
        const void *offsets, const float *weights, void *dst,
        const embag_params_t &params) {
    const data_type_t table_dtype = params.dtypes.table;
    const data_type_t output_dtype = params.dtypes.output;

    const auto run = [&](auto in_tag, auto index_tag, auto out_tag) {
        using InType = decltype(in_tag);
        using IndexType = decltype(index_tag);
        using OutType = decltype(out_tag);
        const status_t kernel_status
                = embag_avx2_kernel<InType, IndexType, IndexType, OutType>(
                        static_cast<const InType *>(table), weights,
                        static_cast<const IndexType *>(indices),
                        static_cast<const IndexType *>(offsets),
                        static_cast<OutType *>(dst), params.embedding_dim,
                        params.num_indices, params.num_bags, params.padding_idx,
                        params.is_weights, params.algo, params.dst_stride,
                        params.include_last_offset);
        if (kernel_status != status_t::success) {
            log_error(
                    "embedding_bag_direct: the AVX2 kernel could not "
                    "allocate its accumulator scratch");
        }
        return kernel_status;
    };
    const auto run_types = [&](auto index_tag) {
        if (table_dtype == data_type_t::f32
                && output_dtype == data_type_t::f32) {
            return run(float {}, index_tag, float {});
        } else if (table_dtype == data_type_t::f32
                && output_dtype == data_type_t::bf16) {
            return run(float {}, index_tag, uint16_t {});
        } else if (table_dtype == data_type_t::bf16
                && output_dtype == data_type_t::f32) {
            return run(uint16_t {}, index_tag, float {});
        } else if (table_dtype == data_type_t::bf16
                && output_dtype == data_type_t::bf16) {
            return run(uint16_t {}, index_tag, uint16_t {});
        } else {
            log_error("embedding_bag_direct: the AVX2 kernel does not support ",
                    dtype_to_string(table_dtype), " table with ",
                    dtype_to_string(output_dtype), " output");
            return status_t::unimplemented;
        }
    };

    // For embedding lookup (algo == none), offsets is nullptr
    const bool is_offsets = (offsets != nullptr);
    if (params.dtypes.indices == data_type_t::s64
            && (!is_offsets || params.dtypes.offsets == data_type_t::s64)) {
        return run_types(int64_t {});
    }
    if (params.dtypes.indices == data_type_t::s32
            && (!is_offsets || params.dtypes.offsets == data_type_t::s32)) {
        return run_types(int32_t {});
    }
    log_error("embedding_bag_direct: unsupported indices/offsets data types");
    return status_t::unimplemented;
}

/** @brief Kernel family that runs one embedding bag operation */
enum class embag_backend_t { reference, fbgemm, avx512, avx2, unsupported };

/**
 * @brief Whether this call may use the native AVX-512 kernels: the CPU can
 *        run them and ZENDNNL_EMBAG_ISA does not ask for AVX2
 */
static inline bool embag_avx512_allowed() {
    return can_use_avx512_embag_kernel() && !embag_isa_env_requests_avx2();
}

/**
 * @brief Pick the kernel family for an operation whose params.kernel
 *        kernel_select has resolved
 *
 * FBGEMM selects its own ISA at run time. Without the AVX-512 kernels, the
 * AVX2 kernel covers FP32 and BF16 tables and outputs and nothing else runs.
 *
 * @param avx512_allowed embag_avx512_allowed() for this call.
 */
static inline embag_backend_t select_embag_backend(
        const embag_params_t &params, bool avx512_allowed) {
    if (params.kernel == embag_kernel_t::reference) {
        return embag_backend_t::reference;
    }
#if ZENDNNL_DEPENDS_FBGEMM
    if (params.kernel == embag_kernel_t::fbgemm && can_use_fbgemm(params)) {
        return embag_backend_t::fbgemm;
    }
#endif
    if (avx512_allowed) { return embag_backend_t::avx512; }

    const auto is_f32_or_bf16 = [](data_type_t dtype) {
        return dtype == data_type_t::f32 || dtype == data_type_t::bf16;
    };
    if (can_use_avx2_embag_kernel() && is_f32_or_bf16(params.dtypes.table)
            && is_f32_or_bf16(params.dtypes.output)) {
        return embag_backend_t::avx2;
    }
    return embag_backend_t::unsupported;
}

/**
 * @brief Run one embedding bag operation on the kernel family that
 *        select_embag_backend picked for it
 *
 * Callers resolve params.kernel with kernel_select and pick @p backend on
 * the calling thread, before group_embedding_bag_direct enters its OpenMP
 * region.
 *
 * @return The kernel's status, or status_t::isa_unsupported when no kernel
 *         this CPU can run covers the operation's types.
 */
static status_t dispatch_avx512_kernel(const void *table, const void *indices,
        const void *offsets, const float *weights, void *dst,
        embag_params_t &params, embag_backend_t backend) {

    // Native and FBGEMM also run from that OpenMP region. They do not
    // publish accumulation precision on embag_config_t: a process-wide
    // write races even when every table stores the same value, and the
    // LOWOHA reference path does not read the field. FBGEMM and the AVX2
    // kernel accumulate in F32. Native AVX512 uses F16 FMA only for
    // F16-touching dtypes when can_use_f16_fma_kernel() is true; that choice
    // stays inside the kernel. ops::embag_ref_kernel still reads the value
    // published by the operator execute path on the calling thread.

    switch (backend) {
        case embag_backend_t::reference:
            // group_embedding_bag_direct resolves each table with
            // kernel_select before its AVX512-FP16 preflight and applies that
            // gate only to FBGEMM and native, so an F16 reference table
            // reaches this branch on hosts without that ISA.
            // embedding_bag_direct skips the same gate for reference.
            // embedding_bag_ref_direct passes F16 vs F32 accumulation into its
            // templates per call and does not read or write
            // embag_config_t::accum_type, so mixed-dtype groups can run it
            // concurrently.
            log_info("Using reference kernel");
            return embedding_bag_ref_direct(
                    table, indices, offsets, weights, dst, params);
#if ZENDNNL_DEPENDS_FBGEMM
        case embag_backend_t::fbgemm:
            log_info("Using FBGEMM kernel");
            return embag_fbgemm_kernel(
                    table, indices, offsets, weights, dst, params);
#endif
        case embag_backend_t::avx512:
        case embag_backend_t::avx2:
            // This includes fbgemm requests that can_use_fbgemm() rejected
            // (non-sum algo, s8/s4 table, fp32 scale/bias) or that this build
            // has no FBGEMM for. Record the backend that actually runs so the
            // apilog line and the caller's params agree with it instead of
            // still reporting fbgemm.
            params.kernel = embag_kernel_t::native;
            if (backend == embag_backend_t::avx2) {
                log_info("Using ZenDNN AVX2 kernel");
                return embag_avx2_native_kernel(
                        table, indices, offsets, weights, dst, params);
            }
            log_info("Using ZenDNN kernel");
            return embag_native_kernel(
                    table, indices, offsets, weights, dst, params);
        default: break;
    }

    log_error("embedding_bag_direct: no kernel for ",
            dtype_to_string(params.dtypes.table), " table with ",
            dtype_to_string(params.dtypes.output),
            " output: the native kernels for these types need AVX-512F "
            "and AVX512-BF16, which this CPU lacks or ZENDNNL_EMBAG_ISA=avx2 "
            "rules out");
    return status_t::isa_unsupported;
}

} // namespace embag
} // namespace lowoha
} // namespace zendnnl

#endif // _LOWOHA_DISPATCH_KERNEL_HPP
