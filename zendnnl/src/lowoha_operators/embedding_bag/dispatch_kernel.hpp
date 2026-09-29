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
 * @return status_t::success when a kernel runs, or status_t::unimplemented
 *         when the table/output or indices/offsets combination is unsupported.
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
            invoke_fbgemm_kernel<false, float, int64_t, int64_t, float>(table,
                    indices, offsets, weights, dst, params,
                    /*bit_rate=*/0, /*is_bf16_in=*/false,
                    /*is_bf16_out=*/false);
        } else if (table_dtype == data_type_t::bf16
                && output_dtype == data_type_t::bf16) {
            invoke_fbgemm_kernel<false, uint16_t, int64_t, int64_t, uint16_t>(
                    table, indices, offsets, weights, dst, params,
                    /*bit_rate=*/0, /*is_bf16_in=*/true, /*is_bf16_out=*/true);
        } else if (table_dtype == data_type_t::bf16
                && output_dtype == data_type_t::f32) {
            invoke_fbgemm_kernel<false, uint16_t, int64_t, int64_t, float>(
                    table, indices, offsets, weights, dst, params,
                    /*bit_rate=*/0, /*is_bf16_in=*/true, /*is_bf16_out=*/false);
        } else if (table_dtype == data_type_t::f32
                && output_dtype == data_type_t::bf16) {
            invoke_fbgemm_kernel<false, float, int64_t, int64_t, uint16_t>(
                    table, indices, offsets, weights, dst, params,
                    /*bit_rate=*/0, /*is_bf16_in=*/false, /*is_bf16_out=*/true);
        } else if (table_dtype == data_type_t::f16
                && output_dtype == data_type_t::f16) {
            invoke_fbgemm_kernel<false, uint16_t, int64_t, int64_t, uint16_t>(
                    table, indices, offsets, weights, dst, params,
                    /*bit_rate=*/0, /*is_bf16_in=*/false,
                    /*is_bf16_out=*/false);
        } else if (table_dtype == data_type_t::f16
                && output_dtype == data_type_t::f32) {
            invoke_fbgemm_kernel<false, uint16_t, int64_t, int64_t, float>(
                    table, indices, offsets, weights, dst, params,
                    /*bit_rate=*/0, /*is_bf16_in=*/false,
                    /*is_bf16_out=*/false);
        } else if (table_dtype == data_type_t::f32
                && output_dtype == data_type_t::f16) {
            invoke_fbgemm_kernel<false, float, int64_t, int64_t, uint16_t>(
                    table, indices, offsets, weights, dst, params,
                    /*bit_rate=*/0, /*is_bf16_in=*/false,
                    /*is_bf16_out=*/false);
        }
        // TODO: Explore the feasibility of using FBGEMM for s4/s8 quantized types.
        // s4 and s8 are handled by the native kernel.
        else if (table_dtype == data_type_t::u4
                && output_dtype == data_type_t::f32) {
            invoke_fbgemm_kernel<true, uint8_t, int64_t, int64_t, float>(table,
                    indices, offsets, weights, dst, params,
                    /*bit_rate=*/4, /*is_bf16_in=*/false,
                    /*is_bf16_out=*/false);
        } else if (table_dtype == data_type_t::u4
                && output_dtype == data_type_t::bf16) {
            invoke_fbgemm_kernel<true, uint8_t, int64_t, int64_t, uint16_t>(
                    table, indices, offsets, weights, dst, params,
                    /*bit_rate=*/4, /*is_bf16_in=*/false, /*is_bf16_out=*/true);
        } else if (table_dtype == data_type_t::u4
                && output_dtype == data_type_t::f16) {
            invoke_fbgemm_kernel<true, uint8_t, int64_t, int64_t, uint16_t>(
                    table, indices, offsets, weights, dst, params,
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
            invoke_fbgemm_kernel<false, float, int32_t, int32_t, float>(table,
                    indices, offsets, weights, dst, params,
                    /*bit_rate=*/0, /*is_bf16_in=*/false,
                    /*is_bf16_out=*/false);
        } else if (table_dtype == data_type_t::bf16
                && output_dtype == data_type_t::bf16) {
            invoke_fbgemm_kernel<false, uint16_t, int32_t, int32_t, uint16_t>(
                    table, indices, offsets, weights, dst, params,
                    /*bit_rate=*/0, /*is_bf16_in=*/true, /*is_bf16_out=*/true);
        } else if (table_dtype == data_type_t::bf16
                && output_dtype == data_type_t::f32) {
            invoke_fbgemm_kernel<false, uint16_t, int32_t, int32_t, float>(
                    table, indices, offsets, weights, dst, params,
                    /*bit_rate=*/0, /*is_bf16_in=*/true, /*is_bf16_out=*/false);
        } else if (table_dtype == data_type_t::f32
                && output_dtype == data_type_t::bf16) {
            invoke_fbgemm_kernel<false, float, int32_t, int32_t, uint16_t>(
                    table, indices, offsets, weights, dst, params,
                    /*bit_rate=*/0, /*is_bf16_in=*/false, /*is_bf16_out=*/true);
        } else if (table_dtype == data_type_t::f16
                && output_dtype == data_type_t::f16) {
            invoke_fbgemm_kernel<false, uint16_t, int32_t, int32_t, uint16_t>(
                    table, indices, offsets, weights, dst, params,
                    /*bit_rate=*/0, /*is_bf16_in=*/false,
                    /*is_bf16_out=*/false);
        } else if (table_dtype == data_type_t::f16
                && output_dtype == data_type_t::f32) {
            invoke_fbgemm_kernel<false, uint16_t, int32_t, int32_t, float>(
                    table, indices, offsets, weights, dst, params,
                    /*bit_rate=*/0, /*is_bf16_in=*/false,
                    /*is_bf16_out=*/false);
        } else if (table_dtype == data_type_t::f32
                && output_dtype == data_type_t::f16) {
            invoke_fbgemm_kernel<false, float, int32_t, int32_t, uint16_t>(
                    table, indices, offsets, weights, dst, params,
                    /*bit_rate=*/0, /*is_bf16_in=*/false,
                    /*is_bf16_out=*/false);
        } else if (table_dtype == data_type_t::u4
                && output_dtype == data_type_t::f32) {
            invoke_fbgemm_kernel<true, uint8_t, int32_t, int32_t, float>(table,
                    indices, offsets, weights, dst, params,
                    /*bit_rate=*/4, /*is_bf16_in=*/false,
                    /*is_bf16_out=*/false);
        } else if (table_dtype == data_type_t::u4
                && output_dtype == data_type_t::bf16) {
            invoke_fbgemm_kernel<true, uint8_t, int32_t, int32_t, uint16_t>(
                    table, indices, offsets, weights, dst, params,
                    /*bit_rate=*/4, /*is_bf16_in=*/false, /*is_bf16_out=*/true);
        } else if (table_dtype == data_type_t::u4
                && output_dtype == data_type_t::f16) {
            invoke_fbgemm_kernel<true, uint8_t, int32_t, int32_t, uint16_t>(
                    table, indices, offsets, weights, dst, params,
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
    return status_t::success;
}
#endif

/**
 * @brief Dispatch to optimized AVX512 embedding bag kernel
 *
 * Dispatches to the appropriate AVX512 kernel instantiation based on
 * indices, offsets, table, and output data types.
 */
static status_t dispatch_avx512_kernel(const void *table, const void *indices,
        const void *offsets, const float *weights, void *dst,
        embag_params_t &params) {

    kernel_select(params);

    // Reference backend. group_embedding_bag_direct resolves each table
    // with kernel_select before its AVX512-FP16 preflight and applies that
    // gate only to FBGEMM and native, so an F16 reference table reaches
    // this branch on hosts without that ISA. embedding_bag_direct skips
    // the same gate for reference. embedding_bag_ref_direct passes F16 vs
    // F32 accumulation into its templates per call and does not read or
    // write embag_config_t::accum_type, so mixed-dtype groups can run it
    // concurrently.
    if (params.kernel == embag_kernel_t::reference) {
        log_info("Using reference kernel");
        return embedding_bag_ref_direct(
                table, indices, offsets, weights, dst, params);
    }

    // Native and FBGEMM also run from that OpenMP region. They do not
    // publish accumulation precision on embag_config_t: a process-wide
    // write races even when every table stores the same value, and the
    // LOWOHA reference path does not read the field. FBGEMM accumulates
    // in F32. Native AVX512 uses F16 FMA only for F16-touching dtypes when
    // can_use_f16_fma_kernel() is true; that choice stays inside the
    // kernel. ops::embag_ref_kernel still reads the value published by the
    // operator execute path on the calling thread.

#if ZENDNNL_DEPENDS_FBGEMM
    if (params.kernel == embag_kernel_t::fbgemm && can_use_fbgemm(params)) {
        log_info("Using FBGEMM kernel");
        return embag_fbgemm_kernel(
                table, indices, offsets, weights, dst, params);
    }
#endif

    // Everything that reaches this point runs the native kernel, including an
    // fbgemm request that can_use_fbgemm() rejected (non-sum algo, s8/s4 table,
    // fp32 scale/bias) or that this build has no FBGEMM for. Record the backend
    // that actually runs so the apilog line and the caller's params agree with
    // it instead of still reporting fbgemm.
    params.kernel = embag_kernel_t::native;

    log_info("Using ZenDNN kernel");
    return embag_native_kernel(table, indices, offsets, weights, dst, params);
}

} // namespace embag
} // namespace lowoha
} // namespace zendnnl

#endif // _LOWOHA_DISPATCH_KERNEL_HPP
