/*******************************************************************************
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

#ifndef LOWOHA_MATMUL_UTILS_HPP
#define LOWOHA_MATMUL_UTILS_HPP

#include <functional>
#include <limits>
#include <mutex>
#include <numeric>
#include <omp.h>
#include <utility>
#include "lowoha_operators/matmul/lowoha_common.hpp"
#include "lowoha_operators/matmul/lowoha_matmul.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

/**
 * @brief Get global mutex for thread-safe lowoha operations
 * @return Reference to the lowoha mutex
 */
std::mutex &get_lowoha_mutex();

inline bool is_aocl_prepacked_weight(const matmul_params &params) noexcept {
    // Scalar ZenTorch prepack currently publishes only mem_format_b='r', so
    // unset provenance is the legacy AOCL representation.
    return params.mem_format_b == 'r'
            && (params.lowoha_algo == matmul_algo_t::none
                    || params.lowoha_algo == matmul_algo_t::aocl_dlp_blocked);
}

inline bool is_foreign_prepacked_weight(const matmul_params &params) noexcept {
    return params.mem_format_b == 'r'
            && params.lowoha_algo != matmul_algo_t::none
            && params.lowoha_algo != matmul_algo_t::aocl_dlp_blocked
            && params.lowoha_algo != matmul_algo_t::moe_custom_kernel;
}

inline int64_t divup(int64_t x, int64_t y) {
    return (x + y - 1) / y;
}

template <class F>
inline void zendnnl_parallel_for(const int64_t begin, const int64_t end,
        const int64_t grain_size, const F &f) {

    if (begin >= end) { return; }
    std::atomic_flag err_flag = ATOMIC_FLAG_INIT;
    std::exception_ptr eptr;
    // choose number of tasks based on grain size and number of threads
    int64_t num_threads = omp_in_parallel() ? 1 : omp_get_max_threads();
    if (grain_size > 0) {
        num_threads = std::min(num_threads, divup((end - begin), grain_size));
    }

#pragma omp parallel num_threads(num_threads)
    {
        int64_t num_threads = omp_get_num_threads();
        int64_t tid = omp_get_thread_num();
        int64_t chunk_size = divup((end - begin), num_threads);
        int64_t begin_tid = begin + tid * chunk_size;
        if (begin_tid < end) {
            try {
                f(begin_tid, std::min(end, chunk_size + begin_tid));
            } catch (...) {
                if (!err_flag.test_and_set()) {
                    eptr = std::current_exception();
                }
            }
        }
    }
    if (eptr) { std::rethrow_exception(eptr); }
}

inline const void *get_matrix_block(const void *base, int row_start,
        int col_start, int lda, bool trans, size_t type_size) {
    if (trans) {
        // Accessing column-major layout when transposed
        return static_cast<const uint8_t *>(base)
                + (col_start * lda + row_start) * type_size;
    } else {
        return static_cast<const uint8_t *>(base)
                + (row_start * lda + col_start) * type_size;
    }
}

inline void *get_output_block(
        void *base, int row_start, int col_start, int ldc, size_t type_size) {
    return static_cast<uint8_t *>(base)
            + (row_start * ldc + col_start) * type_size;
}

inline int get_batch_index(int b, int batch_size) {
    return (batch_size == 1) ? 0 : (b % batch_size);
}

/**
 * @brief Check if post-op tensor has batch dimension (3D)
 *
 * A post-op is considered 3D if it has 3 dimensions and the batch dimension > 1.
 * This is used to determine if batch-specific offsets need to be applied during BMM.
 *
 * @param po The post-op to check
 * @return true if the post-op has a non-trivial batch dimension
 */
inline bool is_3d_postop(const matmul_post_op &po) {
    return po.dims.size() >= 3 && po.dims[0] > 1;
}

/**
 * @brief Calculate the batch stride for a 3D post-op tensor
 *
 * For a 3D tensor with dims [Batch, M, N], the batch stride is M * N elements.
 * This function returns the stride in bytes.
 *
 * @param po The post-op containing dimension and type information
 * @return The batch stride in bytes, or 0 if not a 3D tensor
 */
size_t get_postop_batch_stride(const matmul_post_op &po);

/**
 * @brief Apply batch and row offsets to post-op buffers for BMM
 *
 * This function modifies post-op buffer pointers to account for:
 * 1. Batch offset: For 3D post-ops, offset by batch_idx * batch_stride
 * 2. Row offset: For partitioned execution, offset by m_start * N
 *
 * @param params The matmul_params to modify (post-op buffers will be updated)
 * @param batch_idx The current batch index
 * @param m_start The starting row index for partitioned execution
 * @param N The number of columns in the output matrix
 */
void apply_bmm_postop_offsets(
        matmul_params &params, int batch_idx, int m_start, int N);

/**
* @brief Validates input parameters for matrix multiplication direct operation.
*
* This function performs comprehensive validation of all input parameters to ensure
* they are valid and compatible for matrix multiplication. It checks for null pointers,
* valid dimensions, supported data types, and parameter consistency.
*
* @param src Pointer to the source matrix A data
* @param weight Pointer to the weight matrix B data
* @param dst Pointer to the destination matrix C data
* @param M Number of rows in matrix A and output matrix C
* @param N Number of columns in matrix B and output matrix C
* @param K Number of columns in matrix A and rows in matrix B
* @param Batch_A Number of batches for matrix A
* @param Batch_B Number of batches for matrix B
* @param params Const reference to matmul_params containing operation configuration
* @param is_weights_const Boolean indicating if weights are constant
* @return status_t::success if all validations pass, status_t::failure otherwise
*/
status_t validate_w4a8_inputs(const matmul_params &params, int M, int N, int K);

status_t validate_matmul_direct_inputs(const void *src, const void *weight,
        const void *dst, const int M, const int N, const int K,
        const int Batch_A, const int Batch_B, const matmul_params &params,
        const bool is_weights_const);

/**
 * @brief Checked quant-parameter element count with scalar empty-dims semantics.
 *
 * Quant metadata uses `{}`, `{1}`, and `{1,1}` for per-tensor storage in
 * different frontends. Returns one for empty dims, or -1 for non-positive /
 * overflowing dimensions.
 */
inline int64_t checked_quant_param_num_elements(
        const std::vector<int64_t> &dims) noexcept {
    int64_t count = 1;
    for (const int64_t dim : dims) {
        if (dim <= 0 || count > std::numeric_limits<int64_t>::max() / dim)
            return -1;
        count *= dim;
    }
    return count;
}

inline bool quant_param_dims_are_per_tensor(
        const std::vector<int64_t> &dims) noexcept {
    return checked_quant_param_num_elements(dims) == 1;
}

inline bool quant_param_is_per_tensor(
        const matmul_quantization_params_t::matmul_quant_t &q) noexcept {
    return q.buff != nullptr && quant_param_dims_are_per_tensor(q.dims);
}

inline bool quant_param_is_per_channel(
        const matmul_quantization_params_t::matmul_quant_t &q, int N) noexcept {
    if (q.buff == nullptr || N <= 0) return false;
    return (q.dims.size() == 1 && q.dims[0] == static_cast<int64_t>(N))
            || (q.dims.size() == 2 && q.dims[0] == 1
                    && q.dims[1] == static_cast<int64_t>(N));
}

inline bool quant_param_is_per_token(
        const matmul_quantization_params_t::matmul_quant_t &q, int M) noexcept {
    return M > 0 && q.dims.size() == 2 && q.dims[0] == static_cast<int64_t>(M)
            && q.dims[1] == 1;
}

inline bool quant_scale_dtype_supported(data_type_t dt) noexcept {
    return dt == data_type_t::f32 || dt == data_type_t::bf16;
}

inline bool read_positive_per_tensor_scale(
        const matmul_quantization_params_t::matmul_quant_t &q,
        float &value) noexcept {
    if (!quant_param_is_per_tensor(q) || !quant_scale_dtype_supported(q.dt))
        return false;
    value = q.dt == data_type_t::f32
            ? *static_cast<const float *>(q.buff)
            : static_cast<float>(
                      *static_cast<const zendnnl::common::bfloat16_t *>(
                              q.buff));
    return std::isfinite(value) && value > 0.0f;
}

inline bool valid_positive_per_tensor_scale(
        const matmul_quantization_params_t::matmul_quant_t &q) noexcept {
    float value = 0.0f;
    return read_positive_per_tensor_scale(q, value);
}

inline bool read_u8_per_tensor_zero_point(
        const matmul_quantization_params_t::matmul_quant_t &q,
        int32_t &value) noexcept {
    if (!quant_param_is_per_tensor(q) || q.dt != data_type_t::s32) return false;
    value = *static_cast<const int32_t *>(q.buff);
    return value >= 0 && value <= 255;
}

inline bool valid_u8_per_tensor_zero_point(
        const matmul_quantization_params_t::matmul_quant_t &q) noexcept {
    int32_t value = 0;
    return read_u8_per_tensor_zero_point(q, value);
}

inline bool is_static_u8s8_quant_config(const matmul_params &params) noexcept {
    return !params.dynamic_quant && params.dtypes.src == data_type_t::u8
            && params.dtypes.wei == data_type_t::s8
            && (params.dtypes.compute == data_type_t::none
                    || params.dtypes.compute == data_type_t::u8);
}

inline bool static_u8s8_quant_supported(
        const matmul_params &params, int N) noexcept {
    if (!is_static_u8s8_quant_config(params)) return false;
    const auto &q = params.quant_params;
    const bool weight_scale_supported
            = valid_positive_per_tensor_scale(q.wei_scale)
            || (quant_param_is_per_channel(q.wei_scale, N)
                    && quant_scale_dtype_supported(q.wei_scale.dt));
    if (!valid_positive_per_tensor_scale(q.src_scale)
            || !valid_u8_per_tensor_zero_point(q.src_zp)
            || !weight_scale_supported || q.wei_zp.buff != nullptr) {
        return false;
    }
    if (params.dtypes.dst == data_type_t::u8) {
        return valid_positive_per_tensor_scale(q.dst_scale)
                && valid_u8_per_tensor_zero_point(q.dst_zp);
    }
    return (params.dtypes.dst == data_type_t::bf16
                   || params.dtypes.dst == data_type_t::f32)
            && q.dst_scale.buff == nullptr && q.dst_zp.buff == nullptr;
}

/**
 * @brief Detects a symmetric-quantization INT8 configuration.
 *
 * Returns true when @p params describes the AOCL-DLP sym_quant kernel
 * path: s8 source, s8 weight, no source zero-point, a per-token or
 * per-group source scale (more than one scale element), and an f32 or
 * bf16 destination.  Mirrors the inline definitions used by
 * `kernel_select` and the AOCL backend.  Used by the validator and the
 * LOWOHA dispatcher to gate features (e.g. post-ops) that the classic
 * AOCL-DLP sym_quant kernel does not support.
 *
 * @param params Matmul parameters.
 * @return true if the config matches the sym_quant path; false otherwise.
 */
inline bool is_sym_quant_config(const matmul_params &params) {
    // Cheapest short-circuit first: src/wei dtype.  Non-s8 sources skip
    // every subsequent check (the vast majority of matmul calls).
    if (params.dtypes.src != data_type_t::s8
            || params.dtypes.wei != data_type_t::s8) {
        return false;
    }
    if (params.quant_params.src_zp.buff) { return false; }
    if (params.dtypes.dst != data_type_t::f32
            && params.dtypes.dst != data_type_t::bf16) {
        return false;
    }
    return checked_quant_param_num_elements(params.quant_params.src_scale.dims)
            > 1;
}

/**
 * @brief Detects a dynamic-quantization INT8 configuration.
 *
 * Returns true when @p params describes the dynamic-quantization path
 * that `reorder_quantization_wrapper` will pick up: `dynamic_quant`
 * flag set, an s8 (or GGML-packed) weight, bf16/f32 source, and an s8/u8
 * compute dtype.
 *
 * @param params Matmul parameters.
 * @return true if the config will be dynamic-quantized; false otherwise.
 */
inline bool is_dynamic_quant_config(const matmul_params &params) {
    // Cheapest short-circuit: the dynamic_quant flag is unset for the
    // vast majority of matmul calls.
    if (!params.dynamic_quant) { return false; }
    // GGML packed weights are exempt from the s8 requirement: a Q4_0 weight is
    // still s4 here and only widens to s8 AFTER this source quantization, which
    // group_matmul_direct runs before the unpack.  Demanding s8 would skip the
    // quant and leave a bf16 source that ggml_is_sym_quant then rejects.  Plain
    // (unpacked) s4 stays out — that is W4A8, which reaches
    // `reorder_quantization_wrapper` through is_w4a8_config instead and carries
    // different source-scale shapes.
    if (params.dtypes.wei != data_type_t::s8
            && params.packing.pack_format_b != 1) {
        return false;
    }
    if (params.dtypes.src != data_type_t::bf16
            && params.dtypes.src != data_type_t::f32) {
        return false;
    }
    return params.dtypes.compute == data_type_t::s8
            || params.dtypes.compute == data_type_t::u8;
}

inline bool has_quant_scale_metadata(
        const matmul_quantization_params_t::matmul_quant_t &scale) {
    return scale.buff != nullptr && !scale.dims.empty()
            && scale.dt != data_type_t::none;
}

/**
 * @brief W4A8: s4 weights, bf16 output, s8 compute, symmetric.
 *        Dynamic entry: bf16 src + dynamic_quant. Static entry: pre-quantized
 *        s8 src with populated src/wei scale buffers (dynamic_quant=false).
 */
inline bool is_w4a8_config(const matmul_params &params) {
    if (params.dtypes.wei != data_type_t::s4) { return false; }
    if (params.dtypes.dst != data_type_t::bf16) { return false; }
    if (params.dtypes.compute != data_type_t::s8) { return false; }
    if (params.quant_params.wei_zp.buff || params.quant_params.src_zp.buff) {
        return false;
    }
    // Runtime/pre-quantized path: source is already quantized to s8.
    if (params.dtypes.src == data_type_t::s8) {
        return !params.dynamic_quant
                && has_quant_scale_metadata(params.quant_params.src_scale)
                && has_quant_scale_metadata(params.quant_params.wei_scale);
    }
    // Entry path: bf16 source with dynamic quant enabled.
    if (!params.dynamic_quant) { return false; }
    if (params.dtypes.src == data_type_t::bf16) {
        return !params.quant_params.src_scale.dims.empty()
                && params.quant_params.src_scale.dt != data_type_t::none
                && !params.quant_params.wei_scale.dims.empty()
                && params.quant_params.wei_scale.dt != data_type_t::none;
    }
    return false;
}

inline int effective_postop_leading_dim(
        const matmul_post_op &postop, int N) noexcept {
    return postop.leading_dim == -1 ? N : postop.leading_dim;
}

inline bool int8_ck_binary_postops_supported(
        const matmul_params &params, int M, int N) noexcept {
    if (params.postop_.size() != 2 || M <= 0 || N <= 0) return false;
    const auto valid
            = [&](const matmul_post_op &postop, post_op_type_t expected_type) {
        return postop.po_type == expected_type && postop.buff != nullptr
                && postop.dtype == data_type_t::f32 && postop.dims.size() == 2
                && postop.dims[0] == static_cast<int64_t>(M)
                && postop.dims[1] == static_cast<int64_t>(N)
                && effective_postop_leading_dim(postop, N) >= N;
    };
    return valid(params.postop_[0], post_op_type_t::binary_mul)
            && valid(params.postop_[1], post_op_type_t::binary_add);
}

inline bool int8_ck_postops_supported(
        const matmul_params &params, int M, int N) noexcept {
    if (params.postop_.empty()) return true;
    if (params.postop_.size() == 1) {
        const auto &postop = params.postop_[0];
        return postop.buff == nullptr
                && (postop.po_type == post_op_type_t::relu
                        || postop.po_type == post_op_type_t::sigmoid);
    }
    return int8_ck_binary_postops_supported(params, M, N);
}

/** @brief AOCL sym_quant requires K/G divisible and group_size % 4 == 0. */
inline bool is_w4a8_sym_quant_group_valid(int K, int64_t num_groups) {
    if (num_groups <= 0) { return false; }
    if (static_cast<int64_t>(K) % num_groups != 0) { return false; }
    const int64_t group_size = static_cast<int64_t>(K) / num_groups;
    return group_size % 4 == 0;
}

/**
 * @brief Convert post-op names to a comma-separated string.
 *
 * This function takes a matmul_params structure and converts all post-op types
 * to a comma-separated string representation.
 *
 * @param params The matmul_params structure containing post-op information.
 * @return A string containing comma-separated post-op names, or "none" if no post-ops.
 */
std::string post_op_names_to_string(const matmul_params &params);

/**
 * @brief Convert matmul_algo_t enum to string representation.
 *
 * This function converts a matmul_algo_t enum value to its string representation.
 *
 * @param kernel The matmul_algo_t enum value to convert.
 * @return A const char* pointer to the string representation of the kernel type.
 */
const char *kernel_to_string(matmul_algo_t kernel);

/**
 * @brief Get post-op data types as a comma-separated string for binary_add/binary_mul.
 *
 * This function extracts data types from post-ops that are binary_add or binary_mul
 * and returns them as a comma-separated string.
 *
 * @param params The matmul_params structure containing post-op information.
 * @return A string containing comma-separated data types, or empty string if none.
 */
std::string post_op_data_types_to_string(const matmul_params &params);

inline bool may_i_use_dlp_partition(
        int batch_count, int M, int N, int num_threads, data_type_t dtype);

inline matmul_algo_t select_algo_by_heuristics_bf16_bmm(
        int BS, int M, int N, int K, int num_threads);

inline matmul_algo_t select_algo_by_heuristics_bf16_mm(int M, int N, int K);

/**
* @brief Selects the optimal matrix multiplication kernel algorithm.
*
* This function analyzes the matrix multiplication parameters and system characteristics
* to select the most appropriate kernel algorithm for optimal performance. It considers
* factors such as matrix dimensions, batch sizes, data types, available hardware features,
* and library dependencies.
*
* @param params Reference to matmul_params containing matrix multiplication configuration
* @param Batch_A Number of batches for matrix A (input tensor)
* @param Batch_B Number of batches for matrix B (weight tensor)
* @param batch_count Total number of batch operations to perform
* @param M Number of rows in matrix A and output matrix C
* @param N Number of columns in matrix B and output matrix C
* @param K Number of columns in matrix A and rows in matrix B
* @param num_threads Number of available threads for parallel execution
* @param bias Pointer to bias data; nullptr if no bias is used
* @param is_weights_const Indicates if the weights are constant
* @return matmul_algo_t The selected kernel algorithm (e.g., AOCL DLP, OneDNN, LibXSMM)
*/
matmul_algo_t kernel_select(matmul_params &params, int Batch_A, int Batch_B,
        int batch_count, int M, int N, int K, int num_threads, const void *bias,
        const bool is_weights_const, bool transB = false);

/**
 * @brief Get the auto-tuner version number
 *
 * This function returns the version of the auto-tuner implementation currently in use.
 * The version number can be used to select between different auto-tuning strategies.
 *
 * @return unsigned int The auto-tuner version number
 */
unsigned int get_auto_tuner_ver();

/**
 * @brief Check if MM partitioner should be enabled
 *
 * Returns true if either the runtime environment variable
 * ZENDNN_ENABLE_MM_PARTITIONER is set, or the selected kernel
 * is libxsmm or libxsmm_blocked (which require the partitioner
 * for tiled M/N execution).
 *
 * @param kernel The selected matmul algorithm
 * @return true if MM partitioner should be used, false otherwise
 */
bool should_use_mm_partitioner(const matmul_algo_t &kernel);

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // LOWOHA_MATMUL_UTILS_HPP
