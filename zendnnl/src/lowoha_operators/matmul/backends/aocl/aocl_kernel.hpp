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

#ifndef _AOCL_KERNEL_HPP
#define _AOCL_KERNEL_HPP

#include "common/float16.hpp"
#include "lowoha_operators/matmul/lowoha_common.hpp"

#if ZENDNNL_DEPENDS_AOCLDLP
#include "aocl_dlp.h"
#else
#include <cstddef>
#include <cstdint>
using md_t = std::
        int64_t; // matches aocl-dlp md_t (int64_t); previously dim_t from blis.h
using msz_t = std::
        size_t; // matches aocl-dlp msz_t (pointer-width) in non-DLP stub builds
#endif
namespace zendnnl {
namespace lowoha {
namespace matmul {

using get_reorder_buff_size_func_ptr
        = msz_t (*)(const char, const char, const char, const md_t, const md_t
#if ZENDNNL_DEPENDS_AOCLDLP
                ,
                dlp_metadata_t *
#endif
        );

template <typename T>
using reorder_func_ptr = void (*)(const char, const char, const char, const T *,
        T *, const md_t, const md_t, const md_t
#if ZENDNNL_DEPENDS_AOCLDLP
        ,
        dlp_metadata_t *
#endif
);

/**
 * @brief Reorders and caches weight matrices for optimized memory access patterns
 *
 * This template function performs weight reordering to optimize memory access patterns
 * for specific GEMM kernels and implements a caching mechanism to avoid redundant
 * reordering operations for the same weight tensors. The function checks if the weights
 * have been previously reordered and cached; if so, it returns the cached version.
 * Otherwise, it performs the reordering operation and stores the result in the cache.
 *
 * @tparam T Data type of the weight elements (e.g., float, bfloat16, int8_t)
 * @param key Unique key identifying the weight tensor and reordering parameters
 * @param weights Pointer to the original (non-reordered) weight data
 * @param reorder_weights Reference to pointer that will hold the reordered weights
 * @param k Matrix dimension K (inner dimension for GEMM operation)
 * @param n Matrix dimension N (number of columns in the output matrix)
 * @param ldb Leading dimension of matrix B (must be >= k or n depending on layout)
 * @param order Memory layout order ('r' for row-major, 'c' for column-major)
 * @param trans Transpose flag ('t' for transposed, 'n' for not transposed)
 * @param mem_format_b Memory format specifier for matrix B
 * @param get_reorder_buf_size Function pointer to calculate required buffer size for reordering
 * @param reorder_func Function pointer to perform the actual reordering operation
 * @param weight_cache_type Caching strategy to use:
 *        - 0: caching disabled, reorder into a freshly allocated buffer that
 *          the caller must free.
 *        - 1: out-of-place caching. Reordered weights live in a freshly
 *          allocated buffer owned by the LRU cache; the user's weight buffer
 *          is left untouched.
 *        - 2: in-place caching. The user's weight buffer is reused as the
 *          reorder destination (a temporary buffer is used during the
 *          reorder, then copied back). The cache stores a borrowed pointer
 *          to the user's buffer, so no extra persistent allocation is kept.
 *          Falls back to out-of-place caching when the AOCL blocked size
 *          differs from the plain k*n size or the aligned allocation size
 *          would exceed the user buffer.
 * @return true if reordering was performed (cache miss), false if cached version was used (cache hit)
 */
template <typename T>
bool reorderAndCacheWeights(Key_matmul key, const void *weights,
        void *&reorder_weights, const int k, const int n, const int ldb,
        const char order, const char trans, char mem_format_b,
        get_reorder_buff_size_func_ptr get_reorder_buf_size,
        reorder_func_ptr<T> reorder_func, int weight_cache_type);

#if ZENDNNL_DEPENDS_AOCLDLP
// The new AOCL DLP reorder API dropped the dedicated DLP_SYMM_STAT_QUANT
// argument; the B-side quantization group size now travels inside the
// dlp_metadata_t (via b_quant_op->group_size), which is the sole trailing
// metadata parameter.
using get_reorder_buf_size_sym_quant_func_ptr = msz_t (*)(const char,
        const char, const char, const md_t, const md_t, dlp_metadata_t *);

template <typename T>
using reorder_sym_quant_func_ptr = void (*)(const char, const char, const char,
        const T *, T *, const md_t, const md_t, const md_t, dlp_metadata_t *);

template <typename T>
bool reorderAndCacheWeightsSymQuant(Key_matmul key, const void *weights,
        void *&reorder_weights, const int k, const int n, const int ldb,
        const char order, const char trans, char mem_format_b,
        get_reorder_buf_size_sym_quant_func_ptr get_reorder_buf_size,
        reorder_sym_quant_func_ptr<T> reorder_func, dlp_metadata_t *symq_meta,
        int weight_cache_type);
#endif

// Alignment (bytes) of the appended static-quant per-column weight-sum buffer.
// The prepack writer and the matmul reader MUST agree on this value; both
// derive it from this single constant so they can never diverge. 64 bytes is
// the AOCL AVX-512 packed-weight alignment.
constexpr size_t kStaticQuantColsumAlign = 64;

// Round @p bytes up to the next multiple of @p align (a power of two).
inline size_t round_up_to_align(size_t bytes, size_t align) {
    return (bytes + align - 1) & ~(align - 1);
}

#if ZENDNNL_DEPENDS_AOCLDLP
/**
 * @brief Byte offset of the ZenDNN-appended per-column weight-sum buffer
 *        inside a static-quant INT8 prepacked weight buffer.
 *
 * The prepack writer (lowoha_prepack.cpp) and the matmul reader (run_dlp)
 * MUST agree on this offset, so it is derived from a single expression here.
 * The offset equals the AOCL packed-weight size rounded up to
 * @ref kStaticQuantColsumAlign bytes; the @c N * int32 column-sum buffer
 * starts there.
 *
 * The weight-sum buffer is only ever produced/consumed for a u8 source (the
 * asymmetric static-quant path), so the offset is always keyed to the
 * u8s8s32os32 reorder size.
 *
 * @param order AOCL order ('r').
 * @param trans AOCL transpose flag ('t' / 'n').
 * @param k     Weight rows (K).
 * @param n     Weight cols (N).
 * @return @ref kStaticQuantColsumAlign -aligned byte offset of the column-sum
 *         buffer.
 */
inline size_t static_quant_colsum_offset(
        char order, char trans, md_t k, md_t n) {
    const size_t req = aocl_get_reorder_buf_size_u8s8s32os32(
            order, trans, 'B', k, n, nullptr);
    return round_up_to_align(req, kStaticQuantColsumAlign);
}
#endif

/**
 * @brief Widen packed signed-s4 weights to a K×N s8 buffer.
 *
 * Sign-extends each 4-bit nibble to a full int8 code (range [-8, 7]); performs
 * NO dequantization.  Shared by the W4A8 AOCL sym-quant reorder and the GGML
 * Q4_0 unpack path (which upcasts to s8 before the per-group sym-quant reorder).
 *
 * @param weights       Packed s4 source (two nibbles per byte).
 * @param wei_s8        Destination, written in K×N row-major (non-transposed).
 * @param k             Logical row count of the output.
 * @param n             Logical column count of the output.
 * @param ldb           Leading dimension of the packed source (in elements).
 * @param is_transposed true when the packed source is column-major (ba).
 */
void cvt_s4_to_s8(const int8_t *weights, int8_t *wei_s8, int k, int n, int ldb,
        bool is_transposed);

/** Clear AOCL matmul weight caches and zero-point compensation LRU cache. */
void clear_aocl_matmul_weight_caches();
/// W4A8 s4→s8 expansion + cache (plain row-major, NO blocked reorder).
/// Returns a pointer to a plain [k, n] s8 buffer stored in a dedicated
/// process-lifetime LRU.  The pointer is stable across calls for the same
/// key (original s4 weight pointer + shape).  Used by the ALGO 3 N-tile
/// path which needs a column-sliceable s8 buffer for per-tile reordering.
/// @param[out] s8_plain  Set to the cached plain s8 buffer on success,
///                       nullptr on failure.
void w4a8_cvt_and_cache_plain_s8(Key_matmul key, const int8_t *weights,
        void *&s8_plain, int k, int n, int ldb, bool is_transposed);

/// High-level plain-s8 materialization: for every W4A8 expert in the group,
/// populates the plain-s8 LRU (cvt_s4_to_s8 cached) and fills
/// `w4a8_s8_out[e]` with the cached s8 pointer.  Iterates ALL experts
/// (including M[e]==0 cold experts) so rotating MoE doesn't pay a
/// first-fire conversion spike.  Does NOT mutate `weight[]` or `params`.
/// @param[out] w4a8_s8_out  Resized to num_ops; non-W4A8 slots are nullptr.
/// @param[out] any_w4a8     Set to true if at least one expert was W4A8.
void w4a8_populate_plain_s8_cache(const std::vector<const void *> &weight,
        const std::vector<int> &K, const std::vector<int> &N,
        const std::vector<int> &ldb, const std::vector<bool> &transB,
        const std::vector<matmul_params> &params, int num_ops,
        std::vector<void *> &w4a8_s8_out, bool &any_w4a8);

/// W4A8 weight reorder + cache: converts s4→s8 then packs through the
/// AOCL sym-quant s8s8s32os32 path into the dedicated W4A8 LRU cache.
/// Called by run_dlp at GEMM time; also callable from prepack to eagerly
/// warm the cache for all experts before inference begins.
void w4a8ReorderAndCacheWeightsAocl(Key_matmul key, const int8_t *weights,
        void *&reorder_weights, const int k, const int n, const int ldb,
        const bool is_weights_const, const char order, const char trans,
        data_type_t wei_dt, data_type_t src_dt, int weight_cache_type,
        int sym_quant_group_size);

/// Broadcast per-token/per-tensor src_scale to per-group {M, G} shape
/// so AOCL sym-quant derives the correct group_size = K/G.
status_t broadcast_w4a8_src_scale(
        matmul_params &params, int M, std::vector<uint8_t> &expanded_src_scale);

/**
 * @brief Execute single matrix multiplication using AOCL DLP backend
 *
 * Performs C = alpha * op(A) * op(B) + beta * C using AMD's optimized
 * AOCL library with support for post-operations and bias addition.
 * Automatically dispatches to appropriate data type specializations.
 *
 * @param layout Memory layout ('r' for row-major, 'c' for column-major)
 * @param transA Transpose flag for matrix A ('t' for transpose, 'n' for no transpose)
 * @param transB Transpose flag for matrix B ('t' for transpose, 'n' for no transpose)
 * @param M Number of rows in matrix A and output matrix C
 * @param N Number of columns in matrix B and output matrix C
 * @param K Inner dimension (columns of A, rows of B after potential transpose)
 * @param alpha Scaling factor for the product of A and B
 * @param beta Scaling factor for the existing values in C
 * @param lda Leading dimension of matrix A (stride between rows/columns)
 * @param ldb Leading dimension of matrix B (stride between rows/columns)
 * @param ldc Leading dimension of matrix C (stride between rows/columns)
 * @param mem_format_a Memory format specifier for matrix A
 * @param mem_format_b Memory format specifier for matrix B
 * @param A Pointer to matrix A data buffer
 * @param B Pointer to matrix B (weights) data buffer
 * @param C Pointer to matrix C (output) data buffer
 * @param dtypes Data types structure specifying src, weight, and dst tensor types
 * @param lowoha_param Parameters containing the post-operations chain
 * @param bias Optional bias vector pointer (can be nullptr if no bias)
 * @param kernel Algorithm selection for GEMM execution
 * @param is_weights_const Flag indicating if weights are constant (enables caching)
 */
void run_dlp(char layout, char transA, char transB, int M, int N, int K,
        float alpha, float beta, int lda, int ldb, int ldc, char mem_format_a,
        char mem_format_b, const void *A, const void *B, void *C,
        const matmul_data_types &dtypes, const matmul_params &lowoha_param,
        const void *bias, zendnnl::ops::matmul_algo_t kernel,
        bool is_weights_const);

/**
 * @brief Execute batched matrix multiplication using AOCL backend
 *
 * @param layout Memory layout ('r' for row-major, 'c' for column-major)
 * @param transA Transpose flag for matrices A ('t' for transpose, 'n' for no transpose)
 * @param transB Transpose flag for matrices B ('t' for transpose, 'n' for no transpose)
 * @param M Number of rows in each A matrix and output C matrix
 * @param N Number of columns in each B matrix and output C matrix
 * @param K Inner dimension for each GEMM operation
 * @param alpha Scaling factor applied to all A*B products across all batches
 * @param A Base pointer to the first batch of matrix A data
 * @param lda Leading dimension of each matrix A
 * @param B Base pointer to the first batch of matrix B (weights) data
 * @param ldb Leading dimension of each matrix B
 * @param beta Scaling factor applied to all existing C values across all batches
 * @param C Base pointer to the first batch of matrix C (output) data
 * @param ldc Leading dimension of each matrix C
 * @param dtypes Data types structure specifying src, weight, and dst tensor types
 * @param batch_count Number of independent matrix multiplications to perform
 * @param Batch_A Number of A matrices (1 for broadcasting A across all batches)
 * @param Batch_B Number of B matrices (1 for broadcasting B across all batches)
 * @param mem_format_a Memory format specifier for matrices A
 * @param mem_format_b Memory format specifier for matrices B
 * @param src_stride Byte offset between consecutive A matrices in the batch
 * @param weight_stride Byte offset between consecutive B matrices in the batch
 * @param dst_stride Byte offset between consecutive C matrices in the batch
 * @param lowoha_param Parameters containing post-operations chain applied to all batches
 * @param bias Optional bias vector pointer applied to all batches (can be nullptr)
 */
void matmul_batch_gemm_wrapper(char layout, char transA, char transB, int M,
        int N, int K, float alpha, const void *A, int lda, const void *B,
        int ldb, float beta, void *C, int ldc, matmul_data_types &dtypes,
        int batch_count, int Batch_A, int Batch_B, char mem_format_a,
        char mem_format_b, size_t src_stride, size_t weight_stride,
        size_t dst_stride, const matmul_params &lowoha_param, const void *bias,
        int num_threads);

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif //_AOCL_KERNEL_HPP
