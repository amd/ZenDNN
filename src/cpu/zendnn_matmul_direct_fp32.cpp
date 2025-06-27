/*******************************************************************************
* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
*******************************************************************************/


#include "zendnn.hpp"
#include <algorithm>
#include <cmath>
#include "common/zendnn_lru_cache.hpp"
#include <stdio.h>
#include <omp.h>
#include "zendnn_matmul_direct_fp32.hpp"


#ifndef ZENDNN_USE_AOCL_BLIS_API
    #include <cblas.h>
#else // ZENDNN_USE_AOCL_BLIS_API
    #include "cblas_with_blis_api.hpp"
#endif // ZENDNN_USE_AOCL_BLIS_API


extern std::mutex map_mutex;

namespace zendnn {


int get_env_variable_as_int(const char *var_name, int default_value) {
    const char *env_val = std::getenv(var_name);
    return env_val ? std::atoi(env_val) : default_value;
}

void configure_tile_size(int tile_algo, int &MR, int &NR) {
    NR = 16; // NR is fixed
    switch (tile_algo) {
    case 1:
        MR = 1;
        break;
    case 2:
        MR = 4;
        break;
    case 3:
        MR = 6;
        break;
    case 4:
        MR = 8;
        break;
    case 5:
        MR = 12;
        break;
    default:
        zendnnInfo(ZENDNN_PROFLOG, "Unknown kernel ID. Using default MR=6, NR=16.");
        MR = 6;
    }
}

float *get_transposed_weight_if_needed(float *weight, int K, int N,
                                       bool transB) {
    if (!transB) {
        return weight;
    }

    int zenWeightCache = zendnn_getenv_int("ZENDNN_WEIGHT_CACHING", 1);

    static zendnn::impl::lru_weight_cache_t<float *, float *> matmul_weight_cache;
    float *weightCache = (float *)malloc(K * N * sizeof(float));
    auto found_obj = matmul_weight_cache.find_key(weight);

    if (!found_obj || zenWeightCache == 0) {
        std::lock_guard<std::mutex> lock(map_mutex);
        transpose_matrix(weight, weightCache, N, K);
        matmul_weight_cache.add(weight, weightCache);
    }
    else {
        weightCache = matmul_weight_cache.get(weight);
    }

    return weightCache;
}

/*
    This function evaluates the matrix dimensions (M, N, K) and estimates:
    - Total FLOPs (to gauge problem size)
    - Tile working set size (based on a 12x16 kernel)
    - Total B matrix size (to assess cache pressure)
    - Number of column tiles (to estimate reuse potential of B and C)

   For a kernel with tile size MR × NR, the memory footprint (in bytes) of a single tile is:
   Tile Size=sizeof(float)×(A_tile+B_tile+C_tile)
   Where:
    A_tile = MR × K
    B_tile = K × NR
    C_tile = MR × NR (output tile)
    sizeof(float) = 4 bytes

   For 12x16:
    A_tile: 12 rows × K columns
    B_tile: K rows × 16 columns
    C_tile: 12 rows × 16 columns
    Total: 4 * (28 * K + 192) bytes
*/
size_t zendnn_custom_op::matmul_direct_select_kernel(int M, int N, int K) {
    size_t flops = static_cast<size_t>(2) * M * N * K;

    // Cache sizes
    constexpr size_t L2_CACHE_SIZE = 1 * 1024 * 1024; // 1 MB

    // FLOPs thresholds
    constexpr size_t FLOP_THRESHOLD_LOW = 1e6;
    constexpr size_t FLOP_THRESHOLD_HIGH = 1e8;

    // Estimate tile size for 12x16 kernel
    size_t tile_size_12x16 = sizeof(float) * (12 * K + K * 16 + 12 * 16);

    // Estimate total B matrix size
    size_t total_B_size = sizeof(float) * K * N;

    // Estimate number of column tiles
    // Kernel processes 16 columns at a time (NR = 16).
    // Tiles_N is large, it means: More unique B and C tiles, Higher cache pressure, Lower reuse of B tiles
    int tiles_N = (N + 15) / 16;

    // Decision logic
    // Case 1: Very small problem — custom kernel is faster due to lower overhead
    if (flops < FLOP_THRESHOLD_LOW) {
        return 1;
    }

    // Case 2: Very large problem with large B — BLAS likely more efficient due to prepacking
    if (flops > FLOP_THRESHOLD_HIGH && K > 512 && N > 512) {
        return 0;
    }

    // Case 3: If tile fits in L2 and B matrix is not too large, use custom kernel
    // This is a heuristic to allow some reuse of B tiles without evicting them too quickly.
    // This assumes that up to 8 tiles (128 columns) can be reasonably reused from L2 cache.
    // Can tune this multiplier (2x, 3x, etc.) based on profiling.
    if (tile_size_12x16 <= L2_CACHE_SIZE &&
            total_B_size <= 2 * L2_CACHE_SIZE &&
            tiles_N <= 8) {
        return 1;
    }

    return 0;
}

void execute_aocl_gemm(char order, char transa, char transb,
                       int Batch_A, int Batch_B, int M, int N, int K, float alpha, float beta,
                       const void *src, const void *weight, void *dst, float *bias, data_types dt,
                       char mem_format_a, char mem_format_b, const int lda, const int ldb,
                       const int ldc, int thread_qty,
                       ActivationPostOp activation_post_op = ActivationPostOp::NONE) {
    // Create aocl_post_op
    aocl_post_op post_op = {};

    int postop_count = alpha != 1 ? 1 : 0;
    if (bias != nullptr) {
        ++postop_count;
    }
    if (activation_post_op != ActivationPostOp::NONE) {
        ++postop_count;
    }

    post_op.seq_length = postop_count;
    AOCL_POST_OP_TYPE seq_vector[postop_count];
    post_op.bias = nullptr;
    post_op.eltwise = nullptr;

    int post_op_index = 0;
    float *bias_ = nullptr;
    if (dt.src_dt == zendnn_bf16) {
        if (bias != NULL) {
            // Add bias postop
            seq_vector[post_op_index++] = AOCL_POST_OP_TYPE::BIAS;
            post_op.bias = new aocl_post_op_bias{bias, dt.bia_dt == zendnn_bf16 ? AOCL_PARAMS_STORAGE_TYPES::AOCL_GEMM_BF16 : AOCL_PARAMS_STORAGE_TYPES::AOCL_GEMM_F32 };
            if (post_op.bias == NULL) {
                std::free(seq_vector);
                return;
            }
        }
        //Scale post-op
        if (alpha != 1.0) {
            post_op.sum = (aocl_post_op_sum *) malloc(sizeof(
                              aocl_post_op_sum));

            if (post_op.sum == NULL) {
                if (post_op.bias != NULL) {
                    delete post_op.bias;
                }
                std::free(post_op.seq_vector);
                return;
            }
            seq_vector[post_op_index++] = SCALE;
            (post_op.sum)->is_power_of_2 = FALSE;
            (post_op.sum)->scale_factor = NULL;
            (post_op.sum)->buff = NULL;
            (post_op.sum)->zero_point = NULL;

            (post_op.sum)->scale_factor = malloc(sizeof(float));
            (post_op.sum)->zero_point = malloc(sizeof(float));

            //SCALE
            float *temp_dscale_ptr = (float *)(post_op.sum)->scale_factor;
            float *temp_dzero_point_ptr = (float *)(post_op.sum)->zero_point;
            temp_dscale_ptr[0] = (float)(alpha);

            temp_dzero_point_ptr[0] = (float)0;

            (post_op.sum)->scale_factor_len = 1;
            (post_op.sum)->zero_point_len = 1;
            (post_op.sum)->sf_stor_type = AOCL_GEMM_F32;
            (post_op.sum)->zp_stor_type = AOCL_GEMM_F32;
        }
    }
    else {
        if (bias != nullptr) {
            // Add bias postop
            if (alpha != 1.0f) {
                bias_ = new float[N]();
                #pragma omp parallel for num_threads(omp_get_max_threads())
                for (int i = 0; i < N; ++i) {
                    bias_[i] = alpha * bias[i];
                }
            }
            post_op.bias = new aocl_post_op_bias{ (alpha != 1.0f) ? bias_ : bias, AOCL_PARAMS_STORAGE_TYPES::AOCL_GEMM_F32 };
            seq_vector[post_op_index++] = AOCL_POST_OP_TYPE::BIAS;
        }
    }
    if (activation_post_op != ActivationPostOp::NONE) {
        switch (activation_post_op) {
        case ActivationPostOp::RELU:
            post_op.eltwise = new aocl_post_op_eltwise{false, nullptr, 0, {nullptr, nullptr, AOCL_ELT_ALGO_TYPE::RELU}};
            seq_vector[post_op_index++] = AOCL_POST_OP_TYPE::ELTWISE;
            break;
        case ActivationPostOp::SIGMOID:
            post_op.eltwise = new aocl_post_op_eltwise{false, nullptr, 0, {nullptr, nullptr, AOCL_ELT_ALGO_TYPE::SIGMOID}};
            seq_vector[post_op_index++] = AOCL_POST_OP_TYPE::ELTWISE;
            break;
        case ActivationPostOp::TANH:
            post_op.eltwise = new aocl_post_op_eltwise{false, nullptr, 0, {nullptr, nullptr, AOCL_ELT_ALGO_TYPE::TANH}};
            seq_vector[post_op_index++] = AOCL_POST_OP_TYPE::ELTWISE;
            break;
        case ActivationPostOp::GELU_TANH:
            post_op.eltwise = new aocl_post_op_eltwise{false, nullptr, 0, {nullptr, nullptr, AOCL_ELT_ALGO_TYPE::GELU_TANH}};
            seq_vector[post_op_index++] = AOCL_POST_OP_TYPE::ELTWISE;
            break;
        case ActivationPostOp::GELU_ERF:
            post_op.eltwise = new aocl_post_op_eltwise{false, nullptr, 0, {nullptr, nullptr, AOCL_ELT_ALGO_TYPE::GELU_ERF}};
            seq_vector[post_op_index++] = AOCL_POST_OP_TYPE::ELTWISE;
            break;
        case ActivationPostOp::SILU:
            post_op.eltwise = new aocl_post_op_eltwise{false, nullptr, 0, {malloc(sizeof(float)), nullptr, AOCL_ELT_ALGO_TYPE::SWISH}};
            *((float *)(post_op.eltwise->algo.alpha)) = 1.0f;
            seq_vector[post_op_index++] = AOCL_POST_OP_TYPE::ELTWISE;
            break;
        }
    }

    post_op.seq_vector = seq_vector;

    if (Batch_A == 1 && Batch_B == 1) {
        zendnnInfo(ZENDNN_CORELOG,
                   "Running single AOCL GEMM kernel");
        if (dt.src_dt == zendnn_f32) {
            // Use float32 kernel
            aocl_gemm_f32f32f32of32(order, transa, transb, M, N, K, alpha,
                                    (float *)src, lda, mem_format_a,
                                    (float *)weight, ldb, mem_format_b,
                                    beta, (float *)dst, ldc, &post_op);
        }
        else if (dt.src_dt == zendnn_bf16) {
            if (dt.dst_dt == zendnn_bf16) {
                // Use bf16->bf16 kernel
                aocl_gemm_bf16bf16f32obf16(order, transa, transb, M, N, K, 1.0,
                                           (int16_t *)src, lda, mem_format_a,
                                           (int16_t *)weight, ldb, mem_format_b,
                                           beta, (int16_t *)dst, ldc, &post_op);
            }
            else {
                // Use bf16->f32 kernel
                aocl_gemm_bf16bf16f32of32(order, transa, transb, M, N, K, 1.0,
                                          (int16_t *)src, lda, mem_format_a,
                                          (int16_t *)weight, ldb, mem_format_b,
                                          beta, (float *)dst, ldc, &post_op);
            }
        }
        else {
            zendnnInfo(ZENDNN_PROFLOG, "Unsupported data type combination for AOCL GEMM.");
            return;
        }
    }
    else if (Batch_A > 1 || Batch_B > 1) {
        zendnnInfo(ZENDNN_CORELOG,
                   "Running batched AOCL GEMM kernel");
        uint batch_size = std::max(Batch_A, Batch_B);
        uint offset_src = transa == 't' ? lda * K : M * lda;
        uint offset_wei = transb == 't' ? ldb * N : K * ldb;
        if (thread_qty == 1) {
            for (int i = 0; i < batch_size; ++i) {
                if (dt.src_dt == zendnn_f32) {
                    // Use float32 kernel
                    aocl_gemm_f32f32f32of32(order, transa, transb, M, N, K, alpha,
                                            Batch_A == 1 ? (float *)src : (float *)src + i * offset_src, lda, mem_format_a,
                                            Batch_B == 1 ? (float *)weight : (float *)weight + i * offset_wei, ldb,
                                            mem_format_b,
                                            beta, (float *)dst + i * M * ldc, ldc, &post_op);
                }
                else if (dt.src_dt == zendnn_bf16) {
                    if (dt.dst_dt == zendnn_bf16) {
                        // Use bf16->bf16 kernel
                        aocl_gemm_bf16bf16f32obf16(order, transa, transb, M, N, K, 1.0,
                                                   Batch_A == 1 ? (int16_t *)src : (int16_t *)src + i * offset_src, lda,
                                                   mem_format_a,
                                                   Batch_B == 1 ? (int16_t *)weight : (int16_t *)weight + i * offset_wei, ldb,
                                                   mem_format_b,
                                                   beta, (int16_t *)dst + i * M * ldc, ldc, &post_op);
                    }
                    else {
                        // Use bf16->f32 kernel
                        aocl_gemm_bf16bf16f32of32(order, transa, transb, M, N, K, 1.0,
                                                  Batch_A == 1 ? (int16_t *)src : (int16_t *)src + i * offset_src, lda,
                                                  mem_format_a,
                                                  Batch_B == 1 ? (int16_t *)weight : (int16_t *)weight + i * offset_wei, ldb,
                                                  mem_format_b,
                                                  beta, (float *)dst + i * M * ldc, ldc, &post_op);
                    }
                }
            }
        }
        else {
            omp_set_max_active_levels(1);
            #pragma omp parallel for num_threads(thread_qty)
            for (int i = 0; i < batch_size; ++i) {
                if (dt.src_dt == zendnn_f32) {
                    // Use float32 kernel
                    aocl_gemm_f32f32f32of32(order, transa, transb, M, N, K, alpha,
                                            Batch_A == 1 ? (float *)src : (float *)src + i * offset_src, lda, mem_format_a,
                                            Batch_B == 1 ? (float *)weight : (float *)weight + i * offset_wei, ldb,
                                            mem_format_b,
                                            beta, (float *)dst + i * M * ldc, ldc, &post_op);
                }
                else if (dt.src_dt == zendnn_bf16) {
                    if (dt.dst_dt == zendnn_bf16) {
                        // Use bf16->bf16 kernel
                        aocl_gemm_bf16bf16f32obf16(order, transa, transb, M, N, K, 1.0,
                                                   Batch_A == 1 ? (int16_t *)src : (int16_t *)src + i * offset_src, lda,
                                                   mem_format_a,
                                                   Batch_B == 1 ? (int16_t *)weight : (int16_t *)weight + i * offset_wei, ldb,
                                                   mem_format_b,
                                                   beta, (int16_t *)dst + i * M * ldc, ldc, &post_op);
                    }
                    else {
                        // Use bf16->f32 kernel
                        aocl_gemm_bf16bf16f32of32(order, transa, transb, M, N, K, 1.0,
                                                  Batch_A == 1 ? (int16_t *)src : (int16_t *)src + i * offset_src, lda,
                                                  mem_format_a,
                                                  Batch_B == 1 ? (int16_t *)weight : (int16_t *)weight + i * offset_wei, ldb,
                                                  mem_format_b,
                                                  beta, (float *)dst + i * M * ldc, ldc, &post_op);
                    }
                }
            }
        }
    }
    else {
        zendnnInfo(ZENDNN_PROFLOG,
                   "Invalid batch size. Exiting...");
    }
    // Clean-up allocated memory
    if (bias_ != nullptr) {
        delete bias_;
    }
    if (dt.src_dt == zendnn_bf16) {
        if (post_op.sum != nullptr) {
            if ((post_op.sum)->scale_factor != nullptr) {
                free((post_op.sum)->scale_factor);
            }
            if ((post_op.sum)->zero_point != nullptr) {
                free((post_op.sum)->zero_point);
            }
            free(post_op.sum);
        }
    }
    if (post_op.bias != nullptr) {
        delete post_op.bias;
    }
    if (activation_post_op != ActivationPostOp::NONE &&
            post_op.eltwise != nullptr) {
        delete post_op.eltwise;
    }
}
std::string getActivationPostOpName(ActivationPostOp post_op) {
    switch (post_op) {
    case ActivationPostOp::NONE:
        return "NONE";
    case ActivationPostOp::RELU:
        return "RELU";
    case ActivationPostOp::SIGMOID:
        return "SIGMOID";
    case ActivationPostOp::TANH:
        return "TANH";
    case ActivationPostOp::GELU_TANH:
        return "GELU_TANH";
    case ActivationPostOp::GELU_ERF:
        return "GELU_ERF";
    case ActivationPostOp::SILU:
        return "SILU";
    default:
        return "UNKNOWN";
    }
}
void zendnn_custom_op::zendnn_matmul_direct_fp32(const void *src,
        const void *weight, void *dst,
        const void *bias, float alpha, float beta,
        int M, int N, int K, bool transA, bool transB, int lda, int ldb, int ldc,
        data_types dt,
        ActivationPostOp post_op, int Batch_A, int Batch_B) {

//Input validation
    if (!src || !weight || !dst || M <= 0 || N <= 0 || K <= 0 || Batch_A <= 0 ||
            Batch_B <= 0) {
        zendnnInfo(ZENDNN_PROFLOG,
                   "Invalid input parameters. Aborting........");
        return;
    }
    zendnn::zendnnEnv zenEnvObj = readEnv();
    uint thread_qty = zenEnvObj.omp_num_threads;

    int MR = 6, NR = 16;

    uint algo_type = get_env_variable_as_int("ZENDNN_MATMUL_DIRECT_ALGO", 5);
    uint tile_algo = get_env_variable_as_int("ZENDNN_MATMUL_DIRECT_TILE_ALGO", 3);
    configure_tile_size(tile_algo, MR, NR);

    bool check_ld = (transA) || (transB) || (ldc > N) ||
                    (transB ? ldb > K : ldb > N)  ||
                    (lda > K) || (thread_qty > 1) || (Batch_A != Batch_B) ||
                    (dt.src_dt == zendnn_bf16);

    auto start_ms = std::chrono::high_resolution_clock::now();
    if (check_ld) {
        execute_aocl_gemm('r', transA ? 't' : 'n', transB ? 't' : 'n', Batch_A, Batch_B,
                          M, N, K, alpha, beta, src, weight, dst,
                          (float *)bias, dt,
                          'n', 'n', lda, ldb, ldc, thread_qty, post_op);
    }
    else {
        if (algo_type == 1) {
            zendnn_registerBlocking_kernel_fp32::matmul_avx512_fp32_registerBlocking_batch((
                        float *)src, (float *)weight,
                    (float *)dst, (float *)bias, alpha, beta, M, N, K,
                    MR, NR, transB, post_op, Batch_A);
        }
        else if (algo_type == 2) {
            zendnn_registerBlocking_kernel_fp32::matmul_avx512_fp32_registerBlocking_auto_batch((
                        float *)src, (float *)weight,
                    (float *)dst, (float *)bias, alpha, beta, M, N, K,
                    transB, post_op, Batch_A);
        }
        else if (algo_type == 3) {
            zendnn_registerBlocking_kernel_fp32_batch::matmul_avx512_fp32_registerBlocking_batched((
                        float *)src, (float *)weight,
                    (float *)dst, (float *)bias, alpha, beta, M, N, K, MR, NR,
                    transB, post_op, Batch_A);
        }
        else if (algo_type == 4) {
            zendnn_registerBlocking_kernel_fp32_batch::matmul_avx512_fp32_registerBlocking_batched_auto((
                        float *)src, (float *)weight,
                    (float *)dst, (float *)bias, alpha, beta, M, N, K,
                    transB, post_op, Batch_A);
        }
        else if (algo_type == 5) {
            execute_aocl_gemm('r', transA ? 't' : 'n', transB ? 't' : 'n', Batch_A, Batch_B,
                              M, N, K, alpha, beta, src, weight, dst,
                              (float *)bias, dt,
                              'n', 'n', lda, ldb, ldc, thread_qty, post_op);
        }
        else if (algo_type == 6) {
            zendnn_registerBlocking_kernel_fp32_ref::matmul_avx512_fp32_registerBlocking_batch((
                        float *)src, (float *)weight,
                    (float *)dst, (float *)bias, alpha, beta, M, N, K,
                    MR, NR, transB, post_op, Batch_A);
        }
        else {
            zendnn_registerBlocking_kernel_fp32_ref::matmul_avx512_fp32_registerBlocking_auto_batch((
                        float *)src, (float *)weight,
                    (float *)dst, (float *)bias, alpha, beta, M, N, K,
                    transB, post_op, Batch_A);
        }
    }

    // Code for time profiling of this kernel
    auto end_ms = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration<double, std::milli>(end_ms -
                       start_ms).count();

    zendnnVerbose(ZENDNN_PROFLOG,
                  "zenDirectMatMul, Layout=CblasRowMajor group_count=1 group_size[0]=",
                  std::max(Batch_A, Batch_B),
                  " M_Array[0]=", M, " N_Array[0]=", N,
                  " K_Array[0]=", K, " alpha_Array[0]=", alpha,
                  " beta_Array[0]=", beta, " lda=", lda, " ldb=", ldb, " ldc=", ldc,
                  " PostOp=", getActivationPostOpName(post_op), " Time=",
                  duration_ms, "ms");
}


void zendnn_custom_op::zendnn_batched_matmul_fp32(const std::vector<float *>
        &src_batch,
        const std::vector<float *> &weight_batch,
        std::vector<float *> &dst_batch,
        const std::vector<float *> &bias_batch,
        const std::vector<float> &alpha_array,
        const std::vector<float> &beta_array,
        const std::vector<int> &m_array,
        const std::vector<int> &n_array,
        const std::vector<int> &k_array,
        const std::vector<bool> &transB_array,
        const std::vector<ActivationPostOp> &post_op_array,
        int group_count,
        const std::vector<int> &group_size_array) {
    int idx = 0;
    for (int group_idx = 0; group_idx < group_count; ++group_idx) {
        float alpha = alpha_array[group_idx];
        float beta = beta_array[group_idx];
        int M = m_array[group_idx];
        int N = n_array[group_idx];
        int K = k_array[group_idx];
        bool transB = transB_array[group_idx];
        ActivationPostOp post_op = post_op_array[group_idx];
        data_types dt;
        for (int matrix_idx = 0; matrix_idx < group_size_array[group_idx];
                ++matrix_idx) {
            if (!src_batch[idx] || !weight_batch[idx] || !dst_batch[idx]) {
                std::cerr << "Null pointer detected at index " << idx << std::endl;
                return;
            }
            zendnn_matmul_direct_fp32(src_batch[idx], weight_batch[idx],
                                      dst_batch[idx], bias_batch[idx],
                                      alpha, beta, M, N, K, false, transB, K, N, N, dt, post_op, 1);
            idx++;
        }
    }
}

}