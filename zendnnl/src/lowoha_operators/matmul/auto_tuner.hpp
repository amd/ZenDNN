/********************************************************************************
# * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef _AUTO_TUNER_HPP
#define _AUTO_TUNER_HPP

#include "lowoha_operators/matmul/lowoha_common.hpp"
#include "lowoha_operators/matmul/lowoha_matmul.hpp"

#include <unordered_map>
#include <vector>
#include <cmath>
#include <climits>
#include <utility>
#include <tuple>

namespace zendnnl {
namespace lowoha {

/**
 * @brief Number of initial warmup iterations before algorithm evaluation begins.
 *
 * During these skip iterations, the autotuner uses a different algorithm
 * to warm up caches and stabilize the system before
 * performance measurements are taken.
 *
 * Can be overridden via environment variable: ZENDNNL_MATMUL_SKIP_ITER
 */
#define MATMUL_SKIP_ITER 2

/**
 * @brief Number of iterations used to evaluate and compare algorithm performance.
 *
 * During the evaluation phase, the autotuner cycles through available algorithms
 * (NUM_OF_ALGO) and measures their execution times over this many iterations.
 * The algorithm with the best average performance is selected and cached.
 *
 * Can be overridden via environment variable: ZENDNNL_MATMUL_EVAL_ITER
 */
#define MATMUL_EVALUATE_ITER 3

/**
 * @brief Total number of matmul algorithms available for autotuning.
 *
 * Currently supports two algorithms:
 * 1. oneDNN blocked algorithm
 * 2. AOCL BLIS blocked algorithm
 *
 * This value is used to cycle through algorithms during the evaluation phase
 * using modulo arithmetic: (iteration_count % NUM_OF_ALGO) + 1
 */
#define NUM_OF_ALGO 2

/**
 * @brief Auto-tunes and executes matrix multiplication using the optimal algorithm
 *
 * This function implements an auto-tuning mechanism to select the best-performing
 * matrix multiplication algorithm for the given operation parameters. It uses a
 * multi-phase approach:
 *
 * 1. Skip Phase: Runs initial iterations with different algorithm
 * 2. Evaluation Phase: Tests multiple algorithms and measures their execution times
 * 3. Execution Phase: Uses the cached best-performing algorithm for subsequent calls
 *
 * The function maintains an internal cache (static unordered_map) that stores the
 * optimal algorithm choice for each unique combination of matrix parameters. This
 * enables efficient algorithm selection without repeated benchmarking after the
 * initial tuning phase completes.
 *
 * Environment Variables:
 * - ZENDNNL_MATMUL_SKIP_ITER: Number of skip iterations (default: 2)
 * - ZENDNNL_MATMUL_EVAL_ITER: Number of evaluation iterations (default: 3)
 *
 * @param layout Memory layout ('r' for row-major, 'c' for column-major)
 * @param transA Transpose flag for matrix A ('t' for transpose, 'n' for no transpose)
 * @param transB Transpose flag for matrix B ('t' for transpose, 'n' for no transpose)
 * @param M Number of rows in matrix A and output matrix C
 * @param N Number of columns in matrix B and output matrix C
 * @param K Inner dimension (columns of A, rows of B after potential transpose)
 * @param alpha Scaling factor for the product of A and B
 * @param A Pointer to matrix A data buffer
 * @param lda Leading dimension of matrix A
 * @param B Pointer to matrix B (weights) data buffer
 * @param ldb Leading dimension of matrix B
 * @param beta Scaling factor for the existing values in C
 * @param C Pointer to matrix C (output) data buffer
 * @param ldc Leading dimension of matrix C
 * @param dtypes Data types structure specifying src, weight, and dst tensor types
 * @param kernel Initial algorithm hint (may be overridden by auto-tuner)
 * @param mem_format_a Memory format specifier for matrix A
 * @param mem_format_b Memory format specifier for matrix B
 * @param lowoha_param Parameters containing post-operations chain
 * @param batch_params Batch parameters for batched matrix multiplication
 * @param bias Optional bias vector pointer (can be nullptr)
 * @param is_weights_const Flag indicating if weights are constant (considers weight address as param for key)
 * @return matmul_algo_t The selected algorithm that was used for execution
 */
matmul_algo_t auto_compute_matmul_v1(char layout, char transA, char transB,
                                     int M,
                                     int N, int K, float alpha, const void *A, int lda, const void *B, int ldb,
                                     float beta, void *C, int ldc, data_types &dtypes,
                                     zendnnl::ops::matmul_algo_t kernel, char mem_format_a, char mem_format_b,
                                     lowoha_params &lowoha_param, batch_params_t &batch_params,
                                     const void *bias, bool is_weights_const);

/**
 * @brief Auto-tunes and executes matrix multiplication using alternative tuning strategy
 *
 * This is an alternative implementation of the auto-tuning mechanism for matrix
 * multiplication that uses a different strategy for algorithm selection. Unlike
 * auto_compute_matmul which stores the best algorithm directly, this version
 * maintains execution time statistics in a vector and computes the optimal
 * algorithm selection based on accumulated timing data.
 *
 * Key differences from auto_compute_matmul:
 * - Uses iteration counting approach with simpler map structure
 * - Accumulates timing data in a vector for post-analysis
 * - Selects best algorithm for complete workload after evaluation phase completes
 * - May provide different performance characteristics for certain workloads
 *
 * The function follows the same multi-phase approach:
 * 1. Skip Phase: Initial warmup iterations
 * 2. Evaluation Phase: Collect timing data for different algorithms
 * 3. Execution Phase: Use the algorithm with minimum accumulated time
 *
 * Environment Variables:
 * - ZENDNNL_MATMUL_SKIP_ITER: Number of skip iterations (default: 2)
 * - ZENDNNL_MATMUL_EVAL_ITER: Number of evaluation iterations (default: 3)
 *
 * @param layout Memory layout ('r' for row-major, 'c' for column-major)
 * @param transA Transpose flag for matrix A ('t' for transpose, 'n' for no transpose)
 * @param transB Transpose flag for matrix B ('t' for transpose, 'n' for no transpose)
 * @param M Number of rows in matrix A and output matrix C
 * @param N Number of columns in matrix B and output matrix C
 * @param K Inner dimension (columns of A, rows of B after potential transpose)
 * @param alpha Scaling factor for the product of A and B
 * @param A Pointer to matrix A data buffer
 * @param lda Leading dimension of matrix A
 * @param B Pointer to matrix B (weights) data buffer
 * @param ldb Leading dimension of matrix B
 * @param beta Scaling factor for the existing values in C
 * @param C Pointer to matrix C (output) data buffer
 * @param ldc Leading dimension of matrix C
 * @param dtypes Data types structure specifying src, weight, and dst tensor types
 * @param kernel Initial algorithm hint (may be overridden by auto-tuner)
 * @param mem_format_a Memory format specifier for matrix A
 * @param mem_format_b Memory format specifier for matrix B
 * @param lowoha_param Parameters containing post-operations chain
 * @param batch_params Parameters related to batching
 * @param bias Optional bias vector pointer (can be nullptr)
 * @param is_weights_const Flag indicating if weights are constant (considers weight address as param for key)
 * @return matmul_algo_t The selected algorithm that was used for execution
 */
matmul_algo_t auto_compute_matmul_v2(char layout, char transA, char transB,
                                     int M,
                                     int N, int K, float alpha, const void *A, int lda, const void *B, int ldb,
                                     float beta, void *C, int ldc, data_types &dtypes,
                                     zendnnl::ops::matmul_algo_t kernel, char mem_format_a, char mem_format_b,
                                     lowoha_params &lowoha_param, batch_params_t &batch_params,
                                     const void *bias, bool is_weights_const);
} // namespace lowoha
} // namespace zendnnl

#endif //_AUTO_TUNER_HPP