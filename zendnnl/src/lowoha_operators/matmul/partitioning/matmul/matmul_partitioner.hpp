/*******************************************************************************
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

#ifndef MATMUL_PARTITIONER_HPP
#define MATMUL_PARTITIONER_HPP

#include "lowoha_operators/matmul/lowoha_common.hpp"

#include <functional>
#include <utility>

namespace zendnnl {
namespace lowoha {
namespace matmul {

/**
 * @brief Configuration structure for matrix multiplication partitioning
 *
 * Contains all necessary parameters for partitioning and executing
 * matrix multiplication operations across multiple threads and tiles.
 */
struct matmul_partition_config_t {
  int M;                      ///< Number of rows in matrix A
  int N;                      ///< Number of columns in matrix B
  int K;                      ///< Inner dimension (columns of A, rows of B)
  int num_threads;            ///< Number of OpenMP threads to use
  matmul_algo_t kernel;       ///< Selected kernel algorithm
  size_t src_type_size;       ///< Size in bytes of source data type
  size_t out_type_size;       ///< Size in bytes of output data type
  int lda;                    ///< Leading dimension of matrix A
  int ldb;                    ///< Leading dimension of matrix B
  int ldc;                    ///< Leading dimension of matrix C (output)
  bool transA;                ///< Whether matrix A is transposed
  bool transB;                ///< Whether matrix B is transposed
  matmul_data_types dtypes;   ///< Data types for source, weight, bias, and output
};

/**
 * @brief Blocking parameters for stride-based BRGEMM.
 *
 * Pre-computed once from (M, N, K) and reused across all tiles.
 */
/**
 * @brief Blocking parameters for stride-based BRGEMM (Batch-Reduce GEMM).
 *
 * These parameters are computed from the input matrix dimensions (M, N, K) and
 * data layout, and determine how the work is partitioned and tiled for optimal performance.
 * These parameters are reused for each tile in the matmul execution.
 */
struct brgemm_blocking_params_t {
  int num_k_blocks;          ///< Number of full K blocks in K dimension (K / k_block_size)
  int k_block_size;          ///< Size of each K block
  int k_block_rem;           ///< Remaining elements in K dimension (K % k_block_size)
  int num_n_blocks;          ///< Number of full N blocks in N dimension (N / n_block_size)
  int n_block_size;          ///< Size of each N block
  int n_block_rem;           ///< Remaining elements in N dimension (N % n_block_size)
  int num_m_blocks;          ///< Number of full M blocks in M dimension (M / m_block_size)
  int m_block_size;          ///< Size of each M block
  int m_block_rem;           ///< Remaining elements in M dimension (M % m_block_size)
  int k_blocks_per_reduce;   ///< Number of K-blocks processed in one batch-reduce call
  int k_blocks_reduce_rem;   ///< Remainder K-blocks in the last reduction (num_k_blocks % k_blocks_per_reduce)
  bool weight_reuse;         ///< True if weights are reused across multiple K-loop iterations (enables blocked loop schemes)
  bool use_blocked_weight;   ///< True if weights are pre-converted into blocked layout ([num_n_blocks, num_k_blocks, k_block_size, n_block_size])
  unsigned long long
  stride_a;   ///< Byte stride between consecutive A panels for tiled access
  unsigned long long
  stride_b;   ///< Byte stride between consecutive B panels for tiled access
  const char
  *loop_scheme;       ///< Loop order scheme string ("aCb", "aCB", etc.), used by parallel loopers
};

/**
 * @brief Blocking parameters for blocked-weight BRGEMM
 *
 * This is a simplified version of brgemm_blocking_params_t, used when weights have been
 * physically blocked in memory. There is no tail in N/K since blocked weights
 * always fill complete blocks; only M may have a remainder tile.
 */
struct blocked_brgemm_params_t {
  int num_k_blocks;          ///< Number of K blocks in blocked weights
  int k_block_size;          ///< Size of each K block
  int num_n_blocks;          ///< Number of N blocks in blocked weights
  int n_block_size;          ///< Size of each N block
  int m_block_size;          ///< Size of each M block (for batch dimension)
  int m_block_rem;           ///< Remaining elements in M dimension (M % m_block_size)
  int num_m_blocks;          ///< Number of full M blocks (M / m_block_size)
  int k_blocks_per_reduce;   ///< K-blocks processed per batch-reduce (same as stride-based variant)
  int k_blocks_reduce_rem;   ///< Remaining K blocks for last reduction call
  bool weight_reuse;         ///< True if weights are reused across multiple K-loop iterations (applies to blocked weights too)
  const char *loop_scheme;   ///< Loop order scheme string
};

/**
 * @brief Compute BRGEMM blocking parameters from matrix dimensions.
 *
 * Block sizes default to 64 for N, K and 32 for M.
 */
brgemm_blocking_params_t compute_brgemm_blocking(
  const matmul_partition_config_t &config,
  bool use_blocked_weight = false
);

/**
 * @brief Cache key for BRGEMM dispatch cache.
 *
 * Two configs with identical key produce identical blocking params and
 * identical JIT kernels, so they can share the same pre-dispatched set.
 */
struct brgemm_cache_key_t {
  int M, N, K, lda, ldb, ldc;
  bool transA, transB;
  int src_dt, dst_dt;
  bool blocked;

  brgemm_cache_key_t(const matmul_partition_config_t &c, bool blocked = false)
    : M(c.M), N(c.N), K(c.K), lda(c.lda), ldb(c.ldb), ldc(c.ldc),
      transA(c.transA), transB(c.transB),
      src_dt(static_cast<int>(c.dtypes.src)),
      dst_dt(static_cast<int>(c.dtypes.dst)),
      blocked(blocked) {}

  bool operator==(const brgemm_cache_key_t &o) const {
    return M == o.M && N == o.N && K == o.K &&
           lda == o.lda && ldb == o.ldb && ldc == o.ldc &&
           transA == o.transA && transB == o.transB &&
           src_dt == o.src_dt && dst_dt == o.dst_dt &&
           blocked == o.blocked;
  }

  struct hash {
    std::size_t operator()(const brgemm_cache_key_t &k) const {
      std::size_t h = 0;
      auto mix = [&](std::size_t v) {
        h ^= v + 0x9e3779b9 + (h << 6) + (h >> 2);
      };
      mix(std::hash<int>()(k.M));
      mix(std::hash<int>()(k.N));
      mix(std::hash<int>()(k.K));
      mix(std::hash<int>()(k.lda));
      mix(std::hash<int>()(k.ldb));
      mix(std::hash<int>()(k.ldc));
      mix(std::hash<bool>()(k.transA));
      mix(std::hash<bool>()(k.transB));
      mix(std::hash<int>()(k.src_dt));
      mix(std::hash<int>()(k.dst_dt));
      mix(std::hash<bool>()(k.blocked));
      return h;
    }
  };
};

/**
 * @brief Callback function type for standard tiled kernel execution
 *
 * This callback is invoked for each tile when using standard tiling
 * without K-blocking (e.g., libxsmm or other non-BRGEMM kernels).
 *
 * @param m_start Starting row index in the output matrix
 * @param m_len Number of rows in this tile
 * @param n_start Starting column index in the output matrix
 * @param n_len Number of columns in this tile
 * @param A_tile Pointer to A matrix tile (m_len × K)
 * @param B_tile Pointer to B matrix tile (K × n_len)
 * @param C_tile Pointer to output tile (m_len × n_len)
 * @param tile_bias Pointer to bias vector for this tile (n_len elements)
 */
using tile_kernel_invoker_t = std::function<void(
                                int m_start, int m_len,
                                int n_start, int n_len,
                                const void *A_tile,
                                const void *B_tile,
                                void *C_tile,
                                const void *tile_bias
                              )>;

/**
 * @brief Execute matrix multiplication using standard tiling
 *
 * Divides the computation into M×N tiles without K-blocking.
 * Uses OpenMP parallelization across tiles. This path is used for
 * standard kernels (libxsmm etc.).
 *
 * @param src Source matrix A (M × K)
 * @param weight Weight matrix B (K × N)
 * @param dst Destination matrix C (M × N)
 * @param bias Optional bias vector, can be nullptr
 * @param config Partitioning configuration
 * @param tile_callback User-provided kernel invoker for standard tiles
 */
void execute_partitioned_matmul_standard(
  const void *src,
  const void *weight,
  void *dst,
  const void *bias,
  const matmul_partition_config_t &config,
  const tile_kernel_invoker_t &tile_callback
);

/**
 * @brief Calculate optimal tile sizes for M and N dimensions
 *
 * Uses heuristics based on matrix dimensions, K size, and thread count
 * to determine optimal M_BLOCK and N_BLOCK sizes. Can be overridden
 * using ZENDNN_MATMUL_M_TILE and ZENDNN_MATMUL_N_TILE environment variables.
 *
 * @param config Partitioning configuration containing M, N, K, num_threads
 * @return Pair of (M_BLOCK, N_BLOCK) tile sizes
 *
 * @note Default heuristics are tuned for BF16 performance
 */
std::pair<int, int> calculate_optimal_tile_sizes(
  const matmul_partition_config_t &config
);

/**
 * @brief Determine if KC-blocking (BRGEMM path) should be used
 * @param config Partitioning configuration
 * @param params Matmul parameters (for future extensions)
 * @return true if BRGEMM path should be used, false otherwise
 */
bool should_use_kc_blocking(
  const matmul_partition_config_t &config,
  const matmul_params &params
);

/**
 * @brief Get tile sizes from configuration with fallback to defaults
 *
 * Reads tile size values from matmul_config_t (which reads from environment
 * variables ZENDNN_MM_TILE_M and ZENDNN_MM_TILE_N). Returns the default
 * values if the configuration values are not set or invalid (≤ 0).
 *
 * @param default_m Fallback value for M tile size if not configured
 * @param default_n Fallback value for N tile size if not configured
 * @return Pair of (tile_m, tile_n) sizes from config or defaults
 *
 * @note This replaces get_tile_size_from_env() to use centralized config
 *
 * @example
 * auto [m_tile, n_tile] = get_tile_sizes_from_config(128, 64);
 */
std::pair<int, int> get_tile_sizes_from_config(int default_m, int default_n);

/**
 * @brief Select optimal tile sizes for BF16 matrix multiplication
 *
 * Internal heuristic function that determines M and N tile sizes based on
 * matrix dimensions and available thread count. Used by calculate_optimal_tile_sizes().
 *
 * @param M Number of rows in matrix A
 * @param N Number of columns in matrix B
 * @param K Inner dimension (columns of A, rows of B)
 * @param num_threads Number of threads available for parallel execution
 * @return Tuple of (M_tile, N_tile) sizes
 *
 * @note Heuristics:
 *       - Small matrices (M≤2048, N≤128): 32×32 tiles
 *       - Medium matrices (M≤4096, 768<N≤1024): 64×64 tiles
 *       - Large matrices: 128×64 tiles (default)
 *
 * @see calculate_optimal_tile_sizes() for the public interface
 */
std::tuple<int, int> select_tile(int M, int N, int K, int num_threads);

/**
 * @brief Compute byte offset for post-op buffer access
 *
 * Calculates the offset in bytes for accessing post-op buffers
 * (e.g., binary add/mul operands) at a specific tile location.
 *
 * @param row_start Starting row index in the post-op buffer
 * @param col_start Starting column index in the post-op buffer
 * @param leading_dim Leading dimension of the post-op buffer
 * @param dtype Data type of the post-op buffer elements
 * @return Byte offset from buffer start to (row_start, col_start)
 */
size_t compute_postop_offset(
  int row_start,
  int col_start,
  int leading_dim,
  data_type_t dtype
);

/**
 * @brief Apply byte offset to a buffer pointer
 *
 * @param buffer Original buffer pointer
 * @param offset Offset in bytes to apply
 * @return New pointer offset by the specified number of bytes, or nullptr if buffer is nullptr
 */
void *apply_offset(void *buffer, size_t offset);

/**
 * @brief Compute bias buffer offset for a specific tile column
 *
 * Calculates the pointer to the bias elements corresponding to
 * a tile starting at column col_start.
 *
 * @param bias Original bias buffer pointer
 * @param col_start Starting column index of the tile
 * @param bias_dtype Data type of the bias elements
 * @return Pointer to bias elements for this tile, or nullptr if bias is nullptr
 */
const void *compute_bias_offset(
  const void *bias,
  int col_start,
  data_type_t bias_dtype
);

/**
 * @brief Execute matrix multiplication using stride-based BRGEMM.
 *
 * Loop order: K(outer, sequential) -> N(parallel) -> M(inner).
 * At c==0 the output tile is initialized with bias (or zeros).
 * All BRGEMM calls use beta=1.0 to accumulate onto the tile.
 * PostOps are applied after the last K iteration.
 *
 * @param trans_input  Transpose flag for input matrix  ('N' or 'T')
 * @param trans_weight Transpose flag for weight matrix ('N' or 'T')
 * @param src    Source matrix A (M × K)
 * @param weight Weight matrix B (K × N)
 * @param dst    Destination matrix C (M × N)
 * @param bias   Optional bias vector, can be nullptr
 * @param config Partitioning configuration
 * @param params Matmul parameters (post-ops, data types)
 * @param beta   Scaling factor for C accumulation
 */
void execute_brgemm_tiled(
  const char trans_input,
  const char trans_weight,
  const void *src,
  const void *weight,
  void *dst,
  const void *bias,
  const matmul_partition_config_t &config,
  matmul_params &params,
  float beta
);

/**
 * @brief Execute matmul using blocked-weight BRGEMM.
 *
 * Weight must be pre-blocked as tiles indexed by
 * No K/N tail handling — only full blocks. No transpose support.
 * Only M (batch) remainder is handled with a separate kernel.
 *
 * @param src            Input matrix A (M x K), row-major
 * @param blocked_weight Pre-blocked weight
 * @param dst            Output matrix C (M x N), row-major
 * @param bias           Bias vector (N elements), or nullptr
 * @param bp             Blocked BRGEMM parameters
 * @param config         Partitioning configuration
 * @param params         Matmul params (post-ops, dtypes)
 * @param beta           Scaling factor for C
 */
void execute_brgemm_tiled_blocked(
  const void *src,
  const void *blocked_weight,
  void *dst,
  const void *bias,
  const blocked_brgemm_params_t &bp,
  const matmul_partition_config_t &config,
  matmul_params &params,
  float beta
);

/**
 * @brief Main entry point for partitioned matrix multiplication
 *
 * Orchestrates the entire partitioned matrix multiplication workflow:
 * 1. Selects the optimal kernel algorithm based on matrix properties
 * 2. Determines whether to use KC-blocking (BRGEMM) or standard tiling
 * 3. Creates appropriate callback functions for kernel execution
 * 4. Dispatches to the appropriate execution path
 *
 * @param layout Matrix layout ('R' for row-major, 'C' for column-major)
 * @param trans_input Transpose flag for input matrix ('N' or 'T')
 * @param trans_weight Transpose flag for weight matrix ('N' or 'T')
 * @param src Source matrix A (M × K)
 * @param weight Weight matrix B (K × N)
 * @param dst Destination matrix C (M × N)
 * @param bias Optional bias vector, can be nullptr
 * @param config Partitioning configuration (kernel may be modified)
 * @param params Matmul parameters including post-ops and data types
 * @param batch_params Batch parameters for batch matmul operations
 * @param is_weights_const Whether weights are constant (enables optimizations)
 * @param alpha Scaling factor for A*B product
 * @param beta Scaling factor for C (0.0 = overwrite, 1.0 = accumulate)
 */
void execute_partitioned_matmul(
  const char layout,
  const char trans_input,
  const char trans_weight,
  const void *src,
  const void *weight,
  void *dst,
  const void *bias,
  matmul_partition_config_t config,
  matmul_params &params,
  matmul_batch_params_t &batch_params,
  bool is_weights_const,
  float alpha,
  float beta
);

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_PARTITIONER_HPP
