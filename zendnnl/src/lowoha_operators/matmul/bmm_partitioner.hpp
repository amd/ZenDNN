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

#ifndef BMM_PARTITIONER_HPP
#define BMM_PARTITIONER_HPP

#include <functional>
#include "lowoha_operators/matmul/lowoha_common.hpp"

namespace zendnnl {
namespace lowoha {

/**
 * @struct bmm_partition_config_t
 * @brief Configuration for parallel partitioning of BMM operations
 *
 * This structure encapsulates all parameters needed for parallel partitioning
 * of batch matrix multiplication operations across multiple threads.
 */
struct bmm_partition_config_t {
  int M;                        // Number of rows in output matrix
  int N;                        // Number of columns in output matrix
  int K;                        // Inner dimension
  int batch_count;              // Total number of batches
  int num_threads;              // Number of available threads
  matmul_algo_t kernel;         // Selected kernel algorithm
  size_t src_batch_stride_bytes;    // Batch stride for source in bytes
  size_t weight_batch_stride_bytes; // Batch stride for weights in bytes
  size_t dst_batch_stride_bytes;    // Batch stride for destination in bytes

  /**
   * @brief Default constructor for bmm_partition_config_t
   */
  bmm_partition_config_t() : M(0), N(0), K(0), batch_count(1), num_threads(1),
    kernel(matmul_algo_t::none), src_batch_stride_bytes(0), weight_batch_stride_bytes(0),
    dst_batch_stride_bytes(0) {}
};

/**
 * @brief Callback type for processing a single tile/block
 *
 * @param batch_idx     Batch index
 * @param m_start       Starting row index within the batch
 * @param m_len         Number of rows to process
 * @param src_ptr       Pointer to source data for this batch
 * @param weight_ptr    Pointer to weight data for this batch
 * @param dst_ptr       Pointer to destination data for this batch
 */
using tile_callback_t = std::function<void(
    int batch_idx,
    int m_start,
    int m_len,
    const uint8_t *src_ptr,
    const uint8_t *weight_ptr,
    uint8_t *dst_ptr
)>;

/**
 * @brief Calculate optimal M_block size for parallel partitioning
 *
 * This function determines the optimal block size for M dimension partitioning
 * based on batch count, matrix dimensions, and kernel type. The heuristics
 * are tuned for different scenarios to maximize performance.
 *
 * @param config Partition configuration containing matrix dimensions and parameters
 * @return Optimal M_block size for partitioning
 */
int calculate_optimal_m_block(const bmm_partition_config_t &config);

/**
 * @brief Determine if zendnnl_parallel_for should be used over OpenMP
 *
 * Based on MFLOPs threshold, decides which parallelization strategy to use.
 * zendnnl_parallel_for is preferred for larger workloads (MFLOPs > 6.0),
 * while OpenMP parallel for is used for smaller tasks to avoid overhead.
 *
 * @param M Number of rows
 * @param N Number of columns
 * @param K Inner dimension
 * @return true if zendnnl_parallel_for should be used, false for OpenMP
 */
bool should_use_zendnnl_parallel(int M, int N, int K);

/**
 * @brief Execute parallel partitioned BMM using zendnnl_parallel_for
 *
 * This function partitions work across batches and M-dimension blocks,
 * using the custom zendnnl_parallel_for for better load balancing on
 * larger workloads. Work items are linearized and distributed across threads.
 *
 * @param src           Source data pointer
 * @param weight        Weight data pointer
 * @param dst           Destination data pointer
 * @param config        Partition configuration
 * @param batch_params  Batch parameters (Batch_A, Batch_B)
 * @param M_block       Block size for M dimension
 * @param callback      Callback function to process each tile
 */
void execute_parallel_zendnnl(
    const void *src,
    const void *weight,
    void *dst,
    const bmm_partition_config_t &config,
    const matmul::matmul_batch_params_t &batch_params,
    int M_block,
    const tile_callback_t &callback);

/**
 * @brief Execute parallel partitioned BMM using OpenMP
 *
 * This function partitions work across batches and M-dimension blocks,
 * using OpenMP parallel for with collapse(2) for smaller workloads.
 * This approach has less overhead for smaller tasks.
 *
 * @param src           Source data pointer
 * @param weight        Weight data pointer
 * @param dst           Destination data pointer
 * @param config        Partition configuration
 * @param batch_params  Batch parameters (Batch_A, Batch_B)
 * @param M_block       Block size for M dimension
 * @param callback      Callback function to process each tile
 */
void execute_parallel_omp(
    const void *src,
    const void *weight,
    void *dst,
    const bmm_partition_config_t &config,
    const matmul::matmul_batch_params_t &batch_params,
    int M_block,
    const tile_callback_t &callback);

/**
 * @brief Execute partitioned BMM with automatic strategy selection
 *
 * This is the main entry point that automatically selects between
 * zendnnl_parallel_for and OpenMP based on workload characteristics.
 * It calculates the optimal M_block size and dispatches to the
 * appropriate parallelization strategy.
 *
 * @param src           Source data pointer
 * @param weight        Weight data pointer
 * @param dst           Destination data pointer
 * @param config        Partition configuration
 * @param batch_params  Batch parameters
 * @param callback      Callback function to process each tile
 */
void execute_partitioned_bmm(
    const void *src,
    const void *weight,
    void *dst,
    const bmm_partition_config_t &config,
    const matmul::matmul_batch_params_t &batch_params,
    const tile_callback_t &callback);

} // namespace lowoha
} // namespace zendnnl

#endif // LOWOHA_PARTITIONER_HPP

