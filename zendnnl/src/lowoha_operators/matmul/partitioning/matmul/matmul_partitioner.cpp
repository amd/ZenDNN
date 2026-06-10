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

#include "matmul_partitioner.hpp"

#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <omp.h>

#include "lowoha_operators/matmul/lowoha_matmul.hpp"
#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"
#include "lowoha_operators/matmul/backends/libxsmm/libxsmm_utils.hpp"
#include "lowoha_operators/matmul/backends/onednn/onednn_kernel.hpp"
#include "lowoha_operators/matmul/backends/libxsmm/libxsmm_kernel.hpp"

#if ZENDNNL_DEPENDS_PARLOOPER
  #include "threaded_loops.h"
#endif
namespace zendnnl {
namespace lowoha {
namespace matmul {

constexpr int BRGEMM_M_BLOCK = 32;
constexpr int BRGEMM_N_BLOCK = 64;
constexpr int BRGEMM_K_BLOCK = 64;
constexpr int BRGEMM_K_BLOCKS_PER_REDUCE = 64;
constexpr int WEIGHT_REUSE_THRESHOLD = 4;
static constexpr const char *LOOP_SCHEME_REUSE     = "aCB";
static constexpr const char *LOOP_SCHEME_STREAMING = "aCb";

brgemm_blocking_params_t compute_brgemm_blocking(
  const matmul_partition_config_t &config
) {
  brgemm_blocking_params_t bp{};

  bp.m_block_size = std::min(BRGEMM_M_BLOCK, config.M);
  bp.n_block_size = std::min(BRGEMM_N_BLOCK, config.N);
  bp.k_block_size = std::min(BRGEMM_K_BLOCK, config.K);

  bp.m_block_rem = config.M % bp.m_block_size;
  bp.n_block_rem = config.N % bp.n_block_size;
  bp.k_block_rem = config.K % bp.k_block_size;

  bp.num_m_blocks = config.M / bp.m_block_size;
  bp.num_n_blocks = config.N / bp.n_block_size;
  bp.num_k_blocks = config.K / bp.k_block_size;

  bp.weight_reuse = bp.num_m_blocks > WEIGHT_REUSE_THRESHOLD;
  if (bp.weight_reuse) {
    bp.k_blocks_per_reduce = std::min(BRGEMM_K_BLOCKS_PER_REDUCE, bp.num_k_blocks);
  }
  else {
    bp.k_blocks_per_reduce = bp.num_k_blocks;
  }

  bp.k_blocks_reduce_rem = (bp.k_blocks_per_reduce > 0) ?
                           (bp.num_k_blocks % bp.k_blocks_per_reduce) : 0;
  bp.loop_scheme = bp.weight_reuse ? LOOP_SCHEME_REUSE
                   : LOOP_SCHEME_STREAMING;

  if (config.transA) {
    bp.stride_a = static_cast<unsigned long long>(bp.k_block_size) *
                  config.lda * config.src_type_size;
  }
  else {
    bp.stride_a = static_cast<unsigned long long>(bp.k_block_size) *
                  config.src_type_size;
  }

  if (config.transB) {
    bp.stride_b = static_cast<unsigned long long>(bp.k_block_size) *
                  config.src_type_size;
  }
  else {
    bp.stride_b = static_cast<unsigned long long>(bp.k_block_size) *
                  config.ldb * config.src_type_size;
  }

  return bp;
}

size_t compute_postop_offset(
  int row_start,
  int col_start,
  int leading_dim,
  data_type_t dtype
) {
  size_t element_size = size_of(dtype);
  return (static_cast<size_t>(row_start) * leading_dim + col_start) *
         element_size;
}

void *apply_offset(void *buffer, size_t offset) {
  if (buffer == nullptr) {
    return nullptr;
  }
  return static_cast<uint8_t *>(buffer) + offset;
}

const void *compute_bias_offset(
  const void *bias,
  int col_start,
  data_type_t bias_dtype
) {
  if (bias == nullptr) {
    return nullptr;
  }
  size_t bias_element_size = size_of(bias_dtype);
  return static_cast<const uint8_t *>(bias) +
         (static_cast<size_t>(col_start) * bias_element_size);
}

static bool is_strided_gemm_layout(const matmul_partition_config_t &config) {
  const int min_lda = config.transA ? config.M : config.K;
  const int min_ldb = config.transB ? config.K : config.N;
  return config.lda != min_lda || config.ldb != min_ldb ||
         config.ldc != config.N;
}

static matmul_algo_t select_partition_kernel(
  const char trans_input,
  const char trans_weight,
  float alpha, float beta,
  const matmul_partition_config_t &config,
  const matmul_params &params,
  bool is_weights_const
) {
  const matmul_algo_t kernel = config.kernel;
  if (kernel != matmul_algo_t::libxsmm &&
      kernel != matmul_algo_t::libxsmm_blocked) {
    return kernel;
  }

  if (is_strided_gemm_layout(config)) {
    apilog_info("LibXSMM partitioned kernel does not support strided layouts, falling back to DLP");
    return matmul_algo_t::aocl_dlp;
  }

  // LibXSMM matmul path (libxsmm / libxsmm_blocked) is currently supported
  // only for the BF16_BF16 case. Any other
  // dtype combination must fall back to the DLP kernel.
  // TODO: Add support for other dtype combinations.
  if (config.dtypes.src != data_type_t::bf16 ||
      config.dtypes.wei != data_type_t::bf16 ||
      config.dtypes.dst != data_type_t::bf16) {
    apilog_info("LibXSMM kernel is only supported for BF16_BF16 combination, falling back to DLP");
    return matmul_algo_t::aocl_dlp;
  }

  if (!can_use_libxsmm(trans_input, trans_weight,
                       config.M, config.N, config.K,
                       alpha, beta, params, false)) {
    apilog_info("LibXSMM kernel cannot be used for current configuration, falling back to DLP");
    return matmul_algo_t::aocl_dlp;
  }
  if (kernel == matmul_algo_t::libxsmm_blocked && !is_weights_const) {
    apilog_info("LibXSMM blocked kernel is not supported for non-constant weights, falling back to LibXSMM");
    return matmul_algo_t::libxsmm;
  }
  // If libxsmm_blocked, check if N and K are multiples of blocking sizes.
  if (kernel == matmul_algo_t::libxsmm_blocked) {
    if ((config.N % BRGEMM_N_BLOCK != 0) || (config.K % BRGEMM_K_BLOCK != 0)) {
      apilog_info("LibXSMM blocked kernel requires N and K to be multiples of nc and kc. Falling back to libxsmm.");
      return matmul_algo_t::libxsmm;
    }
  }

  return kernel;
}

#if ZENDNNL_DEPENDS_LIBXSMM
/**
 * @brief Apply all post-ops on a tile, with binary-op operand pointers relocated to the tile's (m_start, n_start) position.
 */
static void apply_tile_postops(
  int m_start, int n_start, int m_len, int n_len, int ldc,
  void *C_tile,
  const matmul_params &params,
  const matmul_data_types &dtypes) {

  for (const auto &po : params.postop_) {
    matmul_post_op tile_po = po;
    if ((tile_po.po_type == post_op_type_t::binary_add ||
         tile_po.po_type == post_op_type_t::binary_mul) &&
        tile_po.buff != nullptr) {
      size_t offset;
      if (tile_po.dims.size() == 2 && tile_po.dims[0] == 1) {
        // Row vector {1, N}: only column offset applies (broadcast over rows).
        offset = compute_postop_offset(0, n_start, tile_po.leading_dim,
                                       tile_po.dtype);
      }
      else {
        offset = compute_postop_offset(
                   m_start, n_start, tile_po.leading_dim, tile_po.dtype);
      }
      tile_po.buff = apply_offset(const_cast<void *>(tile_po.buff), offset);
    }

    if (dtypes.dst == data_type_t::f32) {
      libxsmm_postop<float>(m_len, n_len, ldc, C_tile, tile_po);
    }
    else {
      libxsmm_postop<libxsmm_bfloat16>(m_len, n_len, ldc, C_tile, tile_po);
    }
  }
}

struct BrgemmCachedContext {
  brgemm_blocking_params_t bp;
  PreDispatchedBrgemm pd;
};

/**
 * @brief Block and cache the weight matrix for BRGEMM using the existing LRU cache.
 *
 * On cache miss: allocates, blocks via libxsmm_weight_block(), stores in LRU.
 * On cache hit: returns the cached pointer directly.
 * Also fills blocking params for the caller.
 *
 * @return Pointer to blocked weight, or nullptr if blocking is not applicable.
 */
static const void *libxsmm_weight_prepack(
  const void *weight, int M, int K, int N, int ldb,
  bool transB, int k_block_size, int n_block_size,
  data_type_t src_dtype, bool is_weights_const,
  blocked_brgemm_params_t &bp) {

  apilog_info("libxsmm_weight_prepack: K=", K, " N=", N, " k_block_size=",
              k_block_size,
              " n_block_size=", n_block_size, " is_weights_const=", is_weights_const);

  if (!is_weights_const) {
    return nullptr;
  }

  if (K % k_block_size != 0 || N % n_block_size != 0) {
    return nullptr;
  }

  int num_k_blocks = K / k_block_size;
  int num_n_blocks = N / n_block_size;

  if (num_k_blocks <= 0 || num_n_blocks <= 0) {
    return nullptr;
  }

  bp.num_k_blocks  = num_k_blocks;
  bp.k_block_size  = k_block_size;
  bp.num_n_blocks  = num_n_blocks;
  bp.n_block_size  = n_block_size;
  bp.m_block_size  = std::min(BRGEMM_M_BLOCK, M);
  bp.m_block_rem   = M % bp.m_block_size;
  bp.num_m_blocks  = M / bp.m_block_size;
  bp.weight_reuse  = bp.num_m_blocks > WEIGHT_REUSE_THRESHOLD;
  if (bp.weight_reuse) {
    bp.k_blocks_per_reduce = std::min(BRGEMM_K_BLOCKS_PER_REDUCE, num_k_blocks);
  }
  else {
    bp.k_blocks_per_reduce = num_k_blocks;
  }
  bp.k_blocks_reduce_rem = (bp.k_blocks_per_reduce > 0) ?
                           (num_k_blocks % bp.k_blocks_per_reduce) : 0;
  bp.loop_scheme = bp.weight_reuse ? LOOP_SCHEME_REUSE
                   : LOOP_SCHEME_STREAMING;

  static lru_cache_t<Key_matmul, void *> brgemm_weight_cache;

  Key_matmul key(transB, static_cast<unsigned int>(K),
                 static_cast<unsigned int>(N),
                 static_cast<unsigned int>(ldb), weight,
                 static_cast<uint32_t>(matmul_algo_t::libxsmm_blocked));

  void *cached_blocked = nullptr;
  if (brgemm_weight_cache.try_get(key, cached_blocked)) {
    return cached_blocked;
  }

  size_t buf_sz = libxsmm_weight_block_size(K, N, src_dtype);
  void *blocked = std::aligned_alloc(64, buf_sz);
  if (!blocked) {
    return nullptr;
  }

  libxsmm_weight_block(weight, blocked, K, N, ldb, k_block_size, n_block_size,
                       src_dtype, transB);
  brgemm_weight_cache.add(key, blocked);
  return blocked;
}

static std::unordered_map<brgemm_cache_key_t, BrgemmCachedContext,
       brgemm_cache_key_t::hash> brgemm_ctx_cache;
static std::mutex brgemm_ctx_mutex;

static const BrgemmCachedContext &get_brgemm_context(
  char trans_input, char trans_weight,
  const matmul_partition_config_t &config) {

  brgemm_cache_key_t key(config);

  {
    std::lock_guard<std::mutex> lock(brgemm_ctx_mutex);
    auto it = brgemm_ctx_cache.find(key);
    if (it != brgemm_ctx_cache.end()) {
      return it->second;
    }
  }

  BrgemmCachedContext ctx;
  ctx.bp = compute_brgemm_blocking(config);
  ctx.pd = predispatch_brgemm_kernels(
             trans_input, trans_weight,
             ctx.bp.m_block_size, ctx.bp.m_block_rem,
             ctx.bp.n_block_size, ctx.bp.n_block_rem,
             ctx.bp.k_block_size, ctx.bp.k_block_rem,
             ctx.bp.num_k_blocks, ctx.bp.k_blocks_per_reduce,
             config.lda, config.ldb, config.ldc,
             ctx.bp.stride_a, ctx.bp.stride_b,
             config.dtypes);

  std::lock_guard<std::mutex> lock(brgemm_ctx_mutex);
  auto [it, inserted] = brgemm_ctx_cache.emplace(key, ctx);
  return it->second;
}

static const PreDispatchedBrgemm &get_brgemm_context_blocked(
  const blocked_brgemm_params_t &bp,
  const matmul_partition_config_t &config) {

  brgemm_cache_key_t key(config, /*blocked=*/true);

  {
    std::lock_guard<std::mutex> lock(brgemm_ctx_mutex);
    auto it = brgemm_ctx_cache.find(key);
    if (it != brgemm_ctx_cache.end()) {
      return it->second.pd;
    }
  }

  bool vnni = (config.dtypes.src == data_type_t::bf16);
  unsigned long long blk_stride_a = static_cast<unsigned long long>
                                    (bp.k_block_size) * config.src_type_size;
  unsigned long long blk_stride_b = static_cast<unsigned long long>
                                    (bp.n_block_size) * bp.k_block_size * config.src_type_size;

  BrgemmCachedContext ctx{};
  ctx.pd = predispatch_brgemm_kernels(
             'N', 'N',
             bp.m_block_size, bp.m_block_rem,
             bp.n_block_size, 0,
             bp.k_block_size, 0,
             bp.num_k_blocks, bp.k_blocks_per_reduce,
             config.lda, bp.n_block_size, config.ldc,
             blk_stride_a, blk_stride_b,
             config.dtypes, vnni);

  std::lock_guard<std::mutex> lock(brgemm_ctx_mutex);
  auto [it, inserted] = brgemm_ctx_cache.emplace(key, ctx);
  return it->second.pd;
}

void execute_brgemm_std(
  const char trans_input,
  const char trans_weight,
  const void *src,
  const void *weight,
  void *dst,
  const void *bias,
  const matmul_partition_config_t &config,
  matmul_params &params,
  float beta
) {
  const auto &cached = get_brgemm_context(trans_input, trans_weight, config);
  const brgemm_blocking_params_t &bp = cached.bp;
  const PreDispatchedBrgemm &pd = cached.pd;

  const uint8_t *src_ptr    = static_cast<const uint8_t *>(src);
  const uint8_t *weight_ptr = static_cast<const uint8_t *>(weight);
  uint8_t       *dst_ptr    = static_cast<uint8_t *>(dst);

  const int M = config.M;
  const int N = config.N;
  const bool has_postops = params.postop_.size() > 0;

  const int k_full = bp.num_k_blocks * bp.k_block_size;
  const int k_step = bp.k_blocks_per_reduce * bp.k_block_size;

  auto tile_body = [&](int k_off, int m_off,
  int n_off) __attribute__((always_inline)) {
    int is_m_rem = (m_off + bp.m_block_size > M) ? 1 : 0;
    int is_n_rem = (n_off + bp.n_block_size > N) ? 1 : 0;
    int m_len = is_m_rem ? bp.m_block_rem : bp.m_block_size;
    int n_len = is_n_rem ? bp.n_block_rem : bp.n_block_size;

    void *C_tile = get_output_block(
                     dst_ptr, m_off, n_off, config.ldc, config.out_type_size);

    if (k_off == 0) {
      const void *tile_bias = compute_bias_offset(
                                bias, n_off, config.dtypes.bias);
      init_output_tile(C_tile, tile_bias, m_len, n_len,
                       config.ldc, config.dtypes, beta);
    }

    bool is_last_k = (k_off + k_step >= k_full);
    int count = is_last_k ? (bp.k_blocks_reduce_rem > 0 ? bp.k_blocks_reduce_rem :
                             bp.k_blocks_per_reduce) : bp.k_blocks_per_reduce;

    const void *A_base = get_matrix_block(
                           src_ptr, m_off, k_off, config.lda,
                           config.transA, config.src_type_size);

    const void *B_base;

    B_base = get_matrix_block(
               weight_ptr, k_off, n_off, config.ldb,
               config.transB, config.src_type_size);
    if (count > 0) {
      unsigned long long ull_count = static_cast<unsigned long long>(count);
      run_brgemm(pd.main_ker[is_m_rem][is_n_rem],
                 A_base, B_base, C_tile, ull_count);
    }

    if (is_last_k && bp.k_block_rem > 0) {
      const void *A_tail = get_matrix_block(
                             src_ptr, m_off, k_full, config.lda,
                             config.transA, config.src_type_size);
      const void *B_tail = get_matrix_block(
                             weight_ptr, k_full, n_off, config.ldb,
                             config.transB, config.src_type_size);
      unsigned long long one = 1;
      run_brgemm(pd.tail_ker[is_m_rem][is_n_rem],
                 A_tail, B_tail, C_tile, one);
    }

    if (is_last_k && has_postops) {
      apply_tile_postops(m_off, n_off, m_len, n_len, config.ldc,
                         C_tile, params, config.dtypes);
    }
  };

#if ZENDNNL_DEPENDS_PARLOOPER
  auto gemm_loop = ThreadedLoop<3>({
    LoopSpecs{0L, (long)k_full, (long)k_step, false},
    LoopSpecs{0L, (long)M, (long)bp.m_block_size},
    LoopSpecs{0L, (long)N, (long)bp.n_block_size}
  },
  bp.loop_scheme);

  gemm_loop([&](int *ind) {
    int k_off = ind[0], m_off = ind[1], n_off = ind[2];
    tile_body(k_off, m_off, n_off);
  });
#else
  if (bp.weight_reuse) {
    for (int k_off = 0; k_off < k_full; k_off += k_step) {
      #pragma omp parallel for collapse(2) schedule(static) num_threads(config.num_threads)
      for (int n_off = 0; n_off < N; n_off += bp.n_block_size) {
        for (int m_off = 0; m_off < M; m_off += bp.m_block_size) {
          tile_body(k_off, m_off, n_off);
        }
      }
    }
  }
  else {
    for (int k_off = 0; k_off < k_full; k_off += k_step) {
      #pragma omp parallel for schedule(static) num_threads(config.num_threads)
      for (int n_off = 0; n_off < N; n_off += bp.n_block_size) {
        for (int m_off = 0; m_off < M; m_off += bp.m_block_size) {
          tile_body(k_off, m_off, n_off);
        }
      }
    }
  }
#endif
}

void execute_brgemm_prepacked(
  const void *src,
  const void *blocked_weight,
  void *dst,
  const void *bias,
  const blocked_brgemm_params_t &bp,
  const matmul_partition_config_t &config,
  matmul_params &params,
  float beta
) {
  apilog_info("execute_brgemm_prepacked: M=", config.M,
              " num_k_blocks=", bp.num_k_blocks, " num_n_blocks=", bp.num_n_blocks,
              " k_block_size=", bp.k_block_size, " n_block_size=", bp.n_block_size,
              " m_block_size=", bp.m_block_size,
              " k_blocks_per_reduce=", bp.k_blocks_per_reduce,
              " weight_reuse=", bp.weight_reuse);

  const PreDispatchedBrgemm &pd = get_brgemm_context_blocked(bp, config);

  const uint8_t *src_ptr = static_cast<const uint8_t *>(src);
  const uint8_t *blk_ptr = static_cast<const uint8_t *>(blocked_weight);
  uint8_t       *dst_ptr = static_cast<uint8_t *>(dst);

  const int M              = config.M;
  const int num_k_blocks   = bp.num_k_blocks;
  const int num_n_blocks   = bp.num_n_blocks;
  const int k_block_size   = bp.k_block_size;
  const int n_block_size   = bp.n_block_size;
  const bool has_postops   = params.postop_.size() > 0;

  const size_t src_elem   = config.src_type_size;
  const size_t out_elem   = config.out_type_size;
  const size_t tile_bytes = static_cast<size_t>(n_block_size) * k_block_size *
                            src_elem;

  auto tile_body = [&](int kblk, int m_off, int nblk)
  __attribute__((always_inline)) {
    int is_m_rem = (m_off + bp.m_block_size > M) ? 1 : 0;
    int m_len = is_m_rem ? bp.m_block_rem : bp.m_block_size;

    void *C_tile = dst_ptr +
                   (static_cast<size_t>(m_off) * config.ldc + nblk * n_block_size) * out_elem;

    if (kblk == 0) {
      const void *tile_bias = compute_bias_offset(
                                bias, nblk * n_block_size, config.dtypes.bias);
      init_output_tile(C_tile, tile_bias, m_len, n_block_size,
                       config.ldc, config.dtypes, beta);
    }

    int count = std::min(bp.k_blocks_per_reduce, num_k_blocks - kblk);

    const void *A_base = src_ptr +
                         (static_cast<size_t>(m_off) * config.lda + kblk * k_block_size) * src_elem;
    const void *B_base = blk_ptr +
                         (static_cast<size_t>(nblk) * num_k_blocks + kblk) * tile_bytes;

    if (count > 0) {
      unsigned long long ull_count = static_cast<unsigned long long>(count);
      run_brgemm(pd.main_ker[is_m_rem][0], A_base, B_base, C_tile, ull_count);
    }

    bool is_last_k = (kblk + bp.k_blocks_per_reduce >= num_k_blocks);
    if (is_last_k && has_postops) {
      apply_tile_postops(m_off, nblk * n_block_size, m_len, n_block_size, config.ldc,
                         C_tile, params, config.dtypes);
    }
  };

#if ZENDNNL_DEPENDS_PARLOOPER
  auto gemm_loop = ThreadedLoop<3>({
    LoopSpecs{0L, (long)num_k_blocks, (long)bp.k_blocks_per_reduce, false},
    LoopSpecs{0L, (long)M, (long)bp.m_block_size},
    LoopSpecs{0L, (long)num_n_blocks, 1L}
  },
  bp.loop_scheme);

  gemm_loop([&](int *ind) {
    int kblk = ind[0], m_off = ind[1], nblk = ind[2];
    tile_body(kblk, m_off, nblk);
  });
#else
  if (bp.weight_reuse) {
    for (int kblk = 0; kblk < num_k_blocks; kblk += bp.k_blocks_per_reduce) {
      #pragma omp parallel for collapse(2) schedule(static) num_threads(config.num_threads)
      for (int nblk = 0; nblk < num_n_blocks; ++nblk) {
        for (int m_off = 0; m_off < M; m_off += bp.m_block_size) {
          tile_body(kblk, m_off, nblk);
        }
      }
    }
  }
  else {
    for (int kblk = 0; kblk < num_k_blocks; kblk += bp.k_blocks_per_reduce) {
      #pragma omp parallel for schedule(static) num_threads(config.num_threads)
      for (int nblk = 0; nblk < num_n_blocks; ++nblk) {
        for (int m_off = 0; m_off < M; m_off += bp.m_block_size) {
          tile_body(kblk, m_off, nblk);
        }
      }
    }
  }
#endif
}
#endif

void execute_partitioned_matmul(
  const char layout,
  const char trans_input,
  const char trans_weight,
  const void *src,
  const void *weight,
  void *dst,
  const void *bias,
  matmul_partition_config_t &config,
  matmul_params &params,
  matmul_batch_params_t &batch_params,
  bool is_weights_const,
  float alpha,
  float beta
) {
  config.kernel = select_partition_kernel(
                    trans_input, trans_weight,
                    alpha, beta, config, params, is_weights_const);
#if ZENDNNL_DEPENDS_LIBXSMM
  if (config.kernel == matmul_algo_t::libxsmm_blocked) {
    blocked_brgemm_params_t bbp{};
    const void *blocked_wt = libxsmm_weight_prepack(
                               weight, config.M, config.K, config.N, config.ldb,
                               config.transB, BRGEMM_K_BLOCK, BRGEMM_N_BLOCK,
                               config.dtypes.src, is_weights_const, bbp);
    if (blocked_wt) {
      apilog_info("Executing LIBXSMM brgemm prepacked");
      execute_brgemm_prepacked(
        src, blocked_wt, dst, bias, bbp, config, params, beta);
      return;
    }
    config.kernel = matmul_algo_t::libxsmm;
  }
  if (config.kernel == matmul_algo_t::libxsmm) {
    apilog_info("Executing LIBXSMM brgemm strided");
    execute_brgemm_std(
      trans_input, trans_weight,
      src, weight, dst, bias,
      config, params, beta);
    return;
  }
#endif
#if ZENDNNL_DEPENDS_ONEDNN
  if (config.kernel == matmul_algo_t::onednn ||
      config.kernel == matmul_algo_t::onednn_blocked) {
    apilog_info("Given combination is not supported for matmul parallel primitive, executing matmul LOWOHA kernel with OneDNN fallback, algo: ",
                static_cast<int>(config.kernel));
    matmul_onednn_wrapper(trans_input, trans_weight,
                          config.M, config.N, config.K, alpha,
                          src, config.lda,
                          weight, config.ldb,
                          beta, dst, config.ldc,
                          params, batch_params, bias, config.kernel,
                          is_weights_const);
    return;
  }
#endif
  config.kernel = matmul_algo_t::aocl_dlp;
  apilog_info("Given combination is not supported for matmul parallel primitive, executing matmul LOWOHA kernel with DLP fallback, algo: ",
              static_cast<int>(config.kernel));
  matmul_kernel_wrapper(layout, trans_input, trans_weight,
                        config.M, config.N, config.K, alpha,
                        src, config.lda, weight, config.ldb,
                        beta, dst, config.ldc,
                        params.dtypes, config.kernel,
                        params.mem_format_a, params.mem_format_b,
                        params, batch_params,
                        bias, is_weights_const);
}
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
