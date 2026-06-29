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

#ifndef _LOWOHA_COMMON_HPP
#define _LOWOHA_COMMON_HPP

#include "memory/memory_utils.hpp"
#include "operators/common/post_op.hpp"
#include "operators/matmul/matmul_config.hpp"
#include "lowoha_operators/matmul/lru_cache/zendnnl_key.hpp"
#include "lowoha_operators/matmul/lru_cache/lru_cache.hpp"
#include <algorithm>
#include <cstdint>

namespace zendnnl {
namespace lowoha {
namespace matmul {

using namespace zendnnl::memory;
using namespace zendnnl::ops;

/**
 * @brief Structure to hold data types for matrix multiplication operands
 */
struct matmul_data_types {
  data_type_t src = data_type_t::none;     ///< Source matrix data type
  data_type_t wei = data_type_t::none;     ///< Weight matrix data type
  data_type_t dst = data_type_t::none;     ///< Destination matrix data type
  data_type_t bias = data_type_t::none;    ///< Bias vector data type
  data_type_t compute = data_type_t::none; ///< Computation data type
};

/**
 * @brief Structure for post-operation parameters
 */
struct matmul_post_op {
  zendnnl::ops::post_op_type_t po_type;    ///< Type of post-operation
  void *buff;                              ///< Buffer for binary operations
  data_type_t dtype;                       ///< Data type of the buffer
  std::vector<int64_t> dims;               ///< Dimensions of the buffer
  float alpha;                             ///< Alpha parameter for operations
  float beta;                              ///< Beta parameter for operations
  int leading_dim;                         ///< Leading dimension for the buffer

  /**
   * @brief Default constructor for matmul_post_op
   */
  matmul_post_op() : po_type(zendnnl::ops::post_op_type_t::none), buff(nullptr),
    dtype(data_type_t::none), dims(), alpha(0.0f), beta(0.0f), leading_dim(-1) {}
};

/**
 * @brief Structure for quantization parameters (scales and zero-points)
 */
struct matmul_quantization_params_t {
  /**
   * @brief Individual quantization parameter (scale or zero-point)
   *
   * Dimensions determine quantization granularity for weight matrix [K, N]:
   *   - Per-tensor:  dims = {} or {1}     → single scale for all weights
   *   - Per-channel: dims = {1, N}        → one scale per output channel
   *   - Per-group:   dims = {G, N}        → G groups along K, where G = K/group_size
   */
  struct matmul_quant_t {
    const void *buff;              ///< Pointer to quantization data buffer
    data_type_t dt;                ///< Data type of the buffer
    std::vector<int64_t> dims;     ///< Dimensions of the quantization tensor

    /**
     * @brief Default constructor for matmul_quant_t
     */
    matmul_quant_t() : buff(nullptr), dt(data_type_t::none), dims() {}
  };

  matmul_quant_t src_scale;  ///< Source tensor scale
  matmul_quant_t wei_scale;  ///< Weight tensor scale
  matmul_quant_t dst_scale;  ///< Destination tensor scale
  matmul_quant_t src_zp;     ///< Source tensor zero-point
  matmul_quant_t wei_zp;     ///< Weight tensor zero-point
  matmul_quant_t dst_zp;     ///< Destination tensor zero-point

  /**
   * @brief Default constructor for quantization parameters
   */
  matmul_quantization_params_t() : src_scale(), wei_scale(), dst_scale(),
    src_zp(), wei_zp(), dst_zp() {}
};

/**
 * @struct batch_params_t
 * @brief A structure to encapsulate batch dimensions and batch strides.
 *
 * This structure contains batch sizes (Batch_A and Batch_B) and batch strides
 * for source, weight, and destination tensors. The batch strides specify the
 * byte offset between consecutive batches in memory.
 */
struct matmul_batch_params_t {
  int Batch_A = 1;              /**< Batch size for source tensor. */
  int Batch_B = 1;              /**< Batch size for weight tensor. */
  size_t batch_stride_src =
    -1;  /**< Byte stride between batches for source tensor (-1 means calculate from dimensions). */
  size_t batch_stride_wei =
    -1;  /**< Byte stride between batches for weight tensor (-1 means calculate from dimensions). */
  size_t batch_stride_dst =
    -1;  /**< Byte stride between batches for destination tensor (-1 means calculate from dimensions). */

  /**
   * @brief Default constructor for `matmul_batch_params_t`.
   *
   * Initializes Batch_A and Batch_B to 1, and all strides to -1.
   */
  matmul_batch_params_t() : Batch_A(1), Batch_B(1), batch_stride_src(-1),
    batch_stride_wei(-1), batch_stride_dst(-1) {}
};

/**
 * @brief Structure describing the packing format of weight matrix B.
 *
 * pack_format_b = 0 (default): weights are in standard unpacked layout.
 * pack_format_b = 1: weights are in GGML Q8_0 packed format (int8 weights
 *                     with interleaved fp16 scales) and must be unpacked.
 */
struct pack_format {
  int pack_format_b;  ///< 0 = unpacked (default), 1 = GGML Q8_0 packed weights

  pack_format() : pack_format_b(0) {}
};

/**
 * @brief Main parameter structure for LOWOHA matrix multiplication
 */
struct matmul_params {
  matmul_data_types dtypes;                    ///< Data types for operands
  std::vector<matmul_post_op> postop_;         ///< Post-operation chain
  matmul_quantization_params_t quant_params;   ///< Quantization parameters
  char mem_format_a;                           ///< Memory format for matrix A
  ///< Memory format for matrix B.
  ///< - 'n' (default): standard row-major weights; matmul backend
  ///<                   runs its own reorder/blocking step.
  ///< - 'r' : weights are already in the AOCL DLP blocked layout
  ///<         (produced by a prior @c reorder_direct() prepack call).
  ///<         Matmul backend skips its internal weight-reorder /
  ///<         cache-blocking step and uses the buffer as-is. Requires
  ///<         @c lowoha_algo == matmul_algo_t::aocl_dlp_blocked and
  ///<         the prepack to have used matching
  ///<         K / N / ldb / dtypes / transposed / sym_group_size --
  ///<         a mismatch produces silently wrong results.
  char mem_format_b;
  matmul_algo_t lowoha_algo;                   ///< Selected algorithm
  //num_threads is int32_t to match the type used by OpenMP APIs
  int32_t num_threads;                        ///< Number of threads
  std::string plugin_op;                       ///< Plugin op name
  bool dynamic_quant;                          ///< Enable dynamic quantization of source
  pack_format packing;                         ///< Weight packing format for matrix B

  // ── group_matmul prepack-extras contract (read from params[0] only) ──
  //
  // Optional hint used by `group_matmul_direct` to enable ahead-of-time
  // weight prepack for an MoE-style "all-weights, some-firing" call.
  // When non-zero, the caller passes weight buffers for `total_matmul`
  // experts but only the first `active_matmul` are computed in this
  // call.  The library:
  //
  //   1. Validates the contract.  Three rules apply in opt-in mode
  //      (`active_matmul > 0`):
  //        a) `active_matmul <= total_matmul` (when `total_matmul > 0`).
  //        b) Every per-expert vector accepted by the dispatcher
  //           (`weight`, `K`, `N`, `ldb`, `transB`, `is_weights_const`,
  //           plus the input-side `alpha`, `bias`, `beta`, `ldc`,
  //           `params`, etc.) must be `>= active_matmul`.
  //        c) When `total_matmul > active_matmul` (prepack-extras
  //           tail present), the SIX weight-side metadata vectors
  //           that the prepack module iterates over
  //           (`weight`, `K`, `N`, `ldb`, `transB`,
  //           `is_weights_const`) must additionally be
  //           `>= total_matmul`.  This tighter requirement prevents
  //           silent prepack truncation — without it, an undersized
  //           `weight.size()` (etc.) would let the warmer's
  //           `bound = std::min({total_matmul, weight.size(),
  //           K.size(), ...})` clamp the warm to a shorter length,
  //           leaving tail experts un-warmed.  All other vectors
  //           still need only `>= active_matmul` (compact or padded
  //           — both are legitimate).
  //   2. Computes only the first `active_matmul` GEMMs.
  //   3. Pre-warms the inner-kernel weight cache for ALL `total_matmul`
  //      experts ahead of time, so any expert firing on a future call
  //      hits a warm cache and avoids the on-the-fly reorder spike.
  //
  // The caller fills `params[0].active_matmul` and `params[0].total_matmul`
  // exactly — the dispatcher reads them from the first entry only and
  // ignores the same fields on `params[1..N]`.  Leave both at the
  // default `0` for the legacy "every supplied weight fires" contract:
  // the library then derives `num_ops = M.size()` and requires every
  // weight-side vector to be exactly `num_ops` long.
  //
  // The eager prepack is gated by the `ZENDNNL_GRP_MATMUL_PREPACK`
  // environment variable (default ON).  See
  // `docs/operator/low_overhead_operator/lowoha_group_matmul_operator.md` for the full
  // contract and worked example.
  uint32_t total_matmul;                       ///< Total expert weight slots present in the call (>= active_matmul).
  uint32_t active_matmul;                      ///< Count of firing experts (the leading prefix of all weight-side vectors).

  // Per-call cap on weight-cache mode: 0 = disabled, 1 = out-of-place,
  // 2 = allow in-place when the process-wide setting also permits it.
  int32_t weight_cache_type;

  /**
   * @brief Default constructor for matmul_params
   */
  matmul_params() : dtypes(), postop_(), quant_params(), mem_format_a('n'),
    mem_format_b('n'), lowoha_algo(matmul_algo_t::none), num_threads(0),
    plugin_op(""), dynamic_quant(false), packing(), total_matmul(0), active_matmul(0),
    weight_cache_type(2) {}
};

/**
 * @brief Returns the cache mode allowed by both process and call settings.
 *
 * Reads the process-wide weight-cache setting LIVE on every call (not a
 * one-time static cache): the grouped dispatcher may downgrade WC 2->1
 * mid-process before compute, and every other reader (prepack warmers,
 * dispatch gate, CK runtime) already observes `get_weight_cache()` live,
 * so this stays consistent with them.  `get_weight_cache()` is a relaxed
 * atomic load, negligible against the reorder/GEMM work that follows.
 */
static inline int32_t effective_weight_cache_type(int32_t weight_cache_type) {
  const int32_t env_cache_type =
    zendnnl::ops::matmul_config_t::instance().get_weight_cache();
  return std::min(env_cache_type, weight_cache_type);
}

// ── Grouped-matmul weight-cache policy helpers ──────────────────────
// Single source of truth for the two predicates that previously lived
// open-coded across the prepack warmers, the prelude, the CK runtime,
// and the AOCL runtime.  Keeping them here keeps WC=0/1/2 + mixed-in-place
// gating uniform: callers ask the policy rather than re-deriving it.

/**
 * @brief True when the grouped AUTO mixed in-place mode is active.
 *
 * Mixed mode keeps the process weight-cache at 2 while letting ONLY the
 * bf16 full-weight (prompt) AOCL reorder mutate the caller's weight buffer
 * in place; every decode layout (CK, AOCL per-tile) stays out-of-place.
 * The grouped dispatcher sets `grp_auto_mixed_inplace` when WC==2 + AUTO +
 * PREPACK + CROSS_WARM + unlimited LRU capacity all hold.
 */
static inline bool is_grp_auto_mixed_inplace_active() {
  auto &cfg = zendnnl::ops::matmul_config_t::instance();
  return cfg.get_weight_cache() == 2 && cfg.get_grp_auto_mixed_inplace();
}

/**
 * @brief True when a prepack warmer should populate the weight cache for
 *        the given per-call cap.
 *
 * The runtime only consults the LRU cache for WC==1 (out-of-place) and for
 * WC==2 under mixed in-place mode (decode layouts are warmed out-of-place
 * before the prompt in-place mutation).  WC==0 (disabled) and pinned WC==2
 * (no mixed flag) skip warming — the runtime either owns per-call buffers
 * or does a lazy first-call in-place pack.
 */
static inline bool should_warm_weight_cache(int32_t weight_cache_type) {
  return weight_cache_type == 1 ||
         (weight_cache_type == 2 &&
          zendnnl::ops::matmul_config_t::instance()
              .get_grp_auto_mixed_inplace());
}

/**
 * @brief Weight-cache type a warmer should pass for the bf16 full-weight
 *        (prompt) AOCL reorder: in-place (2) only under mixed mode, else
 *        out-of-place (1).  Decode/per-tile warmers always pass 1.
 *        Callers must have already checked `should_warm_weight_cache()`.
 */
static inline int32_t warm_wct_for_full_weight_bf16(int32_t weight_cache_type) {
  return (weight_cache_type == 2 &&
          zendnnl::ops::matmul_config_t::instance()
              .get_grp_auto_mixed_inplace())
             ? 2 : 1;
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // _LOWOHA_COMMON_HPP