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

#ifndef _LOWOHA_REORDER_COMMON_HPP
#define _LOWOHA_REORDER_COMMON_HPP

#define LOWOHA_REORDER_GRAIN_SIZE 1024

#include "memory/memory_utils.hpp"
#include "common/zendnnl_global.hpp"
#include "lowoha_operators/reorder/prepack/lowoha_prepack.hpp"
#include <vector>
#include <cstdint>
#include <cstdlib>

namespace zendnnl {
namespace lowoha {
namespace reorder {

using namespace zendnnl::memory;

/**
 * @brief Supported reorder algorithms
 */
enum class reorder_algo_t : int {
  none = -1,          ///< No specific algorithm
  DT = 0,             ///< Decision tree based algorithm selection
  native = 1,         ///< Native vectorized implementation (AVX512)
  reference = 2,      ///< Reference scalar implementation
  algo_count          ///< Number of algorithms (must be last)
};

/**
 * @brief Structure for reorder quantization parameters
 *
 * Used for quantization (bf16/f32/f16 -> int8/uint8) and dequantization (int8/uint8 -> bf16/f32/f16).
 * Also used for f32 <-> bf16, f32 <-> f16 and bf16 <-> f16 type conversions
 * with optional scaling.
 *
 * Note: For float-only conversions (f32 <-> bf16, f32 <-> f16, bf16 <-> f16),
 *       scale and zero_point are OPTIONAL.  If not provided (buff = nullptr),
 *       simple type conversion is performed.
 *       Default values: scale = 1.0, zero_point = 0
 *
 * Granularity convention for dims (must match tensor dimensionality):
 *
 * For 1D tensor (shape = [N]):
 *   - per-tensor:  dims = {1}       (1 value for all elements)
 *   - per-channel: dims = {N}       (N values, one per element)
 *
 * For 2D tensor (shape = [M, N]):
 *   - per-tensor:      dims = {1, 1}    (1 value for all elements)
 *   - per-channel-col: dims = {1, N}    (N values, one per column)
 *   - per-channel-row: dims = {M, 1}    (M values, one per row, same across columns)
 *   - per-group-row:   dims = {G, N}    (G*N values, M % G == 0, groups divide rows)
 *   - per-group-col:   dims = {M, G}    (M*G values, N % G == 0, groups divide columns)
 *
 * For 3D tensor (shape = [batch, M, N]):
 *   - per-tensor:      dims = {1, 1, 1} (1 value for all elements)
 *   - per-channel-col: dims = {1, 1, N} (N values, one per column)
 *   - per-channel-row: dims = {1, M, 1} (M values, one per row, same across columns)
 *   - per-group-row:   dims = {1, G, N} (G*N values, M % G == 0, groups divide rows)
 *   - per-group-col:   dims = {1, M, G} (M*G values, N % G == 0, groups divide columns)
 *
 * Per-group-row indexing: index = group_idx * N + col, where group_idx = row / (M / G)
 * Per-group-col indexing: index = row * G + group_idx, where group_idx = col / (N / G)
 *
 * Currently supported:
 *   - scale: f32, bf16, or f16 (bf16/f16 are converted to f32 internally on read)
 *   - zero_point: s32 only
 *
 * Dynamic Quantization Mode (when reorder_params_t::dynamic_quant = true):
 *   - User provides mutable buffers (via buff) for scale and zero_point
 *   - dims specifies the desired granularity
 *   - The API computes min/max from source data and fills the buffers
 *   - If dst is nullptr, only computes scale/zp without performing quantization
 */
struct reorder_quant_params_t {
  /**
   * @brief Individual quantization parameter (scale or zero-point)
   *
   * Supports different data types and quantization granularities.
   * For static quantization: user provides pre-computed values (read).
   * For dynamic quantization: user provides buffer to be filled (write).
   */
  struct quant_t {
    void *buff;                    ///< Pointer to quantization data buffer (read for static, write for dynamic)
    data_type_t
    dt;                ///< Data type of the buffer (f32, bf16, or f16 for scale; s32 for zp)
    std::vector<int64_t> dims;     ///< Dimensions matching tensor dimensionality

    /**
     * @brief Default constructor
     */
    quant_t() : buff(nullptr), dt(data_type_t::none), dims() {}
  };

  quant_t scale;                              ///< Scale factor (f32, bf16, or f16; bf16/f16 converted to f32 on read)
  quant_t zero_point;                         ///< Zero point offset (currently s32 only)

  /**
   * @brief Default constructor
   */
  reorder_quant_params_t() : scale(), zero_point() {}
};

/**
 * @brief Main parameter structure for LOWOHA reorder operation
 *
 * Shape format (vector of int64_t):
 *   - Size 1: 1D array [nelems]
 *   - Size 2: 2D matrix [M, N]
 *   - Size 3: 3D batched matrix [batch, M, N]
 *
 * Strides format (vector of int64_t):
 *   - Empty: contiguous memory
 *   - Size 1: 1D array with stride
 *   - Size 2: 2D matrix with strides [stride_M, stride_N]
 *   - Size 3: 3D batched with strides [stride_batch, stride_M, stride_N]
 *
 * Dynamic Quantization Mode (dynamic_quant = true):
 *   - Computes quantization parameters (scale, zero_point) from source data at runtime
 *   - User provides output buffers via quant_params.scale.buff and quant_params.zero_point.buff
 *   - Granularity is determined from quant_params.scale.dims and quant_params.zero_point.dims
 *   - Supported granularities: per-tensor, per-channel-row, per-channel-col, per-group-row, per-group-col
 *   - If dst is nullptr, only computes and fills scale/zp buffers without performing quantization
 *   - Formula: scale = (max - min) / (qmax - qmin), zero_point = qmin - round(min / scale)
 *
 * @note src_shape and dst_shape must be identical. An error will be thrown if they differ.
 * @note dst_strides is reserved for future implementation and is currently not supported.
 *       The destination is always written in contiguous format.
 */
struct reorder_params_t {
  data_type_t src_dtype;                  ///< Source data type
  data_type_t dst_dtype;                  ///< Destination data type
  reorder_quant_params_t quant_params;    ///< Quantization parameters
  reorder_algo_t algo;                    ///< Selected algorithm
  //num_threads is int32_t to match the type used by OpenMP APIs
  int32_t num_threads;                    ///< Number of threads (0 = auto)
  std::vector<int64_t>
  src_shape;         ///< Source shape: [nelems] or [M, N] or [batch, M, N]
  std::vector<int64_t>
  dst_shape;         ///< Destination shape: must match src_shape
  std::vector<int64_t>
  src_strides;       ///< Source strides for non-contiguous memory access
  std::vector<int64_t>
  dst_strides;       ///< Destination strides (reserved for future, not currently supported)
  bool dynamic_quant;                     ///< Enable dynamic quantization (compute scale/zp from source data)
  bool is_prepack;                        ///< True when this call is a weight-prepack request.

  /**
   * @brief Weight-prepack request piggy-backed on reorder_direct.
   *
   * When @c is_prepack is true, @ref reorder_direct dispatches to the
   * weight-prepack pipeline using the fields in @c prepack (algo,
   * wei_dtype, K, N, ldb, ...). The other reorder fields — src_dtype,
   * dst_dtype, shape, quant_params, ... — are then ignored. Default-
   * constructed @c is_prepack is false, so existing reorder_direct
   * callers are unaffected.
   *
   * Workflow:
   *   1. Fill @c reorder_params_t::prepack fields (algo / wei_dtype /
   *      K / N / ldb / transposed / ...) and set @c is_prepack = true.
   *   2. @c size_t size = weight_prepack_size(rp); — returns 0 on error.
   *   3. Allocate a buffer of that size.
   *   4. @c reorder_direct(weights, dst, rp);
   *
   * Note: the same @c rp is reused for both calls — there is no need
   * to keep a separate prepack_params_t variable around.
   */
  prepack_params_t prepack;

  /**
   * @brief Default constructor
   */
  reorder_params_t()
    : src_dtype(data_type_t::none), dst_dtype(data_type_t::none),
      quant_params(), algo(reorder_algo_t::DT), num_threads(0),
      src_shape(), dst_shape(), src_strides(), dst_strides(),
      dynamic_quant(false), is_prepack(false), prepack() {}

  /**
   * @brief Check if this is a 1D shape
   */
  bool is_1d() const {
    return src_shape.size() == 1;
  }

  /**
   * @brief Check if this is a 2D shape
   */
  bool is_2d() const {
    return src_shape.size() == 2;
  }

  /**
   * @brief Check if this is a 3D shape (batched)
   */
  bool is_3d() const {
    return src_shape.size() == 3;
  }

  /**
   * @brief Check if shape is provided and valid
   */
  bool is_shaped() const {
    if (src_shape.empty()) {
      return false;
    }
    for (auto dim : src_shape) {
      if (dim <= 0) {
        return false;
      }
    }
    return true;
  }

  /**
   * @brief Check if src_shape and dst_shape match
   * @return true if shapes are identical, false otherwise
   */
  bool shapes_match() const {
    return src_shape == dst_shape;
  }

  /**
   * @brief Get total number of elements
   */
  int64_t nelems() const {
    if (src_shape.empty()) {
      return 0;
    }
    int64_t n = 1;
    for (auto dim : src_shape) {
      n *= dim;
    }
    return n;
  }

  /**
   * @brief Get batch dimension (1 for 1D/2D)
   */
  int64_t batch() const {
    return is_3d() ? src_shape[0] : 1;
  }

  /**
   * @brief Get M dimension (rows)
   */
  int64_t M() const {
    if (is_1d()) {
      return src_shape[0];
    }
    if (is_2d()) {
      return src_shape[0];
    }
    if (is_3d()) {
      return src_shape[1];
    }
    return 0;
  }

  /**
   * @brief Get N dimension (columns)
   */
  int64_t N() const {
    if (is_1d()) {
      return 1;
    }
    if (is_2d()) {
      return src_shape[1];
    }
    if (is_3d()) {
      return src_shape[2];
    }
    return 0;
  }

  /**
   * @brief Check if source strides are specified
   */
  bool has_src_strides() const {
    return !src_strides.empty();
  }

  /**
   * @brief Check if destination strides are specified (reserved for future)
   */
  bool has_dst_strides() const {
    return !dst_strides.empty();
  }

  /**
   * @brief Check if source memory layout is contiguous
   */
  bool is_src_contiguous() const {
    if (!has_src_strides() || !is_shaped()) {
      return true;  // No strides or no shape means contiguous
    }

    // Check if strides match contiguous layout
    if (src_strides.size() == 1) {
      return src_strides[0] == 1;
    }
    else if (src_strides.size() == 2 && is_2d()) {
      return src_strides[0] == N() && src_strides[1] == 1;
    }
    else if (src_strides.size() == 3 && is_3d()) {
      return src_strides[0] == (M() * N()) &&
             src_strides[1] == N() &&
             src_strides[2] == 1;
    }
    return false;
  }
};

/**
 * @brief Parameters for grouped per-token dynamic quantization.
 *
 * The grouped API processes independently-contiguous source matrices as one
 * logical row collection. This is intended for grouped MoE GEMMs where each
 * expert owns a separate [M_i, K_i] source buffer, but all rows should share a
 * single OpenMP schedule.
 *
 * Current implementation scope:
 *   - per-token symmetric dynamic quantization only
 *   - bf16/f32 source to s8 destination
 *   - scale dtype f32 or bf16
 */
struct group_dynamic_quant_params_t {
  data_type_t src_dtype;
  data_type_t dst_dtype;
  data_type_t scale_dtype;
  int32_t num_threads;

  group_dynamic_quant_params_t()
      : src_dtype(data_type_t::none), dst_dtype(data_type_t::none),
        scale_dtype(data_type_t::f32), num_threads(0) {}
};

/**
 * @brief Get dynamic quantization per-token algorithm override
 *
 * Reads ZENDNNL_DYNAMIC_QUANT_ALGO environment variable once on first call.
 *   0 (or unset) = default (respects API algo: native -> vector fused,
 *                  reference -> scalar unfused)
 *   1 = vector fused,   2 = vector unfused,
 *   3 = scalar fused,   4 = scalar unfused
 *
 * @return algorithm override value (0-4)
 */
inline int32_t get_dynamic_quant_algo_override() {
  static const int32_t val = []() -> int32_t {
    const char *env = std::getenv("ZENDNNL_DYNAMIC_QUANT_ALGO");
    if (env) {
      int32_t v = std::atoi(env);
      if (v >= 1 && v <= 4) {
        return v;
      }
    }
    return 0;
  }();
  return val;
}

/**
 * @brief Check if the AVX512-FP16 (__m512h) FMA kernels are usable.
 *
 * Reorder ships two FP16 AVX-512 backends side-by-side:
 *   - F32-FMA       — AVX-512F + F16C load/store, math in __m512. Every
 *                     shipping AVX-512F-capable CPU also has F16C (the two
 *                     CPUID bits are independent, but commercial silicon
 *                     introduced F16C in 2012 and AVX-512F in 2017, so the
 *                     superset relation holds in practice).
 *   - FP16-FMA      — native __m512h math, requires the AVX512-FP16 ISA
 *                     (Granite Rapids / Sapphire Rapids / Turin) and a
 *                     toolchain that supports __m512h intrinsics
 *                     (GCC 12+).
 *
 * This helper decides which backend the dispatchers should pick.
 * Returns true to authorise the FP16-FMA kernels, false to fall back to
 * the F32-FMA kernels.
 *
 * Selection policy (single source of truth; the dispatchers simply call
 * this helper):
 *   - Library built with @c -DZENDNNL_NATIVE_F32_ACCUM=ON → false
 *     (build-time kill switch for numerical-reproducibility studies).
 *   - Otherwise + GCC < 12                                → false
 *     (__m512h intrinsics unavailable; the FP16-FMA TUs compile to
 *     empty link stubs on older toolchains, so we must avoid calling
 *     them).
 *   - Otherwise                                           → runtime
 *     AVX512-FP16 ISA status reported by zendnnl_platform_info().
 *
 * No runtime environment variable is exposed for this choice.
 */
inline bool can_use_f16_fma_kernel() {
#if !defined(ZENDNNL_NATIVE_F32_ACCUM) && defined(__GNUC__) && (__GNUC__ >= 12)
  return zendnnl::common::zendnnl_platform_info().get_avx512_f16_status();
#else
  return false;
#endif
}

} // namespace reorder
} // namespace lowoha
} // namespace zendnnl

#endif // _LOWOHA_REORDER_COMMON_HPP
