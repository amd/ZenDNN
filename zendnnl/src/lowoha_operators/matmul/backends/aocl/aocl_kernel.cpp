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

#include "lowoha_operators/matmul/backends/aocl/aocl_kernel.hpp"
#include "lowoha_operators/matmul/ggml_weight_unpack.hpp"
#include "lowoha_operators/matmul/backends/aocl/aocl_postop.hpp"
#include "lowoha_operators/matmul/lru_cache/lowoha_cache.hpp"
#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"
#include <cstdlib>
#include <cstring>   // std::memcpy (WC=2 in-place reorder write-back)
#include <mutex>

namespace zendnnl {
namespace lowoha {
namespace matmul {

// Extract nibble from packed 4-bit byte (low nibble if is_low_nibble=true, else high nibble)
// For s4 (signed int4): sign-extends from bit 3, yielding range [-8, 7]
// For u4 (unsigned int4): returns raw nibble, yielding range [0, 15]
inline int8_t extract_4bit_nibble(int8_t packed_byte, bool is_low_nibble,
                                  data_type_t dt) {
  uint8_t ubyte = static_cast<uint8_t>(packed_byte);
  int8_t value = is_low_nibble ? (ubyte & 0x0F) : ((ubyte >> 4) & 0x0F);
  if (dt == data_type_t::s4 && (value & 0x08)) {
    value |= 0xF0;
  }
  return value;
}

void cvt_4bit_to_bf16(const int8_t *weights, bfloat16_t *wei_bf16, int k,
                      int n,
                      int ldb, bool is_transposed, data_type_t wei_dt,
                      const void *scales, const std::vector<int64_t> &scale_dims,
                      data_type_t scale_dt,
                      const void *zp, const std::vector<int64_t> &zp_dims, data_type_t zp_dt) {
  const size_t scale_size = scale_dims.size() > 0 ? get_num_elements(
                              scale_dims) : 0;
  const size_t zp_size = zp_dims.size() > 0 ? get_num_elements(zp_dims) : 0;

  // Cache per-tensor scale/zp values for efficiency
  const float per_tensor_scale = (scale_size == 1) ? read_and_cast<float>(scales,
                                 scale_dt, 0) : 0.0f;
  const float per_tensor_zp = (zp_size == 1) ? read_and_cast<float>(zp, zp_dt,
                              0) : 0.0f;
  bool is_float_domain_u4 = zp_dt == data_type_t::bf16 &&
                            wei_dt == data_type_t::u4;

  // Determine quantization granularity for group size calculation
  int num_groups = 1;
  int group_size = k;
  if (scale_size > 1 && scale_size != static_cast<size_t>(n)) {
    // Per-group quantization: scale_size = G * N
    num_groups = scale_size / n;
    if (num_groups > 0) {
      group_size = k / num_groups;
    }
  }

  // Helper lambda to compute scale/zp offset based on logical (row, col) position
  // For per-tensor:  offset = 0
  // For per-channel: offset = col (N dimension)
  // For per-group:   offset = group * n + col
  auto compute_quant_offset = [&](int row, int col, size_t qsize) -> size_t {
    if (qsize == 0) {
      return 0;  // No quantization
    }
    else if (qsize == 1) {
      return 0;  // Per-tensor
    }
    else if (qsize == static_cast<size_t>(n)) {
      return col;  // Per-channel
    }
    else {
      // Per-group
      int group = row / group_size;
      return group * n + col;
    }
  };

  // Iterate over logical K×N elements
  // Output is always in K×N layout for subsequent reorder
  #pragma omp parallel for collapse(2)
  for (int row = 0; row < k; ++row) {
    for (int col = 0; col < n; ++col) {
      // Calculate physical index in packed S4 buffer based on transpose
      // For non-transposed (ab): physical element at row*ldb + col
      // For transposed (ba):     physical element at col*ldb + row
      size_t physical_idx = is_transposed ?
                            (static_cast<size_t>(col) * ldb + row) :
                            (static_cast<size_t>(row) * ldb + col);
      size_t packed_byte_idx = physical_idx / 2;
      bool is_low_nibble = (physical_idx % 2) == 0;

      // Extract 4-bit value (signed or unsigned based on wei_dt)
      int8_t s4_value = extract_4bit_nibble(weights[packed_byte_idx], is_low_nibble,
                                            wei_dt);
      float dequant_value = static_cast<float>(s4_value);

      float scale_value = 1.0f;
      if (scale_size == 1) {
        scale_value = per_tensor_scale;
      }
      else {
        size_t scale_offset = compute_quant_offset(row, col, scale_size);
        scale_value = read_and_cast<float>(scales, scale_dt, scale_offset);
      }

      float zp_value = 0.0f;
      if (zp_size == 1) {
        zp_value = per_tensor_zp;
      }
      else if (zp_size > 1) {
        size_t zp_offset = compute_quant_offset(row, col, zp_size);
        zp_value = read_and_cast<float>(zp, zp_dt, zp_offset);
      }
      // WOQ FLOAT DOMAIN U4
      if (is_float_domain_u4) {
        dequant_value = (dequant_value -8) * scale_value + zp_value;
      }
      else {
        dequant_value = (dequant_value -zp_value) * scale_value;
      }

      // Store in K×N output layout (row-major, non-transposed)
      size_t out_idx = static_cast<size_t>(row) * n + col;
      wei_bf16[out_idx] = bfloat16_t(dequant_value);
    }
  }
}

// W4A8: widen packed s4 to K×N s8 (sign-extended nibble codes, no dequant).
void cvt_s4_to_s8(const int8_t *weights, int8_t *wei_s8, int k, int n,
                  int ldb, bool is_transposed) {
  #pragma omp parallel for collapse(2)
  for (int row = 0; row < k; ++row) {
    for (int col = 0; col < n; ++col) {
      // Packed s4 index (ab: row*ldb+col; ba: col*ldb+row).
      size_t physical_idx = is_transposed ?
                            (static_cast<size_t>(col) * ldb + row) :
                            (static_cast<size_t>(row) * ldb + col);
      size_t packed_byte_idx = physical_idx / 2;
      bool is_low_nibble = (physical_idx % 2) == 0;

      int8_t s8_value = extract_4bit_nibble(weights[packed_byte_idx],
                                            is_low_nibble, data_type_t::s4);
      size_t out_idx = static_cast<size_t>(row) * n + col;
      wei_s8[out_idx] = s8_value;
    }
  }
}

// W4A8 AOCL sym-quant derives its source group size from the source-scale
// buffer shape. Normalize compact per-tensor/per-token shapes before reorder.
status_t broadcast_w4a8_src_scale(matmul_params &params, int M,
                                  std::vector<uint8_t> &expanded_src_scale) {
  const auto &src_dims = params.quant_params.src_scale.dims;
  if (src_dims.size() != 2) {
    log_error("[AOCL.run_dlp W4A8] source scale dims must be "
              "{1,1}, {M,1}, or {M,G}");
    return status_t::failure;
  }

  if (params.quant_params.wei_scale.dims.size() != 2) {
    log_error("[AOCL.run_dlp W4A8] source scale broadcast requires "
              "per-group weight scale dims {G,N}");
    return status_t::failure;
  }

  const int64_t src_rows = src_dims[0];
  const int64_t src_cols = src_dims[1];
  const int64_t G_dim = params.quant_params.wei_scale.dims[0];

  const void *scale_buff = params.quant_params.src_scale.buff;
  if (!scale_buff) {
    log_error("[AOCL.run_dlp W4A8] source scale buffer is null");
    return status_t::failure;
  }

  if (src_cols != 1) {
    if (src_rows != M || src_cols != G_dim) {
      log_error("[AOCL.run_dlp W4A8] per-group source scale dims must be "
                "{M,G} (rows=", src_rows, ", cols=", src_cols,
                ", M=", M, ", G=", G_dim, ")");
      return status_t::failure;
    }
    return status_t::success;
  }

  const bool is_per_tensor = src_rows == 1;
  if (!is_per_tensor && src_rows != M) {
    log_error("[AOCL.run_dlp W4A8] source scale rows must be 1 or M "
              "(rows=", src_rows, ", M=", M, ")");
    return status_t::failure;
  }

  const int64_t target_rows = M;
  const int64_t target_cols = G_dim;

  const size_t elem_size = size_of(params.quant_params.src_scale.dt);
  expanded_src_scale.resize(static_cast<size_t>(target_rows * target_cols) *
                            elem_size);
  const uint8_t *src_scale_src = static_cast<const uint8_t *>(scale_buff);
  uint8_t *expanded = expanded_src_scale.data();

  #pragma omp parallel for collapse(2)
  for (int64_t m = 0; m < target_rows; ++m) {
    for (int64_t g = 0; g < target_cols; ++g) {
      const int64_t src_m = is_per_tensor ? 0 : m;
      std::memcpy(expanded + static_cast<size_t>(m * target_cols + g) *
                  elem_size,
                  src_scale_src + static_cast<size_t>(src_m) * elem_size,
                  elem_size);
    }
  }

  params.quant_params.src_scale.buff = expanded;
  params.quant_params.src_scale.dims = {target_rows, target_cols};
  apilog_info("[AOCL.run_dlp W4A8] broadcast source scales to shape [",
              target_rows, ",", target_cols, "]");
  return status_t::success;
}

namespace {
template <typename T>
lru_cache_t<Key_matmul, void *> &get_aocl_weight_cache() {
  static lru_cache_t<Key_matmul, void *> c;
  return c;
}
template <typename T>
std::mutex &get_aocl_weight_cache_mutex() {
  static std::mutex m;
  return m;
}
lru_cache_t<Key_matmul, void *> &get_aocl_symquant_weight_cache() {
  static lru_cache_t<Key_matmul, void *> c;
  return c;
}
std::mutex &get_aocl_symquant_weight_cache_mutex() {
  static std::mutex m;
  return m;
}
lru_cache_t<Key_matmul, void *> &get_aocl_woq_weight_cache() {
  static lru_cache_t<Key_matmul, void *> c;
  return c;
}
std::mutex &get_aocl_woq_weight_cache_mutex() {
  static std::mutex m;
  return m;
}
// Separate W4A8 weight cache (s4 widened to blocked s8 layout).
lru_cache_t<Key_matmul, void *> &get_aocl_w4a8_weight_cache() {
  static lru_cache_t<Key_matmul, void *> c;
  return c;
}
std::mutex &get_aocl_w4a8_weight_cache_mutex() {
  static std::mutex m;
  return m;
}

template <typename T>
void clear_aocl_typed_weight_cache_under_lock() {
  std::lock_guard<std::mutex> lock(get_aocl_weight_cache_mutex<T>());
  get_aocl_weight_cache<T>().clear();
}
void clear_aocl_woq_weight_cache_under_lock() {
  std::lock_guard<std::mutex> lock(get_aocl_woq_weight_cache_mutex());
  get_aocl_woq_weight_cache().clear();
}
void clear_aocl_w4a8_weight_cache_under_lock() {
  std::lock_guard<std::mutex>
  lock(get_aocl_w4a8_weight_cache_mutex());
  get_aocl_w4a8_weight_cache().clear();
}

void clear_aocl_symquant_weight_cache_under_lock() {
  std::lock_guard<std::mutex> lock(get_aocl_symquant_weight_cache_mutex());
  get_aocl_symquant_weight_cache().clear();
}
}  // namespace

void clear_aocl_matmul_weight_caches() {
  // Lock only the mutex paired with each cache so clear does not interleave
  // reorderAndCacheWeights* between find_key() and get()/add(), without
  // blocking all dtypes for the entire teardown.
  clear_aocl_typed_weight_cache_under_lock<float>();
  clear_aocl_typed_weight_cache_under_lock<int16_t>();
  clear_aocl_typed_weight_cache_under_lock<uint16_t>();
  clear_aocl_typed_weight_cache_under_lock<int8_t>();
  clear_aocl_symquant_weight_cache_under_lock();
  clear_aocl_woq_weight_cache_under_lock();
  clear_aocl_w4a8_weight_cache_under_lock();
  clear_ggml_weight_unpack_cache();
  clear_zp_compensation_cache();
}

template <typename T>
bool reorderAndCacheWeights(Key_matmul key, const void *weights,
                            void *&reorder_weights, const int k, const int n, const int ldb,
                            const char order, const char trans, char mem_format_b,
                            get_reorder_buff_size_func_ptr get_reorder_buf_size,
                            reorder_func_ptr<T> reorder_func, int weight_cache_type) {
  // Weight caching
  lru_cache_t<Key_matmul, void *> &matmul_weight_cache =
    get_aocl_weight_cache<T>();
  std::mutex &weight_cache_mutex = get_aocl_weight_cache_mutex<T>();

  // Weights are already reordered and algo is aocl_dlp_blocked
  // Add the key into map and value as nullptr
  // Modify the reorder_weight as weight.
  if (mem_format_b == 'r') {
    matmul_weight_cache.add(key, nullptr);
    reorder_weights = const_cast<void *>(weights);
    return true;
  }

  // The three per-call cache-event lines below are demoted to
  // apilog_verbose: they fire per-expert from inside the ALGO 3 /
  // ALGO 1 OMP region (when called via group_matmul) and can amplify
  // to thousands of lines per benchmark iteration.  Info level
  // (`ZENDNNL_API_LOG_LEVEL=3`) keeps the consolidated
  // [GRP_MATMUL.PREPACK] / [GRP_MATMUL.PROBE] summary lines instead;
  // verbose level (=4) brings the per-event detail back.
  if (weight_cache_type == 0) {
    apilog_verbose("[AOCL.reorder] WEIGHT_CACHE_DISABLE — out-of-place reorder");
    size_t b_reorder_buf_siz_req = get_reorder_buf_size(order, trans, 'B',
                                   k, n,nullptr);
    size_t alignment      = 64;
    size_t reorder_size   = (b_reorder_buf_siz_req + alignment - 1) & ~
                            (alignment - 1);
    reorder_weights       = (T *)aligned_alloc(alignment, reorder_size);
    reorder_func(order, trans, 'B', (T *)weights, (T *)reorder_weights, k, n, ldb,
                 nullptr);
  }
  // Out-of-place reordering
  else if (weight_cache_type == 1) {
    std::lock_guard<std::mutex> lock(weight_cache_mutex);
    void *cached_ptr = nullptr;
    bool found_obj = matmul_weight_cache.try_get(key, cached_ptr);
    if (!found_obj) {
      apilog_verbose("[AOCL.reorder MISS] weight cache miss — packing weights");
      size_t b_reorder_buf_siz_req = get_reorder_buf_size(order, trans, 'B', k, n,
                                     nullptr);
      size_t alignment      = 64;
      size_t reorder_size   = (b_reorder_buf_siz_req + alignment - 1) & ~
                              (alignment - 1);
      reorder_weights = (T *)aligned_alloc(alignment, reorder_size);
      reorder_func(order, trans, 'B', (T *)weights, (T *)reorder_weights, k, n, ldb,
                   nullptr);
      // Create new entry
      matmul_weight_cache.add(key, reorder_weights);
    }
    else {
      // The same cache key may already hold a nullptr from the prepacked
      // mem_format_b == 'r' path above, or from an earlier in-place reorder.
      // In both cases the caller's weight buffer is the reordered buffer.
      if (cached_ptr == nullptr) {
        apilog_info("Read AOCL cached weights WEIGHT_CACHE_OUT_OF_PLACE "
                    "(reusing user buffer)");
        reorder_weights = const_cast<void *>(weights);
      }
      else {
        apilog_info("Read AOCL cached weights WEIGHT_CACHE_OUT_OF_PLACE");
        reorder_weights = cached_ptr;
      }
    }
  }
  // In-place reordering: the user's weight buffer is reused as the reorder
  // destination, so the cache only needs to remember that the reorder has
  // been performed; the reordered bytes already live in the user's buffer.
  // We therefore store @c nullptr as the cache value (the same convention
  // used by the @c mem_format_b == 'r' early-return path above): this
  // avoids the LRU eviction path treating the cached entry as an owned
  // allocation and calling @c std::free on the user's weight buffer.
  //
  // If the reordered layout cannot fit inside the user's allocation we
  // transparently fall back to an out-of-place allocation; in that case the
  // cache stores the actual reordered buffer pointer just like
  // @c weight_cache_type == 1. The cache-hit branch distinguishes the two
  // by inspecting whether the stored pointer is null.
  else if (weight_cache_type == 2) {
    std::lock_guard<std::mutex> lock(weight_cache_mutex);
    void *cached_ptr = nullptr;
    if (matmul_weight_cache.try_get(key, cached_ptr)) {
      if (cached_ptr == nullptr) {
        apilog_info("Read AOCL cached weights WEIGHT_CACHE_IN_PLACE "
                    "(reusing user buffer)");
        reorder_weights = const_cast<void *>(weights);
      }
      else {
        apilog_info("Read AOCL cached weights WEIGHT_CACHE_IN_PLACE "
                    "(fall-back out-of-place buffer)");
        reorder_weights = cached_ptr;
      }
      return true;
    }

    size_t b_reorder_buf_siz_req = get_reorder_buf_size(order, trans, 'B', k, n,
                                   nullptr);
    // Two-part gate for engaging the in-place path:
    //
    //   1. b_reorder_buf_siz_req == plain_size
    //      Strict size equality matches the old library's
    //      WEIGHT_CACHE_INPLACE branch (zendnn_reorder_cache.cpp).
    //      Any padding/internal blocking changes the size and means the
    //      in-place mutation cannot be safely re-derived from the user's
    //      buffer on a later call.
    //   2. reorder_size == plain_size
    //      The out-of-place path allocates @c reorder_size bytes (rounded
    //      up to the 64-byte alignment) so AOCL's SIMD-aware kernels can
    //      safely overscan the blocked layout. The user's weight
    //      allocation is only guaranteed to be @c plain_size bytes; if
    //      @c reorder_size > plain_size, an in-place mutation would leave
    //      0..63 bytes of overscan reading past the user's allocation.
    //      Requiring @c plain_size to already be 64-byte aligned closes
    //      that window.
    //
    // When either part fails, fall through to allocating a fresh
    // out-of-place cache buffer and leave the user's buffer untouched.
    size_t plain_size   = static_cast<size_t>(k) * static_cast<size_t>(n)
                          * sizeof(T);
    size_t alignment    = 64;
    size_t reorder_size = (b_reorder_buf_siz_req + alignment - 1) & ~
                          (alignment - 1);

    if (b_reorder_buf_siz_req == plain_size && reorder_size == plain_size) {
      apilog_info("AOCL reorder weights WEIGHT_CACHE_IN_PLACE");
      T *interim = (T *)aligned_alloc(alignment, reorder_size);
      if (!interim) {
        apilog_error("AOCL in-place reorder: aligned_alloc failed, "
                     "falling back to out-of-place");
        reorder_weights = (T *)aligned_alloc(alignment, reorder_size);
        if (!reorder_weights) {
          return false;
        }
        reorder_func(order, trans, 'B', (T *)weights, (T *)reorder_weights,
                     k, n, ldb, nullptr);
        matmul_weight_cache.add(key, reorder_weights);
        return true;
      }
      reorder_func(order, trans, 'B', (T *)weights, interim, k, n, ldb, nullptr);
      std::memcpy(const_cast<void *>(weights), interim, b_reorder_buf_siz_req);
      std::free(interim);
      reorder_weights = const_cast<void *>(weights);
      // Store nullptr sentinel: the user buffer itself holds the
      // reordered bytes. This keeps the LRU evictor from free()-ing the
      // user's buffer. Caller contract: do not clear/evict this cache
      // while reusing the in-place-mutated weight buffer.
      matmul_weight_cache.add(key, nullptr);
    }
    else {
      apilog_info("AOCL reorder weights WEIGHT_CACHE_IN_PLACE "
                  "(blocked size != plain size, falling back to out-of-place)");
      reorder_weights = (T *)aligned_alloc(alignment, reorder_size);
      if (reorder_weights == nullptr) {
        apilog_error("AOCL in-place reorder fall-back: aligned_alloc failed");
        return false;
      }
      reorder_func(order, trans, 'B', (T *)weights, (T *)reorder_weights,
                   k, n, ldb, nullptr);
      matmul_weight_cache.add(key, reorder_weights);
    }
  }
  return true;
}

template bool reorderAndCacheWeights<int16_t>(Key_matmul, const void *, void *&,
    int, int, int, char, char, char, get_reorder_buff_size_func_ptr,
    reorder_func_ptr<int16_t>, int);
template bool reorderAndCacheWeights<float>(Key_matmul, const void *, void *&,
    int, int, int, char, char, char, get_reorder_buff_size_func_ptr,
    reorder_func_ptr<float>, int);
template bool reorderAndCacheWeights<int8_t>(Key_matmul, const void *, void *&,
    int, int, int, char, char, char, get_reorder_buff_size_func_ptr,
    reorder_func_ptr<int8_t>, int);
template bool reorderAndCacheWeights<uint16_t>(Key_matmul, const void *,
    void *&,
    int, int, int, char, char, char, get_reorder_buff_size_func_ptr,
    reorder_func_ptr<uint16_t>, int);

template <typename T>
bool reorderAndCacheWeightsSymQuant(Key_matmul key, const void *weights,
                                    void *&reorder_weights, const int k, const int n, const int ldb,
                                    const char order, const char trans, char mem_format_b,
                                    get_reorder_buf_size_sym_quant_func_ptr get_reorder_buf_size,
                                    reorder_sym_quant_func_ptr<T> reorder_func,
                                    DLP_SYMM_STAT_QUANT *symq_meta, int weight_cache_type) {

  lru_cache_t<Key_matmul, void *> &matmul_weight_cache =
    get_aocl_symquant_weight_cache();
  std::mutex &weight_cache_mutex = get_aocl_symquant_weight_cache_mutex();

  if (mem_format_b == 'r') {
    matmul_weight_cache.add(key, nullptr);
    reorder_weights = const_cast<void *>(weights);
    return true;
  }

  if (weight_cache_type == 0) {
    apilog_verbose("[AOCL.reorder symquant] WEIGHT_CACHE_DISABLE — "
                   "out-of-place reorder for s8 weights (GEMM = s8s8_sym_quant)");
    size_t b_reorder_buf_siz_req = get_reorder_buf_size(order, trans, 'B',
                                   k, n, symq_meta, nullptr);
    size_t alignment    = 64;
    size_t reorder_size = (b_reorder_buf_siz_req + alignment - 1) & ~
                          (alignment - 1);
    reorder_weights     = (T *)aligned_alloc(alignment, reorder_size);
    if (!reorder_weights) {
      apilog_error("AOCL sym_quant reorder weights: aligned_alloc failed");
      return false;
    }
    reorder_func(order, trans, 'B', (T *)weights, (T *)reorder_weights,
                 k, n, ldb, symq_meta, nullptr);
  }
  else if (weight_cache_type == 1) {
    std::lock_guard<std::mutex> lock(weight_cache_mutex);
    void *cached_ptr = nullptr;
    bool found_obj = matmul_weight_cache.try_get(key, cached_ptr);
    if (!found_obj) {
      apilog_verbose("[AOCL.reorder symquant MISS] weight cache miss — packing");
      size_t b_reorder_buf_siz_req = get_reorder_buf_size(order, trans, 'B',
                                     k, n, symq_meta, nullptr);
      size_t alignment    = 64;
      size_t reorder_size = (b_reorder_buf_siz_req + alignment - 1) & ~
                            (alignment - 1);
      reorder_weights = (T *)aligned_alloc(alignment, reorder_size);
      reorder_func(order, trans, 'B', (T *)weights, (T *)reorder_weights,
                   k, n, ldb, symq_meta, nullptr);
      matmul_weight_cache.add(key, reorder_weights);
    }
    else {
      // See reorderAndCacheWeights: nullptr means a prior prepack or
      // in-place path made the caller's weight buffer the reordered buffer.
      if (cached_ptr == nullptr) {
        apilog_info("Read AOCL sym_quant cached weights "
                    "WEIGHT_CACHE_OUT_OF_PLACE (reusing user buffer)");
        reorder_weights = const_cast<void *>(weights);
      }
      else {
        apilog_info("Read AOCL sym_quant cached weights WEIGHT_CACHE_OUT_OF_PLACE");
        reorder_weights = cached_ptr;
      }
    }
  }
  // In-place reordering for the sym_quant path. Mirrors the standard
  // reorderAndCacheWeights in-place branch: reorder into a temporary
  // buffer, memcpy the result back into the user's weight buffer, and
  // record the reorder by adding @c nullptr to the cache (the user buffer
  // itself holds the reordered bytes). On the fall-back path (reorder
  // doesn't fit in the user's buffer) the cache stores the actual reorder
  // buffer just like the out-of-place branch.
  else if (weight_cache_type == 2) {
    std::lock_guard<std::mutex> lock(weight_cache_mutex);
    void *cached_ptr = nullptr;
    if (matmul_weight_cache.try_get(key, cached_ptr)) {
      if (cached_ptr == nullptr) {
        apilog_info("Read AOCL sym_quant cached weights WEIGHT_CACHE_IN_PLACE "
                    "(reusing user buffer)");
        reorder_weights = const_cast<void *>(weights);
      }
      else {
        apilog_info("Read AOCL sym_quant cached weights WEIGHT_CACHE_IN_PLACE "
                    "(fall-back out-of-place buffer)");
        reorder_weights = cached_ptr;
      }
      return true;
    }

    size_t b_reorder_buf_siz_req = get_reorder_buf_size(order, trans, 'B',
                                   k, n, symq_meta, nullptr);
    // See reorderAndCacheWeights for the rationale -- two-part gate:
    // (1) blocked size equals the plain k*n size (old library's
    //     WEIGHT_CACHE_INPLACE check), and
    // (2) plain_size is already 64-byte aligned so the user's
    //     allocation is large enough to absorb any SIMD overscan the
    //     AOCL kernel may perform on the blocked layout.
    size_t plain_size   = static_cast<size_t>(k) * static_cast<size_t>(n)
                          * sizeof(T);
    size_t alignment    = 64;
    size_t reorder_size = (b_reorder_buf_siz_req + alignment - 1) & ~
                          (alignment - 1);

    if (b_reorder_buf_siz_req == plain_size && reorder_size == plain_size) {
      apilog_info("AOCL sym_quant reorder weights WEIGHT_CACHE_IN_PLACE");
      T *interim = (T *)aligned_alloc(alignment, reorder_size);
      if (!interim) {
        apilog_error("AOCL sym_quant in-place reorder: aligned_alloc failed, "
                     "falling back to out-of-place");
        reorder_weights = (T *)aligned_alloc(alignment, reorder_size);
        if (!reorder_weights) {
          return false;
        }
        reorder_func(order, trans, 'B', (T *)weights, (T *)reorder_weights,
                     k, n, ldb, symq_meta, nullptr);
        matmul_weight_cache.add(key, reorder_weights);
        return true;
      }
      reorder_func(order, trans, 'B', (T *)weights, interim, k, n, ldb,
                   symq_meta, nullptr);
      std::memcpy(const_cast<void *>(weights), interim, b_reorder_buf_siz_req);
      std::free(interim);
      reorder_weights = const_cast<void *>(weights);
      // See reorderAndCacheWeights: nullptr means the user buffer holds
      // the reordered bytes and must remain associated with this cache
      // entry while the buffer is reused.
      matmul_weight_cache.add(key, nullptr);
    }
    else {
      apilog_info("AOCL sym_quant reorder weights WEIGHT_CACHE_IN_PLACE "
                  "(blocked size != plain size, falling back to out-of-place)");
      reorder_weights = (T *)aligned_alloc(alignment, reorder_size);
      if (!reorder_weights) {
        return false;
      }
      reorder_func(order, trans, 'B', (T *)weights, (T *)reorder_weights,
                   k, n, ldb, symq_meta, nullptr);
      matmul_weight_cache.add(key, reorder_weights);
    }
  }
  return true;
}

template bool reorderAndCacheWeightsSymQuant<int8_t>(Key_matmul, const void *,
    void *&, int, int, int, char, char, char,
    get_reorder_buf_size_sym_quant_func_ptr,
    reorder_sym_quant_func_ptr<int8_t>, DLP_SYMM_STAT_QUANT *, int);

void woqReorderAndCacheWeightsAocl(Key_matmul key, const int8_t *weights,
                                   void *&reorder_weights, const int k, const int n, const int ldb,
                                   const bool is_weights_const, const char order, const char trans,
                                   char mem_format_b,
                                   const matmul_quantization_params_t &quant_params,
                                   data_type_t wei_dt,
                                   int weight_cache_type) {
  // WOQ always converts 4-bit weights to bf16 before reordering, so the
  // reordered buffer is strictly larger than the caller's 4-bit weight
  // buffer. In-place caching is therefore not feasible for the WOQ path; the
  // caller (run_dlp) downgrades weight_cache_type == 2 to 1 before invoking
  // this function so the cache, allocation and cleanup logic all stay
  // consistent.
  lru_cache_t<Key_matmul, void *> &matmul_weight_cache_woq =
    get_aocl_woq_weight_cache();
  std::mutex &woq_cache_mutex = get_aocl_woq_weight_cache_mutex();

  bool is_transposed = (trans == 't');

  // Use lock guard to protect the entire check-compute-cache operation
  std::lock_guard<std::mutex> lock(woq_cache_mutex);
  // Short-circuit: only consult (and timestamp-bump) the cache when weights
  // are const. When !is_weights_const the value is recomputed regardless, so
  // skipping the lookup preserves the original no-bump behavior of that path.
  void *cached_woq = nullptr;
  bool found_obj = is_weights_const &&
                   matmul_weight_cache_woq.try_get(key, cached_woq);

  if (!found_obj) {
    apilog_verbose("[AOCL.reorder WOQ MISS] simulated WOQ reorder "
                   "(weight_cache_type=", weight_cache_type, ")");
    size_t alignment = 64;
    size_t cvt_weights_size = (sizeof(bfloat16_t)*k*n + alignment - 1) & ~
                              (alignment - 1);
    bfloat16_t *cvt_weights = (bfloat16_t *)aligned_alloc(alignment,
                              cvt_weights_size);
    cvt_4bit_to_bf16(weights, cvt_weights, k, n, ldb, is_transposed, wei_dt,
                     quant_params.wei_scale.buff, quant_params.wei_scale.dims,
                     quant_params.wei_scale.dt, quant_params.wei_zp.buff,
                     quant_params.wei_zp.dims, quant_params.wei_zp.dt);
    // After cvt_4bit_to_bf16, weights are in K×N layout (non-transposed), ldb_cvt = n
    int ldb_cvt = n;
    size_t b_reorder_buf_siz_req = aocl_get_reorder_buf_size_bf16bf16f32of32(order,
                                   'n', 'B', k, n
                                   ,nullptr);
    size_t reorder_size   = (b_reorder_buf_siz_req + alignment - 1) & ~
                            (alignment - 1);
    reorder_weights = (int16_t *)aligned_alloc(alignment, reorder_size);
    aocl_reorder_bf16bf16f32of32(order, 'n', 'B', (int16_t *)cvt_weights,
                                 (int16_t *)reorder_weights, k, n, ldb_cvt
                                 ,nullptr);
    free(cvt_weights);
    if (is_weights_const && weight_cache_type == 1) {
      // Create new entry
      matmul_weight_cache_woq.add(key, reorder_weights);
    }
  }
  else {
    apilog_verbose("[AOCL.reorder WOQ HIT] simulated WOQ cache hit — "
                   "reusing cached pack");
    reorder_weights = cached_woq;
  }
}

// W4A8: s4 -> s8 widen, AOCL sym_quant int8 reorder, optional cache.
void w4a8ReorderAndCacheWeightsAocl(Key_matmul key, const int8_t *weights,
                                    void *&reorder_weights, const int k, const int n, const int ldb,
                                    const bool is_weights_const, const char order, const char trans,
                                    data_type_t wei_dt, data_type_t src_dt,
                                    int weight_cache_type,
                                    int sym_quant_group_size) {
  lru_cache_t<Key_matmul, void *> &matmul_weight_cache_w4a8 =
    get_aocl_w4a8_weight_cache();
  std::mutex &w4a8_cache_mutex =
    get_aocl_w4a8_weight_cache_mutex();

  bool is_transposed = (trans == 't');

  // Only signed s4 weights are supported -- unsigned u4 needs explicit
  // zero-point semantics and a different conversion routine.
  if (wei_dt != data_type_t::s4) {
    apilog_error("[AOCL.reorder W4A8] supports only s4 weights; "
                 "wei_dt is not s4");
    reorder_weights = nullptr;
    return;
  }
  if (src_dt != data_type_t::s8) {
    apilog_error("[AOCL.reorder W4A8] supports only s8 source; "
                 "src_dt is unsupported");
    reorder_weights = nullptr;
    return;
  }

  std::lock_guard<std::mutex> lock(w4a8_cache_mutex);
  // Short-circuit: only consult (and timestamp-bump) the cache when
  // weights are const. A non-const-weight call always recomputes and
  // discards the buffer.
  void *cached_w4a8 = nullptr;
  bool found_obj = is_weights_const &&
                   matmul_weight_cache_w4a8.try_get(key, cached_w4a8);

  if (!found_obj) {
    apilog_verbose("[AOCL.reorder W4A8 MISS] W4A8 reorder "
                   "(weight_cache_type=", weight_cache_type, ", src_dt=",
                   static_cast<int>(src_dt), ")");
    size_t alignment = 64;
    // Temp K×N s8 buffer; freed before return (cache holds blocked reorder).
    size_t cvt_weights_size = (sizeof(int8_t) * k * n + alignment - 1) & ~
                              (alignment - 1);
    int8_t *cvt_weights = (int8_t *)aligned_alloc(alignment, cvt_weights_size);
    if (!cvt_weights) {
      apilog_error("[AOCL.reorder W4A8] failed to allocate convert weights");
      reorder_weights = nullptr;
      return;
    }
    cvt_s4_to_s8(weights, cvt_weights, k, n, ldb, is_transposed);

    // After cvt_s4_to_s8, weights are in K×N row-major layout
    // (non-transposed), so ldb_cvt = n regardless of the original packed
    // buffer's layout.
    int ldb_cvt = n;
    char trans_cvt = 'n';

    size_t b_reorder_buf_siz_req = 0;
    size_t reorder_size = 0;
    DLP_SYMM_STAT_QUANT symq_meta;
    symq_meta.group_size = sym_quant_group_size > 0 ? sym_quant_group_size : k;
    b_reorder_buf_siz_req =
      aocl_get_reorder_buf_size_s8s8s32os32_sym_quant(order, trans_cvt, 'B',
          k, n, &symq_meta, nullptr);
    reorder_size = (b_reorder_buf_siz_req + alignment - 1) & ~
                   (alignment - 1);
    reorder_weights = (int8_t *)aligned_alloc(alignment, reorder_size);
    if (!reorder_weights) {
      apilog_error("[AOCL.reorder W4A8] failed to allocate reorder weights");
      free(cvt_weights);
      reorder_weights = nullptr;
      return;
    }
    aocl_reorder_s8s8s32os32_sym_quant(order, trans_cvt, 'B', cvt_weights,
                                       (int8_t *)reorder_weights, k, n,
                                       ldb_cvt, &symq_meta, nullptr);
    free(cvt_weights);

    if (is_weights_const && weight_cache_type == 1) {
      matmul_weight_cache_w4a8.add(key, reorder_weights);
    }
  }
  else {
    apilog_verbose("[AOCL.reorder W4A8 HIT] W4A8 cache hit — "
                   "reusing cached pack");
    reorder_weights = cached_w4a8;
  }
}

void run_dlp(char layout, char transA, char transB, int M, int N,
             int K,
             float alpha, float beta, int lda, int ldb, int ldc,
             char mem_format_a, char mem_format_b, const void *A,
             const void *B, void *C, const matmul_data_types &dtypes,
             const matmul_params &lowoha_param, const void *bias,
             zendnnl::ops::matmul_algo_t kernel,
             bool is_weights_const) {

  bool is_weight_blocked = false;
  void *reordered_mem = nullptr;
  bool simulated_woq_free_buff = false;
  matmul_config_t &matmul_config = matmul_config_t::instance();
  int32_t weight_cache_type =
    effective_weight_cache_type(lowoha_param.weight_cache_type);

  // Grouped AUTO mixed-in-place mode keeps the process WC at 2, but in that
  // mode the ONLY layout allowed to mutate the weight buffer in place is the
  // bf16 full-weight (prompt) AOCL reorder.  That is safe because it runs
  // LAST: cross-warm first packs every decode layout (CK pack, AOCL per-tile)
  // OUT-OF-PLACE from the RAW weights, so by the time the full-weight reorder
  // mutates the buffer nothing else still needs the raw bytes.
  //
  // Non-bf16 weights have no in-place-safe pre-warm (the full-weight in-place
  // warmer is bf16-only).  f32 has no warmer at all, so leaving it at WC=2
  // here would let this reorder mutate the buffer in place lazily at runtime,
  // after which a later / concurrent out-of-place reorder of the SAME buffer
  // would read corrupted bytes.  (int8 sym-quant is already warmed + served
  // out-of-place -- its blocked layout carries a compensation row so it can
  // never satisfy the in-place size gate -- but forcing it here keeps it
  // unambiguously on the out-of-place branch.)  Force every non-bf16 dtype
  // out-of-place under mixed mode so only bf16 ever mutates in place.
  //
  // Gated on is_grp_auto_mixed_inplace_active() (process WC==2 AND the grouped
  // AUTO mixed flag, which only the grouped dispatcher sets), so this is INERT
  // for single matmul / BMM and for pinned WC=2 -- none of those set the mixed
  // flag, so the branch never fires for them.
  if (weight_cache_type == 2
      && dtypes.wei != data_type_t::bf16
      && is_grp_auto_mixed_inplace_active()) {
    weight_cache_type = 1;
  }

  size_t run_src_scale_nelems = get_num_elements(
                                  lowoha_param.quant_params.src_scale.dims);

  // s8×s8 sym_quant uses a distinct blocked weight layout. bf16/f32 per-token
  // still runs bf16s8/f32s8 GEMMs with standard s8 blocked weights + a_pre/a_post.
  const bool is_s8_sym_quant_scales =
    dtypes.wei == data_type_t::s8 &&
    !lowoha_param.quant_params.src_zp.buff &&
    run_src_scale_nelems > 1 &&
    (dtypes.dst == data_type_t::f32 || dtypes.dst == data_type_t::bf16) &&
    dtypes.src == data_type_t::s8;

  // W4A8: s4 wei + dynamic-quant s8 src. Used to skip WOQ s4 reorder below.
  const bool is_w4a8 =
    (kernel == zendnnl::ops::matmul_algo_t::aocl_dlp ||
     kernel == zendnnl::ops::matmul_algo_t::aocl_dlp_blocked) &&
    is_w4a8_config(lowoha_param);

  // W4A8 post-op wiring needs broadcast src scales and wei typed as s8.
  matmul_params w4a8_lowoha_param = lowoha_param;
  std::vector<uint8_t> w4a8_expanded_src_scale;
  matmul_data_types dtypes_for_postop = dtypes;
  if (is_w4a8) {
    if (broadcast_w4a8_src_scale(w4a8_lowoha_param, M,
                                 w4a8_expanded_src_scale) !=
        status_t::success) {
      return;
    }
    dtypes_for_postop.wei = data_type_t::s8;
  }

  size_t cache_extra_hash = 0;
  if (is_s8_sym_quant_scales) {
    int64_t src_grp = (run_src_scale_nelems == static_cast<size_t>(M))
                      ? K : K / (static_cast<int64_t>(run_src_scale_nelems) / M);
    cache_extra_hash = std::hash<int64_t> {}(src_grp);
  }
  Key_matmul cache_key(transB == 't', K, N, ldb, B,
                       static_cast<uint32_t>(matmul_algo_t::aocl_dlp_blocked),
                       cache_extra_hash);

  // AOCL blocked kernel reordering for 2D MatMul
  if (kernel==zendnnl::ops::matmul_algo_t::aocl_dlp_blocked &&
      is_weights_const && !is_w4a8) {
    //call reorder and cache function
    bool blocked_flag = false;
    if (lowoha_param.dtypes.wei == data_type_t::f32) {
      blocked_flag = reorderAndCacheWeights<float>(cache_key, B, reordered_mem, K, N,
                     ldb,
                     'r', transB, mem_format_b,
                     aocl_get_reorder_buf_size_f32f32f32of32, aocl_reorder_f32f32f32of32,
                     weight_cache_type);
    }
    else if (lowoha_param.dtypes.wei == data_type_t::bf16) {
      blocked_flag = reorderAndCacheWeights<int16_t>(cache_key, B, reordered_mem, K,
                     N, ldb,
                     'r', transB, mem_format_b,
                     aocl_get_reorder_buf_size_bf16bf16f32of32, aocl_reorder_bf16bf16f32of32,
                     weight_cache_type);
    }
    else if (lowoha_param.dtypes.wei == data_type_t::f16) {
      blocked_flag = reorderAndCacheWeights<uint16_t>(cache_key, B, reordered_mem, K,
                     N, ldb, 'r', transB, mem_format_b,
                     aocl_get_reorder_buf_size_f16f16f16of16, aocl_reorder_f16f16f16of16,
                     weight_cache_type);
    }
    else if (lowoha_param.dtypes.wei == data_type_t::s4 ||
             lowoha_param.dtypes.wei == data_type_t::u4) {
      blocked_flag = reorderAndCacheWeights<int8_t>(cache_key, B, reordered_mem, K,
                     N, ldb,
                     'r', transB, mem_format_b,
                     aocl_get_reorder_buf_size_bf16s4f32of32, aocl_reorder_bf16s4f32of32,
                     weight_cache_type);
    }
    else if (lowoha_param.dtypes.wei == data_type_t::s8) {
      if (is_s8_sym_quant_scales) {
        int64_t src_grp = (run_src_scale_nelems == static_cast<size_t>(M))
                          ? K : K / (static_cast<int64_t>(run_src_scale_nelems) / M);
        DLP_SYMM_STAT_QUANT symq_meta;
        symq_meta.group_size = static_cast<int>(src_grp);
        blocked_flag = reorderAndCacheWeightsSymQuant<int8_t>(cache_key, B,
                       reordered_mem, K, N, ldb,
                       'r', transB, mem_format_b,
                       aocl_get_reorder_buf_size_s8s8s32os32_sym_quant,
                       aocl_reorder_s8s8s32os32_sym_quant,
                       &symq_meta, weight_cache_type);
      }
      else if (lowoha_param.dtypes.src == data_type_t::s8 ||
               lowoha_param.dtypes.src == data_type_t::bf16 ||
               lowoha_param.dtypes.src == data_type_t::f32) {
        blocked_flag = reorderAndCacheWeights<int8_t>(cache_key, B, reordered_mem, K,
                       N, ldb,
                       'r', transB, mem_format_b,
                       aocl_get_reorder_buf_size_s8s8s32os32, aocl_reorder_s8s8s32os32,
                       weight_cache_type);
      }
      else if (lowoha_param.dtypes.src == data_type_t::u8) {
        blocked_flag = reorderAndCacheWeights<int8_t>(cache_key, B, reordered_mem, K,
                       N, ldb,
                       'r', transB, mem_format_b,
                       aocl_get_reorder_buf_size_u8s8s32os32, aocl_reorder_u8s8s32os32,
                       weight_cache_type);
      }
    }
    if (blocked_flag) {
      is_weight_blocked = true;
      mem_format_b = 'r';
    }
  }
  else if (kernel == zendnnl::ops::matmul_algo_t::aocl_dlp &&
           (dtypes.wei == data_type_t::s4 || dtypes.wei == data_type_t::u4) &&
           dtypes.src == data_type_t::bf16) {
    // WOQ expands the 4-bit input into a bf16-blocked layout, so the
    // reordered buffer cannot fit inside the caller's weight buffer.
    // Treat in-place caching the same as out-of-place caching for this path.
    //
    // bf16 src only: WOQ path; W4A8 (s8 src) is handled in the kernel section.
    int32_t woq_weight_cache_type = (weight_cache_type == 2) ? 1 :
                                    weight_cache_type;
    //call woq reorder and cache function
    woqReorderAndCacheWeightsAocl(cache_key, static_cast<const int8_t *>(B),
                                  reordered_mem, K,
                                  N, ldb, is_weights_const, 'r', transB, mem_format_b,
                                  lowoha_param.quant_params, dtypes.wei,
                                  woq_weight_cache_type);
    is_weight_blocked = true;
    mem_format_b = 'r';
    simulated_woq_free_buff = !is_weights_const || woq_weight_cache_type != 1;
  }

  // Compute zero-point compensation for INT8 (with caching for 1D case)
  int32_t *zp_comp_acc = nullptr;
  int zp_comp_ndim = 0;
  int32_t src_zp = 0;
  int32_t wei_zp = 0;
  bool is_int8 = dtypes.wei == data_type_t::s8;
  if (is_int8) {
    // Extract zero-point values
    if (lowoha_param.quant_params.src_zp.buff &&
        dtypes.src != data_type_t::bf16 &&
        dtypes.src != data_type_t::f32) {
      src_zp = read_and_cast<int32_t>(lowoha_param.quant_params.src_zp.buff,
                                      lowoha_param.quant_params.src_zp.dt);
    }
    if (lowoha_param.quant_params.wei_zp.buff) {
      wei_zp = read_and_cast<int32_t>(lowoha_param.quant_params.wei_zp.buff,
                                      lowoha_param.quant_params.wei_zp.dt);
    }

    // Compute or retrieve cached zero-point compensation
    if (src_zp != 0 || wei_zp != 0) {
      zp_comp_acc = cache_or_compute_zp_compensation(
                      cache_key, M, N, K, A, B,
                      src_zp, wei_zp,
                      transA == 't', transB == 't',
                      lda, ldb,
                      dtypes.src,
                      is_weights_const,
                      zp_comp_ndim);

      if (zp_comp_acc) {
        bool is_cacheable = (wei_zp == 0 && is_weights_const &&
                             matmul_config.get_zp_comp_cache());
        apilog_info("INT8 ZP compensation: src_zp=", src_zp, ", wei_zp=", wei_zp,
                    ", ndim=", zp_comp_ndim, ", cached=", (is_cacheable ? "yes" : "no"));
      }
    }
  }

  //TODO: remove the check for is_w4a8
  dlp_metadata_t *aocl_po = create_dlp_post_op(
                              is_w4a8 ? w4a8_lowoha_param : lowoha_param,
                              bias,
                              is_w4a8 ? dtypes_for_postop : dtypes,
                              N, K, M, zp_comp_acc, zp_comp_ndim, kernel, B);

  if (dtypes.src == data_type_t::f32 && dtypes.wei == data_type_t::f32 &&
      dtypes.dst == data_type_t::f32) {
    aocl_gemm_f32f32f32of32(layout,transA,transB,M,N,K,alpha,
                            static_cast<const float *>(A),lda,mem_format_a,
                            is_weight_blocked ? (float *)reordered_mem : static_cast<const float *>(B),
                            ldb, mem_format_b, beta,static_cast<float *>(C),ldc,aocl_po);
  }
  // WOQ s4 blocked path (not W4A8).
  else if (dtypes.wei == data_type_t::s4 &&
           kernel == zendnnl::ops::matmul_algo_t::aocl_dlp_blocked &&
           !is_w4a8) {
    if (dtypes.dst == data_type_t::bf16) {
      aocl_gemm_bf16s4f32obf16(layout,transA,transB,M,N,K,alpha,
                               static_cast<const int16_t *>(A),lda,mem_format_a,
                               is_weight_blocked ? (int8_t *)reordered_mem : static_cast<const int8_t *>(B),
                               ldb, mem_format_b, beta,static_cast<int16_t *>(C),ldc,aocl_po);
    }
    else if (dtypes.dst == data_type_t::f32) {
      aocl_gemm_bf16s4f32of32(layout,transA,transB,M,N,K,alpha,
                              static_cast<const int16_t *>(A),lda,mem_format_a,
                              is_weight_blocked ? (int8_t *)reordered_mem : static_cast<const int8_t *>(B),
                              ldb,mem_format_b, beta,static_cast<float *>(C),ldc,aocl_po);
    }
    else {
      log_error("Unsupported data type for matmul");
    }
  }
  // U4 blocked AOCL-DLP kernel path (skip this path for non-blocked kernels)
  else if (dtypes.wei == data_type_t::u4 &&
           kernel == zendnnl::ops::matmul_algo_t::aocl_dlp_blocked) {
    if (dtypes.dst == data_type_t::bf16) {
      aocl_gemm_bf16u4f32obf16(layout,transA,transB,M,N,K,alpha,
                               static_cast<const int16_t *>(A),lda,mem_format_a,
                               is_weight_blocked ? (uint8_t *)reordered_mem : static_cast<const uint8_t *>(B),
                               ldb, mem_format_b, beta,static_cast<int16_t *>(C),ldc,aocl_po);
    }
    else if (dtypes.dst == data_type_t::f32) {
      aocl_gemm_bf16u4f32of32(layout,transA,transB,M,N,K,alpha,
                              static_cast<const int16_t *>(A),lda,mem_format_a,
                              is_weight_blocked ? (uint8_t *)reordered_mem : static_cast<const uint8_t *>(B),
                              ldb,mem_format_b, beta,static_cast<float *>(C),ldc,aocl_po);
    }
    else {
      log_error("Unsupported data type for matmul");
    }
  }
  // W4A8: reorder s4->s8, sym_quant GEMM (bf16 output only).
  else if (is_w4a8) {
    const size_t w4a8_src_scale_nelems = get_num_elements(
                                           w4a8_lowoha_param.quant_params.src_scale.dims);
    const int64_t src_grp = (w4a8_src_scale_nelems == static_cast<size_t>(M))
                            ? K
                            : K / (static_cast<int64_t>(w4a8_src_scale_nelems) / M);
    const Key_matmul w4a8_cache_key(
      transB == 't', K, N, ldb, B,
      static_cast<uint32_t>(matmul_algo_t::aocl_dlp_blocked),
      std::hash<int64_t> {}(src_grp));

    void *w4a8_reordered_mem = nullptr;
    const int32_t w4a8_weight_cache_type = (weight_cache_type == 2) ? 1 :
                                           weight_cache_type;
    const int w4a8_sym_group_size =
      (w4a8_src_scale_nelems == static_cast<size_t>(M))
      ? K
      : K / (static_cast<int>(w4a8_src_scale_nelems) / M);
    w4a8ReorderAndCacheWeightsAocl(w4a8_cache_key,
                                   static_cast<const int8_t *>(B),
                                   w4a8_reordered_mem, K, N, ldb,
                                   is_weights_const, 'r', transB,
                                   dtypes.wei, dtypes.src,
                                   w4a8_weight_cache_type,
                                   w4a8_sym_group_size);
    if (w4a8_reordered_mem == nullptr) {
      apilog_error("[AOCL.run_dlp W4A8] weight reorder failed");
      cleanup_dlp_post_op(aocl_po);
      return;
    }

    if (dtypes.dst != data_type_t::bf16) {
      log_error("Unsupported output data type for W4A8; expected bf16");
    }
    else {
      //W4A8 matmul call with s8 weights and bf16 output
      aocl_gemm_s8s8s32obf16_sym_quant(layout, transA, transB, M, N, K, alpha,
                                       static_cast<const int8_t *>(A), lda,
                                       mem_format_a,
                                       static_cast<const int8_t *>(w4a8_reordered_mem),
                                       ldb, 'r', beta,
                                       static_cast<int16_t *>(C), ldc, aocl_po);
    }

    cleanup_dlp_post_op(aocl_po);
    if (!is_weights_const || w4a8_weight_cache_type != 1) {
      free(w4a8_reordered_mem);
    }
    return;
  }
  // BF16 kernels: bf16 source and weight or simulated WOQ S4 weights
  else if (dtypes.src == data_type_t::bf16 && (dtypes.wei == data_type_t::bf16 ||
           (dtypes.wei == data_type_t::s4 || dtypes.wei == data_type_t::u4))) {
    if (dtypes.dst == data_type_t::bf16) {
      aocl_gemm_bf16bf16f32obf16(layout,transA,transB,M,N,K,alpha,
                                 static_cast<const int16_t *>(A),lda,mem_format_a,
                                 is_weight_blocked ? (int16_t *)reordered_mem : static_cast<const int16_t *>(B),
                                 ldb, mem_format_b, beta,static_cast<int16_t *>(C),ldc,aocl_po);
    }
    else if (dtypes.dst == data_type_t::f32) {
      aocl_gemm_bf16bf16f32of32(layout,transA,transB,M,N,K,alpha,
                                static_cast<const int16_t *>(A),lda,mem_format_a,
                                is_weight_blocked ? (int16_t *)reordered_mem : static_cast<const int16_t *>(B),
                                ldb,mem_format_b, beta,static_cast<float *>(C),ldc,aocl_po);
    }
  }
  else if (dtypes.src == data_type_t::bf16 && dtypes.wei == data_type_t::s8) {
    switch (dtypes.dst) {
    case data_type_t::bf16:
      aocl_gemm_bf16s8s32obf16(layout,transA,transB,M,N,K,alpha,
                               static_cast<const int16_t *>(A),lda,mem_format_a,
                               is_weight_blocked ? (int8_t *)reordered_mem : static_cast<const int8_t *>(B),
                               ldb, mem_format_b, beta,static_cast<int16_t *>(C),ldc,aocl_po);
      break;
    case data_type_t::s8:
      aocl_gemm_bf16s8s32os8(layout,transA,transB,M,N,K,alpha,
                             static_cast<const int16_t *>(A),lda,mem_format_a,
                             is_weight_blocked ? (int8_t *)reordered_mem : static_cast<const int8_t *>(B),
                             ldb, mem_format_b, beta,static_cast<int8_t *>(C),ldc,aocl_po);
      break;
    case data_type_t::u8:
      aocl_gemm_bf16s8s32ou8(layout,transA,transB,M,N,K,alpha,
                             static_cast<const int16_t *>(A),lda,mem_format_a,
                             is_weight_blocked ? (int8_t *)reordered_mem : static_cast<const int8_t *>(B),
                             ldb, mem_format_b, beta,static_cast<uint8_t *>(C),ldc,aocl_po);
      break;
    case data_type_t::f32:
      aocl_gemm_bf16s8s32of32(layout,transA,transB,M,N,K,alpha,
                              static_cast<const int16_t *>(A),lda,mem_format_a,
                              is_weight_blocked ? (int8_t *)reordered_mem : static_cast<const int8_t *>(B),
                              ldb, mem_format_b, beta,static_cast<float *>(C),ldc,aocl_po);
      break;
    default:
      log_error("Unsupported output data type for bf16 source and s8 weight");
      break;
    }
  }
  else if (dtypes.src == data_type_t::f32 && dtypes.wei == data_type_t::s8) {
    switch (dtypes.dst) {
    case data_type_t::f32:
      aocl_gemm_f32s8s32of32(layout,transA,transB,M,N,K,alpha,
                             static_cast<const float *>(A),lda,mem_format_a,
                             is_weight_blocked ? (int8_t *)reordered_mem : static_cast<const int8_t *>(B),
                             ldb, mem_format_b, beta,static_cast<float *>(C),ldc,aocl_po);
      break;
    case data_type_t::s8:
      aocl_gemm_f32s8s32os8(layout,transA,transB,M,N,K,alpha,
                            static_cast<const float *>(A),lda,mem_format_a,
                            is_weight_blocked ? (int8_t *)reordered_mem : static_cast<const int8_t *>(B),
                            ldb, mem_format_b, beta,static_cast<int8_t *>(C),ldc,aocl_po);
      break;
    case data_type_t::u8:
      aocl_gemm_f32s8s32ou8(layout,transA,transB,M,N,K,alpha,
                            static_cast<const float *>(A),lda,mem_format_a,
                            is_weight_blocked ? (int8_t *)reordered_mem : static_cast<const int8_t *>(B),
                            ldb, mem_format_b, beta,static_cast<uint8_t *>(C),ldc,aocl_po);
      break;
    case data_type_t::bf16:
      aocl_gemm_f32s8s32obf16(layout,transA,transB,M,N,K,alpha,
                              static_cast<const float *>(A),lda,mem_format_a,
                              is_weight_blocked ? (int8_t *)reordered_mem : static_cast<const int8_t *>(B),
                              ldb, mem_format_b, beta,static_cast<int16_t *>(C),ldc,aocl_po);
      break;
    default:
      log_error("Unsupported output data type for f32 source and s8 weight");
      break;
    }
  }
  // INT8 kernels: u8 source
  else if (dtypes.src == data_type_t::u8 && dtypes.wei == data_type_t::s8) {
    const int8_t *weight_ptr = is_weight_blocked ? static_cast<int8_t *>
                               (reordered_mem) : static_cast<const int8_t *>(B);
    switch (dtypes.dst) {
    case data_type_t::u8:
      aocl_gemm_u8s8s32ou8(layout, transA, transB, M, N, K, alpha,
                           static_cast<const uint8_t *>(A), lda, mem_format_a,
                           weight_ptr, ldb, mem_format_b, beta,
                           static_cast<uint8_t *>(C), ldc, aocl_po);
      break;
    case data_type_t::s8:
      aocl_gemm_u8s8s32os8(layout, transA, transB, M, N, K, alpha,
                           static_cast<const uint8_t *>(A), lda, mem_format_a,
                           weight_ptr, ldb, mem_format_b, beta,
                           static_cast<int8_t *>(C), ldc, aocl_po);
      break;
    case data_type_t::s32:
      aocl_gemm_u8s8s32os32(layout, transA, transB, M, N, K, alpha,
                            static_cast<const uint8_t *>(A), lda, mem_format_a,
                            weight_ptr, ldb, mem_format_b, beta,
                            static_cast<int32_t *>(C), ldc, aocl_po);
      break;
    case data_type_t::f32:
      aocl_gemm_u8s8s32of32(layout, transA, transB, M, N, K, alpha,
                            static_cast<const uint8_t *>(A), lda, mem_format_a,
                            weight_ptr, ldb, mem_format_b, beta,
                            static_cast<float *>(C), ldc, aocl_po);
      break;
    case data_type_t::bf16:
      aocl_gemm_u8s8s32obf16(layout, transA, transB, M, N, K, alpha,
                             static_cast<const uint8_t *>(A), lda, mem_format_a,
                             weight_ptr, ldb, mem_format_b, beta,
                             static_cast<int16_t *>(C), ldc, aocl_po);
      break;
    case data_type_t::f16:
      aocl_gemm_u8s8s32of16(layout, transA, transB, M, N, K, alpha,
                            static_cast<const uint8_t *>(A), lda, mem_format_a,
                            weight_ptr, ldb, mem_format_b, beta,
                            static_cast<uint16_t *>(C), ldc, aocl_po);
      break;
    default:
      log_error("Unsupported output data type for u8 source");
      break;
    }
  }
  // INT8 kernels: s8 source
  else if (dtypes.src == data_type_t::s8 && dtypes.wei == data_type_t::s8) {
    const int8_t *weight_ptr = is_weight_blocked ? static_cast<int8_t *>
                               (reordered_mem) : static_cast<const int8_t *>(B);
    if (is_s8_sym_quant_scales) {
      switch (dtypes.dst) {
      case data_type_t::f32:
        aocl_gemm_s8s8s32of32_sym_quant(layout, transA, transB, M, N, K, alpha,
                                        static_cast<const int8_t *>(A), lda, mem_format_a,
                                        weight_ptr, ldb, mem_format_b, beta,
                                        static_cast<float *>(C), ldc, aocl_po);
        break;
      case data_type_t::bf16:
        aocl_gemm_s8s8s32obf16_sym_quant(layout, transA, transB, M, N, K, alpha,
                                         static_cast<const int8_t *>(A), lda, mem_format_a,
                                         weight_ptr, ldb, mem_format_b, beta,
                                         static_cast<int16_t *>(C), ldc, aocl_po);
        break;
      default:
        log_error("Unsupported output data type for sym_quant s8 source");
        break;
      }
    }
    else {
      switch (dtypes.dst) {
      case data_type_t::u8:
        aocl_gemm_s8s8s32ou8(layout, transA, transB, M, N, K, alpha,
                             static_cast<const int8_t *>(A), lda, mem_format_a,
                             weight_ptr, ldb, mem_format_b, beta,
                             static_cast<uint8_t *>(C), ldc, aocl_po);
        break;
      case data_type_t::s8:
        aocl_gemm_s8s8s32os8(layout, transA, transB, M, N, K, alpha,
                             static_cast<const int8_t *>(A), lda, mem_format_a,
                             weight_ptr, ldb, mem_format_b, beta,
                             static_cast<int8_t *>(C), ldc, aocl_po);
        break;
      case data_type_t::s32:
        aocl_gemm_s8s8s32os32(layout, transA, transB, M, N, K, alpha,
                              static_cast<const int8_t *>(A), lda, mem_format_a,
                              weight_ptr, ldb, mem_format_b, beta,
                              static_cast<int32_t *>(C), ldc, aocl_po);
        break;
      case data_type_t::f32:
        aocl_gemm_s8s8s32of32(layout, transA, transB, M, N, K, alpha,
                              static_cast<const int8_t *>(A), lda, mem_format_a,
                              weight_ptr, ldb, mem_format_b, beta,
                              static_cast<float *>(C), ldc, aocl_po);
        break;
      case data_type_t::bf16:
        aocl_gemm_s8s8s32obf16(layout, transA, transB, M, N, K, alpha,
                               static_cast<const int8_t *>(A), lda, mem_format_a,
                               weight_ptr, ldb, mem_format_b, beta,
                               static_cast<int16_t *>(C), ldc, aocl_po);
        break;
      case data_type_t::f16:
        aocl_gemm_s8s8s32of16(layout, transA, transB, M, N, K, alpha,
                              static_cast<const int8_t *>(A), lda, mem_format_a,
                              weight_ptr, ldb, mem_format_b, beta,
                              static_cast<uint16_t *>(C), ldc, aocl_po);
        break;
      default:
        log_error("Unsupported output data type for s8 source");
        break;
      }
    }
  }
  else if (dtypes.src == data_type_t::f16 && dtypes.wei == data_type_t::f16) {
    const uint16_t alpha_f16 = common::float16_t::f32_to_f16_val(alpha);
    const uint16_t beta_f16  = common::float16_t::f32_to_f16_val(beta);
    switch (dtypes.dst) {
    case data_type_t::f16:
      aocl_gemm_f16f16f16of16(layout, transA, transB, M, N, K, alpha_f16,
                              static_cast<const uint16_t *>(A), lda, mem_format_a,
                              is_weight_blocked ? (uint16_t *)reordered_mem : static_cast<const uint16_t *>
                              (B), ldb, mem_format_b, beta_f16, static_cast<uint16_t *>(C), ldc,
                              aocl_po);
      break;
    case data_type_t::f32:
      aocl_gemm_f16f16f16of32(layout, transA, transB, M, N, K, alpha_f16,
                              static_cast<const uint16_t *>(A), lda, mem_format_a,
                              is_weight_blocked ? (uint16_t *)reordered_mem : static_cast<const uint16_t *>
                              (B), ldb, mem_format_b, beta_f16, static_cast<float *>(C), ldc,
                              aocl_po);
      break;
    default:
      log_error("Unsupported output data type for f16 source");
      break;
    }
  }
  else {
    apilog_info("Data type not supported");
  }
  // Free reordered buffer for AOCL blocked non-cached
  bool weight_cache_disabled = (weight_cache_type == 0 &&
                                reordered_mem != nullptr &&
                                lowoha_param.mem_format_b != 'r'
                                && kernel==zendnnl::ops::matmul_algo_t::aocl_dlp_blocked);
  if (weight_cache_disabled || simulated_woq_free_buff) {
    free(reordered_mem);
    reordered_mem = nullptr;
  }
  // Free zero-point compensation buffer (only if not cached)
  // 1D compensation (wei_zp == 0) is cached, 2D compensation is always freed
  bool zp_cache_enabled = matmul_config.get_zp_comp_cache();
  if (zp_comp_acc && (!zp_cache_enabled || wei_zp != 0)) {
    std::free(zp_comp_acc);
  }
  // Per-call teardown for the metadata returned by create_dlp_post_op().
  // No-op for cached holders (the common case); frees the per-call
  // holder + its heap-owned inv_scales[] for the BF16/INT8 per-token-sym
  // path. Safe with nullptr.
  cleanup_dlp_post_op(aocl_po);
}

void matmul_batch_gemm_wrapper(char layout, char transA, char transB, int M,
                               int N, int K, float alpha, const void *A, int lda, const void *B, int ldb,
                               float beta, void *C, int ldc, matmul_data_types &dtypes, int batch_count,
                               int Batch_A, int Batch_B, char mem_format_a, char mem_format_b,
                               size_t src_stride, size_t weight_stride,
                               size_t dst_stride, const matmul_params &lowoha_param, const void *bias,
                               int num_threads) {


  dlp_metadata_t *metadata_array = create_dlp_post_op(lowoha_param, bias, dtypes,
                                   N, K, /*M=*/0, /*zp_comp_acc=*/nullptr,
                                   /*zp_comp_ndim=*/0,
                                   zendnnl::ops::matmul_algo_t::aocl_dlp, B);
  md_t m_ = M;
  md_t n_ = N;
  md_t k_ = K;
  md_t lda_ = lda;
  md_t ldb_ = ldb;
  md_t ldc_ = ldc;
  md_t group_size = batch_count;

  // Helper lambda for batch index calculation (handles broadcasting)
  auto get_batch_idx = [](int b, int batch_size) {
    return (batch_size == 1) ? 0 : (b % batch_size);
  };

  // Prepare pointer arrays for matrices
  std::vector<const void *> a_ptrs(batch_count);
  std::vector<const void *> b_ptrs(batch_count);
  std::vector<void *> c_ptrs(batch_count);

  // Set up pointers for each batch (with broadcasting support)
  #pragma omp parallel for num_threads(num_threads)
  for (int b = 0; b < batch_count; ++b) {
    a_ptrs[b] = static_cast<const uint8_t *>(A) + get_batch_idx(b,
                Batch_A) * src_stride;
    b_ptrs[b] = static_cast<const uint8_t *>(B) + get_batch_idx(b,
                Batch_B) * weight_stride;
    c_ptrs[b] = static_cast<uint8_t *>(C) + b * dst_stride;
  }

  // Call appropriate batch GEMM based on data types
  if (dtypes.src == data_type_t::f32 && dtypes.dst == data_type_t::f32) {
    apilog_info("executing aocl_batch_gemm_f32f32f32of32");
    aocl_batch_gemm_f32f32f32of32(
      &layout, &transA, &transB,
      &m_, &n_, &k_,
      &alpha,
      reinterpret_cast<const float **>(a_ptrs.data()), &lda_,
      reinterpret_cast<const float **>(b_ptrs.data()), &ldb_,
      &beta,
      reinterpret_cast<float **>(c_ptrs.data()), &ldc_,
      1, // single group
      &group_size,
      &mem_format_a, &mem_format_b,
      &metadata_array);
  }
  else if (dtypes.src == data_type_t::bf16 && dtypes.dst == data_type_t::f32) {
    apilog_info("executing aocl_batch_gemm_bf16bf16f32of32");
    aocl_batch_gemm_bf16bf16f32of32(
      &layout, &transA, &transB,
      &m_, &n_, &k_,
      &alpha,
      reinterpret_cast<const bfloat16 **>(a_ptrs.data()), &lda_,
      reinterpret_cast<const bfloat16 **>(b_ptrs.data()), &ldb_,
      &beta,
      reinterpret_cast<float **>(c_ptrs.data()), &ldc_,
      1, // single group
      &group_size,
      &mem_format_a, &mem_format_b,
      &metadata_array);
  }
  else if (dtypes.src == data_type_t::bf16 && dtypes.dst == data_type_t::bf16) {
    apilog_info("executing aocl_batch_gemm_bf16bf16f32obf16");
    aocl_batch_gemm_bf16bf16f32obf16(
      &layout, &transA, &transB,
      &m_, &n_, &k_,
      &alpha,
      reinterpret_cast<const bfloat16 **>(a_ptrs.data()), &lda_,
      reinterpret_cast<const bfloat16 **>(b_ptrs.data()), &ldb_,
      &beta,
      reinterpret_cast<bfloat16 **>(c_ptrs.data()), &ldc_,
      1, // single group
      &group_size,
      &mem_format_a, &mem_format_b,
      &metadata_array);
  }
  else if (dtypes.src == data_type_t::f16 && dtypes.dst == data_type_t::f16) {
    const uint16_t alpha_f16 = common::float16_t::f32_to_f16_val(alpha);
    const uint16_t beta_f16  = common::float16_t::f32_to_f16_val(beta);
    apilog_info("executing aocl_batch_gemm_f16f16f16of16");
    aocl_batch_gemm_f16f16f16of16(
      &layout, &transA, &transB,
      &m_, &n_, &k_,
      &alpha_f16,
      reinterpret_cast<const float16 **>(a_ptrs.data()), &lda_,
      reinterpret_cast<const float16 **>(b_ptrs.data()), &ldb_,
      &beta_f16,
      reinterpret_cast<float16 **>(c_ptrs.data()), &ldc_,
      1, // single group
      &group_size,
      &mem_format_a, &mem_format_b,
      &metadata_array);
  }
  else if (dtypes.src == data_type_t::f16 && dtypes.dst == data_type_t::f32) {
    const uint16_t alpha_f16 = common::float16_t::f32_to_f16_val(alpha);
    const uint16_t beta_f16  = common::float16_t::f32_to_f16_val(beta);
    apilog_info("executing aocl_batch_gemm_f16f16f16of32");
    aocl_batch_gemm_f16f16f16of32(
      &layout, &transA, &transB,
      &m_, &n_, &k_,
      &alpha_f16,
      reinterpret_cast<const float16 **>(a_ptrs.data()), &lda_,
      reinterpret_cast<const float16 **>(b_ptrs.data()), &ldb_,
      &beta_f16,
      reinterpret_cast<float **>(c_ptrs.data()), &ldc_,
      1, // single group
      &group_size,
      &mem_format_a, &mem_format_b,
      &metadata_array);
  }
  else {
    log_error("Unsupported data type combination for batch GEMM");
  }
  // Per-call teardown for the metadata returned by create_dlp_post_op().
  // No-op for cached holders (the common case); frees the per-call
  // holder + its heap-owned inv_scales[] for the BF16/INT8 per-token-sym
  // path. Safe with nullptr.
  cleanup_dlp_post_op(metadata_array);
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl