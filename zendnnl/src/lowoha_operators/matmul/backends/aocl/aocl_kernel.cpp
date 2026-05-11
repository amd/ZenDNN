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
#include "lowoha_operators/matmul/backends/aocl/aocl_postop.hpp"
#include "lowoha_operators/matmul/lru_cache/lowoha_cache.hpp"
#include <cstdlib>
#include <cstring>
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

template <typename T>
void clear_aocl_typed_weight_cache_under_lock() {
  std::lock_guard<std::mutex> lock(get_aocl_weight_cache_mutex<T>());
  get_aocl_weight_cache<T>().clear();
}
void clear_aocl_woq_weight_cache_under_lock() {
  std::lock_guard<std::mutex> lock(get_aocl_woq_weight_cache_mutex());
  get_aocl_woq_weight_cache().clear();
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

  if (weight_cache_type == 0) {
    apilog_info("AOCL reorder weights (WEIGHT_CACHE_DISABLE)");
    size_t b_reorder_buf_siz_req = get_reorder_buf_size(order, trans, 'B',
                                   k, n,nullptr);
    size_t alignment      = 64;
    size_t reorder_size   = (b_reorder_buf_siz_req + alignment - 1) & ~
                            (alignment - 1);
    reorder_weights       = (T *)aligned_alloc(alignment, reorder_size);
    reorder_func(order, trans, 'B', (T *)weights, (T *)reorder_weights, k, n, ldb, nullptr);
  }
  // Out-of-place reordering
  else if (weight_cache_type == 1) {
    std::lock_guard<std::mutex> lock(weight_cache_mutex);
    auto found_obj = matmul_weight_cache.find_key(key);
    if (!found_obj) {
      apilog_info("AOCL reorder weights WEIGHT_CACHE_OUT_OF_PLACE");
      size_t b_reorder_buf_siz_req = get_reorder_buf_size(order, trans, 'B', k, n,nullptr);
      size_t alignment      = 64;
      size_t reorder_size   = (b_reorder_buf_siz_req + alignment - 1) & ~
                              (alignment - 1);
      reorder_weights = (T *)aligned_alloc(alignment, reorder_size);
      reorder_func(order, trans, 'B', (T *)weights, (T *)reorder_weights, k, n, ldb, nullptr);
      // Create new entry
      matmul_weight_cache.add(key, reorder_weights);
    }
    else {
      apilog_info("Read AOCL cached weights WEIGHT_CACHE_OUT_OF_PLACE");
      reorder_weights = matmul_weight_cache.get(key);
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
    apilog_info("AOCL sym_quant reorder weights (WEIGHT_CACHE_DISABLE)");
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
    auto found_obj = matmul_weight_cache.find_key(key);
    if (!found_obj) {
      apilog_info("AOCL sym_quant reorder weights WEIGHT_CACHE_OUT_OF_PLACE");
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
      apilog_info("Read AOCL sym_quant cached weights WEIGHT_CACHE_OUT_OF_PLACE");
      reorder_weights = matmul_weight_cache.get(key);
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
  // Weight caching inplace support cannot be added since buffer size is
  // always expanded.
  lru_cache_t<Key_matmul, void *> &matmul_weight_cache_woq =
    get_aocl_woq_weight_cache();
  std::mutex &woq_cache_mutex = get_aocl_woq_weight_cache_mutex();

  bool is_transposed = (trans == 't');

  // Use lock guard to protect the entire check-compute-cache operation
  std::lock_guard<std::mutex> lock(woq_cache_mutex);
  auto found_obj = matmul_weight_cache_woq.find_key(key);

  if (!is_weights_const || !found_obj) {
    apilog_info("WOQ Simulated AOCL reorder weights (weight_cache_type=",
                weight_cache_type, ")");
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
    apilog_info("Read WOQ Simulated AOCL cached weights");
    reorder_weights = matmul_weight_cache_woq.get(key);
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
  int32_t weight_cache_type = matmul_config.get_weight_cache();

  size_t run_src_scale_nelems = get_num_elements(
                                  lowoha_param.quant_params.src_scale.dims);
  bool is_sym_quant = dtypes.wei == data_type_t::s8 &&
                      dtypes.src == data_type_t::s8 &&
                      !lowoha_param.quant_params.src_zp.buff &&
                      run_src_scale_nelems > 1 &&
                      (dtypes.dst == data_type_t::f32 || dtypes.dst == data_type_t::bf16);

  size_t cache_extra_hash = 0;
  if (is_sym_quant) {
    int64_t src_grp = (run_src_scale_nelems == static_cast<size_t>(M))
                      ? K : K / (static_cast<int64_t>(run_src_scale_nelems) / M);
    cache_extra_hash = std::hash<int64_t> {}(src_grp);
  }
  Key_matmul cache_key(transB == 't', K, N, ldb, B,
                       static_cast<uint32_t>(matmul_algo_t::aocl_dlp_blocked),
                       cache_extra_hash);

  // AOCL blocked kernel reordering for 2D MatMul
  if (kernel==zendnnl::ops::matmul_algo_t::aocl_dlp_blocked &&
      is_weights_const) {
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
      if (is_sym_quant) {
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
           (dtypes.wei == data_type_t::s4 || dtypes.wei == data_type_t::u4)) {
    //call woq reorder and cache function
    woqReorderAndCacheWeightsAocl(cache_key, static_cast<const int8_t *>(B),
                                  reordered_mem, K,
                                  N, ldb, is_weights_const, 'r', transB, mem_format_b,
                                  lowoha_param.quant_params, dtypes.wei, weight_cache_type);
    is_weight_blocked = true;
    mem_format_b = 'r';
    simulated_woq_free_buff = !is_weights_const || weight_cache_type != 1;
  }

  // Compute zero-point compensation for INT8 (with caching for 1D case)
  int32_t *zp_comp_acc = nullptr;
  int zp_comp_ndim = 0;
  int32_t src_zp = 0;
  int32_t wei_zp = 0;
  bool is_int8 = dtypes.wei == data_type_t::s8;
  if (is_int8) {
    // Extract zero-point values
    if (lowoha_param.quant_params.src_zp.buff && dtypes.src != data_type_t::bf16 &&
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

  // Create aocl_post_op structure for bias, post-ops, and WOQ pre-ops
  dlp_metadata_t *aocl_po = create_dlp_post_op(lowoha_param, bias, dtypes, N, K,
                            M, zp_comp_acc, zp_comp_ndim, kernel);

  if (dtypes.src == data_type_t::f32 && dtypes.wei == data_type_t::f32 &&
      dtypes.dst == data_type_t::f32) {
    aocl_gemm_f32f32f32of32(layout,transA,transB,M,N,K,alpha,
                            static_cast<const float *>(A),lda,mem_format_a,
                            is_weight_blocked ? (float *)reordered_mem : static_cast<const float *>(B),
                            ldb, mem_format_b, beta,static_cast<float *>(C),ldc,aocl_po);
  }
  // S4 blocked AOCL-DLP kernel path (skip this path for non-blocked kernels)
  else if (dtypes.wei == data_type_t::s4 &&
           kernel == zendnnl::ops::matmul_algo_t::aocl_dlp_blocked) {
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
    default:
      log_error("Unsupported output data type for u8 source");
      break;
    }
  }
  // INT8 kernels: s8 source
  else if (dtypes.src == data_type_t::s8 && dtypes.wei == data_type_t::s8) {
    const int8_t *weight_ptr = is_weight_blocked ? static_cast<int8_t *>
                               (reordered_mem) : static_cast<const int8_t *>(B);
    if (is_sym_quant) {
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
      default:
        log_error("Unsupported output data type for s8 source");
        break;
      }
    }
  }
  else if (dtypes.src == data_type_t::f16 && dtypes.wei == data_type_t::f16) {
    switch (dtypes.dst) {
    case data_type_t::f16: {
      const uint16_t alpha_f16 = common::float16_t::f32_to_f16_val(alpha);
      const uint16_t beta_f16  = common::float16_t::f32_to_f16_val(beta);
      aocl_gemm_f16f16f16of16(layout, transA, transB, M, N, K, alpha_f16,
                              static_cast<const uint16_t *>(A), lda, mem_format_a,
                              is_weight_blocked ? (uint16_t *)reordered_mem : static_cast<const uint16_t *>
                              (B), ldb, mem_format_b, beta_f16, static_cast<uint16_t *>(C), ldc,
                              nullptr);  // post-ops are not supported for f16
      break;
    }
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

  // Clean up aocl_post_op structure
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
                                   N, K);
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
  else {
    log_error("Unsupported data type combination for batch GEMM");
  }

  // Clean up aocl_post_op structure
  cleanup_dlp_post_op(metadata_array);
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl