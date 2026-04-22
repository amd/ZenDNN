/********************************************************************************
# * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# *******************************************************************************/

#include "lowoha_sdpa_flash_cpu.hpp"
#include "sdpa_simd.hpp"
#include "lowoha_operators/matmul/lowoha_matmul.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <optional>

#include <omp.h>

#include "common/logging.hpp"
#include "common/zendnnl_global.hpp"
#include "lowoha_operators/common/omp_thread_control.hpp"

// Suppress GCC's informational ABI-change note for 64-byte vector types
// (__m512).  The note warns that the calling convention for these types
// differs from GCC 4.6 — irrelevant for any modern toolchain.  The
// template helpers below pass __m512 to/from SimdOps<avx512_tag> methods
// that carry __attribute__((target("avx512f,..."))), which is correct.
#if defined(__GNUC__) && !defined(__clang__)
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wpsabi"
#endif

namespace zendnnl {
namespace lowoha {
namespace sdpa {

static_assert(simd::SimdOps<simd::avx512_tag>::kFloatLanes >= 1 &&
              simd::SimdOps<simd::scalar_tag>::kFloatLanes >= 1,
              "sdpa_simd.hpp must provide at least 1 float lane per tag");

struct scratch_buffer {
  void  *ptr = nullptr;
  size_t cap = 0;
  ~scratch_buffer() {
    free(ptr);
  }
  scratch_buffer() = default;
  scratch_buffer(const scratch_buffer &) = delete;
  scratch_buffer &operator=(const scratch_buffer &) = delete;
};

static thread_local scratch_buffer g_flash_scratch;

namespace {

inline void *flash_scratch_acquire(size_t bytes) {
  if (g_flash_scratch.cap < bytes) {
    free(g_flash_scratch.ptr);
    g_flash_scratch.ptr = malloc(bytes);
    g_flash_scratch.cap = g_flash_scratch.ptr ? bytes : 0;
  }
  return g_flash_scratch.ptr;
}

#define SDPA_SA_CHECK(cond, msg)                                               \
  do {                                                                         \
    if (!(cond))                                                               \
      throw std::invalid_argument(msg);                                        \
  } while (0)

template <typename T> struct is_reduced_fp : std::false_type {};
template <> struct is_reduced_fp<bf16_elem> : std::true_type {};
template <typename T>
inline constexpr bool is_reduced_fp_v = is_reduced_fp<T>::value;

using accum_t = float;

inline float bf16_bits_to_float(uint16_t bits) {
  uint32_t u = static_cast<uint32_t>(bits) << 16;
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

inline uint16_t float_to_bf16_bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  uint32_t rounding_bias = ((u >> 16) & 1) + 0x7FFF;
  return static_cast<uint16_t>((u + rounding_bias) >> 16);
}

template <typename T2>
inline float mask_elem_at(T2 const *b, int i) {
  if constexpr(std::is_same_v<T2, float>) {
    return b[i];
  }
  else {
    return bf16_bits_to_float(b[i].x);
  }
}

template <typename scalar_t> inline float scalar_to_float(scalar_t v) {
  if constexpr(std::is_same_v<scalar_t, float>) {
    return v;
  }
  else {
    return bf16_bits_to_float(v.x);
  }
}

template <typename scalar_t> inline scalar_t float_to_scalar(float f) {
  if constexpr(std::is_same_v<scalar_t, float>) {
    return f;
  }
  else {
    bf16_elem r;
    r.x = float_to_bf16_bits(f);
    return r;
  }
}

inline void sdpa_data_index_init(int64_t linear_idx, int64_t &i, int64_t dim_i,
                                 int64_t &j, int64_t dim_j, int64_t &k,
                                 int64_t dim_k) {
  const int64_t slice = dim_j * dim_k;
  i = linear_idx / slice;
  int64_t rem = linear_idx % slice;
  j = rem / dim_k;
  k = rem % dim_k;
}

template <typename T>
inline void zendnn_gemm(int64_t m, int64_t n, int64_t k, float alpha,
                        const T *a, int64_t lda, const T *b, int64_t ldb,
                        float beta, float *c, int64_t ldc, bool TransA,
                        bool TransB) {
  constexpr bool is_input_float = std::is_same_v<T, float>;
  zendnnl::lowoha::matmul::matmul_params params;
  zendnnl::lowoha::matmul::matmul_data_types matmul_dtype;
  matmul_dtype.bias = zendnnl::common::data_type_t::none;
  matmul_dtype.compute = zendnnl::common::data_type_t::none;
  matmul_dtype.src =
    is_input_float ? zendnnl::common::data_type_t::f32 :
    zendnnl::common::data_type_t::bf16;
  matmul_dtype.wei =
    is_input_float ? zendnnl::common::data_type_t::f32 :
    zendnnl::common::data_type_t::bf16;
  matmul_dtype.dst = zendnnl::common::data_type_t::f32;
  params.dtypes = matmul_dtype;
  params.lowoha_algo = zendnnl::ops::matmul_algo_t::aocl_dlp;

  zendnnl::lowoha::matmul::matmul_batch_params_t batch_params;
  batch_params.Batch_A = 1;
  batch_params.Batch_B = 1;

  zendnnl::lowoha::matmul::matmul_direct(
    'r', TransA, TransB, m, n, k, alpha, a, lda, b, ldb, nullptr, beta, c,
    ldc, false, batch_params, params);
}

// ---------------------------------------------------------------------------
// SIMD-templated helper functions.
// All functions below are parameterised on SimdTag (avx512_tag | scalar_tag).
//
// #pragma GCC target enables AVX-512 so that __m512 locals inside the
// avx512_tag instantiation use the correct ABI (zmm registers, 64-byte
// alignment).  The scalar_tag instantiation never touches __m512 types
// (VecF32 = struct{float}), so the target is harmless there.
//
// #pragma GCC optimize("no-tree-vectorize") prevents the compiler from
// auto-vectorising scalar-path loops with AVX-512 instructions, which
// would SIGILL on machines without AVX-512 support.
// ---------------------------------------------------------------------------
#if defined(__GNUC__) && !defined(__clang__)
  #pragma GCC push_options
  #pragma GCC target("avx512f,avx512bw,avx512vl,fma")
  #pragma GCC optimize("no-tree-vectorize")
#endif

template <typename SimdTag, typename T1, typename T2>
inline void scale_attn_mask_fusion(T1 *a, T2 const *b, int size, T1 *out,
                                   T1 val) {
  using Ops = simd::SimdOps<SimdTag>;
  using VecF32 = typename Ops::VecF32;
  const int L = Ops::kFloatLanes;
  int i = 0;
  if constexpr(std::is_same_v<T1, float> &&std::is_same_v<T2, float>) {
    const VecF32 vs = Ops::vec_set1(val);
    for (; i + L <= size; i += L) {
      const VecF32 va = Ops::vec_loadu(a + i);
      const VecF32 vb = Ops::vec_loadu(reinterpret_cast<const float *>(b) + i);
      Ops::vec_storeu(out + i, Ops::vec_fmadd(va, vs, vb));
    }
  }
  else if constexpr(std::is_same_v<T1, float> &&
                    std::is_same_v<T2, bf16_elem>) {
    const VecF32 vs = Ops::vec_set1(val);
    const auto *bp = reinterpret_cast<const uint16_t *>(b);
    for (; i + L <= size; i += L) {
      const VecF32 va = Ops::vec_loadu(a + i);
      const VecF32 vb = Ops::vec_mask_bf16_loadu(bp + i);
      Ops::vec_storeu(out + i, Ops::vec_fmadd(va, vs, vb));
    }
  }
  for (; i < size; ++i) {
    const float bf = mask_elem_at(b, i);
    out[i] = a[i] * val + static_cast<T1>(bf);
  }
}

template <typename SimdTag>
inline void exp_reduce_sum_fusion(const accum_t *a, int size, accum_t *out,
                                  accum_t &val) {
  using Ops = simd::SimdOps<SimdTag>;
  using VecF32 = typename Ops::VecF32;
  const int L = Ops::kFloatLanes;
  const VecF32 vmb = Ops::vec_set1(val);
  VecF32 vsum = Ops::vec_set1(0.f);
  int i = 0;
  for (; i + L <= size; i += L) {
    const VecF32 x = Ops::vec_loadu(a + i);
    const VecF32 d = Ops::vec_sub(x, vmb);
    const VecF32 e = Ops::vec_exp_u20(d);
    vsum = Ops::vec_add(vsum, e);
    Ops::vec_storeu(out + i, e);
  }
  accum_t tmp_sum = Ops::vec_reduce_sum(vsum);
  for (; i < size; ++i) {
    const accum_t e = std::exp(a[i] - val);
    out[i] = e;
    tmp_sum += e;
  }
  val = tmp_sum;
}

template <typename SimdTag, typename T2>
inline void exp_reduce_sum_fusion_to(const accum_t *a, int size, T2 *out,
                                     accum_t &val) {
  using Ops = simd::SimdOps<SimdTag>;
  using VecF32 = typename Ops::VecF32;
  const int L = Ops::kFloatLanes;
  const VecF32 vmb = Ops::vec_set1(val);
  VecF32 vsum = Ops::vec_set1(0.f);
  int i = 0;
  for (; i + L <= size; i += L) {
    const VecF32 x = Ops::vec_loadu(a + i);
    const VecF32 d = Ops::vec_sub(x, vmb);
    const VecF32 e = Ops::vec_fexp_u20(d);
    vsum = Ops::vec_add(vsum, e);
    if constexpr(std::is_same_v<T2, bf16_elem>) {
      Ops::vec_bf16_storeu(reinterpret_cast<uint16_t *>(&out[i].x), e);
    }
    else {
      Ops::vec_storeu(reinterpret_cast<float *>(out + i), e);
    }
  }
  accum_t tmp_sum = Ops::vec_reduce_sum(vsum);
  for (; i < size; ++i) {
    const accum_t e = std::exp(a[i] - val);
    out[i] = float_to_scalar<T2>(e);
    tmp_sum += e;
  }
  val = tmp_sum;
}

template <typename SimdTag>
inline void mul_reduce_max_fusion(const accum_t *a, accum_t scale, int size,
                                  accum_t *out, accum_t &maxv) {
  using Ops = simd::SimdOps<SimdTag>;
  using VecF32 = typename Ops::VecF32;
  const int L = Ops::kFloatLanes;
  VecF32 vtmp_max = Ops::vec_set1(-std::numeric_limits<accum_t>::infinity());
  const VecF32 vs = Ops::vec_set1(scale);
  int i = 0;
  for (; i + L <= size; i += L) {
    const VecF32 x = Ops::vec_loadu(a + i);
    const VecF32 y = Ops::vec_mul(x, vs);
    vtmp_max = Ops::vec_max(vtmp_max, y);
    Ops::vec_storeu(out + i, y);
  }
  maxv = Ops::vec_reduce_max(vtmp_max);
  for (; i < size; ++i) {
    const accum_t t = a[i] * scale;
    out[i] = t;
    maxv = std::max(maxv, t);
  }
}

template <typename scalar_t>
inline void fill_stub(scalar_t *data, scalar_t val, int64_t size) {
  for (int64_t d = 0; d < size; ++d) {
    data[d] = val;
  }
}

template <typename SimdTag>
inline void fill_stub_f32(float *data, float val, int64_t size) {
  using Ops = simd::SimdOps<SimdTag>;
  using VecF32 = typename Ops::VecF32;
  const int L = Ops::kFloatLanes;
  const VecF32 vv = Ops::vec_set1(val);
  int64_t d = 0;
  for (; d + L <= size; d += L) {
    Ops::vec_storeu(data + d, vv);
  }
  for (; d < size; ++d) {
    data[d] = val;
  }
}

template <typename scalar_t>
inline scalar_t *conditional_data_ptr(scalar_t *ptr, scalar_t *ptr2) {
  SDPA_SA_CHECK(ptr2 == nullptr, "conditional_data_ptr: unexpected bf16 qk buf");
  return ptr;
}

template <typename scalar_t,
          typename std::enable_if_t<is_reduced_fp_v<scalar_t>, int> = 0>
inline scalar_t *conditional_data_ptr(float *ptr, scalar_t *ptr2) {
  return ptr2;
}

template <typename SimdTag>
inline accum_t row_max(const accum_t *row, int64_t len) {
  using Ops = simd::SimdOps<SimdTag>;
  using VecF32 = typename Ops::VecF32;
  const int L = Ops::kFloatLanes;
  VecF32 vtmp_max = Ops::vec_set1(-std::numeric_limits<accum_t>::infinity());
  int64_t c = 0;
  for (; c + L <= len; c += L) {
    const VecF32 x = Ops::vec_loadu(row + c);
    vtmp_max = Ops::vec_max(vtmp_max, x);
  }
  accum_t m = Ops::vec_reduce_max(vtmp_max);
  for (; c < len; ++c) {
    m = std::max(m, row[c]);
  }
  return m;
}

template <typename SimdTag>
inline void scale_dst_row(accum_t *row, int64_t len, accum_t exp_tmp) {
  using Ops = simd::SimdOps<SimdTag>;
  using VecF32 = typename Ops::VecF32;
  const int L = Ops::kFloatLanes;
  const VecF32 vs = Ops::vec_set1(exp_tmp);
  int64_t c = 0;
  for (; c + L <= len; c += L) {
    const VecF32 x = Ops::vec_loadu(row + c);
    Ops::vec_storeu(row + c, Ops::vec_mul(x, vs));
  }
  for (; c < len; ++c) {
    row[c] *= exp_tmp;
  }
}

template <typename SimdTag>
inline void vec_f32_scaled_bf16_store(bf16_elem *dst, const accum_t *src,
                                      int64_t len, accum_t scale) {
  using Ops = simd::SimdOps<SimdTag>;
  using VecF32 = typename Ops::VecF32;
  const int L = Ops::kFloatLanes;
  const VecF32 vs = Ops::vec_set1(scale);
  int64_t c = 0;
  for (; c + L <= len; c += L) {
    const VecF32 x = Ops::vec_loadu(src + c);
    Ops::vec_bf16_storeu(reinterpret_cast<uint16_t *>(&dst[c].x),
                         Ops::vec_mul(x, vs));
  }
  for (; c < len; ++c) {
    dst[c].x = float_to_bf16_bits(src[c] * scale);
  }
}

template <typename SimdTag, typename scalar_t>
inline void write_scaled_output_row(scalar_t *out_base, int64_t out_stride_d,
                                    const accum_t *dst_row, int64_t headSize,
                                    accum_t sum_reciprocal) {
  using Ops = simd::SimdOps<SimdTag>;
  using VecF32 = typename Ops::VecF32;
  const int L = Ops::kFloatLanes;
  const VecF32 vr = Ops::vec_set1(sum_reciprocal);
  if (out_stride_d == 1) {
    if constexpr(std::is_same_v<scalar_t, float>) {
      int64_t c = 0;
      for (; c + L <= headSize; c += L) {
        const VecF32 x = Ops::vec_loadu(dst_row + c);
        Ops::vec_storeu(out_base + c, Ops::vec_mul(x, vr));
      }
      for (; c < headSize; ++c) {
        out_base[c] = dst_row[c] * sum_reciprocal;
      }
      return;
    }
    else if constexpr(std::is_same_v<scalar_t, bf16_elem>) {
      vec_f32_scaled_bf16_store<SimdTag>(out_base, dst_row, headSize,
                                         sum_reciprocal);
      return;
    }
  }
  for (int64_t c = 0; c < headSize; ++c) {
    const accum_t v = dst_row[c] * sum_reciprocal;
    out_base[c * out_stride_d] = float_to_scalar<scalar_t>(v);
  }
}

struct TransposedBHSD {
  const void *base;
  int64_t size_b, size_m, size_h, size_d;
  int64_t stride_b, stride_m, stride_h, stride_d;
};

inline TransposedBHSD transpose_bh_sd(const sdpa_flash_cpu_tensor_view &v) {
  TransposedBHSD t;
  t.base = v.data;
  t.size_b = v.size_b;
  t.size_m = v.size_s;
  t.size_h = v.size_h;
  t.size_d = v.size_d;
  t.stride_b = v.stride_b;
  t.stride_m = v.stride_s;
  t.stride_h = v.stride_h;
  t.stride_d = v.stride_d;
  return t;
}

struct NormalizedMask {
  const void *data;
  int64_t m_stride_b;
  int64_t m_stride_h;
  int64_t m_stride_m;
};

inline NormalizedMask normalize_mask(const sdpa_flash_cpu_mask_view &mv,
                                     int64_t batchSize,
                                     int64_t num_head, int64_t qSize,
                                     int64_t kvSize) {
  int64_t v0 = 1, v1 = 1, v2, v3;
  int64_t t0 = 0, t1 = 0, t2, t3;

  if (mv.ndim == 2) {
    v2 = mv.sizes[0];
    v3 = mv.sizes[1];
    t2 = mv.strides[0];
    t3 = mv.strides[1];
  }
  else if (mv.ndim == 4) {
    const int64_t o0 = mv.sizes[0], o1 = mv.sizes[1], o2 = mv.sizes[2],
                  o3 = mv.sizes[3];
    v0 = (o0 == batchSize) ? batchSize : 1;
    v1 = (o1 == num_head) ? num_head : 1;
    v2 = o2;
    v3 = o3;
    t0 = mv.strides[0];
    t1 = mv.strides[1];
    t2 = mv.strides[2];
    t3 = mv.strides[3];
  }
  else {
    throw std::invalid_argument("normalize_mask: ndim must be 2 or 4");
  }

  auto expand_stride = [&](int64_t vs, int64_t ts, int64_t es) -> int64_t {
    if (es == vs) {
      return ts;
    }
    if (vs == 1) {
      return 0;
    }
    throw std::invalid_argument("normalize_mask: incompatible mask shape");
  };

  int64_t rs0 = expand_stride(v0, t0, batchSize);
  int64_t rs1 = expand_stride(v1, t1, num_head);
  int64_t rs2 = expand_stride(v2, t2, qSize);
  int64_t rs3 = expand_stride(v3, t3, kvSize);
  (void)rs3;
  SDPA_SA_CHECK(rs3 == 1,
                "normalize_mask: last dim stride must be 1 (contiguous KV)");

  NormalizedMask nm;
  nm.data = mv.data;
  nm.m_stride_b = (batchSize > 1) ? rs0 : 0;
  nm.m_stride_h = (num_head > 1) ? rs1 : 0;
  nm.m_stride_m = rs2;
  return nm;
}

inline float calculate_scale_value(std::optional<double> scale,
                                   int64_t head_dim) {
  if (scale.has_value()) {
    return static_cast<float>(scale.value());
  }
  return 1.0f / std::sqrt(static_cast<float>(head_dim));
}

// ---------------------------------------------------------------------------
// Main flash-attention kernel — templated on SimdTag for SIMD dispatch.
// ---------------------------------------------------------------------------

template <typename SimdTag, typename scalar_t, typename mask_t,
          int64_t q_split_size, int64_t kv_split_size>
void cpu_flash_attention_sa(
  const sdpa_flash_cpu_tensor_view &output,
  const sdpa_flash_cpu_tensor_view &query_bh,
  const sdpa_flash_cpu_tensor_view &key_bh,
  const sdpa_flash_cpu_tensor_view &value_bh, double dropout_p, bool is_causal,
  std::optional<sdpa_flash_cpu_mask_view> attn_mask, std::optional<double> scale,
  int num_threads_hint) {
  (void)dropout_p;
  SDPA_SA_CHECK(!dropout_p, "dropout must be 0");

  TransposedBHSD query = transpose_bh_sd(query_bh);
  TransposedBHSD key = transpose_bh_sd(key_bh);
  TransposedBHSD value = transpose_bh_sd(value_bh);
  TransposedBHSD out = transpose_bh_sd(output);

  constexpr bool is_reduced_type = is_reduced_fp_v<scalar_t>;
  accum_t scaling_factor =
    calculate_scale_value(scale, static_cast<int64_t>(query.size_d));

  SDPA_SA_CHECK(query.size_d == value.size_d && key.size_d == value.size_d,
                "Q/K/V head dim mismatch");

  int64_t batchSize = query.size_b;
  int64_t qSize = query.size_m;
  int64_t kvSize = value.size_m;
  int64_t num_head = query.size_h;
  int64_t headSize = query.size_d;

  NormalizedMask nmask{};
  const void *mask_data_void = nullptr;
  bool has_attn_mask =
    attn_mask.has_value() &&
    attn_mask->data != nullptr &&
    (attn_mask->ndim == 2 || attn_mask->ndim == 4) &&
    (attn_mask->ndim == 2
     ? (attn_mask->sizes[0] * attn_mask->sizes[1] > 0)
     : (attn_mask->sizes[0] * attn_mask->sizes[1] * attn_mask->sizes[2] *
        attn_mask->sizes[3] >
        0));

  if (has_attn_mask) {
    nmask = normalize_mask(*attn_mask, batchSize, num_head, qSize, kvSize);
    mask_data_void = nmask.data;
  }

  int64_t qStrideB = query.stride_b;
  int64_t qStrideM = query.stride_m;
  int64_t qStrideH = query.stride_h;
  int64_t kStrideB = key.stride_b;
  int64_t kStrideN = key.stride_m;
  int64_t kStrideH = key.stride_h;
  int64_t vStrideB = value.stride_b;
  int64_t vStrideN = value.stride_m;
  int64_t vStrideH = value.stride_h;
  int64_t oStrideB = out.stride_b;
  int64_t oStrideM = out.stride_m;
  int64_t oStrideH = out.stride_h;

  int64_t mStrideB =
    has_attn_mask ? nmask.m_stride_b : 0;
  int64_t mStrideH =
    has_attn_mask ? nmask.m_stride_h : 0;
  int64_t mStrideM = has_attn_mask ? nmask.m_stride_m : 0;

  int zen_q_split_size = static_cast<int>(q_split_size);
  if (batchSize > 4) {
    zen_q_split_size = 512;
  }
  int64_t qSplitSize =
    zen_q_split_size > qSize ? qSize : static_cast<int64_t>(zen_q_split_size);
  int64_t kvSplitSize =
    kv_split_size > kvSize ? kvSize : kv_split_size;
  int64_t qSlice = (qSize - 1) / qSplitSize + 1;

  int64_t num_thread = (num_threads_hint > 0) ? num_threads_hint
                       : omp_get_max_threads();
  int64_t size_per_thread = qSplitSize * kvSplitSize + qSplitSize + qSplitSize +
                            qSplitSize * headSize;

  const size_t buf_bytes = static_cast<size_t>(num_thread * size_per_thread)
                           * sizeof(accum_t);
  const size_t buf_reduced_bytes = is_reduced_type
                                   ? static_cast<size_t>(num_thread * qSplitSize * kvSplitSize)
                                   * sizeof(scalar_t)
                                   : 0;
  void *scratch = flash_scratch_acquire(buf_bytes + buf_reduced_bytes);
  SDPA_SA_CHECK(scratch != nullptr, "flash scratch allocation failed");

  auto *q_data = static_cast<const scalar_t *>(query.base);
  auto *k_data = static_cast<const scalar_t *>(key.base);
  auto *v_data = static_cast<const scalar_t *>(value.base);
  auto *mask_data = static_cast<const mask_t *>(mask_data_void);
  auto *out_data = static_cast<scalar_t *>(const_cast<void *>(out.base));
  accum_t *buf_data = static_cast<accum_t *>(scratch);
  scalar_t *buf_reduced_data = is_reduced_type
                               ? reinterpret_cast<scalar_t *>(static_cast<char *>(scratch) + buf_bytes)
                               : nullptr;

  scoped_active_levels active_levels_guard(1);

  #pragma omp parallel for schedule(static) num_threads(num_thread)
  for (int64_t z = 0; z < batchSize * num_head * qSlice; ++z) {
    int64_t i = 0, j = 0, k = 0;
    sdpa_data_index_init(z, i, batchSize, j, num_head, k, qSlice);
    int ompIdx = omp_get_thread_num();
    accum_t *buf_ptr = buf_data + ompIdx * size_per_thread;
    accum_t *qk_data = buf_ptr;
    accum_t *qk_max_data = qk_data + qSplitSize * kvSplitSize;
    accum_t *qk_sum_data = qk_max_data + qSplitSize;
    accum_t *dst_data = qk_sum_data + qSplitSize;
    scalar_t *qk_reduced_data =
      is_reduced_type
      ? buf_reduced_data + ompIdx * qSplitSize * kvSplitSize
      : nullptr;

    int64_t m = k * qSplitSize;
    int64_t qBlockSize = std::min(qSplitSize, qSize - m);
    fill_stub_f32<SimdTag>(qk_max_data,
                           -std::numeric_limits<accum_t>::infinity(),
                           qBlockSize);
    fill_stub_f32<SimdTag>(qk_sum_data, static_cast<accum_t>(0), qBlockSize);
    int64_t num_keys = is_causal ? std::min(m + qBlockSize, kvSize) : kvSize;

    for (int64_t n = 0; n < num_keys; n += kvSplitSize) {
      int64_t kvBlockSize = std::min(kvSplitSize, kvSize - n);
      // GEMM: Q_block @ K_block^T -> qk_data.
      //   alpha=1: softmax scale (1/sqrt(d) or user scale) is fused into
      //            mul_reduce_max_fusion / scale_attn_mask_fusion below.
      //   beta =0: qk_data is reused per kv-tile, no prior contents to keep.
      zendnn_gemm<scalar_t>(
        qBlockSize, kvBlockSize, headSize, 1.0f,
        q_data + i * qStrideB + j * qStrideH + m * qStrideM, qStrideM,
        k_data + i * kStrideB + j * kStrideH + n * kStrideN, kStrideN, 0.0f,
        qk_data, kvBlockSize, false, true);

      if (is_causal && num_keys - n <= kvSplitSize) {
        for (int64_t row = 0; row < qBlockSize; ++row) {
          int64_t last_col = m + row - n;
          accum_t *row_ptr = qk_data + row * kvBlockSize;
          if (last_col < 0) {
            fill_stub_f32<SimdTag>(row_ptr,
                                   -std::numeric_limits<accum_t>::infinity(),
                                   kvBlockSize);
          }
          else if (last_col + 1 < kvBlockSize) {
            fill_stub_f32<SimdTag>(row_ptr + last_col + 1,
                                   -std::numeric_limits<accum_t>::infinity(),
                                   kvBlockSize - last_col - 1);
          }
        }
      }

      if (has_attn_mask) {
        for (int64_t row = 0; row < qBlockSize; ++row) {
          scale_attn_mask_fusion<SimdTag>(
            qk_data + row * kvBlockSize,
            mask_data + i * mStrideB + j * mStrideH +
            (m + row) * mStrideM + n,
            static_cast<int>(kvBlockSize), qk_data + row * kvBlockSize,
            scaling_factor);
        }
      }

      accum_t tmp_max = 0, tmp_sum = 0, exp_tmp = 0;
      for (int64_t row = 0; row < qBlockSize; ++row) {
        if (has_attn_mask) {
          tmp_max = row_max<SimdTag>(qk_data + row * kvBlockSize, kvBlockSize);
        }
        else {
          mul_reduce_max_fusion<SimdTag>(qk_data + row * kvBlockSize,
                                         scaling_factor,
                                         static_cast<int>(kvBlockSize),
                                         qk_data + row * kvBlockSize, tmp_max);
        }
        tmp_max = qk_max_data[row] > tmp_max ? qk_max_data[row] : tmp_max;
        tmp_sum = tmp_max;
        if constexpr(is_reduced_type) {
          exp_reduce_sum_fusion_to<SimdTag>(
            qk_data + row * kvBlockSize, static_cast<int>(kvBlockSize),
            conditional_data_ptr(qk_data, qk_reduced_data) +
            row * kvBlockSize,
            tmp_sum);
        }
        else {
          exp_reduce_sum_fusion<SimdTag>(qk_data + row * kvBlockSize,
                                         static_cast<int>(kvBlockSize),
                                         qk_data + row * kvBlockSize, tmp_sum);
        }
        exp_tmp = std::exp(qk_max_data[row] - tmp_max);
        qk_sum_data[row] = tmp_sum + exp_tmp * qk_sum_data[row];
        qk_max_data[row] = tmp_max;
        if (n > 0) {
          scale_dst_row<SimdTag>(dst_data + row * headSize, headSize, exp_tmp);
        }
      }
      // GEMM: softmax_block @ V_block -> dst_data (online-softmax accumulator).
      //   alpha=1: prior dst rows already rescaled by exp(prev_max - new_max)
      //            via scale_dst_row above.
      //   beta = n==0 ? 0 : 1: first kv-tile initialises dst_data, subsequent
      //            tiles accumulate (this is the flash-attention update rule).
      zendnn_gemm<scalar_t>(
        qBlockSize, headSize, kvBlockSize, 1.0f,
        conditional_data_ptr(qk_data, qk_reduced_data), kvBlockSize,
        v_data + i * vStrideB + j * vStrideH + n * vStrideN, vStrideN,
        n == 0 ? 0.0f : 1.0f, dst_data, headSize, false, false);
    }

    for (int64_t row = 0; row < qBlockSize; ++row) {
      const accum_t sum_reciprocal = 1 / qk_sum_data[row];
      write_scaled_output_row<SimdTag, scalar_t>(
        out_data + i * oStrideB + j * oStrideH + m * oStrideM + row * oStrideM,
        out.stride_d, dst_data + row * headSize, headSize, sum_reciprocal);
    }
  }
}

template <typename SimdTag, typename input_type, typename attention_mask>
void flash_attention_kernel_sa_dispatch(
  const sdpa_flash_cpu_tensor_view &output,
  const sdpa_flash_cpu_tensor_view &query, const sdpa_flash_cpu_tensor_view &key,
  const sdpa_flash_cpu_tensor_view &value, double dropout_p, bool is_causal,
  std::optional<sdpa_flash_cpu_mask_view> attn_mask, std::optional<double> scale,
  int num_threads) {
  int64_t q_seq_len = query.size_s;
  if (q_seq_len >= 768) {
    cpu_flash_attention_sa<SimdTag, input_type, attention_mask, 256, 512>(
      output, query, key, value, dropout_p, is_causal, attn_mask,
      scale, num_threads);
  }
  else if (q_seq_len >= 192) {
    cpu_flash_attention_sa<SimdTag, input_type, attention_mask, 64, 512>(
      output, query, key, value, dropout_p, is_causal, attn_mask,
      scale, num_threads);
  }
  else {
    cpu_flash_attention_sa<SimdTag, input_type, attention_mask, 32, 512>(
      output, query, key, value, dropout_p, is_causal, attn_mask,
      scale, num_threads);
  }
}

} // namespace

#if defined(__GNUC__) && !defined(__clang__)
  #pragma GCC pop_options
#endif

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

void sdpa_flash_cpu_free_scratch() {
  free(g_flash_scratch.ptr);
  g_flash_scratch.ptr = nullptr;
  g_flash_scratch.cap = 0;
}

status_t sdpa_flash_cpu_run_internal(
  const sdpa_flash_cpu_tensor_view &output,
  const sdpa_flash_cpu_tensor_view &query,
  const sdpa_flash_cpu_tensor_view &key,
  const sdpa_flash_cpu_tensor_view &value,
  double dropout_p,
  bool is_causal,
  const sdpa_flash_cpu_mask_view *mask,
  const double *scale_opt,
  data_type_t qkv_dt, data_type_t mask_dtype,
  int num_threads) {
  try {
    std::optional<sdpa_flash_cpu_mask_view> mopt;
    if (mask != nullptr && mask->data != nullptr) {
      mopt = *mask;
    }
    std::optional<double> scale;
    if (scale_opt != nullptr) {
      scale = *scale_opt;
    }
    if (dropout_p != 0.0) {
      log_error("sdpa_flash_cpu: dropout must be 0");
      return status_t::failure;
    }

    // Runtime SIMD dispatch: AVX-512 when available, scalar fallback otherwise.
    auto run = [&](auto simd_tag) {
      using Tag = decltype(simd_tag);
      // No-mask and f32-mask paths share the same mask_t = float
      // instantiation; mopt is already std::nullopt when no mask is present.
      if (qkv_dt == data_type_t::f32) {
        flash_attention_kernel_sa_dispatch<Tag, float, float>(
          output, query, key, value, dropout_p, is_causal, mopt, scale,
          num_threads);
      }
      else if (qkv_dt == data_type_t::bf16) {
        if (!mopt.has_value() || mask_dtype == data_type_t::f32) {
          flash_attention_kernel_sa_dispatch<Tag, bf16_elem, float>(
            output, query, key, value, dropout_p, is_causal, mopt, scale,
            num_threads);
        }
        else {
          flash_attention_kernel_sa_dispatch<Tag, bf16_elem, bf16_elem>(
            output, query, key, value, dropout_p, is_causal, mopt, scale,
            num_threads);
        }
      }
      else {
        log_error("sdpa_flash_cpu: unsupported Q/K/V dtype");
        throw std::invalid_argument("unsupported Q/K/V dtype");
      }
    };

    if (zendnnl::common::zendnnl_platform_info().get_avx512f_status()) {
      apilog_info("sdpa_flash_cpu: using AVX-512 SIMD");
      run(simd::avx512_tag{});
    }
    else {
      apilog_info("sdpa_flash_cpu: using scalar SIMD");
      run(simd::scalar_tag{});
    }

    return status_t::success;
  }
  catch (const std::invalid_argument &e) {
    log_error("sdpa_flash_cpu: invalid argument (check mask/shapes): %s",
              e.what());
    return status_t::failure;
  }
}

} // namespace sdpa
} // namespace lowoha
} // namespace zendnnl

#if defined(__GNUC__) && !defined(__clang__)
  #pragma GCC diagnostic pop
#endif
