/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#include "lowoha_operators/matmul/matmul_native/common/kernel_cache.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/bf16/bf16_gemv_bkc.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/int8/int8_gemv_bkc.hpp"
#include <cstring>
#include <algorithm>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

PrepackedWeightCache &PrepackedWeightCache::instance() {
  static PrepackedWeightCache inst;
  return inst;
}

const PrepackedWeight *PrepackedWeightCache::get_or_prepack(
  const PrepackedWeightKey &key, const float *weight) {

  // TODO: The mutex is held during the O(K*N) packing below. For large
  // weight matrices this blocks threads that need different keys.
  // Consider per-key locking or double-checked locking to allow
  // concurrent packing of independent weight matrices.
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = cache_.find(key);
  if (it != cache_.end())
    return it->second.get();

  const int K = key.K, N = key.N, ldb = key.ldb;
  const bool transB = key.transB;
  if (K <= 0 || N <= 0) return nullptr;
  const int np = (N + NR_PACK - 1) / NR_PACK;
  const size_t total = static_cast<size_t>(np) * K * NR_PACK;

  float *buf = static_cast<float *>(
    std::aligned_alloc(64, ((total * sizeof(float) + 63) & ~size_t(63))));
  if (!buf) return nullptr;

  // Pack into NR_PACK-wide K-contiguous panels.
  // Each panel: packed[kk * NR_PACK + nr] = B_logical[kk][panel*NR_PACK + nr]
  //
  // For non-transposed B with NR_PACK=64:
  //   Each k-row copies min(NR_PACK, N - j0) floats via memcpy.
  //   This gives perfect 256-byte stride for the microkernel.
  for (int jp = 0; jp < np; ++jp) {
    const int j0 = jp * NR_PACK;
    const int nr_act = std::min(NR_PACK, N - j0);
    float *dst = buf + static_cast<size_t>(jp) * K * NR_PACK;

    if (!transB) {
      // B row-major: B[k][n] = weight[k * ldb + n]
      for (int kk = 0; kk < K; ++kk) {
        float *d = dst + kk * NR_PACK;
        std::memcpy(d, weight + static_cast<size_t>(kk) * ldb + j0,
              nr_act * sizeof(float));
        if (nr_act < NR_PACK)
          std::memset(d + nr_act, 0,
                (NR_PACK - nr_act) * sizeof(float));
      }
    } else {
      // B transposed: B_logical[k][n] = weight[n * ldb + k]
      for (int kk = 0; kk < K; ++kk) {
        float *d = dst + kk * NR_PACK;
        for (int nr = 0; nr < nr_act; ++nr)
          d[nr] = weight[static_cast<size_t>(j0 + nr) * ldb + kk];
        for (int nr = nr_act; nr < NR_PACK; ++nr)
          d[nr] = 0.0f;
      }
    }
  }

  auto pw = std::make_unique<PrepackedWeight>();
  pw->buf.reset(buf);
  pw->data = buf;
  pw->K = K;
  pw->N = N;
  pw->n_panels = np;

  const PrepackedWeight *raw = pw.get();
  cache_[key] = std::move(pw);
  return raw;
}

// ============================================================================
// BF16 VNNI prepacked weight cache
// ============================================================================

BF16PrepackedWeightCache &BF16PrepackedWeightCache::instance() {
  static BF16PrepackedWeightCache inst;
  return inst;
}

const BF16PrepackedWeight *BF16PrepackedWeightCache::get_or_prepack(
  const PrepackedWeightKey &key, const uint16_t *weight) {

  std::lock_guard<std::mutex> lock(mutex_);
  auto it = cache_.find(key);
  if (it != cache_.end())
    return it->second.get();

  const int K = key.K, N = key.N, ldb = key.ldb;
  const bool transB = key.transB;
  if (K <= 0 || N <= 0) return nullptr;
  const int K_padded = (K + 1) & ~1;
  const int np = (N + NR_PACK - 1) / NR_PACK;
  const int k_pairs = K_padded / 2;
  const int vnni_stride = NR_PACK * VNNI_PAIR;
  const size_t total = static_cast<size_t>(np) * k_pairs * vnni_stride;

  // Allocate 64-byte aligned buffer
  uint16_t *buf = static_cast<uint16_t *>(
    std::aligned_alloc(64, ((total * sizeof(uint16_t) + 63) & ~size_t(63))));
  if (!buf) return nullptr;

  // Pack into VNNI format: k-pairs interleaved at the N dimension.
  // For each panel and k-pair:
  //   packed[n * 2 + 0] = B_logical[2*kp  ][panel*NR_PACK + n]
  //   packed[n * 2 + 1] = B_logical[2*kp+1][panel*NR_PACK + n]
  for (int jp = 0; jp < np; ++jp) {
    const int j0 = jp * NR_PACK;
    const int nr_act = std::min(NR_PACK, N - j0);
    uint16_t *dst = buf + static_cast<size_t>(jp) * k_pairs * vnni_stride;

    for (int kp = 0; kp < k_pairs; ++kp) {
      uint16_t *d = dst + kp * vnni_stride;
      const int k0 = kp * 2;
      const int k1 = k0 + 1;

      if (!transB) {
        // B row-major: B_logical[k][n] = weight[k * ldb + n]
        for (int n = 0; n < nr_act; ++n) {
          d[n * VNNI_PAIR + 0] = (k0 < K) ? weight[static_cast<size_t>(k0) * ldb + (j0 + n)] : 0;
          d[n * VNNI_PAIR + 1] = (k1 < K) ? weight[static_cast<size_t>(k1) * ldb + (j0 + n)] : 0;
        }
      } else {
        // B transposed: B_logical[k][n] = weight[n * ldb + k]
        for (int n = 0; n < nr_act; ++n) {
          d[n * VNNI_PAIR + 0] = (k0 < K) ? weight[static_cast<size_t>(j0 + n) * ldb + k0] : 0;
          d[n * VNNI_PAIR + 1] = (k1 < K) ? weight[static_cast<size_t>(j0 + n) * ldb + k1] : 0;
        }
      }
      // Zero-pad unused columns
      for (int n = nr_act; n < NR_PACK; ++n) {
        d[n * VNNI_PAIR + 0] = 0;
        d[n * VNNI_PAIR + 1] = 0;
      }
    }
  }

  auto pw = std::make_unique<BF16PrepackedWeight>();
  pw->buf.reset(buf);
  pw->data = buf;
  pw->K = K;
  pw->K_padded = K_padded;
  pw->N = N;
  pw->n_panels = np;

  const BF16PrepackedWeight *raw = pw.get();
  cache_[key] = std::move(pw);
  return raw;
}

// ============================================================================
// BF16 Blocked K-contiguous (BKC) VNNI packed weight cache
// ============================================================================

BF16BKCWeightCache &BF16BKCWeightCache::instance() {
  static BF16BKCWeightCache inst;
  return inst;
}

const BF16BKCWeight *BF16BKCWeightCache::get_or_pack(
  const PrepackedWeightKey &key, const uint16_t *weight) {

  // Fast path: check cache under lock.
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = cache_.find(key);
    if (it != cache_.end())
      return it->second.get();
  }

  // Cache miss: pack outside the lock to avoid serializing threads.
  const int K = key.K, N = key.N, ldb = key.ldb;
  const bool transB = key.transB;
  if (K <= 0 || N <= 0) return nullptr;
  const int K_padded = (K + 1) & ~1;
  const int N_padded = ((N + NR_PACK - 1) / NR_PACK) * NR_PACK;
  const size_t total = static_cast<size_t>(K_padded) * N_padded;

  uint16_t *buf = static_cast<uint16_t *>(
    std::aligned_alloc(64, ((total * sizeof(uint16_t) + 63) & ~size_t(63))));
  if (!buf) return nullptr;

  pack_b_bkc_ext(weight, ldb, K, N, transB, buf);

  auto pw = std::make_unique<BF16BKCWeight>();
  pw->buf.reset(buf);
  pw->data = buf;
  pw->K = K;
  pw->K_padded = K_padded;
  pw->N = N;
  pw->N_padded = N_padded;
  pw->total = total;

  // Re-acquire lock and insert (another thread may have raced).
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = cache_.find(key);
  if (it != cache_.end())
    return it->second.get();  // race loser — discard our pack, use winner's

  const BF16BKCWeight *raw = pw.get();
  cache_[key] = std::move(pw);
  return raw;
}

// ============================================================================
// INT8 K-contiguous VNNI packed weight cache
// ============================================================================

INT8KContiguousWeightCache &INT8KContiguousWeightCache::instance() {
  static INT8KContiguousWeightCache inst;
  return inst;
}

const INT8KContiguousWeight *INT8KContiguousWeightCache::get_or_pack(
  const PrepackedWeightKey &key,
  const int8_t *weight,
  float src_scale, int32_t src_zp,
  const float *bias,
  const float *wei_scale, int wei_scale_count) {

  // Fast path: check cache under lock.
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      auto *entry = it->second.get();
      if (entry->cached_src_scale == src_scale
          && entry->cached_src_zp == src_zp)
        return entry;
      // Quant params changed — return nullptr to force thread-local path.
      // Avoids in-place mutation that races with concurrent readers.
      return nullptr;
    }
  }

  // Cache miss: pack outside the lock.
  const int K = key.K, N = key.N, ldb = key.ldb;
  const bool transB = key.transB;
  if (K <= 0 || N <= 0) return nullptr;
  const int K_padded = (K + 3) & ~3;
  const int N_padded = ((N + NR_PACK - 1) / NR_PACK) * NR_PACK;
  const size_t packed_total = static_cast<size_t>(K_padded) * N_padded;

  int8_t *pbuf = static_cast<int8_t *>(
    std::aligned_alloc(64, ((packed_total + 63) & ~size_t(63))));
  if (!pbuf) return nullptr;

  int32_t *cs_buf = static_cast<int32_t *>(
    std::aligned_alloc(64, ((N_padded * sizeof(int32_t) + 63) & ~size_t(63))));
  if (!cs_buf) { std::free(pbuf); return nullptr; }

  float *cscale_buf = static_cast<float *>(
    std::aligned_alloc(64, ((N_padded * sizeof(float) + 63) & ~size_t(63))));
  if (!cscale_buf) { std::free(pbuf); std::free(cs_buf); return nullptr; }

  float *ebias_buf = static_cast<float *>(
    std::aligned_alloc(64, ((N_padded * sizeof(float) + 63) & ~size_t(63))));
  if (!ebias_buf) { std::free(pbuf); std::free(cs_buf); std::free(cscale_buf); return nullptr; }

  pack_b_int8_bkc(weight, ldb, K, N, transB, pbuf, cs_buf);
  precompute_int8_dequant(
      cs_buf, bias, src_scale, src_zp,
      wei_scale, wei_scale_count,
      N, N_padded, cscale_buf, ebias_buf);

  auto pw = std::make_unique<INT8KContiguousWeight>();
  pw->packed_buf.reset(pbuf);
  pw->col_sum_buf.reset(cs_buf);
  pw->combined_scale_buf.reset(cscale_buf);
  pw->effective_bias_buf.reset(ebias_buf);
  pw->data = pbuf;
  pw->col_sum = cs_buf;
  pw->combined_scale = cscale_buf;
  pw->effective_bias = ebias_buf;
  pw->K = K;
  pw->K_padded = K_padded;
  pw->N = N;
  pw->N_padded = N_padded;
  pw->total = packed_total;
  pw->cached_src_scale = src_scale;
  pw->cached_src_zp = src_zp;

  // Re-acquire lock and insert.
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = cache_.find(key);
  if (it != cache_.end())
    return it->second.get();

  const INT8KContiguousWeight *raw = pw.get();
  cache_[key] = std::move(pw);
  return raw;
}

// ============================================================================
// INT8 VNNI panel-format prepacked weight cache (for BRGEMM looper)
// ============================================================================

INT8PrepackedWeightCache &INT8PrepackedWeightCache::instance() {
  static INT8PrepackedWeightCache inst;
  return inst;
}

const INT8PrepackedWeight *INT8PrepackedWeightCache::get_or_prepack(
  const PrepackedWeightKey &key, const int8_t *weight) {

  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = cache_.find(key);
    if (it != cache_.end())
      return it->second.get();
  }

  const int K = key.K, N = key.N, ldb = key.ldb;
  const bool transB = key.transB;
  if (K <= 0 || N <= 0) return nullptr;
  const int K_padded = (K + 3) & ~3;
  const int k_quads  = K_padded / 4;
  const int np = (N + NR_PACK - 1) / NR_PACK;
  const int vnni_stride = NR_PACK * INT8_VNNI_GRP;
  const size_t pack_total = static_cast<size_t>(np) * k_quads * vnni_stride;

  int8_t *pbuf = static_cast<int8_t *>(
    std::aligned_alloc(64, ((pack_total + 63) & ~size_t(63))));
  if (!pbuf) return nullptr;

  int32_t *cs_buf = static_cast<int32_t *>(
    std::aligned_alloc(64, ((N * sizeof(int32_t) + 63) & ~size_t(63))));
  if (!cs_buf) { std::free(pbuf); return nullptr; }

  std::memset(cs_buf, 0, N * sizeof(int32_t));
  for (int jp = 0; jp < np; ++jp) {
    const int j0 = jp * NR_PACK;
    const int nr_act = std::min(NR_PACK, N - j0);
    int8_t *dst = pbuf + static_cast<size_t>(jp) * k_quads * vnni_stride;
    for (int kq = 0; kq < k_quads; ++kq) {
      int8_t *d = dst + kq * vnni_stride;
      const int k_base = kq * 4;
      for (int n = 0; n < nr_act; ++n) {
        int32_t sum = 0;
        for (int i = 0; i < 4; ++i) {
          const int k = k_base + i;
          int8_t val = 0;
          if (k < K)
            val = transB ? weight[(j0+n)*ldb+k] : weight[k*ldb+(j0+n)];
          d[n*4+i] = val;
          sum += val;
        }
        cs_buf[j0+n] += sum;
      }
      for (int n = nr_act; n < NR_PACK; ++n) {
        d[n*4+0] = 0; d[n*4+1] = 0; d[n*4+2] = 0; d[n*4+3] = 0;
      }
    }
  }

  auto pw = std::make_unique<INT8PrepackedWeight>();
  pw->packed_buf.reset(pbuf);
  pw->col_sum_buf.reset(cs_buf);
  pw->data = pbuf;
  pw->col_sum = cs_buf;
  pw->K = K;
  pw->K_padded = K_padded;
  pw->N = N;
  pw->n_panels = np;

  std::lock_guard<std::mutex> lock(mutex_);
  auto it = cache_.find(key);
  if (it != cache_.end())
    return it->second.get();

  const INT8PrepackedWeight *raw = pw.get();
  cache_[key] = std::move(pw);
  return raw;
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
