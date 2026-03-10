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

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
