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

#ifndef MATMUL_NATIVE_KERNEL_CACHE_HPP
#define MATMUL_NATIVE_KERNEL_CACHE_HPP

#include <mutex>
#include <memory>
#include <unordered_map>
#include <functional>
#include <cstddef>
#include <cstdlib>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

/// Canonical panel width for B prepacking.
/// Must be >= max NR across all kernel variants (64).
/// All kernels work with this panel width regardless of their own NR.
/// A kernel with NR=48 loads 3 of 4 ZMMs per k-row; the unused 16 floats
/// are zero-padded and ignored.
inline constexpr int NR_PACK = 64;

/// Custom deleter for aligned_alloc'd memory.
struct AlignedFreeDeleter {
  void operator()(float *p) const { std::free(p); }
};

/// Prepacked weight buffer: NR_PACK-wide K-contiguous panels.
///
/// Layout:
///   n_panels = ceil(N / NR_PACK)
///   Panel p covers columns [p*NR_PACK .. min((p+1)*NR_PACK, N)-1]
///   Within panel: packed[kk * NR_PACK + nr]
///   Total size: n_panels * K * NR_PACK floats
///
/// This is the optimal AVX-512 layout:
///   - b_stride = NR_PACK = 64 (256 bytes between k-rows)
///   - HW prefetcher handles 256-byte stride perfectly
///   - Micro-tile's B working set (KB × 64 × 4) fits in L2
///   - Any kernel NR (16, 32, 48, 64) reads a prefix of the panel
///   - Decoupled from MR/NR: cache key has no tile dependency
///
/// Indexing:
///   get_panel(pc, panel_idx) → buf + panel_idx * K * NR_PACK + pc * NR_PACK
///   b_stride = NR_PACK
struct PrepackedWeight {
  std::unique_ptr<float[], AlignedFreeDeleter> buf;
  const float *data;   ///< Points to buf.get() (always owns for panel format)
  int K;         ///< K dimension
  int N;         ///< Original N dimension
  int n_panels;    ///< ceil(N / NR_PACK)

  /// Get B panel pointer for K-offset pc, N-panel index.
  const float *get_panel(int pc, int panel_idx) const {
    return data + static_cast<size_t>(panel_idx) * K * NR_PACK + static_cast<size_t>(pc) * NR_PACK;
  }

  /// Row stride for microkernel (constant).
  static constexpr int stride() { return NR_PACK; }
};

/// Cache key — no NR dependency.
struct PrepackedWeightKey {
  const void *weight_ptr;
  int K, N, ldb;
  bool transB;

  bool operator==(const PrepackedWeightKey &o) const {
    return weight_ptr == o.weight_ptr &&
         K == o.K && N == o.N && ldb == o.ldb &&
         transB == o.transB;
  }
};

struct PrepackedWeightKeyHash {
  size_t operator()(const PrepackedWeightKey &k) const {
    size_t h = std::hash<const void *>()(k.weight_ptr);
    auto combine = [](size_t &seed, size_t val) {
      seed ^= val + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    };
    combine(h, std::hash<int>()(k.K));
    combine(h, std::hash<int>()(k.N));
    combine(h, std::hash<int>()(k.ldb));
    combine(h, std::hash<bool>()(k.transB));
    return h;
  }
};

/// Thread-safe singleton cache for prepacked weight matrices.
class PrepackedWeightCache {
public:
  static PrepackedWeightCache &instance();
  const PrepackedWeight *get_or_prepack(
    const PrepackedWeightKey &key, const float *weight);
  PrepackedWeightCache(const PrepackedWeightCache &) = delete;
  PrepackedWeightCache &operator=(const PrepackedWeightCache &) = delete;
private:
  PrepackedWeightCache() = default;
  std::unordered_map<PrepackedWeightKey,
             std::unique_ptr<PrepackedWeight>,
             PrepackedWeightKeyHash> cache_;
  std::mutex mutex_;
};

// ============================================================================
// BF16 VNNI prepacked weight cache
// ============================================================================

/// VNNI pair constant: 2 BF16 values per element position per k-pair.
inline constexpr int VNNI_PAIR = 2;

/// Custom deleter for uint16_t aligned_alloc'd memory.
struct AlignedFreeU16Deleter {
  void operator()(uint16_t *p) const { std::free(p); }
};

/// BF16 VNNI prepacked weight buffer.
///
/// Layout: for each NR_PACK-wide panel and each k-pair:
///   packed[kp * NR_PACK * 2 + n * 2 + 0] = B[2*kp  ][panel*NR_PACK + n]
///   packed[kp * NR_PACK * 2 + n * 2 + 1] = B[2*kp+1][panel*NR_PACK + n]
///
/// K is padded to even for VNNI alignment.
struct BF16PrepackedWeight {
  std::unique_ptr<uint16_t[], AlignedFreeU16Deleter> buf;
  const uint16_t *data;  ///< Points to buf.get()
  int K;           ///< Original K
  int K_padded;    ///< K rounded up to even
  int N;
  int n_panels;

  /// Get B panel pointer for k-pair index kp, N-panel index.
  const uint16_t *get_panel(int kp, int panel_idx) const {
    const int vnni_stride = NR_PACK * VNNI_PAIR;
    const int k_pairs = K_padded / 2;
    return data
        + static_cast<size_t>(panel_idx) * k_pairs * vnni_stride
        + static_cast<size_t>(kp) * vnni_stride;
  }

  /// Row stride in uint16_t units for one k-pair.
  static constexpr int stride() { return NR_PACK * VNNI_PAIR; }
};

/// Thread-safe singleton cache for BF16 VNNI prepacked weight matrices.
class BF16PrepackedWeightCache {
public:
  static BF16PrepackedWeightCache &instance();
  const BF16PrepackedWeight *get_or_prepack(
    const PrepackedWeightKey &key, const uint16_t *weight);
  BF16PrepackedWeightCache(const BF16PrepackedWeightCache &) = delete;
  BF16PrepackedWeightCache &operator=(const BF16PrepackedWeightCache &) = delete;
private:
  BF16PrepackedWeightCache() = default;
  std::unordered_map<PrepackedWeightKey,
             std::unique_ptr<BF16PrepackedWeight>,
             PrepackedWeightKeyHash> cache_;
  std::mutex mutex_;
};

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_KERNEL_CACHE_HPP
