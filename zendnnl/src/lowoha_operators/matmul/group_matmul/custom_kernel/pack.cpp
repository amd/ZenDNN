/*******************************************************************************
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#include "pack.hpp"

#include <cstdlib>
#include <limits>
#include <mutex>

#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"
#include "lowoha_operators/matmul/lru_cache/lru_cache.hpp"
#include "lowoha_operators/matmul/lru_cache/zendnnl_key.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace custom_kernel {

namespace {

// VNNI pack of [K, N] row-major BF16 → [O/pack_nr, K_pad/2, pack_nr, 2].
// K_pad = K rounded up to even; trailing K-tail (when K is odd) is
// zero-padded so a 4-byte VDPBF16PS load is always safe.
void pack_bf16_vnni(const bfloat16_t *weight, int K, int N, int pack_nr,
                    bfloat16_t *packed) {
  const int K_pair = (K + 1) / 2;
  const int n_blocks = N / pack_nr;

  for (int o_blk = 0; o_blk < n_blocks; ++o_blk) {
    bfloat16_t *blk_base = packed + static_cast<size_t>(o_blk)
        * K_pair * pack_nr * kVNNIPair;
    const int n_base = o_blk * pack_nr;

    for (int kp = 0; kp < K_pair; ++kp) {
      const int k_lo = kp * 2;
      const int k_hi = k_lo + 1;
      bfloat16_t *kp_base = blk_base
          + static_cast<size_t>(kp) * pack_nr * kVNNIPair;

      for (int n = 0; n < pack_nr; ++n) {
        const int col = n_base + n;
        kp_base[n * kVNNIPair + 0] =
            weight[static_cast<size_t>(k_lo) * N + col];
        kp_base[n * kVNNIPair + 1] = (k_hi < K)
            ? weight[static_cast<size_t>(k_hi) * N + col]
            : bfloat16_t(0.0f);
      }
    }
  }
}

// ── Shared per-process cache state (see doc-block below) ────────────
// Pulled out of `get_or_pack_weight_bf16()` so
// `clear_custom_kernel_pack_cache()` can reach the same singleton.
// The outer anonymous namespace keeps everything invisible to other
// translation units.
//
// ── Why eviction is DISABLED (UINT32_MAX capacity) ────────────────
// The dispatcher layer (`prepare_for_call`) stores raw packed_ptrs
// in per-call `CallContext` and the OMP region (`dispatch_tile`)
// reads them WITHOUT re-acquiring this mutex.  If eviction were
// allowed to run on a concurrent `add()` call, it would
// `std::free()` a packed buffer that another thread is still
// reading from — a use-after-free in the microkernel.  Giving the
// cache an infinite capacity sidesteps that race without needing
// per-pointer refcounts.  This explicitly overrides any global
// `ZENDNNL_LRU_CACHE_CAPACITY` setting, because the custom kernel
// has no generation-guarded fast-path retry like the outer
// prepack cache (`kernel_cache.hpp`) does.
//
// ── RSS bound in steady state ─────────────────────────────────────
// Live cache footprint is:
//     Σ_unique(weight_ptr, N, K, pack_nr)  K_pad × N × sizeof(BF16)
// where `K_pad = (K + 1) & ~1` (VNNI pair padding).  For typical
// MoE serving with a fixed model:
//     experts × pack_nr_variants × per-expert pack size
// which is the same order of magnitude as the weights already
// resident in RAM.  The cache is model-bounded, not workload-
// bounded, for fixed-weight deployments.
//
// ── Failure mode for weight-rotating workloads ────────────────────
// Frameworks that repeatedly rebuild or reallocate weight tensors
// (e.g. dynamic LoRA hot-swap, retraining loops, compile/decompile
// cycles) will see the cache footprint grow by one entry per
// distinct (weight pointer, N, K, pack_nr) tuple observed — the
// cache cannot distinguish "the old weight at ptr P is gone" from
// "a fresh weight at ptr P is arriving", so old packed buffers
// stay resident.  For those deployments call
// `clear_custom_kernel_pack_cache()` during a known quiescent
// window between workload phases (see the safety note in pack.hpp).
lru_cache_t<Key_matmul, void *> &pack_cache_singleton() {
  static lru_cache_t<Key_matmul, void *> pack_cache(
      std::numeric_limits<uint32_t>::max());
  return pack_cache;
}
std::mutex &pack_mutex_singleton() {
  static std::mutex pack_mutex;
  return pack_mutex;
}

} // namespace

status_t get_or_pack_weight_bf16(
    const bfloat16_t *weight,
    int K, int N, int pack_nr,
    bool transB,
    const bfloat16_t **out_packed) {

  if (weight == nullptr || K <= 0 || N <= 0
      || (pack_nr != kNRMin && pack_nr != kNRMax)
      || (N % pack_nr) != 0
      || out_packed == nullptr) {
    log_error("custom_kernel pack: invalid arg "
              "(weight, K, N must be valid; pack_nr in {",
              kNRMin, ",", kNRMax, "}; N %% pack_nr == 0)");
    return status_t::failure;
  }
  if (transB) {
    log_error("custom_kernel pack: transB=true not supported by the "
              "custom BF16 microkernel");
    return status_t::failure;
  }

  // Canonical Key_matmul / lru_cache_t pair — same pattern AOCL DLP
  // and oneDNN reorder paths use.  `extra_input_hash` carries the
  // pack-width discriminator so different NR variants of the same
  // weight pointer do not alias.  See the `pack_cache_singleton()`
  // doc-block above for RSS bounds, eviction rationale, and the
  // weight-rotating-workload failure mode.
  auto &pack_cache = pack_cache_singleton();

  // Mark the cache slot as belonging to the custom BF16 microkernel
  // (so it doesn't collide with any existing cache that uses the
  // same `Key_matmul`); fold pack_nr into the discriminator.
  static constexpr uint32_t kCustomKernelAlgoMarker = 0xC0DE0000U;
  const size_t extra_hash = static_cast<size_t>(
      kCustomKernelAlgoMarker | static_cast<uint32_t>(pack_nr));
  Key_matmul key(weight, static_cast<unsigned>(N),
                 static_cast<unsigned>(K), extra_hash);

  std::lock_guard<std::mutex> lock(pack_mutex_singleton());

  if (pack_cache.find_key(key)) {
    *out_packed = static_cast<const bfloat16_t *>(pack_cache.get(key));
    return status_t::success;
  }

  // Miss — allocate, pack, insert.  64-byte aligned so the
  // microkernel's vmovdqu reads start on a cache-line boundary.
  const int K_pair = (K + 1) / 2;
  const size_t bytes = static_cast<size_t>(N / pack_nr)
      * K_pair * pack_nr * kVNNIPair * sizeof(bfloat16_t);
  const size_t alignment = 64;
  const size_t bytes_aligned = (bytes + alignment - 1) & ~(alignment - 1);

  void *raw = std::aligned_alloc(alignment, bytes_aligned);
  if (raw == nullptr) {
    log_error("custom_kernel pack: aligned_alloc failed for ",
              bytes_aligned, " bytes");
    return status_t::failure;
  }

  pack_bf16_vnni(weight, K, N, pack_nr, static_cast<bfloat16_t *>(raw));
  pack_cache.add(key, raw);

  *out_packed = static_cast<const bfloat16_t *>(raw);
  return status_t::success;
}

// See pack.hpp for the quiescent-window safety contract.  The
// implementation is a two-step capacity toggle on the same
// lru_cache_t used by `get_or_pack_weight_bf16()`:
//   1. Lower the capacity to 0 — the LRU's internal `evict()` pass
//      runs on the next mutating op (below) and `std::free()`s every
//      cached packed buffer via `if constexpr(is_pointer)` path.
//   2. Nudge eviction immediately by calling `set_capacity(0)`
//      (which evicts synchronously inside its own mutex).
//   3. Restore capacity to UINT32_MAX so subsequent
//      `get_or_pack_weight_bf16()` hits go back to the eviction-
//      disabled regime.  Without this step the cache would begin
//      evicting packs as callers add new entries — reintroducing
//      the UAF concern that motivated the UINT32_MAX default.
//
// We also hold the outer `pack_mutex_singleton()` for the duration
// so any racing `get_or_pack_weight_bf16()` call blocks until this
// completes; combined with the caller's quiescent-window guarantee,
// no in-flight microkernel can observe a partially-cleared cache.
void clear_custom_kernel_pack_cache() {
  std::lock_guard<std::mutex> lock(pack_mutex_singleton());
  auto &pack_cache = pack_cache_singleton();
  pack_cache.set_capacity(0);  // evict-all, free()s every pointer
  pack_cache.set_capacity(std::numeric_limits<uint32_t>::max());
}

} // namespace custom_kernel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
