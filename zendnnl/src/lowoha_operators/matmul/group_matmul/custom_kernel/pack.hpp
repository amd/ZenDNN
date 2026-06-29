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

/// BF16 custom kernel — adaptive weight pack and per-process pack cache.
/// Shared by both non-fused flat_n_tile (ALGO 3) and the fused-MoE
/// tight-arena path.
///
/// PACK LAYOUT (BF16, VDPBF16PS-friendly, "VNNI" style — same as
/// `bf16_brgemm_ukernel.cpp`'s NR-templated layout):
///
///     packed[O / pack_nr][K / 2][pack_nr][2]
///
///   * `pack_nr` is a runtime parameter — supported values 32 and 64.
///     Larger NR = fewer microkernel calls per N-tile, smaller NR =
///     more N-divisibility flexibility and tighter register budget.
///   * The trailing dim 2 packs two consecutive K rows, byte-laid as
///       byte[0..1] = row k        (BF16, lo)
///       byte[2..3] = row k+1      (BF16, hi)
///     so each `vmovdqu` of 64 bytes loads 16 N-cols of one K-pair
///     as 16 contiguous (k_lo, k_hi) FP32-lane-formatted pairs.
///   * For odd K the implementation pads the trailing K-pair with a
///     zero second element (no contribution to the dot product).
///
/// CACHE: shared `lru_cache_t<Key_matmul, void *>` (the same template
/// the AOCL DLP / oneDNN reorder caches in `aocl_kernel.cpp` use).
/// Keyed on (weight_ptr, K, N, pack_nr, ldb, transB) — different
/// NR / layout / stride variants of the same weight occupy distinct
/// cache entries.
///
/// Eviction policy: the per-process custom-kernel pack cache uses a
/// hard-capped capacity of `UINT32_MAX` entries (effectively
/// unbounded for any realistic workload) and DOES NOT EVICT during
/// regular operation — `ZENDNNL_LRU_CACHE_CAPACITY` does NOT govern
/// this cache.  The reason: raw packed-weight pointers are captured
/// into `CallContext.packed_ptrs[]` and read by OMP workers outside
/// the cache mutex, so any eviction during in-flight dispatch would
/// be a use-after-free.  The cache is reset only by an explicit
/// call to `clear_custom_kernel_pack_cache()` in a quiescent window
/// (see safety contract on that function).

#ifndef ZENDNNL_GROUP_MATMUL_CUSTOM_KERNEL_PACK_HPP
#define ZENDNNL_GROUP_MATMUL_CUSTOM_KERNEL_PACK_HPP

#include "common/bfloat16.hpp"
#include "common/error_status.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace custom_kernel {

using zendnnl::common::bfloat16_t;
using zendnnl::error_handling::status_t;

/// VNNI K-pair multiplicity (= 2: two consecutive K-rows per VDPBF16PS lane).
inline constexpr int kVNNIPair = 2;

/// VNNI K-quad multiplicity (= 4: four consecutive K-rows per
/// VPDPBUSD lane).  Mirror constant for the int8 pack family; the
/// int8 microkernel header re-uses this name and the two declarations
/// MUST stay in sync (compile-time consistency enforced indirectly
/// by the pack ↔ microkernel layout contract, see
/// `int8_microkernel.{hpp,cpp}`).
inline constexpr int kVNNIInt8Quad = 4;

/// Supported pack widths.  Match the `bf16_brgemm_ukernel.cpp` set
/// minus NR=16 (gated activation needs at least 32 cols for one
/// (gate, up) pair to fit in a single (acc_lo, acc_hi) zmm pair).
inline constexpr int kNRMin = 32;
inline constexpr int kNRMax = 64;

/// Look up the pre-packed BF16 weight for one
/// (weight, K, N, pack_nr, transB) tuple, packing on the first miss
/// and caching for subsequent calls.
///
/// Contract:
///   * `weight`  — caller-owned BF16 buffer.  Logical layout depends
///                 on `transB`:
///                   * `transB == false` → row-major [K, N] with row
///                     stride `ldb` (typically `ldb == N`).
///                   * `transB == true`  → row-major [N, K] with row
///                     stride `ldb` (typically `ldb == K`).  This
///                     matches the standard `nn.Linear` weight
///                     convention used by major LLM serving stacks
///                     (`y = x @ w^T`, weight stored as
///                     `[out_features, in_features]`).
///                 Pointer identity is part of the cache key (callers
///                 must keep it alive while the entry is cached).
///   * `K`, `N`  — logical dimensions; `N` MUST be a multiple of
///                 `pack_nr` regardless of `transB`.
///   * `ldb`     — caller's row stride.  Pack reads the weight as
///                 `weight[row * ldb + col]` (row-major) — passing the
///                 actual stride (rather than assuming N or K) lets
///                 the pack handle non-contiguous slices correctly.
///   * `pack_nr` — pack width, MUST be one of {kNRMin, kNRMax}.
///   * `transB`  — false: standard [K, N] caller layout.
///                 true:  PyTorch [N, K] caller layout.  The pack
///                 produces an identical VNNI output for both
///                 layouts; only the input access pattern differs.
///
/// On success `*out_packed` points into cache-owned storage that is
/// retained for the lifetime of the process (or until
/// `clear_custom_kernel_pack_cache()` is called during a quiescent
/// window).  The cache never evicts during regular operation so raw
/// pointers held in `CallContext.packed_ptrs[]` across an OMP region
/// cannot be freed underneath the microkernel.  Returns failure on
/// bad arguments or out-of-memory while packing.
///
/// `transB` is folded into the cache key so transB=true and
/// transB=false on the same weight pointer occupy distinct entries
/// (would otherwise alias and produce silent-wrong output).
///
/// `interleave_split_halves` (default false) — when true, the pack
/// reads source columns in interleaved order so the resulting packed
/// arena physically matches the `swiglu_oai_mul` layout.  Used for
/// the `silu_and_mul` and `gelu_and_mul` fused-CK paths: in both
/// cases the caller's W13 weight is in canonical split-halves
/// `[gate_cols | up_cols]` layout, but the CK kernel's in-register
/// pair-store epilogues (`silu_and_mul_store_pair`,
/// `gelu_and_mul_store_pair`) require the interleaved
/// `[g0, u0, g1, u1, ...]` order.  silu and gelu share the SAME
/// permutation — only the kernel-side activation math differs — so
/// the cache-key bit is shared between them and a packed arena
/// warmed for silu can be reused for gelu on the same weight pointer.
/// When the flag is set, pack output column `2i+0` reads canonical
/// column `i` (gate i) and pack output column `2i+1` reads canonical
/// column `I + i` (up i), where `I = N / 2`.  Requires N to be even.
/// Folded into the cache key so the same weight pointer packed with
/// and without interleaving occupies distinct entries.
///
/// `was_hit_out` (optional) — when non-null, set to `true` if this
/// call served from the cache (no pack work) or `false` if it had
/// to allocate + pack on a miss.  Used by the prepack-all probe
/// helper (see `warm_pack_all_custom_kernel_experts`) to report
/// per-call HIT/MISS counts at apilog level.  Default `nullptr`
/// preserves the pre-existing call sites unchanged.
///
/// `disable_cache` (default false) — library-wide weight-cache
/// toggle.  When `false`, the function behaves as documented above:
/// served from the per-process LRU on a hit, allocated + packed +
/// inserted on a miss, retained for the process lifetime.  When
/// `true`, the LRU is BYPASSED entirely: the function always
/// allocates a fresh aligned buffer, packs into it, and returns
/// the raw pointer to the caller WITHOUT inserting into the LRU
/// (so no future call observes this buffer through the cache key).
/// In disable-cache mode `*out_packed` is therefore CALLER-OWNED
/// storage: the caller MUST free it via
/// `free_owned_packed_weight()` once `dispatch_tile()` has finished
/// consuming it (typical lifetime: from end of `prepare_for_call()`
/// to right after the OMP parallel region exits).  Disable-cache
/// mode is the runtime counterpart of the prepack-side
/// `ZENDNNL_MATMUL_WEIGHT_CACHE != 1` gate (see
/// `prepack/prepack_custom_kernel.cpp::warm_pack_all_custom_kernel_experts`)
/// — together they let frameworks with churning weight addresses
/// keep the CK kernel math while paying a per-call pack cost
/// (no inter-call amortisation, no stale pointer hits).
/// `was_hit_out` is forced to `false` in disable-cache mode (the
/// pack always runs).
/// `in_place` (WEIGHT_CACHE=2): when set AND the packed layout is
/// byte-for-byte the same size as the raw weight (bf16 even-K, where
/// `packed_bytes == K*N*sizeof(bf16)`), the pack is written back into
/// the caller's `weight` buffer itself and the cache stores a nullptr
/// sentinel (so `*out_packed == weight` and the LRU never frees the
/// caller's buffer).  This saves one packed buffer per weight.
/// Alignment is NOT required: the bf16 microkernel reads the packed
/// B-stream with the unaligned `_mm512_loadu_si512`, so handing it the
/// caller's (possibly unaligned) weight buffer is correct — an
/// unaligned base only costs cache-line split loads on the weight
/// stream, free when the buffer happens to be aligned.  CALLER CONTRACT
/// (mirrors the AOCL in-place path): the weight buffer is mutated to the
/// packed layout, so the caller must NOT read it raw afterwards, and
/// must NOT clear the pack cache while still reusing the buffer (a
/// post-clear miss would re-pack the already-packed bytes).  Falls back
/// to the out-of-place cache on a size mismatch (e.g. odd K), so it is
/// always safe to request.  Ignored unless the cache is active
/// (`!disable_cache`).
status_t get_or_pack_weight_bf16(
    const bfloat16_t *weight,
    int K, int N, int ldb, int pack_nr,
    bool transB,
    bool interleave_split_halves,
    const bfloat16_t **out_packed,
    bool *was_hit_out = nullptr,
    bool disable_cache = false,
    bool in_place = false);

/// Free a packed-weight buffer returned by
/// `get_or_pack_weight_bf16(..., disable_cache=true)`.  Safe with
/// `nullptr`.  Use ONLY for caller-owned buffers obtained from
/// disable-cache mode — cached pointers are owned by the LRU and
/// MUST NOT be passed here (would double-free on
/// `clear_custom_kernel_pack_cache()`).  The CallContext
/// destructor in `dispatch.hpp` routes every entry it owns
/// through this helper, so application code typically never
/// calls it directly.
void free_owned_packed_weight(const bfloat16_t *packed);

// ── DQ-INT8 pack ────────────────────────────────────────────────
// Adaptive INT8 weight pack with a per-column compensation row.
// Used by the INT8 microkernel (custom_kernel/ukernel/
// int8_microkernel.{hpp,cpp}) for the s8 × {s8, u8} → bf16
// fast path on bf16 src reorder-quantised down to s8 / u8.
//
// PACK LAYOUT (per o-block of `pack_nr` cols):
//
//     [O/pack_nr][
//        [K_pad/4][pack_nr][4]    ← VNNI-quad packed weight bytes
//        [pack_nr] int32_t         ← per-column compensation row
//                                    (sum_k wei_s8[k, col])
//     ]
//
// `pack_nr` is a runtime parameter, supported values 32 and 64
// (shared with the bf16 pack).  `K_pad = ceil(K / 4) * 4`.  The
// trailing K-quad is zero-padded so a 4-byte VPDPBUSD broadcast
// is always safe.  Compensation is computed at pack time from
// the FULL K column (no zero-pad contribution since wei_s8[K_pad-1
// down to K] is zero) and stored as one int32 per column —
// `comp[v_col] = sum_{k=0}^{K-1} wei_s8[k, v_col]`.
//
// Cache: SEPARATE singleton from the BF16 pack — different value
// type (`int8_t *`) and different cache-key marker bit prevent
// collision.  The cache-key `extra_hash` ORs `kCustomKernelInt8Marker`
// (a distinct constant from `kCustomKernelAlgoMarker`) so a bf16
// pack and an int8 pack for the same weight pointer occupy
// disjoint LRU entries.

/// Look up the pre-packed INT8 weight for one
/// (weight, K, N, pack_nr, transB) tuple, packing on the first
/// miss and caching for subsequent calls.
///
/// Contract is the same as `get_or_pack_weight_bf16` (see above)
/// with three differences:
///   * `weight` is `int8_t *` (signed); the pack writes raw s8
///     bytes into the VNNI-quad slab plus an `int32_t` compensation
///     row at the tail of each o-block.
///   * `interleave_split_halves` carries the same semantic as in
///     the bf16 path — `silu_and_mul` / `gelu_and_mul` callers
///     present canonical split-halves `[gate_cols | up_cols]`
///     weight and the pack re-orders source columns into the
///     interleaved `[g0, u0, g1, u1, ...]` layout the in-register
///     pair-pack epilogues expect.  Compensation is computed
///     AFTER re-ordering so the per-column sums match the
///     re-permuted output column index.
///   * Output is `int8_t *` — the caller / dispatcher will read it
///     as the VNNI-quad slab and then the int32 compensation row
///     using byte arithmetic; see int8_microkernel.cpp for the
///     consumer side.
///
/// `disable_cache` (default false) behaves identically to the
/// bf16 sibling: when true, allocate a fresh aligned buffer per
/// call, skip the LRU singleton, and let the caller free via
/// `free_owned_packed_weight_int8()`.
status_t get_or_pack_weight_int8(
    const int8_t *weight,
    int K, int N, int ldb, int pack_nr,
    bool transB,
    bool interleave_split_halves,
    const int8_t **out_packed,
    bool *was_hit_out = nullptr,
    bool disable_cache = false);

/// Free a packed-weight buffer returned by
/// `get_or_pack_weight_int8(..., disable_cache=true)`.  Safe with
/// `nullptr`.  Same lifetime contract as
/// `free_owned_packed_weight` (the bf16 sibling).
void free_owned_packed_weight_int8(const int8_t *packed);

/// Release every cached packed BF16 weight and reset the cache to
/// empty.  Intended for weight-rotating deployments (dynamic
/// LoRA hot-swap, retraining loops, recompile / redeploy cycles)
/// that would otherwise accumulate entries across workload phases
/// and eventually exhaust RSS.
///
/// ⚠ SAFETY: the pack cache has no eviction during regular
/// operation because raw packed pointers are captured into
/// `CallContext.packed_ptrs[]` and read by OMP workers outside the
/// cache mutex (see the detailed comment in `pack.cpp`).  Calling
/// this function while ANY other thread is in an active custom-
/// kernel dispatch would use-after-free whatever entry the
/// dispatcher is currently reading.  Callers MUST ensure a
/// quiescent window — no in-flight calls to
/// `group_matmul_direct(...)` or its fused-MoE variants with
/// `ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL=1` on ANY thread — before
/// invoking.  Typical integration points are:
///   * Between inference batches in a serving framework that joins
///     its worker threads at a batch boundary.
///   * After model unload / before reload in a LoRA hot-swap
///     sequence that serialises model-lifecycle events through a
///     single control thread.
///   * During graceful shutdown.
///
/// After return the cache is empty and future calls to
/// `get_or_pack_weight_bf16()` pay the repack cost on first miss.
void clear_custom_kernel_pack_cache();

/// Release every cached packed INT8 weight and reset the cache to
/// empty.  Mirror of `clear_custom_kernel_pack_cache()` for the
/// disjoint INT8 LRU singleton.  See the warning above — same
/// quiescence contract applies (no in-flight INT8 CK dispatch on
/// any thread).
void clear_custom_kernel_pack_cache_int8();

} // namespace custom_kernel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_CUSTOM_KERNEL_PACK_HPP
