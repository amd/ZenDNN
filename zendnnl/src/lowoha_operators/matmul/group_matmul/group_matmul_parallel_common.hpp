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

/// Library-internal helpers shared by the ALGO 1..5 parallel paths.
///
/// Each ALGO implementation (`sequential_experts`, `flat_m_tile`,
/// `flat_n_tile`, `parallel_multilevel`, `parallel_per_expert`) is
/// split into its own translation unit so the files stay small and
/// ownership is clear.  This header hosts the bits they all need:
///   - env-driven feature flags
///   - the `resolve_kernel()` / `execute_expert_slice()` primitives
///   - tile-size constants referenced by ALGO 0 auto-select
///   - forward declarations of the per-strategy entry points
///
/// None of these symbols are part of the public ZenDNN API.

#ifndef ZENDNNL_GROUP_MATMUL_PARALLEL_COMMON_HPP
#define ZENDNNL_GROUP_MATMUL_PARALLEL_COMMON_HPP

#include <algorithm>
#include <cstdlib>
#include <vector>

#include "group_matmul_direct.hpp"
#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"
#include "lowoha_operators/common/omp_thread_control.hpp"
#include "operators/matmul/matmul_config.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

// Match the original group_matmul_parallel.cpp using-declarations so
// source files including this header can refer to matmul_algo_t,
// matmul_config_t, post_op_type_t, etc. without namespace prefixes.
using namespace zendnnl::ops;
using zendnnl::common::size_of;

// ──────────────────────────────────────────────────────────────────────
// Shared tile-size constants.
//
// Used by both ALGO 0 auto-select (to score ntile/mtile viability) and
// the tile kernels themselves (to decide per-thread min-work).
// ──────────────────────────────────────────────────────────────────────

/// Upper bound on per-expert M for a shape to be considered "decode".
/// Above this threshold ALGO 0 switches to prompt-oriented heuristics
/// and the tile kernels use different per-thread minimums.
inline constexpr int kDecodeMaxM = 32;

/// Minimum N-columns per thread in prompt-path N-tile.
inline constexpr int kMinNTile = 512;

/// Minimum N-columns per thread in decode-path N-tile.
inline constexpr int kDecodeNTile = 256;

// ──────────────────────────────────────────────────────────────────────
// Env-driven feature flags.
// ──────────────────────────────────────────────────────────────────────

/// ZENDNNL_GRP_MATMUL_ALGO: force a specific ALGO (1..5), or 0/unset
/// to let ALGO 0 auto-select.
///
/// Intentionally NOT cached — gtests toggle this mid-process via
/// AlgoEnvGuard (see test_group_matmul.cpp) to parameterize the
/// ALGO-under-test across test cases within a single binary run.
/// Per-call getenv is ~1μs, negligible vs the matmul it gates.
inline int get_grp_matmul_algo() {
  const char *env = std::getenv("ZENDNNL_GRP_MATMUL_ALGO");
  return (env && env[0] >= '1' && env[0] <= '5') ? (env[0] - '0') : 0;
}

// ──────────────────────────────────────────────────────────────────────
// ALGO 3 fused gated-activation epilogue
//
//   The N-tile path (flat_n_tile) can fold a gated activation into its
//   epilogue when the activation's layout keeps each thread's input
//   self-contained.  The per-thread compaction saves a second OMP pass
//   and can keep the matmul tile hot in L2 for small-M decode shapes.
//
//   Supported today:
//     swiglu_oai_mul  — interleaved (g, u) pair layout, with an
//                       epilogue that splits rows (not columns) across
//                       threads so the in-place N/2 compaction never
//                       aliases another thread's column range.  See
//                       apply_n_tile_paired_swiglu_oai in
//                       group_matmul_n_tile.cpp for the correctness
//                       argument.
//   Not fuseable in A3 today (still run as a separate post-pass):
//     silu_and_mul, gelu_and_mul — split-halves layout where gate and
//     up live in disjoint column ranges that a naive N-split does not
//     colocate on the same thread.
//
//   ZENDNNL_GRP_N_TILE_FUSED_ACT (env):
//     0 / unset  → fused epilogue disabled (default; caller does the
//                  activation in a separate OMP pass, matching the
//                  ALGO 3 behavior used by every framework integration
//                  today).
//     1          → opt in to the fused epilogue.  Correctness is
//                  protected by gtests that force this on (see the
//                  FusedActEnvGuard cases in test_group_matmul.cpp),
//                  but the default stays off until a framework/serving
//                  stack explicitly wires weight unpacking + dispatch
//                  policy to actually reach ALGO 3 with a supported
//                  activation.
// ──────────────────────────────────────────────────────────────────────

// Per-call lookup — intentionally NOT cached so AlgoEnvGuard-style
// tests (and env-driven A/B sweeps) can toggle the flag mid-process.
// Cost is a single getenv (~1 μs) per group_matmul call, negligible
// vs the GEMM it gates.
inline bool get_grp_n_tile_fused_act() {
  const char *env = std::getenv("ZENDNNL_GRP_N_TILE_FUSED_ACT");
  if (env == nullptr || env[0] == '\0') return false;
  return env[0] != '0';
}

/// Returns true when A3 can run a per-thread fused epilogue for `act`.
/// Add new layouts here as they become supported (e.g. a paired-split
/// kernel for silu_and_mul / gelu_and_mul in a follow-up PR).
inline bool a3_can_fuse_act(grp_matmul_gated_act_t act) {
  return act == grp_matmul_gated_act_t::swiglu_oai_mul;
}

// ──────────────────────────────────────────────────────────────────────
// ALGO 3 decode-shape tuning knobs (per-call, so tests / A-B sweeps can
// toggle mid-process without rebuilding).
//
//   ZENDNNL_GRP_N_FALLBACK_V1 (default: 1 — ON)
//     1 → when ntile_viable=false, run experts sequentially with
//         num_threads threads per GEMM (= ALGO 1 fallback).  Closes the
//         down_proj E=4 warmup cliff (pre-fix: 14x slower than ALGO 1;
//         post-fix: matches ALGO 1).  Aggregate win ~+2.1% on the
//         GPT-OSS decode profile (gpt_oss_moe_decode_profiled.txt, 128t,
//         MATMUL_ALGO=1; see algo3_tune/ comparison for per-shape data).
//     0 → legacy: 1 OMP thread per expert, num_ops experts in parallel.
//         Only leave off for A/B sanity.
//
//   ZENDNNL_GRP_N_DECODE_TILE_AB (default: 0 — OFF, opt-in)
//     1 → when max_M ≤ kDecodeMaxM, use kDecodeNTile (256) instead of
//         kMinNTile (512) as the per-thread N-tile bound in paths (A)
//         and (B).  Doubles max_n_thr so decode-shape down_proj at
//         num_ops 12..32 can use more threads; intended to help the
//         13 configs where V3 still trails V1 after FB_V1.
//         Measured effect on the GPT-OSS decode profile: mix of small
//         wins (E=13–15 down: -10%..-18%) and small regressions (E=24
//         down: +6..+9%) that roughly cancel (aggregate neutral,
//         -0.2%).  Kept as an opt-in knob so a future ALGO 0 auto-
//         select revision can turn it on selectively for the shapes
//         where it actually wins.
//     0 → legacy: use kMinNTile in (A)/(B) regardless of max_M.
// ──────────────────────────────────────────────────────────────────────
inline bool get_grp_n_fallback_v1() {
  const char *env = std::getenv("ZENDNNL_GRP_N_FALLBACK_V1");
  if (env == nullptr || env[0] == '\0') return true;
  return env[0] != '0';
}
inline bool get_grp_n_decode_tile_ab() {
  const char *env = std::getenv("ZENDNNL_GRP_N_DECODE_TILE_AB");
  if (env == nullptr || env[0] == '\0') return false;
  return env[0] != '0';
}

/// Aggregate L3 capacity used by the N-tile / multilevel L3-aware batch
/// planners.  Default is 512 MB which matches a typical Zen 4 EPYC
/// (8 CCDs × 32 MB + shared victim headroom).  Other SKUs (Zen 3,
/// Zen 5, Turin, embedded) have different L3, so allow an env override.
///
///   ZENDNNL_GRP_L3_TOTAL_MB = <positive integer>  (megabytes)
///
/// Cached once per process via C++11 magic-static.
inline size_t get_grp_l3_total_bytes() {
  static const size_t v = []() -> size_t {
    const char *env = std::getenv("ZENDNNL_GRP_L3_TOTAL_MB");
    if (env != nullptr && env[0] != '\0') {
      char *end = nullptr;
      long mb = std::strtol(env, &end, 10);
      if (end != env && mb > 0) {
        return static_cast<size_t>(mb) * 1024UL * 1024UL;
      }
    }
    // Default: Zen 4 EPYC aggregate (8 × 32 MB L3 + shared victim).
    return 512UL * 1024UL * 1024UL;
  }();
  return v;
}

// ──────────────────────────────────────────────────────────────────────
// Shared per-expert primitives.
// ──────────────────────────────────────────────────────────────────────

/// Resolves the effective matmul algo ID from the runtime config,
/// falling back to AOCL DLP blocked when the config is unset/invalid.
inline matmul_algo_t resolve_kernel() {
  static const matmul_algo_t algo = []() {
    int32_t a = matmul_config_t::instance().get_algo();
    if (a <= 0 || a >= static_cast<int32_t>(matmul_algo_t::algo_count))
      return matmul_algo_t::aocl_dlp_blocked;
    return static_cast<matmul_algo_t>(a);
  }();
  return algo;
}

/// Thin wrapper around matmul_execute that packages per-expert slice
/// arguments into the batch/params objects the kernel expects.
inline void execute_expert_slice(
    char layout, bool transA, bool transB,
    int M, int N, int K, float alpha,
    const void *src, int lda,
    const void *weight, int ldb,
    const void *bias, float beta,
    void *dst, int ldc,
    bool is_weights_const, int num_thr,
    matmul_params &params,
    matmul_algo_t algo) {

  matmul_batch_params_t bp;
  bp.Batch_A = 1;
  bp.Batch_B = 1;
  matmul_algo_t kernel = algo;
  matmul_execute(layout, transA, transB,
      M, N, K, alpha, src, lda, weight, ldb, bias, beta, dst, ldc,
      is_weights_const, size_of(params.dtypes.src),
      size_of(params.dtypes.dst),
      num_thr, kernel, params, bp, 0);
}

// ──────────────────────────────────────────────────────────────────────
// Tile-strategy entry points (defined in their own .cpp files).
//
// These functions have external (library-internal) linkage so that
// group_matmul_parallel.cpp (dispatcher) can call them.  They are NOT
// part of any public header; they remain library-private.
// ──────────────────────────────────────────────────────────────────────

/// ALGO 2 — M-tile parallel GEMM.  Defined in group_matmul_m_tile.cpp.
void flat_m_tile(
    const std::vector<char> &layout,
    const std::vector<bool> &transA, const std::vector<bool> &transB,
    const std::vector<int> &M, const std::vector<int> &N,
    const std::vector<int> &K, const std::vector<float> &alpha,
    const std::vector<const void *> &src, const std::vector<int> &lda,
    const std::vector<const void *> &weight, const std::vector<int> &ldb,
    const std::vector<const void *> &bias, const std::vector<float> &beta,
    const std::vector<void *> &dst, const std::vector<int> &ldc,
    grp_matmul_gated_act_t fused_act, data_type_t act_dtype,
    const std::vector<bool> &is_weights_const,
    std::vector<matmul_params> &params,
    int num_threads);

/// ALGO 3 — N-tile parallel GEMM, with optional fused-swiglu-oai
/// epilogue.  Defined in group_matmul_n_tile.cpp.
void flat_n_tile(
    const std::vector<char> &layout,
    const std::vector<bool> &transA, const std::vector<bool> &transB,
    const std::vector<int> &M, const std::vector<int> &N,
    const std::vector<int> &K, const std::vector<float> &alpha,
    const std::vector<const void *> &src, const std::vector<int> &lda,
    const std::vector<const void *> &weight, const std::vector<int> &ldb,
    const std::vector<const void *> &bias, const std::vector<float> &beta,
    const std::vector<void *> &dst, const std::vector<int> &ldc,
    const std::vector<bool> &is_weights_const,
    std::vector<matmul_params> &params,
    int num_threads,
    grp_matmul_gated_act_t fused_act = grp_matmul_gated_act_t::none,
    data_type_t act_dtype = data_type_t::none);

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif  // ZENDNNL_GROUP_MATMUL_PARALLEL_COMMON_HPP
