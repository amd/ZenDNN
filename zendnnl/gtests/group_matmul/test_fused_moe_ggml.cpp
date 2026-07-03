/********************************************************************************
# * Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

/// @file test_fused_moe_ggml.cpp
/// @brief Fused-MoE (Op1 → gated act → Op2 in one call) with GGML Q8_0
///        block-quantized weights on BOTH projections.
///
/// `params[i].packing.pack_format_b == 1` marks an expert's gate/up weight
/// (`weight[i]`) AND its down weight (`fused.down_weight[i]`) as GGML-packed;
/// `group_matmul_direct` unpacks + AOCL sym-quant-reorders both per active
/// expert, then runs the per-group dynamic-quant two-pass.
///
/// Strategy: the fused single call is compared against a 2-call **non-fused**
/// GGML reference (Op1+act, then Op2) built from the SAME packed weights.  That
/// isolates the new fused wiring (both-weights unpack + Op2 scale/format
/// plumbing); the GGML unpack's numerical correctness vs an f32 reference is
/// covered separately by test_ggml_per_group.cpp.  Sparse routing (inactive
/// experts with M=0) and several gated activations are exercised.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "gtest_utils.hpp"
#include "moe_test_utils.hpp"
#include "lowoha_operators/matmul/ggml_weight_unpack.hpp"

namespace {

using namespace moe_test_utils;
using zendnnl::lowoha::matmul::matmul_params;
using zendnnl::lowoha::matmul::grp_matmul_gated_act_t;
using zendnnl::lowoha::matmul::grp_matmul_gated_act_params;
using zendnnl::lowoha::matmul::grp_matmul_fused_moe_params;
using zendnnl::lowoha::matmul::group_matmul_direct;
using zendnnl::error_handling::status_t;
using data_type_t = zendnnl::common::data_type_t;

constexpr int kBlk = 32;          // GGML Q8_0 group size
constexpr size_t kBlkBytes = 34;  // fp16 scale + 32 int8 per block

// Pack a plain s8 [K, N] weight (+ its per-group {K/32, N} bf16 scale) into the
// GGML Q8_0 (N-major) byte layout the API unpacks.  Mirrors the single-matmul
// INT8_PER_GROUP_GGML_PACKED packing.
void pack_q8_0_from_s8(const int8_t *s8_kn, const uint16_t *scale_bf16,
                       int64_t K, int64_t N, std::vector<uint8_t> &out) {
  const int64_t ng = K / kBlk;
  std::vector<int8_t> wt_nk(static_cast<size_t>(K) * N);
  for (int64_t ki = 0; ki < K; ++ki)
    for (int64_t ni = 0; ni < N; ++ni)
      wt_nk[ni * K + ki] = s8_kn[ki * N + ni];
  std::vector<float> scl(static_cast<size_t>(ng) * N);
  for (size_t i = 0; i < scl.size(); ++i) {
    uint32_t bits = static_cast<uint32_t>(scale_bf16[i]) << 16;  // bf16 -> f32
    std::memcpy(&scl[i], &bits, sizeof(float));
  }
  out.resize(static_cast<size_t>(N) * ng * kBlkBytes);
  repack_weights_q8_0(wt_nk.data(), scl.data(), N, K, out.data());
}

// Build the per-group dynamic-INT8 + GGML-packed params for one GEMM.
// src bf16 → s8 (dynamic, per-group {M, K/32}); weight GGML-packed s8.
matmul_params make_ggml_dyn_params(int M, int K) {
  matmul_params p;
  p.dtypes.src     = data_type_t::bf16;
  p.dtypes.wei     = data_type_t::s8;
  p.dtypes.dst     = data_type_t::bf16;
  p.dtypes.compute = data_type_t::s8;
  p.dynamic_quant  = true;
  p.packing.pack_format_b = 1;  // GGML packed (both Op1 & Op2 by contract)
  p.num_threads = 0;
  p.quant_params.src_scale.buff = nullptr;  // dynamic: group-DQ allocates
  // Must match the GGML weight scale dtype (bf16) — the AOCL sym-quant GEMM
  // requires the A (activation) and B (weight) scale factor types to agree.
  p.quant_params.src_scale.dt   = data_type_t::bf16;
  // Per-group, group_size 32; dims[0] must equal this expert's row count.
  p.quant_params.src_scale.dims = {M, K / kBlk};
  return p;
}

/// One scenario: build E experts with GGML-packed gate/up + down weights, run
/// the fused single call, and compare against the 2-call non-fused GGML
/// reference.  `rows[e] == 0` => inactive expert (no routed tokens).
///
/// `use_vertical_fusion` selects how the fused call is dispatched:
///   * false (default) — ALGO 1 full-N AOCL sym-quant two-pass (the original
///     coverage: exercises the fused unpack + Op2 scale/format plumbing).
///   * true            — ALGO 2 + forced vertical fusion (M-tile pipeline).
///     The per-group / GGML-reordered weights flow through the single-pass
///     executor; the test additionally asserts the capture tag is
///     `kVerticalFusionDQINT8` (the executor really engaged, not a silent
///     fall-back) and still checks the result against the ALGO-1 reference.
void run_fused_ggml_scenario(const std::string &label,
                             const std::vector<int> &rows, int H, int dim,
                             int act_int, bool use_vertical_fusion = false) {
  ASSERT_EQ(H % kBlk, 0) << label << ": H must be a multiple of 32";
  ASSERT_GE(H / kBlk, 2) << label;
  const int N_gate_up = 2 * dim;       // gate + up
  const int K_down = dim;              // gated activation halves Op1 output
  ASSERT_EQ(K_down % kBlk, 0) << label << ": dim must be a multiple of 32";
  ASSERT_GE(K_down / kBlk, 2) << label;

  const auto act = static_cast<grp_matmul_gated_act_t>(act_int);
  ASSERT_NE(act, grp_matmul_gated_act_t::none) << label;

  const int E = static_cast<int>(rows.size());
  int M_buf = 1;
  for (int r : rows) M_buf = std::max(M_buf, r);

  // Independent weight set per scenario (the GGML reorder cache is keyed by
  // weight pointer, and these buffers are freed on return).
  zendnnl::lowoha::matmul::clear_ggml_weight_unpack_cache();

  tensor_factory_t factory;
  const data_type_t dt = data_type_t::bf16;       // fused Op2 quant is bf16-only
  const data_type_t scale_dt = data_type_t::bf16;

  std::vector<tensor_t> w1_s8(E), w1_scale(E), w1_zp(E);
  std::vector<tensor_t> w2_s8(E), w2_scale(E), w2_zp(E);
  std::vector<tensor_t> src_t(E), w1_pk_t(E), w2_pk_t(E);
  std::vector<std::vector<uint8_t>> w1_pk(E), w2_pk(E);
  std::vector<std::vector<uint8_t>> w1_before(E), w2_before(E);
  std::vector<const void *> srcs(E), w1_pk_p(E), w2_pk_p(E);

  // uniform_dist_tensor seeds from the global `seed`; bump it per expert for
  // independent data, then restore so the mutation does not leak into other
  // tests in the process.
  const int64_t saved_seed = seed;

  for (int e = 0; e < E; ++e) {
    // Independent data per expert so a routing/weight mix-up is detectable.
    seed = saved_seed + 1 + e;

    auto w1_ref = factory.uniform_dist_tensor(
        {static_cast<uint64_t>(H), static_cast<uint64_t>(N_gate_up)}, dt, 2.0);
    ASSERT_EQ(quant_params_compute(factory, w1_ref, dt, data_type_t::s8,
                                   {H / kBlk, N_gate_up}, scale_dt,
                                   w1_scale[e], w1_zp[e], &w1_s8[e]),
              status_t::success)
        << label << ": w1 quant (e=" << e << ")";
    pack_q8_0_from_s8(
        static_cast<const int8_t *>(w1_s8[e].get_raw_handle_unsafe()),
        static_cast<const uint16_t *>(w1_scale[e].get_raw_handle_unsafe()), H,
        N_gate_up, w1_pk[e]);
    w1_pk_t[e] = factory.copy_tensor(
        {static_cast<uint64_t>(H), static_cast<uint64_t>(N_gate_up)},
        data_type_t::s8,
        std::make_pair(w1_pk[e].size(), static_cast<void *>(w1_pk[e].data())),
        /*trans=*/true, /*is_blocked=*/false);

    auto w2_ref = factory.uniform_dist_tensor(
        {static_cast<uint64_t>(K_down), static_cast<uint64_t>(H)}, dt, 2.0);
    ASSERT_EQ(quant_params_compute(factory, w2_ref, dt, data_type_t::s8,
                                   {K_down / kBlk, H}, scale_dt, w2_scale[e],
                                   w2_zp[e], &w2_s8[e]),
              status_t::success)
        << label << ": w2 quant (e=" << e << ")";
    pack_q8_0_from_s8(
        static_cast<const int8_t *>(w2_s8[e].get_raw_handle_unsafe()),
        static_cast<const uint16_t *>(w2_scale[e].get_raw_handle_unsafe()),
        K_down, H, w2_pk[e]);
    w2_pk_t[e] = factory.copy_tensor(
        {static_cast<uint64_t>(K_down), static_cast<uint64_t>(H)},
        data_type_t::s8,
        std::make_pair(w2_pk[e].size(), static_cast<void *>(w2_pk[e].data())),
        /*trans=*/true, /*is_blocked=*/false);

    src_t[e] = factory.uniform_dist_tensor(
        {static_cast<uint64_t>(M_buf), static_cast<uint64_t>(H)}, dt, 2.0);

    srcs[e]    = src_t[e].get_raw_handle_unsafe();
    w1_pk_p[e] = w1_pk_t[e].get_raw_handle_unsafe();
    w2_pk_p[e] = w2_pk_t[e].get_raw_handle_unsafe();
    w1_before[e].assign(w1_pk[e].begin(), w1_pk[e].end());
    w2_before[e].assign(w2_pk[e].begin(), w2_pk[e].end());
  }
  seed = saved_seed;  // restore global RNG seed

  const std::vector<int> Ms = rows;
  std::vector<char> layout(E, 'r');
  std::vector<bool> transA(E, false), transB(E, true), is_wc(E, true);
  std::vector<float> alpha(E, 1.0f), beta(E, 0.0f);
  std::vector<const void *> no_bias(E, nullptr);

  TypedBuffers d1_ref, d2_ref, d1_test, d2_test;
  d1_ref .alloc(E, static_cast<size_t>(M_buf) * N_gate_up, /*is_bf16=*/true);
  d2_ref .alloc(E, static_cast<size_t>(M_buf) * H,         true);
  d1_test.alloc(E, static_cast<size_t>(M_buf) * N_gate_up, true);
  d2_test.alloc(E, static_cast<size_t>(M_buf) * H,         true);

  grp_matmul_gated_act_params act_params{};
  act_params.act = act;

  // ── Reference: 2 separate non-fused GGML calls (Op1+act, then Op2) ──
  {
    // The per-group GGML reference always runs the full-N AOCL sym-quant
    // two-pass (ALGO 1) — the known-good baseline the fused result (whether
    // two-pass or vertical-fusion) is verified against.
    AlgoEnvGuard ref_algo(1);
    // Op1 (gate/up) + gated activation.
    std::vector<matmul_params> p1(E);
    for (int e = 0; e < E; ++e) p1[e] = make_ggml_dyn_params(Ms[e], H);
    std::vector<int> Ns(E, N_gate_up), Ks(E, H), lda(E, H), ldb(E, H),
        ldc(E, N_gate_up);
    auto d1_p = d1_ref.ptrs(true);
    ASSERT_EQ(group_matmul_direct(layout, transA, transB, Ms, Ns, Ks, alpha,
                                  srcs, lda, w1_pk_p, ldb, no_bias, beta, d1_p,
                                  ldc, is_wc, p1, nullptr, &act_params),
              status_t::success)
        << label << ": ref Op1";

    // Op2 (down): source is the activated Op1 output [M, K_down] read at the
    // gate+up stride.
    std::vector<matmul_params> p2(E);
    for (int e = 0; e < E; ++e) p2[e] = make_ggml_dyn_params(Ms[e], K_down);
    std::vector<int> Ns2(E, H), Ks2(E, K_down), lda2(E, N_gate_up),
        ldb2(E, K_down), ldc2(E, H);
    std::vector<const void *> srcs2(E);
    auto d1_cp = d1_ref.cptrs(true);
    for (int e = 0; e < E; ++e) srcs2[e] = d1_cp[e];
    auto d2_p = d2_ref.ptrs(true);
    ASSERT_EQ(group_matmul_direct(layout, transA, transB, Ms, Ns2, Ks2, alpha,
                                  srcs2, lda2, w2_pk_p, ldb2, no_bias, beta,
                                  d2_p, ldc2, is_wc, p2),
              status_t::success)
        << label << ": ref Op2";
  }

  // ── Test: single fused call (Op1 → act → Op2) with GGML weights ──
  int mtile_tag = 0;
  {
    // Default: ALGO 1 two-pass.  Opt-in: ALGO 2 + FORCED vertical fusion with
    // a generous per-thread scratch budget so the GGML / per-group M-tile
    // pipeline engages on these shapes.  All guards restore on scope exit.
    std::unique_ptr<AlgoEnvGuard> fused_algo;
    std::unique_ptr<moe_test_utils::MoEVerticalFusionOverride> vf_guard;
    std::unique_ptr<moe_test_utils::MoEPipelineScratchKbOverride> scratch_guard;
    if (use_vertical_fusion) {
      fused_algo = std::make_unique<AlgoEnvGuard>(2);
      vf_guard =
          std::make_unique<moe_test_utils::MoEVerticalFusionOverride>(1);
      scratch_guard =
          std::make_unique<moe_test_utils::MoEPipelineScratchKbOverride>(1024);
    } else {
      fused_algo = std::make_unique<AlgoEnvGuard>(1);
    }

    std::vector<matmul_params> pt(E);
    for (int e = 0; e < E; ++e) {
      pt[e] = make_ggml_dyn_params(Ms[e], H);
      // Pin the team size so the M-tile planner's wide-N / round-based gates
      // resolve deterministically regardless of host core count: the VF
      // scenarios below are sized so total_need > num_threads/2 at 32 threads
      // (same rationale as the per-token VF fixtures in test_fused_moe.cpp).
      // The Op2 (down) params inherit this via build_op2_dispatch_params.
      if (use_vertical_fusion) pt[e].num_threads = 32;
    }
    std::vector<int> Ns(E, N_gate_up), Ks(E, H), lda(E, H), ldb(E, H),
        ldc(E, N_gate_up);

    auto fused = make_fused_moe_op2(E, H, w2_pk_p, no_bias);
    fused.dst_down = d2_test.ptrs(true);
    fused.ldc_down = std::vector<int>(E, H);
    fused.ldb_down = std::vector<int>(E, K_down);  // transB=true => ldb >= K_down
    // down_scale intentionally left empty: GGML scales are embedded in the
    // packed blocks; group_matmul_direct fills the unpacked {K_down/32, H}
    // scale per expert.

    auto d1_p = d1_test.ptrs(true);
    moe_test_utils::MTilePathCaptureGuard cap;
    ASSERT_EQ(group_matmul_direct(layout, transA, transB, Ms, Ns, Ks, alpha,
                                  srcs, lda, w1_pk_p, ldb, no_bias, beta, d1_p,
                                  ldc, is_wc, pt, /*moe_postop=*/nullptr,
                                  &act_params, &fused),
              status_t::success)
        << label << ": fused call";
    mtile_tag = zendnnl::lowoha::matmul::test_api::s_last_m_tile_path.load(
        std::memory_order_relaxed);
  }

  // The opt-in path MUST have engaged the DQ-INT8 vertical-fusion executor on
  // the GGML / per-group weights — a silent fall-back to the two-pass would
  // still produce a correct result (and pass the numeric check below) but
  // defeat the single-pass fusion this scenario is here to cover.
  if (use_vertical_fusion) {
    EXPECT_EQ(mtile_tag,
              zendnnl::lowoha::matmul::test_api::m_tile_path_tag
                  ::kVerticalFusionDQINT8)
        << label << ": per-group GGML vertical fusion did NOT engage — "
        << "capture tag = " << mtile_tag << " (expected kVerticalFusionDQINT8 = "
        << zendnnl::lowoha::matmul::test_api::m_tile_path_tag
               ::kVerticalFusionDQINT8
        << ")";
  }

  // ── Sanity: outputs are non-trivial (catch an all-zero short-circuit) ──
  double ref_sum = 0.0, test_sum = 0.0;
  for (int e = 0; e < E; ++e) {
    if (Ms[e] == 0) continue;
    for (int i = 0; i < Ms[e] * H; ++i) {
      ref_sum  += std::abs(static_cast<float>(d2_ref.bf16[e][i]));
      test_sum += std::abs(static_cast<float>(d2_test.bf16[e][i]));
    }
  }
  bool any_active = false;
  for (int r : rows) any_active = any_active || (r > 0);
  if (any_active) {
    ASSERT_GT(ref_sum, 1e-3) << label << ": reference produced all-zero output";
    ASSERT_GT(test_sum, 1e-3) << label << ": fused produced all-zero output";
  }

  // ── Fused output must match the 2-call reference on every routed expert ──
  verify_per_expert_2d(d2_test, static_cast<size_t>(H), d2_ref,
                       static_cast<size_t>(H), Ms, H, /*is_bf16=*/true,
                       tol_fused(true), label);

  // ── Warm-all: the fused call must have unpacked + cached EVERY expert's
  //    gate/up AND down weight (2 entries per expert), not just the routed
  //    ones — so a later iteration that routes to a now-cold expert is a
  //    cache hit instead of a first-fire unpack spike.  (Every expert here
  //    owns valid const weights; the 2-call reference above only warmed the
  //    active subset via the active-only non-fused path, so reaching 2*E
  //    proves the fused call warmed the inactive experts too.) ──
  EXPECT_EQ(zendnnl::lowoha::matmul::ggml_weight_unpack_cache_size(),
            static_cast<size_t>(2 * E))
      << label << ": expected all " << E
      << " experts warmed (gate/up + down = " << (2 * E) << " cache entries)";

  // ── GGML packed weights are read-only on both passes ──
  for (int e = 0; e < E; ++e) {
    EXPECT_EQ(std::memcmp(w1_before[e].data(),
                          w1_pk_t[e].get_raw_handle_unsafe(),
                          w1_before[e].size()),
              0)
        << label << ": gate/up packed bytes mutated (e=" << e << ")";
    EXPECT_EQ(std::memcmp(w2_before[e].data(),
                          w2_pk_t[e].get_raw_handle_unsafe(),
                          w2_before[e].size()),
              0)
        << label << ": down packed bytes mutated (e=" << e << ")";
  }
}

}  // namespace

// Dense: every expert routed, silu gate.
TEST(FusedMoEGgml, DenseSiluBF16) {
  run_fused_ggml_scenario("dense/4 silu", std::vector<int>(4, 16),
                          /*H=*/64, /*dim=*/64, /*act=*/1);
}

// Sparse routing: 8 experts, only 4 fire (interleaved) — the MoE decode case
// that exercises the M==0 skip for both the unpack and the GEMM.
TEST(FusedMoEGgml, SparseSiluBF16) {
  std::vector<int> rows(8, 0);
  for (int e : {1, 3, 4, 6}) rows[e] = 16;
  run_fused_ggml_scenario("8/4 sparse silu", rows, 64, 64, 1);
}

// Sparse with expert 0 inactive (stresses prepack representative skip).
TEST(FusedMoEGgml, SparseLastActiveGeluBF16) {
  std::vector<int> rows(6, 0);
  for (int e = 3; e < 6; ++e) rows[e] = 16;
  run_fused_ggml_scenario("6/3 last gelu", rows, 64, 64, 2);
}

// SwiGLU-OAI gate, dense.
TEST(FusedMoEGgml, DenseSwigluBF16) {
  run_fused_ggml_scenario("dense/4 swiglu", std::vector<int>(4, 16), 64, 64, 3);
}

// Ragged per-expert token counts (incl. a single inactive) + larger groups.
TEST(FusedMoEGgml, VariedTokensSiluBF16) {
  std::vector<int> rows = {16, 0, 32, 16, 0, 24};
  run_fused_ggml_scenario("varied silu", rows, /*H=*/128, /*dim=*/64, 1);
}

// Single routed expert in the middle of an otherwise-cold layer.
TEST(FusedMoEGgml, SingleActiveSiluBF16) {
  std::vector<int> rows(8, 0);
  rows[5] = 16;
  run_fused_ggml_scenario("8/1 single silu", rows, 64, 64, 1);
}

// ── Vertical-fusion (ALGO 2, M-tile single-pass) coverage ──────────────────
// Same scenarios as above, but the fused call is forced through the
// per-group / GGML-reordered vertical-fusion pipeline instead of the ALGO-1
// two-pass.  Each asserts (a) the capture tag is `kVerticalFusionDQINT8` (the
// executor engaged) AND (b) the result matches the ALGO-1 two-pass reference.

// Dense: every expert routed, silu gate — the prompt-class M-tile case.
TEST(FusedMoEGgml, VerticalFusionDenseSiluBF16) {
  run_fused_ggml_scenario("VF dense/4 silu", std::vector<int>(4, 128),
                          /*H=*/64, /*dim=*/64, /*act=*/1,
                          /*use_vertical_fusion=*/true);
}

// Sparse routing: 8 experts, only 4 fire — vertical fusion must engage on the
// active subset while the warm-all still caches every expert.  M=128 keeps
// total_need = 4·ceil(128/16) = 32 > 16 (clears the wide-N gate at 32 threads).
TEST(FusedMoEGgml, VerticalFusionSparseSiluBF16) {
  std::vector<int> rows(8, 0);
  for (int e : {1, 3, 4, 6}) rows[e] = 128;
  run_fused_ggml_scenario("VF 8/4 sparse silu", rows, 64, 64, 1,
                          /*use_vertical_fusion=*/true);
}

// Larger K (4 groups per row at group_size 32) + gelu gate.
TEST(FusedMoEGgml, VerticalFusionVariedGeluBF16) {
  std::vector<int> rows = {128, 0, 96, 128};
  run_fused_ggml_scenario("VF varied gelu", rows, /*H=*/128, /*dim=*/128,
                          /*act=*/2, /*use_vertical_fusion=*/true);
}

// SwiGLU-OAI gate, dense — exercises the third fused-activation arm.
TEST(FusedMoEGgml, VerticalFusionDenseSwigluBF16) {
  run_fused_ggml_scenario("VF dense/4 swiglu", std::vector<int>(4, 128), 64, 64,
                          3, /*use_vertical_fusion=*/true);
}
