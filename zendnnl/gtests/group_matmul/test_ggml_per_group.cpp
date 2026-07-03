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

/// @file test_ggml_per_group.cpp
/// @brief Grouped (MoE) symmetric INT8 per-group matmul fed GGML Q8_0 packed
///        weights — the group-matmul analogue of the single-matmul
///        `INT8_PER_GROUP_GGML_PACKED` test (`test_matmul.cpp`).
///
/// Each expert owns:
///   * a pre-quantized s8 source `[M, K]` with a per-group `{M, K/32}`
///     src_scale (group_size fixed at 32 to pair with GGML Q8_0 blocks), and
///   * a GGML Q8_0 block-quantized weight (`pack_format_b = 1`) that the API
///     unpacks per-expert into a cached AOCL sym-quant-reordered s8 buffer +
///     per-group `{K/32, N}` wei_scale before the grouped GEMM.
///
/// The suite drives `group_matmul_direct` with many experts (15) and a sparse
/// active set (e.g. only 6 routed), modelling one MoE decode iteration.
/// Inactive experts carry `M == 0` (no routed tokens): the GEMM skips them and
/// the per-expert GGML unpack leaves their packed bytes untouched.  Routed
/// experts are validated bit-for-bit-tolerantly against the unpacked s8
/// reference, and every expert's packed bytes must survive the call unchanged
/// (the unpack/reorder is out-of-place and cached).

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "gtest_utils.hpp"
#include "group_matmul_test_helpers.hpp"
#include "lowoha_operators/matmul/ggml_weight_unpack.hpp"

namespace {

constexpr int kGgmlBlock = 32;          // GGML Q8_0 group size
constexpr size_t kGgmlBlockBytes = 34;  // fp16 scale + 32 int8 per block

/// Build every expert with full-size GGML-packed per-group INT8 weights and a
/// pre-quantized s8 per-group source, then drive `group_matmul_direct` with the
/// supplied per-expert active row counts (`rows[e] == 0` => inactive expert /
/// no routed tokens).  Routed experts are compared against the unpacked s8
/// reference; packed bytes must survive the call untouched.
void run_ggml_per_group_scenario(const std::string &label,
                                 const std::vector<int> &rows, uint64_t K,
                                 uint64_t N, bool use_bf16_out) {
  ASSERT_EQ(K % kGgmlBlock, 0u) << label << ": K must be a multiple of 32";
  const uint64_t ng = K / kGgmlBlock;
  ASSERT_GE(ng, 2u) << label << ": need >= 2 groups for per-group scaling";

  const int E = static_cast<int>(rows.size());
  ASSERT_GT(E, 0) << label;

  // The GGML reorder cache is keyed by the weight *pointer*; this test frees
  // each scenario's packed buffers on return, so a later scenario can reuse a
  // freed address and hit a stale entry.  Real frameworks keep weights
  // resident (no reuse), so clear the cache to model an independent weight set
  // per scenario.
  zendnnl::lowoha::matmul::clear_ggml_weight_unpack_cache();

  const matmul_algo_t algo = matmul_algo_t::aocl_dlp_blocked;
  const data_type_t ref_dt = use_bf16_out ? data_type_t::bf16 : data_type_t::f32;
  const data_type_t out_dt = ref_dt;
  // Bias dtype must be uniform across experts (m_tile_safe gate).
  const data_type_t bias_dt = ref_dt;
  const data_type_t scale_dt = data_type_t::bf16;
  const std::vector<int64_t> wei_sd = {static_cast<int64_t>(ng),
                                       static_cast<int64_t>(N)};

  tensor_factory_t tf;

  std::vector<tensor_t> inp(E), wpacked(E), wref(E), bias(E), out(E),
      out_ref(E);
  // `copy_tensor` aliases (does not copy) the byte buffer it is handed, so the
  // packed storage must outlive the whole call — keep it here rather than in
  // a per-expert loop local.
  std::vector<std::vector<uint8_t>> packed_storage(E);
  std::vector<std::vector<uint8_t>> packed_before(E);
  std::vector<int> pack_fmt(E, 1);  // every expert owns GGML-packed weights
  std::vector<int> active(E);

  // Give each expert independent data (so a routing mix-up that feeds an
  // expert the wrong weight is caught) while keeping the magnitude fixed and
  // moderate — the per-element quant noise then stays inside the comparator's
  // fixed accumulation bound regardless of expert count.  `uniform_dist_tensor`
  // seeds its RNG from the global `seed`, so bump it per expert and restore it
  // once the inputs are built.
  const int64_t saved_seed = seed;
  const float wval = 2.0f;   // matches the single-matmul GGML packed test
  const float sval = 25.0f;

  for (int e = 0; e < E; ++e) {
    active[e] = rows[e];
    seed = saved_seed + 1 + static_cast<int64_t>(e);
    // Inactive experts still own a (tiny) valid source so quantization
    // succeeds; `active[e] == 0` is what tells the GEMM to process no rows.
    const uint64_t Mbuf = static_cast<uint64_t>(rows[e] > 0 ? rows[e] : 4);

    // ── Reference (unpacked) weight: s8 [K, N] + per-group {ng, N} scale ──
    auto wei_ref_f = tf.uniform_dist_tensor({K, N}, ref_dt, wval, false);
    tensor_t weight_tensor, wei_scale, wei_zp;
    ASSERT_EQ(quant_params_compute(tf, wei_ref_f, ref_dt, data_type_t::s8,
                                   wei_sd, scale_dt, wei_scale, wei_zp,
                                   &weight_tensor),
              status_t::success)
        << label << ": weight quant failed (expert " << e << ")";
    wref[e] = weight_tensor;

    // ── Pack that same weight into GGML Q8_0 (N-major) ──
    const int8_t *raw_wt =
        static_cast<const int8_t *>(weight_tensor.get_raw_handle_unsafe());
    const uint16_t *raw_scl =
        static_cast<const uint16_t *>(wei_scale.get_raw_handle_unsafe());
    const size_t num_scales = static_cast<size_t>(ng * N);
    std::vector<float> scl_f32(num_scales);
    for (size_t i = 0; i < num_scales; ++i) {
      uint32_t bits = static_cast<uint32_t>(raw_scl[i]) << 16;  // bf16 -> f32
      std::memcpy(&scl_f32[i], &bits, sizeof(float));
    }
    std::vector<int8_t> wt_nk(static_cast<size_t>(K) * N);
    for (uint64_t ki = 0; ki < K; ++ki)
      for (uint64_t ni = 0; ni < N; ++ni)
        wt_nk[ni * K + ki] = raw_wt[ki * N + ni];
    const size_t packed_size = static_cast<size_t>(N) * ng * kGgmlBlockBytes;
    packed_storage[e].resize(packed_size);
    repack_weights_q8_0(wt_nk.data(), scl_f32.data(),
                        static_cast<int64_t>(N), static_cast<int64_t>(K),
                        packed_storage[e].data());
    wpacked[e] = tf.copy_tensor(
        {K, N}, data_type_t::s8,
        std::make_pair(packed_size,
                       static_cast<void *>(packed_storage[e].data())),
        /*trans=*/true, /*is_blocked=*/false);
    packed_before[e].resize(packed_size);
    std::memcpy(packed_before[e].data(), wpacked[e].get_raw_handle_unsafe(),
                packed_size);

    // ── Pre-quantized s8 source + per-group {Mbuf, ng} scale ──
    const std::vector<int64_t> src_sd = {static_cast<int64_t>(Mbuf),
                                         static_cast<int64_t>(ng)};
    auto src_ref_f = tf.uniform_dist_tensor({Mbuf, K}, ref_dt, sval, false);
    tensor_t input_tensor, src_scale, src_zp;
    ASSERT_EQ(quant_params_compute(tf, src_ref_f, ref_dt, data_type_t::s8,
                                   src_sd, scale_dt, src_scale, src_zp,
                                   &input_tensor),
              status_t::success)
        << label << ": source quant failed (expert " << e << ")";
    inp[e] = input_tensor;

    bias[e] = tf.uniform_dist_tensor({1, N}, bias_dt, 2.0);
    out[e] = tf.uniform_dist_tensor({Mbuf, N}, out_dt, 2.0);
    out_ref[e] = tf.uniform_dist_tensor({Mbuf, N}, out_dt, 2.0);
  }
  seed = saved_seed;  // restore global RNG seed for subsequent tests

  // ── Drive the grouped per-group GGML path ──
  status_t st = group_matmul_kernel_test(inp, wpacked, bias, out, algo, 1.0f,
                                         0.0f, /*moe_postop=*/nullptr,
                                         /*gated_act=*/nullptr, pack_fmt,
                                         active);
  ASSERT_EQ(st, status_t::success) << label << ": group_matmul_direct failed";

  // ── Compare every routed expert against the unpacked s8 reference ──
  const std::vector<post_op_type_t> ref_po;
  for (int e = 0; e < E; ++e) {
    if (rows[e] == 0) continue;  // inactive -> nothing computed
    std::vector<tensor_t> bin;
    status_t rst = matmul_forced_ref_kernel_test(inp[e], wref[e], bias[e],
                                                 out_ref[e], ref_po, bin,
                                                 /*use_LOWOHA=*/true, algo,
                                                 1.0f, 0.0f);
    ASSERT_EQ(rst, status_t::success)
        << label << ": reference failed (expert " << e << ")";
    bool expert_ok = true;
    // INT8 sym-quant comparison tolerance: the int8-GEMM-vs-f32-ref rounding
    // gap is dwarfed by the per-element accumulation bound for a bf16 dst, but
    // an f32 dst exposes it fully and a small fraction of cancellation-heavy
    // elements exceed the 1x bound once many experts are summed.  Use the same
    // 18x epsilon margin the INT8 sym-quant matmul cache tests rely on (see
    // test_postop_cache.cpp).
    compare_tensor_2D_matrix(out[e], out_ref[e], static_cast<uint64_t>(rows[e]),
                             N, K, rtol_bf16, 18.0f * epsilon_bf16, expert_ok,
                             /*enable_f32_relaxation=*/false, 1.0f);
    EXPECT_TRUE(expert_ok) << label << ": output mismatch on expert " << e
                           << " (rows=" << rows[e] << ")";
  }

  // ── GGML packed weights are read-only: the out-of-place unpack/reorder
  //    must never mutate the caller's packed bytes (active OR inactive). ──
  for (int e = 0; e < E; ++e) {
    EXPECT_EQ(std::memcmp(packed_before[e].data(),
                          wpacked[e].get_raw_handle_unsafe(),
                          packed_before[e].size()),
              0)
        << label << ": packed weight bytes mutated on expert " << e;
  }
}

}  // namespace

// Headline scenario: 15 experts, only 6 routed (interleaved) — one MoE decode
// iteration that fires 6 of 15 experts.
TEST(GroupMatmulGgmlPerGroup, FifteenExpertsSixActiveInterleavedBF16) {
  std::vector<int> rows(15, 0);
  for (int e : {1, 3, 5, 8, 11, 14}) rows[e] = 32;
  run_ggml_per_group_scenario("15/6 interleaved bf16", rows, /*K=*/128,
                              /*N=*/64, /*bf16=*/true);
}

// First 6 experts routed, F32 accumulation.
TEST(GroupMatmulGgmlPerGroup, FifteenExpertsSixActiveContiguousFirstF32) {
  std::vector<int> rows(15, 0);
  for (int e = 0; e < 6; ++e) rows[e] = 24;
  run_ggml_per_group_scenario("15/6 first-6 f32", rows, 64, 48, false);
}

// Last 6 experts routed => expert 0 is inactive (M[0] == 0).  Stresses the
// prepack representative-expert selection (must skip M[i] <= 0) and the
// unpack M == 0 guard.
TEST(GroupMatmulGgmlPerGroup, FifteenExpertsSixActiveContiguousLastBF16) {
  std::vector<int> rows(15, 0);
  for (int e = 9; e < 15; ++e) rows[e] = 16;
  run_ggml_per_group_scenario("15/6 last-6 bf16", rows, 256, 32, true);
}

// Non-uniform token counts across the routed experts (incl. M == 1) — stresses
// per-group quant over ragged M and the row-level grouped quant scheduler.
TEST(GroupMatmulGgmlPerGroup, FifteenExpertsVariedTokenCountsBF16) {
  std::vector<int> rows(15, 0);
  const int idx[6] = {0, 2, 4, 6, 9, 13};
  const int act[6] = {1, 2, 3, 5, 8, 13};
  for (int j = 0; j < 6; ++j) rows[idx[j]] = act[j];
  run_ggml_per_group_scenario("15 varied tokens bf16", rows, 128, 64, true);
}

// Single routed expert in the middle of the inactive set.
TEST(GroupMatmulGgmlPerGroup, FifteenExpertsSingleActiveBF16) {
  std::vector<int> rows(15, 0);
  rows[7] = 8;
  run_ggml_per_group_scenario("15/1 single active bf16", rows, 96, 80, true);
}

// Dense routing: every expert fires (no inactive experts).
TEST(GroupMatmulGgmlPerGroup, FifteenExpertsAllActiveF32) {
  std::vector<int> rows(15, 12);
  run_ggml_per_group_scenario("15/15 all active f32", rows, 128, 64, false);
}

// Degenerate routing: no expert fires (all M == 0).  The call must succeed,
// touch no packed bytes, and produce no output.
TEST(GroupMatmulGgmlPerGroup, FifteenExpertsNoneActiveBF16) {
  std::vector<int> rows(15, 0);
  run_ggml_per_group_scenario("15/0 none active bf16", rows, 128, 64, true);
}
