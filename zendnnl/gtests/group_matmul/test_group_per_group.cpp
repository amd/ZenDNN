/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

/// @file test_group_per_group.cpp
/// @brief Unit tests for the grouped (MoE) PER-GROUP dynamic-quant path.
///
/// The grouped per-group kernel reuses the same single-group quant body as
/// the single-tensor per-group kernels (`quant_one_group_*_s8`), so the
/// grouped scheduler must produce bit-identical s8 output + scales to the
/// single-tensor per-group reference run per expert.  These tests pin that
/// parity (across active/inactive experts) plus the K % G validation.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "common/zendnnl_global.hpp"
#include "lowoha_operators/reorder/lowoha_reorder.hpp"
#include "lowoha_operators/reorder/reorder_data_type/dynamic_quant_impl/dynamic_kernels.hpp"

namespace {

using zendnnl::common::data_type_t;
using status_t = zendnnl::error_handling::status_t;
using zendnnl::lowoha::reorder::group_dynamic_quant;
using zendnnl::lowoha::reorder::group_dynamic_quant_params_t;
using zendnnl::lowoha::reorder::dynamic_per_group_quant_bf16_s8_native;
using zendnnl::lowoha::reorder::dynamic_per_group_quant_f32_s8_native;

// The native per-group quant kernels (`dynamic_per_group_quant_*_native`)
// emit AVX-512F intrinsics, so any test that reaches them — directly or via
// `group_dynamic_quant` — SIGILLs on hosts without AVX-512F.  Gate at runtime
// for consistency with the other ISA-specific gtests.
bool host_has_avx512f() {
  return zendnnl::common::zendnnl_platform_info().get_avx512f_status();
}

// Truncating float -> bf16 (top 16 bits).  The kernels reconstruct f32 by
// shifting the bits back up, so the round-trip is exact and parity is
// independent of rounding policy.
uint16_t f2bf16(float f) {
  uint32_t b;
  std::memcpy(&b, &f, sizeof(b));
  return static_cast<uint16_t>(b >> 16);
}

// Drives `group_dynamic_quant` (grouped per-group) and compares against the
// single-tensor per-group kernel run per active expert.  `is_bf16` selects
// the source dtype path.
void run_parity(const std::vector<int> &M, int K, int G, bool is_bf16) {
  if (!host_has_avx512f()) {
    GTEST_SKIP() << "Requires AVX-512F for native per-group quant kernels";
  }
  const size_t E = M.size();
  std::mt19937 rng(1234 + K + G + (is_bf16 ? 1 : 0));
  std::uniform_real_distribution<float> dist(-4.0f, 4.0f);

  std::vector<std::vector<uint16_t>> src_bf16(E);
  std::vector<std::vector<float>>    src_f32(E);
  std::vector<std::vector<int8_t>>   dst_grp(E), dst_ref(E);
  std::vector<std::vector<float>>    scl_grp(E), scl_ref(E);
  std::vector<const void *> src(E, nullptr);
  std::vector<void *>       dst(E, nullptr), scl(E, nullptr);
  std::vector<int>          Kv(E, K);

  for (size_t e = 0; e < E; ++e) {
    const int m = M[e];
    const size_t n = static_cast<size_t>(std::max(0, m)) * K;
    src_bf16[e].resize(is_bf16 ? n : 0);
    src_f32[e].resize(is_bf16 ? 0 : n);
    for (size_t i = 0; i < n; ++i) {
      const float v = dist(rng);
      if (is_bf16) src_bf16[e][i] = f2bf16(v);
      else         src_f32[e][i]  = v;
    }
    dst_grp[e].assign(n, 0);
    dst_ref[e].assign(n, 0);
    const size_t ns = static_cast<size_t>(std::max(0, m)) * G;
    scl_grp[e].assign(ns, 0.0f);
    scl_ref[e].assign(ns, 0.0f);

    src[e] = is_bf16 ? static_cast<const void *>(src_bf16[e].data())
                     : static_cast<const void *>(src_f32[e].data());
    if (n == 0) src[e] = nullptr;
    dst[e] = dst_grp[e].empty() ? nullptr : dst_grp[e].data();
    scl[e] = scl_grp[e].empty() ? nullptr : scl_grp[e].data();
  }

  group_dynamic_quant_params_t gp;
  gp.src_dtype   = is_bf16 ? data_type_t::bf16 : data_type_t::f32;
  gp.dst_dtype   = data_type_t::s8;
  gp.scale_dtype = data_type_t::f32;
  gp.num_threads = 4;
  gp.num_groups  = G;
  std::vector<std::vector<int64_t>> no_strides;

  ASSERT_EQ(group_dynamic_quant(src, M, Kv, no_strides, dst, no_strides, scl,
                                gp),
            status_t::success);

  for (size_t e = 0; e < E; ++e) {
    if (M[e] == 0) continue;
    if (is_bf16) {
      dynamic_per_group_quant_bf16_s8_native(src_bf16[e].data(),
                                             dst_ref[e].data(),
                                             scl_ref[e].data(), M[e], K, G);
    } else {
      dynamic_per_group_quant_f32_s8_native(src_f32[e].data(),
                                            dst_ref[e].data(),
                                            scl_ref[e].data(), M[e], K, G);
    }
    EXPECT_EQ(dst_grp[e], dst_ref[e]) << "s8 mismatch on expert " << e;
    EXPECT_EQ(scl_grp[e], scl_ref[e]) << "scale mismatch on expert " << e;
  }
}

}  // namespace

// Grouped per-group BF16->S8 matches the single-tensor reference, including
// an inactive (M==0) expert interleaved and a single-token (M==1) expert.
TEST(GroupPerGroupQuant, GroupedMatchesSingleTensorBF16) {
  run_parity(/*M=*/{3, 0, 1, 5}, /*K=*/64, /*G=*/4, /*is_bf16=*/true);
}

TEST(GroupPerGroupQuant, GroupedMatchesSingleTensorF32) {
  run_parity(/*M=*/{2, 4, 0, 1}, /*K=*/128, /*G=*/8, /*is_bf16=*/false);
}

// G == K (group_size == 1) and G == 2 (large groups) edge group counts.
TEST(GroupPerGroupQuant, GroupCountExtremes) {
  run_parity(/*M=*/{4}, /*K=*/32, /*G=*/2,  /*is_bf16=*/true);
  run_parity(/*M=*/{2}, /*K=*/32, /*G=*/32, /*is_bf16=*/true);
}

// An INACTIVE (M==0) expert whose K is not divisible by num_groups must NOT
// fail the op — its data is never quantized.  Regression: the per-group
// `K % num_groups` check used to run before the `M[i] == 0` skip, so a single
// unrouted expert with a non-divisible K aborted the whole grouped call.
TEST(GroupPerGroupQuant, InactiveExpertBadKDoesNotFail) {
  if (!host_has_avx512f()) {
    GTEST_SKIP() << "Requires AVX-512F for native per-group quant kernels";
  }
  const int G = 4;
  // Expert 0: active, K=64 (64 % 4 == 0).  Expert 1: inactive (M==0), K=30
  // (30 % 4 != 0) — must be skipped, not rejected.
  const std::vector<int> M  = {2, 0};
  const std::vector<int> Kv = {64, 30};

  std::vector<uint16_t> src0(static_cast<size_t>(M[0]) * Kv[0]);
  std::mt19937 rng(99);
  std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
  for (auto &v : src0) v = f2bf16(dist(rng));
  std::vector<int8_t> dst0(static_cast<size_t>(M[0]) * Kv[0], 0);
  std::vector<float>  scl0(static_cast<size_t>(M[0]) * G, 0.0f);

  // Inactive expert carries null buffers — the convention for unrouted experts.
  std::vector<const void *> src = {src0.data(), nullptr};
  std::vector<void *>       dst = {dst0.data(), nullptr};
  std::vector<void *>       scl = {scl0.data(), nullptr};

  group_dynamic_quant_params_t gp;
  gp.src_dtype   = data_type_t::bf16;
  gp.dst_dtype   = data_type_t::s8;
  gp.scale_dtype = data_type_t::f32;
  gp.num_threads = 1;
  gp.num_groups  = G;
  std::vector<std::vector<int64_t>> no_strides;

  ASSERT_EQ(group_dynamic_quant(src, M, Kv, no_strides, dst, no_strides, scl,
                                gp),
            status_t::success)
      << "inactive expert with non-divisible K must be skipped, not rejected";

  // The active expert must still be quantized correctly (parity with the
  // single-tensor per-group reference).
  std::vector<int8_t> dst_ref(static_cast<size_t>(M[0]) * Kv[0], 0);
  std::vector<float>  scl_ref(static_cast<size_t>(M[0]) * G, 0.0f);
  dynamic_per_group_quant_bf16_s8_native(src0.data(), dst_ref.data(),
                                         scl_ref.data(), M[0], Kv[0], G);
  EXPECT_EQ(dst0, dst_ref) << "active expert s8 mismatch";
  EXPECT_EQ(scl0, scl_ref) << "active expert scale mismatch";
}

// K not divisible by num_groups is rejected.
TEST(GroupPerGroupQuant, RejectsKNotDivisibleByG) {
  if (!host_has_avx512f()) {
    GTEST_SKIP() << "Requires AVX-512F for native per-group quant kernels";
  }
  const std::vector<int> M = {2};
  const int K = 30, G = 4;  // 30 % 4 != 0
  std::vector<uint16_t> src_bank(static_cast<size_t>(M[0]) * K, f2bf16(0.5f));
  std::vector<int8_t>   dst_bank(static_cast<size_t>(M[0]) * K, 0);
  std::vector<float>    scl_bank(static_cast<size_t>(M[0]) * G, 0.0f);

  std::vector<const void *> src = {src_bank.data()};
  std::vector<void *>       dst = {dst_bank.data()};
  std::vector<void *>       scl = {scl_bank.data()};
  std::vector<int>          M_v = M, K_v = {K};

  group_dynamic_quant_params_t gp;
  gp.src_dtype   = data_type_t::bf16;
  gp.dst_dtype   = data_type_t::s8;
  gp.scale_dtype = data_type_t::f32;
  gp.num_threads = 1;
  gp.num_groups  = G;
  std::vector<std::vector<int64_t>> no_strides;

  EXPECT_EQ(group_dynamic_quant(src, M_v, K_v, no_strides, dst, no_strides,
                                scl, gp),
            status_t::failure);
}
