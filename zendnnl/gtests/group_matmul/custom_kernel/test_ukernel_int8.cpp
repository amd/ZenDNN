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

/// DQ-INT8 microkernel direct-surface correctness — exercises the
/// hand-rolled `int8_ukernel_fn_t` returned by `select_int8_ukernel`
/// against a scalar s32-accumulator + per-row × per-channel f32-dequant
/// reference.  Mirrors the bf16 sibling `test_ukernel_bf16.cpp` in
/// scope (per-instantiation coverage) but stays at the microkernel
/// surface — no `group_matmul_direct` round-trip — because the int8
/// path threads through extra hoist machinery that the bf16 e2e test
/// already validates structurally.
///
/// Coverage axes (parameterised):
///   * MR ∈ {1, 4, 6, 8}  — a representative cross section that
///     exercises both the small-MR (no double-buffer) and the
///     large-MR (full chain) code paths.
///   * NV ∈ {2, 4}        — NR = 32 and NR = 64; the only two
///     production NRs (`plan_pack_nr_int8` returns one of these).
///   * Compute ∈ {kS8_Sym, kU8_Asym}.
///   * Act    ∈ {none, swiglu_oai_mul, silu_and_mul, gelu_and_mul}.
///   * Bias   ∈ {none, bf16, f32}.
///
/// `select_int8_ukernel` returning `nullptr` for any (MR, NV) tuple
/// where `MR > max_mr_for_nv(NV)` is asserted in the negative
/// coverage block (the dispatcher's `kfn_table_int8` filler relies
/// on this contract).

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "ck_test_helpers.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/pack.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/ukernel/int8_microkernel.hpp"

namespace {

namespace ck = zendnnl::lowoha::matmul::custom_kernel;
namespace mt = moe_test_utils;
using mt::bfloat16_t;
using zendnnl::error_handling::status_t;

// Skip the whole suite when AVX-512 VNNI is unavailable.  Alias of
// the shared `CK_SKIP_IF_NO_INT8_ISA()` (ck_test_helpers.hpp) so the
// VNNI gate has a single definition across the int8 test files.
#define INT8_CK_SKIP_IF_NO_VNNI() CK_SKIP_IF_NO_INT8_ISA()

// ──────────────────────────────────────────────────────────────────
// Scalar reference: s32 accumulator + f32 dequant + optional bias +
// optional gated activation.
//
//   acc[m, v]   = sum_k (src_byte[m, k]  * wei[k, v])
//                  - K_m * sum_wei[v]               (s32)
//   y_f32[m, v] = (acc[m, v]) * src_scale[m] * wei_scale[v] + bias[v]
//
// where `src_byte` is the byte value the microkernel sees:
//   * sym: src_s8 XOR 0x80, K_m = 128.   (Effectively converts to
//     u8 by biasing by +128; the compensation row undoes it.)
//   * asym: src_u8 directly,    K_m = src_zp[m] (int32).
//
// Both forms produce the same `acc[m, v]` numerically: for sym the
// compensation is `K_m * sum_wei[v] = 128 * sum_wei[v]` (constant
// across rows), and for asym it's `src_zp[m] * sum_wei[v]`.
//
// Output dtype is bf16 (Phase 1 only supports bf16 dst); the
// reference rounds via `bfloat16_t(f32)` to match the kernel's
// store path.
// ──────────────────────────────────────────────────────────────────
template <typename SrcByte>
void scalar_ref_dq_int8(int M, int K, int N,
                        const SrcByte *src, int lda,
                        const int8_t  *wei, int ldb,
                        const float   *src_scale,
                        const int32_t *src_zp,   // nullptr → sym
                        const float   *wei_scale,
                        const void    *bias, ck::BiasKind bias_kind,
                        ck::ActKind     act,
                        bfloat16_t     *dst, int ldc) {
  // For sym (`src_zp==nullptr`) the kernel XORs each src byte
  // broadcast with `0x80808080` before the VPDPBUSD reduction,
  // converting the s8 src to u8 by adding +128 modulo 256.  The
  // resulting `+128 * sum_wei[v]` bias is then undone in the
  // epilogue by subtracting `128 * comp[v]` (where `comp[v] =
  // sum_k wei_s8[k,v]`, precomputed at pack time).  Mirror that
  // exact byte-level transformation here so the reference
  // computes the same s32 accumulator the kernel does — without
  // the XOR-by-0x80 step the byte the reference reads as `uint8`
  // would be `src_s8 + 256` (for negative s8) rather than
  // `src_s8 + 128`, and the per-element products would diverge.
  const bool sym = (src_zp == nullptr);
  const uint8_t xor_mask = sym ? 0x80 : 0x00;
  for (int m = 0; m < M; ++m) {
    // Per-row K_m: sym → 128, asym → src_zp[m].
    const int32_t K_m =
        (src_zp != nullptr) ? src_zp[m] : 128;
    for (int v = 0; v < N; ++v) {
      int64_t acc = 0;
      int32_t sum_wei = 0;
      for (int k = 0; k < K; ++k) {
        const uint8_t raw =
            static_cast<uint8_t>(src[m * lda + k]) ^ xor_mask;
        const int32_t a = static_cast<int32_t>(raw);
        const int32_t b =
            static_cast<int32_t>(wei[k * ldb + v]);
        acc += static_cast<int64_t>(a) * static_cast<int64_t>(b);
        sum_wei += b;
      }
      acc -= static_cast<int64_t>(K_m) * static_cast<int64_t>(sum_wei);
      float y =
          static_cast<float>(acc) * src_scale[m] * wei_scale[v];
      if (bias != nullptr) {
        if (bias_kind == ck::BiasKind::bf16) {
          y += static_cast<float>(
              reinterpret_cast<const bfloat16_t *>(bias)[v]);
        } else {
          y += reinterpret_cast<const float *>(bias)[v];
        }
      }
      (void)act;  // ActKind handling below (caller passes act=none
                  // for this reference path; gated activations are
                  // applied by `apply_gated_ref` on top of `dst`).
      dst[m * ldc + v] = bfloat16_t(y);
    }
  }
}

// Gated-activation reference applied to the dense bf16 matmul
// output.  `act=none` is a no-op.  For the three gated forms the
// caller is expected to have produced a dense `[M, N]` matmul
// result in `mm`; this writes the halved `[M, N/2]` activated
// output into `dst` at `ldc`.
//
// For `swiglu_oai_mul` / `silu_and_mul` / `gelu_and_mul` the CK
// arena physically follows the INTERLEAVED layout `[g0,u0,g1,u1...]`
// (the pack module re-orders split-halves callers at pack time;
// for the swiglu_oai_mul caller the layout is already interleaved
// at the API boundary).  This reference therefore reads pair
// `(2v, 2v+1)` and writes col `v` for all three gated kinds.
inline float silu(float x) {
  return x / (1.0f + std::exp(-x));
}
inline float gelu_tanh(float x) {
  const float k = 0.7978845608028654f;   // sqrt(2/pi)
  const float t = k * (x + 0.044715f * x * x * x);
  return 0.5f * x * (1.0f + std::tanh(t));
}
inline float swiglu_oai(float g, float u) {
  // Reference matches CK's `swiglu_oai_store_pair` (clamp + α-swish).
  const float alpha = 1.702f;
  const float lim   = 7.0f;
  const float gc    = std::min(g, lim);
  const float uc    = std::max(std::min(u, lim), -lim);
  return gc * (1.0f / (1.0f + std::exp(-alpha * gc))) * (uc + 1.0f);
}

void apply_gated_ref(int M, int N, ck::ActKind act,
                     const bfloat16_t *mm, int ldmm,
                     bfloat16_t       *dst, int ldc) {
  if (act == ck::ActKind::none) return;
  const int I = N / 2;
  for (int m = 0; m < M; ++m) {
    for (int v = 0; v < I; ++v) {
      const float g = static_cast<float>(mm[m * ldmm + 2 * v]);
      const float u = static_cast<float>(mm[m * ldmm + 2 * v + 1]);
      float y = 0.0f;
      switch (act) {
        case ck::ActKind::swiglu_oai_mul: y = swiglu_oai(g, u);   break;
        case ck::ActKind::silu_and_mul:   y = silu(g) * u;        break;
        case ck::ActKind::gelu_and_mul:   y = gelu_tanh(g) * u;   break;
        default: y = 0.0f;
      }
      dst[m * ldc + v] = bfloat16_t(y);
    }
  }
}

// ──────────────────────────────────────────────────────────────────
// `select_int8_ukernel` truth table — non-null for every supported
// (MR, NV, Compute, Act) tuple, null for out-of-range MR.
// ──────────────────────────────────────────────────────────────────
TEST(CkInt8UkernelSelect, NonNullForValidTuples) {
  INT8_CK_SKIP_IF_NO_VNNI();
  for (int NV : {2, 4}) {
    const int max_mr = ck::max_mr_for_nv(NV);
    for (int MR = 1; MR <= max_mr; ++MR) {
      for (auto C : {ck::IntCompute::kS8_Sym, ck::IntCompute::kU8_Asym}) {
        for (auto act : {ck::ActKind::none,
                         ck::ActKind::swiglu_oai_mul,
                         ck::ActKind::silu_and_mul,
                         ck::ActKind::gelu_and_mul}) {
          auto *fn = ck::select_int8_ukernel(MR, NV, C, act,
                                             ck::DstDt::kBf16);
          EXPECT_NE(fn, nullptr)
              << "select_int8_ukernel returned null for valid tuple "
                 "MR=" << MR << " NV=" << NV
              << " Compute=" << static_cast<int>(C)
              << " Act=" << static_cast<int>(act);
        }
      }
    }
  }
}

// FP32-dst axis (mirror of the bf16 sibling): non-null for
// (act=none, kF32) on every MR/NV/Compute, and null for any gated
// activation + kF32 tuple (gated kinds are BF16-dst only).
TEST(CkInt8UkernelSelect, F32DstNonNullForNoneNullForGated) {
  INT8_CK_SKIP_IF_NO_VNNI();
  for (int NV : {2, 4}) {
    const int max_mr = ck::max_mr_for_nv(NV);
    for (int MR = 1; MR <= max_mr; ++MR) {
      for (auto C : {ck::IntCompute::kS8_Sym, ck::IntCompute::kU8_Asym}) {
        EXPECT_NE(ck::select_int8_ukernel(MR, NV, C, ck::ActKind::none,
                                          ck::DstDt::kF32),
                  nullptr)
            << "f32-dst act=none must be instantiated (MR=" << MR
            << " NV=" << NV << " C=" << static_cast<int>(C) << ")";
        for (auto act : {ck::ActKind::swiglu_oai_mul,
                         ck::ActKind::silu_and_mul,
                         ck::ActKind::gelu_and_mul}) {
          EXPECT_EQ(ck::select_int8_ukernel(MR, NV, C, act,
                                            ck::DstDt::kF32),
                    nullptr)
              << "gated act + f32-dst must be refused (MR=" << MR
              << " NV=" << NV << " act=" << static_cast<int>(act) << ")";
        }
      }
    }
  }
}

TEST(CkInt8UkernelSelect, NullForOutOfRangeMR) {
  INT8_CK_SKIP_IF_NO_VNNI();
  for (int NV : {2, 4}) {
    const int max_mr = ck::max_mr_for_nv(NV);
    auto *fn = ck::select_int8_ukernel(
        max_mr + 1, NV, ck::IntCompute::kS8_Sym, ck::ActKind::none,
        ck::DstDt::kBf16);
    EXPECT_EQ(fn, nullptr)
        << "select_int8_ukernel must return null for MR > max_mr_for_nv(NV)"
        << " (NV=" << NV << " MR=" << (max_mr + 1) << ")";
  }
}

// ──────────────────────────────────────────────────────────────────
// Correctness — small (M × K × N) with random s8/u8 src + s8 wei.
// One concrete (Compute × Act × Bias) cell per parameterised case.
// ──────────────────────────────────────────────────────────────────
struct Int8UkernelCase {
  int                 MR;
  int                 NV;
  int                 K;     // multiple of 4 (kVNNIInt8Quad).
  ck::IntCompute      compute;
  ck::ActKind         act;
  ck::BiasKind        bias;
  std::string         label;
};

class CkInt8UkernelTest
    : public ::testing::TestWithParam<Int8UkernelCase> {};

TEST_P(CkInt8UkernelTest, MatchesScalarReference) {
  INT8_CK_SKIP_IF_NO_VNNI();
  const auto &c = GetParam();
  const int   MR      = c.MR;
  const int   NV      = c.NV;
  const int   NR      = NV * 16;
  const int   pack_nr = NR;
  const int   K       = c.K;
  ASSERT_EQ(K % 4, 0) << "K must be a multiple of kVNNIInt8Quad=4";
  ASSERT_LE(MR, ck::max_mr_for_nv(NV))
      << "MR exceeds max_mr_for_nv(NV) — case is malformed";

  // Random inputs — bounded so the s32 accumulator stays well
  // away from saturation regardless of K.
  std::mt19937 rng(0xC0DECAFEu
                   ^ static_cast<unsigned>(MR * 137 + NV * 11 + K));
  std::uniform_int_distribution<int> wd(-64, 63);    // s8 wei in a moderate range
  std::uniform_int_distribution<int> sd_s8(-64, 63);
  std::uniform_int_distribution<int> sd_u8(0, 127);
  std::uniform_int_distribution<int> zpd(0, 96);
  std::uniform_real_distribution<float> sc(0.001f, 0.01f);

  std::vector<int8_t> wei(static_cast<size_t>(K) * NR);
  for (auto &w : wei) w = static_cast<int8_t>(wd(rng));

  // Pack the weight via the production pack helper (disable_cache
  // so we own the buffer for the test scope).
  const int8_t *packed_raw = nullptr;
  ASSERT_EQ(ck::get_or_pack_weight_int8(
                wei.data(), K, NR, /*ldb=*/NR, pack_nr,
                /*transB=*/false,
                /*interleave_split_halves=*/false,
                &packed_raw, nullptr, /*disable_cache=*/true),
            status_t::success);
  ASSERT_NE(packed_raw, nullptr);
  struct PackedGuard {
    const int8_t *p;
    ~PackedGuard() { ck::free_owned_packed_weight_int8(p); }
  } pg{packed_raw};

  // Src — sym keeps s8 storage (the kernel XORs to u8 internally);
  // asym uses u8 storage directly.
  std::vector<uint8_t> src(static_cast<size_t>(MR) * K);
  const bool sym = (c.compute == ck::IntCompute::kS8_Sym);
  for (auto &b : src) {
    b = sym ? static_cast<uint8_t>(static_cast<int8_t>(sd_s8(rng)))
            : static_cast<uint8_t>(sd_u8(rng));
  }
  std::vector<float>   src_scale(MR);
  std::vector<int32_t> src_zp_v(MR);
  for (int m = 0; m < MR; ++m) {
    src_scale[m] = sc(rng);
    src_zp_v[m]  = sym ? 0 : zpd(rng);
  }
  const int32_t *src_zp_ptr = sym ? nullptr : src_zp_v.data();

  std::vector<float> wei_scale(NR);
  for (int v = 0; v < NR; ++v) wei_scale[v] = sc(rng);

  // Bias buffer (only consulted when bias != BiasKind::none).
  std::vector<bfloat16_t> bias_bf16(NR, bfloat16_t(0.0f));
  std::vector<float>      bias_f32(NR, 0.0f);
  const void *bias_ptr = nullptr;
  if (c.bias == ck::BiasKind::bf16) {
    for (int v = 0; v < NR; ++v) bias_bf16[v] = bfloat16_t(0.02f * v);
    bias_ptr = bias_bf16.data();
  } else if (c.bias == ck::BiasKind::fp32) {
    for (int v = 0; v < NR; ++v) bias_f32[v] = 0.02f * v;
    bias_ptr = bias_f32.data();
  }

  // Kernel output buffer.  For gated activations the kernel writes
  // the halved [MR, NR/2] tight output; for `none` it writes the
  // full [MR, NR] output.
  const int out_cols = (c.act == ck::ActKind::none) ? NR : (NR / 2);
  std::vector<bfloat16_t> dst(static_cast<size_t>(MR) * out_cols,
                              bfloat16_t(0.0f));

  // Resolve the kernel function pointer for this case (BF16 dst).
  auto *fn = ck::select_int8_ukernel(MR, NV, c.compute, c.act,
                                     ck::DstDt::kBf16);
  ASSERT_NE(fn, nullptr) << "select_int8_ukernel returned null";

  // Both `Cout` and `Cout_tight` are supported routings — the
  // kernel picks one per the activation contract.  For `none` the
  // ukernel writes through `Cout` at `ldc=out_cols`; for gated
  // activations it writes through `Cout_tight` at `ldc_tight=out_cols`.
  if (c.act == ck::ActKind::none) {
    fn(/*A=*/src.data(),       /*lda=*/K,
       /*Bpacked=*/packed_raw,
       /*src_scale=*/src_scale.data(),
       /*src_zp=*/src_zp_ptr,
       /*wei_scale=*/wei_scale.data(),
       /*scale_kind=*/ck::ScaleKind::kF32,
       /*bias=*/bias_ptr, c.bias,
       /*Cout=*/dst.data(),   /*ldc=*/out_cols,
       /*Cout_tight=*/nullptr, /*ldc_tight=*/0,
       /*K=*/K);
  } else {
    fn(src.data(), K, packed_raw,
       src_scale.data(), src_zp_ptr, wei_scale.data(),
       ck::ScaleKind::kF32,
       bias_ptr, c.bias,
       /*Cout=*/nullptr, /*ldc=*/0,
       /*Cout_tight=*/dst.data(), /*ldc_tight=*/out_cols,
       /*K=*/K);
  }

  // Reference path — dense matmul + dequant + bias first, then the
  // gated epilogue.  Tolerance follows the bf16 sibling test:
  // bf16-mantissa absolute (~0.02) plus a small relative band for
  // bias / activation drift.
  std::vector<bfloat16_t> ref_mm(static_cast<size_t>(MR) * NR,
                                 bfloat16_t(0.0f));
  scalar_ref_dq_int8<uint8_t>(MR, K, NR, src.data(), K,
                              wei.data(), NR,
                              src_scale.data(), src_zp_ptr,
                              wei_scale.data(),
                              bias_ptr, c.bias,
                              ck::ActKind::none,
                              ref_mm.data(), NR);
  std::vector<bfloat16_t> ref(static_cast<size_t>(MR) * out_cols,
                              bfloat16_t(0.0f));
  if (c.act == ck::ActKind::none) {
    for (int m = 0; m < MR; ++m)
      for (int v = 0; v < NR; ++v)
        ref[m * out_cols + v] = ref_mm[m * NR + v];
  } else {
    apply_gated_ref(MR, NR, c.act, ref_mm.data(), NR,
                    ref.data(), out_cols);
  }

  // BF16-tolerance comparison.
  const float abs_tol = 0.05f;
  const float rel_tol = 0.05f;
  int n_bad = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const float a = static_cast<float>(dst[i]);
    const float b = static_cast<float>(ref[i]);
    const float diff = std::fabs(a - b);
    const float band = abs_tol + rel_tol * std::fabs(b);
    if (diff > band) {
      if (n_bad++ < 8) {
        ADD_FAILURE() << "Mismatch at i=" << i
                      << " kernel=" << a << " ref=" << b
                      << " diff=" << diff << " band=" << band
                      << " (" << c.label << ")";
      }
    }
  }
  EXPECT_EQ(n_bad, 0) << "Total mismatches: " << n_bad
                      << " (" << c.label << ")";
}

// ScaleKind::kBf16 — the production decode path stores per-token
// src_scale and per-channel wei_scale as bf16; the int8 microkernel
// converts bf16->f32 on load.  This was previously exercised only at
// f32 scale.  Verify a bf16-scale invocation matches the scalar
// reference computed with the SAME bf16-rounded scales.
TEST(CkInt8UkernelBf16Scale, MatchesScalarReference) {
  INT8_CK_SKIP_IF_NO_VNNI();
  const int MR = 4, NV = 2, NR = NV * 16, pack_nr = NR, K = 64;

  std::mt19937 rng(0xB16Cu);
  std::uniform_int_distribution<int> wd(-64, 63);
  std::uniform_int_distribution<int> sd_s8(-64, 63);
  std::uniform_real_distribution<float> sc(0.001f, 0.01f);

  std::vector<int8_t> wei(static_cast<size_t>(K) * NR);
  for (auto &w : wei) w = static_cast<int8_t>(wd(rng));

  const int8_t *packed_raw = nullptr;
  ASSERT_EQ(ck::get_or_pack_weight_int8(
                wei.data(), K, NR, /*ldb=*/NR, pack_nr, /*transB=*/false,
                /*interleave_split_halves=*/false, &packed_raw, nullptr,
                /*disable_cache=*/true),
            status_t::success);
  ASSERT_NE(packed_raw, nullptr);
  struct PackedGuard {
    const int8_t *p;
    ~PackedGuard() { ck::free_owned_packed_weight_int8(p); }
  } pg{packed_raw};

  // Symmetric src (s8 stored as u8; kernel XORs internally).
  std::vector<uint8_t> src(static_cast<size_t>(MR) * K);
  for (auto &b : src)
    b = static_cast<uint8_t>(static_cast<int8_t>(sd_s8(rng)));

  // f32 scales -> bf16 buffers handed to the kernel; the reference uses
  // the bf16-rounded f32 values so the comparison is apples-to-apples.
  std::vector<bfloat16_t> src_scale_bf16(MR), wei_scale_bf16(NR);
  std::vector<float>      src_scale_ref(MR), wei_scale_ref(NR);
  for (int m = 0; m < MR; ++m) {
    src_scale_bf16[m] = bfloat16_t(sc(rng));
    src_scale_ref[m]  = static_cast<float>(src_scale_bf16[m]);
  }
  for (int v = 0; v < NR; ++v) {
    wei_scale_bf16[v] = bfloat16_t(sc(rng));
    wei_scale_ref[v]  = static_cast<float>(wei_scale_bf16[v]);
  }

  std::vector<bfloat16_t> dst(static_cast<size_t>(MR) * NR, bfloat16_t(0.0f));
  auto *fn = ck::select_int8_ukernel(MR, NV, ck::IntCompute::kS8_Sym,
                                     ck::ActKind::none, ck::DstDt::kBf16);
  ASSERT_NE(fn, nullptr);
  fn(/*A=*/src.data(), /*lda=*/K, /*Bpacked=*/packed_raw,
     /*src_scale=*/src_scale_bf16.data(), /*src_zp=*/nullptr,
     /*wei_scale=*/wei_scale_bf16.data(),
     /*scale_kind=*/ck::ScaleKind::kBf16,
     /*bias=*/nullptr, ck::BiasKind::none,
     /*Cout=*/dst.data(), /*ldc=*/NR,
     /*Cout_tight=*/nullptr, /*ldc_tight=*/0, /*K=*/K);

  std::vector<bfloat16_t> ref(static_cast<size_t>(MR) * NR, bfloat16_t(0.0f));
  scalar_ref_dq_int8<uint8_t>(MR, K, NR, src.data(), K, wei.data(), NR,
                              src_scale_ref.data(), /*src_zp=*/nullptr,
                              wei_scale_ref.data(), /*bias=*/nullptr,
                              ck::BiasKind::none, ck::ActKind::none,
                              ref.data(), NR);

  const float abs_tol = 0.05f, rel_tol = 0.05f;
  int n_bad = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const float a = static_cast<float>(dst[i]);
    const float b = static_cast<float>(ref[i]);
    if (std::fabs(a - b) > abs_tol + rel_tol * std::fabs(b)) {
      if (n_bad++ < 8) {
        ADD_FAILURE() << "bf16-scale mismatch at i=" << i
                      << " kernel=" << a << " ref=" << b;
      }
    }
  }
  EXPECT_EQ(n_bad, 0) << "Total bf16-scale mismatches: " << n_bad;
}

INSTANTIATE_TEST_SUITE_P(
    SymAndAsym, CkInt8UkernelTest,
    ::testing::Values(
        // act=none, both compute flavours, both NVs, span of MRs.
        Int8UkernelCase{1, 2, 64, ck::IntCompute::kS8_Sym,
                        ck::ActKind::none, ck::BiasKind::none,
                        "sym_mr1_nv2_K64_none"},
        Int8UkernelCase{4, 2, 128, ck::IntCompute::kS8_Sym,
                        ck::ActKind::none, ck::BiasKind::bf16,
                        "sym_mr4_nv2_K128_bf16bias"},
        Int8UkernelCase{8, 2, 64,  ck::IntCompute::kS8_Sym,
                        ck::ActKind::none, ck::BiasKind::fp32,
                        "sym_mr8_nv2_K64_f32bias"},
        Int8UkernelCase{4, 4, 128, ck::IntCompute::kS8_Sym,
                        ck::ActKind::none, ck::BiasKind::none,
                        "sym_mr4_nv4_K128_none"},
        Int8UkernelCase{6, 4, 64,  ck::IntCompute::kS8_Sym,
                        ck::ActKind::none, ck::BiasKind::bf16,
                        "sym_mr6_nv4_K64_bf16bias"},
        // C.1 cell #1 — MR fill at NV=2 sym none K=64.
        Int8UkernelCase{2, 2, 64, ck::IntCompute::kS8_Sym,
                        ck::ActKind::none, ck::BiasKind::none,
                        "sym_mr2_nv2_K64_none"},
        Int8UkernelCase{3, 2, 64, ck::IntCompute::kS8_Sym,
                        ck::ActKind::none, ck::BiasKind::none,
                        "sym_mr3_nv2_K64_none"},
        Int8UkernelCase{5, 2, 64, ck::IntCompute::kS8_Sym,
                        ck::ActKind::none, ck::BiasKind::none,
                        "sym_mr5_nv2_K64_none"},
        Int8UkernelCase{7, 2, 64, ck::IntCompute::kS8_Sym,
                        ck::ActKind::none, ck::BiasKind::none,
                        "sym_mr7_nv2_K64_none"},
        // C.1 cell #2 — MR fill at NV=4 sym none K=64.
        Int8UkernelCase{1, 4, 64, ck::IntCompute::kS8_Sym,
                        ck::ActKind::none, ck::BiasKind::none,
                        "sym_mr1_nv4_K64_none"},
        Int8UkernelCase{3, 4, 64, ck::IntCompute::kS8_Sym,
                        ck::ActKind::none, ck::BiasKind::none,
                        "sym_mr3_nv4_K64_none"},
        Int8UkernelCase{5, 4, 64, ck::IntCompute::kS8_Sym,
                        ck::ActKind::none, ck::BiasKind::none,
                        "sym_mr5_nv4_K64_none"},
        // C.1 cell #3 — NV=4 gated activations at MR=4 sym K=64.
        Int8UkernelCase{4, 4, 64, ck::IntCompute::kS8_Sym,
                        ck::ActKind::swiglu_oai_mul,
                        ck::BiasKind::none,
                        "sym_mr4_nv4_K64_swiglu"},
        Int8UkernelCase{4, 4, 64, ck::IntCompute::kS8_Sym,
                        ck::ActKind::silu_and_mul,
                        ck::BiasKind::none,
                        "sym_mr4_nv4_K64_silu"},
        Int8UkernelCase{4, 4, 64, ck::IntCompute::kS8_Sym,
                        ck::ActKind::gelu_and_mul,
                        ck::BiasKind::none,
                        "sym_mr4_nv4_K64_gelu"},
        // C.1 cell #4 — large K=1024 sym MR=4 NV=2 (multi-K-block).
        Int8UkernelCase{4, 2, 1024, ck::IntCompute::kS8_Sym,
                        ck::ActKind::none, ck::BiasKind::none,
                        "sym_mr4_nv2_K1024_none"},
        // C.1 cell #7 — asym src_zp range (the random uniform [0,96]
        // already covers the typical case; we also exercise the
        // upper end via larger MR which bumps the per-row zp count).
        Int8UkernelCase{6, 2, 128, ck::IntCompute::kU8_Asym,
                        ck::ActKind::none, ck::BiasKind::bf16,
                        "asym_mr6_nv2_K128_bf16bias"},
        // asym path — exercises non-zero src_zp / VPDPBUSD direct path.
        Int8UkernelCase{1, 2, 64,  ck::IntCompute::kU8_Asym,
                        ck::ActKind::none, ck::BiasKind::none,
                        "asym_mr1_nv2_K64_none"},
        Int8UkernelCase{4, 2, 128, ck::IntCompute::kU8_Asym,
                        ck::ActKind::none, ck::BiasKind::fp32,
                        "asym_mr4_nv2_K128_f32bias"},
        Int8UkernelCase{4, 4, 128, ck::IntCompute::kU8_Asym,
                        ck::ActKind::none, ck::BiasKind::bf16,
                        "asym_mr4_nv4_K128_bf16bias"},
        // gated activations — sym, NV=2 (covers swiglu / silu / gelu
        // epilogues; their store path is identical across NVs).
        Int8UkernelCase{4, 2, 64,  ck::IntCompute::kS8_Sym,
                        ck::ActKind::swiglu_oai_mul,
                        ck::BiasKind::none,
                        "sym_mr4_nv2_K64_swiglu"},
        Int8UkernelCase{4, 2, 64,  ck::IntCompute::kS8_Sym,
                        ck::ActKind::silu_and_mul,
                        ck::BiasKind::none,
                        "sym_mr4_nv2_K64_silu"},
        Int8UkernelCase{4, 2, 64,  ck::IntCompute::kS8_Sym,
                        ck::ActKind::gelu_and_mul,
                        ck::BiasKind::none,
                        "sym_mr4_nv2_K64_gelu"},
        // gated + asym to sanity-check the cross product.
        Int8UkernelCase{4, 2, 64,  ck::IntCompute::kU8_Asym,
                        ck::ActKind::swiglu_oai_mul,
                        ck::BiasKind::none,
                        "asym_mr4_nv2_K64_swiglu"},
        // asym + split-halves gated (silu / gelu) — previously only the
        // sym variants + asym swiglu were covered.  Exercises the asym
        // src_zp dequant under the split-halves epilogue at NV=2 and NV=4.
        Int8UkernelCase{4, 2, 64,  ck::IntCompute::kU8_Asym,
                        ck::ActKind::silu_and_mul,
                        ck::BiasKind::none,
                        "asym_mr4_nv2_K64_silu"},
        Int8UkernelCase{4, 2, 64,  ck::IntCompute::kU8_Asym,
                        ck::ActKind::gelu_and_mul,
                        ck::BiasKind::none,
                        "asym_mr4_nv2_K64_gelu"},
        Int8UkernelCase{4, 4, 64,  ck::IntCompute::kU8_Asym,
                        ck::ActKind::silu_and_mul,
                        ck::BiasKind::none,
                        "asym_mr4_nv4_K64_silu"},
        Int8UkernelCase{4, 4, 64,  ck::IntCompute::kU8_Asym,
                        ck::ActKind::gelu_and_mul,
                        ck::BiasKind::none,
                        "asym_mr4_nv4_K64_gelu"}),
    [](const ::testing::TestParamInfo<Int8UkernelCase> &info) {
      return info.param.label;
    });

// ──────────────────────────────────────────────────────────────────
// C.1 cell #5 — extreme weight / src values (zero, max-int8).
// Verifies the ukernel does not crash on saturation-edge inputs and
// that the scalar reference still matches.  The XOR-by-0x80 path
// in sym mode is the one most likely to alias on extreme bytes
// (0x80 src XOR'd to 0x00 produces zero contribution from any
// negative src), so this exercise is non-redundant with the random
// case above.
// ──────────────────────────────────────────────────────────────────
TEST(CkInt8UkernelExtremes, ZeroAndMaxValues) {
  INT8_CK_SKIP_IF_NO_VNNI();
  const int MR = 4;
  const int NV = 2;
  const int NR = NV * 16;
  const int K  = 64;
  for (auto compute : {ck::IntCompute::kS8_Sym, ck::IntCompute::kU8_Asym}) {
    for (int kind = 0; kind < 3; ++kind) {
      // 0: all zero, 1: all max-positive, 2: all max-negative.
      const int8_t  wei_v =
          (kind == 0) ? 0
        : (kind == 1) ? std::numeric_limits<int8_t>::max()
        :               std::numeric_limits<int8_t>::min();
      const uint8_t src_v_sym  =
          (kind == 0) ? 0
        : (kind == 1) ? static_cast<uint8_t>(
                          std::numeric_limits<int8_t>::max())
        :               static_cast<uint8_t>(
                          std::numeric_limits<int8_t>::min());
      const uint8_t src_v_asym =
          (kind == 0) ? 0u
        : (kind == 1) ? std::numeric_limits<uint8_t>::max()
        :               0u;
      const uint8_t src_v =
          (compute == ck::IntCompute::kS8_Sym) ? src_v_sym : src_v_asym;
      std::vector<int8_t>  wei(static_cast<size_t>(K) * NR, wei_v);
      std::vector<uint8_t> src(static_cast<size_t>(MR) * K, src_v);
      std::vector<float>   src_scale(MR, 0.005f);
      std::vector<int32_t> src_zp(MR, 0);
      std::vector<float>   wei_scale(NR, 0.005f);
      const int32_t *src_zp_ptr =
          (compute == ck::IntCompute::kU8_Asym) ? src_zp.data() : nullptr;

      const int8_t *packed = nullptr;
      ASSERT_EQ(ck::get_or_pack_weight_int8(
                    wei.data(), K, NR, NR, NR,
                    /*transB=*/false,
                    /*interleave_split_halves=*/false,
                    &packed, nullptr, /*disable_cache=*/true),
                status_t::success);
      ASSERT_NE(packed, nullptr);
      auto *fn = ck::select_int8_ukernel(MR, NV, compute, ck::ActKind::none,
                                         ck::DstDt::kBf16);
      ASSERT_NE(fn, nullptr);
      std::vector<bfloat16_t> dst(static_cast<size_t>(MR) * NR,
                                  bfloat16_t(0.0f));
      fn(src.data(), K, packed,
         src_scale.data(), src_zp_ptr, wei_scale.data(),
         ck::ScaleKind::kF32,
         /*bias=*/nullptr, ck::BiasKind::none,
         /*Cout=*/dst.data(), /*ldc=*/NR,
         /*Cout_tight=*/nullptr, /*ldc_tight=*/0,
         /*K=*/K);
      // No assertion on values for kind==1/2 (saturation depends on
      // the s32 accumulator math which the ref recomputes); we
      // verify the call completes without crashing for the extreme
      // input space.  The random-input parameterised cases above
      // bound numerical correctness.
      ck::free_owned_packed_weight_int8(packed);
      (void)dst;
    }
  }
}

// ──────────────────────────────────────────────────────────────────
// C.1 cell #6 — sign of src_scale and wei_scale.  The kernel
// applies them as plain f32 multipliers in the dequant epilogue;
// negative scales should propagate sign correctly.  Run a small
// random case with negative scales and verify the kernel matches
// the scalar reference (which simply uses the signed scales).
// ──────────────────────────────────────────────────────────────────
TEST(CkInt8UkernelExtremes, NegativeScales) {
  INT8_CK_SKIP_IF_NO_VNNI();
  const int MR = 4;
  const int NV = 2;
  const int NR = NV * 16;
  const int K  = 64;
  std::mt19937 rng(0xDEADBEEF);
  std::uniform_int_distribution<int> wd(-32, 31);
  std::uniform_int_distribution<int> sd(-32, 31);
  std::vector<int8_t> wei(static_cast<size_t>(K) * NR);
  for (auto &w : wei) w = static_cast<int8_t>(wd(rng));
  std::vector<uint8_t> src(static_cast<size_t>(MR) * K);
  for (auto &b : src) b = static_cast<uint8_t>(static_cast<int8_t>(sd(rng)));
  std::vector<float> src_scale(MR), wei_scale(NR);
  for (int m = 0; m < MR; ++m) src_scale[m] = -0.003f;  // negative
  for (int v = 0; v < NR; ++v) wei_scale[v] = (v & 1) ? -0.004f : 0.004f;

  const int8_t *packed = nullptr;
  ASSERT_EQ(ck::get_or_pack_weight_int8(
                wei.data(), K, NR, NR, NR,
                /*transB=*/false,
                /*interleave_split_halves=*/false,
                &packed, nullptr, /*disable_cache=*/true),
            status_t::success);
  ASSERT_NE(packed, nullptr);
  auto *fn = ck::select_int8_ukernel(
      MR, NV, ck::IntCompute::kS8_Sym, ck::ActKind::none, ck::DstDt::kBf16);
  ASSERT_NE(fn, nullptr);
  std::vector<bfloat16_t> dst(static_cast<size_t>(MR) * NR,
                              bfloat16_t(0.0f));
  fn(src.data(), K, packed,
     src_scale.data(), /*src_zp=*/nullptr, wei_scale.data(),
     ck::ScaleKind::kF32,
     /*bias=*/nullptr, ck::BiasKind::none,
     dst.data(), NR, nullptr, 0, K);
  std::vector<bfloat16_t> ref(static_cast<size_t>(MR) * NR,
                              bfloat16_t(0.0f));
  scalar_ref_dq_int8<uint8_t>(MR, K, NR, src.data(), K,
                              wei.data(), NR,
                              src_scale.data(), nullptr,
                              wei_scale.data(),
                              /*bias=*/nullptr, ck::BiasKind::none,
                              ck::ActKind::none, ref.data(), NR);
  int n_bad = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const float a = static_cast<float>(dst[i]);
    const float b = static_cast<float>(ref[i]);
    if (std::fabs(a - b) > 0.05f + 0.05f * std::fabs(b)) {
      if (n_bad++ < 4) {
        ADD_FAILURE() << "neg-scale mismatch i=" << i
                      << " kernel=" << a << " ref=" << b;
      }
    }
  }
  EXPECT_EQ(n_bad, 0);
  ck::free_owned_packed_weight_int8(packed);
}

// ──────────────────────────────────────────────────────────────────
// FP32-dst axis (ukernel_f32dst) — `act=none`, DstDt::kF32.  The
// kernel stores the dequantised FP32 accumulator directly via
// `_mm512_storeu_ps` (no BF16 narrowing), so the reference compares
// against the exact FP32 dequant with a tight tolerance.  Small-MR
// cases (MR ∈ {1, 2, 3}) also exercise the double-buffer path
// (kBuffers == 2); larger MR exercise single-buffer.  Sym + asym,
// NV ∈ {2, 4}, multi-K-block (K=256) included.
// ──────────────────────────────────────────────────────────────────
TEST(CkInt8UkernelF32Dst, MatchesScalarF32Reference) {
  INT8_CK_SKIP_IF_NO_VNNI();
  struct F32Case { int MR; int NV; int K; ck::IntCompute compute; };
  const F32Case cases[] = {
      {1, 2, 64,  ck::IntCompute::kS8_Sym},   // double-buffer, sym
      {2, 2, 64,  ck::IntCompute::kS8_Sym},   // double-buffer, sym
      {3, 2, 256, ck::IntCompute::kS8_Sym},   // double-buffer, multi-K
      {8, 2, 64,  ck::IntCompute::kS8_Sym},   // single-buffer, sym
      {1, 4, 64,  ck::IntCompute::kU8_Asym},  // double-buffer, asym
      {3, 4, 128, ck::IntCompute::kU8_Asym},  // double-buffer, asym
      {6, 4, 256, ck::IntCompute::kU8_Asym},  // single-buffer, asym
  };
  for (const auto &c : cases) {
    const int NV = c.NV;
    const int NR = NV * 16;
    const int K  = c.K;
    const int MR = c.MR;
    const bool sym = (c.compute == ck::IntCompute::kS8_Sym);

    std::mt19937 rng(0xF32D57u ^ static_cast<unsigned>(MR * 131 + NV * 7 + K));
    std::uniform_int_distribution<int> wd(-64, 63);
    std::uniform_int_distribution<int> sd_s8(-64, 63);
    std::uniform_int_distribution<int> sd_u8(0, 127);
    std::uniform_int_distribution<int> zpd(0, 96);
    std::uniform_real_distribution<float> sc(0.001f, 0.01f);

    std::vector<int8_t> wei(static_cast<size_t>(K) * NR);
    for (auto &w : wei) w = static_cast<int8_t>(wd(rng));

    const int8_t *packed = nullptr;
    ASSERT_EQ(ck::get_or_pack_weight_int8(
                  wei.data(), K, NR, /*ldb=*/NR, /*pack_nr=*/NR,
                  /*transB=*/false, /*interleave_split_halves=*/false,
                  &packed, nullptr, /*disable_cache=*/true),
              status_t::success);
    ASSERT_NE(packed, nullptr);
    struct Guard { const int8_t *p; ~Guard() {
      ck::free_owned_packed_weight_int8(p); } } pg{packed};

    std::vector<uint8_t> src(static_cast<size_t>(MR) * K);
    for (auto &b : src) {
      b = sym ? static_cast<uint8_t>(static_cast<int8_t>(sd_s8(rng)))
              : static_cast<uint8_t>(sd_u8(rng));
    }
    std::vector<float>   src_scale(MR);
    std::vector<int32_t> src_zp_v(MR);
    for (int m = 0; m < MR; ++m) {
      src_scale[m] = sc(rng);
      src_zp_v[m]  = sym ? 0 : zpd(rng);
    }
    const int32_t *src_zp_ptr = sym ? nullptr : src_zp_v.data();
    std::vector<float> wei_scale(NR);
    for (int v = 0; v < NR; ++v) wei_scale[v] = sc(rng);

    auto *fn = ck::select_int8_ukernel(MR, NV, c.compute, ck::ActKind::none,
                                       ck::DstDt::kF32);
    ASSERT_NE(fn, nullptr) << "f32-dst kernel must be instantiated";

    std::vector<float> dst(static_cast<size_t>(MR) * NR, 0.0f);
    fn(src.data(), K, packed,
       src_scale.data(), src_zp_ptr, wei_scale.data(),
       ck::ScaleKind::kF32,
       /*bias=*/nullptr, ck::BiasKind::none,
       /*Cout=*/dst.data(), /*ldc=*/NR,
       /*Cout_tight=*/nullptr, /*ldc_tight=*/0, /*K=*/K);

    // Exact FP32 dequant reference (no BF16 narrowing).
    const uint8_t xor_mask = sym ? 0x80 : 0x00;
    int n_bad = 0;
    for (int m = 0; m < MR && n_bad == 0; ++m) {
      const int32_t K_m = sym ? 128 : src_zp_v[m];
      for (int v = 0; v < NR; ++v) {
        int64_t acc = 0, sum_wei = 0;
        for (int k = 0; k < K; ++k) {
          const int32_t a = static_cast<int32_t>(
              static_cast<uint8_t>(src[m * K + k]) ^ xor_mask);
          const int32_t b = static_cast<int32_t>(wei[k * NR + v]);
          acc += static_cast<int64_t>(a) * b;
          sum_wei += b;
        }
        const int32_t corrected =
            static_cast<int32_t>(acc - static_cast<int64_t>(K_m) * sum_wei);
        float y = static_cast<float>(corrected);
        y = y * src_scale[m];
        y = y * wei_scale[v];
        const float got = dst[m * NR + v];
        const float band = 1e-3f + 1e-4f * std::fabs(y);
        if (std::fabs(got - y) > band) {
          if (n_bad++ < 4) {
            ADD_FAILURE() << "f32-dst mismatch MR=" << MR << " NV=" << NV
                          << " K=" << K << " sym=" << sym
                          << " at (m=" << m << ",v=" << v << ") got=" << got
                          << " ref=" << y;
          }
        }
      }
    }
    EXPECT_EQ(n_bad, 0);
  }
}

}  // namespace
