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

/// Custom-kernel gtest scaffolding.
///
/// Header-only.  Lives under `gtests/group_matmul/custom_kernel/`
/// alongside the suite that exercises the BF16 microkernel, dispatcher,
/// and pack module directly (rather than through `group_matmul_direct`
/// like the rest of `gtests/group_matmul/`).  Most fixtures here build
/// minimally-valid inputs to `prepare_for_call` so a single test case
/// can probe one gate at a time.
///
/// What this file does NOT do: replicate the full reference path.  End-
/// to-end correctness comparisons against the AOCL DLP / native BLAS
/// reference live in [test_ukernel_bf16.cpp](test_ukernel_bf16.cpp) and
/// reuse the existing `moe_test_utils.hpp` machinery
/// (`make_uniform_params`, `GemmVecs::uniform`, `tol_act`).

#ifndef ZENDNNL_GTESTS_GROUP_MATMUL_CUSTOM_KERNEL_CK_TEST_HELPERS_HPP
#define ZENDNNL_GTESTS_GROUP_MATMUL_CUSTOM_KERNEL_CK_TEST_HELPERS_HPP

#include <gtest/gtest.h>

#include <vector>

#include "lowoha_operators/matmul/group_matmul/custom_kernel/dispatch.hpp"
#include "moe_test_utils.hpp"

namespace ck_test {

namespace ck = zendnnl::lowoha::matmul::custom_kernel;
using moe_test_utils::bfloat16_t;
using moe_test_utils::data_type_t;
using zendnnl::error_handling::status_t;
using zendnnl::lowoha::matmul::grp_matmul_gated_act_t;

// ──────────────────────────────────────────────────────────────────
// ISA gate — every test that calls into `prepare_for_call` or runs
// the per-tile kernel needs this.  On a host that doesn't have
// AVX-512-BF16 the dispatcher refuses every shape, so end-to-end /
// per-tile / prepare_for_call tests would fail spuriously.
//
// `resolve_variant` is the one exception: it is a pure switch over
// POD enums with no ISA dependency, so its tests run identically
// on every host and intentionally skip this guard.  Tests that
// invoke `prepare_for_call` (gating matrix, refusal cases) DO call
// this macro because the dispatcher's CPUID gate is the first
// refusal it can return.
// ──────────────────────────────────────────────────────────────────
#define CK_SKIP_IF_NO_BF16_ISA()                                       \
  do {                                                                 \
    if (!::ck_test::ck::dispatch_supported()) {                        \
      GTEST_SKIP()                                                     \
          << "AVX-512-BF16 not available on this host; the custom "    \
             "kernel cannot run, so any per-tile test would refuse";   \
    }                                                                  \
  } while (0)

// DQ-INT8 ISA gate — needed for tests that exercise the int8 variant
// resolution through `prepare_for_call`.  The int8 microkernel uses
// VPDPBUSD (AVX-512 VNNI), which on Zen 4+ is a strict superset of
// AVX-512-BF16 but on broader x86 silicon is an independent feature
// flag.  Tests that probe the int8 path call this macro instead of
// (or in addition to) `CK_SKIP_IF_NO_BF16_ISA`.
#define CK_SKIP_IF_NO_INT8_ISA()                                       \
  do {                                                                 \
    if (!::ck_test::ck::avx512vnni_available()) {                      \
      GTEST_SKIP()                                                     \
          << "AVX-512 VNNI not available on this host; the DQ-INT8 "   \
             "custom kernel cannot run, so any per-tile or "           \
             "prepare_for_call probe of the int8 variants would "      \
             "refuse cleanly at the ISA gate";                         \
    }                                                                  \
  } while (0)

// ──────────────────────────────────────────────────────────────────
// PrepCallCase — minimal valid inputs for `prepare_for_call`.
//
// Default values (single expert with a small uniform shape) succeed
// today — every field is what the dispatcher's gate accepts on the
// supported variant.  Tests negate one field at a time to probe the
// individual gate, leaving the rest at default so the failure is
// attributable to that field.  The expected status is encoded in
// `expect_success` so parameterised suites can mix positive + negative
// cases in one INSTANTIATE_TEST_SUITE_P.
//
// Bias buffer is materialised inside `run_prepare()` only when
// `bias_dt != none`; weight + bias storage live in the helper-owned
// vectors so test fixtures don't have to thread them through.
// ──────────────────────────────────────────────────────────────────
struct PrepCallCase {
  data_type_t           src_dt        = data_type_t::bf16;
  data_type_t           wei_dt        = data_type_t::bf16;
  data_type_t           dst_dt        = data_type_t::bf16;
  data_type_t           act_dt        = data_type_t::bf16;
  data_type_t           bias_dt       = data_type_t::none;
  grp_matmul_gated_act_t act           = grp_matmul_gated_act_t::none;
  int                   M             = 16;
  int                   K             = 64;
  int                   N             = 256;        // multiple of pack_nr=32
  bool                  is_wc         = true;
  bool                  transA        = false;
  bool                  transB        = false;
  // DQ-INT8 discriminators threaded through to `prepare_for_call`.
  // Defaults keep the BF16-only behaviour (`dynamic_quant=false`,
  // `compute_dtype=none`) so existing tests are unaffected; int8
  // cases set these explicitly along with `wei_dt=s8`.
  bool                  dynamic_quant = false;
  data_type_t           compute_dt    = data_type_t::none;
  // Linear-form modifiers and ldb override.  Defaults are the
  // identity (alpha=1, beta=0) and `ldb=auto` so the existing
  // positive cases are unchanged; the new int8 refusal cells set
  // these explicitly to probe the corresponding gates.
  float                 alpha         = 1.0f;
  float                 beta          = 0.0f;
  int                   ldb_override  = -1;          // -1 → auto from K/N
  // When >0, override num_ops to drive multi-expert refusal cases
  // (e.g. null weight in second expert, all-M-zero with N>1).
  int                   num_ops_override = 1;
  // When true, the second expert (only used if num_ops_override>=2)
  // gets a null weight pointer.  Drives the
  // `null_weight_in_active_expert` refusal.
  bool                  null_second_weight = false;
  std::string           label;                       // human-readable
  bool                  expect_success = true;
};

// Storage owned by the caller; pointers in `run_prepare` reference these.
// Kept as a single struct so a per-test parameter doesn't need to declare
// each vector individually.
struct PrepCallStorage {
  std::vector<bfloat16_t>     wei_storage;
  std::vector<int8_t>         wei_int8_storage;
  std::vector<bfloat16_t>     bias_bf16_storage;
  std::vector<float>          bias_f32_storage;
};

// One-shot driver for `prepare_for_call` — populates `storage` with
// the right buffers, fills the per-expert vectors, and calls into
// the dispatcher.  Returns the status `prepare_for_call` returned;
// `kctx` is left at its post-call state (caller can inspect
// `kctx.enabled` / `kctx.variant` etc.).
inline status_t run_prepare(const PrepCallCase &c,
                            PrepCallStorage   &storage,
                            ck::CallContext   &kctx) {
  // Pick the weight storage based on wei_dt — int8 cases need a
  // K×N region of `int8_t`, every other case (including negative
  // dtype probes) gets bf16 storage and the dispatcher refuses
  // before reading the bytes.
  const void *wei_ptr = nullptr;
  if (c.wei_dt == data_type_t::s8) {
    storage.wei_int8_storage.assign(
        static_cast<size_t>(c.K) * c.N, static_cast<int8_t>(1));
    wei_ptr = storage.wei_int8_storage.data();
  } else {
    storage.wei_storage.assign(
        static_cast<size_t>(c.K) * c.N, bfloat16_t(0.05f));
    wei_ptr = storage.wei_storage.data();
  }

  void *bias_ptr = nullptr;
  if (c.bias_dt == data_type_t::bf16) {
    storage.bias_bf16_storage.assign(c.N, bfloat16_t(0.01f));
    bias_ptr = storage.bias_bf16_storage.data();
  } else if (c.bias_dt == data_type_t::f32) {
    storage.bias_f32_storage.assign(c.N, 0.01f);
    bias_ptr = storage.bias_f32_storage.data();
  }

  std::vector<bool>  transA_v;
  std::vector<bool>  transB_v;
  std::vector<int>   M_v;
  std::vector<int>   N_v;
  std::vector<int>   K_v;
  std::vector<int>   ldb_v;
  std::vector<float> alpha_v;
  std::vector<float> beta_v;
  std::vector<const void *> weight_v;
  std::vector<bool>  is_wc_v;
  const int auto_ldb = c.transB ? c.K : c.N;
  const int eff_ldb  = (c.ldb_override > 0) ? c.ldb_override : auto_ldb;
  const int num_ops  = (c.num_ops_override < 1) ? 1 : c.num_ops_override;
  for (int i = 0; i < num_ops; ++i) {
    transA_v.push_back(c.transA);
    transB_v.push_back(c.transB);
    M_v.push_back(c.M);
    N_v.push_back(c.N);
    K_v.push_back(c.K);
    ldb_v.push_back(eff_ldb);
    alpha_v.push_back(c.alpha);
    beta_v.push_back(c.beta);
    is_wc_v.push_back(c.is_wc);
    if (i == 1 && c.null_second_weight) {
      weight_v.push_back(nullptr);
    } else {
      weight_v.push_back(wei_ptr);
    }
  }

  (void)bias_ptr;  // bias buffer is per-tile; prepare_for_call only
                   // consults bias_dtype at the dispatcher level.

  return ck::prepare_for_call(c.act, c.src_dt, c.wei_dt, c.dst_dt,
                              c.act_dt, c.bias_dt, transA_v, transB_v,
                              M_v, N_v, K_v, ldb_v, alpha_v, beta_v,
                              weight_v, is_wc_v, kctx,
                              c.dynamic_quant, c.compute_dt);
}

// Pretty-name a data_type_t for TestParam labels.  Must return a
// distinct string for every value in `data_type_t` — gtest rejects
// `INSTANTIATE_TEST_SUITE_P` instantiations with duplicate parameter
// names, so any new dtype added to the enum must extend this switch
// before it can appear in a parameterised test sweep.
inline const char *dt_name(data_type_t dt) noexcept {
  switch (dt) {
    case data_type_t::none: return "none";
    case data_type_t::f32:  return "f32";
    case data_type_t::f16:  return "f16";
    case data_type_t::bf16: return "bf16";
    case data_type_t::s32:  return "s32";
    case data_type_t::s64:  return "s64";
    case data_type_t::s16:  return "s16";
    case data_type_t::s8:   return "s8";
    case data_type_t::s4:   return "s4";
    case data_type_t::u32:  return "u32";
    case data_type_t::u16:  return "u16";
    case data_type_t::u8:   return "u8";
    case data_type_t::u4:   return "u4";
    default:                return "unk";
  }
}

}  // namespace ck_test

#endif  // ZENDNNL_GTESTS_GROUP_MATMUL_CUSTOM_KERNEL_CK_TEST_HELPERS_HPP
