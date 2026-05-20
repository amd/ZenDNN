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
  std::string           label;                       // human-readable
  bool                  expect_success = true;
};

// Storage owned by the caller; pointers in `run_prepare` reference these.
// Kept as a single struct so a per-test parameter doesn't need to declare
// each vector individually.
struct PrepCallStorage {
  std::vector<bfloat16_t>     wei_storage;
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
  // Weights are bf16 by contract on the supported variants; for
  // negative dtype tests we still allocate as bf16 so the storage is
  // valid (the dispatcher refuses before reading the bytes).
  storage.wei_storage.assign(static_cast<size_t>(c.K) * c.N,
                             bfloat16_t(0.05f));

  void *bias_ptr = nullptr;
  if (c.bias_dt == data_type_t::bf16) {
    storage.bias_bf16_storage.assign(c.N, bfloat16_t(0.01f));
    bias_ptr = storage.bias_bf16_storage.data();
  } else if (c.bias_dt == data_type_t::f32) {
    storage.bias_f32_storage.assign(c.N, 0.01f);
    bias_ptr = storage.bias_f32_storage.data();
  }

  std::vector<bool>  transA_v{c.transA};
  std::vector<bool>  transB_v{c.transB};
  std::vector<int>   M_v{c.M};
  std::vector<int>   N_v{c.N};
  std::vector<int>   K_v{c.K};
  std::vector<int>   ldb_v{c.transB ? c.K : c.N};
  std::vector<float> alpha_v{1.0f};
  std::vector<float> beta_v{0.0f};
  std::vector<const void *> weight_v{storage.wei_storage.data()};
  std::vector<bool>  is_wc_v{c.is_wc};

  (void)bias_ptr;  // bias buffer is per-tile; prepare_for_call only
                   // consults bias_dtype at the dispatcher level.

  return ck::prepare_for_call(c.act, c.src_dt, c.wei_dt, c.dst_dt,
                              c.act_dt, c.bias_dt, transA_v, transB_v,
                              M_v, N_v, K_v, ldb_v, alpha_v, beta_v,
                              weight_v, is_wc_v, kctx);
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
