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

/// `dispatch_supported()` ISA gate — the cheap CPUID check the
/// dispatcher exposes so callers can early-out before building any
/// inputs to `prepare_for_call`.  These tests assert the function's
/// observable contract on this host:
///   1. Returns the same value across calls (cached after first).
///   2. Is consistent with the per-call `prepare_for_call` ISA branch
///      (when `dispatch_supported()` is false, `prepare_for_call`
///      MUST refuse — the dispatcher relies on this invariant to
///      decide whether the per-tile path is safe to invoke).
///
/// Runtime: O(1) — three cases.

#include "ck_test_helpers.hpp"

namespace {

namespace ck = ck_test::ck;

TEST(CkDispatchSupported, IsCachedAcrossCalls) {
  const bool a = ck::dispatch_supported();
  const bool b = ck::dispatch_supported();
  const bool c = ck::dispatch_supported();
  EXPECT_EQ(a, b);
  EXPECT_EQ(b, c);
}

TEST(CkDispatchSupported, MatchesPrepareForCallIsaGate) {
  // On a host without AVX-512-BF16, `prepare_for_call` MUST refuse
  // every shape (that's the dispatcher's first run-once invariant
  // check).  On a host with the ISA, the supported BF16 variant
  // succeeds — so we use the supported tuple as the probe.
  ck_test::PrepCallCase c{};  // defaults to (bf16, bf16, bf16)
  ck_test::PrepCallStorage storage;
  ck::CallContext kctx;
  const auto status = ck_test::run_prepare(c, storage, kctx);

  if (ck::dispatch_supported()) {
    EXPECT_EQ(status,
              zendnnl::error_handling::status_t::success)
        << "dispatch_supported()=true but prepare_for_call refused "
           "the canonical bf16:bf16:bf16 supported tuple";
    EXPECT_TRUE(kctx.enabled);
    EXPECT_NE(kctx.variant, ck::KernelVariant::kUnsupported);
  } else {
    EXPECT_EQ(status,
              zendnnl::error_handling::status_t::failure)
        << "dispatch_supported()=false but prepare_for_call accepted "
           "the call — kernel would run with no-AVX512BF16 host UB";
    EXPECT_FALSE(kctx.enabled);
  }
}

TEST(CkDispatchSupported, NoSideEffects) {
  // Calling dispatch_supported() must not mutate any global state
  // observable to a follow-up prepare_for_call.  Probe twice with a
  // call in between; both `enabled` outcomes must match.
  ck_test::PrepCallCase c{};
  ck_test::PrepCallStorage s1, s2;
  ck::CallContext k1, k2;

  const auto st1 = ck_test::run_prepare(c, s1, k1);
  (void)ck::dispatch_supported();   // intervening probe
  const auto st2 = ck_test::run_prepare(c, s2, k2);

  EXPECT_EQ(st1, st2);
  EXPECT_EQ(k1.enabled, k2.enabled);
  EXPECT_EQ(static_cast<int>(k1.variant),
            static_cast<int>(k2.variant));
}

}  // namespace
