/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

/// Caller-declared weight-buffer capacity -> int8 in-place weight cache.
///
/// WHAT THIS GUARDS.  The int8 blocked layouts are `N*4` bytes larger than
/// the logical `K*N` weight: they carry a per-output-column int32
/// compensation row that `VPDPBUSD` makes mandatory (it multiplies
/// unsigned x signed, so signed activations are shifted by +128 and the
/// row subtracts the resulting `128 * sum_wei` bias -- see
/// int8_microkernel.hpp).  The row therefore cannot be dropped, only
/// relocated, and the in-place weight-cache gate demands the blocked bytes
/// fit the caller's allocation.  Result: every int8 weight is duplicated
/// out-of-place at every ALGO, while bf16 -- which needs no such row --
/// reorders in place for free.
///
/// `matmul_params::wei_buffer_capacity_bytes` lets a framework declare the
/// real allocation so int8 can take the in-place path with no layout
/// change on either side.  Until one does, it stays 0 and the gate must
/// behave EXACTLY as before; that back-compatibility is the main thing
/// asserted here.

#include <gtest/gtest.h>

#include "lowoha_operators/matmul/lowoha_common.hpp"

namespace {

using zendnnl::lowoha::matmul::wei_inplace_fits;

// A representative per-expert MoE Op1 shape: K=2048, N=1024.
constexpr size_t kPlainS8 = 2048u * 1024u; // 2 097 152
constexpr size_t kBlockedS8 = kPlainS8 + 1024u * 4u; // + int32 per column
constexpr size_t kPlainBf16 = 2048u * 1024u * 2u;

// ── Undeclared capacity: must reproduce the historical strict equality ──

TEST(WeiBufferCapacity, UndeclaredKeepsBf16InPlace) {
    // bf16's blocked layout is exactly the logical extent, so it was
    // already in-place before this feature and must stay so.
    EXPECT_TRUE(wei_inplace_fits(
            kPlainBf16, kPlainBf16, kPlainBf16, /*declared=*/0));
}

TEST(WeiBufferCapacity, UndeclaredKeepsInt8OutOfPlace) {
    // The whole reason int8 duplicates today.  A regression here would
    // silently start mutating past the caller's allocation.
    EXPECT_FALSE(
            wei_inplace_fits(kBlockedS8, kBlockedS8, kPlainS8, /*declared=*/0));
}

// ── Declared capacity: the opt-in a framework will use ──

TEST(WeiBufferCapacity, ExactDeclarationEnablesInt8InPlace) {
    // The framework allocated weights + compensation and said so.
    EXPECT_TRUE(wei_inplace_fits(kBlockedS8, kBlockedS8, kPlainS8, kBlockedS8));
}

TEST(WeiBufferCapacity, OverDeclarationEnablesInt8InPlace) {
    EXPECT_TRUE(wei_inplace_fits(
            kBlockedS8, kBlockedS8, kPlainS8, kBlockedS8 + 4096u));
}

// ── Corner cases: partial slack must NOT be accepted ──

TEST(WeiBufferCapacity, SlackShorterThanCompensationRowIsRejected) {
    // A framework that padded to a page/alignment boundary rather than to
    // the reorder requirement.  One byte short still overruns, so this is
    // the case most likely to appear in the field and silently corrupt.
    EXPECT_FALSE(wei_inplace_fits(
            kBlockedS8, kBlockedS8, kPlainS8, kBlockedS8 - 1u));
    EXPECT_FALSE(wei_inplace_fits(
            kBlockedS8, kBlockedS8, kPlainS8, kPlainS8 + 2048u));
}

TEST(WeiBufferCapacity, DeclarationBelowLogicalExtentIsRejected) {
    // Nonsensical input (smaller than the weights themselves) must not be
    // read as permission.
    EXPECT_FALSE(
            wei_inplace_fits(kBlockedS8, kBlockedS8, kPlainS8, kPlainS8 / 2u));
}

TEST(WeiBufferCapacity, DeclarationIrrelevantWhenLayoutAlreadyFits) {
    // bf16 does not need the opt-in and must not be made conditional on
    // it: a caller that declares nothing, too little, or plenty all get
    // the in-place path.
    EXPECT_TRUE(wei_inplace_fits(kPlainBf16, kPlainBf16, kPlainBf16, 0u));
    EXPECT_TRUE(wei_inplace_fits(kPlainBf16, kPlainBf16, kPlainBf16, 1u));
    EXPECT_TRUE(wei_inplace_fits(
            kPlainBf16, kPlainBf16, kPlainBf16, kPlainBf16 * 2u));
}

// ── Default wiring ──

TEST(WeiBufferCapacity, ParamsDefaultToUndeclared) {
    // The field must default to "assume exactly K*N", or every existing
    // caller would silently opt in to in-place on the first rebuild.
    zendnnl::lowoha::matmul::matmul_params p {};
    EXPECT_EQ(p.wei_buffer_capacity_bytes, 0u);
    EXPECT_FALSE(wei_inplace_fits(
            kBlockedS8, kBlockedS8, kPlainS8, p.wei_buffer_capacity_bytes));
}

} // namespace
