/********************************************************************************
# * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

/// @file test_group_reorder.cpp
/// @brief Unit tests for `reorder::group_reorder` — the grouped wrapper
///        that applies `reorder_direct` to a batch of independent reorder
///        ops in a sequential loop.
///
/// These are plain `TEST()` cases (not the parameterised `TestReorder`
/// round-trip fixture) because `group_reorder` is a thin batching wrapper:
/// the per-op math is already covered by the other reorder suites, so the
/// contract under test here is purely the wrapper's — namely that a group
/// produces exactly what the equivalent per-op `reorder_direct` loop does,
/// plus the size / abort / mixed-mode edge behaviour documented on the API.

#include <gtest/gtest.h>

#include <cstring>
#include <vector>

#include "gtest_utils.hpp"
#include "lowoha_operators/reorder/lowoha_reorder.hpp"

namespace {

namespace rdr = zendnnl::lowoha::reorder;
using data_type_t = zendnnl::memory::data_type_t;
using status_t = zendnnl::memory::status_t;
using zendnnl::common::bfloat16_t;

// Build a plain F32 -> BF16 type-conversion op over an [M, N] tensor.
// Scale/zp are left unset (buff == nullptr), so reorder_direct performs a
// simple element-wise conversion and does NOT mutate the params — which
// keeps the per-op reference run and the grouped run byte-identical.
rdr::reorder_params_t make_convert_params(int M, int N) {
    rdr::reorder_params_t p;
    p.src_dtype = data_type_t::f32;
    p.dst_dtype = data_type_t::bf16;
    p.src_shape = {M, N};
    p.dst_shape = {M, N};
    p.num_threads = 1;
    return p;
}

// Deterministic source fill so per-op and grouped runs see identical input.
void fill_f32(std::vector<float> &v, int seed) {
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = static_cast<float>((static_cast<int>(i) * 7 + seed) % 97)
                        * 0.03125f
                - 1.5f;
    }
}

// ──────────────────────────────────────────────────────────────────
// Core contract: a group produces exactly what the per-op
// reorder_direct loop produces (byte-for-byte) for the same inputs.
// ──────────────────────────────────────────────────────────────────
TEST(GroupReorder, MatchesPerOpReorderLoop) {
    struct Shape {
        int M, N;
    };
    const std::vector<Shape> shapes = {{4, 8}, {2, 16}, {8, 4}, {1, 32}};
    const int num_ops = static_cast<int>(shapes.size());

    std::vector<std::vector<float>> src_store(num_ops);
    std::vector<std::vector<bfloat16_t>> ref_store(num_ops);
    std::vector<std::vector<bfloat16_t>> grp_store(num_ops);

    std::vector<const void *> src;
    std::vector<void *> ref_dst;
    std::vector<void *> grp_dst;
    std::vector<rdr::reorder_params_t> ref_params;
    std::vector<rdr::reorder_params_t> grp_params;

    for (int i = 0; i < num_ops; ++i) {
        const int M = shapes[i].M, N = shapes[i].N;
        const size_t nelems = static_cast<size_t>(M) * N;
        src_store[i].resize(nelems);
        ref_store[i].assign(nelems, bfloat16_t(0.0f));
        grp_store[i].assign(nelems, bfloat16_t(0.0f));
        fill_f32(src_store[i], i);

        src.push_back(src_store[i].data());
        ref_dst.push_back(ref_store[i].data());
        grp_dst.push_back(grp_store[i].data());
        ref_params.push_back(make_convert_params(M, N));
        grp_params.push_back(make_convert_params(M, N));
    }

    // Reference: per-op reorder_direct, one call per op.
    for (int i = 0; i < num_ops; ++i) {
        ASSERT_EQ(rdr::reorder_direct(src[i], ref_dst[i], ref_params[i]),
                status_t::success)
                << "per-op reorder_direct failed at op " << i;
    }

    // Grouped: a single group_reorder over the whole batch.
    ASSERT_EQ(rdr::group_reorder(src, grp_dst, grp_params), status_t::success);

    // Byte-equality per op — group_reorder must be exactly the per-op loop.
    for (int i = 0; i < num_ops; ++i) {
        EXPECT_EQ(0,
                std::memcmp(ref_store[i].data(), grp_store[i].data(),
                        ref_store[i].size() * sizeof(bfloat16_t)))
                << "group_reorder output differs from the per-op "
                   "reorder_direct "
                   "output at op "
                << i;
    }
}

// ──────────────────────────────────────────────────────────────────
// Empty group → failure (nothing to do; treated as a usage error).
// ──────────────────────────────────────────────────────────────────
TEST(GroupReorder, EmptyGroupFails) {
    std::vector<const void *> src;
    std::vector<void *> dst;
    std::vector<rdr::reorder_params_t> params; // empty
    EXPECT_EQ(rdr::group_reorder(src, dst, params), status_t::failure);
}

// ──────────────────────────────────────────────────────────────────
// Mismatched vector lengths → failure (src/dst must line up 1:1 with
// params).
// ──────────────────────────────────────────────────────────────────
TEST(GroupReorder, VectorSizeMismatchFails) {
    std::vector<float> s0(4 * 8), s1(4 * 8);
    std::vector<bfloat16_t> d0(4 * 8), d1(4 * 8);
    fill_f32(s0, 0);
    fill_f32(s1, 1);

    // params describes 2 ops, but src only has 1 entry → mismatch.
    std::vector<rdr::reorder_params_t> params {
            make_convert_params(4, 8), make_convert_params(4, 8)};
    std::vector<const void *> src {s0.data()}; // size 1
    std::vector<void *> dst {d0.data(), d1.data()}; // size 2

    EXPECT_EQ(rdr::group_reorder(src, dst, params), status_t::failure);
}

// ──────────────────────────────────────────────────────────────────
// Abort-on-first-failure: a bad op mid-group stops the loop and
// returns that op's failure, while ops completed BEFORE it keep their
// finished output (no rollback).
// ──────────────────────────────────────────────────────────────────
TEST(GroupReorder, AbortsOnFirstOpFailureKeepingEarlierResults) {
    constexpr int M = 4, N = 8;
    const size_t nelems = static_cast<size_t>(M) * N;

    std::vector<float> s0(nelems), s1(nelems);
    std::vector<bfloat16_t> ref0(nelems, bfloat16_t(0.0f));
    std::vector<bfloat16_t> d0(nelems, bfloat16_t(0.0f));
    fill_f32(s0, 3);
    fill_f32(s1, 4);

    // Reference completion of op0 alone (for the "earlier result kept" check).
    rdr::reorder_params_t ref_p0 = make_convert_params(M, N);
    ASSERT_EQ(rdr::reorder_direct(s0.data(), ref0.data(), ref_p0),
            status_t::success);

    // Group: op0 valid; op1 has a null dst → reorder_direct rejects it,
    // so group_reorder must abort with failure after completing op0.
    std::vector<rdr::reorder_params_t> params {
            make_convert_params(M, N), make_convert_params(M, N)};
    std::vector<const void *> src {s0.data(), s1.data()};
    std::vector<void *> dst {d0.data(), nullptr};

    EXPECT_EQ(rdr::group_reorder(src, dst, params), status_t::failure);

    // op0 ran before op1 failed — its output must already be written.
    EXPECT_EQ(
            0, std::memcmp(ref0.data(), d0.data(), nelems * sizeof(bfloat16_t)))
            << "op0 output should have been produced before the op1 failure "
               "aborted the group (no rollback expected)";
}

} // namespace
