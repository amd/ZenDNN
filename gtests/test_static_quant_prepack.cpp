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

// =============================================================================
// Static-INT8 prepack weight-summation-buffer tests (AOCL-DLP backend).
//
// Exercises the 1D asymmetric static-quant compensation rework:
//   * Reorder/prepack implicitly appends an N*int32 per-column weight-sum
//     buffer (colsum[n] = sum_k wei[k,n]) for the u8-source s8 static-quant
//     path (no caller flag; keyed purely on src_dtype == u8).
//   * Matmul, when the caller supplies prepacked weights (mem_format_b == 'r')
//     with a u8 source, reads that buffer and applies a DLP bias post-op
//     scaled by -src_zp, instead of recomputing / caching the compensation
//     per call.
//
// The non-prepacked path (mem_format_b == 'n') still uses the existing
// cache_or_compute_zp_compensation flow; the two must produce identical
// output. Values are kept small so the integer accumulation is exact in f32,
// allowing an exact comparison.
// =============================================================================

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <gtest/gtest.h>

#include "gtest_utils.hpp"
// Prepack column-sum offset helper + weight-cache clear (not in the
// gtest_utils.hpp transitive closure).
#include "lowoha_operators/matmul/backends/aocl/aocl_kernel.hpp"
#include "lowoha_operators/matmul/backends/aocl/aocl_postop.hpp"

namespace {

using zendnnl::common::matmul_algo_t;
using zendnnl::memory::data_type_t;
using zendnnl::memory::status_t;

struct PrepackCase {
    uint64_t m, k, n;
    data_type_t src_dt; // u8 or s8
    int32_t src_zp;
};

void PrintTo(const PrepackCase &p, std::ostream *os) {
    *os << (p.src_dt == data_type_t::u8 ? "U8" : "S8") << "_m" << p.m << "_k"
        << p.k << "_n" << p.n << "_zp" << p.src_zp;
}

// Round up to the shared static-quant colsum alignment. Reuses the production
// constant + helper (aocl_kernel.hpp) so the test stays valid if the alignment
// ever changes.
size_t round_up_align(size_t b) {
    return zendnnl::lowoha::matmul::round_up_to_align(
            b, zendnnl::lowoha::matmul::kStaticQuantColsumAlign);
}

class TestStaticQuantPrepack : public ::testing::TestWithParam<PrepackCase> {
protected:
    void SetUp() override {
        zendnnl::lowoha::matmul::clear_aocl_matmul_weight_caches();
        zendnnl::lowoha::matmul::clear_aocl_postop_metadata_cache();
    }
    void TearDown() override {
        zendnnl::lowoha::matmul::clear_aocl_matmul_weight_caches();
        zendnnl::lowoha::matmul::clear_aocl_postop_metadata_cache();
    }

    // Fill a matmul_params for the static-INT8 1D asymmetric case with
    // per-tensor unit scales (so the int accumulation is preserved exactly).
    matmul_params make_params(const PrepackCase &p, char mem_format_b) {
        matmul_params params;
        params.dtypes.src = p.src_dt;
        params.dtypes.wei = data_type_t::s8;
        params.dtypes.dst = data_type_t::f32;
        params.dtypes.bias = data_type_t::none;
        params.lowoha_algo = matmul_algo_t::aocl_dlp_blocked;
        params.mem_format_a = 'n';
        params.mem_format_b = mem_format_b;
        // Out-of-place weight caching: never mutate the caller's raw weight
        // buffer (keeps the reference run and the prepack input pristine).
        params.weight_cache_type = 1;

        params.quant_params.src_scale.buff = &one_f_;
        params.quant_params.src_scale.dt = data_type_t::f32;
        params.quant_params.src_scale.dims = {1, 1};
        params.quant_params.wei_scale.buff = &one_f_;
        params.quant_params.wei_scale.dt = data_type_t::f32;
        params.quant_params.wei_scale.dims = {1, 1};
        params.quant_params.src_zp.buff = &src_zp_;
        params.quant_params.src_zp.dt = data_type_t::s32;
        params.quant_params.src_zp.dims = {1, 1};
        // Weight zero-point (2D "both zero-points" case) wired only when set.
        if (wei_zp_ != 0) {
            params.quant_params.wei_zp.buff = &wei_zp_;
            params.quant_params.wei_zp.dt = data_type_t::s32;
            params.quant_params.wei_zp.dims = {1, 1};
        }
        return params;
    }

    float one_f_ = 1.0f;
    int32_t src_zp_ = 0;
    int32_t wei_zp_ = 0;
};

// dst_ref[m,n] = sum_k (src[m,k] - src_zp) * (wei[k,n] - wei_zp)
std::vector<float> reference_gemm(const std::vector<uint8_t> &src_u8,
        const std::vector<int8_t> &src_s8, bool src_is_u8,
        const std::vector<int8_t> &wei, int m, int k, int n, int32_t src_zp,
        int32_t wei_zp = 0) {
    std::vector<float> dst(static_cast<size_t>(m) * n, 0.0f);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            int32_t acc = 0;
            for (int kk = 0; kk < k; ++kk) {
                int32_t s = src_is_u8
                        ? static_cast<int32_t>(src_u8[i * k + kk])
                        : static_cast<int32_t>(src_s8[i * k + kk]);
                acc += (s - src_zp)
                        * (static_cast<int32_t>(wei[kk * n + j]) - wei_zp);
            }
            dst[i * n + j] = static_cast<float>(acc);
        }
    }
    return dst;
}

} // namespace

// Prepacked (mem_format_b == 'r', colsum buffer) vs non-prepacked
// (mem_format_b == 'n', cache_or_compute) must match each other and the
// hand-computed reference exactly.
TEST_P(TestStaticQuantPrepack, PrepackedMatchesReferenceAndFallback) {
    const PrepackCase p = GetParam();
    const int m = static_cast<int>(p.m), k = static_cast<int>(p.k),
              n = static_cast<int>(p.n);
    src_zp_ = p.src_zp;

    std::mt19937 rng(0xABCDu ^ (m * 131 + k * 17 + n));
    std::uniform_int_distribution<int> wdist(-8, 7);
    const bool src_is_u8 = (p.src_dt == data_type_t::u8);

    std::vector<int8_t> wei(static_cast<size_t>(k) * n);
    for (auto &w : wei)
        w = static_cast<int8_t>(wdist(rng));

    std::vector<uint8_t> src_u8;
    std::vector<int8_t> src_s8;
    if (src_is_u8) {
        std::uniform_int_distribution<int> sdist(0, 15);
        src_u8.resize(static_cast<size_t>(m) * k);
        for (auto &s : src_u8)
            s = static_cast<uint8_t>(sdist(rng));
    } else {
        std::uniform_int_distribution<int> sdist(-8, 7);
        src_s8.resize(static_cast<size_t>(m) * k);
        for (auto &s : src_s8)
            s = static_cast<int8_t>(sdist(rng));
    }
    const void *src_ptr = src_is_u8 ? static_cast<const void *>(src_u8.data())
                                    : static_cast<const void *>(src_s8.data());

    std::vector<float> ref
            = reference_gemm(src_u8, src_s8, src_is_u8, wei, m, k, n, p.src_zp);

    const int lda = k, ldb = n, ldc = n;
    matmul_batch_params_t batch_params;

    // --- Non-prepacked path (existing zp_comp cache flow) ---
    std::vector<float> dst_plain(static_cast<size_t>(m) * n, -1.0f);
    {
        matmul_params params = make_params(p, /*mem_format_b=*/'n');
        status_t st = zendnnl::lowoha::matmul::matmul_direct('r', false, false,
                m, n, k, 1.0f, src_ptr, lda, wei.data(), ldb,
                /*bias=*/nullptr, 0.0f, dst_plain.data(), ldc,
                /*is_weights_const=*/true, batch_params, params);
        ASSERT_EQ(st, status_t::success);
    }

    // --- Prepack the weights WITH the appended column-sum buffer ---
    reorder_params_t rp;
    rp.is_prepack = true;
    rp.prepack.algo = matmul_algo_t::aocl_dlp_blocked;
    rp.prepack.wei_dtype = data_type_t::s8;
    rp.prepack.src_dtype = p.src_dt;
    rp.prepack.K = k;
    rp.prepack.N = n;
    rp.prepack.ldb = ldb;
    rp.prepack.transposed = false;

    const size_t packed_size
            = zendnnl::lowoha::reorder::weight_prepack_size(rp);
    ASSERT_GT(packed_size, 0u);

    // Size cross-check: the appended buffer begins at the shared colsum offset
    // (the base packed size rounded to 64B) and is round_up(N*4, 64) bytes.
    const size_t colsum_off
            = zendnnl::lowoha::matmul::static_quant_colsum_offset(
                    'r', 'n', k, n);
    EXPECT_EQ(packed_size - colsum_off,
            round_up_align(static_cast<size_t>(n) * sizeof(int32_t)));

    std::vector<uint8_t> packed(packed_size, 0);
    ASSERT_EQ(zendnnl::lowoha::reorder::reorder_direct(
                      wei.data(), packed.data(), rp),
            status_t::success);

    // Column-sum content check: read the appended buffer at the shared offset.
    const int32_t *colsum
            = reinterpret_cast<const int32_t *>(packed.data() + colsum_off);
    for (int j = 0; j < n; ++j) {
        int32_t expect = 0;
        for (int kk = 0; kk < k; ++kk)
            expect += wei[kk * n + j];
        EXPECT_EQ(colsum[j], expect) << "colsum mismatch at n=" << j;
    }

    // --- Prepacked matmul path (reads colsum, applies -src_zp) ---
    std::vector<float> dst_prepacked(static_cast<size_t>(m) * n, -1.0f);
    {
        matmul_params params = make_params(p, /*mem_format_b=*/'r');
        status_t st = zendnnl::lowoha::matmul::matmul_direct('r', false, false,
                m, n, k, 1.0f, src_ptr, lda, packed.data(), ldb,
                /*bias=*/nullptr, 0.0f, dst_prepacked.data(), ldc,
                /*is_weights_const=*/true, batch_params, params);
        ASSERT_EQ(st, status_t::success);
    }

    for (size_t i = 0; i < ref.size(); ++i) {
        EXPECT_FLOAT_EQ(dst_prepacked[i], ref[i])
                << "prepacked vs ref at " << i;
        EXPECT_FLOAT_EQ(dst_prepacked[i], dst_plain[i])
                << "prepacked vs non-prepacked at " << i;
    }
}

// Transposed weights: prepack.transposed = true, consumed with transB = true.
//
// The appended column-sum is produced and consumed through two INDEPENDENT
// derivations of the same quantities, and this is the only test that exercises
// the transposed side of either:
//
//   * Stride. write_weight_colsum() reduces the raw weights with
//     (wei_s0, wei_s1) = transposed ? (1, ldb) : (ldb, 1). A swapped pair
//     silently sums along the wrong axis -- and for k > n it also indexes past
//     the weight buffer.
//   * Offset. The writer locates the tail with
//     static_quant_colsum_offset('r', transposed ? 't' : 'n', K, N) while
//     run_dlp re-derives it from the matmul call's transB. The two must agree.
//
// Runs the same logical matrix three ways -- non-prepacked transposed,
// prepacked transposed, and (for the tail only) prepacked non-transposed --
// so a regression in either derivation shows up as a mismatch rather than as
// plausible-looking wrong numbers.
TEST_P(TestStaticQuantPrepack, TransposedPrepackedMatchesReferenceAndFallback) {
    const PrepackCase p = GetParam();
    const int m = static_cast<int>(p.m), k = static_cast<int>(p.k),
              n = static_cast<int>(p.n);
    src_zp_ = p.src_zp;

    std::mt19937 rng(0xBEEFu ^ (m * 131 + k * 17 + n));
    std::uniform_int_distribution<int> wdist(-8, 7);
    const bool src_is_u8 = (p.src_dt == data_type_t::u8);

    // wei is the logical [K, N] row-major weight; wei_t is the SAME matrix
    // stored column-major (i.e. [N, K] row-major with ldb = K), which is what
    // transposed = true / transB = true describes.
    std::vector<int8_t> wei(static_cast<size_t>(k) * n);
    for (auto &w : wei)
        w = static_cast<int8_t>(wdist(rng));
    std::vector<int8_t> wei_t(static_cast<size_t>(n) * k);
    for (int kk = 0; kk < k; ++kk) {
        for (int j = 0; j < n; ++j) {
            wei_t[static_cast<size_t>(j) * k + kk]
                    = wei[static_cast<size_t>(kk) * n + j];
        }
    }

    std::vector<uint8_t> src_u8;
    std::vector<int8_t> src_s8;
    if (src_is_u8) {
        std::uniform_int_distribution<int> sdist(0, 15);
        src_u8.resize(static_cast<size_t>(m) * k);
        for (auto &s : src_u8)
            s = static_cast<uint8_t>(sdist(rng));
    } else {
        std::uniform_int_distribution<int> sdist(-8, 7);
        src_s8.resize(static_cast<size_t>(m) * k);
        for (auto &s : src_s8)
            s = static_cast<int8_t>(sdist(rng));
    }
    const void *src_ptr = src_is_u8 ? static_cast<const void *>(src_u8.data())
                                    : static_cast<const void *>(src_s8.data());

    // Reference is computed from the logical [K, N] view, so it is layout
    // independent -- both the transposed and non-transposed runs must match it.
    std::vector<float> ref
            = reference_gemm(src_u8, src_s8, src_is_u8, wei, m, k, n, p.src_zp);

    // Transposed weights are [N, K], so the leading dimension is K, not N.
    const int lda = k, ldb_t = k, ldc = n;
    matmul_batch_params_t batch_params;

    // --- Non-prepacked transposed path (existing zp_comp cache flow) ---
    std::vector<float> dst_plain(static_cast<size_t>(m) * n, -1.0f);
    {
        matmul_params params = make_params(p, /*mem_format_b=*/'n');
        status_t st = zendnnl::lowoha::matmul::matmul_direct('r',
                /*transA=*/false, /*transB=*/true, m, n, k, 1.0f, src_ptr, lda,
                wei_t.data(), ldb_t,
                /*bias=*/nullptr, 0.0f, dst_plain.data(), ldc,
                /*is_weights_const=*/true, batch_params, params);
        ASSERT_EQ(st, status_t::success);
    }

    // --- Prepack the TRANSPOSED weights ---
    reorder_params_t rp_t;
    rp_t.is_prepack = true;
    rp_t.prepack.algo = matmul_algo_t::aocl_dlp_blocked;
    rp_t.prepack.wei_dtype = data_type_t::s8;
    rp_t.prepack.src_dtype = p.src_dt;
    rp_t.prepack.K = k;
    rp_t.prepack.N = n;
    rp_t.prepack.ldb = ldb_t;
    rp_t.prepack.transposed = true;

    const size_t packed_size_t
            = zendnnl::lowoha::reorder::weight_prepack_size(rp_t);
    ASSERT_GT(packed_size_t, 0u);

    // Offset cross-check against the 't' spelling -- the same expression
    // run_dlp evaluates when the matmul is issued with transB = true.
    const size_t colsum_off_t
            = zendnnl::lowoha::matmul::static_quant_colsum_offset(
                    'r', 't', k, n);
    EXPECT_EQ(packed_size_t - colsum_off_t,
            round_up_align(static_cast<size_t>(n) * sizeof(int32_t)))
            << "Transposed prepack must reserve exactly the colsum tail past "
               "the "
               "'t'-derived offset";

    std::vector<uint8_t> packed_t(packed_size_t, 0);
    ASSERT_EQ(zendnnl::lowoha::reorder::reorder_direct(
                      wei_t.data(), packed_t.data(), rp_t),
            status_t::success);

    // Column-sum content: colsum[j] must still be sum_k of the LOGICAL column j,
    // independent of how the weights were laid out in memory. This is what a
    // swapped (wei_s0, wei_s1) pair breaks.
    const int32_t *colsum_t
            = reinterpret_cast<const int32_t *>(packed_t.data() + colsum_off_t);
    for (int j = 0; j < n; ++j) {
        int32_t expect = 0;
        for (int kk = 0; kk < k; ++kk) {
            expect += wei[static_cast<size_t>(kk) * n + j];
        }
        EXPECT_EQ(colsum_t[j], expect)
                << "transposed colsum mismatch at n=" << j;
    }

    // --- Same logical matrix prepacked NON-transposed: the tails must agree ---
    // Pins the two stride paths against each other directly, so a regression in
    // either one is a mismatch here even if both still produce a plausible sum.
    {
        reorder_params_t rp_n;
        rp_n.is_prepack = true;
        rp_n.prepack.algo = matmul_algo_t::aocl_dlp_blocked;
        rp_n.prepack.wei_dtype = data_type_t::s8;
        rp_n.prepack.src_dtype = p.src_dt;
        rp_n.prepack.K = k;
        rp_n.prepack.N = n;
        rp_n.prepack.ldb = n;
        rp_n.prepack.transposed = false;

        const size_t packed_size_n
                = zendnnl::lowoha::reorder::weight_prepack_size(rp_n);
        ASSERT_GT(packed_size_n, 0u);
        std::vector<uint8_t> packed_n(packed_size_n, 0);
        ASSERT_EQ(zendnnl::lowoha::reorder::reorder_direct(
                          wei.data(), packed_n.data(), rp_n),
                status_t::success);

        const size_t colsum_off_n
                = zendnnl::lowoha::matmul::static_quant_colsum_offset(
                        'r', 'n', k, n);
        const int32_t *colsum_n = reinterpret_cast<const int32_t *>(
                packed_n.data() + colsum_off_n);
        for (int j = 0; j < n; ++j) {
            EXPECT_EQ(colsum_t[j], colsum_n[j])
                    << "transposed vs non-transposed colsum diverge at n=" << j;
        }
    }

    // --- Prepacked transposed matmul (reads the tail, applies -src_zp) ---
    std::vector<float> dst_prepacked(static_cast<size_t>(m) * n, -1.0f);
    {
        matmul_params params = make_params(p, /*mem_format_b=*/'r');
        status_t st = zendnnl::lowoha::matmul::matmul_direct('r',
                /*transA=*/false, /*transB=*/true, m, n, k, 1.0f, src_ptr, lda,
                packed_t.data(), ldb_t,
                /*bias=*/nullptr, 0.0f, dst_prepacked.data(), ldc,
                /*is_weights_const=*/true, batch_params, params);
        ASSERT_EQ(st, status_t::success);
    }

    for (size_t i = 0; i < ref.size(); ++i) {
        EXPECT_FLOAT_EQ(dst_prepacked[i], ref[i])
                << "transposed prepacked vs ref at " << i;
        EXPECT_FLOAT_EQ(dst_prepacked[i], dst_plain[i])
                << "transposed prepacked vs non-prepacked at " << i;
    }
}

// Symmetric case (src_zp == 0): the prepacked buffer is still appended, but
// matmul must NOT emit the compensation bias (no zero-point to correct).
TEST_P(TestStaticQuantPrepack, SymmetricUnaffectedByColsumBuffer) {
    PrepackCase p = GetParam();
    p.src_zp = 0;
    const int m = static_cast<int>(p.m), k = static_cast<int>(p.k),
              n = static_cast<int>(p.n);
    src_zp_ = 0;

    std::mt19937 rng(0x1234u ^ (m * 7 + k * 13 + n));
    std::uniform_int_distribution<int> wdist(-8, 7);
    const bool src_is_u8 = (p.src_dt == data_type_t::u8);

    std::vector<int8_t> wei(static_cast<size_t>(k) * n);
    for (auto &w : wei)
        w = static_cast<int8_t>(wdist(rng));

    std::vector<uint8_t> src_u8;
    std::vector<int8_t> src_s8;
    if (src_is_u8) {
        std::uniform_int_distribution<int> sdist(0, 15);
        src_u8.resize(static_cast<size_t>(m) * k);
        for (auto &s : src_u8)
            s = static_cast<uint8_t>(sdist(rng));
    } else {
        std::uniform_int_distribution<int> sdist(-8, 7);
        src_s8.resize(static_cast<size_t>(m) * k);
        for (auto &s : src_s8)
            s = static_cast<int8_t>(sdist(rng));
    }
    const void *src_ptr = src_is_u8 ? static_cast<const void *>(src_u8.data())
                                    : static_cast<const void *>(src_s8.data());

    std::vector<float> ref = reference_gemm(
            src_u8, src_s8, src_is_u8, wei, m, k, n, /*src_zp=*/0);

    const int lda = k, ldb = n, ldc = n;
    matmul_batch_params_t batch_params;

    reorder_params_t rp;
    rp.is_prepack = true;
    rp.prepack.algo = matmul_algo_t::aocl_dlp_blocked;
    rp.prepack.wei_dtype = data_type_t::s8;
    rp.prepack.src_dtype = p.src_dt;
    rp.prepack.K = k;
    rp.prepack.N = n;
    rp.prepack.ldb = ldb;
    rp.prepack.transposed = false;
    const size_t packed_size
            = zendnnl::lowoha::reorder::weight_prepack_size(rp);
    ASSERT_GT(packed_size, 0u);
    std::vector<uint8_t> packed(packed_size, 0);
    ASSERT_EQ(zendnnl::lowoha::reorder::reorder_direct(
                      wei.data(), packed.data(), rp),
            status_t::success);

    std::vector<float> dst(static_cast<size_t>(m) * n, -1.0f);
    matmul_params params = make_params(p, /*mem_format_b=*/'r');
    // Symmetric: no source zero-point.
    params.quant_params.src_zp.buff = nullptr;
    status_t st = zendnnl::lowoha::matmul::matmul_direct('r', false, false, m,
            n, k, 1.0f, src_ptr, lda, packed.data(), ldb,
            /*bias=*/nullptr, 0.0f, dst.data(), ldc,
            /*is_weights_const=*/true, batch_params, params);
    ASSERT_EQ(st, status_t::success);

    for (size_t i = 0; i < ref.size(); ++i) {
        EXPECT_FLOAT_EQ(dst[i], ref[i])
                << "symmetric prepacked vs ref at " << i;
    }
}

// 2D "both zero-points" (src_zp != 0 AND wei_zp != 0) with prepacked weights:
// the fast 1D path is skipped (wei_zp != 0), so run_dlp routes through
// cache_or_compute_zp_compensation, feeding the appended column-sum buffer for
// the weight column-sum term (the blocked weight bytes cannot be summed
// directly). Must match both the reference and the non-prepacked path.
TEST_P(TestStaticQuantPrepack, PrepackedBothZeroPointsMatchesFallback) {
    const PrepackCase p = GetParam();
    const int m = static_cast<int>(p.m), k = static_cast<int>(p.k),
              n = static_cast<int>(p.n);
    src_zp_ = p.src_zp;
    wei_zp_ = 4; // non-zero weight zero-point -> 2D "both" path

    std::mt19937 rng(0x5A5Au ^ (m * 31 + k * 7 + n));
    std::uniform_int_distribution<int> wdist(-8, 7);
    const bool src_is_u8 = (p.src_dt == data_type_t::u8);

    std::vector<int8_t> wei(static_cast<size_t>(k) * n);
    for (auto &w : wei)
        w = static_cast<int8_t>(wdist(rng));

    std::vector<uint8_t> src_u8;
    std::vector<int8_t> src_s8;
    if (src_is_u8) {
        std::uniform_int_distribution<int> sdist(0, 15);
        src_u8.resize(static_cast<size_t>(m) * k);
        for (auto &s : src_u8)
            s = static_cast<uint8_t>(sdist(rng));
    } else {
        std::uniform_int_distribution<int> sdist(-8, 7);
        src_s8.resize(static_cast<size_t>(m) * k);
        for (auto &s : src_s8)
            s = static_cast<int8_t>(sdist(rng));
    }
    const void *src_ptr = src_is_u8 ? static_cast<const void *>(src_u8.data())
                                    : static_cast<const void *>(src_s8.data());

    std::vector<float> ref = reference_gemm(
            src_u8, src_s8, src_is_u8, wei, m, k, n, p.src_zp, wei_zp_);

    const int lda = k, ldb = n, ldc = n;
    matmul_batch_params_t batch_params;

    // --- Non-prepacked path (2D compensation from plain weights) ---
    std::vector<float> dst_plain(static_cast<size_t>(m) * n, -1.0f);
    {
        matmul_params params = make_params(p, /*mem_format_b=*/'n');
        status_t st = zendnnl::lowoha::matmul::matmul_direct('r', false, false,
                m, n, k, 1.0f, src_ptr, lda, wei.data(), ldb,
                /*bias=*/nullptr, 0.0f, dst_plain.data(), ldc,
                /*is_weights_const=*/true, batch_params, params);
        ASSERT_EQ(st, status_t::success);
    }

    // --- Prepack the weights WITH the appended column-sum buffer ---
    reorder_params_t rp;
    rp.is_prepack = true;
    rp.prepack.algo = matmul_algo_t::aocl_dlp_blocked;
    rp.prepack.wei_dtype = data_type_t::s8;
    rp.prepack.src_dtype = p.src_dt;
    rp.prepack.K = k;
    rp.prepack.N = n;
    rp.prepack.ldb = ldb;
    rp.prepack.transposed = false;
    const size_t packed_size
            = zendnnl::lowoha::reorder::weight_prepack_size(rp);
    ASSERT_GT(packed_size, 0u);
    std::vector<uint8_t> packed(packed_size, 0);
    ASSERT_EQ(zendnnl::lowoha::reorder::reorder_direct(
                      wei.data(), packed.data(), rp),
            status_t::success);

    // --- Prepacked matmul path (2D via appended colsum) ---
    std::vector<float> dst_prepacked(static_cast<size_t>(m) * n, -1.0f);
    {
        matmul_params params = make_params(p, /*mem_format_b=*/'r');
        status_t st = zendnnl::lowoha::matmul::matmul_direct('r', false, false,
                m, n, k, 1.0f, src_ptr, lda, packed.data(), ldb,
                /*bias=*/nullptr, 0.0f, dst_prepacked.data(), ldc,
                /*is_weights_const=*/true, batch_params, params);
        ASSERT_EQ(st, status_t::success);
    }

    for (size_t i = 0; i < ref.size(); ++i) {
        EXPECT_FLOAT_EQ(dst_prepacked[i], ref[i])
                << "prepacked vs ref at " << i;
        EXPECT_FLOAT_EQ(dst_prepacked[i], dst_plain[i])
                << "prepacked vs non-prepacked at " << i;
    }
}

// The appended weight-sum buffer (and the matmul path that consumes it) is
// now implicit and u8-source only, so these cases cover u8 sources exclusively.
INSTANTIATE_TEST_SUITE_P(StaticQuantPrepack, TestStaticQuantPrepack,
        ::testing::Values(PrepackCase {8, 16, 8, data_type_t::u8, 5},
                PrepackCase {16, 64, 32, data_type_t::u8, 17},
                PrepackCase {32, 128, 64, data_type_t::u8, 3}));

TEST(StaticQuantPrepackGuard, U8SrcWithSymGroupRejected) {
    reorder_params_t rp;
    rp.is_prepack = true;
    rp.prepack.algo = matmul_algo_t::aocl_dlp_blocked;
    rp.prepack.wei_dtype = data_type_t::s8;
    rp.prepack.src_dtype = data_type_t::u8;
    rp.prepack.K = 32;
    rp.prepack.N = 16;
    rp.prepack.ldb = 16;
    rp.prepack.sym_group_size = 16;

    EXPECT_EQ(zendnnl::lowoha::reorder::weight_prepack_size(rp), 0u);
}

// Explicit guard: prepacked s8 weights (mem_format_b == 'r') with an s8 source
// and a non-zero source zero-point are unsupported (the source-zp compensation
// buffer is u8-source only). matmul_direct must reject the call up front rather
// than silently compute a wrong result. A symmetric s8 source (src_zp == 0) on
// the same prepacked buffer stays valid.
TEST(StaticQuantPrepackGuard, S8SrcAsymmetricPrepackedRejected) {
    using zendnnl::common::matmul_algo_t;
    using zendnnl::memory::data_type_t;
    using zendnnl::memory::status_t;

    const int m = 8, k = 32, n = 16;
    const int lda = k, ldb = n, ldc = n;

    std::mt19937 rng(0xC0FFEEu);
    std::uniform_int_distribution<int> wdist(-8, 7);
    std::vector<int8_t> wei(static_cast<size_t>(k) * n);
    for (auto &w : wei)
        w = static_cast<int8_t>(wdist(rng));

    std::uniform_int_distribution<int> sdist(-8, 7);
    std::vector<int8_t> src(static_cast<size_t>(m) * k);
    for (auto &s : src)
        s = static_cast<int8_t>(sdist(rng));

    // Prepack s8 weights (s8 source path -> no appended colsum).
    reorder_params_t rp;
    rp.is_prepack = true;
    rp.prepack.algo = matmul_algo_t::aocl_dlp_blocked;
    rp.prepack.wei_dtype = data_type_t::s8;
    rp.prepack.src_dtype = data_type_t::s8;
    rp.prepack.K = k;
    rp.prepack.N = n;
    rp.prepack.ldb = ldb;
    rp.prepack.transposed = false;
    const size_t packed_size
            = zendnnl::lowoha::reorder::weight_prepack_size(rp);
    ASSERT_GT(packed_size, 0u);
    std::vector<uint8_t> packed(packed_size, 0);
    ASSERT_EQ(zendnnl::lowoha::reorder::reorder_direct(
                      wei.data(), packed.data(), rp),
            status_t::success);

    float one_f = 1.0f;
    int32_t src_zp = 7; // non-zero -> asymmetric

    auto make_s8_params = [&](int32_t zp_value, int32_t *zp_buf) {
        matmul_params params;
        params.dtypes.src = data_type_t::s8;
        params.dtypes.wei = data_type_t::s8;
        params.dtypes.dst = data_type_t::f32;
        params.lowoha_algo = matmul_algo_t::aocl_dlp_blocked;
        params.mem_format_a = 'n';
        params.mem_format_b = 'r';
        params.weight_cache_type = 1;
        params.quant_params.src_scale.buff = &one_f;
        params.quant_params.src_scale.dt = data_type_t::f32;
        params.quant_params.src_scale.dims = {1, 1};
        params.quant_params.wei_scale.buff = &one_f;
        params.quant_params.wei_scale.dt = data_type_t::f32;
        params.quant_params.wei_scale.dims = {1, 1};
        if (zp_buf) {
            *zp_buf = zp_value;
            params.quant_params.src_zp.buff = zp_buf;
            params.quant_params.src_zp.dt = data_type_t::s32;
            params.quant_params.src_zp.dims = {1, 1};
        }
        return params;
    };

    matmul_batch_params_t batch_params;
    std::vector<float> dst(static_cast<size_t>(m) * n, -1.0f);

    // Asymmetric s8 + prepacked -> rejected.
    {
        matmul_params params = make_s8_params(src_zp, &src_zp);
        status_t st = zendnnl::lowoha::matmul::matmul_direct('r', false, false,
                m, n, k, 1.0f, src.data(), lda, packed.data(), ldb,
                /*bias=*/nullptr, 0.0f, dst.data(), ldc,
                /*is_weights_const=*/true, batch_params, params);
        EXPECT_EQ(st, status_t::unimplemented);
    }

    // Symmetric s8 + prepacked (no src zero-point) -> still accepted.
    {
        matmul_params params = make_s8_params(0, /*zp_buf=*/nullptr);
        status_t st = zendnnl::lowoha::matmul::matmul_direct('r', false, false,
                m, n, k, 1.0f, src.data(), lda, packed.data(), ldb,
                /*bias=*/nullptr, 0.0f, dst.data(), ldc,
                /*is_weights_const=*/true, batch_params, params);
        EXPECT_EQ(st, status_t::success);
    }
}
