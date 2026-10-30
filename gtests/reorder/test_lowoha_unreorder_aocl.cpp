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

/// @file test_lowoha_unreorder_aocl.cpp
/// @brief LOWOHA AOCL weight prepack/unreorder round-trips.
///
/// Packs a plain weight, unpacks it, and compares within REORDER_TOL.
/// Numerical cases are TEST_P(TestReorder) on the fixture K/N. They skip
/// without AOCL-DLP; f16 also skips without AVX512-FP16. Dense cases pin
/// m_hint to 0. *_M_HINT pins it to 32, and pack and unpack must share
/// that hint. Guards are plain TEST() with a fixed shape.

#include "reorder_test_common.hpp"

namespace {

// transposed stores the weight as [N, K] and recovers that same layout.
// pad row-pads source and destination (ldb = cols + pad).
// m_hint is the process ZENDNNL_DLP_M_HINT for this call; 0 disables it.
void run_unreorder_roundtrip(tensor_factory_t &tf, const ReorderType &params,
        data_type_t wei_dtype, data_type_t src_dtype, bool transposed,
        uint64_t pad, bool consider_m_hint = true, int64_t m_hint = 0) {
    DlpMHintGuard hint_guard(m_hint);

    // f16 prepack needs AVX512-FP16.
    if (wei_dtype == data_type_t::f16
            && !zendnnl::common::zendnnl_platform_info()
                        .get_avx512_f16_status()) {
        GTEST_SKIP() << "f16 prepack requires the AVX512-FP16 ISA";
    }

    const uint64_t K = params.M;
    const uint64_t N = params.N;
    const uint64_t rows = transposed ? N : K;
    const uint64_t cols = transposed ? K : N;
    const int64_t src_ldb = static_cast<int64_t>(cols + pad);

    tensor_t weight = (pad == 0)
            ? tf.uniform_dist_tensor({rows, cols}, wei_dtype, 2.0f)
            : tf.uniform_dist_strided_tensor(
                      {rows, cols}, {rows, cols + pad}, wei_dtype, 2.0f, false);

    void *packed = nullptr;
    size_t packed_size = 0;
    status_t prep = lowoha_prepack_weight_test(weight, params, wei_dtype,
            src_dtype, transposed, static_cast<int64_t>(K),
            static_cast<int64_t>(N), src_ldb, &packed, &packed_size);
    auto free_packed = [&]() {
        if (packed) {
            zendnnl_aligned_free(packed);
            packed = nullptr;
        }
    };
    if (prep == status_t::isa_unsupported) {
        free_packed();
        GTEST_SKIP() << "requires AVX512-FP16 ISA";
    }
#if !ZENDNNL_DEPENDS_AOCLDLP
    if (prep == status_t::unimplemented) {
        free_packed();
        GTEST_SKIP() << "prepack unavailable (no AOCL-DLP)";
    }
#endif
    EXPECT_EQ(prep, status_t::success);
    if (prep != status_t::success) {
        free_packed();
        return;
    }

    tensor_t recovered = (pad == 0)
            ? tf.zero_tensor({rows, cols}, wei_dtype)
            : tf.uniform_dist_strided_tensor({rows, cols}, {rows, cols + pad},
                      wei_dtype, /*range=*/0.0f, false);
#if ZENDNNL_DEPENDS_AOCLDLP
    {
        reorder_params_t sized;
        sized.is_unreorder = true;
        sized.prepack.algo = matmul_algo_t::aocl_dlp_blocked;
        sized.prepack.wei_dtype = wei_dtype;
        sized.prepack.src_dtype = src_dtype;
        sized.prepack.K = static_cast<int64_t>(K);
        sized.prepack.N = static_cast<int64_t>(N);
        sized.prepack.ldb = src_ldb;
        sized.prepack.transposed = transposed;
        sized.prepack.consider_m_hint = consider_m_hint;
        const size_t elem = zendnnl::common::size_of(wei_dtype);
        const size_t expect = static_cast<size_t>(rows)
                * static_cast<size_t>(src_ldb) * elem;
        EXPECT_EQ(weight_unreorder_size(sized), expect);
    }
#endif
    status_t un = lowoha_unreorder_weight_test(packed, recovered, params,
            wei_dtype, src_dtype, transposed, static_cast<int64_t>(K),
            static_cast<int64_t>(N), src_ldb, consider_m_hint);
#if !ZENDNNL_DEPENDS_AOCLDLP
    if (un == status_t::unimplemented) {
        free_packed();
        GTEST_SKIP() << "unreorder unavailable (no AOCL-DLP)";
    }
#endif
    EXPECT_EQ(un, status_t::success);
    if (un != status_t::success) {
        free_packed();
        return;
    }

    bool is_test_successful = true;
    compare_tensor_2D(
            recovered, weight, rows, cols, REORDER_TOL, is_test_successful);
    EXPECT_TRUE(is_test_successful);

    free_packed();
}

} // namespace

/** @fn TEST_P
 *  @param TestReorder parameterized test class to initialize parameters
 *  @param AOCL_UNREORDER_F32 user-defined name of test
 *  @brief F32 plain -> blocked -> plain round-trip (m_hint = 0)
 */
TEST_P(TestReorder, AOCL_UNREORDER_F32) {
    if (!use_LOWOHA) { GTEST_SKIP(); }
    run_unreorder_roundtrip(tensor_factory, lowoha_params, data_type_t::f32,
            data_type_t::f32, /*transposed=*/false, /*pad=*/0);
}

/** @fn TEST_P
 *  @param TestReorder parameterized test class to initialize parameters
 *  @param AOCL_UNREORDER_BF16 user-defined name of test
 *  @brief BF16 plain -> blocked -> plain round-trip (m_hint = 0)
 */
TEST_P(TestReorder, AOCL_UNREORDER_BF16) {
    if (!use_LOWOHA) { GTEST_SKIP(); }
    run_unreorder_roundtrip(tensor_factory, lowoha_params, data_type_t::bf16,
            data_type_t::bf16, /*transposed=*/false, /*pad=*/0);
}

/** @fn TEST_P
 *  @param TestReorder parameterized test class to initialize parameters
 *  @param AOCL_UNREORDER_S8 user-defined name of test
 *  @brief S8 plain -> blocked -> plain round-trip, s8 source (m_hint = 0)
 */
TEST_P(TestReorder, AOCL_UNREORDER_S8) {
    if (!use_LOWOHA) { GTEST_SKIP(); }
    run_unreorder_roundtrip(tensor_factory, lowoha_params, data_type_t::s8,
            data_type_t::s8, /*transposed=*/false, /*pad=*/0);
}

/** @fn TEST_P
 *  @param TestReorder parameterized test class to initialize parameters
 *  @param AOCL_UNREORDER_F16 user-defined name of test
 *  @brief F16 round-trip. Skips without AVX512-FP16 (m_hint = 0)
 */
TEST_P(TestReorder, AOCL_UNREORDER_F16) {
    if (!use_LOWOHA) { GTEST_SKIP(); }
    run_unreorder_roundtrip(tensor_factory, lowoha_params, data_type_t::f16,
            data_type_t::f16, /*transposed=*/false, /*pad=*/0);
}

/** @fn TEST_P
 *  @param TestReorder parameterized test class to initialize parameters
 *  @param AOCL_UNREORDER_U8SRC user-defined name of test
 *  @brief S8 weights, u8 source. Unreorder returns the s8 weights only
 */
TEST_P(TestReorder, AOCL_UNREORDER_U8SRC) {
    if (!use_LOWOHA) { GTEST_SKIP(); }
    run_unreorder_roundtrip(tensor_factory, lowoha_params, data_type_t::s8,
            data_type_t::u8, /*transposed=*/false, /*pad=*/0);
}

/** @fn TEST_P
 *  @param TestReorder parameterized test class to initialize parameters
 *  @param AOCL_UNREORDER_F32_TRANSPOSED user-defined name of test
 *  @brief F32 transposed round-trip. Recovers the [N, K] layout
 */
TEST_P(TestReorder, AOCL_UNREORDER_F32_TRANSPOSED) {
    if (!use_LOWOHA) { GTEST_SKIP(); }
    run_unreorder_roundtrip(tensor_factory, lowoha_params, data_type_t::f32,
            data_type_t::f32, /*transposed=*/true, /*pad=*/0);
}

/** @fn TEST_P
 *  @param TestReorder parameterized test class to initialize parameters
 *  @param AOCL_UNREORDER_S8_TRANSPOSED user-defined name of test
 *  @brief S8 transposed round-trip. Recovers the [N, K] layout
 */
TEST_P(TestReorder, AOCL_UNREORDER_S8_TRANSPOSED) {
    if (!use_LOWOHA) { GTEST_SKIP(); }
    run_unreorder_roundtrip(tensor_factory, lowoha_params, data_type_t::s8,
            data_type_t::s8, /*transposed=*/true, /*pad=*/0);
}

/** @fn TEST_P
 *  @param TestReorder parameterized test class to initialize parameters
 *  @param AOCL_UNREORDER_F32_STRIDED user-defined name of test
 *  @brief F32 round-trip with ldb > N on source and destination
 */
TEST_P(TestReorder, AOCL_UNREORDER_F32_STRIDED) {
    if (!use_LOWOHA) { GTEST_SKIP(); }
    const uint64_t pad = 1 + lowoha_params.row_padding;
    run_unreorder_roundtrip(tensor_factory, lowoha_params, data_type_t::f32,
            data_type_t::f32, /*transposed=*/false, pad);
}

/** @fn TEST_P
 *  @param TestReorder parameterized test class to initialize parameters
 *  @param AOCL_UNREORDER_S8_STRIDED user-defined name of test
 *  @brief S8 round-trip with ldb > N on source and destination
 */
TEST_P(TestReorder, AOCL_UNREORDER_S8_STRIDED) {
    if (!use_LOWOHA) { GTEST_SKIP(); }
    const uint64_t pad = 1 + lowoha_params.row_padding;
    run_unreorder_roundtrip(tensor_factory, lowoha_params, data_type_t::s8,
            data_type_t::s8, /*transposed=*/false, pad);
}

/** @fn TEST_P
 *  @param TestReorder parameterized test class to initialize parameters
 *  @param AOCL_UNREORDER_BF16_M_HINT user-defined name of test
 *  @brief BF16 round-trip with m_hint = 32. Pack and unpack share it
 */
TEST_P(TestReorder, AOCL_UNREORDER_BF16_M_HINT) {
    if (!use_LOWOHA) { GTEST_SKIP(); }
    run_unreorder_roundtrip(tensor_factory, lowoha_params, data_type_t::bf16,
            data_type_t::bf16, /*transposed=*/false, /*pad=*/0,
            /*consider_m_hint=*/true, /*m_hint=*/32);
}

/** @fn TEST_P
 *  @param TestReorder parameterized test class to initialize parameters
 *  @param AOCL_UNREORDER_S8_M_HINT user-defined name of test
 *  @brief S8 round-trip with m_hint = 32. Pack and unpack share it
 */
TEST_P(TestReorder, AOCL_UNREORDER_S8_M_HINT) {
    if (!use_LOWOHA) { GTEST_SKIP(); }
    run_unreorder_roundtrip(tensor_factory, lowoha_params, data_type_t::s8,
            data_type_t::s8, /*transposed=*/false, /*pad=*/0,
            /*consider_m_hint=*/true, /*m_hint=*/32);
}

/** @fn TEST_P
 *  @param TestReorder parameterized test class to initialize parameters
 *  @param AOCL_UNREORDER_U8SRC_M_HINT user-defined name of test
 *  @brief U8-source s8 round-trip with m_hint = 32
 */
TEST_P(TestReorder, AOCL_UNREORDER_U8SRC_M_HINT) {
    if (!use_LOWOHA) { GTEST_SKIP(); }
    run_unreorder_roundtrip(tensor_factory, lowoha_params, data_type_t::s8,
            data_type_t::u8, /*transposed=*/false, /*pad=*/0,
            /*consider_m_hint=*/true, /*m_hint=*/32);
}

/** @fn TEST_P
 *  @param TestReorder parameterized test class to initialize parameters
 *  @param AOCL_UNREORDER_M_HINT_OPT_OUT user-defined name of test
 *  @brief consider_m_hint = false unpacks a pack that also omitted the hint
 */
TEST_P(TestReorder, AOCL_UNREORDER_M_HINT_OPT_OUT) {
    if (!use_LOWOHA) { GTEST_SKIP(); }
    DlpMHintGuard guard(0);
    const uint64_t K = lowoha_params.M;
    const uint64_t N = lowoha_params.N;
    tensor_t weight = tensor_factory.uniform_dist_tensor(
            {K, N}, data_type_t::bf16, 2.0f);
    void *packed = nullptr;
    size_t packed_size = 0;
    status_t prep = lowoha_prepack_weight_test(weight, lowoha_params,
            data_type_t::bf16, data_type_t::bf16, /*transposed=*/false,
            static_cast<int64_t>(K), static_cast<int64_t>(N),
            static_cast<int64_t>(N), &packed, &packed_size);
    auto free_packed = [&]() {
        if (packed) {
            zendnnl_aligned_free(packed);
            packed = nullptr;
        }
    };
#if !ZENDNNL_DEPENDS_AOCLDLP
    if (prep == status_t::unimplemented) {
        free_packed();
        GTEST_SKIP() << "prepack unavailable (no AOCL-DLP)";
    }
#endif
    ASSERT_EQ(prep, status_t::success);

    matmul_config_t::instance().set_dlp_m_hint(32);
    tensor_t recovered = tensor_factory.zero_tensor({K, N}, data_type_t::bf16);
    status_t un = lowoha_unreorder_weight_test(packed, recovered, lowoha_params,
            data_type_t::bf16, data_type_t::bf16, /*transposed=*/false,
            static_cast<int64_t>(K), static_cast<int64_t>(N),
            static_cast<int64_t>(N), /*consider_m_hint=*/false);
    EXPECT_EQ(un, status_t::success);
    bool is_test_successful = un == status_t::success;
    if (is_test_successful) {
        compare_tensor_2D(
                recovered, weight, K, N, REORDER_TOL, is_test_successful);
    }
    EXPECT_TRUE(is_test_successful);
    free_packed();
}

namespace {

constexpr int64_t kGuardK = 8;
constexpr int64_t kGuardN = 16;

reorder_params_t make_unreorder_params(data_type_t wei_dtype, int64_t K,
        int64_t N, int64_t ldb,
        matmul_algo_t algo = matmul_algo_t::aocl_dlp_blocked,
        data_type_t src_dtype = data_type_t::f32, int sym_group_size = 0) {
    reorder_params_t rp;
    rp.is_unreorder = true;
    rp.prepack.algo = algo;
    rp.prepack.wei_dtype = wei_dtype;
    rp.prepack.src_dtype = src_dtype;
    rp.prepack.K = K;
    rp.prepack.N = N;
    rp.prepack.ldb = ldb;
    rp.prepack.sym_group_size = sym_group_size;
    return rp;
}

// ──────────────────────────────────────────────────────────────────
// is_prepack and is_unreorder together -> failure.
// ──────────────────────────────────────────────────────────────────
TEST(AoclUnreorderGuard, BothFlagsFail) {
    reorder_params_t rp;
    rp.is_prepack = true;
    rp.is_unreorder = true;
    rp.prepack.algo = matmul_algo_t::aocl_dlp_blocked;
    rp.prepack.wei_dtype = data_type_t::f32;
    rp.prepack.K = kGuardK;
    rp.prepack.N = kGuardN;
    rp.prepack.ldb = kGuardN;

    tensor_factory_t tf;
    auto a = tf.uniform_dist_tensor(
            {static_cast<uint64_t>(kGuardK), static_cast<uint64_t>(kGuardN)},
            data_type_t::f32, 2.0f);
    auto b = tf.zero_tensor(
            {static_cast<uint64_t>(kGuardK), static_cast<uint64_t>(kGuardN)},
            data_type_t::f32);
    EXPECT_EQ(reorder_direct(
                      a.get_raw_handle_unsafe(), b.get_raw_handle_unsafe(), rp),
            status_t::failure);
}

// ──────────────────────────────────────────────────────────────────
// Only aocl_dlp_blocked can be unpacked. Other algos -> size 0.
// ──────────────────────────────────────────────────────────────────
TEST(AoclUnreorderGuard, UnsupportedAlgoSizeZero) {
    tensor_factory_t tf;
    auto dst = tf.zero_tensor(
            {static_cast<uint64_t>(kGuardK), static_cast<uint64_t>(kGuardN)},
            data_type_t::f32);
    for (matmul_algo_t algo :
            {matmul_algo_t::moe_custom_kernel, matmul_algo_t::onednn_blocked,
                    matmul_algo_t::libxsmm_blocked}) {
        reorder_params_t rp = make_unreorder_params(
                data_type_t::f32, kGuardK, kGuardN, kGuardN, algo);
        EXPECT_EQ(weight_unreorder_size(rp), 0u)
                << "algo " << static_cast<int>(algo)
                << " must not be unreorderable";
        char dummy = 0;
        EXPECT_EQ(reorder_direct(&dummy, dst.get_raw_handle_unsafe(), rp),
                status_t::unimplemented)
                << "algo " << static_cast<int>(algo);
    }
}

// ──────────────────────────────────────────────────────────────────
// s4 and s8 sym-quant have no unreorder kernel.
// ──────────────────────────────────────────────────────────────────
TEST(AoclUnreorderGuard, UnsupportedDtypeSizeZero) {
    tensor_factory_t tf;
    char dummy = 0;

    reorder_params_t s4
            = make_unreorder_params(data_type_t::s4, kGuardK, kGuardN, kGuardN);
    EXPECT_EQ(weight_unreorder_size(s4), 0u) << "s4 must be rejected";
    auto dst_s4 = tf.zero_tensor(
            {static_cast<uint64_t>(kGuardK), static_cast<uint64_t>(kGuardN)},
            data_type_t::s4);
    EXPECT_EQ(reorder_direct(&dummy, dst_s4.get_raw_handle_unsafe(), s4),
            status_t::unimplemented);

    reorder_params_t sym = make_unreorder_params(data_type_t::s8, kGuardK,
            kGuardN, kGuardN, matmul_algo_t::aocl_dlp_blocked,
            data_type_t::bf16, /*sym_group_size=*/16);
    EXPECT_EQ(weight_unreorder_size(sym), 0u) << "sym-quant must be rejected";
    auto dst_s8 = tf.zero_tensor(
            {static_cast<uint64_t>(kGuardK), static_cast<uint64_t>(kGuardN)},
            data_type_t::s8);
    EXPECT_EQ(reorder_direct(&dummy, dst_s8.get_raw_handle_unsafe(), sym),
            status_t::unimplemented);
}

// ──────────────────────────────────────────────────────────────────
// ldb < N, K or N <= 0, or a null pointer -> failure.
// ──────────────────────────────────────────────────────────────────
TEST(AoclUnreorderGuard, BadParamsFail) {
    tensor_factory_t tf;
    char dummy = 0;
    auto dst = tf.zero_tensor({8, 16}, data_type_t::f32);

    // ldb < N.
    reorder_params_t bad_ldb = make_unreorder_params(
            data_type_t::f32, kGuardK, kGuardN, /*ldb=*/4);
    EXPECT_EQ(weight_unreorder_size(bad_ldb), 0u);
    EXPECT_EQ(reorder_direct(&dummy, dst.get_raw_handle_unsafe(), bad_ldb),
            status_t::failure);

    // K or N <= 0.
    reorder_params_t bad_k = make_unreorder_params(
            data_type_t::f32, /*K=*/0, kGuardN, kGuardN);
    EXPECT_EQ(weight_unreorder_size(bad_k), 0u);
    EXPECT_EQ(reorder_direct(&dummy, dst.get_raw_handle_unsafe(), bad_k),
            status_t::failure);
    reorder_params_t bad_n = make_unreorder_params(
            data_type_t::f32, kGuardK, /*N=*/0, kGuardN);
    EXPECT_EQ(weight_unreorder_size(bad_n), 0u);
    EXPECT_EQ(reorder_direct(&dummy, dst.get_raw_handle_unsafe(), bad_n),
            status_t::failure);

    // Null packed pointer.
    reorder_params_t rp = make_unreorder_params(
            data_type_t::f32, kGuardK, kGuardN, kGuardN);
    EXPECT_EQ(reorder_direct(nullptr, dst.get_raw_handle_unsafe(), rp),
            status_t::failure);

    // Null destination.
    EXPECT_EQ(reorder_direct(&dummy, nullptr, rp), status_t::failure);
}

} // namespace
