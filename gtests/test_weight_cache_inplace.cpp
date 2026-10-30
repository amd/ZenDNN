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
// In-place weight cache vs. INT8 zero-point compensation.
//
// Under ZENDNNL_MATMUL_WEIGHT_CACHE=2 the AOCL-DLP blocked matmul reorders the
// caller's weight buffer IN PLACE on the first call and reuses the mutated
// buffer on later calls. The INT8 zero-point compensation is derived from the
// PLAIN (pre-reorder) weights, so if it is (re)computed after the in-place
// reorder it reads scrambled bytes and miscomputes.
//
// This test drives the full u8*s8 quantized matmul through the same harness as
// test_matmul.cpp (matmul_kernel_test, which does a warmup + real call on the
// SAME weight buffer, then matmul_kernel_test(..., use_reference=true) for the
// golden baseline),
// pinned to WEIGHT_CACHE=2 so the in-place reorder path is exercised. It covers
// the common case: u8 activations (nonzero src_zp) + symmetric s8 weights
// (wei_zp == 0). The matmul is run several times on the same in-place-reordered
// weight buffer to confirm accuracy holds across cache reuse, not just the
// first warmup+real pair.
//
// DeclaredCapacity* / MatmulDirect*: WC=2 rewrites a padded INT8 buffer
// when the blocked image exceeds k*n, and leaves an undeclared buffer raw.
// BmmSharedWeight*: shared tiles cap the weight cache at 1 so a source
// zero-point sum still sees raw columns, including when the blocked size
// equals k*n; a caller's 0 stays 0.
//
// The two related vulnerable paths (asymmetric wei_zp != 0, and ZP-comp caching
// disabled) are documented as a TODO next to the compensation code in
// aocl_kernel.cpp (run_dlp).
//
// WeightCacheGuard / ZpCompCacheGuard are shared RAII helpers from
// gtest_utils.hpp.
// =============================================================================

#include <gtest/gtest.h>

#include <cstring>
#include <vector>

#include "common/op_config.hpp"
#include "gtest_utils.hpp"
#include "lowoha_operators/matmul/backends/aocl/aocl_kernel.hpp"

namespace {

class TestWeightCacheInplace : public ::testing::Test {
protected:
    void TearDown() override { clear_matmul_test_caches(); }
    tensor_factory_t tensor_factory {};
};

// The scalar custom-kernel route caps a u8×s8 weight at weight-cache mode 1
// so a later grouped pack still sees raw bytes. These direct tests turn that
// route off; an environment variable is too late because the flag is cached.
struct CustomKernelRouteGuard {
    zendnnl::common::matmul_config_t &config
            = zendnnl::common::matmul_config_t::instance();
    bool previous = config.get_custom_kernel_route();
    explicit CustomKernelRouteGuard(bool enable) {
        config.set_custom_kernel_route(enable);
    }
    ~CustomKernelRouteGuard() { config.set_custom_kernel_route(previous); }
};

} // namespace

// Common case: u8 activations (nonzero src_zp) x symmetric s8 weights
// (wei_zp == 0) -> 1D zero-point compensation. Run under WEIGHT_CACHE=2 (in-place
// reorder + weight-buffer reuse across the warmup and real call) and compare
// against the reference kernel. The 1D compensation is computed from the plain
// weights and cached, so the in-place reorder cannot corrupt it.
TEST_F(TestWeightCacheInplace, Cached1D_InPlaceWeightMutation) {
    const uint64_t m = 32, k = 256, n = 256;
    const bool use_LOWOHA = true;
    const auto algo = matmul_algo_t::aocl_dlp_blocked;
    const std::vector<post_op_type_t> po_types {};
    const std::vector<tensor_t> binary_tensors {};

    // Symmetric s8 weights (per-tensor scale, wei_zp == 0).
    auto wei_ref = tensor_factory.uniform_dist_tensor(
            {k, n}, data_type_t::bf16, 25.0, /*transB=*/false);
    tensor_t weight_tensor, wei_scale, wei_zp;
    ASSERT_EQ(quant_params_compute(tensor_factory, wei_ref, data_type_t::bf16,
                      data_type_t::s8, {1, 1}, data_type_t::f32, wei_scale,
                      wei_zp, &weight_tensor),
            status_t::success)
            << "weight quantization failed";

    // u8 activations -> asymmetric -> nonzero src_zp.
    auto src_ref = tensor_factory.uniform_dist_tensor(
            {m, k}, data_type_t::bf16, 25.0, /*transA=*/false);
    tensor_t input_tensor, src_scale, src_zp;
    ASSERT_EQ(quant_params_compute(tensor_factory, src_ref, data_type_t::bf16,
                      data_type_t::u8, {1, 1}, data_type_t::f32, src_scale,
                      src_zp, &input_tensor),
            status_t::success)
            << "source quantization failed";

    auto bias_tensor
            = tensor_factory.uniform_dist_tensor({1, n}, data_type_t::f32, 2.0);
    auto output_tensor = tensor_factory.uniform_dist_tensor(
            {m, n}, data_type_t::bf16, 2.0);
    auto output_tensor_ref = tensor_factory.uniform_dist_tensor(
            {m, n}, data_type_t::bf16, 2.0);

    // Reference FIRST, on the pristine weight buffer: the WEIGHT_CACHE=2 run below
    // reorders those weights IN PLACE, so computing the reference afterwards would
    // read the mutated (blocked) bytes and produce a wrong baseline.
    status_t ref_status = matmul_kernel_test(input_tensor, weight_tensor,
            bias_tensor, output_tensor_ref, po_types, binary_tensors,
            use_LOWOHA, algo, 1.0, 0.0, true);
    ASSERT_EQ(ref_status, status_t::success) << "reference kernel failed";
    clear_matmul_test_caches();

    // Pin WEIGHT_CACHE=2 (in-place reorder + weight-buffer reuse) and force the
    // 1D ZP-comp cache ON so the assertion below is not perturbed by an external
    // ZENDNNL_ZP_COMP_CACHE=0 in the environment. Both are restored on scope exit.
    WeightCacheGuard wc_guard(2);
    ZpCompCacheGuard zp_guard(true);

    // Run the matmul repeatedly on the SAME weight buffer. matmul_kernel_test
    // warms up (reordering the weights in place on the first touch) and runs the
    // real matmul; subsequent iterations reuse the in-place-reordered weights and
    // the cached 1D compensation. Every iteration must match the reference — if
    // the compensation were derived from the reordered bytes, reuse would diverge.
    constexpr int kIterations = 3;
    for (int iter = 0; iter < kIterations; ++iter) {
        status_t status = matmul_kernel_test(input_tensor, weight_tensor,
                bias_tensor, output_tensor, po_types, binary_tensors,
                use_LOWOHA, algo, 1.0, 0.0);
        if (status == status_t::isa_unsupported) {
            GTEST_SKIP() << "AOCL-DLP blocked INT8 not supported on this ISA";
        }
        ASSERT_EQ(status, status_t::success)
                << "matmul failed on iteration " << iter;

        bool ok = true;
        compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m, n, k,
                rtol_bf16, epsilon_bf16, ok, false, 1.0f,
                /*is_quant=*/true);
        EXPECT_TRUE(ok)
                << "WEIGHT_CACHE=2 output diverged from the reference on "
                   "iteration "
                << iter
                << " — in-place reorder corrupted "
                   "the zero-point compensation";
    }
}

// The AOCL DLP m_hint / nt_hint pair changes the packed panel width, so
// run_dlp() folds it into the weight-cache key (extra_input_hash). Drive the
// pack cache directly to pin that contract: the same key must reuse the packed
// buffer, and a key carrying the hint fold must produce a separate pack instead
// of the layout built without hints.
TEST_F(TestWeightCacheInplace, HintFoldedKeyPacksSeparately) {
    constexpr int k = 256, n = 256;
    const auto algo = static_cast<uint32_t>(matmul_algo_t::aocl_dlp_blocked);

    auto weights = tensor_factory.uniform_dist_tensor(
            {static_cast<uint64_t>(k), static_cast<uint64_t>(n)},
            data_type_t::bf16, 2.0, /*transB=*/false);
    const void *weight_ptr = weights.get_raw_handle_unsafe();

    const Key_matmul unhinted(/*TransB=*/false, k, n, /*ldb=*/n, weight_ptr,
            algo, /*extra_input_hash=*/0);
    std::size_t hinted_hash = 0;
    hinted_hash = zendnnl::common::hash_combine(hinted_hash, int64_t {32});
    hinted_hash = zendnnl::common::hash_combine(hinted_hash, int64_t {24});
    const Key_matmul hinted(
            /*TransB=*/false, k, n, /*ldb=*/n, weight_ptr, algo, hinted_hash);

    // weight_cache_type 1: packs live in LRU-owned buffers (freed by
    // clear_matmul_test_caches() in TearDown) and the plain weights stay
    // untouched, so every call below packs from the same source bytes.
    auto pack = [&](const Key_matmul &key) {
        void *packed = nullptr;
        zendnnl::lowoha::matmul::reorderAndCacheWeights<int16_t>(key,
                weight_ptr, packed, k, n, /*ldb=*/n, /*order=*/'r',
                /*trans=*/'n', /*mem_format_b=*/'n',
                aocl_get_reorder_buf_size_bf16bf16f32of32,
                aocl_reorder_bf16bf16f32of32, /*weight_cache_type=*/1);
        return packed;
    };

    void *unhinted_pack = pack(unhinted);
    ASSERT_NE(unhinted_pack, nullptr) << "unhinted weight pack failed";

    EXPECT_EQ(pack(unhinted), unhinted_pack)
            << "same key must hit the cache and reuse the packed buffer";

    EXPECT_NE(pack(hinted), unhinted_pack)
            << "hint-folded key reused the pack built without hints — a hint "
               "change must re-pack";
}

// WC=2 with a declared capacity rewrites the caller's INT8 buffer. The
// blocked image must match the out-of-place pack, and bytes past it stay
// untouched. Cap 0 stays out-of-place for this shape.
TEST_F(TestWeightCacheInplace, DeclaredCapacityRewritesInt8BufferInPlace) {
    constexpr int k = 128, n = 128;
    const size_t plain = static_cast<size_t>(k) * static_cast<size_t>(n);
    const size_t req = aocl_get_reorder_buf_size_s8s8s32os32(
            'r', 'n', 'B', k, n, nullptr);
    ASSERT_GT(req, 0u) << "AOCL INT8 reorder size query failed";
    const size_t reorder_size = (req + 63u) & ~static_cast<size_t>(63u);
    const size_t capacity = reorder_size + 4096u;

    std::vector<int8_t> pristine(plain);
    for (size_t i = 0; i < pristine.size(); ++i) {
        pristine[i] = static_cast<int8_t>((i * 37) % 251 - 125);
    }

    const auto pack
            = [&](std::vector<int8_t> &buf, int32_t wc, size_t cap) -> void * {
        void *out = nullptr;
        Key_matmul key(/*TransB=*/false, k, n, /*ldb=*/n, buf.data(),
                static_cast<uint32_t>(matmul_algo_t::aocl_dlp_blocked),
                /*extra_input_hash=*/0);
        EXPECT_TRUE(reorderAndCacheWeights<int8_t>(key, buf.data(), out, k, n,
                /*ldb=*/n, /*order=*/'r', /*trans=*/'n', /*mem_format_b=*/'n',
                aocl_get_reorder_buf_size_s8s8s32os32, aocl_reorder_s8s8s32os32,
                wc, cap, nullptr));
        return out;
    };

    std::vector<int8_t> oop_buf(pristine);
    void *oop = pack(oop_buf, /*wc=*/1, /*cap=*/0);
    ASSERT_NE(oop, nullptr);
    ASSERT_NE(oop, oop_buf.data())
            << "WC=1 must leave the caller's buffer untouched";

    std::vector<int8_t> ip_buf(capacity, 0);
    std::memcpy(ip_buf.data(), pristine.data(), plain);
    // Canary past the blocked image.
    std::memset(ip_buf.data() + req, 0xA5, capacity - req);
    ASSERT_TRUE(wei_inplace_fits(req, reorder_size, plain, capacity))
            << "test capacity " << capacity << " must cover blocked "
            << reorder_size << " (req=" << req << ", plain=" << plain << ")";
    void *ip = pack(ip_buf, /*wc=*/2, capacity);
    ASSERT_EQ(ip, ip_buf.data())
            << "declared capacity under WC=2 must reorder INTO the caller's "
               "buffer; if this fails the comparison below is vacuous because "
               "no in-place reorder happened";

    EXPECT_EQ(std::memcmp(oop, ip, req), 0)
            << "the in-place blocked image differs from the out-of-place one "
               "over "
            << req
            << " bytes -- same reorder, same input, different destination";

    for (size_t i = req; i < ip_buf.size(); ++i) {
        ASSERT_EQ(static_cast<uint8_t>(ip_buf[i]), 0xA5u)
                << "in-place reorder wrote past its blocked image at byte " << i
                << " (declared need was " << req << ")";
    }

    // Cap 0 must stay out-of-place for this shape.
    ASSERT_FALSE(wei_inplace_fits(req, reorder_size, plain, /*declared_cap=*/0))
            << "s8s8 " << k << "x" << n
            << " blocked image fits k*n; the undeclared control would not "
               "prove the capacity gate";
    std::vector<int8_t> und_buf(pristine);
    void *und = pack(und_buf, /*wc=*/2, /*cap=*/0);
    ASSERT_NE(und, und_buf.data())
            << "undeclared INT8 under WC=2 must stay out-of-place when "
               "the blocked image does not fit k*n";
    EXPECT_EQ(std::memcmp(und_buf.data(), pristine.data(), plain), 0)
            << "out-of-place fallback mutated the caller's weight buffer";
    EXPECT_EQ(std::memcmp(und, oop, req), 0);
}

// 130x130 u8 is larger than k*n, so cap 0 stays raw and a declared
// capacity mutates. The output must still match the reference.
// The custom-kernel route is off so this call is not capped at WC=1.
TEST_F(TestWeightCacheInplace, MatmulDirectDeclaredCapacityInPlaceMatchesRef) {
    const uint64_t m = 32, k = 130, n = 130;
    const bool use_LOWOHA = true;
    const auto algo = matmul_algo_t::aocl_dlp_blocked;
    const std::vector<post_op_type_t> po_types {};
    const std::vector<tensor_t> binary_tensors {};

    auto wei_ref = tensor_factory.uniform_dist_tensor(
            {k, n}, data_type_t::bf16, 25.0, /*transB=*/false);
    tensor_t weight_tensor, wei_scale, wei_zp;
    ASSERT_EQ(quant_params_compute(tensor_factory, wei_ref, data_type_t::bf16,
                      data_type_t::s8, {1, 1}, data_type_t::f32, wei_scale,
                      wei_zp, &weight_tensor),
            status_t::success)
            << "weight quantization failed";

    auto src_ref = tensor_factory.uniform_dist_tensor(
            {m, k}, data_type_t::bf16, 25.0, /*transA=*/false);
    tensor_t input_tensor, src_scale, src_zp;
    ASSERT_EQ(quant_params_compute(tensor_factory, src_ref, data_type_t::bf16,
                      data_type_t::u8, {1, 1}, data_type_t::f32, src_scale,
                      src_zp, &input_tensor),
            status_t::success)
            << "source quantization failed";

    auto bias_tensor
            = tensor_factory.uniform_dist_tensor({1, n}, data_type_t::f32, 2.0);
    auto output_tensor = tensor_factory.uniform_dist_tensor(
            {m, n}, data_type_t::bf16, 2.0);
    auto output_tensor_ref = tensor_factory.uniform_dist_tensor(
            {m, n}, data_type_t::bf16, 2.0);

    status_t ref_status = matmul_kernel_test(input_tensor, weight_tensor,
            bias_tensor, output_tensor_ref, po_types, binary_tensors,
            use_LOWOHA, algo, 1.0, 0.0, true);
    ASSERT_EQ(ref_status, status_t::success) << "reference kernel failed";
    clear_matmul_test_caches();

    const int ki = static_cast<int>(k), ni = static_cast<int>(n);
    const size_t plain = static_cast<size_t>(ki) * static_cast<size_t>(ni);
    const size_t req = aocl_get_reorder_buf_size_u8s8s32os32(
            'r', 'n', 'B', ki, ni, nullptr);
    ASSERT_GT(req, 0u);
    const size_t reorder_size = (req + 63u) & ~static_cast<size_t>(63u);
    const size_t capacity = reorder_size + 4096u;
    ASSERT_TRUE(wei_inplace_fits(req, reorder_size, plain, capacity));
    ASSERT_FALSE(wei_inplace_fits(req, reorder_size, plain, /*declared_cap=*/0))
            << "u8s8 " << k << "x" << n
            << " fits k*n with no declared capacity, so this test cannot "
               "tell a forwarded capacity from the historical equality gate";

    std::vector<int8_t> padded(capacity, 0);
    std::memcpy(padded.data(), weight_tensor.get_raw_handle_unsafe(), plain);
    const std::vector<int8_t> pristine(padded.begin(),
            padded.begin() + static_cast<std::ptrdiff_t>(plain));

    auto fill_q = [](matmul_quantization_params_t::matmul_quant_t &q,
                          const tensor_t &t, bool scale) {
        if (!t.is_quantized()) return;
        if (scale) {
            q.buff = t.get_quant_scale_raw_handle_const();
            q.dt = t.get_quant_scale_data_type();
            auto dims = t.get_quant_scale_size();
            q.dims.assign(dims.begin(), dims.end());
            return;
        }
        if (t.get_quant_subtype() != quant_subtype_t::asymmetric) return;
        q.buff = t.get_quant_zero_raw_handle_const();
        if (!q.buff) return;
        q.dt = t.get_quant_zero_data_type();
        auto dims = t.get_quant_zero_size();
        q.dims.assign(dims.begin(), dims.end());
    };

    matmul_params params;
    params.lowoha_algo = algo;
    params.dtypes.src = data_type_t::u8;
    params.dtypes.wei = data_type_t::s8;
    params.dtypes.dst = data_type_t::bf16;
    params.dtypes.bias = data_type_t::f32;
    params.weight_cache_type = 2;
    params.wei_buffer_capacity_bytes = capacity;
    fill_q(params.quant_params.src_scale, input_tensor, true);
    fill_q(params.quant_params.src_zp, input_tensor, false);
    fill_q(params.quant_params.wei_scale, weight_tensor, true);
    fill_q(params.quant_params.wei_zp, weight_tensor, false);

    WeightCacheGuard wc_guard(2);
    ZpCompCacheGuard zp_guard(true);
    CustomKernelRouteGuard route_off(false);
    matmul_batch_params_t batch;

    std::vector<int8_t> plain_buf(pristine);
    matmul_params cap0_params = params;
    cap0_params.wei_buffer_capacity_bytes = 0;
    status_t cap0_status = matmul_direct('r', /*transA=*/false,
            /*transB=*/false, static_cast<int>(m), ni, ki, 1.0f,
            input_tensor.get_raw_handle_unsafe(), ki, plain_buf.data(), ni,
            bias_tensor.get_raw_handle_unsafe(), 0.0f,
            output_tensor.get_raw_handle_unsafe(), ni,
            /*is_weights_const=*/true, batch, cap0_params);
    if (cap0_status == status_t::isa_unsupported) {
        GTEST_SKIP() << "AOCL-DLP blocked INT8 not supported on this ISA";
    }
    ASSERT_EQ(cap0_status, status_t::success);
    EXPECT_EQ(std::memcmp(plain_buf.data(), pristine.data(), plain), 0)
            << "undeclared capacity mutated the caller's weight buffer";
    clear_matmul_test_caches();

    status_t status = matmul_direct('r', /*transA=*/false, /*transB=*/false,
            static_cast<int>(m), ni, ki, 1.0f,
            input_tensor.get_raw_handle_unsafe(), ki, padded.data(), ni,
            bias_tensor.get_raw_handle_unsafe(), 0.0f,
            output_tensor.get_raw_handle_unsafe(), ni,
            /*is_weights_const=*/true, batch, params);
    if (status == status_t::isa_unsupported) {
        GTEST_SKIP() << "AOCL-DLP blocked INT8 not supported on this ISA";
    }
    ASSERT_EQ(status, status_t::success);

    EXPECT_NE(std::memcmp(padded.data(), pristine.data(), plain), 0)
            << "matmul_direct left the padded INT8 weight raw under WC=2 "
               "with a declared capacity that covers the blocked image";

    bool ok = true;
    compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m, n, k,
            rtol_bf16, epsilon_bf16, ok, false, 1.0f, /*is_quant=*/true);
    EXPECT_TRUE(ok) << "in-place matmul_direct output diverged from the "
                       "reference";
}

// Same shape as the test above, but with ZP-comp caching DISABLED — one of the
// two paths the TODO next to the compensation code called out as unguarded.
//
// With caching off the 1D compensation is recomputed from the weight buffer on
// every call, so it is no longer computed once from the pristine bytes. Under
// WEIGHT_CACHE=2 an in-place reorder would leave iterations 2+ summing over
// blocked bytes and silently returning wrong output. run_dlp now refuses
// in-place for int8 whenever the compensation is not cacheable
// (`wei_zp == 0 && is_weights_const && get_zp_comp_cache()`), which is what
// keeps every iteration on the reference here.
//
// The sibling path — asymmetric weights with `wei_zp != 0` — takes the same
// refusal branch but is not expressible through this harness:
// `quant_params_compute` only emits a zero-point for a u8 destination, so s8
// weights always come back symmetric.
TEST_F(TestWeightCacheInplace, Uncached1D_ZpCompCacheOff_InPlaceRefused) {
    const uint64_t m = 32, k = 256, n = 256;
    const bool use_LOWOHA = true;
    const auto algo = matmul_algo_t::aocl_dlp_blocked;
    const std::vector<post_op_type_t> po_types {};
    const std::vector<tensor_t> binary_tensors {};

    auto wei_ref = tensor_factory.uniform_dist_tensor(
            {k, n}, data_type_t::bf16, 25.0, /*transB=*/false);
    tensor_t weight_tensor, wei_scale, wei_zp;
    ASSERT_EQ(quant_params_compute(tensor_factory, wei_ref, data_type_t::bf16,
                      data_type_t::s8, {1, 1}, data_type_t::f32, wei_scale,
                      wei_zp, &weight_tensor),
            status_t::success)
            << "weight quantization failed";

    // u8 activations -> nonzero src_zp -> a compensation term is required.
    auto src_ref = tensor_factory.uniform_dist_tensor(
            {m, k}, data_type_t::bf16, 25.0, /*transA=*/false);
    tensor_t input_tensor, src_scale, src_zp;
    ASSERT_EQ(quant_params_compute(tensor_factory, src_ref, data_type_t::bf16,
                      data_type_t::u8, {1, 1}, data_type_t::f32, src_scale,
                      src_zp, &input_tensor),
            status_t::success)
            << "source quantization failed";

    auto bias_tensor
            = tensor_factory.uniform_dist_tensor({1, n}, data_type_t::f32, 2.0);
    auto output_tensor = tensor_factory.uniform_dist_tensor(
            {m, n}, data_type_t::bf16, 2.0);
    auto output_tensor_ref = tensor_factory.uniform_dist_tensor(
            {m, n}, data_type_t::bf16, 2.0);

    // Reference first, on pristine weights (see the note in the test above).
    status_t ref_status = matmul_kernel_test(input_tensor, weight_tensor,
            bias_tensor, output_tensor_ref, po_types, binary_tensors,
            use_LOWOHA, algo, 1.0, 0.0, true);
    ASSERT_EQ(ref_status, status_t::success) << "reference kernel failed";
    clear_matmul_test_caches();

    WeightCacheGuard wc_guard(2);
    ZpCompCacheGuard zp_guard(false);

    constexpr int kIterations = 3;
    for (int iter = 0; iter < kIterations; ++iter) {
        status_t status = matmul_kernel_test(input_tensor, weight_tensor,
                bias_tensor, output_tensor, po_types, binary_tensors,
                use_LOWOHA, algo, 1.0, 0.0);
        if (status == status_t::isa_unsupported) {
            GTEST_SKIP() << "AOCL-DLP blocked INT8 not supported on this ISA";
        }
        ASSERT_EQ(status, status_t::success)
                << "matmul failed on iteration " << iter;

        bool ok = true;
        compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m, n, k,
                rtol_bf16, epsilon_bf16, ok, false, 1.0f,
                /*is_quant=*/true);
        EXPECT_TRUE(ok)
                << "output diverged from the reference on iteration " << iter
                << " with ZENDNNL_ZP_COMP_CACHE=0 — in-place must be refused "
                   "when the compensation is re-derived from the weights on "
                   "every call";
    }
}

// Shared BMM tiles sum raw columns for a nonzero s8 source zero-point.
// The weight must stay raw and the output must match the reference.
TEST_F(TestWeightCacheInplace, BmmSharedWeightKeepsAsymmetricS8OutOfPlace) {
    constexpr int kBatch = 4;
    constexpr int kThreads = 4;
    const uint64_t m = 32, k = 128, n = 128;
    const int mi = static_cast<int>(m);
    const int ki = static_cast<int>(k);
    const int ni = static_cast<int>(n);
    const auto algo = matmul_algo_t::aocl_dlp_blocked;

    auto wei_ref = tensor_factory.uniform_dist_tensor(
            {k, n}, data_type_t::bf16, 25.0, /*transB=*/false);
    tensor_t weight_tensor, wei_scale, wei_zp;
    ASSERT_EQ(quant_params_compute(tensor_factory, wei_ref, data_type_t::bf16,
                      data_type_t::s8, {1, 1}, data_type_t::f32, wei_scale,
                      wei_zp, &weight_tensor),
            status_t::success)
            << "weight quantization failed";

    // s8 quant emits no zero-point; attach one below.
    auto src_ref = tensor_factory.uniform_dist_tensor(
            {static_cast<uint64_t>(kBatch) * m, k}, data_type_t::bf16, 25.0,
            /*transA=*/false);
    tensor_t input_tensor, src_scale, src_zp_tensor;
    ASSERT_EQ(quant_params_compute(tensor_factory, src_ref, data_type_t::bf16,
                      data_type_t::s8, {1, 1}, data_type_t::f32, src_scale,
                      src_zp_tensor, &input_tensor),
            status_t::success)
            << "source quantization failed";

    auto output_tensor = tensor_factory.uniform_dist_tensor(
            {static_cast<uint64_t>(kBatch) * m, n}, data_type_t::bf16, 2.0);
    auto output_tensor_ref = tensor_factory.uniform_dist_tensor(
            {static_cast<uint64_t>(kBatch) * m, n}, data_type_t::bf16, 2.0);

    const size_t plain = static_cast<size_t>(ki) * static_cast<size_t>(ni);
    const size_t req = aocl_get_reorder_buf_size_s8s8s32os32(
            'r', 'n', 'B', ki, ni, nullptr);
    ASSERT_GT(req, 0u) << "AOCL INT8 reorder size query failed";
    const size_t reorder_size = (req + 63u) & ~static_cast<size_t>(63u);
    const size_t capacity = reorder_size + 4096u;
    ASSERT_TRUE(wei_inplace_fits(req, reorder_size, plain, capacity))
            << "test capacity must be large enough that WC=2 would rewrite "
               "the buffer if a tile forwarded it";

    std::vector<int8_t> padded(capacity, 0);
    std::memcpy(padded.data(), weight_tensor.get_raw_handle_unsafe(), plain);
    const std::vector<int8_t> pristine(padded);

    auto fill_q = [](matmul_quantization_params_t::matmul_quant_t &q,
                          const tensor_t &t, bool scale) {
        if (!t.is_quantized()) return;
        if (scale) {
            q.buff = t.get_quant_scale_raw_handle_const();
            q.dt = t.get_quant_scale_data_type();
            auto dims = t.get_quant_scale_size();
            q.dims.assign(dims.begin(), dims.end());
            return;
        }
        if (t.get_quant_subtype() != quant_subtype_t::asymmetric) return;
        q.buff = t.get_quant_zero_raw_handle_const();
        if (!q.buff) return;
        q.dt = t.get_quant_zero_data_type();
        auto dims = t.get_quant_zero_size();
        q.dims.assign(dims.begin(), dims.end());
    };

    // Explicit source zero-point. quant_params stores the pointer.
    const int32_t src_zp_value = 13;

    matmul_params params;
    params.lowoha_algo = algo;
    params.dtypes.src = data_type_t::s8;
    params.dtypes.wei = data_type_t::s8;
    params.dtypes.dst = data_type_t::bf16;
    params.weight_cache_type = 2;
    params.wei_buffer_capacity_bytes = capacity;
    params.num_threads = kThreads;
    fill_q(params.quant_params.src_scale, input_tensor, true);
    fill_q(params.quant_params.wei_scale, weight_tensor, true);
    params.quant_params.src_zp.buff = &src_zp_value;
    params.quant_params.src_zp.dt = data_type_t::s32;
    params.quant_params.src_zp.dims = {1};

    matmul_batch_params_t batch;
    batch.Batch_A = kBatch;
    batch.Batch_B = 1;
    batch.batch_stride_src = static_cast<size_t>(-1);
    batch.batch_stride_wei = static_cast<size_t>(-1);
    batch.batch_stride_dst = static_cast<size_t>(-1);

    matmul_params ref_params = params;
    ref_params.lowoha_algo = matmul_algo_t::reference;
    ref_params.num_threads = 1;
    ref_params.weight_cache_type = 0;
    ref_params.wei_buffer_capacity_bytes = 0;

    status_t ref_status = matmul_direct('r', /*transA=*/false,
            /*transB=*/false, mi, ni, ki, 1.0f,
            input_tensor.get_raw_handle_unsafe(), ki, padded.data(), ni,
            /*bias=*/nullptr, 0.0f, output_tensor_ref.get_raw_handle_unsafe(),
            ni, /*is_weights_const=*/true, batch, ref_params);
    ASSERT_EQ(ref_status, status_t::success) << "reference kernel failed";
    EXPECT_EQ(std::memcmp(padded.data(), pristine.data(), padded.size()), 0)
            << "reference kernel mutated the weight buffer";
    clear_matmul_test_caches();

    WeightCacheGuard wc_guard(2);
    ZpCompCacheGuard zp_guard(true);

    constexpr int kIterations = 2;
    for (int iter = 0; iter < kIterations; ++iter) {
        status_t status = matmul_direct('r', /*transA=*/false,
                /*transB=*/false, mi, ni, ki, 1.0f,
                input_tensor.get_raw_handle_unsafe(), ki, padded.data(), ni,
                /*bias=*/nullptr, 0.0f, output_tensor.get_raw_handle_unsafe(),
                ni, /*is_weights_const=*/true, batch, params);
        if (status == status_t::isa_unsupported) {
            GTEST_SKIP() << "AOCL-DLP blocked INT8 not supported on this ISA";
        }
        ASSERT_EQ(status, status_t::success)
                << "BMM failed on iteration " << iter;

        EXPECT_EQ(std::memcmp(padded.data(), pristine.data(), padded.size()), 0)
                << "shared BMM weight was rewritten in place on iteration "
                << iter
                << " — a shared tile took WC=2 while another tile can still "
                   "be summing the raw columns";

        bool ok = true;
        compare_tensor_2D_matrix(output_tensor, output_tensor_ref,
                static_cast<uint64_t>(kBatch) * m, n, k, rtol_bf16,
                epsilon_bf16, ok, false, 1.0f, /*is_quant=*/true);
        EXPECT_TRUE(ok) << "BMM output diverged from the reference on "
                           "iteration "
                        << iter << " with a nonzero s8 source zero-point";
    }
}

// u8 128x128 blocked size equals k*n, so cap 0 would still allow WC=2.
// Shared tiles must force out-of-place and keep the weight raw.
TEST_F(TestWeightCacheInplace, BmmSharedWeightKeepsAsymmetricU8OutOfPlace) {
    constexpr int kBatch = 4;
    constexpr int kThreads = 4;
    const uint64_t m = 32, k = 128, n = 128;
    const int mi = static_cast<int>(m);
    const int ki = static_cast<int>(k);
    const int ni = static_cast<int>(n);
    const auto algo = matmul_algo_t::aocl_dlp_blocked;

    auto wei_ref = tensor_factory.uniform_dist_tensor(
            {k, n}, data_type_t::bf16, 25.0, /*transB=*/false);
    tensor_t weight_tensor, wei_scale, wei_zp;
    ASSERT_EQ(quant_params_compute(tensor_factory, wei_ref, data_type_t::bf16,
                      data_type_t::s8, {1, 1}, data_type_t::f32, wei_scale,
                      wei_zp, &weight_tensor),
            status_t::success)
            << "weight quantization failed";

    auto src_ref = tensor_factory.uniform_dist_tensor(
            {static_cast<uint64_t>(kBatch) * m, k}, data_type_t::bf16, 25.0,
            /*transA=*/false);
    tensor_t input_tensor, src_scale, src_zp;
    ASSERT_EQ(quant_params_compute(tensor_factory, src_ref, data_type_t::bf16,
                      data_type_t::u8, {1, 1}, data_type_t::f32, src_scale,
                      src_zp, &input_tensor),
            status_t::success)
            << "source quantization failed";

    auto output_tensor = tensor_factory.uniform_dist_tensor(
            {static_cast<uint64_t>(kBatch) * m, n}, data_type_t::bf16, 2.0);
    auto output_tensor_ref = tensor_factory.uniform_dist_tensor(
            {static_cast<uint64_t>(kBatch) * m, n}, data_type_t::bf16, 2.0);

    const size_t plain = static_cast<size_t>(ki) * static_cast<size_t>(ni);
    const size_t req = aocl_get_reorder_buf_size_u8s8s32os32(
            'r', 'n', 'B', ki, ni, nullptr);
    ASSERT_GT(req, 0u) << "AOCL INT8 reorder size query failed";
    const size_t reorder_size = (req + 63u) & ~static_cast<size_t>(63u);
    const size_t capacity = reorder_size + 4096u;
    ASSERT_TRUE(wei_inplace_fits(req, reorder_size, plain, /*declared_cap=*/0))
            << "u8s8 " << k << "x" << n
            << " does not fit k*n; this case would already be refused by "
               "clearing wei_buffer_capacity_bytes";
    ASSERT_TRUE(wei_inplace_fits(req, reorder_size, plain, capacity));

    std::vector<int8_t> padded(capacity, 0);
    std::memcpy(padded.data(), weight_tensor.get_raw_handle_unsafe(), plain);
    const std::vector<int8_t> pristine(padded);

    auto fill_q = [](matmul_quantization_params_t::matmul_quant_t &q,
                          const tensor_t &t, bool scale) {
        if (!t.is_quantized()) return;
        if (scale) {
            q.buff = t.get_quant_scale_raw_handle_const();
            q.dt = t.get_quant_scale_data_type();
            auto dims = t.get_quant_scale_size();
            q.dims.assign(dims.begin(), dims.end());
            return;
        }
        if (t.get_quant_subtype() != quant_subtype_t::asymmetric) return;
        q.buff = t.get_quant_zero_raw_handle_const();
        if (!q.buff) return;
        q.dt = t.get_quant_zero_data_type();
        auto dims = t.get_quant_zero_size();
        q.dims.assign(dims.begin(), dims.end());
    };

    matmul_params params;
    params.lowoha_algo = algo;
    params.dtypes.src = data_type_t::u8;
    params.dtypes.wei = data_type_t::s8;
    params.dtypes.dst = data_type_t::bf16;
    params.weight_cache_type = 2;
    params.wei_buffer_capacity_bytes = capacity;
    params.num_threads = kThreads;
    fill_q(params.quant_params.src_scale, input_tensor, true);
    fill_q(params.quant_params.src_zp, input_tensor, false);
    fill_q(params.quant_params.wei_scale, weight_tensor, true);
    ASSERT_NE(params.quant_params.src_zp.buff, nullptr)
            << "u8 source must carry a zero-point so tiles sum raw columns";

    matmul_batch_params_t batch;
    batch.Batch_A = kBatch;
    batch.Batch_B = 1;
    batch.batch_stride_src = static_cast<size_t>(-1);
    batch.batch_stride_wei = static_cast<size_t>(-1);
    batch.batch_stride_dst = static_cast<size_t>(-1);

    matmul_params ref_params = params;
    ref_params.lowoha_algo = matmul_algo_t::reference;
    ref_params.num_threads = 1;
    ref_params.weight_cache_type = 0;
    ref_params.wei_buffer_capacity_bytes = 0;

    status_t ref_status = matmul_direct('r', /*transA=*/false,
            /*transB=*/false, mi, ni, ki, 1.0f,
            input_tensor.get_raw_handle_unsafe(), ki, padded.data(), ni,
            /*bias=*/nullptr, 0.0f, output_tensor_ref.get_raw_handle_unsafe(),
            ni, /*is_weights_const=*/true, batch, ref_params);
    ASSERT_EQ(ref_status, status_t::success) << "reference kernel failed";
    EXPECT_EQ(std::memcmp(padded.data(), pristine.data(), padded.size()), 0)
            << "reference kernel mutated the weight buffer";
    clear_matmul_test_caches();

    WeightCacheGuard wc_guard(2);
    ZpCompCacheGuard zp_guard(true);

    constexpr int kIterations = 2;
    for (int iter = 0; iter < kIterations; ++iter) {
        status_t status = matmul_direct('r', /*transA=*/false,
                /*transB=*/false, mi, ni, ki, 1.0f,
                input_tensor.get_raw_handle_unsafe(), ki, padded.data(), ni,
                /*bias=*/nullptr, 0.0f, output_tensor.get_raw_handle_unsafe(),
                ni, /*is_weights_const=*/true, batch, params);
        if (status == status_t::isa_unsupported) {
            GTEST_SKIP() << "AOCL-DLP blocked INT8 not supported on this ISA";
        }
        ASSERT_EQ(status, status_t::success)
                << "BMM failed on iteration " << iter;

        EXPECT_EQ(std::memcmp(padded.data(), pristine.data(), padded.size()), 0)
                << "shared u8 BMM weight was rewritten in place on iteration "
                << iter
                << " — blocked size equals k*n, so WC=2 must be forced off";

        bool ok = true;
        compare_tensor_2D_matrix(output_tensor, output_tensor_ref,
                static_cast<uint64_t>(kBatch) * m, n, k, rtol_bf16,
                epsilon_bf16, ok, false, 1.0f, /*is_quant=*/true);
        EXPECT_TRUE(ok) << "u8 BMM output diverged from the reference on "
                           "iteration "
                        << iter;
    }
}

// Shared tiles lower the weight cache to 1, never above the caller's own
// value: with weight_cache_type 0, a weight rewritten at the same address is
// reordered again on the next call.  With ZP-compensation caching off the
// weight cache is the only place the old weight could come from; with it on
// (the default) the compensation must be recomputed from the new weight too.
TEST_F(TestWeightCacheInplace, BmmSharedWeightHonoursPerCallCacheOff) {
    constexpr int kBatch = 4;
    constexpr int kThreads = 4;
    const uint64_t m = 32, k = 128, n = 128;
    const int mi = static_cast<int>(m);
    const int ki = static_cast<int>(k);
    const int ni = static_cast<int>(n);

    auto wei_ref = tensor_factory.uniform_dist_tensor(
            {k, n}, data_type_t::bf16, 25.0, /*transB=*/false);
    tensor_t weight_tensor, wei_scale, wei_zp;
    ASSERT_EQ(quant_params_compute(tensor_factory, wei_ref, data_type_t::bf16,
                      data_type_t::s8, {1, 1}, data_type_t::f32, wei_scale,
                      wei_zp, &weight_tensor),
            status_t::success)
            << "weight quantization failed";

    auto src_ref = tensor_factory.uniform_dist_tensor(
            {static_cast<uint64_t>(kBatch) * m, k}, data_type_t::bf16, 25.0,
            /*transA=*/false);
    tensor_t input_tensor, src_scale, src_zp_tensor;
    ASSERT_EQ(quant_params_compute(tensor_factory, src_ref, data_type_t::bf16,
                      data_type_t::s8, {1, 1}, data_type_t::f32, src_scale,
                      src_zp_tensor, &input_tensor),
            status_t::success)
            << "source quantization failed";

    auto output_tensor = tensor_factory.uniform_dist_tensor(
            {static_cast<uint64_t>(kBatch) * m, n}, data_type_t::bf16, 2.0);
    auto output_tensor_ref = tensor_factory.uniform_dist_tensor(
            {static_cast<uint64_t>(kBatch) * m, n}, data_type_t::bf16, 2.0);

    const size_t plain = static_cast<size_t>(ki) * static_cast<size_t>(ni);
    std::vector<int8_t> wei(plain);
    std::memcpy(wei.data(), weight_tensor.get_raw_handle_unsafe(), plain);

    auto fill_scale = [](matmul_quantization_params_t::matmul_quant_t &q,
                              const tensor_t &t) {
        if (!t.is_quantized()) return;
        q.buff = t.get_quant_scale_raw_handle_const();
        q.dt = t.get_quant_scale_data_type();
        auto dims = t.get_quant_scale_size();
        q.dims.assign(dims.begin(), dims.end());
    };

    const int32_t src_zp_value = 13;

    matmul_params params;
    params.lowoha_algo = matmul_algo_t::aocl_dlp_blocked;
    params.dtypes.src = data_type_t::s8;
    params.dtypes.wei = data_type_t::s8;
    params.dtypes.dst = data_type_t::bf16;
    params.weight_cache_type = 0;
    params.num_threads = kThreads;
    fill_scale(params.quant_params.src_scale, input_tensor);
    fill_scale(params.quant_params.wei_scale, weight_tensor);
    params.quant_params.src_zp.buff = &src_zp_value;
    params.quant_params.src_zp.dt = data_type_t::s32;
    params.quant_params.src_zp.dims = {1};

    matmul_batch_params_t batch;
    batch.Batch_A = kBatch;
    batch.Batch_B = 1;
    batch.batch_stride_src = static_cast<size_t>(-1);
    batch.batch_stride_wei = static_cast<size_t>(-1);
    batch.batch_stride_dst = static_cast<size_t>(-1);

    matmul_params ref_params = params;
    ref_params.lowoha_algo = matmul_algo_t::reference;
    ref_params.num_threads = 1;

    WeightCacheGuard wc_guard(2);
    const std::vector<int8_t> original = wei;

    for (const bool zp_comp_cache : {false, true}) {
        SCOPED_TRACE(zp_comp_cache ? "ZP-compensation cache on"
                                   : "ZP-compensation cache off");
        ZpCompCacheGuard zp_guard(zp_comp_cache);
        clear_matmul_test_caches();
        wei = original;
        for (int pass = 0; pass < 2; ++pass) {
            if (pass == 1) {
                for (size_t i = 0; i < plain; ++i) {
                    wei[i] = static_cast<int8_t>(3 - wei[i] / 2);
                }
            }
            ASSERT_EQ(
                    matmul_direct('r', /*transA=*/false, /*transB=*/false, mi,
                            ni, ki, 1.0f, input_tensor.get_raw_handle_unsafe(),
                            ki, wei.data(), ni, /*bias=*/nullptr, 0.0f,
                            output_tensor_ref.get_raw_handle_unsafe(), ni,
                            /*is_weights_const=*/true, batch, ref_params),
                    status_t::success)
                    << "reference kernel failed on pass " << pass;
            status_t status = matmul_direct('r', /*transA=*/false,
                    /*transB=*/false, mi, ni, ki, 1.0f,
                    input_tensor.get_raw_handle_unsafe(), ki, wei.data(), ni,
                    /*bias=*/nullptr, 0.0f,
                    output_tensor.get_raw_handle_unsafe(), ni,
                    /*is_weights_const=*/true, batch, params);
            if (status == status_t::isa_unsupported) {
                GTEST_SKIP()
                        << "AOCL-DLP blocked INT8 not supported on this ISA";
            }
            ASSERT_EQ(status, status_t::success)
                    << "BMM failed on pass " << pass;

            bool ok = true;
            compare_tensor_2D_matrix(output_tensor, output_tensor_ref,
                    static_cast<uint64_t>(kBatch) * m, n, k, rtol_bf16,
                    epsilon_bf16, ok, false, 1.0f, /*is_quant=*/true);
            EXPECT_TRUE(ok)
                    << "BMM output diverged from the reference on pass " << pass
                    << (pass == 1 ? " (a cached reorder or compensation "
                                    "served the weight from before the "
                                    "rewrite)"
                                  : "");
        }
    }
}

// The simulated W4A8 path (aocl_dlp) caches the blocked reorder of an s4
// weight while caching is on.  After the caller rewrites the weight at the
// same address, a call whose effective cache mode is 0 must not reuse it: it
// matches a run with every cache dropped.
TEST_F(TestWeightCacheInplace, W4A8CacheOffRereadsRewrittenWeight) {
    const uint64_t m = 32, k = 128, n = 64, group_size = 32;
    auto wei_scale = tensor_factory.uniform_dist_tensor(
            {k / group_size, n}, data_type_t::bf16, 2.0);
    auto weight = tensor_factory.uniform_dist_tensor(
            {k, n}, data_type_t::s4, 7.0, false, wei_scale);
    auto src_scale = tensor_factory.zero_tensor({m, 1u}, data_type_t::bf16);
    auto input = tensor_factory.uniform_dist_tensor(
            {m, k}, data_type_t::bf16, 2.0, false, src_scale, tensor_t());
    auto bias = tensor_factory.zero_tensor({1u, n}, data_type_t::bf16);
    auto first = tensor_factory.zero_tensor({m, n}, data_type_t::bf16);
    auto second = tensor_factory.zero_tensor({m, n}, data_type_t::bf16);
    auto ref = tensor_factory.zero_tensor({m, n}, data_type_t::bf16);

    CustomKernelRouteGuard route_off(false);
    const std::vector<post_op_type_t> po_types;
    const std::vector<tensor_t> binary_tensors;
    const auto run = [&](tensor_t &out, int32_t mode) {
        WeightCacheGuard wc_guard(mode);
        ASSERT_EQ(matmul_kernel_test(input, weight, bias, out, po_types,
                          binary_tensors, /*use_LOWOHA=*/true,
                          matmul_algo_t::aocl_dlp, 1.0f, 0.0f),
                status_t::success);
    };

    clear_matmul_test_caches();
    ASSERT_NO_FATAL_FAILURE(run(first, 1));
    // Flip the low bit of every s4 value: still in range, other weights.
    auto *w = static_cast<uint8_t *>(weight.get_raw_handle_unsafe());
    for (size_t i = 0; i < weight.get_buffer_sz_bytes(); ++i) {
        w[i] ^= 0x11;
    }
    ASSERT_NO_FATAL_FAILURE(run(second, 0));
    clear_matmul_test_caches();
    ASSERT_NO_FATAL_FAILURE(run(ref, 0));

    const size_t bytes = m * n * sizeof(bfloat16_t);
    EXPECT_NE(std::memcmp(first.get_raw_handle_unsafe(),
                      ref.get_raw_handle_unsafe(), bytes),
            0)
            << "the rewrite must change the output";
    EXPECT_EQ(std::memcmp(second.get_raw_handle_unsafe(),
                      ref.get_raw_handle_unsafe(), bytes),
            0)
            << "a W4A8 reorder cached before the rewrite was reused";
}
