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

// oneDNN blocked WC=2 in-place / OOP cache.

#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include "gtest_utils.hpp"
#include "lowoha_operators/matmul/lowoha_matmul.hpp"

#if ZENDNNL_DEPENDS_ONEDNN

namespace {

using zendnnl::lowoha::matmul::matmul_batch_params_t;
using zendnnl::lowoha::matmul::matmul_direct;
using zendnnl::lowoha::matmul::matmul_params;

class TestOnednnWeightCacheInplace : public ::testing::Test {
protected:
    void TearDown() override { clear_matmul_test_caches(); }
    tensor_factory_t tensor_factory {};
};

tensor_t make_2d(tensor_factory_t &factory, int rows, int cols,
        data_type_t dtype, double range) {
    return factory.uniform_dist_tensor(
            {static_cast<uint64_t>(rows), static_cast<uint64_t>(cols)}, dtype,
            range);
}

tensor_t make_operand(
        tensor_factory_t &factory, int rows, int cols, data_type_t dtype) {
    return make_2d(factory, rows, cols, dtype, 25.0);
}

tensor_t make_dst(
        tensor_factory_t &factory, int rows, int cols, data_type_t dtype) {
    return make_2d(factory, rows, cols, dtype, 2.0);
}

tensor_t make_bias(tensor_factory_t &factory, int n) {
    return make_2d(factory, 1, n, data_type_t::f32, 2.0);
}

matmul_params make_params(data_type_t elem_dtype, int32_t weight_cache_type) {
    matmul_params params;
    params.dtypes.src = elem_dtype;
    params.dtypes.wei = elem_dtype;
    params.dtypes.dst = elem_dtype;
    params.dtypes.bias = (elem_dtype == data_type_t::bf16
                                 || elem_dtype == data_type_t::f16)
            ? data_type_t::f32
            : elem_dtype;
    params.lowoha_algo = matmul_algo_t::onednn_blocked;
    params.weight_cache_type = weight_cache_type;
    return params;
}

void attach_quant_scale(
        const tensor_t &t, matmul_quantization_params_t::matmul_quant_t &dst) {
    dst.buff = const_cast<void *>(t.get_quant_scale_raw_handle_const());
    dst.dt = t.get_quant_scale_data_type();
    const auto dims = t.get_quant_scale_size();
    dst.dims.assign(dims.begin(), dims.end());
}

void attach_quant_zero(
        const tensor_t &t, matmul_quantization_params_t::matmul_quant_t &dst) {
    dst.buff = const_cast<void *>(t.get_quant_zero_raw_handle_const());
    dst.dt = t.get_quant_zero_data_type();
    const auto dims = t.get_quant_zero_size();
    dst.dims.assign(dims.begin(), dims.end());
}

matmul_params make_int8_params(
        int32_t weight_cache_type, const tensor_t &input, const tensor_t &wei) {
    matmul_params params;
    params.dtypes.src = data_type_t::u8;
    params.dtypes.wei = data_type_t::s8;
    params.dtypes.dst = data_type_t::bf16;
    params.dtypes.bias = data_type_t::f32;
    params.lowoha_algo = matmul_algo_t::onednn_blocked;
    params.weight_cache_type = weight_cache_type;
    attach_quant_scale(input, params.quant_params.src_scale);
    if (input.get_quant_subtype() == quant_subtype_t::asymmetric) {
        attach_quant_zero(input, params.quant_params.src_zp);
    }
    attach_quant_scale(wei, params.quant_params.wei_scale);
    if (wei.get_quant_subtype() == quant_subtype_t::asymmetric) {
        attach_quant_zero(wei, params.quant_params.wei_zp);
    }
    return params;
}

status_t run_blocked(matmul_params &params, tensor_t &input, tensor_t &weight,
        const void *bias, tensor_t &output, int m) {
    const int k = static_cast<int>(input.get_size()[1]);
    const int n = static_cast<int>(weight.get_size()[1]);
    matmul_batch_params_t batch_params;
    return matmul_direct('r', false, false, m, n, k, 1.0f,
            input.get_raw_handle_unsafe(), k, weight.get_raw_handle_unsafe(), n,
            bias, 0.0f, output.get_raw_handle_unsafe(), n,
            /*is_weights_const=*/true, batch_params, params);
}

status_t run_reference(
        tensor_t &input, tensor_t &weight, tensor_t &bias, tensor_t &output) {
    WeightCacheGuard wc_guard(1);
    const std::vector<post_op_type_t> po_types {};
    const std::vector<tensor_t> binary_tensors {};
    return matmul_kernel_test(input, weight, bias, output, po_types,
            binary_tensors, /*use_LOWOHA=*/true, matmul_algo_t::onednn_blocked,
            1.0f, 0.0f);
}

#define ASSERT_RUN_BLOCKED(...) \
    do { \
        const status_t blocked_status_ = run_blocked(__VA_ARGS__); \
        if (blocked_status_ == status_t::unimplemented) { \
            GTEST_SKIP() << "onednn_blocked unavailable in this build"; \
        } \
        ASSERT_EQ(blocked_status_, status_t::success); \
    } while (0)

#define ASSERT_RUN_REFERENCE(...) \
    do { \
        const status_t reference_status_ = run_reference(__VA_ARGS__); \
        if (reference_status_ == status_t::unimplemented \
                || reference_status_ == status_t::isa_unsupported) { \
            GTEST_SKIP() << "onednn_blocked reference unavailable on this " \
                            "host"; \
        } \
        ASSERT_EQ(reference_status_, status_t::success) \
                << "reference kernel failed"; \
    } while (0)

void compare_onednn_blocked_output(tensor_t &output, tensor_t &reference, int m,
        int n, int k, bool is_quant, data_type_t elem_dtype,
        const char *context) {
    const bool half
            = elem_dtype == data_type_t::bf16 || elem_dtype == data_type_t::f16;
    bool ok = true;
    compare_tensor_2D_matrix(output, reference, m, n, k,
            half ? rtol_bf16 : rtol_f32, half ? epsilon_bf16 : epsilon_f32, ok,
            false, 1.0f, is_quant);
    EXPECT_TRUE(ok) << context;
}

std::vector<uint8_t> snapshot_tensor_bytes(const tensor_t &t) {
    const uint8_t *p = static_cast<const uint8_t *>(t.get_raw_handle_const());
    return std::vector<uint8_t>(p, p + t.get_buffer_sz_bytes());
}

bool tensor_bytes_match(const tensor_t &t, const std::vector<uint8_t> &bytes) {
    return bytes.size() == t.get_buffer_sz_bytes()
            && std::memcmp(t.get_raw_handle_const(), bytes.data(), bytes.size())
            == 0;
}

void restore_tensor_bytes(const std::vector<uint8_t> &bytes, tensor_t &t) {
    ASSERT_EQ(bytes.size(), t.get_buffer_sz_bytes());
    std::memcpy(t.get_raw_handle_unsafe(), bytes.data(), bytes.size());
}

#define ASSERT_RESTORE_TENSOR_BYTES(bytes, tensor) \
    ASSERT_NO_FATAL_FAILURE(restore_tensor_bytes((bytes), (tensor)))

tensor_t make_independent_weight_copy(tensor_factory_t &factory,
        const std::vector<uint8_t> &bytes, int rows, int cols,
        data_type_t dtype) {
    auto weight = make_operand(factory, rows, cols, dtype);
    restore_tensor_bytes(bytes, weight);
    return weight;
}

#define ASSERT_INDEPENDENT_WEIGHT_COPY(dst, bytes, rows, cols, dtype) \
    ASSERT_NO_FATAL_FAILURE( \
            (dst) = make_independent_weight_copy( \
                    tensor_factory, (bytes), (rows), (cols), (dtype)))

struct quantized_operand_t {
    tensor_t value;
    tensor_t scale;
    tensor_t zp;
};

status_t quantize_operand(tensor_factory_t &factory, int rows, int cols,
        data_type_t dtype, quantized_operand_t &out) {
    auto ref = make_operand(factory, rows, cols, data_type_t::bf16);
    return quant_params_compute(factory, ref, data_type_t::bf16, dtype, {1, 1},
            data_type_t::f32, out.scale, out.zp, &out.value);
}

} // namespace

TEST_F(TestOnednnWeightCacheInplace, Bf16Blocked_InPlaceWeightReuse) {
    const int m = 32, k = 256, n = 256;
    const data_type_t dtype = data_type_t::bf16;

    auto input = make_operand(tensor_factory, m, k, dtype);
    auto weight = make_operand(tensor_factory, k, n, dtype);
    auto bias = make_bias(tensor_factory, n);
    auto output = make_dst(tensor_factory, m, n, dtype);
    auto reference = make_dst(tensor_factory, m, n, dtype);
    const auto pristine_weight = snapshot_tensor_bytes(weight);

    ASSERT_RUN_REFERENCE(input, weight, bias, reference);
    clear_matmul_test_caches();
    ASSERT_RESTORE_TENSOR_BYTES(pristine_weight, weight);

    WeightCacheGuard wc_guard(2);
    matmul_params params = make_params(dtype, 2);
    for (int iter = 0; iter < 3; ++iter) {
        ASSERT_RUN_BLOCKED(
                params, input, weight, bias.get_raw_handle_unsafe(), output, m);
        EXPECT_FALSE(tensor_bytes_match(weight, pristine_weight))
                << "WC=2 must mutate BF16 weights before cache reuse";
        compare_onednn_blocked_output(output, reference, m, n, k,
                /*is_quant=*/false, dtype,
                "WEIGHT_CACHE=2 oneDNN BF16 output diverged from the "
                "reference");
    }
}

TEST_F(TestOnednnWeightCacheInplace, F16Blocked_InPlaceWeightReuse) {
    const int m = 32, k = 256, n = 256;
    const data_type_t dtype = data_type_t::f16;

    auto input = make_operand(tensor_factory, m, k, dtype);
    auto weight = make_operand(tensor_factory, k, n, dtype);
    auto bias = make_bias(tensor_factory, n);
    auto output = make_dst(tensor_factory, m, n, dtype);
    auto reference = make_dst(tensor_factory, m, n, dtype);
    const auto pristine_weight = snapshot_tensor_bytes(weight);

    ASSERT_RUN_REFERENCE(input, weight, bias, reference);
    clear_matmul_test_caches();
    ASSERT_RESTORE_TENSOR_BYTES(pristine_weight, weight);

    WeightCacheGuard wc_guard(2);
    matmul_params params = make_params(dtype, 2);
    for (int iter = 0; iter < 3; ++iter) {
        ASSERT_RUN_BLOCKED(
                params, input, weight, bias.get_raw_handle_unsafe(), output, m);
        EXPECT_FALSE(tensor_bytes_match(weight, pristine_weight))
                << "WC=2 must mutate F16 weights before cache reuse";
        compare_onednn_blocked_output(output, reference, m, n, k,
                /*is_quant=*/false, dtype,
                "WEIGHT_CACHE=2 oneDNN F16 output diverged from the reference");
    }
}

TEST_F(TestOnednnWeightCacheInplace, Bf16Blocked_RectangularLdbUsesN) {
    const int m = 32, k = 128, n = 256;
    const data_type_t dtype = data_type_t::bf16;

    auto input = make_operand(tensor_factory, m, k, dtype);
    auto weight = make_operand(tensor_factory, k, n, dtype);
    auto bias = make_bias(tensor_factory, n);
    auto output = make_dst(tensor_factory, m, n, dtype);
    auto reference = make_dst(tensor_factory, m, n, dtype);
    const auto pristine_weight = snapshot_tensor_bytes(weight);

    ASSERT_RUN_REFERENCE(input, weight, bias, reference);
    clear_matmul_test_caches();
    ASSERT_RESTORE_TENSOR_BYTES(pristine_weight, weight);

    WeightCacheGuard wc_guard(2);
    matmul_params params = make_params(dtype, 2);
    ASSERT_RUN_BLOCKED(
            params, input, weight, bias.get_raw_handle_unsafe(), output, m);
    compare_onednn_blocked_output(output, reference, m, n, k,
            /*is_quant=*/false, dtype,
            "onednn_blocked BF16 K!=N must use ldb=N");
    EXPECT_FALSE(tensor_bytes_match(weight, pristine_weight))
            << "WC=2 must mutate rectangular BF16 weights in place";
}

TEST_F(TestOnednnWeightCacheInplace, Bf16Blocked_InPlaceSurvivesMChange) {
    const int k = 256, n = 256, m_prompt = 32, m_decode = 1;
    const data_type_t dtype = data_type_t::bf16;

    auto input_prompt = make_operand(tensor_factory, m_prompt, k, dtype);
    auto input_decode = make_operand(tensor_factory, m_decode, k, dtype);
    auto weight = make_operand(tensor_factory, k, n, dtype);
    auto bias = make_bias(tensor_factory, n);
    auto out_prompt = make_dst(tensor_factory, m_prompt, n, dtype);
    auto out_decode = make_dst(tensor_factory, m_decode, n, dtype);
    auto ref_prompt = make_dst(tensor_factory, m_prompt, n, dtype);
    auto ref_decode = make_dst(tensor_factory, m_decode, n, dtype);
    const auto pristine_weight = snapshot_tensor_bytes(weight);

    tensor_t weight_for_ref;
    ASSERT_INDEPENDENT_WEIGHT_COPY(
            weight_for_ref, pristine_weight, k, n, dtype);
    ASSERT_RUN_REFERENCE(input_prompt, weight_for_ref, bias, ref_prompt);
    ASSERT_RESTORE_TENSOR_BYTES(pristine_weight, weight_for_ref);
    clear_matmul_test_caches();
    ASSERT_RUN_REFERENCE(input_decode, weight_for_ref, bias, ref_decode);
    clear_matmul_test_caches();

    WeightCacheGuard wc_guard(2);
    matmul_params params = make_params(dtype, 2);
    ASSERT_RUN_BLOCKED(params, input_prompt, weight,
            bias.get_raw_handle_unsafe(), out_prompt, m_prompt);
    EXPECT_FALSE(tensor_bytes_match(weight, pristine_weight))
            << "WC=2 must take the in-place path before the M=1 reuse";
    ASSERT_RUN_BLOCKED(params, input_decode, weight,
            bias.get_raw_handle_unsafe(), out_decode, m_decode);

    compare_onednn_blocked_output(out_prompt, ref_prompt, m_prompt, n, k,
            /*is_quant=*/false, dtype,
            "BF16 M=32 after in-place pack must match the reference");
    compare_onednn_blocked_output(out_decode, ref_decode, m_decode, n, k,
            /*is_quant=*/false, dtype,
            "BF16 M=1 must reuse the in-place-packed weight buffer");
}

TEST_F(TestOnednnWeightCacheInplace,
        Bf16Blocked_InPlaceSurvivesDecodeThenPrompt) {
    const int k = 256, n = 256, m_prompt = 32, m_decode = 1;
    const data_type_t dtype = data_type_t::bf16;

    auto input_prompt = make_operand(tensor_factory, m_prompt, k, dtype);
    auto input_decode = make_operand(tensor_factory, m_decode, k, dtype);
    auto weight = make_operand(tensor_factory, k, n, dtype);
    auto bias = make_bias(tensor_factory, n);
    auto out_prompt = make_dst(tensor_factory, m_prompt, n, dtype);
    auto out_decode = make_dst(tensor_factory, m_decode, n, dtype);
    auto ref_prompt = make_dst(tensor_factory, m_prompt, n, dtype);
    auto ref_decode = make_dst(tensor_factory, m_decode, n, dtype);
    const auto pristine_weight = snapshot_tensor_bytes(weight);

    tensor_t weight_for_ref;
    ASSERT_INDEPENDENT_WEIGHT_COPY(
            weight_for_ref, pristine_weight, k, n, dtype);
    ASSERT_RUN_REFERENCE(input_prompt, weight_for_ref, bias, ref_prompt);
    ASSERT_RESTORE_TENSOR_BYTES(pristine_weight, weight_for_ref);
    clear_matmul_test_caches();
    ASSERT_RUN_REFERENCE(input_decode, weight_for_ref, bias, ref_decode);
    clear_matmul_test_caches();

    WeightCacheGuard wc_guard(2);
    matmul_params params = make_params(dtype, 2);
    ASSERT_RUN_BLOCKED(params, input_decode, weight,
            bias.get_raw_handle_unsafe(), out_decode, m_decode);
    EXPECT_FALSE(tensor_bytes_match(weight, pristine_weight))
            << "WC=2 must take the in-place path on the first M=1 call";
    ASSERT_RUN_BLOCKED(params, input_prompt, weight,
            bias.get_raw_handle_unsafe(), out_prompt, m_prompt);

    compare_onednn_blocked_output(out_decode, ref_decode, m_decode, n, k,
            /*is_quant=*/false, dtype,
            "BF16 M=1 after in-place pack must match the reference");
    compare_onednn_blocked_output(out_prompt, ref_prompt, m_prompt, n, k,
            /*is_quant=*/false, dtype,
            "BF16 M=32 must reuse the in-place-packed weight buffer from M=1");
}

TEST_F(TestOnednnWeightCacheInplace, Int8Blocked_Wc2StaysOutOfPlace) {
    const int m = 32, k = 256, n = 256;

    quantized_operand_t wei_q, src_q;
    ASSERT_EQ(quantize_operand(tensor_factory, k, n, data_type_t::s8, wei_q),
            status_t::success);
    ASSERT_EQ(quantize_operand(tensor_factory, m, k, data_type_t::u8, src_q),
            status_t::success);
    auto bias = make_bias(tensor_factory, n);
    auto output = make_dst(tensor_factory, m, n, data_type_t::bf16);
    auto reference = make_dst(tensor_factory, m, n, data_type_t::bf16);
    const auto weight_snapshot = snapshot_tensor_bytes(wei_q.value);

    ASSERT_RUN_REFERENCE(src_q.value, wei_q.value, bias, reference);
    clear_matmul_test_caches();

    WeightCacheGuard wc_guard(2);
    ZpCompCacheGuard zp_guard(true);
    matmul_params params = make_int8_params(2, src_q.value, wei_q.value);
    ASSERT_RUN_BLOCKED(params, src_q.value, wei_q.value,
            bias.get_raw_handle_unsafe(), output, m);

    EXPECT_TRUE(tensor_bytes_match(wei_q.value, weight_snapshot))
            << "INT8 oneDNN blocked weight must stay out-of-place under WC=2 "
               "(blocked size > plain buffer)";
    compare_onednn_blocked_output(output, reference, m, n, k,
            /*is_quant=*/true, data_type_t::bf16,
            "INT8 oneDNN WC=2 output must match the reference");
}

TEST_F(TestOnednnWeightCacheInplace, Int8Blocked_Wc2ReuseMatchesReference) {
    const int m = 32, k = 256, n = 256;

    quantized_operand_t wei_q, src_q;
    ASSERT_EQ(quantize_operand(tensor_factory, k, n, data_type_t::s8, wei_q),
            status_t::success);
    ASSERT_EQ(quantize_operand(tensor_factory, m, k, data_type_t::u8, src_q),
            status_t::success);
    auto bias = make_bias(tensor_factory, n);
    auto output = make_dst(tensor_factory, m, n, data_type_t::bf16);
    auto reference = make_dst(tensor_factory, m, n, data_type_t::bf16);
    const auto weight_snapshot = snapshot_tensor_bytes(wei_q.value);

    ASSERT_RUN_REFERENCE(src_q.value, wei_q.value, bias, reference);
    clear_matmul_test_caches();

    WeightCacheGuard wc_guard(2);
    ZpCompCacheGuard zp_guard(true);
    matmul_params params = make_int8_params(2, src_q.value, wei_q.value);
    for (int iter = 0; iter < 3; ++iter) {
        ASSERT_RUN_BLOCKED(params, src_q.value, wei_q.value,
                bias.get_raw_handle_unsafe(), output, m);
        EXPECT_TRUE(tensor_bytes_match(wei_q.value, weight_snapshot))
                << "INT8 oneDNN WC=2 must keep the caller weight buffer "
                   "untouched across cache reuse";
        compare_onednn_blocked_output(output, reference, m, n, k,
                /*is_quant=*/true, data_type_t::bf16,
                "INT8 oneDNN WC=2 output diverged from the reference on reuse");
    }
}

TEST_F(TestOnednnWeightCacheInplace,
        Int8Blocked_Wc1BiasChangeMatchesReference) {
    const int m = 32, k = 256, n = 256;

    quantized_operand_t wei_q, src_q;
    ASSERT_EQ(quantize_operand(tensor_factory, k, n, data_type_t::s8, wei_q),
            status_t::success);
    ASSERT_EQ(quantize_operand(tensor_factory, m, k, data_type_t::u8, src_q),
            status_t::success);
    auto bias = make_bias(tensor_factory, n);
    auto out_bias = make_dst(tensor_factory, m, n, data_type_t::bf16);
    auto out_nobias = make_dst(tensor_factory, m, n, data_type_t::bf16);
    auto ref_bias = make_dst(tensor_factory, m, n, data_type_t::bf16);
    auto ref_nobias = make_dst(tensor_factory, m, n, data_type_t::bf16);
    const auto weight_snapshot = snapshot_tensor_bytes(wei_q.value);

    tensor_t no_bias;
    ASSERT_RUN_REFERENCE(src_q.value, wei_q.value, bias, ref_bias);
    clear_matmul_test_caches();
    ASSERT_RESTORE_TENSOR_BYTES(weight_snapshot, wei_q.value);
    ASSERT_RUN_REFERENCE(src_q.value, wei_q.value, no_bias, ref_nobias);
    clear_matmul_test_caches();
    ASSERT_RESTORE_TENSOR_BYTES(weight_snapshot, wei_q.value);

    WeightCacheGuard wc_guard(1);
    ZpCompCacheGuard zp_guard(true);
    matmul_params params = make_int8_params(1, src_q.value, wei_q.value);
    ASSERT_RUN_BLOCKED(params, src_q.value, wei_q.value,
            bias.get_raw_handle_unsafe(), out_bias, m);
    ASSERT_RUN_BLOCKED(
            params, src_q.value, wei_q.value, nullptr, out_nobias, m);

    compare_onednn_blocked_output(out_bias, ref_bias, m, n, k,
            /*is_quant=*/true, data_type_t::bf16,
            "INT8 WC=1 with bias must not be polluted by a later no-bias call");
    compare_onednn_blocked_output(out_nobias, ref_nobias, m, n, k,
            /*is_quant=*/true, data_type_t::bf16,
            "INT8 WC=1 no-bias after a with-bias pack must match its own "
            "reference");
}

TEST_F(TestOnednnWeightCacheInplace, F32Blocked_Wc2OutOfPlaceReuse) {
    const int m = 32, k = 256, n = 256;
    const data_type_t dtype = data_type_t::f32;

    auto input = make_operand(tensor_factory, m, k, dtype);
    auto weight = make_operand(tensor_factory, k, n, dtype);
    auto bias = make_bias(tensor_factory, n);
    auto output = make_dst(tensor_factory, m, n, dtype);
    auto reference = make_dst(tensor_factory, m, n, dtype);
    const auto pristine_weight = snapshot_tensor_bytes(weight);

    ASSERT_RUN_REFERENCE(input, weight, bias, reference);
    clear_matmul_test_caches();
    ASSERT_RESTORE_TENSOR_BYTES(pristine_weight, weight);

    WeightCacheGuard wc_guard(2);
    matmul_params params = make_params(dtype, 2);
    for (int iter = 0; iter < 3; ++iter) {
        ASSERT_RUN_BLOCKED(
                params, input, weight, bias.get_raw_handle_unsafe(), output, m);
        EXPECT_TRUE(tensor_bytes_match(weight, pristine_weight))
                << "WC=2 must leave F32 weights plain for the M=1 AOCL "
                   "fallback";
        compare_onednn_blocked_output(output, reference, m, n, k,
                /*is_quant=*/false, dtype,
                "WEIGHT_CACHE=2 oneDNN F32 output diverged from the "
                "reference");
    }
}

// TODO(ZENAI-3355): when F32 M=1 stays on oneDNN, assert in-place like BF16.
TEST_F(TestOnednnWeightCacheInplace, F32Blocked_Wc2SurvivesM1Fallback) {
    const int k = 256, n = 256, m_prompt = 32, m_decode = 1;
    const data_type_t dtype = data_type_t::f32;

    auto input_prompt = make_operand(tensor_factory, m_prompt, k, dtype);
    auto input_decode = make_operand(tensor_factory, m_decode, k, dtype);
    auto weight = make_operand(tensor_factory, k, n, dtype);
    auto bias = make_bias(tensor_factory, n);
    auto ref_prompt = make_dst(tensor_factory, m_prompt, n, dtype);
    auto ref_decode = make_dst(tensor_factory, m_decode, n, dtype);
    auto out_prompt = make_dst(tensor_factory, m_prompt, n, dtype);
    auto out_decode = make_dst(tensor_factory, m_decode, n, dtype);
    const auto pristine_weight = snapshot_tensor_bytes(weight);

    tensor_t weight_for_ref;
    ASSERT_INDEPENDENT_WEIGHT_COPY(
            weight_for_ref, pristine_weight, k, n, dtype);
    ASSERT_RUN_REFERENCE(input_prompt, weight_for_ref, bias, ref_prompt);
    ASSERT_RESTORE_TENSOR_BYTES(pristine_weight, weight_for_ref);
    clear_matmul_test_caches();
    ASSERT_RUN_REFERENCE(input_decode, weight_for_ref, bias, ref_decode);
    clear_matmul_test_caches();

    WeightCacheGuard wc_guard(2);
    matmul_params params = make_params(dtype, 2);
    ASSERT_RUN_BLOCKED(params, input_prompt, weight,
            bias.get_raw_handle_unsafe(), out_prompt, m_prompt);
    EXPECT_TRUE(tensor_bytes_match(weight, pristine_weight))
            << "WC=2 must keep F32 weights plain before the M=1 AOCL fallback";
    ASSERT_RUN_BLOCKED(params, input_decode, weight,
            bias.get_raw_handle_unsafe(), out_decode, m_decode);

    compare_onednn_blocked_output(out_prompt, ref_prompt, m_prompt, n, k,
            /*is_quant=*/false, dtype,
            "F32 M=32 WC=2 must match the reference");
    compare_onednn_blocked_output(out_decode, ref_decode, m_decode, n, k,
            /*is_quant=*/false, dtype,
            "F32 M=1 AOCL fallback must consume the unchanged weight buffer");
}

#endif
