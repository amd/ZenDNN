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

/// @file test_dynamic_quant.cpp
/// @brief LOWOHA dynamic-quantization round-trip tests.
///
/// Like the static-quant suite, but the scale/zp are computed by the kernel
/// (dynamic_quant=true) on the forward pass and consumed by the static
/// dequant dispatcher on the reverse pass.  Covers FP32, BF16, and FP16
/// sources (incl. the FP16-typed scale-buffer write/read path).

#include "reorder_test_common.hpp"

/** @fn TEST_P
 *  @param TestReorder parameterized test class to initialize parameters
 *  @param FP32_DYN_QUANT user-defined name of test
 *  @brief Round-trip test: FP32 dynamic quantization (S8/U8) and dequantization
 */
TEST_P(TestReorder, FP32_DYN_QUANT) {
  if (!use_LOWOHA) {
    GTEST_SKIP();
  }
  bool is_symmetric = lowoha_params.is_symmetric;
  data_type_t src_dtype = data_type_t::f32;
  data_type_t compute_dtype = is_symmetric ? data_type_t::s8 : data_type_t::u8;
  log_lowoha_test_info(lowoha_params, src_dtype, compute_dtype, false, true);
  log_info("Dynamic Quantization: ",
           is_symmetric ? "Symmetric (S8)" : "Asymmetric (U8)");

  std::vector<size_t> shape = get_lowoha_shape(lowoha_params);
  std::vector<size_t> quant_shape = get_lowoha_quant_shape(lowoha_params);

  auto src_tensor = tensor_factory.uniform_dist_tensor(shape, src_dtype, 2.0f);
  auto quant_tensor = tensor_factory.zero_tensor(shape, compute_dtype);

  auto scale_tensor = tensor_factory.zero_tensor(quant_shape, data_type_t::f32);
  tensor_t zp_tensor;
  if (!is_symmetric) {
    zp_tensor = tensor_factory.zero_tensor(quant_shape, data_type_t::s32);
  }

  // ---- Step 1: Dynamic Quantization (F32 → S8/U8) ----
  lowoha_params.src_dtype = src_dtype;
  lowoha_params.dst_dtype = compute_dtype;
  lowoha_params.use_strided_src = false;
  lowoha_params.lowoha_algo = reorder_algo_t::native;

  status_t quant_status = lowoha_reorder_kernel_test(
                            src_tensor, quant_tensor, scale_tensor, zp_tensor,
                            lowoha_params, /*dynamic_quant=*/true);
  if (quant_status != status_t::success) {
    log_error("Dynamic quantization (F32 -> ",
              is_symmetric ? "S8" : "U8", ") failed");
    EXPECT_TRUE(false);
    return;
  }

  // ---- Step 2: Dequantization (S8/U8 → F32) using computed scale/zp ----
  auto dequant_tensor = tensor_factory.zero_tensor(shape, src_dtype);

  ReorderType dequant_params = lowoha_params;
  dequant_params.src_dtype = compute_dtype;
  dequant_params.dst_dtype = src_dtype;
  dequant_params.lowoha_algo = reorder_algo_t::native;

  status_t dequant_status = lowoha_reorder_kernel_test(
                              quant_tensor, dequant_tensor, scale_tensor,
                              zp_tensor, dequant_params);
  if (dequant_status != status_t::success) {
    log_error("Dequantization (", is_symmetric ? "S8" : "U8", " -> F32) failed");
    EXPECT_TRUE(false);
    return;
  }

  // ---- Step 3: Compare original F32 with dequantized F32 ----
  bool is_test_successful = true;
  compare_lowoha_quant_output(src_tensor, dequant_tensor, scale_tensor,
                              lowoha_params, is_test_successful);
  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestReorder parameterized test class to initialize parameters
 *  @param BF16_DYN_QUANT user-defined name of test
 *  @brief Round-trip test: BF16 dynamic quantization (S8/U8) and dequantization
 */
TEST_P(TestReorder, BF16_DYN_QUANT) {
  if (!use_LOWOHA) {
    GTEST_SKIP();
  }
  bool is_symmetric = lowoha_params.is_symmetric;
  data_type_t src_dtype = data_type_t::bf16;
  data_type_t compute_dtype = is_symmetric ? data_type_t::s8 : data_type_t::u8;
  log_lowoha_test_info(lowoha_params, src_dtype, compute_dtype, false, true);
  log_info("Dynamic Quantization: ",
           is_symmetric ? "Symmetric (S8)" : "Asymmetric (U8)");

  std::vector<size_t> shape = get_lowoha_shape(lowoha_params);
  std::vector<size_t> quant_shape = get_lowoha_quant_shape(lowoha_params);

  auto src_tensor = tensor_factory.uniform_dist_tensor(shape, src_dtype, 2.0f);
  auto quant_tensor = tensor_factory.zero_tensor(shape, compute_dtype);

  auto scale_tensor = tensor_factory.zero_tensor(quant_shape, data_type_t::f32);
  tensor_t zp_tensor;
  if (!is_symmetric) {
    zp_tensor = tensor_factory.zero_tensor(quant_shape, data_type_t::s32);
  }

  // ---- Step 1: Dynamic Quantization (BF16 → S8/U8) ----
  lowoha_params.src_dtype = src_dtype;
  lowoha_params.dst_dtype = compute_dtype;
  lowoha_params.use_strided_src = false;
  lowoha_params.lowoha_algo = reorder_algo_t::native;

  status_t quant_status = lowoha_reorder_kernel_test(
                            src_tensor, quant_tensor, scale_tensor, zp_tensor,
                            lowoha_params, /*dynamic_quant=*/true);
  if (quant_status != status_t::success) {
    log_error("Dynamic quantization (BF16 -> ",
              is_symmetric ? "S8" : "U8", ") failed");
    EXPECT_TRUE(false);
    return;
  }

  // ---- Step 2: Dequantization (S8/U8 → BF16) using computed scale/zp ----
  auto dequant_tensor = tensor_factory.zero_tensor(shape, src_dtype);

  ReorderType dequant_params = lowoha_params;
  dequant_params.src_dtype = compute_dtype;
  dequant_params.dst_dtype = src_dtype;
  dequant_params.lowoha_algo = reorder_algo_t::native;

  status_t dequant_status = lowoha_reorder_kernel_test(
                              quant_tensor, dequant_tensor, scale_tensor,
                              zp_tensor, dequant_params);
  if (dequant_status != status_t::success) {
    log_error("Dequantization (", is_symmetric ? "S8" : "U8",
              " -> BF16) failed");
    EXPECT_TRUE(false);
    return;
  }

  bool is_test_successful = true;
  compare_lowoha_quant_output(src_tensor, dequant_tensor, scale_tensor,
                              lowoha_params, is_test_successful);
  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestReorder parameterized test class to initialize parameters
 *  @param F16_DYN_QUANT user-defined name of test
 *  @brief Round-trip test: F16 dynamic quantization (S8/U8) and dequantization
 *
 *  The reorder operator only offers F16 in the *forward* dynamic-quant
 *  direction (f16 -> s8/u8); there is no `dynamic_quant=true` path that
 *  consumes s8/u8 input. For the round-trip image we therefore run Step 2
 *  through the static dequant dispatcher. We pick f32 as the dequant
 *  target (instead of f16) because the comparison helper already widens
 *  f16 -> f32 via `tensor.at()` on the source side, so dequanting straight
 *  to f32 keeps the compare in a single dtype with no extra narrow/widen.
 *  The companion test `F16_DYN_QUANT_F16_SCALE` exercises the `s8/u8 -> f16`
 *  static dequant path explicitly.
 *
 *  At dispatch time the FMA backend (F32-FMA vs FP16-FMA) is auto-selected
 *  by can_use_f16_fma_kernel() from the runtime AVX512-FP16 ISA status
 *  and the build-time toolchain version. Rebuild the library with
 *  -DZENDNNL_NATIVE_F32_ACCUM=ON to force F32-FMA at build time
 *  (no runtime knob).
 */
TEST_P(TestReorder, F16_DYN_QUANT) {
  if (!use_LOWOHA) {
    GTEST_SKIP();
  }
  bool is_symmetric = lowoha_params.is_symmetric;
  data_type_t src_dtype = data_type_t::f16;
  data_type_t compute_dtype = is_symmetric ? data_type_t::s8 : data_type_t::u8;
  log_lowoha_test_info(lowoha_params, src_dtype, compute_dtype, false, true);
  log_info("Dynamic Quantization: ",
           is_symmetric ? "Symmetric (S8)" : "Asymmetric (U8)");

  std::vector<size_t> shape = get_lowoha_shape(lowoha_params);
  std::vector<size_t> quant_shape = get_lowoha_quant_shape(lowoha_params);

  auto src_tensor = tensor_factory.uniform_dist_tensor(shape, src_dtype, 2.0f);
  auto quant_tensor = tensor_factory.zero_tensor(shape, compute_dtype);

  auto scale_tensor = tensor_factory.zero_tensor(quant_shape, data_type_t::f32);
  tensor_t zp_tensor;
  if (!is_symmetric) {
    zp_tensor = tensor_factory.zero_tensor(quant_shape, data_type_t::s32);
  }

  // ---- Step 1: Dynamic Quantization (F16 -> S8/U8) ----
  lowoha_params.src_dtype = src_dtype;
  lowoha_params.dst_dtype = compute_dtype;
  lowoha_params.use_strided_src = false;
  lowoha_params.lowoha_algo = reorder_algo_t::native;

  status_t quant_status = lowoha_reorder_kernel_test(
                            src_tensor, quant_tensor, scale_tensor, zp_tensor,
                            lowoha_params, /*dynamic_quant=*/true);
  if (quant_status == status_t::isa_unsupported) {
    GTEST_SKIP() << "F16 not supported: requires AVX512-FP16 ISA";
  }
  if (quant_status != status_t::success) {
    log_error("Dynamic quantization (F16 -> ",
              is_symmetric ? "S8" : "U8", ") failed");
    EXPECT_TRUE(false);
    return;
  }

  // ---- Step 2: Dequantization (S8/U8 -> F32) using computed scale/zp ----
  // Dequant runs through the *static* dequant dispatcher (there is no
  // dynamic_quant=true path that consumes s8/u8). Target dtype is f32 by
  // choice -- the comparison helper widens f16 source via tensor.at() to
  // f32, so dequanting straight to f32 keeps the compare in one dtype.
  // The s8/u8 -> f16 static dequant path is exercised by the companion
  // test F16_DYN_QUANT_F16_SCALE.
  auto dequant_tensor = tensor_factory.zero_tensor(shape, data_type_t::f32);

  ReorderType dequant_params = lowoha_params;
  dequant_params.src_dtype = compute_dtype;
  dequant_params.dst_dtype = data_type_t::f32;
  dequant_params.lowoha_algo = reorder_algo_t::native;

  status_t dequant_status = lowoha_reorder_kernel_test(
                              quant_tensor, dequant_tensor, scale_tensor,
                              zp_tensor, dequant_params);
  if (dequant_status != status_t::success) {
    log_error("Dequantization (", is_symmetric ? "S8" : "U8",
              " -> F32) failed");
    EXPECT_TRUE(false);
    return;
  }

  // ---- Step 3: Compare original F16 with F32 dequant ----
  // compare_lowoha_quant_output reads both tensors via tensor.at() which
  // widens to f32 on the fly, so the f16 source and f32 dequant compare
  // cleanly. Tolerance follows the same scale/2 + epsilon model as the
  // BF16/FP32 paths; F16 has 10 mantissa bits (more than BF16's 7), so
  // the BF16 truncation-noise epsilon is conservative for F16.
  bool is_test_successful = true;
  compare_lowoha_quant_output(src_tensor, dequant_tensor, scale_tensor,
                              lowoha_params, is_test_successful);
  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestReorder parameterized test class to initialize parameters
 *  @param F16_DYN_QUANT_F16_SCALE user-defined name of test
 *  @brief Round-trip test: F16 dynamic quantization (S8/U8) with an FP16
 *  scale buffer and dequantization back to F16.
 *
 *  Exercises the f16 narrowing path on the dynamic-quant WRITE side
 *  (dynamic_dispatch widens scale to f32 internally, then narrows back to
 *  f16 into the user buffer via float16_t::f32_to_f16_val) and the f16
 *  branch of get_scale_value on the static-dequant READ side. The companion
 *  test F16_QUANT_DEQUANT_F16_SCALE covers the read-only path with a
 *  user-supplied f16 scale; this test covers the produce-then-consume path
 *  end-to-end with the scale buffer staying in f16 throughout.
 *
 *  Dequant target is f16 (not f32 like F16_DYN_QUANT) so the same f16 scale
 *  buffer is consumed back through the static dequant dispatcher
 *  (reorder_dtype_dispatch.cpp s8/u8 -> f16 path), maximising f16 scale
 *  coverage in a single test.
 */
TEST_P(TestReorder, F16_DYN_QUANT_F16_SCALE) {
  if (!use_LOWOHA) {
    GTEST_SKIP();
  }
  bool is_symmetric = lowoha_params.is_symmetric;
  data_type_t src_dtype = data_type_t::f16;
  data_type_t compute_dtype = is_symmetric ? data_type_t::s8 : data_type_t::u8;
  log_lowoha_test_info(lowoha_params, src_dtype, compute_dtype, false, true);
  log_info("Dynamic Quantization: ",
           is_symmetric ? "Symmetric (S8)" : "Asymmetric (U8)",
           " (FP16 scale buffer)");

  std::vector<size_t> shape = get_lowoha_shape(lowoha_params);
  std::vector<size_t> quant_shape = get_lowoha_quant_shape(lowoha_params);

  auto src_tensor = tensor_factory.uniform_dist_tensor(shape, src_dtype, 2.0f);
  auto quant_tensor = tensor_factory.zero_tensor(shape, compute_dtype);

  // The scale buffer is FP16 from the outset: dynamic_dispatch computes
  // scale in f32 internally and narrows to f16 on store (see
  // dynamic_dispatch.cpp's scale_needs_narrow branch).
  auto scale_tensor = tensor_factory.zero_tensor(quant_shape, data_type_t::f16);
  tensor_t zp_tensor;
  if (!is_symmetric) {
    zp_tensor = tensor_factory.zero_tensor(quant_shape, data_type_t::s32);
  }

  lowoha_params.src_dtype = src_dtype;
  lowoha_params.dst_dtype = compute_dtype;
  lowoha_params.use_strided_src = false;
  lowoha_params.lowoha_algo = reorder_algo_t::native;

  status_t quant_status = lowoha_reorder_kernel_test(
                            src_tensor, quant_tensor, scale_tensor, zp_tensor,
                            lowoha_params, /*dynamic_quant=*/true);
  if (quant_status == status_t::isa_unsupported) {
    GTEST_SKIP() << "F16 not supported: requires AVX512-FP16 ISA";
  }
  if (quant_status != status_t::success) {
    log_error("Dynamic quantization (F16 -> ",
              is_symmetric ? "S8" : "U8", ", FP16 scale) failed");
    EXPECT_TRUE(false);
    return;
  }

  // Step 2: Static dequant (S8/U8 -> F16) reusing the same FP16 scale buffer.
  // This is the read side of the f16 scale path: get_scale_value widens
  // f16 -> f32 before handing the scalar scale to the dequant kernel.
  auto dequant_tensor = tensor_factory.zero_tensor(shape, src_dtype);
  ReorderType dequant_params = lowoha_params;
  dequant_params.src_dtype = compute_dtype;
  dequant_params.dst_dtype = src_dtype;
  dequant_params.lowoha_algo = reorder_algo_t::native;

  status_t dequant_status = lowoha_reorder_kernel_test(
                              quant_tensor, dequant_tensor, scale_tensor,
                              zp_tensor, dequant_params);
  if (dequant_status != status_t::success) {
    log_error("Dequantization (", is_symmetric ? "S8" : "U8",
              " -> F16, FP16 scale) failed");
    EXPECT_TRUE(false);
    return;
  }

  // Step 3: Compare. compare_lowoha_quant_output reads the f16 scale via
  // tensor.at() (widens to f32). Tolerance follows the shared
  // BF16/FP16 model (max_scale/2 + epsilon); the f16 narrowing on the
  // scale buffer adds at most ~scale * 2^-11 of relative error, well
  // inside that epsilon.
  bool is_test_successful = true;
  compare_lowoha_quant_output(src_tensor, dequant_tensor, scale_tensor,
                              lowoha_params, is_test_successful);
  EXPECT_TRUE(is_test_successful);
}
