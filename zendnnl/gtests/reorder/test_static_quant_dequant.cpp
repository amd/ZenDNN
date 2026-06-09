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

/// @file test_static_quant_dequant.cpp
/// @brief LOWOHA static quantization / dequantization round-trip tests.
///
/// Round-trip methodology: Source -> INT8 (S8/U8) -> Source using a
/// user-provided scale/zp, then compare the reconstructed source against the
/// original within a scale-aware tolerance.  Covers BF16, FP32, and FP16
/// sources (incl. the FP16-typed scale-buffer variant).

#include "reorder_test_common.hpp"

/** @fn TEST_P
 *  @param TestReorder parameterized test class to initialize parameters
 *  @param BF16_QUANT_DEQUANT user-defined name of test
 *  @brief Round-trip test: BF16 quantization (S8/U8) and dequantization
 */
TEST_P(TestReorder, BF16_QUANT_DEQUANT) {
  if (!use_LOWOHA) {
    GTEST_SKIP();
  }
  bool is_symmetric = lowoha_params.is_symmetric;
  data_type_t src_dtype = data_type_t::bf16;
  data_type_t quant_dtype = is_symmetric ? data_type_t::s8 : data_type_t::u8;
  log_lowoha_test_info(lowoha_params, src_dtype, quant_dtype, false, true);
  log_info("Quantization: ",
           is_symmetric ? "Symmetric (S8, no zp)" : "Asymmetric (U8, with zp)");

  std::vector<size_t> shape = get_lowoha_shape(lowoha_params);
  std::vector<size_t> quant_shape = get_lowoha_quant_shape(lowoha_params);

  auto src_tensor = tensor_factory.uniform_dist_tensor(shape, src_dtype, 2.0f);
  auto quant_tensor = tensor_factory.zero_tensor(shape, quant_dtype);

  auto scale_tensor = tensor_factory.uniform_dist_tensor(
                        quant_shape, data_type_t::f32, 0.2f);
  float *scale_ptr = static_cast<float *>(scale_tensor.get_raw_handle_unsafe());
  size_t scale_nelem = scale_tensor.get_nelem();
  for (size_t i = 0; i < scale_nelem; ++i) {
    scale_ptr[i] = 0.04f + std::fabs(scale_ptr[i]);
  }

  // ZP constrained to [55, 183] to prevent saturation with min_scale=0.04
  tensor_t zp_tensor;
  if (!is_symmetric) {
    zp_tensor = tensor_factory.uniform_dist_tensor(
                  quant_shape, data_type_t::s32, 128.0f);
    int32_t *zp_ptr = static_cast<int32_t *>(zp_tensor.get_raw_handle_unsafe());
    size_t zp_nelem = zp_tensor.get_nelem();
    for (size_t i = 0; i < zp_nelem; ++i) {
      zp_ptr[i] = 55 + (std::abs(zp_ptr[i]) % 129);
    }
  }

  // ---- Step 1: Quantization (BF16 → S8/U8) ----
  lowoha_params.src_dtype = src_dtype;
  lowoha_params.dst_dtype = quant_dtype;
  lowoha_params.use_strided_src = false;
  lowoha_params.lowoha_algo = reorder_algo_t::native;

  status_t quant_status = lowoha_reorder_kernel_test(
                            src_tensor, quant_tensor, scale_tensor, zp_tensor,
                            lowoha_params);
  if (quant_status != status_t::success) {
    log_error("Quantization (BF16 -> ",
              is_symmetric ? "S8" : "U8", ") failed");
    EXPECT_TRUE(false);
    return;
  }

  // ---- Step 2: Dequantization (S8/U8 → BF16) using same scale/zp ----
  auto dequant_tensor = tensor_factory.zero_tensor(shape, src_dtype);

  ReorderType dequant_params = lowoha_params;
  dequant_params.src_dtype = quant_dtype;
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

  // ---- Step 3: Compare original BF16 with dequantized BF16 ----
  bool is_test_successful = true;
  compare_lowoha_quant_output(src_tensor, dequant_tensor, scale_tensor,
                              lowoha_params, is_test_successful);
  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestReorder parameterized test class to initialize parameters
 *  @param F16_QUANT_DEQUANT user-defined name of test
 *  @brief Round-trip test: F16 static quantization (S8/U8) and dequantization
 *
 *  Static FP16 quant + dequant exercises both directions of the FP16 reorder
 *  surface (f16 -> s8/u8 and s8/u8 -> f16). At dispatch time the FMA backend
 *  (F32-FMA vs FP16-FMA) is auto-selected by can_use_f16_fma_kernel() from
 *  the runtime AVX512-FP16 ISA status and the build-time toolchain
 *  version. Rebuild the library with -DZENDNNL_NATIVE_F32_ACCUM=ON to
 *  force F32-FMA at build time (no runtime knob).
 */
TEST_P(TestReorder, F16_QUANT_DEQUANT) {
  if (!use_LOWOHA) {
    GTEST_SKIP();
  }
  bool is_symmetric = lowoha_params.is_symmetric;
  data_type_t src_dtype = data_type_t::f16;
  data_type_t quant_dtype = is_symmetric ? data_type_t::s8 : data_type_t::u8;
  log_lowoha_test_info(lowoha_params, src_dtype, quant_dtype, false, true);
  log_info("Quantization: ",
           is_symmetric ? "Symmetric (S8, no zp)" : "Asymmetric (U8, with zp)");

  std::vector<size_t> shape = get_lowoha_shape(lowoha_params);
  std::vector<size_t> quant_shape = get_lowoha_quant_shape(lowoha_params);

  auto src_tensor = tensor_factory.uniform_dist_tensor(shape, src_dtype, 2.0f);
  auto quant_tensor = tensor_factory.zero_tensor(shape, quant_dtype);

  auto scale_tensor = tensor_factory.uniform_dist_tensor(
                        quant_shape, data_type_t::f32, 0.2f);
  float *scale_ptr = static_cast<float *>(scale_tensor.get_raw_handle_unsafe());
  size_t scale_nelem = scale_tensor.get_nelem();
  for (size_t i = 0; i < scale_nelem; ++i) {
    scale_ptr[i] = 0.04f + std::fabs(scale_ptr[i]);
  }

  // ZP constrained to [55, 183] to prevent saturation with min_scale=0.04
  tensor_t zp_tensor;
  if (!is_symmetric) {
    zp_tensor = tensor_factory.uniform_dist_tensor(
                  quant_shape, data_type_t::s32, 128.0f);
    int32_t *zp_ptr = static_cast<int32_t *>(zp_tensor.get_raw_handle_unsafe());
    size_t zp_nelem = zp_tensor.get_nelem();
    for (size_t i = 0; i < zp_nelem; ++i) {
      zp_ptr[i] = 55 + (std::abs(zp_ptr[i]) % 129);
    }
  }

  // ---- Step 1: Quantization (F16 -> S8/U8) ----
  lowoha_params.src_dtype = src_dtype;
  lowoha_params.dst_dtype = quant_dtype;
  lowoha_params.use_strided_src = false;
  lowoha_params.lowoha_algo = reorder_algo_t::native;

  status_t quant_status = lowoha_reorder_kernel_test(
                            src_tensor, quant_tensor, scale_tensor, zp_tensor,
                            lowoha_params);
  if (quant_status == status_t::isa_unsupported) {
    GTEST_SKIP() << "F16 not supported: requires AVX512-FP16 ISA";
  }
  if (quant_status != status_t::success) {
    log_error("Quantization (F16 -> ",
              is_symmetric ? "S8" : "U8", ") failed");
    EXPECT_TRUE(false);
    return;
  }

  // ---- Step 2: Dequantization (S8/U8 -> F16) using same scale/zp ----
  auto dequant_tensor = tensor_factory.zero_tensor(shape, src_dtype);

  ReorderType dequant_params = lowoha_params;
  dequant_params.src_dtype = quant_dtype;
  dequant_params.dst_dtype = src_dtype;
  dequant_params.lowoha_algo = reorder_algo_t::native;

  status_t dequant_status = lowoha_reorder_kernel_test(
                              quant_tensor, dequant_tensor, scale_tensor,
                              zp_tensor, dequant_params);
  if (dequant_status != status_t::success) {
    log_error("Dequantization (", is_symmetric ? "S8" : "U8",
              " -> F16) failed");
    EXPECT_TRUE(false);
    return;
  }

  // ---- Step 3: Compare original F16 with dequantized F16 ----
  // compare_lowoha_quant_output reads both tensors via tensor.at() which
  // widens to f32 on the fly. The BF16 truncation-epsilon branch in the
  // helper is a conservative bound for F16 (10 mantissa bits vs BF16's 7).
  bool is_test_successful = true;
  compare_lowoha_quant_output(src_tensor, dequant_tensor, scale_tensor,
                              lowoha_params, is_test_successful);
  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestReorder parameterized test class to initialize parameters
 *  @param F16_QUANT_DEQUANT_F16_SCALE user-defined name of test
 *  @brief Round-trip test: F16 source with a user-supplied FP16 scale
 *  buffer and S8/U8 destination. Exercises the f16 branch of
 *  get_scale_value on the static quant/dequant read path. The companion
 *  test F16_DYN_QUANT_F16_SCALE covers the dynamic-quant write path that
 *  narrows scale to f16 via compute_dynamic_quant_params /
 *  dynamic_dispatch's scale_needs_narrow branch.
 *
 *  We initialize the f16 scale buffer by widening from an f32 distribution so
 *  every stored scale value is representable in f16 (10-bit mantissa) — the
 *  reference comparison reads back via get_scale_value which widens to f32,
 *  so the round-trip-through-f16 step is lossless for these inputs.
 */
TEST_P(TestReorder, F16_QUANT_DEQUANT_F16_SCALE) {
  if (!use_LOWOHA) {
    GTEST_SKIP();
  }
  bool is_symmetric = lowoha_params.is_symmetric;
  data_type_t src_dtype = data_type_t::f16;
  data_type_t quant_dtype = is_symmetric ? data_type_t::s8 : data_type_t::u8;
  log_lowoha_test_info(lowoha_params, src_dtype, quant_dtype, false, true);
  log_info("Quantization: ",
           is_symmetric ? "Symmetric (S8, no zp)" : "Asymmetric (U8, with zp)",
           " (FP16 scale buffer)");

  std::vector<size_t> shape = get_lowoha_shape(lowoha_params);
  std::vector<size_t> quant_shape = get_lowoha_quant_shape(lowoha_params);

  auto src_tensor = tensor_factory.uniform_dist_tensor(shape, src_dtype, 2.0f);
  auto quant_tensor = tensor_factory.zero_tensor(shape, quant_dtype);

  // Build scale in f32 first (to apply the [0.04, ~0.24] floor), then narrow
  // to f16 so the user buffer holds f16-encoded scale values. f16's normal
  // range easily covers this magnitude (min normal ≈ 6.1e-5).
  auto scale_tensor_f32 = tensor_factory.uniform_dist_tensor(
                            quant_shape, data_type_t::f32, 0.2f);
  float *scale_f32_ptr = static_cast<float *>(scale_tensor_f32.get_raw_handle_unsafe());
  size_t scale_nelem = scale_tensor_f32.get_nelem();
  for (size_t i = 0; i < scale_nelem; ++i) {
    scale_f32_ptr[i] = 0.04f + std::fabs(scale_f32_ptr[i]);
  }

  auto scale_tensor = tensor_factory.zero_tensor(quant_shape, data_type_t::f16);
  uint16_t *scale_f16_ptr = static_cast<uint16_t *>(scale_tensor.get_raw_handle_unsafe());
  for (size_t i = 0; i < scale_nelem; ++i) {
    scale_f16_ptr[i] = float16_t::f32_to_f16_val(scale_f32_ptr[i]);
  }

  tensor_t zp_tensor;
  if (!is_symmetric) {
    zp_tensor = tensor_factory.uniform_dist_tensor(
                  quant_shape, data_type_t::s32, 128.0f);
    int32_t *zp_ptr = static_cast<int32_t *>(zp_tensor.get_raw_handle_unsafe());
    size_t zp_nelem = zp_tensor.get_nelem();
    for (size_t i = 0; i < zp_nelem; ++i) {
      zp_ptr[i] = 55 + (std::abs(zp_ptr[i]) % 129);
    }
  }

  // Step 1: Quantization (F16 src, FP16 scale -> S8/U8)
  lowoha_params.src_dtype = src_dtype;
  lowoha_params.dst_dtype = quant_dtype;
  lowoha_params.use_strided_src = false;
  lowoha_params.lowoha_algo = reorder_algo_t::native;

  status_t quant_status = lowoha_reorder_kernel_test(
                            src_tensor, quant_tensor, scale_tensor, zp_tensor,
                            lowoha_params);
  if (quant_status == status_t::isa_unsupported) {
    GTEST_SKIP() << "F16 not supported: requires AVX512-FP16 ISA";
  }
  if (quant_status != status_t::success) {
    log_error("Quantization (F16 src, FP16 scale -> ",
              is_symmetric ? "S8" : "U8", ") failed");
    EXPECT_TRUE(false);
    return;
  }

  // Step 2: Dequantization (S8/U8 -> F16) reusing the same FP16 scale buffer.
  auto dequant_tensor = tensor_factory.zero_tensor(shape, src_dtype);
  ReorderType dequant_params = lowoha_params;
  dequant_params.src_dtype = quant_dtype;
  dequant_params.dst_dtype = src_dtype;
  dequant_params.lowoha_algo = reorder_algo_t::native;

  status_t dequant_status = lowoha_reorder_kernel_test(
                              quant_tensor, dequant_tensor, scale_tensor,
                              zp_tensor, dequant_params);
  if (dequant_status != status_t::success) {
    log_error("Dequantization (", is_symmetric ? "S8" : "U8",
              " -> F16) failed");
    EXPECT_TRUE(false);
    return;
  }

  // Step 3: Compare. compare_lowoha_quant_output widens the FP16 scale via
  // tensor.at(), then computes max_scale and tolerance off that. The f16
  // narrowing on the scale buffer adds at most ~scale * 2^-11 of relative
  // error; the shared BF16/FP16 tolerance epsilon (max_scale/2 + 0.03)
  // absorbs this comfortably.
  bool is_test_successful = true;
  compare_lowoha_quant_output(src_tensor, dequant_tensor, scale_tensor,
                              lowoha_params, is_test_successful);
  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestReorder parameterized test class to initialize parameters
 *  @param FP32_QUANT_DEQUANT user-defined name of test
 *  @brief Round-trip test: FP32 quantization (S8/U8) and dequantization
 */
TEST_P(TestReorder, FP32_QUANT_DEQUANT) {
  if (!use_LOWOHA) {
    GTEST_SKIP();
  }
  bool is_symmetric = lowoha_params.is_symmetric;
  data_type_t src_dtype = data_type_t::f32;
  data_type_t quant_dtype = is_symmetric ? data_type_t::s8 : data_type_t::u8;
  log_lowoha_test_info(lowoha_params, src_dtype, quant_dtype, false, true);
  log_info("Quantization: ",
           is_symmetric ? "Symmetric (S8, no zp)" : "Asymmetric (U8, with zp)");

  std::vector<size_t> shape = get_lowoha_shape(lowoha_params);
  std::vector<size_t> quant_shape = get_lowoha_quant_shape(lowoha_params);

  auto src_tensor = tensor_factory.uniform_dist_tensor(shape, src_dtype, 2.0f);
  auto quant_tensor = tensor_factory.zero_tensor(shape, quant_dtype);

  auto scale_tensor = tensor_factory.uniform_dist_tensor(
                        quant_shape, data_type_t::f32, 0.2f);
  float *scale_ptr = static_cast<float *>(scale_tensor.get_raw_handle_unsafe());
  size_t scale_nelem = scale_tensor.get_nelem();
  for (size_t i = 0; i < scale_nelem; ++i) {
    scale_ptr[i] = 0.04f + std::fabs(scale_ptr[i]);
  }

  // ZP constrained to [55, 183] to prevent saturation with min_scale=0.04
  tensor_t zp_tensor;
  if (!is_symmetric) {
    zp_tensor = tensor_factory.uniform_dist_tensor(
                  quant_shape, data_type_t::s32, 128.0f);
    int32_t *zp_ptr = static_cast<int32_t *>(zp_tensor.get_raw_handle_unsafe());
    size_t zp_nelem = zp_tensor.get_nelem();
    for (size_t i = 0; i < zp_nelem; ++i) {
      zp_ptr[i] = 55 + (std::abs(zp_ptr[i]) % 129);
    }
  }

  // ---- Step 1: Quantization (FP32 → S8/U8) ----
  lowoha_params.src_dtype = src_dtype;
  lowoha_params.dst_dtype = quant_dtype;
  lowoha_params.use_strided_src = false;
  lowoha_params.lowoha_algo = reorder_algo_t::native;

  status_t quant_status = lowoha_reorder_kernel_test(
                            src_tensor, quant_tensor, scale_tensor, zp_tensor,
                            lowoha_params);
  if (quant_status != status_t::success) {
    log_error("Quantization (FP32 -> ",
              is_symmetric ? "S8" : "U8", ") failed");
    EXPECT_TRUE(false);
    return;
  }

  // ---- Step 2: Dequantization (S8/U8 → FP32) using same scale/zp ----
  auto dequant_tensor = tensor_factory.zero_tensor(shape, src_dtype);

  ReorderType dequant_params = lowoha_params;
  dequant_params.src_dtype = quant_dtype;
  dequant_params.dst_dtype = src_dtype;
  dequant_params.lowoha_algo = reorder_algo_t::native;

  status_t dequant_status = lowoha_reorder_kernel_test(
                              quant_tensor, dequant_tensor, scale_tensor,
                              zp_tensor, dequant_params);
  if (dequant_status != status_t::success) {
    log_error("Dequantization (", is_symmetric ? "S8" : "U8",
              " -> FP32) failed");
    EXPECT_TRUE(false);
    return;
  }

  // ---- Step 3: Compare original FP32 with dequantized FP32 ----
  bool is_test_successful = true;
  compare_lowoha_quant_output(src_tensor, dequant_tensor, scale_tensor,
                              lowoha_params, is_test_successful);
  EXPECT_TRUE(is_test_successful);
}
