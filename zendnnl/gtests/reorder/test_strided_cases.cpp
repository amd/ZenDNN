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

/// @file test_strided_cases.cpp
/// @brief LOWOHA reorder round-trips over strided (non-contiguous) source
/// memory: quant/dequant and scaled type-conversion with optional row
/// padding.  The forward pass reads a strided source; the reverse pass writes
/// back contiguous and the result is compared against the original.

#include "reorder_test_common.hpp"

/** @fn TEST_P
 *  @param TestReorder parameterized test class to initialize parameters
 *  @param BF16_QUANT_DEQUANT_STRIDED user-defined name of test
 *  @brief Round-trip test: strided BF16 quantization (S8/U8) and dequantization
 */
TEST_P(TestReorder, BF16_QUANT_DEQUANT_STRIDED) {
  if (!use_LOWOHA) {
    GTEST_SKIP();
  }
  bool is_symmetric = lowoha_params.is_symmetric;
  data_type_t src_dtype = data_type_t::bf16;
  data_type_t quant_dtype = is_symmetric ? data_type_t::s8 : data_type_t::u8;
  log_lowoha_test_info(lowoha_params, src_dtype, quant_dtype, true, true);
  log_info("Quantization: ",
           is_symmetric ? "Symmetric (S8, no zp)" : "Asymmetric (U8, with zp)");

  std::vector<size_t> shape = get_lowoha_shape(lowoha_params);
  std::vector<size_t> strided_shape = get_lowoha_strided_shape(lowoha_params);
  std::vector<size_t> quant_shape = get_lowoha_quant_shape(lowoha_params);

  auto src_tensor = tensor_factory.uniform_dist_strided_tensor(
                      shape, strided_shape, src_dtype, 2.0f);
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

  lowoha_params.src_dtype = src_dtype;
  lowoha_params.dst_dtype = quant_dtype;
  lowoha_params.use_strided_src = true;
  lowoha_params.lowoha_algo = reorder_algo_t::native;

  status_t quant_status = lowoha_reorder_kernel_test(
                            src_tensor, quant_tensor, scale_tensor, zp_tensor,
                            lowoha_params);
  if (quant_status != status_t::success) {
    log_error("Quantization (strided) failed");
    EXPECT_TRUE(false);
    return;
  }

  auto dequant_tensor = tensor_factory.zero_tensor(shape, src_dtype);

  ReorderType dequant_params = lowoha_params;
  dequant_params.src_dtype = quant_dtype;
  dequant_params.dst_dtype = src_dtype;
  dequant_params.use_strided_src = false;
  dequant_params.lowoha_algo = reorder_algo_t::native;

  status_t dequant_status = lowoha_reorder_kernel_test(
                              quant_tensor, dequant_tensor, scale_tensor,
                              zp_tensor, dequant_params);
  if (dequant_status != status_t::success) {
    log_error("Dequantization (strided) failed");
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
 *  @param FP32_QUANT_DEQUANT_STRIDED user-defined name of test
 *  @brief Round-trip test: strided FP32 quantization (S8/U8) and dequantization
 */
TEST_P(TestReorder, FP32_QUANT_DEQUANT_STRIDED) {
  if (!use_LOWOHA) {
    GTEST_SKIP();
  }
  bool is_symmetric = lowoha_params.is_symmetric;
  data_type_t src_dtype = data_type_t::f32;
  data_type_t quant_dtype = is_symmetric ? data_type_t::s8 : data_type_t::u8;
  log_lowoha_test_info(lowoha_params, src_dtype, quant_dtype, true, true);
  log_info("Quantization: ",
           is_symmetric ? "Symmetric (S8, no zp)" : "Asymmetric (U8, with zp)");

  std::vector<size_t> shape = get_lowoha_shape(lowoha_params);
  std::vector<size_t> strided_shape = get_lowoha_strided_shape(lowoha_params);
  std::vector<size_t> quant_shape = get_lowoha_quant_shape(lowoha_params);

  auto src_tensor = tensor_factory.uniform_dist_strided_tensor(
                      shape, strided_shape, src_dtype, 2.0f);
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

  // ---- Step 1: Quantization (FP32 → S8/U8) with strided source ----
  lowoha_params.src_dtype = src_dtype;
  lowoha_params.dst_dtype = quant_dtype;
  lowoha_params.use_strided_src = true;
  lowoha_params.lowoha_algo = reorder_algo_t::native;

  status_t quant_status = lowoha_reorder_kernel_test(
                            src_tensor, quant_tensor, scale_tensor, zp_tensor,
                            lowoha_params);
  if (quant_status != status_t::success) {
    log_error("Quantization (strided) failed");
    EXPECT_TRUE(false);
    return;
  }

  // ---- Step 2: Dequantization (S8/U8 → FP32) using same scale/zp ----
  auto dequant_tensor = tensor_factory.zero_tensor(shape, src_dtype);

  ReorderType dequant_params = lowoha_params;
  dequant_params.src_dtype = quant_dtype;
  dequant_params.dst_dtype = src_dtype;
  dequant_params.use_strided_src = false;
  dequant_params.lowoha_algo = reorder_algo_t::native;

  status_t dequant_status = lowoha_reorder_kernel_test(
                              quant_tensor, dequant_tensor, scale_tensor,
                              zp_tensor, dequant_params);
  if (dequant_status != status_t::success) {
    log_error("Dequantization (strided) failed");
    EXPECT_TRUE(false);
    return;
  }

  // ---- Step 3: Compare original vs dequantized output ----
  bool is_test_successful = true;
  compare_lowoha_quant_output(src_tensor, dequant_tensor, scale_tensor,
                              lowoha_params, is_test_successful);
  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestReorder parameterized test class to initialize parameters
 *  @param FP32_BF16_CVT_STRIDED user-defined name of test
 *  @brief Round-trip test: strided FP32 <-> BF16 conversion with scale/zp
 */
TEST_P(TestReorder, FP32_BF16_CVT_STRIDED) {
  if (!use_LOWOHA) {
    GTEST_SKIP();
  }
  data_type_t src_dtype = lowoha_params.cvt_direction_swap ? data_type_t::f32 :
                          data_type_t::bf16;
  data_type_t dst_dtype = (src_dtype == data_type_t::f32) ? data_type_t::bf16 :
                          data_type_t::f32;
  log_lowoha_test_info(lowoha_params, src_dtype, dst_dtype, true, true);

  std::vector<size_t> shape = get_lowoha_shape(lowoha_params);
  std::vector<size_t> strided_shape = get_lowoha_strided_shape(lowoha_params);
  std::vector<size_t> quant_shape = get_lowoha_quant_shape(lowoha_params);

  auto src_tensor = tensor_factory.uniform_dist_strided_tensor(
                      shape, strided_shape, src_dtype, 2.0f);

  auto scale_tensor = tensor_factory.uniform_dist_tensor(
                        quant_shape, data_type_t::f32, 0.2f);
  float *scale_ptr = static_cast<float *>(scale_tensor.get_raw_handle_unsafe());
  size_t scale_nelem = scale_tensor.get_nelem();
  for (size_t i = 0; i < scale_nelem; ++i) {
    scale_ptr[i] = 0.04f + std::fabs(scale_ptr[i]);
  }

  auto zp_tensor = tensor_factory.uniform_dist_tensor(
                     quant_shape, data_type_t::s32, 64.0f);

  // ---- Step 1: Forward conversion (src_dtype → dst_dtype) with strided src ----
  auto fwd_tensor = tensor_factory.zero_tensor(shape, dst_dtype);

  lowoha_params.src_dtype = src_dtype;
  lowoha_params.dst_dtype = dst_dtype;
  lowoha_params.use_strided_src = true;
  lowoha_params.lowoha_algo = reorder_algo_t::native;

  status_t fwd_status = lowoha_reorder_kernel_test(
                          src_tensor, fwd_tensor, scale_tensor, zp_tensor,
                          lowoha_params);
  if (fwd_status != status_t::success) {
    log_error("Forward cvt (strided) failed");
    EXPECT_TRUE(false);
    return;
  }

  // ---- Step 2: Backward conversion (dst_dtype → src_dtype) ----
  auto bwd_tensor = tensor_factory.zero_tensor(shape, src_dtype);

  ReorderType bwd_params = lowoha_params;
  bwd_params.src_dtype = dst_dtype;
  bwd_params.dst_dtype = src_dtype;
  bwd_params.use_strided_src = false;
  bwd_params.lowoha_algo = reorder_algo_t::native;

  status_t bwd_status = lowoha_reorder_kernel_test(
                          fwd_tensor, bwd_tensor, scale_tensor,
                          zp_tensor, bwd_params);
  if (bwd_status != status_t::success) {
    log_error("Backward cvt (strided) failed");
    EXPECT_TRUE(false);
    return;
  }

  // ---- Step 3: Compare original vs round-trip output ----
  bool is_test_successful = true;
  compare_lowoha_quant_output(src_tensor, bwd_tensor, scale_tensor,
                              lowoha_params, is_test_successful);
  EXPECT_TRUE(is_test_successful);
}
