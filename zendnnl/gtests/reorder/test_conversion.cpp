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

/// @file test_conversion.cpp
/// @brief LOWOHA type-conversion round-trip tests (no quantization to INT8).
///
/// Round-trips between floating-point dtypes — FP32<->BF16, FP32<->F16,
/// BF16<->F16 — both without scale/zp and with a scale/zp applied.  Each test
/// converts forward then back and compares against the original within a
/// tolerance pinned to the lossy intermediate dtype.

#include "reorder_test_common.hpp"

/** @fn TEST_P
 *  @param TestReorder parameterized test class to initialize parameters
 *  @param FP32_BF16_CVT user-defined name of test
 *  @brief Round-trip test: FP32 <-> BF16 type conversion without scale/zp
 */
TEST_P(TestReorder, FP32_BF16_CVT) {
  if (!use_LOWOHA) {
    GTEST_SKIP();
  }
  data_type_t src_dtype = lowoha_params.cvt_direction_swap ? data_type_t::f32 :
                          data_type_t::bf16;
  data_type_t dst_dtype = (src_dtype == data_type_t::f32) ? data_type_t::bf16 :
                          data_type_t::f32;
  log_lowoha_test_info(lowoha_params, src_dtype, dst_dtype, false, false);

  std::vector<size_t> shape = get_lowoha_shape(lowoha_params);

  auto src_tensor = tensor_factory.uniform_dist_tensor(shape, src_dtype, 2.0f);

  // No scale/zp tensors for simple conversion
  tensor_t scale_tensor;
  tensor_t zp_tensor;

  // ---- Step 1: Forward conversion (src_dtype → dst_dtype) ----
  auto fwd_tensor = tensor_factory.zero_tensor(shape, dst_dtype);

  lowoha_params.src_dtype = src_dtype;
  lowoha_params.dst_dtype = dst_dtype;
  lowoha_params.use_strided_src = false;
  lowoha_params.lowoha_algo = reorder_algo_t::native;

  status_t fwd_status = lowoha_reorder_kernel_test(
                          src_tensor, fwd_tensor, scale_tensor, zp_tensor,
                          lowoha_params);
  if (fwd_status != status_t::success) {
    log_error("Forward cvt failed");
    EXPECT_TRUE(false);
    return;
  }

  // ---- Step 2: Backward conversion (dst_dtype → src_dtype) ----
  auto bwd_tensor = tensor_factory.zero_tensor(shape, src_dtype);

  ReorderType bwd_params = lowoha_params;
  bwd_params.src_dtype = dst_dtype;
  bwd_params.dst_dtype = src_dtype;
  bwd_params.lowoha_algo = reorder_algo_t::native;

  status_t bwd_status = lowoha_reorder_kernel_test(
                          fwd_tensor, bwd_tensor, scale_tensor,
                          zp_tensor, bwd_params);
  if (bwd_status != status_t::success) {
    log_error("Backward cvt failed");
    EXPECT_TRUE(false);
    return;
  }

  // ---- Step 3: Compare original vs round-trip output ----
  ReorderType cmp_params = lowoha_params;
  cmp_params.dst_dtype = data_type_t::bf16;
  bool is_test_successful = true;
  compare_lowoha_reorder_output(bwd_tensor, src_tensor, cmp_params,
                                is_test_successful);
  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestReorder parameterized test class to initialize parameters
 *  @param FP32_BF16_CVT_SCALED user-defined name of test
 *  @brief Round-trip test: FP32 <-> BF16 type conversion with scale/zp
 */
TEST_P(TestReorder, FP32_BF16_CVT_SCALED) {
  if (!use_LOWOHA) {
    GTEST_SKIP();
  }
  data_type_t src_dtype = lowoha_params.cvt_direction_swap ? data_type_t::f32 :
                          data_type_t::bf16;
  data_type_t dst_dtype = (src_dtype == data_type_t::f32) ? data_type_t::bf16 :
                          data_type_t::f32;
  log_lowoha_test_info(lowoha_params, src_dtype, dst_dtype, false, true);

  std::vector<size_t> shape = get_lowoha_shape(lowoha_params);
  std::vector<size_t> quant_shape = get_lowoha_quant_shape(lowoha_params);

  auto src_tensor = tensor_factory.uniform_dist_tensor(shape, src_dtype, 2.0f);

  auto scale_tensor = tensor_factory.uniform_dist_tensor(
                        quant_shape, data_type_t::f32, 0.2f);
  float *scale_ptr = static_cast<float *>(scale_tensor.get_raw_handle_unsafe());
  size_t scale_nelem = scale_tensor.get_nelem();
  for (size_t i = 0; i < scale_nelem; ++i) {
    scale_ptr[i] = 0.04f + std::fabs(scale_ptr[i]);
  }

  auto zp_tensor = tensor_factory.uniform_dist_tensor(
                     quant_shape, data_type_t::s32, 64.0f);

  // ---- Step 1: Forward conversion (src_dtype → dst_dtype) with scale/zp ----
  auto fwd_tensor = tensor_factory.zero_tensor(shape, dst_dtype);

  lowoha_params.src_dtype = src_dtype;
  lowoha_params.dst_dtype = dst_dtype;
  lowoha_params.use_strided_src = false;
  lowoha_params.lowoha_algo = reorder_algo_t::native;

  status_t fwd_status = lowoha_reorder_kernel_test(
                          src_tensor, fwd_tensor, scale_tensor, zp_tensor,
                          lowoha_params);
  if (fwd_status != status_t::success) {
    log_error("Forward cvt (scaled) failed");
    EXPECT_TRUE(false);
    return;
  }

  // ---- Step 2: Backward conversion (dst_dtype → src_dtype) with scale/zp ----
  auto bwd_tensor = tensor_factory.zero_tensor(shape, src_dtype);

  ReorderType bwd_params = lowoha_params;
  bwd_params.src_dtype = dst_dtype;
  bwd_params.dst_dtype = src_dtype;
  bwd_params.lowoha_algo = reorder_algo_t::native;

  status_t bwd_status = lowoha_reorder_kernel_test(
                          fwd_tensor, bwd_tensor, scale_tensor,
                          zp_tensor, bwd_params);
  if (bwd_status != status_t::success) {
    log_error("Backward cvt (scaled) failed");
    EXPECT_TRUE(false);
    return;
  }

  // ---- Step 3: Compare original vs round-trip output ----
  bool is_test_successful = true;
  compare_lowoha_quant_output(src_tensor, bwd_tensor, scale_tensor,
                              lowoha_params, is_test_successful);
  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestReorder parameterized test class to initialize parameters
 *  @param FP32_F16_CVT user-defined name of test
 *  @brief Round-trip test: FP32 <-> F16 type conversion without scale/zp
 */
TEST_P(TestReorder, FP32_F16_CVT) {
  if (!use_LOWOHA) {
    GTEST_SKIP();
  }
  data_type_t src_dtype = lowoha_params.cvt_direction_swap ? data_type_t::f32 :
                          data_type_t::f16;
  data_type_t dst_dtype = (src_dtype == data_type_t::f32) ? data_type_t::f16 :
                          data_type_t::f32;
  log_lowoha_test_info(lowoha_params, src_dtype, dst_dtype, false, false);

  std::vector<size_t> shape = get_lowoha_shape(lowoha_params);

  auto src_tensor = tensor_factory.uniform_dist_tensor(shape, src_dtype, 2.0f);

  // No scale/zp tensors for simple conversion
  tensor_t scale_tensor;
  tensor_t zp_tensor;

  // ---- Step 1: Forward conversion (src_dtype → dst_dtype) ----
  auto fwd_tensor = tensor_factory.zero_tensor(shape, dst_dtype);

  lowoha_params.src_dtype = src_dtype;
  lowoha_params.dst_dtype = dst_dtype;
  lowoha_params.use_strided_src = false;
  lowoha_params.lowoha_algo = reorder_algo_t::native;

  status_t fwd_status = lowoha_reorder_kernel_test(
                          src_tensor, fwd_tensor, scale_tensor, zp_tensor,
                          lowoha_params);
  if (fwd_status != status_t::success) {
    log_error("Forward cvt failed");
    EXPECT_TRUE(false);
    return;
  }

  // ---- Step 2: Backward conversion (dst_dtype → src_dtype) ----
  auto bwd_tensor = tensor_factory.zero_tensor(shape, src_dtype);

  ReorderType bwd_params = lowoha_params;
  bwd_params.src_dtype = dst_dtype;
  bwd_params.dst_dtype = src_dtype;
  bwd_params.lowoha_algo = reorder_algo_t::native;

  status_t bwd_status = lowoha_reorder_kernel_test(
                          fwd_tensor, bwd_tensor, scale_tensor,
                          zp_tensor, bwd_params);
  if (bwd_status != status_t::success) {
    log_error("Backward cvt failed");
    EXPECT_TRUE(false);
    return;
  }

  // ---- Step 3: Compare original vs round-trip output ----
  // Pin tolerance to the lossy intermediate dtype (f16) for the round-trip.
  ReorderType cmp_params = lowoha_params;
  cmp_params.dst_dtype = data_type_t::f16;
  bool is_test_successful = true;
  compare_lowoha_reorder_output(bwd_tensor, src_tensor, cmp_params,
                                is_test_successful);
  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestReorder parameterized test class to initialize parameters
 *  @param FP32_F16_CVT_SCALED user-defined name of test
 *  @brief Round-trip test: FP32 <-> F16 type conversion with scale/zp
 */
TEST_P(TestReorder, FP32_F16_CVT_SCALED) {
  if (!use_LOWOHA) {
    GTEST_SKIP();
  }
  data_type_t src_dtype = lowoha_params.cvt_direction_swap ? data_type_t::f32 :
                          data_type_t::f16;
  data_type_t dst_dtype = (src_dtype == data_type_t::f32) ? data_type_t::f16 :
                          data_type_t::f32;
  log_lowoha_test_info(lowoha_params, src_dtype, dst_dtype, false, true);

  std::vector<size_t> shape = get_lowoha_shape(lowoha_params);
  std::vector<size_t> quant_shape = get_lowoha_quant_shape(lowoha_params);

  auto src_tensor = tensor_factory.uniform_dist_tensor(shape, src_dtype, 2.0f);

  auto scale_tensor = tensor_factory.uniform_dist_tensor(
                        quant_shape, data_type_t::f32, 0.2f);
  float *scale_ptr = static_cast<float *>(scale_tensor.get_raw_handle_unsafe());
  size_t scale_nelem = scale_tensor.get_nelem();
  for (size_t i = 0; i < scale_nelem; ++i) {
    scale_ptr[i] = 0.04f + std::fabs(scale_ptr[i]);
  }

  auto zp_tensor = tensor_factory.uniform_dist_tensor(
                     quant_shape, data_type_t::s32, 64.0f);

  // ---- Step 1: Forward conversion (src_dtype → dst_dtype) with scale/zp ----
  auto fwd_tensor = tensor_factory.zero_tensor(shape, dst_dtype);

  lowoha_params.src_dtype = src_dtype;
  lowoha_params.dst_dtype = dst_dtype;
  lowoha_params.use_strided_src = false;
  lowoha_params.lowoha_algo = reorder_algo_t::reference;

  status_t fwd_status = lowoha_reorder_kernel_test(
                          src_tensor, fwd_tensor, scale_tensor, zp_tensor,
                          lowoha_params);
  if (fwd_status != status_t::success) {
    log_error("Forward cvt (scaled) failed");
    EXPECT_TRUE(false);
    return;
  }

  // ---- Step 2: Backward conversion (dst_dtype → src_dtype) with scale/zp ----
  auto bwd_tensor = tensor_factory.zero_tensor(shape, src_dtype);

  ReorderType bwd_params = lowoha_params;
  bwd_params.src_dtype = dst_dtype;
  bwd_params.dst_dtype = src_dtype;
  bwd_params.lowoha_algo = reorder_algo_t::native;

  status_t bwd_status = lowoha_reorder_kernel_test(
                          fwd_tensor, bwd_tensor, scale_tensor,
                          zp_tensor, bwd_params);
  if (bwd_status != status_t::success) {
    log_error("Backward cvt (scaled) failed");
    EXPECT_TRUE(false);
    return;
  }

  // ---- Step 3: Compare original vs round-trip output ----
  bool is_test_successful = true;
  compare_lowoha_quant_output(src_tensor, bwd_tensor, scale_tensor,
                              lowoha_params, is_test_successful);
  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestReorder parameterized test class to initialize parameters
 *  @param BF16_F16_CVT user-defined name of test
 *  @brief Round-trip test: BF16 <-> F16 type conversion without scale/zp
 */
TEST_P(TestReorder, BF16_F16_CVT) {
  if (!use_LOWOHA) {
    GTEST_SKIP();
  }
  data_type_t src_dtype = lowoha_params.cvt_direction_swap ? data_type_t::bf16 :
                          data_type_t::f16;
  data_type_t dst_dtype = (src_dtype == data_type_t::bf16) ? data_type_t::f16 :
                          data_type_t::bf16;
  log_lowoha_test_info(lowoha_params, src_dtype, dst_dtype, false, false);

  std::vector<size_t> shape = get_lowoha_shape(lowoha_params);

  auto src_tensor = tensor_factory.uniform_dist_tensor(shape, src_dtype, 2.0f);

  // No scale/zp tensors for simple conversion
  tensor_t scale_tensor;
  tensor_t zp_tensor;

  // ---- Step 1: Forward conversion (src_dtype → dst_dtype) ----
  auto fwd_tensor = tensor_factory.zero_tensor(shape, dst_dtype);

  lowoha_params.src_dtype = src_dtype;
  lowoha_params.dst_dtype = dst_dtype;
  lowoha_params.use_strided_src = false;
  lowoha_params.lowoha_algo = reorder_algo_t::native;

  status_t fwd_status = lowoha_reorder_kernel_test(
                          src_tensor, fwd_tensor, scale_tensor, zp_tensor,
                          lowoha_params);
  if (fwd_status != status_t::success) {
    log_error("Forward cvt failed");
    EXPECT_TRUE(false);
    return;
  }

  // ---- Step 2: Backward conversion (dst_dtype → src_dtype) ----
  auto bwd_tensor = tensor_factory.zero_tensor(shape, src_dtype);

  ReorderType bwd_params = lowoha_params;
  bwd_params.src_dtype = dst_dtype;
  bwd_params.dst_dtype = src_dtype;
  bwd_params.lowoha_algo = reorder_algo_t::native;

  status_t bwd_status = lowoha_reorder_kernel_test(
                          fwd_tensor, bwd_tensor, scale_tensor,
                          zp_tensor, bwd_params);
  if (bwd_status != status_t::success) {
    log_error("Backward cvt failed");
    EXPECT_TRUE(false);
    return;
  }

  // ---- Step 3: Compare original vs round-trip output ----
  // Pin tolerance to the narrower mantissa (bf16 = 7 bits) for round-trip.
  ReorderType cmp_params = lowoha_params;
  cmp_params.dst_dtype = data_type_t::bf16;
  bool is_test_successful = true;
  compare_lowoha_reorder_output(bwd_tensor, src_tensor, cmp_params,
                                is_test_successful);
  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestReorder parameterized test class to initialize parameters
 *  @param BF16_F16_CVT_SCALED user-defined name of test
 *  @brief Round-trip test: BF16 <-> F16 type conversion with scale/zp
 */
TEST_P(TestReorder, BF16_F16_CVT_SCALED) {
  if (!use_LOWOHA) {
    GTEST_SKIP();
  }
  data_type_t src_dtype = lowoha_params.cvt_direction_swap ? data_type_t::bf16 :
                          data_type_t::f16;
  data_type_t dst_dtype = (src_dtype == data_type_t::bf16) ? data_type_t::f16 :
                          data_type_t::bf16;
  log_lowoha_test_info(lowoha_params, src_dtype, dst_dtype, false, true);

  std::vector<size_t> shape = get_lowoha_shape(lowoha_params);
  std::vector<size_t> quant_shape = get_lowoha_quant_shape(lowoha_params);

  auto src_tensor = tensor_factory.uniform_dist_tensor(shape, src_dtype, 2.0f);

  auto scale_tensor = tensor_factory.uniform_dist_tensor(
                        quant_shape, data_type_t::f32, 0.2f);
  float *scale_ptr = static_cast<float *>(scale_tensor.get_raw_handle_unsafe());
  size_t scale_nelem = scale_tensor.get_nelem();
  for (size_t i = 0; i < scale_nelem; ++i) {
    scale_ptr[i] = 0.04f + std::fabs(scale_ptr[i]);
  }

  auto zp_tensor = tensor_factory.uniform_dist_tensor(
                     quant_shape, data_type_t::s32, 64.0f);

  // ---- Step 1: Forward conversion (src_dtype → dst_dtype) with scale/zp ----
  auto fwd_tensor = tensor_factory.zero_tensor(shape, dst_dtype);

  lowoha_params.src_dtype = src_dtype;
  lowoha_params.dst_dtype = dst_dtype;
  lowoha_params.use_strided_src = false;
  lowoha_params.lowoha_algo = reorder_algo_t::reference;

  status_t fwd_status = lowoha_reorder_kernel_test(
                          src_tensor, fwd_tensor, scale_tensor, zp_tensor,
                          lowoha_params);
  if (fwd_status != status_t::success) {
    log_error("Forward cvt (scaled) failed");
    EXPECT_TRUE(false);
    return;
  }

  // ---- Step 2: Backward conversion (dst_dtype → src_dtype) with scale/zp ----
  auto bwd_tensor = tensor_factory.zero_tensor(shape, src_dtype);

  ReorderType bwd_params = lowoha_params;
  bwd_params.src_dtype = dst_dtype;
  bwd_params.dst_dtype = src_dtype;
  bwd_params.lowoha_algo = reorder_algo_t::native;

  status_t bwd_status = lowoha_reorder_kernel_test(
                          fwd_tensor, bwd_tensor, scale_tensor,
                          zp_tensor, bwd_params);
  if (bwd_status != status_t::success) {
    log_error("Backward cvt (scaled) failed");
    EXPECT_TRUE(false);
    return;
  }

  // ---- Step 3: Compare original vs round-trip output ----
  // compare_lowoha_quant_output already accounts for both bf16 and f16 being
  // present (involves_bf16 && involves_f16) by widening the tolerance.
  bool is_test_successful = true;
  compare_lowoha_quant_output(src_tensor, bwd_tensor, scale_tensor,
                              lowoha_params, is_test_successful);
  EXPECT_TRUE(is_test_successful);
}
