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

#include <gtest/gtest.h>
#include "gtest_utils.hpp"

/** @brief TestSoftmax is a parameterized test class for softmax (OneDNN vs Reference) */
class TestSoftmax : public ::testing::TestWithParam<SoftmaxType> {
 protected:
  virtual void SetUp() {
    SoftmaxType params = GetParam();
    ndims        = params.ndims;
    for (int i = 0; i < ndims; ++i) {
      shape.push_back(params.shape[i]);
    }
    axis         = params.axis;
    log_softmax  = params.log_softmax;
    softmin      = params.softmin;
    num_threads  = params.num_threads;
    omp_set_num_threads(num_threads);

    total_elements = 1;
    for (auto d : shape) {
      total_elements *= d;
    }

    log_info("ndims: ", ndims,
             " total_elements: ", total_elements,
             " axis: ", axis,
             " log_softmax: ", log_softmax,
             " softmin: ", softmin,
             " num_threads: ", num_threads);
  }

  virtual void TearDown() {}

  /**
   * @brief Build a softmax_params struct from test fixture members.
   *
   * Returns the status of setup_softmax_shape() so callers can abort the
   * test early (e.g., via ASSERT_EQ) if shape/axis validation fails,
   * rather than proceeding with invalid params.
   */
  status_t build_params(data_type_t test_src_dt, data_type_t test_dst_dt,
                        softmax_params &sp) {
    uint64_t shape_arr[SOFTMAX_MAX_NDIMS];
    for (int i = 0; i < ndims; ++i) {
      shape_arr[i] = shape[i];
    }
    status_t st = setup_softmax_shape(sp, shape_arr, ndims, axis);
    if (st != status_t::success) {
      return st;
    }
    sp.src_dt       = test_src_dt;
    sp.dst_dt       = test_dst_dt;
    sp.log_softmax  = log_softmax;
    sp.softmin      = softmin;
    sp.num_threads  = num_threads;
    return st;
  }

  int ndims;
  std::vector<uint64_t> shape;
  int axis;
  bool log_softmax;
  bool softmin;
  int32_t num_threads;
  uint64_t total_elements;

  tensor_factory_t tensor_factory{};
};

/** @fn TEST_P
 *  @param TestSoftmax parameterized test class
 *  @param F32_F32 src=f32, dst=f32
 *  @brief Validates softmax OneDNN kernel against reference kernel (f32 I/O)
 */
TEST_P(TestSoftmax, F32_F32) {
#if !ZENDNNL_DEPENDS_ONEDNN
  GTEST_SKIP() << "OneDNN backend not compiled in; softmax_direct would "
               "fall back to the reference kernel, making this test "
               "reference-vs-reference.";
#endif
  data_type_t dt = data_type_t::f32;

  auto input_tensor      = tensor_factory.uniform_dist_tensor(shape, dt, 2.0f);
  auto output_tensor     = tensor_factory.zero_tensor(shape, dt);
  auto output_tensor_ref = tensor_factory.zero_tensor(shape, dt);

  softmax_params sp{};
  softmax_params ref_sp{};
  ASSERT_EQ(build_params(dt, dt, sp),     status_t::success);
  ASSERT_EQ(build_params(dt, dt, ref_sp), status_t::success);

  status_t status = softmax_kernel_test(
                      input_tensor.get_raw_handle_unsafe(),
                      output_tensor.get_raw_handle_unsafe(), sp);
  log_info("F32_F32 OneDNN kernel status: ",
           (status == status_t::success) ? "success" : "failure");

  status_t ref_status = softmax_forced_ref_kernel_test(
                          input_tensor.get_raw_handle_unsafe(),
                          output_tensor_ref.get_raw_handle_unsafe(), ref_sp);
  log_info("F32_F32 reference kernel status: ",
           (ref_status == status_t::success) ? "success" : "failure");

  bool is_test_successful =
    (status == status_t::success && ref_status == status_t::success);

  if (is_test_successful) {
    compare_softmax_tensors(output_tensor, output_tensor_ref,
                            shape, total_elements,
                            SOFTMAX_F32_TOL, is_test_successful);
  }

  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestSoftmax parameterized test class
 *  @param BF16_BF16 src=bf16, dst=bf16
 *  @brief Validates softmax OneDNN kernel against reference kernel (bf16 I/O)
 */
TEST_P(TestSoftmax, BF16_BF16) {
#if !ZENDNNL_DEPENDS_ONEDNN
  GTEST_SKIP() << "OneDNN backend not compiled in; softmax_direct would "
               "fall back to the reference kernel, making this test "
               "reference-vs-reference.";
#endif
  data_type_t dt = data_type_t::bf16;

  auto input_tensor      = tensor_factory.uniform_dist_tensor(shape, dt, 2.0f);
  auto output_tensor     = tensor_factory.zero_tensor(shape, dt);
  auto output_tensor_ref = tensor_factory.zero_tensor(shape, dt);

  softmax_params sp{};
  softmax_params ref_sp{};
  ASSERT_EQ(build_params(dt, dt, sp),     status_t::success);
  ASSERT_EQ(build_params(dt, dt, ref_sp), status_t::success);

  status_t status = softmax_kernel_test(
                      input_tensor.get_raw_handle_unsafe(),
                      output_tensor.get_raw_handle_unsafe(), sp);
  log_info("BF16_BF16 OneDNN kernel status: ",
           (status == status_t::success) ? "success" : "failure");

  status_t ref_status = softmax_forced_ref_kernel_test(
                          input_tensor.get_raw_handle_unsafe(),
                          output_tensor_ref.get_raw_handle_unsafe(), ref_sp);
  log_info("BF16_BF16 reference kernel status: ",
           (ref_status == status_t::success) ? "success" : "failure");

  bool is_test_successful =
    (status == status_t::success && ref_status == status_t::success);

  if (is_test_successful) {
    compare_softmax_tensors(output_tensor, output_tensor_ref,
                            shape, total_elements,
                            SOFTMAX_BF16_TOL, is_test_successful);
  }

  EXPECT_TRUE(is_test_successful);
}

/** @fn INSTANTIATE_TEST_SUITE_P
 *  @brief Triggers Softmax parameterized test suite
 */
INSTANTIATE_TEST_SUITE_P(Softmax, TestSoftmax,
                         ::testing::ValuesIn(softmax_test));
