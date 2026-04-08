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
#include <cmath>
#include <cstring>
#include "gtest_utils.hpp"

/** @brief TestNormalization is a parameterized test class for all normalization types */
class TestNormalization : public ::testing::TestWithParam<NormalizationType> {
 protected:
  virtual void SetUp() {
    NormalizationType params = GetParam();
    norm_type     = params.norm_type;
    batch         = params.batch;
    norm_size     = params.norm_size;
    num_channels  = params.num_channels;
    shape         = params.shape;
    epsilon       = params.epsilon;
    use_scale     = params.use_scale;
    use_shift     = params.use_shift;
    gamma_dt      = params.gamma_dt;
    beta_dt       = params.beta_dt;
    num_threads   = params.num_threads;
    omp_set_num_threads(num_threads);

    total_elements = 1;
    for (auto d : shape) {
      total_elements *= d;
    }

    gamma_size = (norm_type == norm_type_t::BATCH_NORM) ?
                 num_channels : norm_size;

    log_info("norm_type: ", norm_type_to_str(norm_type),
             " batch: ", batch,
             " norm_size: ", norm_size,
             " total_elements: ", total_elements,
             " epsilon: ", epsilon,
             " use_scale: ", use_scale, " use_shift: ", use_shift,
             " gamma_dt: ", dtype_info(gamma_dt),
             " beta_dt: ", dtype_info(beta_dt),
             " num_threads: ", num_threads);
  }

  virtual void TearDown() {}

  /**
   * @brief Build a norm_params struct from test fixture members
   */
  norm_params build_params(data_type_t src_dt, data_type_t dst_dt) {
    norm_params np;
    np.norm_type    = norm_type;
    np.batch        = batch;
    np.norm_size    = norm_size;
    np.num_channels = num_channels;
    np.epsilon      = epsilon;
    np.use_scale    = use_scale;
    np.use_shift    = use_shift;
    np.src_dt       = src_dt;
    np.dst_dt       = dst_dt;
    np.gamma_dt     = gamma_dt;
    np.beta_dt      = beta_dt;
    np.num_threads  = num_threads;
    return np;
  }

  /**
   * @brief Create optional tensors that depend on the normalization type
   */
  void create_optional_tensors(data_type_t src_dt) {
    gamma_tensor = use_scale ?
                   tensor_factory.uniform_dist_tensor({gamma_size}, gamma_dt, 1.0f) :
                   tensor_t();

    bool needs_beta = use_shift &&
                      (norm_type == norm_type_t::LAYER_NORM ||
                       norm_type == norm_type_t::BATCH_NORM);
    beta_tensor = needs_beta ?
                  tensor_factory.uniform_dist_tensor({gamma_size}, beta_dt, 1.0f) :
                  tensor_t();

    if (norm_type == norm_type_t::BATCH_NORM) {
      running_mean_tensor = tensor_factory.uniform_dist_tensor(
      {num_channels}, data_type_t::f32, 2.0f);
      running_var_tensor  = tensor_factory.uniform_dist_tensor(
      {num_channels}, data_type_t::f32, 1.0f);
      // Ensure running_var is positive
      float *var_ptr = static_cast<float *>(
                         running_var_tensor.get_raw_handle_unsafe());
      for (uint64_t i = 0; i < num_channels; ++i) {
        var_ptr[i] = std::fabs(var_ptr[i]) + 0.1f;
      }
    }
    else {
      running_mean_tensor = tensor_t();
      running_var_tensor  = tensor_t();
    }

    if (norm_type == norm_type_t::FUSED_ADD_RMS_NORM) {
      residual_tensor     = tensor_factory.uniform_dist_tensor(shape, src_dt, 2.0f);
      residual_tensor_ref = tensor_factory.zero_tensor(shape, src_dt);
      size_t elem_bytes = (src_dt == data_type_t::bf16) ? 2 : 4;
      std::memcpy(residual_tensor_ref.get_raw_handle_unsafe(),
                  residual_tensor.get_raw_handle_unsafe(),
                  total_elements * elem_bytes);
    }
    else {
      residual_tensor     = tensor_t();
      residual_tensor_ref = tensor_t();
    }
  }

  norm_type_t norm_type;
  uint64_t batch;
  uint64_t norm_size;
  uint64_t num_channels;
  std::vector<uint64_t> shape;
  float epsilon;
  bool use_scale, use_shift;
  data_type_t gamma_dt, beta_dt;
  uint32_t num_threads;

  uint64_t total_elements;
  uint64_t gamma_size;

  tensor_t gamma_tensor, beta_tensor;
  tensor_t running_mean_tensor, running_var_tensor;
  tensor_t residual_tensor, residual_tensor_ref;

  tensor_factory_t tensor_factory{};
};

/** @fn TEST_P
 *  @param TestNormalization parameterized test class
 *  @param F32_F32 src=f32, dst=f32
 *  @brief Validates normalization native kernel against reference kernel (f32 I/O)
 */
TEST_P(TestNormalization, F32_F32) {
  data_type_t src_dt = data_type_t::f32;
  data_type_t dst_dt = data_type_t::f32;

  auto input_tensor      = tensor_factory.uniform_dist_tensor(shape, src_dt,
                           2.0f);
  auto output_tensor     = tensor_factory.zero_tensor(shape, dst_dt);
  auto output_tensor_ref = tensor_factory.zero_tensor(shape, dst_dt);

  create_optional_tensors(src_dt);

  norm_params np     = build_params(src_dt, dst_dt);
  norm_params ref_np = build_params(src_dt, dst_dt);

  status_t status = normalization_kernel_test(
                      input_tensor, output_tensor, gamma_tensor, beta_tensor,
                      running_mean_tensor, running_var_tensor, residual_tensor, np);
  log_info("F32_F32 native kernel status: ",
           (status == status_t::success) ? "success" : "failure");

  status_t ref_status = normalization_forced_ref_kernel_test(
                          input_tensor, output_tensor_ref, gamma_tensor, beta_tensor,
                          running_mean_tensor, running_var_tensor, residual_tensor_ref, ref_np);
  log_info("F32_F32 reference kernel status: ",
           (ref_status == status_t::success) ? "success" : "failure");

  bool is_test_successful =
    (status == status_t::success && ref_status == status_t::success);

  if (is_test_successful) {
    compare_norm_tensors(output_tensor, output_tensor_ref,
                         shape, total_elements,
                         NORM_F32_TOL, is_test_successful);
  }

  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestNormalization parameterized test class
 *  @param BF16_BF16 src=bf16, dst=bf16
 *  @brief Validates normalization native kernel against reference kernel (bf16 I/O)
 */
TEST_P(TestNormalization, BF16_BF16) {
  data_type_t src_dt = data_type_t::bf16;
  data_type_t dst_dt = data_type_t::bf16;

  auto input_tensor      = tensor_factory.uniform_dist_tensor(shape, src_dt,
                           2.0f);
  auto output_tensor     = tensor_factory.zero_tensor(shape, dst_dt);
  auto output_tensor_ref = tensor_factory.zero_tensor(shape, dst_dt);

  create_optional_tensors(src_dt);

  norm_params np     = build_params(src_dt, dst_dt);
  norm_params ref_np = build_params(src_dt, dst_dt);

  status_t status = normalization_kernel_test(
                      input_tensor, output_tensor, gamma_tensor, beta_tensor,
                      running_mean_tensor, running_var_tensor, residual_tensor, np);
  log_info("BF16_BF16 native kernel status: ",
           (status == status_t::success) ? "success" : "failure");

  status_t ref_status = normalization_forced_ref_kernel_test(
                          input_tensor, output_tensor_ref, gamma_tensor, beta_tensor,
                          running_mean_tensor, running_var_tensor, residual_tensor_ref, ref_np);
  log_info("BF16_BF16 reference kernel status: ",
           (ref_status == status_t::success) ? "success" : "failure");

  bool is_test_successful =
    (status == status_t::success && ref_status == status_t::success);

  if (is_test_successful) {
    compare_norm_tensors(output_tensor, output_tensor_ref,
                         shape, total_elements,
                         NORM_BF16_TOL, is_test_successful);
  }

  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestNormalization parameterized test class
 *  @param BF16_F32 src=bf16, dst=f32
 *  @brief Validates normalization native kernel against reference kernel (bf16 input, f32 output)
 */
TEST_P(TestNormalization, BF16_F32) {
  data_type_t src_dt = data_type_t::bf16;
  data_type_t dst_dt = data_type_t::f32;

  auto input_tensor      = tensor_factory.uniform_dist_tensor(shape, src_dt,
                           2.0f);
  auto output_tensor     = tensor_factory.zero_tensor(shape, dst_dt);
  auto output_tensor_ref = tensor_factory.zero_tensor(shape, dst_dt);

  create_optional_tensors(src_dt);

  norm_params np     = build_params(src_dt, dst_dt);
  norm_params ref_np = build_params(src_dt, dst_dt);

  status_t status = normalization_kernel_test(
                      input_tensor, output_tensor, gamma_tensor, beta_tensor,
                      running_mean_tensor, running_var_tensor, residual_tensor, np);
  log_info("BF16_F32 native kernel status: ",
           (status == status_t::success) ? "success" : "failure");

  status_t ref_status = normalization_forced_ref_kernel_test(
                          input_tensor, output_tensor_ref, gamma_tensor, beta_tensor,
                          running_mean_tensor, running_var_tensor, residual_tensor_ref, ref_np);
  log_info("BF16_F32 reference kernel status: ",
           (ref_status == status_t::success) ? "success" : "failure");

  bool is_test_successful =
    (status == status_t::success && ref_status == status_t::success);

  if (is_test_successful) {
    compare_norm_tensors(output_tensor, output_tensor_ref,
                         shape, total_elements,
                         NORM_F32_TOL, is_test_successful);
  }

  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestNormalization parameterized test class
 *  @param F32_BF16 src=f32, dst=bf16
 *  @brief Validates normalization native kernel against reference kernel (f32 input, bf16 output)
 */
TEST_P(TestNormalization, F32_BF16) {
  data_type_t src_dt = data_type_t::f32;
  data_type_t dst_dt = data_type_t::bf16;

  auto input_tensor      = tensor_factory.uniform_dist_tensor(shape, src_dt,
                           2.0f);
  auto output_tensor     = tensor_factory.zero_tensor(shape, dst_dt);
  auto output_tensor_ref = tensor_factory.zero_tensor(shape, dst_dt);

  create_optional_tensors(src_dt);

  norm_params np     = build_params(src_dt, dst_dt);
  norm_params ref_np = build_params(src_dt, dst_dt);

  status_t status = normalization_kernel_test(
                      input_tensor, output_tensor, gamma_tensor, beta_tensor,
                      running_mean_tensor, running_var_tensor, residual_tensor, np);
  log_info("F32_BF16 native kernel status: ",
           (status == status_t::success) ? "success" : "failure");

  status_t ref_status = normalization_forced_ref_kernel_test(
                          input_tensor, output_tensor_ref, gamma_tensor, beta_tensor,
                          running_mean_tensor, running_var_tensor, residual_tensor_ref, ref_np);
  log_info("F32_BF16 reference kernel status: ",
           (ref_status == status_t::success) ? "success" : "failure");

  bool is_test_successful =
    (status == status_t::success && ref_status == status_t::success);

  if (is_test_successful) {
    compare_norm_tensors(output_tensor, output_tensor_ref,
                         shape, total_elements,
                         NORM_BF16_TOL, is_test_successful);
  }

  EXPECT_TRUE(is_test_successful);
}

/** @fn INSTANTIATE_TEST_SUITE_P
 *  @brief Triggers Normalization parameterized test suite
 */
INSTANTIATE_TEST_SUITE_P(Normalization, TestNormalization,
                         ::testing::ValuesIn(normalization_test));
