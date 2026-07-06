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
      size_t elem_bytes = size_of(src_dt);
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

  norm_params np = build_params(src_dt, dst_dt);

  status_t status = normalization_kernel_test(
                      input_tensor, output_tensor, gamma_tensor, beta_tensor,
                      running_mean_tensor, running_var_tensor, residual_tensor, np);

  status_t ref_status = normalization_forced_ref_kernel_test(
                          input_tensor, output_tensor_ref, gamma_tensor, beta_tensor,
                          running_mean_tensor, running_var_tensor, residual_tensor_ref, np);

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

  norm_params np = build_params(src_dt, dst_dt);

  status_t status = normalization_kernel_test(
                      input_tensor, output_tensor, gamma_tensor, beta_tensor,
                      running_mean_tensor, running_var_tensor, residual_tensor, np);

  status_t ref_status = normalization_forced_ref_kernel_test(
                          input_tensor, output_tensor_ref, gamma_tensor, beta_tensor,
                          running_mean_tensor, running_var_tensor, residual_tensor_ref, np);

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

  norm_params np = build_params(src_dt, dst_dt);

  status_t status = normalization_kernel_test(
                      input_tensor, output_tensor, gamma_tensor, beta_tensor,
                      running_mean_tensor, running_var_tensor, residual_tensor, np);

  status_t ref_status = normalization_forced_ref_kernel_test(
                          input_tensor, output_tensor_ref, gamma_tensor, beta_tensor,
                          running_mean_tensor, running_var_tensor, residual_tensor_ref, np);

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

  norm_params np = build_params(src_dt, dst_dt);

  status_t status = normalization_kernel_test(
                      input_tensor, output_tensor, gamma_tensor, beta_tensor,
                      running_mean_tensor, running_var_tensor, residual_tensor, np);

  status_t ref_status = normalization_forced_ref_kernel_test(
                          input_tensor, output_tensor_ref, gamma_tensor, beta_tensor,
                          running_mean_tensor, running_var_tensor, residual_tensor_ref, np);

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
 *  @param F16_F16 src=f16, dst=f16
 *  @brief Validates normalization native kernel against reference kernel (f16 I/O).
 *         Skipped at runtime if the platform lacks AVX512-FP16 ISA support.
 */
TEST_P(TestNormalization, F16_F16) {
  data_type_t src_dt = data_type_t::f16;
  data_type_t dst_dt = data_type_t::f16;

  auto input_tensor      = tensor_factory.uniform_dist_tensor(shape, src_dt,
                           2.0f);
  auto output_tensor     = tensor_factory.zero_tensor(shape, dst_dt);
  auto output_tensor_ref = tensor_factory.zero_tensor(shape, dst_dt);

  create_optional_tensors(src_dt);

  norm_params np = build_params(src_dt, dst_dt);

  status_t status = normalization_kernel_test(
                      input_tensor, output_tensor, gamma_tensor, beta_tensor,
                      running_mean_tensor, running_var_tensor, residual_tensor, np);
  if (status == status_t::isa_unsupported) {
    GTEST_SKIP() << "F16 not supported: requires F16-capable ISA";
  }

  status_t ref_status = normalization_forced_ref_kernel_test(
                          input_tensor, output_tensor_ref, gamma_tensor, beta_tensor,
                          running_mean_tensor, running_var_tensor, residual_tensor_ref, np);

  bool is_test_successful =
    (status == status_t::success && ref_status == status_t::success);

  if (is_test_successful) {
    compare_norm_tensors(output_tensor, output_tensor_ref,
                         shape, total_elements,
                         NORM_F16_TOL, is_test_successful);
  }

  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestNormalization parameterized test class
 *  @param F16_F32 src=f16, dst=f32
 *  @brief Validates normalization native kernel against reference kernel
 *         (f16 input, f32 output). Skipped at runtime if F16 ISA is unavailable.
 */
TEST_P(TestNormalization, F16_F32) {
  data_type_t src_dt = data_type_t::f16;
  data_type_t dst_dt = data_type_t::f32;

  auto input_tensor      = tensor_factory.uniform_dist_tensor(shape, src_dt,
                           2.0f);
  auto output_tensor     = tensor_factory.zero_tensor(shape, dst_dt);
  auto output_tensor_ref = tensor_factory.zero_tensor(shape, dst_dt);

  create_optional_tensors(src_dt);

  norm_params np = build_params(src_dt, dst_dt);

  status_t status = normalization_kernel_test(
                      input_tensor, output_tensor, gamma_tensor, beta_tensor,
                      running_mean_tensor, running_var_tensor, residual_tensor, np);
  if (status == status_t::isa_unsupported) {
    GTEST_SKIP() << "F16 not supported: requires F16-capable ISA";
  }

  status_t ref_status = normalization_forced_ref_kernel_test(
                          input_tensor, output_tensor_ref, gamma_tensor, beta_tensor,
                          running_mean_tensor, running_var_tensor, residual_tensor_ref, np);

  bool is_test_successful =
    (status == status_t::success && ref_status == status_t::success);

  if (is_test_successful) {
    compare_norm_tensors(output_tensor, output_tensor_ref,
                         shape, total_elements,
                         NORM_F16_TOL, is_test_successful);
  }

  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestNormalization parameterized test class
 *  @param F32_F16 src=f32, dst=f16
 *  @brief Validates normalization native kernel against reference kernel
 *         (f32 input, f16 output). Skipped at runtime if F16 ISA is unavailable.
 */
TEST_P(TestNormalization, F32_F16) {
  data_type_t src_dt = data_type_t::f32;
  data_type_t dst_dt = data_type_t::f16;

  auto input_tensor      = tensor_factory.uniform_dist_tensor(shape, src_dt,
                           2.0f);
  auto output_tensor     = tensor_factory.zero_tensor(shape, dst_dt);
  auto output_tensor_ref = tensor_factory.zero_tensor(shape, dst_dt);

  create_optional_tensors(src_dt);

  norm_params np = build_params(src_dt, dst_dt);

  status_t status = normalization_kernel_test(
                      input_tensor, output_tensor, gamma_tensor, beta_tensor,
                      running_mean_tensor, running_var_tensor, residual_tensor, np);
  if (status == status_t::isa_unsupported) {
    GTEST_SKIP() << "F16 not supported: requires F16-capable ISA";
  }

  status_t ref_status = normalization_forced_ref_kernel_test(
                          input_tensor, output_tensor_ref, gamma_tensor, beta_tensor,
                          running_mean_tensor, running_var_tensor, residual_tensor_ref, np);

  bool is_test_successful =
    (status == status_t::success && ref_status == status_t::success);

  if (is_test_successful) {
    compare_norm_tensors(output_tensor, output_tensor_ref,
                         shape, total_elements,
                         NORM_F16_TOL, is_test_successful);
  }

  EXPECT_TRUE(is_test_successful);
}

/** @fn INSTANTIATE_TEST_SUITE_P
 *  @brief Triggers Normalization parameterized test suite
 */
INSTANTIATE_TEST_SUITE_P(Normalization, TestNormalization,
                         ::testing::ValuesIn(normalization_test));

/** @brief Deterministic norm_size values that are NOT multiples of 32.
 *
 *  These force the masked AVX512-FP16 tail path in the F16 normalization
 *  kernels (f16x32_load_tail_typed / f16x32_store_tail_typed, which route
 *  through the f16_maskz_loadu_vec / f16_mask_storeu_vec shims). The set
 *  covers tail-only (1, 31), single-block + tail (33, 47, 63), and
 *  multi-block + tail (95, 129) cases so every loop remainder branch is
 *  exercised. The random `normalization_test` instantiation above only hits
 *  these sizes by chance; this instantiation makes the coverage guaranteed.
 */
static std::vector<NormalizationType> make_norm_tail_cases() {
  const std::vector<uint64_t> tail_sizes = {1, 31, 33, 47, 63, 95, 129};
  // FusedAddRMSNorm is intentionally NOT in this shared suite: the native FP16
  // fused-add kernel needs a strict src_dt == dst_dt == gamma_dt == f16 gate,
  // but this suite pins gamma_dt=f32 (so the F32_F32 / BF16_* fixture variants,
  // which don't skip on missing F16 ISA, stay valid on non-F16 hosts). The
  // dedicated NormalizationFusedTailF16 test below covers the fused-add masked
  // tail with all-f16 dtypes when built with -DZENDNNL_FUSED_ADD_RMS_F16=ON.
  const std::vector<norm_type_t> norm_types = {
    norm_type_t::RMS_NORM,
    norm_type_t::LAYER_NORM};

  std::vector<NormalizationType> cases;
  cases.reserve(norm_types.size() * tail_sizes.size());
  for (norm_type_t nt : norm_types) {
    for (uint64_t ns : tail_sizes) {
      // Default-construct then fully override every field, so the result is
      // deterministic regardless of the constructor's internal randomization.
      NormalizationType c;
      c.norm_type    = nt;
      c.batch        = 4;            // multiple rows over the tail dimension
      c.norm_size    = ns;
      c.num_channels = 0;            // unused for non-batch-norm
      c.shape        = {c.batch, ns};
      c.epsilon      = (nt == norm_type_t::RMS_NORM) ? 1e-6f : 1e-5f;
      c.use_scale    = true;
      c.use_shift    = (nt == norm_type_t::LAYER_NORM);
      // Keep gamma/beta f32: the F32_F32/BF16 fixture variants don't skip on
      // missing F16 ISA, so f16 gamma/beta there would fail on non-F16 hosts.
      // The masked shim is still exercised via the f16 src/dst variants.
      c.gamma_dt     = data_type_t::f32;
      c.beta_dt      = data_type_t::f32;
      c.num_threads  = 2;
      cases.push_back(c);
    }
  }
  return cases;
}

/** @fn INSTANTIATE_TEST_SUITE_P
 *  @brief Guaranteed masked-tail coverage for the F16 normalization kernels.
 *         Reuses the same fixture (and its F16 isa_unsupported GTEST_SKIP),
 *         so on non-F16 hosts the F16_* variants skip and the F32/BF16
 *         variants still validate the tail handling for those dtypes.
 */
INSTANTIATE_TEST_SUITE_P(NormalizationTail, TestNormalization,
                         ::testing::ValuesIn(make_norm_tail_cases()));

#if defined(ZENDNNL_FUSED_ADD_RMS_F16)
/** @brief Dedicated masked-tail coverage for the native FP16 FusedAddRMSNorm
 *         kernel (only compiled with -DZENDNNL_FUSED_ADD_RMS_F16=ON).
 *
 *  The shared TestNormalization fixture cannot host this case: the native
 *  fused-add FP16 path requires a strict src_dt == dst_dt == gamma_dt == f16
 *  gate, but a gamma_dt=f16 parameter would also fan out to the fixture's
 *  F32_F32 / BF16_* variants, which do not skip on missing F16 ISA and would
 *  therefore fail on non-F16 hosts. This standalone test pins all-f16 dtypes,
 *  loops the same non-multiple-of-32 tail sizes, and skips at runtime when the
 *  host lacks AVX512-FP16 — so it actually exercises the f16x32_load_mask_typed
 *  / f16x32_store_mask_typed shims that only the fused-add kernel uses.
 */
TEST(NormalizationFusedTailF16, MaskedTail) {
  const std::vector<uint64_t> tail_sizes = {1, 31, 33, 47, 63, 95, 129};
  const data_type_t dt   = data_type_t::f16;
  const uint64_t    batch = 4;   // multiple rows over the tail dimension
  tensor_factory_t  tensor_factory{};

  for (uint64_t ns : tail_sizes) {
    const std::vector<uint64_t> shape = {batch, ns};
    const uint64_t total_elements = batch * ns;

    auto input_tensor      = tensor_factory.uniform_dist_tensor(shape, dt, 2.0f);
    auto output_tensor     = tensor_factory.zero_tensor(shape, dt);
    auto output_tensor_ref = tensor_factory.zero_tensor(shape, dt);
    auto gamma_tensor      = tensor_factory.uniform_dist_tensor({ns}, dt, 1.0f);
    tensor_t beta_tensor;                       // RMS family ignores beta
    tensor_t running_mean_tensor, running_var_tensor;

    // Residual is read-modify-written in place by the native kernel; the
    // reference gets an untouched copy of the same initial data.
    auto residual_tensor     = tensor_factory.uniform_dist_tensor(shape, dt, 2.0f);
    auto residual_tensor_ref = tensor_factory.zero_tensor(shape, dt);
    const size_t elem_bytes  = size_of(dt);
    std::memcpy(residual_tensor_ref.get_raw_handle_unsafe(),
                residual_tensor.get_raw_handle_unsafe(),
                total_elements * elem_bytes);

    norm_params np;
    np.norm_type    = norm_type_t::FUSED_ADD_RMS_NORM;
    np.batch        = batch;
    np.norm_size    = ns;
    np.num_channels = 0;
    np.epsilon      = 1e-6f;
    np.use_scale    = true;
    np.use_shift    = false;
    np.src_dt       = dt;
    np.dst_dt       = dt;
    np.gamma_dt     = dt;
    np.beta_dt      = dt;
    np.num_threads  = 2;
    omp_set_num_threads(np.num_threads);

    status_t status = normalization_kernel_test(
                        input_tensor, output_tensor, gamma_tensor, beta_tensor,
                        running_mean_tensor, running_var_tensor, residual_tensor, np);
    if (status == status_t::isa_unsupported) {
      GTEST_SKIP() << "F16 not supported: requires F16-capable ISA";
    }

    // Same np (with the accum_type the native dispatch just recorded) so the
    // reference bit-matches the native FP16 accumulation.
    status_t ref_status = normalization_forced_ref_kernel_test(
                            input_tensor, output_tensor_ref, gamma_tensor, beta_tensor,
                            running_mean_tensor, running_var_tensor, residual_tensor_ref, np);

    bool ok = (status == status_t::success && ref_status == status_t::success);
    if (ok) {
      compare_norm_tensors(output_tensor, output_tensor_ref,
                           shape, total_elements, NORM_F16_TOL, ok);
    }
    EXPECT_TRUE(ok) << "FusedAddRMSNorm F16 masked-tail failed for norm_size="
                    << ns;
  }
}
#endif // ZENDNNL_FUSED_ADD_RMS_F16
