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
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>
#include "gtest_utils.hpp"

namespace {

// -------------------------------------------------------------------------
// Q/K/V/output physical-layout order strings.
//
// All SDPA tests use logical 4D shape [B, H, S, D] (positions a=0, b=1,
// c=2, d=3); the @c set_order() string controls the *physical* layout by
// re-ordering which logical dim is innermost / next-innermost / ... The
// reference operator and the LOWOHA flash backend both read each tensor's
// strides directly, so any per-tensor layout listed here works.
//
//   "abcd" -> BHSD canonical contiguous, strides = [H*S*D, S*D,   D, 1]
//   "acbd" -> BSHD physical layout,      strides = [S*H*D,   D, H*D, 1]
//             (a logical-BHSD view of BSHD memory; equivalent to
//              torch.empty(B, S, H, D).transpose(1, 2) in PyTorch)
// -------------------------------------------------------------------------
constexpr const char *kBhsdOrder = "abcd";
constexpr const char *kBshdOrder = "acbd";

/**
 * @brief Create a 4D tensor with logical shape @p sizes and physical
 *        layout described by @p order, then invoke @p fill on the storage.
 *
 * The standard tensor_factory_t exposes a @c trans flag that only swaps
 * the inner two dims; we need an arbitrary order string to construct BSHD
 * tensors (order "acbd" for a [B, H, S, D] logical shape), so this helper
 * builds the tensor by hand. @p fill receives the created tensor and is
 * expected to populate the storage via @c get_raw_handle_unsafe().
 */
template <typename FillFn>
tensor_t make_4d_tensor_with_order(const std::vector<uint64_t> &sizes,
                                   data_type_t dtype,
                                   const std::string &order,
                                   FillFn fill) {
  tensor_t t = tensor_t()
               .set_name("sdpa test tensor")
               .set_size(sizes)
               .set_data_type(dtype)
               .set_order(order);
  t.set_storage();
  t.create();
  if (!t.check()) {
    log_warning("Failed to create test tensor with order '", order, "'");
    return t;
  }
  fill(t);
  return t;
}

/**
 * @brief Uniform-random 4D tensor with the requested physical layout.
 *
 * Matches @c tensor_factory_t::uniform_dist_tensor exactly when @p order
 * is "abcd" (BHSD); for other orders we fill the underlying storage with
 * the same RNG so the comparison-of-backends pattern keeps working.
 */
tensor_t make_uniform_tensor(tensor_factory_t &tf,
                             const std::vector<uint64_t> &sizes,
                             data_type_t dtype, float val,
                             const std::string &order) {
  if (order == kBhsdOrder) {
    return tf.uniform_dist_tensor(sizes, dtype, val);
  }
  return make_4d_tensor_with_order(sizes, dtype, order, [=](tensor_t &t) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0 * val, 1.0 * val);
    auto buf_nelem = t.get_nelem();
    void *raw      = t.get_raw_handle_unsafe();
    if (dtype == data_type_t::f32) {
      float *p = static_cast<float *>(raw);
      std::generate(p, p + buf_nelem, [&]() {
        return dist(gen);
      });
    }
    else if (dtype == data_type_t::bf16) {
      bfloat16_t *p = static_cast<bfloat16_t *>(raw);
      std::generate(p, p + buf_nelem,
      [&]() {
        return bfloat16_t(dist(gen));
      });
    }
  });
}

/** @brief Zeroed 4D tensor with the requested physical layout. */
tensor_t make_zero_tensor(tensor_factory_t &tf,
                          const std::vector<uint64_t> &sizes,
                          data_type_t dtype,
                          const std::string &order) {
  if (order == kBhsdOrder) {
    return tf.zero_tensor(sizes, dtype);
  }
  return make_4d_tensor_with_order(sizes, dtype, order, [](tensor_t &t) {
    std::memset(t.get_raw_handle_unsafe(), 0, t.get_buffer_sz_bytes());
  });
}

/**
 * @brief Supported additive-mask shape variants.
 *
 * The reference operator and LOWOHA flash backend accept four mask shapes
 * (see sdpa_encoder_operator_impl.cpp::validate). The fixture randomly
 * picks one of these per (seed, params) instance and the
 * @c F32_F32_MASK_LAYOUT / @c BF16_BF16_MASK_LAYOUT tests materialise it
 * via @c resolve_mask_shape; this collapses the per-shape and
 * per-(BHSD/BSHD) test variants into a single test body without losing
 * coverage across the parameterised sweep.
 */
enum class mask_shape_kind_t {
  full_4d,      /*!< [B, H, S_q, S_kv]   per-(b, h) mask              */
  head_bcast,   /*!< [B, 1, S_q, S_kv]   broadcast across heads       */
  batch_bcast,  /*!< [1, H, S_q, S_kv]   broadcast across batch       */
  two_d         /*!< [S_q, S_kv]         broadcast across batch+heads */
};

/** @brief Materialise a @c mask_shape_kind_t into the corresponding
 *         tensor shape vector for the given Q/K/V dims. */
std::vector<uint64_t> resolve_mask_shape(mask_shape_kind_t kind,
    uint64_t batch, uint64_t num_heads,
    uint64_t seq_len_q,
    uint64_t seq_len_kv) {
  switch (kind) {
  case mask_shape_kind_t::full_4d:
    return {batch, num_heads, seq_len_q, seq_len_kv};
  case mask_shape_kind_t::head_bcast:
    return {batch,      1UL,  seq_len_q, seq_len_kv};
  case mask_shape_kind_t::batch_bcast:
    return {1UL,   num_heads, seq_len_q, seq_len_kv};
  case mask_shape_kind_t::two_d:
    return {seq_len_q, seq_len_kv};
  }
  return {};  // unreachable; silences -Wreturn-type on some compilers
}

/**
 * @brief Run an SDPA mask-layout correctness test (LOWOHA vs operator ref).
 *
 * Allocates Q [B, H, S_q, D], K/V [B, H, S_kv, D], output [B, H, S_q, D] of
 * @p qkv_dtype with the requested physical layout (@p qkv_order, applies
 * to all four tensors), an additive mask of shape @p mask_shape and dtype
 * @p mask_dtype, then runs both backends and compares within the dtype's
 * tolerance band. Supports cross-attention via independent @p seq_len_q /
 * @p seq_len_kv.
 *
 * @param tensor_factory Reused factory from the test fixture.
 * @param batch          Q batch size.
 * @param num_heads      Q num_heads.
 * @param seq_len_q      Q sequence length (S_q).
 * @param seq_len_kv     K/V sequence length (S_kv); == seq_len_q for self-attn.
 * @param head_dim       Per-head feature dim (must match across Q/K/V).
 * @param scale          Scaling factor applied to QK^T.
 * @param is_causal      If true, apply causal mask before the explicit mask.
 * @param qkv_dtype      data_type_t::f32 or data_type_t::bf16.
 * @param mask_shape     Mask tensor shape — 2D [S_q, S_kv] or
 *                       4D [B|1, H|1, S_q, S_kv].
 * @param qkv_order      Physical-layout order string for Q/K/V/output
 *                       (kBhsdOrder or kBshdOrder).
 * @param rtol           Relative tolerance for output comparison.
 * @param epsilon        Per-op numerical epsilon for output comparison.
 * @param mask_dtype     Mask tensor dtype. F32 callers must pass
 *                       @c data_type_t::f32 (the operator's validate()
 *                       rejects bf16 mask when QKV is f32). BF16 callers
 *                       typically pass the fixture's randomised
 *                       @c mask_dt member to exercise both supported
 *                       QKV-bf16 mask paths.
 */
void run_sdpa_mask_layout_test(tensor_factory_t &tensor_factory,
                               uint64_t batch, uint64_t num_heads,
                               uint64_t seq_len_q, uint64_t seq_len_kv,
                               uint64_t head_dim,
                               float scale, bool is_causal,
                               data_type_t qkv_dtype,
                               const std::vector<uint64_t> &mask_shape,
                               const std::string &qkv_order,
                               float rtol, float epsilon,
                               data_type_t mask_dtype = data_type_t::f32) {
  auto query_tensor       = make_uniform_tensor(tensor_factory,
  {batch, num_heads, seq_len_q, head_dim},
  qkv_dtype, 1.0, qkv_order);
  auto key_tensor         = make_uniform_tensor(tensor_factory,
  {batch, num_heads, seq_len_kv, head_dim},
  qkv_dtype, 1.0, qkv_order);
  auto value_tensor       = make_uniform_tensor(tensor_factory,
  {batch, num_heads, seq_len_kv, head_dim},
  qkv_dtype, 1.0, qkv_order);
  // The mask validator only accepts canonical row-major contiguous strides
  // (BHSD-style for 4D, [S_q, S_kv] row-major for 2D); the mask dtype
  // itself comes from the caller (FP32 for f32-QKV tests, the fixture's
  // randomised mask_dt for bf16-QKV tests).
  log_info("SDPA mask-layout test: qkv_dtype=", static_cast<int>(qkv_dtype),
           " mask_dtype=", static_cast<int>(mask_dtype));
  auto mask_tensor        = tensor_factory.uniform_dist_tensor(
                              mask_shape, mask_dtype, 0.5);

  auto output_tensor      = make_zero_tensor(tensor_factory,
  {batch, num_heads, seq_len_q, head_dim},
  qkv_dtype, qkv_order);
  auto output_tensor_ref  = make_zero_tensor(tensor_factory,
  {batch, num_heads, seq_len_q, head_dim},
  qkv_dtype, qkv_order);

  status_t status         = sdpa_kernel_test(query_tensor, key_tensor,
                            value_tensor, mask_tensor, output_tensor,
                            scale, is_causal, /*has_mask=*/true);
  status_t ref_status     = sdpa_forced_ref_kernel_test(query_tensor,
                            key_tensor, value_tensor, mask_tensor,
                            output_tensor_ref, scale, is_causal,
                            /*has_mask=*/true);

  bool is_test_successful =
    (status == status_t::success && ref_status == status_t::success);

  if (is_test_successful) {
    compare_tensor_4D_sdpa(output_tensor, output_tensor_ref,
                           batch, num_heads, seq_len_q, seq_len_kv,
                           head_dim, rtol, epsilon, is_test_successful);
  }

  EXPECT_TRUE(is_test_successful);
}

// NOTE: A previous helper run_sdpa_qkv_layout_test() was used to dispatch
// per-test BHSD vs BSHD coverage of the eight inline tests. With the
// fixture's @c qkv_order randomised in SetUp the inline tests now cover
// both layouts directly, so the helper has been removed as dead code.

}  // namespace

/** @brief TestSdpa is a test class to handle SDPA parameters */
class TestSdpa : public ::testing::TestWithParam<SdpaType> {
 protected:
  /** @brief SetUp initializes test parameters from the parameterized fixture
   *
   *  Standard googletest fixture entry point: pulls a SdpaType from the
   *  parameter generator, seeds the RNG so per-test data generation is
   *  reproducible, pins the OpenMP thread count, and picks a random
   *  collection of axes that are independent of the @c SdpaType params:
   *
   *    - @c mask_dt         : f32 or bf16 (BF16 SDPA tests only).
   *    - @c qkv_order       : kBhsdOrder or kBshdOrder (Q/K/V/output
   *                           physical layout). Both backends consume
   *                           per-tensor strides via get_stride() so any
   *                           supported order works for any test body.
   *    - @c mask_shape_kind : full_4d / head_bcast / batch_bcast / two_d
   *                           (mask broadcast shape). Used by the
   *                           *_MASK_LAYOUT tests to collapse the four
   *                           per-shape variants into one body.
   *
   *  Each axis uses a dedicated @c std::mt19937_64 (NOT the global
   *  @c rand() state) seeded from an FNV-1a-style hash of the SdpaType
   *  fields mixed with the global @c seed, so the choice is reproducible
   *  AND varies across parameterised TestSdpa instances rather than
   *  collapsing to a single value when @c srand reseeds with the same
   *  global @c seed in every SetUp call. The per-axis seed offsets keep
   *  the four axes statistically independent.
   *
   *  F32 SDPA tests ignore @c mask_dt and pass @c data_type_t::f32
   *  directly (the operator's validate() rejects bf16 mask when QKV is
   *  f32).
   */
  virtual void SetUp() {
    SdpaType params = GetParam();
    srand(static_cast<unsigned int>(seed));
    batch       = params.batch;
    num_heads   = params.num_heads;
    seq_len     = params.seq_len;
    kv_seq_len  = params.kv_seq_len;
    head_dim    = params.head_dim;
    scale       = params.scale;
    is_causal   = params.is_causal;
    has_mask    = params.has_mask;
    num_threads = params.num_threads;
    omp_set_num_threads(num_threads);

    // Per-instance variant seed (FNV-1a-style mixing of the SdpaType
    // fields). Combined with the global test seed and per-axis offsets
    // below to give independent, reproducible random selections.
    uint64_t variant = 1469598103934665603ULL;  // FNV-1a basis
    auto mix = [&variant](uint64_t v) {
      variant ^= v;
      variant *= 1099511628211ULL;
    };
    mix(batch);
    mix(num_heads);
    mix(seq_len);
    mix(kv_seq_len);
    mix(head_dim);
    mix(is_causal ? 1ULL : 0ULL);
    mix(has_mask  ? 1ULL : 0ULL);
    const uint64_t base_seed = static_cast<uint64_t>(seed) ^ variant;

    // Per-axis offsets keep the three random selections statistically
    // independent (otherwise a single mt19937_64 stream would correlate
    // them via shared state).
    std::mt19937_64 mask_rng(base_seed ^ 0xA5A5A5A5A5A5A5A5ULL);
    std::mt19937_64 order_rng(base_seed ^ 0x5A5A5A5A5A5A5A5AULL);
    std::mt19937_64 shape_rng(base_seed ^ 0xC3C3C3C3C3C3C3C3ULL);

    mask_dt   = (mask_rng()  & 1ULL) ? data_type_t::bf16
                : data_type_t::f32;
    qkv_order = (order_rng() & 1ULL) ? kBshdOrder : kBhsdOrder;
    mask_shape_kind = static_cast<mask_shape_kind_t>(shape_rng() % 4ULL);

    log_info("batch: ", batch, " num_heads: ", num_heads,
             " seq_len: ", seq_len, " kv_seq_len: ", kv_seq_len,
             " head_dim: ", head_dim,
             " scale: ", scale,
             " is_causal: ", is_causal, " has_mask: ", has_mask,
             " num_threads: ", num_threads,
             " mask_dt: ", static_cast<int>(mask_dt),
             " qkv_order: ", qkv_order,
             " mask_shape_kind: ", static_cast<int>(mask_shape_kind));
  }

  /** @brief TearDown is used to free resources used in test */
  virtual void TearDown() {}

  uint64_t batch, num_heads, seq_len, kv_seq_len, head_dim;
  float scale;
  bool is_causal, has_mask;
  int32_t num_threads;
  /**
   * @brief Mask data type for BF16 SDPA tests (f32 or bf16, randomised
   *        in @c SetUp). F32 tests use @c data_type_t::f32 directly.
   */
  data_type_t mask_dt;
  /**
   * @brief Q/K/V/output physical layout order string (@c kBhsdOrder or
   *        @c kBshdOrder), randomised in @c SetUp.
   */
  std::string qkv_order;
  /**
   * @brief Mask shape variant (full 4D, head/batch broadcast, or 2D),
   *        randomised in @c SetUp. Used by the *_MASK_LAYOUT tests.
   */
  mask_shape_kind_t mask_shape_kind;
  tensor_factory_t tensor_factory{};
};

/** @fn TEST_P
 *  @param TestSdpa parameterized test class to initialize SDPA parameters
 *  @param F32_F32 user-defined name of test
 *  @brief Test to validate SDPA F32 LOWOHA flash kernel against the
 *         operator-based reference (sdpa_encoder_operator_t -> FP32 ref kernel).
 *
 *  Q/K/V/output are 4D tensors with logical shape [batch, num_heads,
 *  seq_len, head_dim]; the physical layout (BHSD or BSHD) is selected
 *  randomly per instance via the fixture's @c qkv_order. Both backends
 *  consume per-tensor strides via get_stride() so any supported order
 *  works. The same input tensors are fed to both the LOWOHA flash backend
 *  (sdpa_direct) and the reference operator path; outputs are compared
 *  element-wise within an SDPA-specific error bound.
 */
TEST_P(TestSdpa, F32_F32) {
  auto query_tensor       = make_uniform_tensor(tensor_factory,
  {batch, num_heads, seq_len, head_dim},
  data_type_t::f32, 1.0, qkv_order);
  auto key_tensor         = make_uniform_tensor(tensor_factory,
  {batch, num_heads, kv_seq_len, head_dim},
  data_type_t::f32, 1.0, qkv_order);
  auto value_tensor       = make_uniform_tensor(tensor_factory,
  {batch, num_heads, kv_seq_len, head_dim},
  data_type_t::f32, 1.0, qkv_order);

  // Mask is additive: 0 means "attend", -inf means "ignore". We populate with
  // small magnitudes so both attend / partially-attend behaviour is exercised
  // without producing NaNs from a fully-masked row.
  //
  // Shape is [1, 1, S_q, S_kv] (broadcast across batch/heads): both LOWOHA
  // flash and the operator-based reference agree on this layout regardless of
  // whether self-attention (S_q == S_kv) or cross-attention (S_q != S_kv) is
  // exercised. The full set of supported mask shapes is covered by the
  // F32_F32_MASK_LAYOUT / BF16_BF16_MASK_LAYOUT tests.
  auto mask_tensor        = has_mask ?
                            tensor_factory.uniform_dist_tensor(
  {1UL, 1UL, seq_len, kv_seq_len},
  data_type_t::f32, 0.5)
    : tensor_t();

  auto output_tensor      = make_zero_tensor(tensor_factory,
  {batch, num_heads, seq_len, head_dim},
  data_type_t::f32, qkv_order);
  auto output_tensor_ref  = make_zero_tensor(tensor_factory,
  {batch, num_heads, seq_len, head_dim},
  data_type_t::f32, qkv_order);

  status_t status         = sdpa_kernel_test(query_tensor, key_tensor,
                            value_tensor, mask_tensor, output_tensor,
                            scale, is_causal, has_mask);
  status_t ref_status     = sdpa_forced_ref_kernel_test(query_tensor,
                            key_tensor, value_tensor, mask_tensor,
                            output_tensor_ref, scale, is_causal, has_mask);

  bool is_test_successful =
    (status == status_t::success && ref_status == status_t::success);

  if (is_test_successful) {
    compare_tensor_4D_sdpa(output_tensor, output_tensor_ref,
                           batch, num_heads, seq_len, kv_seq_len, head_dim,
                           rtol_f32, epsilon_f32, is_test_successful);
  }

  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestSdpa parameterized test class
 *  @param F32_F32_NO_MASK Forces non-causal, non-masked code path
 *  @brief Always-on smoke test for the simplest SDPA variant: pure
 *         softmax(QK^T * scale) V with no causal/explicit masking.
 *         Q/K/V/output physical layout (BHSD or BSHD) is randomised
 *         per instance via the fixture's @c qkv_order.
 */
TEST_P(TestSdpa, F32_F32_NO_MASK) {
  auto query_tensor       = make_uniform_tensor(tensor_factory,
  {batch, num_heads, seq_len, head_dim},
  data_type_t::f32, 1.0, qkv_order);
  auto key_tensor         = make_uniform_tensor(tensor_factory,
  {batch, num_heads, kv_seq_len, head_dim},
  data_type_t::f32, 1.0, qkv_order);
  auto value_tensor       = make_uniform_tensor(tensor_factory,
  {batch, num_heads, kv_seq_len, head_dim},
  data_type_t::f32, 1.0, qkv_order);
  tensor_t mask_tensor;  // empty: has_mask=false

  auto output_tensor      = make_zero_tensor(tensor_factory,
  {batch, num_heads, seq_len, head_dim},
  data_type_t::f32, qkv_order);
  auto output_tensor_ref  = make_zero_tensor(tensor_factory,
  {batch, num_heads, seq_len, head_dim},
  data_type_t::f32, qkv_order);

  status_t status         = sdpa_kernel_test(query_tensor, key_tensor,
                            value_tensor, mask_tensor, output_tensor,
                            scale, /*is_causal=*/false, /*has_mask=*/false);
  status_t ref_status     = sdpa_forced_ref_kernel_test(query_tensor,
                            key_tensor, value_tensor, mask_tensor,
                            output_tensor_ref, scale,
                            /*is_causal=*/false, /*has_mask=*/false);

  bool is_test_successful =
    (status == status_t::success && ref_status == status_t::success);

  if (is_test_successful) {
    compare_tensor_4D_sdpa(output_tensor, output_tensor_ref,
                           batch, num_heads, seq_len, kv_seq_len, head_dim,
                           rtol_f32, epsilon_f32, is_test_successful);
  }

  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestSdpa parameterized test class
 *  @param F32_F32_CAUSAL Forces causal masking
 *  @brief Exercises the causal path: scores[i][j>i] = -inf before softmax,
 *         no explicit mask. Mirrors decoder-style self-attention.
 *         Q/K/V/output physical layout (BHSD or BSHD) is randomised
 *         per instance via the fixture's @c qkv_order.
 */
TEST_P(TestSdpa, F32_F32_CAUSAL) {
  auto query_tensor       = make_uniform_tensor(tensor_factory,
  {batch, num_heads, seq_len, head_dim},
  data_type_t::f32, 1.0, qkv_order);
  auto key_tensor         = make_uniform_tensor(tensor_factory,
  {batch, num_heads, kv_seq_len, head_dim},
  data_type_t::f32, 1.0, qkv_order);
  auto value_tensor       = make_uniform_tensor(tensor_factory,
  {batch, num_heads, kv_seq_len, head_dim},
  data_type_t::f32, 1.0, qkv_order);
  tensor_t mask_tensor;

  auto output_tensor      = make_zero_tensor(tensor_factory,
  {batch, num_heads, seq_len, head_dim},
  data_type_t::f32, qkv_order);
  auto output_tensor_ref  = make_zero_tensor(tensor_factory,
  {batch, num_heads, seq_len, head_dim},
  data_type_t::f32, qkv_order);

  status_t status         = sdpa_kernel_test(query_tensor, key_tensor,
                            value_tensor, mask_tensor, output_tensor,
                            scale, /*is_causal=*/true, /*has_mask=*/false);
  status_t ref_status     = sdpa_forced_ref_kernel_test(query_tensor,
                            key_tensor, value_tensor, mask_tensor,
                            output_tensor_ref, scale,
                            /*is_causal=*/true, /*has_mask=*/false);

  bool is_test_successful =
    (status == status_t::success && ref_status == status_t::success);

  if (is_test_successful) {
    compare_tensor_4D_sdpa(output_tensor, output_tensor_ref,
                           batch, num_heads, seq_len, kv_seq_len, head_dim,
                           rtol_f32, epsilon_f32, is_test_successful);
  }

  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestSdpa parameterized test class
 *  @param F32_F32_MASK Forces explicit attention mask
 *  @brief Exercises the explicit-mask code path with a small additive mask.
 *         Mask is intentionally bounded so no row degenerates to all -inf.
 *         Q/K/V/output physical layout (BHSD or BSHD) is randomised
 *         per instance via the fixture's @c qkv_order.
 */
TEST_P(TestSdpa, F32_F32_MASK) {
  auto query_tensor       = make_uniform_tensor(tensor_factory,
  {batch, num_heads, seq_len, head_dim},
  data_type_t::f32, 1.0, qkv_order);
  auto key_tensor         = make_uniform_tensor(tensor_factory,
  {batch, num_heads, kv_seq_len, head_dim},
  data_type_t::f32, 1.0, qkv_order);
  auto value_tensor       = make_uniform_tensor(tensor_factory,
  {batch, num_heads, kv_seq_len, head_dim},
  data_type_t::f32, 1.0, qkv_order);
  // Mask: small additive values (typically 0 or small negatives).
  // Avoid -inf in randomized test to prevent fully-masked rows -> NaN softmax.
  // Use [1, 1, S_q, S_kv] broadcast layout — see F32_F32 test for rationale.
  auto mask_tensor        = tensor_factory.uniform_dist_tensor(
  {1UL, 1UL, seq_len, kv_seq_len},
  data_type_t::f32, 0.5);

  auto output_tensor      = make_zero_tensor(tensor_factory,
  {batch, num_heads, seq_len, head_dim},
  data_type_t::f32, qkv_order);
  auto output_tensor_ref  = make_zero_tensor(tensor_factory,
  {batch, num_heads, seq_len, head_dim},
  data_type_t::f32, qkv_order);

  status_t status         = sdpa_kernel_test(query_tensor, key_tensor,
                            value_tensor, mask_tensor, output_tensor,
                            scale, /*is_causal=*/false, /*has_mask=*/true);
  status_t ref_status     = sdpa_forced_ref_kernel_test(query_tensor,
                            key_tensor, value_tensor, mask_tensor,
                            output_tensor_ref, scale,
                            /*is_causal=*/false, /*has_mask=*/true);

  bool is_test_successful =
    (status == status_t::success && ref_status == status_t::success);

  if (is_test_successful) {
    compare_tensor_4D_sdpa(output_tensor, output_tensor_ref,
                           batch, num_heads, seq_len, kv_seq_len, head_dim,
                           rtol_f32, epsilon_f32, is_test_successful);
  }

  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestSdpa parameterized test class
 *  @param BF16_BF16 BF16 Q/K/V/output (encoder-style self-attention).
 *  @brief Validate SDPA BF16 LOWOHA flash kernel against the operator-based
 *         reference (sdpa_encoder_ref_kernel_t). Mirrors F32_F32 with BF16
 *         storage; arithmetic in both kernels accumulates in FP32 internally.
 */
TEST_P(TestSdpa, BF16_BF16) {
  auto query_tensor       = make_uniform_tensor(tensor_factory,
  {batch, num_heads, seq_len, head_dim},
  data_type_t::bf16, 1.0, qkv_order);
  auto key_tensor         = make_uniform_tensor(tensor_factory,
  {batch, num_heads, kv_seq_len, head_dim},
  data_type_t::bf16, 1.0, qkv_order);
  auto value_tensor       = make_uniform_tensor(tensor_factory,
  {batch, num_heads, kv_seq_len, head_dim},
  data_type_t::bf16, 1.0, qkv_order);

  // For BF16 QKV both FP32 and BF16 additive masks are supported by the
  // reference operator and the LOWOHA flash backend; the fixture's mask_dt
  // (randomised in SetUp) selects one per (seed, params) instance so both
  // code paths get exercised across parameterisations. The mask is applied
  // to the FP32 score buffer either way (the BF16 path converts each
  // element to float at add time). Use [1, 1, S_q, S_kv] broadcast so the
  // reference's per-(b, h) mask advance and LOWOHA's broadcast agree.
  auto mask_tensor        = has_mask ?
                            tensor_factory.uniform_dist_tensor(
  {1UL, 1UL, seq_len, kv_seq_len},
  mask_dt, 0.5)
    : tensor_t();

  auto output_tensor      = make_zero_tensor(tensor_factory,
  {batch, num_heads, seq_len, head_dim},
  data_type_t::bf16, qkv_order);
  auto output_tensor_ref  = make_zero_tensor(tensor_factory,
  {batch, num_heads, seq_len, head_dim},
  data_type_t::bf16, qkv_order);

  status_t status         = sdpa_kernel_test(query_tensor, key_tensor,
                            value_tensor, mask_tensor, output_tensor,
                            scale, is_causal, has_mask);
  status_t ref_status     = sdpa_forced_ref_kernel_test(query_tensor,
                            key_tensor, value_tensor, mask_tensor,
                            output_tensor_ref, scale, is_causal, has_mask);

  bool is_test_successful =
    (status == status_t::success && ref_status == status_t::success);

  if (is_test_successful) {
    compare_tensor_4D_sdpa(output_tensor, output_tensor_ref,
                           batch, num_heads, seq_len, kv_seq_len, head_dim,
                           rtol_bf16, epsilon_bf16, is_test_successful);
  }

  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestSdpa parameterized test class
 *  @param BF16_BF16_NO_MASK BF16 path with no causal mask and no explicit mask
 *  @brief BF16 smoke test for the simplest SDPA shape: pure
 *         softmax(QK^T * scale) V. Q/K/V/output physical layout (BHSD or
 *         BSHD) is randomised per instance via the fixture's @c qkv_order.
 */
TEST_P(TestSdpa, BF16_BF16_NO_MASK) {
  auto query_tensor       = make_uniform_tensor(tensor_factory,
  {batch, num_heads, seq_len, head_dim},
  data_type_t::bf16, 1.0, qkv_order);
  auto key_tensor         = make_uniform_tensor(tensor_factory,
  {batch, num_heads, kv_seq_len, head_dim},
  data_type_t::bf16, 1.0, qkv_order);
  auto value_tensor       = make_uniform_tensor(tensor_factory,
  {batch, num_heads, kv_seq_len, head_dim},
  data_type_t::bf16, 1.0, qkv_order);
  tensor_t mask_tensor;  // empty: has_mask=false

  auto output_tensor      = make_zero_tensor(tensor_factory,
  {batch, num_heads, seq_len, head_dim},
  data_type_t::bf16, qkv_order);
  auto output_tensor_ref  = make_zero_tensor(tensor_factory,
  {batch, num_heads, seq_len, head_dim},
  data_type_t::bf16, qkv_order);

  status_t status         = sdpa_kernel_test(query_tensor, key_tensor,
                            value_tensor, mask_tensor, output_tensor,
                            scale, /*is_causal=*/false, /*has_mask=*/false);
  status_t ref_status     = sdpa_forced_ref_kernel_test(query_tensor,
                            key_tensor, value_tensor, mask_tensor,
                            output_tensor_ref, scale,
                            /*is_causal=*/false, /*has_mask=*/false);

  bool is_test_successful =
    (status == status_t::success && ref_status == status_t::success);

  if (is_test_successful) {
    compare_tensor_4D_sdpa(output_tensor, output_tensor_ref,
                           batch, num_heads, seq_len, kv_seq_len, head_dim,
                           rtol_bf16, epsilon_bf16, is_test_successful);
  }

  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestSdpa parameterized test class
 *  @param BF16_BF16_CAUSAL Forces causal masking with BF16 storage
 *  @brief Exercises the BF16 causal path: scores[i][j>i] = -inf before
 *         softmax, no explicit mask. Q/K/V/output physical layout (BHSD or
 *         BSHD) is randomised per instance via the fixture's @c qkv_order.
 */
TEST_P(TestSdpa, BF16_BF16_CAUSAL) {
  auto query_tensor       = make_uniform_tensor(tensor_factory,
  {batch, num_heads, seq_len, head_dim},
  data_type_t::bf16, 1.0, qkv_order);
  auto key_tensor         = make_uniform_tensor(tensor_factory,
  {batch, num_heads, kv_seq_len, head_dim},
  data_type_t::bf16, 1.0, qkv_order);
  auto value_tensor       = make_uniform_tensor(tensor_factory,
  {batch, num_heads, kv_seq_len, head_dim},
  data_type_t::bf16, 1.0, qkv_order);
  tensor_t mask_tensor;

  auto output_tensor      = make_zero_tensor(tensor_factory,
  {batch, num_heads, seq_len, head_dim},
  data_type_t::bf16, qkv_order);
  auto output_tensor_ref  = make_zero_tensor(tensor_factory,
  {batch, num_heads, seq_len, head_dim},
  data_type_t::bf16, qkv_order);

  status_t status         = sdpa_kernel_test(query_tensor, key_tensor,
                            value_tensor, mask_tensor, output_tensor,
                            scale, /*is_causal=*/true, /*has_mask=*/false);
  status_t ref_status     = sdpa_forced_ref_kernel_test(query_tensor,
                            key_tensor, value_tensor, mask_tensor,
                            output_tensor_ref, scale,
                            /*is_causal=*/true, /*has_mask=*/false);

  bool is_test_successful =
    (status == status_t::success && ref_status == status_t::success);

  if (is_test_successful) {
    compare_tensor_4D_sdpa(output_tensor, output_tensor_ref,
                           batch, num_heads, seq_len, kv_seq_len, head_dim,
                           rtol_bf16, epsilon_bf16, is_test_successful);
  }

  EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestSdpa parameterized test class
 *  @param BF16_BF16_MASK BF16 path with explicit additive mask
 *  @brief Exercises the explicit-mask code path with BF16 QKV and a mask
 *         dtype taken from the fixture's @c mask_dt (FP32 or BF16,
 *         randomised in @c SetUp). Both combinations are supported by the
 *         reference operator and the LOWOHA flash backend. Mask values are
 *         bounded so no row degenerates to all -inf. Q/K/V/output physical
 *         layout (BHSD or BSHD) is randomised per instance via the
 *         fixture's @c qkv_order.
 */
TEST_P(TestSdpa, BF16_BF16_MASK) {
  auto query_tensor       = make_uniform_tensor(tensor_factory,
  {batch, num_heads, seq_len, head_dim},
  data_type_t::bf16, 1.0, qkv_order);
  auto key_tensor         = make_uniform_tensor(tensor_factory,
  {batch, num_heads, kv_seq_len, head_dim},
  data_type_t::bf16, 1.0, qkv_order);
  auto value_tensor       = make_uniform_tensor(tensor_factory,
  {batch, num_heads, kv_seq_len, head_dim},
  data_type_t::bf16, 1.0, qkv_order);
  // Mask dtype is mask_dt (f32 or bf16; see SetUp), [1, 1, S_q, S_kv]
  // broadcast — see BF16_BF16 for the broadcast-shape rationale.
  auto mask_tensor        = tensor_factory.uniform_dist_tensor(
  {1UL, 1UL, seq_len, kv_seq_len},
  mask_dt, 0.5);

  auto output_tensor      = make_zero_tensor(tensor_factory,
  {batch, num_heads, seq_len, head_dim},
  data_type_t::bf16, qkv_order);
  auto output_tensor_ref  = make_zero_tensor(tensor_factory,
  {batch, num_heads, seq_len, head_dim},
  data_type_t::bf16, qkv_order);

  status_t status         = sdpa_kernel_test(query_tensor, key_tensor,
                            value_tensor, mask_tensor, output_tensor,
                            scale, /*is_causal=*/false, /*has_mask=*/true);
  status_t ref_status     = sdpa_forced_ref_kernel_test(query_tensor,
                            key_tensor, value_tensor, mask_tensor,
                            output_tensor_ref, scale,
                            /*is_causal=*/false, /*has_mask=*/true);

  bool is_test_successful =
    (status == status_t::success && ref_status == status_t::success);

  if (is_test_successful) {
    compare_tensor_4D_sdpa(output_tensor, output_tensor_ref,
                           batch, num_heads, seq_len, kv_seq_len, head_dim,
                           rtol_bf16, epsilon_bf16, is_test_successful);
  }

  EXPECT_TRUE(is_test_successful);
}

// ---------------------------------------------------------------------------
// Mask-layout variant: a single test per dtype that exercises per-(b, h)
// mask advance + 2D/4D broadcast, with the mask shape AND the Q/K/V/output
// physical layout (BHSD vs BSHD) randomly selected per parameterised
// instance from the fixture's @c mask_shape_kind / @c qkv_order. This
// collapses what used to be 8 dedicated TESTs per dtype (4 mask shapes x
// {BHSD, BSHD}) into 1 test body each; coverage of every combination is
// preserved across the 400-instance parameter sweep.
// ---------------------------------------------------------------------------

/** @brief F32 SDPA mask-layout coverage with random mask shape and random
 *         BHSD/BSHD Q/K/V/output layout. */
TEST_P(TestSdpa, F32_F32_MASK_LAYOUT) {
  run_sdpa_mask_layout_test(tensor_factory, batch, num_heads,
                            seq_len, kv_seq_len, head_dim,
                            scale, /*is_causal=*/false,
                            data_type_t::f32,
                            resolve_mask_shape(mask_shape_kind, batch,
                                num_heads, seq_len, kv_seq_len),
                            qkv_order, rtol_f32, epsilon_f32);
}

/** @brief BF16 SDPA mask-layout coverage with random mask shape, random
 *         mask dtype (FP32 or BF16), and random BHSD/BSHD Q/K/V/output
 *         layout. */
TEST_P(TestSdpa, BF16_BF16_MASK_LAYOUT) {
  run_sdpa_mask_layout_test(tensor_factory, batch, num_heads,
                            seq_len, kv_seq_len, head_dim,
                            scale, /*is_causal=*/false,
                            data_type_t::bf16,
                            resolve_mask_shape(mask_shape_kind, batch,
                                num_heads, seq_len, kv_seq_len),
                            qkv_order, rtol_bf16, epsilon_bf16, mask_dt);
}

/** @fn INSTANTIATE_TEST_SUITE_P
 *  @brief Triggers SDPA parameterized test suite
 */
INSTANTIATE_TEST_SUITE_P(Sdpa, TestSdpa,
                         ::testing::ValuesIn(sdpa_test));
