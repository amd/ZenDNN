/********************************************************************************
# * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
// Post-op metadata cache tests for the AOCL DLP backend.
//
// The cache under test lives in
//   zendnnl/src/lowoha_operators/matmul/backends/aocl/aocl_postop.cpp
// (per-thread LRU of `dlp_postop_metadata_holder_t*`, keyed by Key_matmul
// = weight_ptr/K/N/algo/postop_signature). Each test body issues TWO
// matmul calls back-to-back using the SAME weight tensor (so the cache
// key matches) and asserts:
//
//   * Call 1 (cold) produces a correct result vs the reference.
//   * Call 2 (hit)  produces a correct result vs the reference EVEN
//                   THOUGH per-call mutable fields (bias buffer, binary
//                   operand buffer, per-token quant scale buffer, dyn-
//                   quant zero-point buffer, ...) have been replaced
//                   between the two calls.
//
// Cache lifetime is preserved within one TEST_P body because
// `clear_matmul_test_caches()` only runs in `TearDown()`.
//
// To avoid duplicating each test body once per dtype, this file uses a
// single value-parameterized fixture `TestPostopCache` whose param
// carries (shape, algo, src_dt, dst_dt). Each scenario is one TEST_P
// body and is instantiated multiple times via INSTANTIATE_TEST_SUITE_P
// (F32/BF16/F16 for floating-point tests; BF16/F32 dst for INT8 tests).
//
// Coverage matrix:
//
//   Scenario                                  | F32 | BF16 | F16 | INT8->BF16 | INT8->F32
//   ------------------------------------------+-----+------+-----+------------+----------
//   HitParity                                 |  Y  |  Y   |  Y* |     -      |    -
//   BiasRefreshOnHit                          |  Y  |  Y   |  -  |     -      |    -
//   BinaryAddRefreshOnHit                     |  Y  |  Y   |  -  |     -      |    -
//   BinaryMulRefreshOnHit                     |  Y  |  Y   |  -  |     -      |    -
//   BiasDtypeKeysDistinct                     |  Y  |  Y   |  -  |     -      |    -
//   PostopOrderKeysDistinct                   |  Y  |  Y   |  -  |     -      |    -
//   LifecycleClear                            |  Y  |  Y   |  Y* |     -      |    -
//   SymQuantPerTokenScaleRefreshOnHit         |  -  |  -   |  -  |     Y      |    Y
//   DynQuantScaleRefreshOnHit                 |  -  |  -   |  -  |     Y      |    Y
//
// * F16 + AOCL-DLP does not support post-ops or bias (see TestMatmul.
//   F16_F16 in test_matmul.cpp), so the F16 instantiation only exercises
//   bias-less / post-op-less paths and skips the bias/binary/order tests.
//
// In addition to the parameterized matrix above, this file ends with a
// small set of NEGATIVE tests (TestPostopCacheZpCompNegative.*) that
// call create_dlp_post_op() directly to pin the API-boundary
// normalization of (zp_comp_ndim > 0, zp_comp_acc == nullptr) — the bad
// pair reachable when cache_or_compute_zp_compensation() OOMs after
// setting ndim but before allocating the comp buffer. These tests
// assert cache-key equivalence with the canonical "no zp_comp" call
// and cache-key distinctness from a real (ndim>0, acc!=null) call on
// the same weight.
//
// Finally, two test fixtures at the very bottom cover the runtime
// kill switch ZENDNNL_ENABLE_POSTOP_CACHE:
//
//   * TestPostopCacheEnableFlag     — env var -> config_postop_cache_t
//                                     plumbing (parser coverage for
//                                     every supported on/off spelling
//                                     and the safety contract that
//                                     unrecognized values leave the
//                                     default untouched).
//   * TestPostopCacheClearMechanism — proves clear_aocl_postop_metadata_cache
//                                     produces a fresh holder for the
//                                     same key on the next call, which
//                                     is the mechanism the disabled-
//                                     mode gate in create_dlp_post_op
//                                     leverages.
//
// The functional "cold-path-on-every-call still produces correct
// results" half of the kill switch is already covered for the full
// parameterized matrix by TestPostopCache.LifecycleClear — see the
// docstring on that test for why.
// =============================================================================

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>

#include "gtest_utils.hpp"
// Needed for the direct-API negative tests at the bottom of the file
// (create_dlp_post_op / clear_aocl_postop_metadata_cache). The header is
// not in gtest_utils.hpp's transitive closure.
#include "lowoha_operators/matmul/backends/aocl/aocl_postop.hpp"
// Needed for TestPostopCacheEnableFlag: the env var
// ZENDNNL_ENABLE_POSTOP_CACHE is parsed inside config_manager_t and
// exposed via get_postop_cache_config().
#include "common/config_manager.hpp"
// Needed for the GTEST_SKIP() guard at the top of each cache-on-
// dependent fixture's SetUp(): when the runtime kill switch
// ZENDNNL_ENABLE_POSTOP_CACHE=0 (or a future default of `false`)
// is in effect, create_dlp_post_op() clears the cache on every
// call so the WITH-cache invariants those fixtures pin (pointer-
// identity on hit, lifecycle clear semantics, refresh-on-hit
// correctness vs. a cold rebuild) no longer hold. We skip the
// affected tests cleanly rather than fail them.
#include "common/zendnnl_global.hpp"

#include <cstdlib>   // setenv / unsetenv for the env-flag tests
#include <string>

namespace {

// Tolerance bundle for compare_tensor_2D_matrix(), keyed off the dst
// dtype (which dominates the post-accumulation rounding budget).
struct postop_cache_tols_t {
  float rtol;
  float epsilon;
  bool  enable_f32_relaxation;
};

postop_cache_tols_t tols_for_dst(data_type_t dst_dt) {
  switch (dst_dt) {
    case data_type_t::f32:
      return {rtol_f32, epsilon_f32, /*enable_f32_relaxation=*/false};
    case data_type_t::bf16:
      return {rtol_bf16, epsilon_bf16, false};
    case data_type_t::f16:
      // No dedicated f16 tolerance constants exist; existing F16 tests
      // (TestMatmul.F16_F16) reuse the bf16 budget, so we do the same.
      return {rtol_bf16, epsilon_bf16, false};
    default:
      return {rtol_f32, epsilon_f32, false};
  }
}

// A short, descriptive token for a data_type_t used to label
// INSTANTIATE_TEST_SUITE_P prefixes and PrintTo output.
std::string dtype_token(data_type_t dt) {
  switch (dt) {
    case data_type_t::f32:  return "F32";
    case data_type_t::bf16: return "BF16";
    case data_type_t::f16:  return "F16";
    case data_type_t::s8:   return "S8";
    case data_type_t::u8:   return "U8";
    default:                return "DT";
  }
}

}  // anonymous namespace

/** @brief Parameters for one TestPostopCache instance.
 *
 *  Mirrors the minimal subset of MatmulType needed by the cache tests
 *  (shape + algo) and adds an explicit (src_dt, dst_dt) pair so the
 *  same test body can be instantiated across dtypes via
 *  INSTANTIATE_TEST_SUITE_P. */
struct PostopCacheParam {
  uint64_t      m;
  uint64_t      k;
  uint64_t      n;
  bool          transA;
  bool          transB;
  float         alpha;
  float         beta;
  matmul_algo_t algo;
  data_type_t   src_dt;
  data_type_t   dst_dt;
  int32_t       num_threads;
};

void PrintTo(const PostopCacheParam &p, ::std::ostream *os) {
  *os << dtype_token(p.src_dt) << "_" << dtype_token(p.dst_dt)
      << "_m" << p.m << "_k" << p.k << "_n" << p.n
      << "_tA" << static_cast<int>(p.transA)
      << "_tB" << static_cast<int>(p.transB)
      << "_algo" << static_cast<int>(p.algo);
}

namespace {

// A small, curated set of (M,K,N) shapes for cache validation. The
// cache key is shape-independent at the M axis (M is intentionally
// excluded from the signature so dynamic batch sizes can hit), so even
// one shape would prove patch correctness, but covering a few small +
// medium shapes catches accidental shape-dependent regressions.
struct cache_shape_t { uint64_t m, k, n; };
const std::vector<cache_shape_t> kCacheShapes = {
  {  8,  16,   8 },
  { 16,  64,  32 },
  { 32, 128,  32 },
  { 64, 128,  64 },
};

// Build a parameter vector for a given (src_dt, dst_dt) pair across
// both AOCL DLP algos (the only backends the cache lives in) and the
// curated shape list. transA/transB are fixed to false to keep the
// suite size predictable; the kernel-correctness suite in
// test_matmul.cpp already covers the transposed paths.
std::vector<PostopCacheParam> make_cache_params(data_type_t src_dt,
                                                data_type_t dst_dt) {
  std::vector<PostopCacheParam> out;
  const std::vector<matmul_algo_t> algos = {
    matmul_algo_t::aocl_dlp,
    matmul_algo_t::aocl_dlp_blocked,
  };
  for (auto algo : algos) {
    for (const auto &s : kCacheShapes) {
      out.push_back(PostopCacheParam{
        s.m, s.k, s.n,
        /*transA=*/false, /*transB=*/false,
        /*alpha=*/1.0f,   /*beta=*/0.0f,
        algo, src_dt, dst_dt,
        /*num_threads=*/1,
      });
    }
  }
  return out;
}

}  // anonymous namespace

/** @brief Value-parameterized fixture for AOCL DLP post-op metadata
 *         cache tests. Each TEST_P body issues two matmul calls with
 *         the same weight tensor so the cache key matches, then
 *         asserts both produce correct results despite per-call mutable
 *         inputs differing. */
class TestPostopCache : public ::testing::TestWithParam<PostopCacheParam> {
 protected:
  void SetUp() override {
    // These TEST_P bodies issue a cold call followed by a hot call on the
    // same cache key, with per-call mutable fields (bias buffer, binary
    // operand, per-token quant scale, dyn-quant zp) intentionally swapped
    // between the two calls. The hot call is the load-bearing assertion:
    // it pins that patch_mutable_fields refreshes those fields onto the
    // cached holder. With the cache disabled, both calls take the cold
    // path and the test no longer exercises that contract -- the
    // assertions still pass (the cold path naturally picks up the new
    // values) but the coverage signal is gone. Skip cleanly so a
    // ZENDNNL_ENABLE_POSTOP_CACHE=0 run doesn't quietly turn these into
    // tautological cold-vs-cold checks.
    if (!zendnnl::common::is_postop_cache_enabled()) {
      GTEST_SKIP() << "AOCL DLP post-op metadata cache is disabled "
                   "(ZENDNNL_ENABLE_POSTOP_CACHE=0); this fixture "
                   "exercises the WITH-cache hit-path contract.";
    }
    const PostopCacheParam params = GetParam();
    srand(static_cast<unsigned int>(seed));
    m           = params.m;
    k           = params.k;
    n           = params.n;
    transA      = params.transA;
    transB      = params.transB;
    alpha       = params.alpha;
    beta        = params.beta;
    algo        = params.algo;
    src_dt      = params.src_dt;
    dst_dt      = params.dst_dt;
    num_threads = params.num_threads;
    omp_set_num_threads(num_threads);
    log_info("PostopCache: src_dt=", dtype_token(src_dt),
             " dst_dt=", dtype_token(dst_dt),
             " m=", m, " k=", k, " n=", n,
             " algo=", static_cast<int>(algo));
  }

  void TearDown() override { clear_matmul_test_caches(); }

  // Returns true if (src_dt, dst_dt) is a floating-point compute path
  // (the FP tests below skip otherwise).
  bool is_fp_path() const {
    return (src_dt == data_type_t::f32  ||
            src_dt == data_type_t::bf16 ||
            src_dt == data_type_t::f16);
  }

  // F16 + AOCL DLP doesn't support post-ops or bias (see TestMatmul.
  // F16_F16 in test_matmul.cpp). Tests that exist solely to exercise
  // bias / binary / chain refresh on hit are not meaningful for F16
  // and skip themselves via this helper.
  bool aocl_dlp_supports_postops_for_src() const {
    return src_dt != data_type_t::f16;
  }

  // Shared body for the BinaryAdd / BinaryMul refresh-on-hit tests.
  // Both scenarios are identical except for the binary post-op enum,
  // so we factor the body into a member function to avoid duplication
  // while still letting the function reach the fixture's protected
  // state (tensor_factory, m/k/n, dtypes, etc.). Returns nothing; the
  // GTEST_SKIP / ASSERT / EXPECT macros operate on the current test.
  void run_binary_refresh_on_hit(post_op_type_t bin_po) {
    if (!is_fp_path()) {
      GTEST_SKIP() << "FP test (INT8 covered separately)";
    }
    if (!aocl_dlp_supports_postops_for_src()) {
      GTEST_SKIP() << "F16 + AOCL DLP does not support binary post-ops";
    }

    const std::vector<post_op_type_t> po = {bin_po};

    auto weight_tensor = tensor_factory.uniform_dist_tensor({k, n},
                         src_dt, 2.0, transB);
    auto input_tensor  = tensor_factory.uniform_dist_tensor({m, k},
                         src_dt, 2.0, transA);
    auto bias_tensor   = tensor_factory.uniform_dist_tensor({1, n}, dst_dt,
                                                            2.0);

    auto binary_1 = make_binary_postop_tensors(tensor_factory, po, {m, n},
                                               dst_dt, 2.0);
    auto binary_2 = make_binary_postop_tensors(tensor_factory, po, {m, n},
                                               dst_dt, 5.0);

    auto out_1 = tensor_factory.uniform_dist_tensor({m, n}, dst_dt, 2.0);
    auto ref_1 = tensor_factory.uniform_dist_tensor({m, n}, dst_dt, 2.0);
    status_t s1 = matmul_kernel_test(input_tensor, weight_tensor, bias_tensor,
                                     out_1, po, binary_1, true, algo, alpha,
                                     beta);
    if (s1 == status_t::isa_unsupported) {
      GTEST_SKIP() << dtype_token(src_dt) << " not supported on this ISA";
    }
    ASSERT_EQ(s1, status_t::success);
    ASSERT_EQ(matmul_forced_ref_kernel_test(input_tensor, weight_tensor,
                                            bias_tensor, ref_1, po, binary_1,
                                            true, algo, alpha, beta),
              status_t::success);

    auto out_2 = tensor_factory.uniform_dist_tensor({m, n}, dst_dt, 2.0);
    auto ref_2 = tensor_factory.uniform_dist_tensor({m, n}, dst_dt, 2.0);
    ASSERT_EQ(matmul_kernel_test(input_tensor, weight_tensor, bias_tensor,
                                 out_2, po, binary_2, true, algo, alpha,
                                 beta), status_t::success);
    ASSERT_EQ(matmul_forced_ref_kernel_test(input_tensor, weight_tensor,
                                            bias_tensor, ref_2, po, binary_2,
                                            true, algo, alpha, beta),
              status_t::success);

    const auto t = tols_for_dst(dst_dt);
    bool ok_1 = true, ok_2 = true;
    compare_tensor_2D_matrix(out_1, ref_1, m, n, k, t.rtol, t.epsilon, ok_1,
                             t.enable_f32_relaxation, alpha);
    compare_tensor_2D_matrix(out_2, ref_2, m, n, k, t.rtol, t.epsilon, ok_2,
                             t.enable_f32_relaxation, alpha);
    EXPECT_TRUE(ok_1);
    EXPECT_TRUE(ok_2);
  }

  uint64_t       m{}, k{}, n{};
  bool           transA{}, transB{};
  float          alpha{1.0f}, beta{0.0f};
  matmul_algo_t  algo{matmul_algo_t::aocl_dlp};
  data_type_t    src_dt{data_type_t::f32};
  data_type_t    dst_dt{data_type_t::f32};
  int32_t        num_threads{1};
  tensor_factory_t tensor_factory{};
};

// =============================================================================
// Tier 1: floating-point tests parameterized on (src_dt, dst_dt).
// =============================================================================

/** @brief Cold/hit parity: two identical FP matmul calls (same key,
 *         same inputs) must each produce the same correct result.
 *         Establishes the pattern used by the rest of the cache suite. */
TEST_P(TestPostopCache, HitParity) {
  if (!is_fp_path()) {
    GTEST_SKIP() << "FP test (INT8 covered separately)";
  }

  // F16 + AOCL DLP: post-ops/bias unsupported. For plain-matmul layers
  // (no post-ops, no bias, not WOQ, not INT8) create_dlp_post_op
  // returns nullptr without allocating or caching a holder, so both
  // calls below take the same early-return path — the test verifies
  // they produce identical output and that nothing blows up across
  // repeated invocations on the same key.
  const bool drop_postops_and_bias = !aocl_dlp_supports_postops_for_src();
  const std::vector<post_op_type_t> po = drop_postops_and_bias
                                         ? std::vector<post_op_type_t>{}
                                         : std::vector<post_op_type_t>{
                                             post_op_type_t::relu};

  auto weight_tensor = tensor_factory.uniform_dist_tensor({k, n},
                       src_dt, 2.0, transB);
  auto input_tensor  = tensor_factory.uniform_dist_tensor({m, k},
                       src_dt, 2.0, transA);
  auto bias_tensor   = drop_postops_and_bias
                       ? tensor_t()
                       : tensor_factory.uniform_dist_tensor({1, n}, dst_dt,
                                                            2.0);
  auto binary_tensors = make_binary_postop_tensors(tensor_factory, po,
                                                   {m, n}, dst_dt);

  // Call 1 (cold path).
  auto out_1 = tensor_factory.uniform_dist_tensor({m, n}, dst_dt, 2.0);
  auto ref_1 = tensor_factory.uniform_dist_tensor({m, n}, dst_dt, 2.0);
  status_t s1 = matmul_kernel_test(input_tensor, weight_tensor, bias_tensor,
                                   out_1, po, binary_tensors,
                                   /*use_LOWOHA=*/true, algo, alpha, beta);
  if (s1 == status_t::isa_unsupported) {
    GTEST_SKIP() << dtype_token(src_dt) << " not supported on this ISA";
  }
  ASSERT_EQ(s1, status_t::success);
  ASSERT_EQ(matmul_forced_ref_kernel_test(input_tensor, weight_tensor,
                                          bias_tensor, ref_1, po,
                                          binary_tensors, true, algo,
                                          alpha, beta), status_t::success);

  // Call 2 (hit path; same weight key, same inputs).
  auto out_2 = tensor_factory.uniform_dist_tensor({m, n}, dst_dt, 2.0);
  auto ref_2 = tensor_factory.uniform_dist_tensor({m, n}, dst_dt, 2.0);
  ASSERT_EQ(matmul_kernel_test(input_tensor, weight_tensor, bias_tensor,
                               out_2, po, binary_tensors, true, algo, alpha,
                               beta), status_t::success);
  ASSERT_EQ(matmul_forced_ref_kernel_test(input_tensor, weight_tensor,
                                          bias_tensor, ref_2, po,
                                          binary_tensors, true, algo, alpha,
                                          beta), status_t::success);

  const auto t = tols_for_dst(dst_dt);
  bool ok_1 = true, ok_2 = true;
  compare_tensor_2D_matrix(out_1, ref_1, m, n, k, t.rtol, t.epsilon,
                           ok_1, t.enable_f32_relaxation, alpha);
  compare_tensor_2D_matrix(out_2, ref_2, m, n, k, t.rtol, t.epsilon,
                           ok_2, t.enable_f32_relaxation, alpha);
  EXPECT_TRUE(ok_1);
  EXPECT_TRUE(ok_2);
}

/** @brief Mutable-field regression for bias[i].bias: same weight key,
 *         distinct bias buffers across the two calls. If
 *         patch_mutable_fields fails to repoint bias on hit, call 2
 *         reads bias_1's stale address and disagrees with the reference
 *         computed against bias_2. */
TEST_P(TestPostopCache, BiasRefreshOnHit) {
  if (!is_fp_path()) {
    GTEST_SKIP() << "FP test (INT8 covered separately)";
  }
  if (!aocl_dlp_supports_postops_for_src()) {
    GTEST_SKIP() << "F16 + AOCL DLP does not support bias";
  }

  const std::vector<post_op_type_t> po;  // bias only, no post-op chain

  auto weight_tensor = tensor_factory.uniform_dist_tensor({k, n},
                       src_dt, 2.0, transB);
  auto input_tensor  = tensor_factory.uniform_dist_tensor({m, k},
                       src_dt, 2.0, transA);

  // Distinct bias buffers (different addresses + different value ranges
  // so any stale-pointer reuse produces an observable output mismatch).
  auto bias_1 = tensor_factory.uniform_dist_tensor({1, n}, dst_dt, 2.0);
  auto bias_2 = tensor_factory.uniform_dist_tensor({1, n}, dst_dt, 5.0);
  std::vector<tensor_t> no_binaries;

  auto out_1 = tensor_factory.uniform_dist_tensor({m, n}, dst_dt, 2.0);
  auto ref_1 = tensor_factory.uniform_dist_tensor({m, n}, dst_dt, 2.0);
  status_t s1 = matmul_kernel_test(input_tensor, weight_tensor, bias_1, out_1,
                                   po, no_binaries, true, algo, alpha, beta);
  if (s1 == status_t::isa_unsupported) {
    GTEST_SKIP() << dtype_token(src_dt) << " not supported on this ISA";
  }
  ASSERT_EQ(s1, status_t::success);
  ASSERT_EQ(matmul_forced_ref_kernel_test(input_tensor, weight_tensor, bias_1,
                                          ref_1, po, no_binaries, true, algo,
                                          alpha, beta), status_t::success);

  // Hit path: same weight key, DIFFERENT bias buffer.
  auto out_2 = tensor_factory.uniform_dist_tensor({m, n}, dst_dt, 2.0);
  auto ref_2 = tensor_factory.uniform_dist_tensor({m, n}, dst_dt, 2.0);
  ASSERT_EQ(matmul_kernel_test(input_tensor, weight_tensor, bias_2, out_2,
                               po, no_binaries, true, algo, alpha, beta),
            status_t::success);
  ASSERT_EQ(matmul_forced_ref_kernel_test(input_tensor, weight_tensor, bias_2,
                                          ref_2, po, no_binaries, true, algo,
                                          alpha, beta), status_t::success);

  const auto t = tols_for_dst(dst_dt);
  bool ok_1 = true, ok_2 = true;
  compare_tensor_2D_matrix(out_1, ref_1, m, n, k, t.rtol, t.epsilon, ok_1,
                           t.enable_f32_relaxation, alpha);
  compare_tensor_2D_matrix(out_2, ref_2, m, n, k, t.rtol, t.epsilon, ok_2,
                           t.enable_f32_relaxation, alpha);
  EXPECT_TRUE(ok_1);
  EXPECT_TRUE(ok_2);
}

/** @brief Mutable-field regression for matrix_add[i].buff: same weight
 *         key, distinct binary_add operand buffers across the two
 *         calls. */
TEST_P(TestPostopCache, BinaryAddRefreshOnHit) {
  run_binary_refresh_on_hit(post_op_type_t::binary_add);
}

/** @brief Mutable-field regression for matrix_mul[i].buff: same weight
 *         key, distinct binary_mul operand buffers across the two
 *         calls. Mirrors BinaryAdd above but exercises the
 *         multiplicative branch of patch_mutable_fields. */
TEST_P(TestPostopCache, BinaryMulRefreshOnHit) {
  run_binary_refresh_on_hit(post_op_type_t::binary_mul);
}

/** @brief Key-distinctness regression for dtypes.bias: same weight,
 *         shapes, and post-op chain, but two different bias dtypes
 *         (bf16 vs f32). compute_postop_signature folds dtypes.bias
 *         into the signature, so the two calls must miss separately
 *         and each produce a result matching its own reference. */
TEST_P(TestPostopCache, BiasDtypeKeysDistinct) {
  if (!is_fp_path()) {
    GTEST_SKIP() << "FP test (INT8 covered separately)";
  }
  if (!aocl_dlp_supports_postops_for_src()) {
    GTEST_SKIP() << "F16 + AOCL DLP does not support bias";
  }

  const std::vector<post_op_type_t> po;

  auto weight_tensor = tensor_factory.uniform_dist_tensor({k, n},
                       src_dt, 2.0, transB);
  auto input_tensor  = tensor_factory.uniform_dist_tensor({m, k},
                       src_dt, 2.0, transA);

  auto bias_bf16 = tensor_factory.uniform_dist_tensor({1, n},
                                                      data_type_t::bf16, 2.0);
  auto bias_f32  = tensor_factory.uniform_dist_tensor({1, n},
                                                      data_type_t::f32, 2.0);
  std::vector<tensor_t> no_binaries;

  auto out_1 = tensor_factory.uniform_dist_tensor({m, n}, dst_dt, 2.0);
  auto ref_1 = tensor_factory.uniform_dist_tensor({m, n}, dst_dt, 2.0);
  status_t s1 = matmul_kernel_test(input_tensor, weight_tensor, bias_bf16,
                                   out_1, po, no_binaries, true, algo, alpha,
                                   beta);
  if (s1 == status_t::isa_unsupported) {
    GTEST_SKIP() << dtype_token(src_dt) << " not supported on this ISA";
  }
  ASSERT_EQ(s1, status_t::success);
  ASSERT_EQ(matmul_forced_ref_kernel_test(input_tensor, weight_tensor,
                                          bias_bf16, ref_1, po, no_binaries,
                                          true, algo, alpha, beta),
            status_t::success);

  // Same weight_ptr/K/N/algo; bias dtype differs => separate cache
  // entry expected.
  auto out_2 = tensor_factory.uniform_dist_tensor({m, n}, dst_dt, 2.0);
  auto ref_2 = tensor_factory.uniform_dist_tensor({m, n}, dst_dt, 2.0);
  ASSERT_EQ(matmul_kernel_test(input_tensor, weight_tensor, bias_f32, out_2,
                               po, no_binaries, true, algo, alpha, beta),
            status_t::success);
  ASSERT_EQ(matmul_forced_ref_kernel_test(input_tensor, weight_tensor,
                                          bias_f32, ref_2, po, no_binaries,
                                          true, algo, alpha, beta),
            status_t::success);

  const auto t = tols_for_dst(dst_dt);
  bool ok_1 = true, ok_2 = true;
  compare_tensor_2D_matrix(out_1, ref_1, m, n, k, t.rtol, t.epsilon, ok_1,
                           t.enable_f32_relaxation, alpha);
  compare_tensor_2D_matrix(out_2, ref_2, m, n, k, t.rtol, t.epsilon, ok_2,
                           t.enable_f32_relaxation, alpha);
  EXPECT_TRUE(ok_1);
  EXPECT_TRUE(ok_2);
}

/** @brief Key-distinctness regression for post-op chain order: same
 *         weight key, same dtypes, two different chains that are
 *         permutations of each other ({relu, gelu_tanh} vs
 *         {gelu_tanh, relu}). The signature folds po_type in iteration
 *         order, so the two chains must hash distinctly. */
TEST_P(TestPostopCache, PostopOrderKeysDistinct) {
  if (!is_fp_path()) {
    GTEST_SKIP() << "FP test (INT8 covered separately)";
  }
  if (!aocl_dlp_supports_postops_for_src()) {
    GTEST_SKIP() << "F16 + AOCL DLP does not support eltwise post-ops";
  }

  const std::vector<post_op_type_t> chain_1 = {post_op_type_t::relu,
                                               post_op_type_t::gelu_tanh};
  const std::vector<post_op_type_t> chain_2 = {post_op_type_t::gelu_tanh,
                                               post_op_type_t::relu};

  auto weight_tensor = tensor_factory.uniform_dist_tensor({k, n},
                       src_dt, 2.0, transB);
  auto input_tensor  = tensor_factory.uniform_dist_tensor({m, k},
                       src_dt, 2.0, transA);
  auto bias_tensor   = tensor_factory.uniform_dist_tensor({1, n}, dst_dt,
                                                          2.0);
  std::vector<tensor_t> no_binaries;

  auto out_1 = tensor_factory.uniform_dist_tensor({m, n}, dst_dt, 2.0);
  auto ref_1 = tensor_factory.uniform_dist_tensor({m, n}, dst_dt, 2.0);
  status_t s1 = matmul_kernel_test(input_tensor, weight_tensor, bias_tensor,
                                   out_1, chain_1, no_binaries, true, algo,
                                   alpha, beta);
  if (s1 == status_t::isa_unsupported) {
    GTEST_SKIP() << dtype_token(src_dt) << " not supported on this ISA";
  }
  ASSERT_EQ(s1, status_t::success);
  ASSERT_EQ(matmul_forced_ref_kernel_test(input_tensor, weight_tensor,
                                          bias_tensor, ref_1, chain_1,
                                          no_binaries, true, algo, alpha,
                                          beta), status_t::success);

  auto out_2 = tensor_factory.uniform_dist_tensor({m, n}, dst_dt, 2.0);
  auto ref_2 = tensor_factory.uniform_dist_tensor({m, n}, dst_dt, 2.0);
  ASSERT_EQ(matmul_kernel_test(input_tensor, weight_tensor, bias_tensor,
                               out_2, chain_2, no_binaries, true, algo, alpha,
                               beta), status_t::success);
  ASSERT_EQ(matmul_forced_ref_kernel_test(input_tensor, weight_tensor,
                                          bias_tensor, ref_2, chain_2,
                                          no_binaries, true, algo, alpha,
                                          beta), status_t::success);

  const auto t = tols_for_dst(dst_dt);
  bool ok_1 = true, ok_2 = true;
  compare_tensor_2D_matrix(out_1, ref_1, m, n, k, t.rtol, t.epsilon, ok_1,
                           t.enable_f32_relaxation, alpha);
  compare_tensor_2D_matrix(out_2, ref_2, m, n, k, t.rtol, t.epsilon, ok_2,
                           t.enable_f32_relaxation, alpha);
  EXPECT_TRUE(ok_1);
  EXPECT_TRUE(ok_2);
}

/** @brief Cache lifecycle: populate -> clear -> repopulate. Call 1
 *         inserts a holder, clear_matmul_test_caches() frees it, and
 *         call 2 must rebuild from scratch. Both results must be
 *         correct; a use-after-free or stale-holder bug in the clear
 *         path would crash or corrupt call 2's output.
 *
 *         This test body is also the parameterized functional proof
 *         for the ZENDNNL_ENABLE_POSTOP_CACHE=0 runtime kill switch
 *         (see is_postop_cache_enabled() in zendnnl_global.hpp and the
 *         gate at the top of create_dlp_post_op). The kill switch
 *         works by clearing the post-op metadata cache before every
 *         lookup, which is mechanically the same as the clear()
 *         between the two calls below — so passing LifecycleClear
 *         across the full (dtype, shape, algo) matrix is equivalent
 *         to proving cold-path-on-every-call correctness in disabled
 *         mode for that matrix. The env-var plumbing and the cache-
 *         clear -> fresh-holder mechanism that the kill switch
 *         depends on are pinned separately by
 *         TestPostopCacheEnableFlag and TestPostopCacheClearMechanism
 *         at the bottom of this file. */
TEST_P(TestPostopCache, LifecycleClear) {
  if (!is_fp_path()) {
    GTEST_SKIP() << "FP test (INT8 covered separately)";
  }

  // F16 + AOCL DLP: post-ops/bias unsupported. create_dlp_post_op
  // returns nullptr without caching for this plain-matmul layer, so
  // clear_matmul_test_caches() has no postop-cache holder to evict —
  // but the test still verifies the clear cycle is a no-op for the
  // no-postop path and call 2 stays correct after it (catches
  // regressions in other caches the clear also touches, e.g. the
  // matmul weight LRU).
  const bool drop_postops_and_bias = !aocl_dlp_supports_postops_for_src();
  const std::vector<post_op_type_t> po = drop_postops_and_bias
                                         ? std::vector<post_op_type_t>{}
                                         : std::vector<post_op_type_t>{
                                             post_op_type_t::relu};

  auto weight_tensor = tensor_factory.uniform_dist_tensor({k, n},
                       src_dt, 2.0, transB);
  auto input_tensor  = tensor_factory.uniform_dist_tensor({m, k},
                       src_dt, 2.0, transA);
  auto bias_tensor   = drop_postops_and_bias
                       ? tensor_t()
                       : tensor_factory.uniform_dist_tensor({1, n}, dst_dt,
                                                            2.0);
  auto binary_tensors = make_binary_postop_tensors(tensor_factory, po,
                                                   {m, n}, dst_dt);

  auto out_1 = tensor_factory.uniform_dist_tensor({m, n}, dst_dt, 2.0);
  auto ref_1 = tensor_factory.uniform_dist_tensor({m, n}, dst_dt, 2.0);
  status_t s1 = matmul_kernel_test(input_tensor, weight_tensor, bias_tensor,
                                   out_1, po, binary_tensors, true, algo,
                                   alpha, beta);
  if (s1 == status_t::isa_unsupported) {
    GTEST_SKIP() << dtype_token(src_dt) << " not supported on this ISA";
  }
  ASSERT_EQ(s1, status_t::success);
  ASSERT_EQ(matmul_forced_ref_kernel_test(input_tensor, weight_tensor,
                                          bias_tensor, ref_1, po,
                                          binary_tensors, true, algo, alpha,
                                          beta), status_t::success);

  // Drop every holder owned by this thread.
  clear_matmul_test_caches();

  // Cold path again (cache was just emptied). Same key, same inputs.
  auto out_2 = tensor_factory.uniform_dist_tensor({m, n}, dst_dt, 2.0);
  auto ref_2 = tensor_factory.uniform_dist_tensor({m, n}, dst_dt, 2.0);
  ASSERT_EQ(matmul_kernel_test(input_tensor, weight_tensor, bias_tensor,
                               out_2, po, binary_tensors, true, algo, alpha,
                               beta), status_t::success);
  ASSERT_EQ(matmul_forced_ref_kernel_test(input_tensor, weight_tensor,
                                          bias_tensor, ref_2, po,
                                          binary_tensors, true, algo, alpha,
                                          beta), status_t::success);

  const auto t = tols_for_dst(dst_dt);
  bool ok_1 = true, ok_2 = true;
  compare_tensor_2D_matrix(out_1, ref_1, m, n, k, t.rtol, t.epsilon, ok_1,
                           t.enable_f32_relaxation, alpha);
  compare_tensor_2D_matrix(out_2, ref_2, m, n, k, t.rtol, t.epsilon, ok_2,
                           t.enable_f32_relaxation, alpha);
  EXPECT_TRUE(ok_1);
  EXPECT_TRUE(ok_2);
}

// =============================================================================
// Tier 2: INT8 quant-cache tests, parameterized on dst_dt.
//
// `src_dt` from the param is ignored here (the source is always s8,
// dynamically quantized inside the body); only `dst_dt` selects the
// output dtype.
// =============================================================================

/** @brief Mutable-field regression for post_op_grp->a_scl in INT8
 *         symmetric per-token quantization. Same s8 weight (so same
 *         cache key), two distinct s8 sources quantized independently:
 *         each call produces a fresh per-token src_scale buffer at a
 *         new address with new values. On hit, patch_mutable_fields
 *         must refresh post_op_grp->a_scl->scale_factor (and its
 *         length) to call 2's buffer. */
TEST_P(TestPostopCache, SymQuantPerTokenScaleRefreshOnHit) {
  // INT8-only test body. Skip on FP instantiations (those rows are covered
  // by the FP test bodies in this same suite) and on any dst that the
  // AOCL DLP INT8 sym-quant path does not support (f16 in particular).
  if (is_fp_path()) {
    GTEST_SKIP() << "INT8 test (FP rows covered separately)";
  }
  if (dst_dt != data_type_t::bf16 && dst_dt != data_type_t::f32) {
    GTEST_SKIP() << "AOCL DLP INT8 sym-quant supports only bf16/f32 dst";
  }

  uint64_t sym_k = (k / 4) * 4;
  if (sym_k == 0) sym_k = 4;

  const data_type_t ref_dt   = (dst_dt == data_type_t::f32)
                               ? data_type_t::f32 : data_type_t::bf16;
  const data_type_t scale_dt = data_type_t::f32;
  const std::vector<int64_t> wei_sd = {1, static_cast<int64_t>(n)};
  const std::vector<int64_t> src_sd = {static_cast<int64_t>(m), 1};

  // Quantize the weight ONCE so the s8 buffer pointer (== cache key)
  // is identical across both matmul calls.
  auto wei_ref = tensor_factory.uniform_dist_tensor({sym_k, n}, ref_dt, 25.0,
                                                    transB);
  tensor_t weight_tensor, wei_scale, wei_zp;
  ASSERT_EQ(quant_params_compute(tensor_factory, wei_ref, ref_dt,
                                 data_type_t::s8, wei_sd, scale_dt, wei_scale,
                                 wei_zp, &weight_tensor),
            status_t::success) << "weight quantization failed";

  auto bias_tensor    = tensor_factory.uniform_dist_tensor({1, n}, ref_dt,
                                                           2.0);
  std::vector<tensor_t> no_binaries;

  // Source 1: independently quantized => fresh src_scale_1 buffer.
  auto src_ref_1 = tensor_factory.uniform_dist_tensor({m, sym_k}, ref_dt,
                                                      25.0, transA);
  tensor_t input_1, src_scale_1, src_zp_1;
  ASSERT_EQ(quant_params_compute(tensor_factory, src_ref_1, ref_dt,
                                 data_type_t::s8, src_sd, scale_dt,
                                 src_scale_1, src_zp_1, &input_1),
            status_t::success) << "source 1 quantization failed";

  // Source 2: different value range + independently quantized.
  auto src_ref_2 = tensor_factory.uniform_dist_tensor({m, sym_k}, ref_dt,
                                                      15.0, transA);
  tensor_t input_2, src_scale_2, src_zp_2;
  ASSERT_EQ(quant_params_compute(tensor_factory, src_ref_2, ref_dt,
                                 data_type_t::s8, src_sd, scale_dt,
                                 src_scale_2, src_zp_2, &input_2),
            status_t::success) << "source 2 quantization failed";

  auto out_1 = tensor_factory.uniform_dist_tensor({m, n}, dst_dt, 2.0);
  auto ref_1 = tensor_factory.uniform_dist_tensor({m, n}, dst_dt, 2.0);
  ASSERT_EQ(matmul_kernel_test(input_1, weight_tensor, bias_tensor, out_1,
                               std::vector<post_op_type_t>{},
                               no_binaries, true, algo, 1.0, 0.0),
            status_t::success);
  ASSERT_EQ(matmul_forced_ref_kernel_test(input_1, weight_tensor, bias_tensor,
                                          ref_1, {}, no_binaries, true, algo,
                                          1.0, 0.0), status_t::success);

  auto out_2 = tensor_factory.uniform_dist_tensor({m, n}, dst_dt, 2.0);
  auto ref_2 = tensor_factory.uniform_dist_tensor({m, n}, dst_dt, 2.0);
  ASSERT_EQ(matmul_kernel_test(input_2, weight_tensor, bias_tensor, out_2,
                               {}, no_binaries, true, algo, 1.0, 0.0),
            status_t::success);
  ASSERT_EQ(matmul_forced_ref_kernel_test(input_2, weight_tensor, bias_tensor,
                                          ref_2, {}, no_binaries, true, algo,
                                          1.0, 0.0), status_t::success);

  // INT8 quant noise dominates the dst rounding budget for any dst dtype,
  // so use bf16 tolerance unconditionally (mirrors INT8_*_GEMM_* in
  // test_matmul.cpp).
  bool ok_1 = true, ok_2 = true;
  compare_tensor_2D_matrix(out_1, ref_1, m, n, sym_k, rtol_bf16,
                           18.0f * epsilon_bf16, ok_1,
                           /*enable_f32_relaxation=*/false, 1.0f,
                           /*is_quant=*/true);
  compare_tensor_2D_matrix(out_2, ref_2, m, n, sym_k, rtol_bf16,
                           18.0f * epsilon_bf16, ok_2,
                           /*enable_f32_relaxation=*/false, 1.0f,
                           /*is_quant=*/true);
  EXPECT_TRUE(ok_1);
  EXPECT_TRUE(ok_2);
}

/** @brief Mutable-field regression for the per-call dynamic-quantization
 *         scale buffer under INT8 dynamic GEMM. The s8 weight is
 *         pre-quantized once (stable cache key); both calls pass FP
 *         sources with their OWN zero_tensor src_scale slots so the
 *         reorder_quantization path inside matmul allocates a fresh
 *         per-call scale buffer for each call. On hit,
 *         patch_mutable_fields must repoint the cached scale buffer at
 *         call 2's allocation. */
TEST_P(TestPostopCache, DynQuantScaleRefreshOnHit) {
  // INT8-only test body. Skip on FP instantiations (those rows are covered
  // by the FP test bodies in this same suite) and on any dst that the
  // AOCL DLP INT8 dyn-quant path does not support (f16 in particular).
  if (is_fp_path()) {
    GTEST_SKIP() << "INT8 test (FP rows covered separately)";
  }
  if (dst_dt != data_type_t::bf16 && dst_dt != data_type_t::f32) {
    GTEST_SKIP() << "AOCL DLP INT8 dyn-quant supports only bf16/f32 dst";
  }

  uint64_t sym_k = (k / 4) * 4;
  if (sym_k == 0) sym_k = 4;

  const data_type_t test_dt  = (dst_dt == data_type_t::f32)
                               ? data_type_t::f32 : data_type_t::bf16;
  const data_type_t scale_dt = data_type_t::f32;
  const std::vector<int64_t>  wei_scale_dims  = {1,
                                                 static_cast<int64_t>(n)};
  const std::vector<uint64_t> src_scale_shape = {m, 1};

  // Pre-quantize the weight ONCE.
  auto weight_tensor_ref = tensor_factory.uniform_dist_tensor({sym_k, n},
                                                              test_dt, 2.0);
  tensor_t weight_tensor_s8, wei_scale, wei_zp;
  ASSERT_EQ(quant_params_compute(tensor_factory, weight_tensor_ref, test_dt,
                                 data_type_t::s8, wei_scale_dims, scale_dt,
                                 wei_scale, wei_zp, &weight_tensor_s8),
            status_t::success) << "weight quantization failed";

  auto bias_tensor = tensor_factory.uniform_dist_tensor({1, n}, test_dt, 2.0);
  std::vector<tensor_t> no_binaries;

  // Two distinct (src, src_scale) pairs; each src_scale is a fresh
  // zero_tensor allocation, so the reorder_quantization path inside
  // matmul writes per-call scales into a NEW buffer pointer per call.
  auto src_scale_1 = tensor_factory.zero_tensor(src_scale_shape, scale_dt);
  auto input_1     = tensor_factory.uniform_dist_tensor({m, sym_k}, test_dt,
                                                        2.0, transA,
                                                        src_scale_1,
                                                        tensor_t());
  auto input_1_ref = tensor_factory.uniform_dist_tensor({m, sym_k}, test_dt,
                                                        2.0, transA);

  auto src_scale_2 = tensor_factory.zero_tensor(src_scale_shape, scale_dt);
  auto input_2     = tensor_factory.uniform_dist_tensor({m, sym_k}, test_dt,
                                                        5.0, transA,
                                                        src_scale_2,
                                                        tensor_t());
  auto input_2_ref = tensor_factory.uniform_dist_tensor({m, sym_k}, test_dt,
                                                        5.0, transA);

  auto out_1 = tensor_factory.uniform_dist_tensor({m, n}, test_dt, 2.0);
  auto ref_1 = tensor_factory.uniform_dist_tensor({m, n}, test_dt, 2.0);
  ASSERT_EQ(matmul_kernel_test(input_1, weight_tensor_s8, bias_tensor, out_1,
                               {}, no_binaries, true, algo, 1.0, 0.0),
            status_t::success);
  ASSERT_EQ(matmul_forced_ref_kernel_test(input_1_ref, weight_tensor_ref,
                                          bias_tensor, ref_1, {}, no_binaries,
                                          true, algo, 1.0, 0.0),
            status_t::success);

  auto out_2 = tensor_factory.uniform_dist_tensor({m, n}, test_dt, 2.0);
  auto ref_2 = tensor_factory.uniform_dist_tensor({m, n}, test_dt, 2.0);
  ASSERT_EQ(matmul_kernel_test(input_2, weight_tensor_s8, bias_tensor, out_2,
                               {}, no_binaries, true, algo, 1.0, 0.0),
            status_t::success);
  ASSERT_EQ(matmul_forced_ref_kernel_test(input_2_ref, weight_tensor_ref,
                                          bias_tensor, ref_2, {}, no_binaries,
                                          true, algo, 1.0, 0.0),
            status_t::success);

  // Dyn-quant introduces quantization error between the kernel (s8 weight,
  // on-the-fly s8 source) and the FP reference; quant noise dominates the
  // dst rounding budget for both bf16 and f32 dst. Use bf16 tolerance
  // unconditionally (matches INT8_DYNAMIC_GEMM_F32 / _BF16 in
  // test_matmul.cpp).
  bool ok_1 = true, ok_2 = true;
  compare_tensor_2D_matrix(out_1, ref_1, m, n, sym_k, rtol_bf16,
                           18.0f * epsilon_bf16, ok_1, false, 1.0f, true);
  compare_tensor_2D_matrix(out_2, ref_2, m, n, sym_k, rtol_bf16,
                           18.0f * epsilon_bf16, ok_2, false, 1.0f, true);
  EXPECT_TRUE(ok_1);
  EXPECT_TRUE(ok_2);
}

// =============================================================================
// INSTANTIATE_TEST_SUITE_P: one instantiation per (src_dt, dst_dt) pair
// we want to cover. Each instantiation fans the four kCacheShapes ×
// two AOCL DLP algos across every TEST_P body above (with body-level
// GTEST_SKIPs for the FP-only or INT8-only tests).
// =============================================================================

INSTANTIATE_TEST_SUITE_P(
  F32_F32, TestPostopCache,
  ::testing::ValuesIn(make_cache_params(data_type_t::f32,  data_type_t::f32)));

INSTANTIATE_TEST_SUITE_P(
  BF16_BF16, TestPostopCache,
  ::testing::ValuesIn(make_cache_params(data_type_t::bf16, data_type_t::bf16)));

INSTANTIATE_TEST_SUITE_P(
  F16_F16, TestPostopCache,
  ::testing::ValuesIn(make_cache_params(data_type_t::f16,  data_type_t::f16)));

// INT8 instantiations: src_dt=s8 selects the INT8 SymQuant/DynQuant test
// bodies via is_fp_path(); FP test bodies in the same suite GTEST_SKIP
// here, and INT8 test bodies GTEST_SKIP on the FP rows above. AOCL DLP
// INT8 only supports bf16/f32 dst, so we instantiate exactly those two.
INSTANTIATE_TEST_SUITE_P(
  INT8_BF16, TestPostopCache,
  ::testing::ValuesIn(make_cache_params(data_type_t::s8,   data_type_t::bf16)));

INSTANTIATE_TEST_SUITE_P(
  INT8_F32, TestPostopCache,
  ::testing::ValuesIn(make_cache_params(data_type_t::s8,   data_type_t::f32)));

// =============================================================================
// Negative tests: (zp_comp_ndim > 0, zp_comp_acc == nullptr) bad-pair handling.
//
// Background: cache_or_compute_zp_compensation() in lowoha_cache.hpp sets
// zp_comp_ndim BEFORE allocating the compensation buffer (lines 102, 152,
// 188) and returns nullptr on aligned_alloc failure WITHOUT resetting
// ndim (lines 117, 157, 193). The matmul kernel caller in
// aocl_kernel.cpp:514-535 does not reset ndim either, so a bad pair
// (ndim > 0, acc == nullptr) flows straight into create_dlp_post_op().
//
// Inside create_dlp_post_op() the signature, total_ops, and the
// bias_count / matrix_add_count bumps branch on zp_comp_ndim alone,
// while the cold-path slot wiring and patch_mutable_fields branch on
// (zp_comp_ndim && zp_comp_acc). Without normalization at the API
// boundary, the bad pair would share a cache slot with a subsequent
// (ndim > 0, acc != null) call on the same weight, and the hit-path
// patch would overwrite the cached user-bias slot with zp_comp_acc,
// producing silently wrong results.
//
// The fix (aocl_postop.cpp, top of create_dlp_post_op) is a single-line
// normalization: `if (!zp_comp_acc) zp_comp_ndim = 0;`. These tests pin
// the resulting invariant directly against the lowest-level API rather
// than relying on an OOM injection through the public matmul path.
// =============================================================================

class TestPostopCacheZpCompNegative : public ::testing::Test {
 protected:
  void SetUp() override    {
    // Both test bodies in this fixture assert pointer-identity of the
    // dlp_metadata_t returned by create_dlp_post_op (`EXPECT_EQ(md_bad_1d,
    // md_zero)` style), which can only hold on a cache hit. With the
    // cache disabled, every call gets its own freshly-allocated holder
    // and the assertions fail with a malloc-reuse-dependent false
    // positive/negative. Skip cleanly under the kill switch.
    if (!zendnnl::common::is_postop_cache_enabled()) {
      GTEST_SKIP() << "AOCL DLP post-op metadata cache is disabled "
                   "(ZENDNNL_ENABLE_POSTOP_CACHE=0); this fixture "
                   "asserts cache-slot identity which requires hits.";
    }
    clear_aocl_postop_metadata_cache();
  }
  void TearDown() override { clear_aocl_postop_metadata_cache(); }

  // Build a minimal matmul_params that takes the INT8 code path inside
  // create_dlp_post_op (so total_ops/bias_count/matrix_add_count
  // accounting actually runs) without wiring any user bias, post-op,
  // or quant scale buffers — keeping the test focused on zp_comp
  // accounting alone.
  static matmul_params make_int8_minimal_params() {
    matmul_params p{};
    p.dtypes.src = data_type_t::s8;
    p.dtypes.wei = data_type_t::s8;
    p.dtypes.dst = data_type_t::bf16;
    return p;
  }
};

// (ndim=1 OR 2, acc=null) MUST canonicalize to the same cache slot as
// (ndim=0, acc=null) on the same key. We assert pointer-identity of the
// dlp_metadata_t returned by create_dlp_post_op (the per-thread LRU
// returns &h->metadata from a single cached holder for each unique
// signature). Without the normalization fix, the bad-pair calls would
// hash to a distinct signature (because compute_postop_signature feeds
// raw zp_comp_ndim into the hash) and miss the cache, returning a
// different holder — the EXPECT_EQ below would fail.
TEST_F(TestPostopCacheZpCompNegative, NullAccCollapsesToNoZpCompKey) {
  const matmul_params       params = make_int8_minimal_params();
  const matmul_data_types   dtypes = params.dtypes;

  // weight_storage is never dereferenced by create_dlp_post_op (it is
  // only used as the identity component of the cache key — see the
  // contract in aocl_postop.hpp). Stack storage is safe.
  alignas(int32_t) uint8_t weight_storage[64] = {0};
  const void *weight_ptr = static_cast<const void *>(weight_storage);
  const int N = 8, K = 16, M = 4;
  const auto algo = matmul_algo_t::aocl_dlp;

  dlp_metadata_t *md_zero = create_dlp_post_op(
                              params, /*bias=*/nullptr, dtypes, N, K, M,
                              /*zp_comp_acc=*/nullptr,
                              /*zp_comp_ndim=*/0,
                              algo, weight_ptr);
  ASSERT_NE(md_zero, nullptr)
      << "INT8 path always builds metadata even with zero ops";

  dlp_metadata_t *md_bad_1d = create_dlp_post_op(
                                params, /*bias=*/nullptr, dtypes, N, K, M,
                                /*zp_comp_acc=*/nullptr,
                                /*zp_comp_ndim=*/1,
                                algo, weight_ptr);
  EXPECT_EQ(md_bad_1d, md_zero)
      << "Bad pair (ndim=1, acc=null) must canonicalize to the "
         "(ndim=0, acc=null) cache slot; differing pointers indicate the "
         "signature still hashes raw ndim and the bad pair created a "
         "separate holder";

  dlp_metadata_t *md_bad_2d = create_dlp_post_op(
                                params, /*bias=*/nullptr, dtypes, N, K, M,
                                /*zp_comp_acc=*/nullptr,
                                /*zp_comp_ndim=*/2,
                                algo, weight_ptr);
  EXPECT_EQ(md_bad_2d, md_zero)
      << "Bad pair (ndim=2, acc=null) must canonicalize to the same "
         "cache slot as (ndim=0, acc=null) for the matrix_add path too";

  // Nothing was wired into seq_vector — no bias, no post-ops, no
  // scales, no zp_comp slot (acc was null in every call).
  EXPECT_EQ(md_zero->seq_length, 0);
}

// (ndim=1, acc=null) and (ndim=1, acc=<real>) on the SAME weight MUST
// land in different cache slots — they describe semantically different
// post-op chains (one has a zp_comp BIAS slot, the other does not). If
// they collided, patch_mutable_fields would overwrite the cached
// user-bias slot with zp_comp_acc on the hit path. This is the exact
// silent-miscompute scenario Copilot flagged.
TEST_F(TestPostopCacheZpCompNegative, NullAccDistinctFromRealAccKey) {
  const matmul_params       params = make_int8_minimal_params();
  const matmul_data_types   dtypes = params.dtypes;

  alignas(int32_t) uint8_t weight_storage[64] = {0};
  const void *weight_ptr = static_cast<const void *>(weight_storage);
  // zp_comp_storage is stored by-pointer into the cached metadata's
  // bias[0].bias / matrix_add[0].matrix; lifetime must outlive the
  // create_dlp_post_op call. TearDown clears the cache before this
  // storage goes out of scope, so the dangling-pointer window never
  // opens within the test process.
  int32_t zp_comp_storage[8] = {0};
  const int N = 8, K = 16, M = 4;
  const auto algo = matmul_algo_t::aocl_dlp;

  // Bad pair first: lands on the "no zp_comp" slot after normalization
  // (asserted in the previous test). seq_length == 0.
  dlp_metadata_t *md_bad = create_dlp_post_op(
                             params, /*bias=*/nullptr, dtypes, N, K, M,
                             /*zp_comp_acc=*/nullptr,
                             /*zp_comp_ndim=*/1,
                             algo, weight_ptr);
  ASSERT_NE(md_bad, nullptr);

  // Real pair: signature now hashes ndim=1 (because acc is non-null,
  // normalization is a no-op). Different signature => different cache
  // slot => different holder => different metadata pointer.
  dlp_metadata_t *md_real_1d = create_dlp_post_op(
                                 params, /*bias=*/nullptr, dtypes, N, K, M,
                                 /*zp_comp_acc=*/zp_comp_storage,
                                 /*zp_comp_ndim=*/1,
                                 algo, weight_ptr);
  ASSERT_NE(md_real_1d, nullptr);
  EXPECT_NE(md_real_1d, md_bad)
      << "Bad pair (ndim=1, acc=null) and real pair (ndim=1, acc!=null) "
         "must NOT share a cache entry — sharing would let "
         "patch_mutable_fields on the hit path overwrite the cached "
         "user-bias slot with zp_comp_acc and silently miscompute";
  EXPECT_EQ(md_bad->seq_length, 0);
  ASSERT_EQ(md_real_1d->seq_length, 1)
      << "Real-pair 1D zp_comp call must wire exactly one BIAS seq op";
  ASSERT_NE(md_real_1d->bias, nullptr);
  EXPECT_EQ(md_real_1d->bias[0].bias, zp_comp_storage)
      << "Real-pair 1D zp_comp must point at the caller-supplied buffer";

  // Symmetric coverage for the 2D (matrix_add) path. Distinct signature
  // from both ndim=0 and ndim=1.
  dlp_metadata_t *md_real_2d = create_dlp_post_op(
                                 params, /*bias=*/nullptr, dtypes, N, K, M,
                                 /*zp_comp_acc=*/zp_comp_storage,
                                 /*zp_comp_ndim=*/2,
                                 algo, weight_ptr);
  ASSERT_NE(md_real_2d, nullptr);
  EXPECT_NE(md_real_2d, md_bad);
  EXPECT_NE(md_real_2d, md_real_1d)
      << "ndim=1 and ndim=2 are semantically different chains and must "
         "occupy distinct cache slots";
  ASSERT_EQ(md_real_2d->seq_length, 1)
      << "Real-pair 2D zp_comp call must wire exactly one MATRIX_ADD seq op";
  ASSERT_NE(md_real_2d->matrix_add, nullptr);
  EXPECT_EQ(md_real_2d->matrix_add[0].matrix, zp_comp_storage)
      << "Real-pair 2D zp_comp must point at the caller-supplied buffer";
}

// =============================================================================
// Kill-switch coverage: ZENDNNL_ENABLE_POSTOP_CACHE.
//
// The kill switch has three layers and each is pinned by a distinct test:
//
//   Layer                                         | Test
//   ---------------------------------------------+---------------------------
//   1. Env var spelling -> config_postop_cache_t | TestPostopCacheEnableFlag
//      (handled in config_manager_t::            |
//       set_env_postop_cache_config)             |
//   2. cache.clear() -> next call returns a      | TestPostopCacheClearMechanism
//      fresh holder (the gate's effect inside   |
//      create_dlp_post_op)                       |
//   3. End-to-end cold-path-on-every-call still  | TestPostopCache.LifecycleClear
//      produces correct results across the full | (parameterized across the
//      (dtype, shape, algo) matrix              |  full coverage matrix)
//
// We can't toggle the env var mid-process to flip is_postop_cache_enabled()
// — that helper samples once via `static const bool cached` (see
// zendnnl_global.hpp), which is exactly the property the runtime-overhead
// argument depends on. Layer 1 therefore tests a LOCAL config_manager_t
// instance (not the global one used by is_postop_cache_enabled), which
// reads the env on every config() call. That's enough to lock the
// parser; layers 2 and 3 prove the rest of the chain still does what
// the flag asks for.
// =============================================================================

/** @brief Plumbing for the ZENDNNL_ENABLE_POSTOP_CACHE env var into
 *         config_manager_t::config_postop_cache_t::enable. Uses a
 *         local config_manager_t instance so the env var is sampled
 *         on every test (the global instance's value is locked by
 *         is_postop_cache_enabled's static-const cache and cannot be
 *         re-sampled mid-process). */
class TestPostopCacheEnableFlag : public ::testing::Test {
 protected:
  // Restore env to the pre-test value (or unset if it wasn't set) so
  // adjacent tests aren't perturbed. Captures in SetUp, restores in
  // TearDown so a test body that throws still leaves the env clean.
  void SetUp() override {
    const char *prev = std::getenv(kEnvVar);
    if (prev) {
      had_prev_  = true;
      prev_val_  = prev;
    } else {
      had_prev_  = false;
      prev_val_.clear();
    }
  }

  void TearDown() override {
    if (had_prev_) {
      ::setenv(kEnvVar, prev_val_.c_str(), /*overwrite=*/1);
    } else {
      ::unsetenv(kEnvVar);
    }
  }

  // Build a fresh config_manager_t and run its env-var-driven config
  // path with ZENDNNL_ENABLE_POSTOP_CACHE set (or unset) to `value`.
  // Returns the resulting enable flag. Each call is independent of
  // every other call (and of the process-global config_manager).
  static bool enable_for(const char *value_or_null) {
    if (value_or_null) {
      ::setenv(kEnvVar, value_or_null, /*overwrite=*/1);
    } else {
      ::unsetenv(kEnvVar);
    }
    zendnnl::common::config_manager_t local_cfg;
    local_cfg.config();
    return local_cfg.get_postop_cache_config().enable;
  }

  static constexpr const char *kEnvVar = "ZENDNNL_ENABLE_POSTOP_CACHE";

 private:
  bool        had_prev_{false};
  std::string prev_val_;
};

// Default (env var unset): cache is enabled. This is the post-soak
// contract integrators rely on — opt-out to disable. The default
// was previously false during the initial validation soak and was
// flipped to true once the cache had been validated across
// integrator releases.
TEST_F(TestPostopCacheEnableFlag, DefaultIsEnabled) {
  EXPECT_TRUE(enable_for(/*value_or_null=*/nullptr))
      << "Unset env var must leave cache enabled (default contract)";
}

// Every spelling we claim to support in the env-setter must actually
// disable the cache. With the current default of true these all
// exercise the real disable branch in the parser (rather than the
// no-op same-as-default path); the test pins that the parser
// recognizes these spellings as enable=false rather than falling
// into the unrecognized-value branch (where the cache would stay on).
TEST_F(TestPostopCacheEnableFlag, EnvSpellingsThatDisable) {
  for (const char *v : {"0", "false", "off", "no",
                        "FALSE", "Off", "No", "fAlSe"}) {
    EXPECT_FALSE(enable_for(v))
        << "Env var spelling '" << v << "' should disable cache";
  }
}

// Every spelling we claim to support in the env-setter must actually
// enable the cache. With the current default of true these match the
// no-op same-as-default path, but the test still pins that the
// parser recognizes these spellings as enable=true rather than
// falling into the unrecognized-value branch (where, if the default
// ever flips again, behavior could drift).
TEST_F(TestPostopCacheEnableFlag, EnvSpellingsThatEnable) {
  for (const char *v : {"1", "true", "on", "yes",
                        "TRUE", "On", "Yes", "tRuE"}) {
    EXPECT_TRUE(enable_for(v))
        << "Env var spelling '" << v << "' should enable cache";
  }
}

// Unrecognized values must NOT silently disable the cache. This is a
// safety contract: a typo like ZENDNNL_ENABLE_POSTOP_CACHE=flase must
// leave the default untouched, not surprise the user by turning the
// cache off. The parser is documented to behave this way; this test
// pins it so the behavior can't drift.
TEST_F(TestPostopCacheEnableFlag, GarbageLeavesDefault) {
  for (const char *v : {"ture", "enabled", "disabled", "2", "-1",
                        "  ", "1.0", "yesno"}) {
    EXPECT_TRUE(enable_for(v))
        << "Unrecognized env var spelling '" << v
        << "' must leave default (true) untouched, not silently disable";
  }
}

// Empty string: a common mistake (e.g. `export FOO=` from a shell).
// The parser treats empty as unrecognized -> default preserved.
TEST_F(TestPostopCacheEnableFlag, EmptyValueLeavesDefault) {
  EXPECT_TRUE(enable_for(""))
      << "Empty env var value must leave default (true) untouched";
}

/** @brief Mechanism the kill switch relies on: cold-path build
 *         inserts a holder (size goes 0 -> 1); a same-key call hits
 *         and does NOT insert again (size stays 1, returned pointer
 *         matches); clear_aocl_postop_metadata_cache() empties the
 *         cache (size goes back to 0); the next same-key call rebuilds
 *         (size 0 -> 1 again).
 *
 *         The cache size signal is independent of malloc address-reuse
 *         — when clear() frees the holder and the next create call
 *         immediately allocates one of the same size, glibc's free list
 *         routinely hands back the SAME address, so pointer-distinctness
 *         is not a reliable post-clear signal. Cache size is. We still
 *         keep the pointer-equality check on the hit path, where it IS
 *         reliable (the holder is never freed between the two calls).
 *
 *         When ZENDNNL_ENABLE_POSTOP_CACHE=0 the gate at the top of
 *         create_dlp_post_op clears the cache on every call, which
 *         is mechanically the same as the manual clear() below. We
 *         can't flip the env var mid-process (see the
 *         TestPostopCacheEnableFlag rationale above) but the underlying
 *         mechanism is identical, so testing it directly here gives us
 *         the kill-switch coverage we need without the subprocess fork
 *         dance. */
class TestPostopCacheClearMechanism : public ::testing::Test {
 protected:
  void SetUp() override    {
    // The single test body in this fixture asserts pointer-identity
    // (`EXPECT_EQ(md_hit, md_first)`) on the hot path -- exactly the
    // invariant the kill switch's cache.clear() gate at the top of
    // create_dlp_post_op breaks (every call becomes a cold rebuild,
    // returning a freshly-allocated holder at a malloc-dependent
    // address). The test is mechanically about the cache being on,
    // so skip cleanly when it isn't.
    if (!zendnnl::common::is_postop_cache_enabled()) {
      GTEST_SKIP() << "AOCL DLP post-op metadata cache is disabled "
                   "(ZENDNNL_ENABLE_POSTOP_CACHE=0); this fixture "
                   "asserts hot-path pointer identity which requires "
                   "the cache.";
    }
    clear_aocl_postop_metadata_cache();
  }
  void TearDown() override { clear_aocl_postop_metadata_cache(); }
};

TEST_F(TestPostopCacheClearMechanism, ClearProducesFreshHolderForSameKey) {
  // Use the tiny INT8 setup: the s8 weight path always allocates a
  // holder even with seq_length=0, so we always get a non-null
  // metadata pointer back. Same trick TestPostopCacheZpCompNegative
  // uses.
  matmul_params params{};
  params.dtypes.src = data_type_t::s8;
  params.dtypes.wei = data_type_t::s8;
  params.dtypes.dst = data_type_t::bf16;

  alignas(int32_t) uint8_t weight_storage[64] = {0};
  const void *weight_ptr = static_cast<const void *>(weight_storage);
  const int N = 8, K = 16, M = 4;
  const auto algo = matmul_algo_t::aocl_dlp;

  // SetUp() already cleared; pin that as our starting point so a
  // future regression that leaves entries behind across tests is
  // caught here instead of leaking into the assertions below.
  ASSERT_EQ(get_aocl_postop_metadata_cache_size(), 0u)
      << "Test fixture SetUp() must leave cache empty";

  // Cold path: cache miss -> allocate + build + insert. Size 0 -> 1.
  dlp_metadata_t *md_first = create_dlp_post_op(
                               params, /*bias=*/nullptr, params.dtypes,
                               N, K, M, /*zp_comp_acc=*/nullptr,
                               /*zp_comp_ndim=*/0, algo, weight_ptr);
  ASSERT_NE(md_first, nullptr);
  EXPECT_EQ(get_aocl_postop_metadata_cache_size(), 1u)
      << "Cold-path build must insert exactly one holder into the cache";

  // Hot path: same key, no clear in between. Must return the same
  // holder pointer AND must NOT add a second entry. If the cache
  // silently behaved like the disabled path (i.e. always cleared
  // before lookup), the pointer would differ and a fresh entry
  // would be inserted (still size==1 because the just-cleared cache
  // was empty before insert, but the pointer would differ).
  dlp_metadata_t *md_hit = create_dlp_post_op(
                             params, /*bias=*/nullptr, params.dtypes,
                             N, K, M, /*zp_comp_acc=*/nullptr,
                             /*zp_comp_ndim=*/0, algo, weight_ptr);
  EXPECT_EQ(md_hit, md_first)
      << "Same key + cache enabled (default) MUST return same holder; "
         "differing pointers indicate the cache is being unconditionally "
         "cleared or the lookup is broken";
  EXPECT_EQ(get_aocl_postop_metadata_cache_size(), 1u)
      << "Cache hit must NOT insert a second holder";

  // Manual clear: emulates exactly what the disabled-mode gate does
  // at the top of create_dlp_post_op. Size must drop to 0; this is
  // the load-bearing assertion for the kill switch — if clear()
  // didn't actually drop the holder, ZENDNNL_ENABLE_POSTOP_CACHE=0
  // would silently keep returning the cached holder forever and the
  // disabled mode would behave identically to enabled mode.
  clear_aocl_postop_metadata_cache();
  EXPECT_EQ(get_aocl_postop_metadata_cache_size(), 0u)
      << "clear_aocl_postop_metadata_cache() MUST drop the holder. "
         "Non-zero size here means the disabled-mode gate cannot "
         "force cold-path rebuilds and the kill switch is broken";

  // Next same-key call must miss the (just-emptied) cache and run
  // the cold path again, reinserting a holder. Cannot assert pointer
  // distinctness here — glibc's free list routinely recycles the
  // just-freed address, and that's a malloc property unrelated to
  // the cache. Size going 0 -> 1 is the correct observable signal.
  dlp_metadata_t *md_after_clear = create_dlp_post_op(
                                     params, /*bias=*/nullptr, params.dtypes,
                                     N, K, M, /*zp_comp_acc=*/nullptr,
                                     /*zp_comp_ndim=*/0, algo, weight_ptr);
  ASSERT_NE(md_after_clear, nullptr);
  EXPECT_EQ(get_aocl_postop_metadata_cache_size(), 1u)
      << "Post-clear cold-path call MUST reinsert into the cache";

  // Second call after clear, no further clear: must hit the freshly
  // inserted holder. Size still 1 (no second insert), returned
  // pointer matches the post-clear pointer (cache hit on the fresh
  // holder). Confirms clear+rebuild leaves the cache in a healthy
  // state for subsequent lookups.
  dlp_metadata_t *md_hit_after_clear = create_dlp_post_op(
                                         params, /*bias=*/nullptr, params.dtypes,
                                         N, K, M, /*zp_comp_acc=*/nullptr,
                                         /*zp_comp_ndim=*/0, algo, weight_ptr);
  EXPECT_EQ(md_hit_after_clear, md_after_clear)
      << "Second post-clear call MUST hit the rebuilt holder; "
         "differing pointers indicate the rebuild path is not "
         "inserting into the cache";
  EXPECT_EQ(get_aocl_postop_metadata_cache_size(), 1u)
      << "Second post-clear hit must NOT insert another holder";
}
