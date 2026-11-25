/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#ifndef ZENDNNL_GTESTS_MATMUL_NATIVE_NATIVE_BRGEMM_TEST_UTILS_HPP
#define ZENDNNL_GTESTS_MATMUL_NATIVE_NATIVE_BRGEMM_TEST_UTILS_HPP

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "common/op_config.hpp"
#include "lowoha_operators/matmul/group_matmul/group_matmul_direct.hpp"
#include "lowoha_operators/matmul/lowoha_matmul.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_planner.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_tuning_keys.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_hw.hpp"

/// Helpers shared by the matmul_native suites: operand generation, weights in
/// the native VNNI layout, one dense call run through matmul_direct, the
/// reference algorithm or the one-op group call, scoped process-wide pins,
/// what the native BRGEMM keeps for a weight, and result comparisons.
namespace native_brgemm_test {

using zendnnl::common::matmul_algo_t;
using zendnnl::memory::data_type_t;
using zendnnl::memory::status_t;
namespace mm = zendnnl::lowoha::matmul;

/// 64-byte aligned, zero-filled storage; copies share it.
struct buffer_t {
    std::shared_ptr<uint8_t> mem;
    size_t bytes = 0;
    void *get() const { return mem.get(); }
    template <typename T>
    T *as() const {
        return reinterpret_cast<T *>(mem.get());
    }
};
buffer_t make_buffer(size_t bytes);
size_t dt_bytes(data_type_t dt);

/// Storage that ends where an unreadable page starts: an access past its end
/// faults.  An unreadable page also precedes its span of whole pages, which
/// the storage starts at only when its size is a whole number of pages.  On
/// Windows, plain heap storage without the guard pages.  get() is nullptr
/// when the storage could not be had.
class guarded_t {
public:
    explicit guarded_t(size_t bytes);
    ~guarded_t();
    guarded_t(const guarded_t &) = delete;
    guarded_t &operator=(const guarded_t &) = delete;
    uint8_t *get() const { return data_; }
    size_t bytes() const { return bytes_; }

private:
    size_t bytes_ = 0;
    uint8_t *data_ = nullptr;
#ifndef _WIN32
    size_t page_ = 0;
    size_t span_ = 0;
    uint8_t *base_ = nullptr;
#else
    buffer_t heap_;
#endif
};

/// b, whose memory stays allocated until the process exits, so that no
/// later buffer takes its address: the group route keeps entries by the
/// addresses in its weights in caches of its worker threads, which only
/// those threads can clear.
buffer_t kept_until_exit(buffer_t b);

/// n elements of dt drawn uniformly from [lo, hi] (integers for s8 / u8 /
/// s32), the same for the same seed.
buffer_t random_buffer(
        data_type_t dt, size_t n, uint32_t seed, float lo, float hi);
float load_f32(const void *base, data_type_t dt, size_t i);
void store_f32(void *base, data_type_t dt, size_t i, float v);

/// a * b rounded to f32 on its own, as a kernel rounds a product it does not
/// fuse: no compiler may contract it with the add that follows it into one
/// fused multiply-add.
inline float mul_unfused(float a, float b) {
    const volatile float p = a * b;
    return p;
}

/// One dense call; a leading dimension of 0 means the tight one.
struct call_t {
    int M = 0, N = 0, K = 0;
    bool transA = false, transB = false;
    int lda = 0, ldb = 0, ldc = 0;
    float alpha = 1.0f, beta = 0.0f;
    const void *src = nullptr;
    const void *wei = nullptr;
    const void *bias = nullptr;
    void *dst = nullptr;
    bool is_weights_const = true;
    mm::matmul_params params;
    /// Passed to matmul_direct as its gated_act argument.
    mm::matmul_gated_act_t gated = mm::matmul_gated_act_t::none;

    int lda_or_tight() const { return lda ? lda : (transA ? M : K); }
    int ldb_or_tight() const { return ldb ? ldb : (transB ? K : N); }
    /// Output columns: N / 2 for a gated activation, N otherwise.
    int out_cols() const;
    int ldc_or_tight() const { return ldc ? ldc : out_cols(); }
};

status_t run_matmul(const call_t &c);

/// The descriptor of c, as matmul_direct makes it for the native BRGEMM.
zendnnl::lowoha::matmul::native::GemmDescriptor call_descriptor(
        const call_t &c);

/// The route of the plan of the INT8 call of desc and params with decision,
/// as the looper gives it to prepare_int8_call (native_problem_t::route()).
zendnnl::lowoha::matmul::native::native_route_t int8_route_of(
        const zendnnl::lowoha::matmul::native::GemmDescriptor &desc,
        const mm::matmul_params &params,
        const zendnnl::lowoha::matmul::native::native_int8_decision_t
                &decision);

/// The status native_brgemm_refusal gives c (success when the native BRGEMM
/// serves it or does not take it), with reason its reason.
status_t refusal_of(const call_t &c, std::string *reason = nullptr);

/// c is a call the native BRGEMM takes and does not serve, refused with
/// status st; then c runs through run_matmul_direct, which checks
/// matmul_direct's status, dst and error line for it.
::testing::AssertionResult refused_with(const call_t &c, status_t st);

/// The call on lowoha_algo = reference (all threads) with raw weight
/// `raw_wei` into `dst`.  A gated activation is applied in the test
/// (gated_act_ref), so for a gated call the reference writes the f32 [M, N]
/// pre-activation (bias included, no post-ops) and dst must hold M x N f32.
/// With a source zero point per row (static per token, or dynamic asymmetric
/// quantization), or a per-token source scale matmul_direct only takes one
/// row at a time, it runs one row per call.  Without AVX512-FP16 an f16 bias
/// goes in widened to f32.
status_t run_reference(const call_t &c, const void *raw_wei, void *dst);

/// group_matmul_direct with one op and fused activation `act`: dst [M, N]
/// with ldc N; the gated result is in its first N / 2 columns.
status_t run_group_one_op(
        const call_t &c, mm::grp_matmul_gated_act_t act, void *dst);

/// out[M, N / 2] = act over the pairs of pre[M, N] (f32, row stride N), in
/// the order of group_matmul_moe_act.cpp: split halves [gate | up] for
/// silu / gelu (erf) / gelu (tanh), interleaved pairs for swiglu_oai.
void gated_act_ref(
        const float *pre, int M, int N, mm::matmul_gated_act_t act, float *out);

/// Scoped process-wide LRU cache capacity (ZENDNNL_LRU_CACHE_CAPACITY).
class lru_capacity_pin_t {
public:
    explicit lru_capacity_pin_t(uint32_t capacity);
    ~lru_capacity_pin_t();
    lru_capacity_pin_t(const lru_capacity_pin_t &) = delete;
    lru_capacity_pin_t &operator=(const lru_capacity_pin_t &) = delete;

private:
    uint32_t prev_;
};

/// Scoped process-wide matmul algo (ZENDNNL_MATMUL_ALGO); none = unset.
class algo_pin_t {
public:
    explicit algo_pin_t(matmul_algo_t algo);
    ~algo_pin_t();
    algo_pin_t(const algo_pin_t &) = delete;
    algo_pin_t &operator=(const algo_pin_t &) = delete;

private:
    int32_t prev_;
};

/// Scoped settings under which a CK-routable dense call reaches the custom
/// kernel: ALGO unset, ZENDNNL_MATMUL_CUSTOM_KERNEL on, group custom kernel
/// (and its INT8 path) on, grouped AUTO, N-tile strategy 2.
class ck_route_pin_t {
public:
    ck_route_pin_t();
    ~ck_route_pin_t();
    ck_route_pin_t(const ck_route_pin_t &) = delete;
    ck_route_pin_t &operator=(const ck_route_pin_t &) = delete;

private:
    struct impl_t;
    std::unique_ptr<impl_t> impl_;
};

/// True when the default route (and with it the custom kernel) is reachable
/// in this process: resolve_matmul_backend() keeps the algorithm the process started
/// with, so no pin brings the default route back under another one.
bool default_route_reachable();

/// The layouts the native BRGEMM keeps for the weight at weight: those it
/// cached for the pointer, plus one when the buffer holds an image it wrote
/// in place.  A call it served on a constant weight at weight cache mode 1
/// or 2 leaves at least one; a call it declined, or any group-matmul call,
/// leaves none.
size_t native_layouts(const void *weight);

/// Whether matmul_direct takes a call with params p to the native BRGEMM at
/// all: ZENDNNL_MATMUL_ALGO=10, or lowoha_algo == native_brgemm.
bool native_selected(const mm::matmul_params &p);

/// Whether the kernel registry of this process holds the F16_PH kernels: on
/// a CPU with AVX512-FP16, in a build that has them, unless
/// ZENDNNL_MATMUL_NATIVE_F16_ACCUM is f32 (in any case).
bool f16_ph_kernels_registered();

/// Bit mr - 1 set for every row count route has a kernel for at nr, a tile
/// of several slabs at its slab, from the selectors: every kernel its calls
/// take (both sources of INT8, both orders of INT4, the fused order of the
/// group kernels also over source groups).
uint32_t route_kernel_rows(mm::native::native_route_t route, int nr);

/// Records whether the dense calls made while it lives went through the
/// custom kernel (the group gemm-mode capture).
class ck_engagement_t {
public:
    ck_engagement_t();
    ~ck_engagement_t();
    bool engaged() const;
    ck_engagement_t(const ck_engagement_t &) = delete;
    ck_engagement_t &operator=(const ck_engagement_t &) = delete;
};

/// The absolute part of the compare_tensor_2D_matrix band of a K-deep matmul
/// (gtest_utils.cpp, matmul_comparison_abs_bound).
float comparison_abs_bound(uint64_t k, float epsilon, float alpha, float beta,
        bool is_low_precision, bool is_quant, bool is_dst_u8);

/// |got - ref| <= abs_bound + rtol * |ref| over [rows, cols] (the band of
/// compare_tensor_2D_matrix with abs_bound from comparison_abs_bound).
::testing::AssertionResult within_band(const void *got, int ldg,
        const void *ref, int ldr, data_type_t dt, int rows, int cols,
        float rtol, float abs_bound);
/// Integer dst: |got - ref| <= steps.  Otherwise (or steps == 0): identical
/// bits.
::testing::AssertionResult within_steps(const void *got, int ldg,
        const void *ref, int ldr, data_type_t dt, int rows, int cols,
        int steps);

// ── Settings read once per process ──────────────────────────────────────

/// The highest layer of the native planner a plan applied (override, table,
/// rules), or heuristic when it applied none.
zendnnl::lowoha::matmul::native::native_plan_source_t top_source(
        const zendnnl::lowoha::matmul::native::native_plan_t &plan);

/// ZENDNNL_MATMUL_NATIVE_CONFIG keys that turn every rule switch off: the
/// plan of the pins, the narrowing, the row split, the K pass rules and the
/// L3 row-group test, which traffic_groups=0 runs in place of the traffic
/// rule (row groups when the source and one domain's B slabs exceed the
/// l3_pct share of its L3 and the column tiles do not divide over the
/// threads), without the cost model's choice.
inline constexpr const char *kNativeRulesOff
        = "makespan=0,l1_pass=0,traffic_groups=0,bf16_alias=0,"
          "bf16_narrow_n=0,blocks=0,cost_model=0";

/// The rule switches of the native planner's config: its int32 keys of range
/// 0 .. 1, in key order.
std::vector<std::string> native_rule_switches();

/// The value names of an enum key type of the native planner's config, in
/// enum order; empty for the other types.
std::vector<std::string> native_enum_values(
        zendnnl::lowoha::matmul::native::native_key_type_t type);

/// hw's value of every hardware key of a tuning file that hw reports, as a
/// JSON hardware block: the block holds on hw.
std::string hw_match_text(
        const zendnnl::lowoha::matmul::native::native_hw_t &hw);

/// Sets an environment variable while it lives; a nullptr value unsets it.
class env_pin_t {
public:
    env_pin_t(const char *name, const char *value);
    ~env_pin_t();
    env_pin_t(const env_pin_t &) = delete;
    env_pin_t &operator=(const env_pin_t &) = delete;

private:
    std::string name_;
    bool had_ = false;
    std::string old_;
};

/// A variable of the environment of expect_in_child's child, and its value
/// (nullptr: unset).
using env_setting_t = std::pair<const char *, const char *>;

/// Runs check in a child process, a death test of the threadsafe style,
/// whose environment adds settings: a setting the library reads once per
/// process (ZENDNNL_MATMUL_NATIVE_CONFIG, ZENDNNL_MATMUL_NATIVE_TUNING_FILE)
/// takes effect there.  The child runs the test again up to this call, with
/// in_child() true, then check, which returns its failures and describes
/// them on stderr; the expectation holds when there are none.
void expect_in_child(const std::vector<env_setting_t> &settings,
        const std::function<int()> &check, const std::string &what);

/// True in a child process of expect_in_child.
bool in_child();

/// A file of the given bytes in the test temp directory, under a name no
/// other process uses; removed with it, or when a child process exits.
class temp_file_t {
public:
    temp_file_t(const std::string &name, const std::string &bytes);
    ~temp_file_t();
    temp_file_t(const temp_file_t &) = delete;
    temp_file_t &operator=(const temp_file_t &) = delete;
    const std::string &path() const { return path_; }

private:
    std::string path_;
};

/// Buffers the parent computes once for its children: publish() saves them
/// for the children expect_in_child starts afterwards, which read them when
/// they construct this object.
class child_buffers_t {
public:
    /// In a child, the buffers the parent published; in the parent, none.
    child_buffers_t();
    ~child_buffers_t();
    child_buffers_t(const child_buffers_t &) = delete;
    child_buffers_t &operator=(const child_buffers_t &) = delete;
    void publish(const std::vector<buffer_t> &buffers);
    const std::vector<buffer_t> &buffers() const { return buffers_; }

private:
    std::vector<buffer_t> buffers_;
    std::unique_ptr<temp_file_t> file_;
    std::unique_ptr<env_pin_t> pin_;
};

} // namespace native_brgemm_test

#endif // ZENDNNL_GTESTS_MATMUL_NATIVE_NATIVE_BRGEMM_TEST_UTILS_HPP
