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

#include "matmul_native/native_brgemm_test_utils.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <random>
#include <set>
#include <sstream>

#include "common/bfloat16.hpp"
#include "common/float16.hpp"
#include "common/zendnnl_compat.hpp"
#include "common/zendnnl_global.hpp"
#include "group_matmul/moe_test_utils.hpp"
#include "gtest_utils.hpp"
#include "lowoha_operators/common/omp_thread_control.hpp"
#include "lowoha_operators/matmul/group_matmul/common/group_matmul_parallel_common.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_brgemm_looper.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_weight_cache.hpp"
#include "lowoha_operators/matmul/matmul_native/native_matmul.hpp"
#include "nlohmann/json.hpp"

namespace native_brgemm_test {

using zendnnl::common::bfloat16_t;
using zendnnl::common::float16_t;
using zendnnl::common::matmul_config_t;

buffer_t make_buffer(size_t bytes) {
    const size_t rounded = (std::max<size_t>(bytes, 1) + 63) / 64 * 64;
    auto *p = static_cast<uint8_t *>(zendnnl_aligned_alloc(64, rounded));
    if (p != nullptr) std::memset(p, 0, rounded);
    buffer_t b;
    b.mem = std::shared_ptr<uint8_t>(
            p, [](uint8_t *q) { zendnnl_aligned_free(q); });
    b.bytes = bytes;
    return b;
}

buffer_t kept_until_exit(buffer_t b) {
    static std::mutex m;
    static std::vector<std::shared_ptr<uint8_t>> kept;
    std::lock_guard<std::mutex> lock(m);
    kept.push_back(b.mem);
    return b;
}

size_t dt_bytes(data_type_t dt) {
    switch (dt) {
        case data_type_t::f32:
        case data_type_t::s32: return 4;
        case data_type_t::bf16:
        case data_type_t::f16: return 2;
        case data_type_t::s8:
        case data_type_t::u8: return 1;
        default: return 0;
    }
}

float load_f32(const void *base, data_type_t dt, size_t i) {
    switch (dt) {
        case data_type_t::f32: return static_cast<const float *>(base)[i];
        case data_type_t::bf16:
            return static_cast<float>(static_cast<const bfloat16_t *>(base)[i]);
        case data_type_t::f16:
            return static_cast<float>(static_cast<const float16_t *>(base)[i]);
        case data_type_t::s32:
            return static_cast<float>(static_cast<const int32_t *>(base)[i]);
        case data_type_t::s8:
            return static_cast<float>(static_cast<const int8_t *>(base)[i]);
        case data_type_t::u8:
            return static_cast<float>(static_cast<const uint8_t *>(base)[i]);
        default: return NAN;
    }
}

void store_f32(void *base, data_type_t dt, size_t i, float v) {
    switch (dt) {
        case data_type_t::f32: static_cast<float *>(base)[i] = v; break;
        case data_type_t::bf16:
            static_cast<bfloat16_t *>(base)[i] = bfloat16_t(v);
            break;
        case data_type_t::f16:
            static_cast<float16_t *>(base)[i] = float16_t(v);
            break;
        case data_type_t::s32:
            static_cast<int32_t *>(base)[i] = static_cast<int32_t>(v);
            break;
        case data_type_t::s8:
            static_cast<int8_t *>(base)[i] = static_cast<int8_t>(v);
            break;
        case data_type_t::u8:
            static_cast<uint8_t *>(base)[i] = static_cast<uint8_t>(v);
            break;
        default: break;
    }
}

buffer_t random_buffer(
        data_type_t dt, size_t n, uint32_t seed, float lo, float hi) {
    buffer_t b = make_buffer(n * dt_bytes(dt));
    std::mt19937 gen(seed);
    const bool integer = dt == data_type_t::s8 || dt == data_type_t::u8
            || dt == data_type_t::s32;
    std::uniform_real_distribution<float> real(lo, hi);
    std::uniform_int_distribution<int> whole(
            static_cast<int>(lo), static_cast<int>(hi));
    for (size_t i = 0; i < n; ++i)
        store_f32(b.get(), dt, i,
                integer ? static_cast<float>(whole(gen)) : real(gen));
    return b;
}

int call_t::out_cols() const {
    return gated == mm::matmul_gated_act_t::none ? N : N / 2;
}

zendnnl::lowoha::matmul::native::GemmDescriptor call_descriptor(
        const call_t &c) {
    zendnnl::lowoha::matmul::native::GemmDescriptor desc;
    desc.M = c.M;
    desc.N = c.N;
    desc.K = c.K;
    desc.lda = c.lda_or_tight();
    desc.ldb = c.ldb_or_tight();
    desc.ldc = c.ldc_or_tight();
    desc.transA = c.transA;
    desc.transB = c.transB;
    desc.alpha = c.alpha;
    desc.beta = c.beta;
    desc.src_dt = c.params.dtypes.src;
    desc.wei_dt = c.params.dtypes.wei;
    desc.dst_dt = c.params.dtypes.dst;
    desc.bias_dt = c.params.dtypes.bias;
    desc.bias = c.bias;
    desc.num_threads = c.params.num_threads;
    desc.gated_act = c.gated;
    return desc;
}

status_t run_matmul(const call_t &c) {
    mm::matmul_batch_params_t batch;
    mm::matmul_gated_act_params gated;
    gated.act = c.gated;
    return mm::matmul_direct('r', c.transA, c.transB, c.M, c.N, c.K, c.alpha,
            c.src, c.lda_or_tight(), c.wei, c.ldb_or_tight(), c.bias, c.beta,
            c.dst, c.ldc_or_tight(), c.is_weights_const, batch, c.params,
            c.gated == mm::matmul_gated_act_t::none ? nullptr : &gated);
}

status_t refusal_of(const call_t &c, std::string *reason) {
    mm::matmul_batch_params_t batch;
    mm::matmul_gated_act_params gated;
    gated.act = c.gated;
    const char *why = nullptr;
    const status_t st = mm::native_brgemm_refusal('r', c.transA, c.transB, c.M,
            c.N, c.K, c.alpha, c.src, c.lda_or_tight(), c.wei, c.ldb_or_tight(),
            c.bias, c.beta, c.dst, c.ldc_or_tight(), c.is_weights_const, batch,
            c.params,
            c.gated == mm::matmul_gated_act_t::none ? nullptr : &gated, &why);
    if (reason != nullptr) *reason = why != nullptr ? why : "";
    return st;
}

::testing::AssertionResult refused_with(const call_t &c, status_t st) {
    std::string reason;
    const status_t refused = refusal_of(c, &reason);
    if (refused != st)
        return ::testing::AssertionFailure()
                << "refused with status " << static_cast<int>(refused) << " ("
                << reason << "), not " << static_cast<int>(st);
    mm::matmul_batch_params_t batch;
    mm::matmul_gated_act_params gated;
    gated.act = c.gated;
    run_matmul_direct('r', c.transA, c.transB, c.M, c.N, c.K, c.alpha, c.src,
            c.lda_or_tight(), c.wei, c.ldb_or_tight(), c.bias, c.beta, c.dst,
            c.ldc_or_tight(), c.is_weights_const, batch, c.params,
            c.gated == mm::matmul_gated_act_t::none ? nullptr : &gated);
    if (!native_refusal_checked())
        return ::testing::AssertionFailure() << "not checked as refused";
    return ::testing::AssertionSuccess() << reason;
}

namespace {

bool per_row(const mm::matmul_quantization_params_t::matmul_quant_t &q, int M) {
    return q.buff != nullptr && !q.dims.empty() && q.dims[0] == M;
}

bool per_channel(
        const mm::matmul_quantization_params_t::matmul_quant_t &q, int N) {
    int64_t n = 1;
    for (int64_t d : q.dims)
        n *= d;
    return q.buff != nullptr && n == N;
}

void row_of(mm::matmul_quantization_params_t::matmul_quant_t &q, int m) {
    q.buff = static_cast<const uint8_t *>(q.buff)
            + static_cast<size_t>(m) * dt_bytes(q.dt);
    q.dims = {1, 1};
}

} // namespace

status_t run_reference(const call_t &c, const void *raw_wei, void *dst) {
    call_t r = c;
    r.wei = raw_wei;
    r.dst = dst;
    r.lda = c.lda_or_tight();
    r.ldb = c.ldb_or_tight();
    r.params.lowoha_algo = matmul_algo_t::reference;
    r.params.mem_format_b = 'n';
    r.gated = mm::matmul_gated_act_t::none;
    r.params.weight_cache_type = 0;
    r.params.num_threads = 0;
    if (c.gated != mm::matmul_gated_act_t::none) {
        r.beta = 0.0f;
        r.params.postop_.clear();
        r.params.dtypes.dst = data_type_t::f32;
        r.ldc = c.N;
    } else {
        r.ldc = c.ldc_or_tight();
    }
    // The reference reads the bias as f32; matmul_direct refuses an f16 bias
    // on a CPU without AVX512-FP16, so the same values go in widened to f32.
    buffer_t wide_bias;
    if (r.bias != nullptr && r.params.dtypes.bias == data_type_t::f16
            && !zendnnl::common::zendnnl_platform_info()
                        .get_avx512_f16_status()) {
        wide_bias = make_buffer(static_cast<size_t>(c.N) * 4);
        for (int n = 0; n < c.N; ++n)
            store_f32(wide_bias.get(), data_type_t::f32, n,
                    load_f32(r.bias, data_type_t::f16, n));
        r.bias = wide_bias.get();
        r.params.dtypes.bias = data_type_t::f32;
    }
    // The reference reads one source zero point per call, and matmul_direct
    // takes a per-token source scale only from a symmetric s8 / bf16 / f32
    // source with an f32 / bf16 destination and per-channel weight scales:
    // other calls with per-row source parameters run row by row.
    const auto &qp = c.params.quant_params;
    const data_type_t src_dt = r.params.dtypes.src;
    const data_type_t dst_dt = r.params.dtypes.dst;
    const bool token_scale_ok = qp.src_zp.buff == nullptr
            && (src_dt == data_type_t::s8 || src_dt == data_type_t::bf16
                    || src_dt == data_type_t::f32)
            && (dst_dt == data_type_t::f32 || dst_dt == data_type_t::bf16)
            && per_channel(qp.wei_scale, c.N);
    const bool by_row = c.M > 1
            && ((per_row(qp.src_scale, c.M) && !token_scale_ok)
                    || per_row(qp.src_zp, c.M)
                    || (c.params.dynamic_quant
                            && c.params.dtypes.compute == data_type_t::u8));
    if (!by_row) return run_matmul(r);
    const size_t a_es = dt_bytes(c.params.dtypes.src);
    const size_t c_es = dt_bytes(r.params.dtypes.dst);
    for (int m = 0; m < c.M; ++m) {
        call_t row = r;
        row.M = 1;
        row.src = static_cast<const uint8_t *>(r.src)
                + (c.transA ? static_cast<size_t>(m)
                            : static_cast<size_t>(m) * r.lda)
                        * a_es;
        row.dst = static_cast<uint8_t *>(dst)
                + static_cast<size_t>(m) * r.ldc * c_es;
        auto &q = row.params.quant_params;
        if (per_row(q.src_scale, c.M)) row_of(q.src_scale, m);
        if (per_row(q.src_zp, c.M)) row_of(q.src_zp, m);
        for (mm::matmul_post_op &po : row.params.postop_) {
            if (po.buff == nullptr || po.dims.size() != 2 || po.dims[0] != c.M)
                continue;
            po.buff = static_cast<uint8_t *>(po.buff)
                    + static_cast<size_t>(m) * po.dims[1] * dt_bytes(po.dtype);
            po.dims[0] = 1;
        }
        const status_t st = run_matmul(row);
        if (st != status_t::success) return st;
    }
    return status_t::success;
}

status_t run_group_one_op(
        const call_t &c, mm::grp_matmul_gated_act_t act, void *dst) {
    mm::grp_matmul_gated_act_params gated;
    gated.act = act;
    return mm::group_matmul_direct({'r'}, {c.transA}, {c.transB}, {c.M}, {c.N},
            {c.K}, {c.alpha}, {c.src}, {c.lda_or_tight()}, {c.wei},
            {c.ldb_or_tight()}, {c.bias}, {c.beta}, {dst}, {c.N},
            {c.is_weights_const}, {c.params}, nullptr,
            act == mm::grp_matmul_gated_act_t::none ? nullptr : &gated);
}

void gated_act_ref(const float *pre, int M, int N, mm::matmul_gated_act_t act,
        float *out) {
    const int I = N / 2;
    for (int m = 0; m < M; ++m) {
        const float *row = pre + static_cast<size_t>(m) * N;
        for (int i = 0; i < I; ++i) {
            float v = 0.0f;
            if (act == mm::matmul_gated_act_t::swiglu_oai_mul) {
                const float g = std::max(-7.0f, std::min(row[2 * i], 7.0f));
                const float u = std::max(-7.0f, std::min(row[2 * i + 1], 7.0f));
                v = (1.0f + u) * g / (1.0f + std::exp(-1.702f * g));
            } else {
                const float g = row[i];
                float a = 0.0f;
                if (act == mm::matmul_gated_act_t::silu_and_mul)
                    a = g / (1.0f + std::exp(-g));
                else if (act == mm::matmul_gated_act_t::gelu_tanh_and_mul)
                    a = 0.5f * g
                            * (1.0f
                                    + std::tanh(0.7978845608f
                                            * (g + 0.044715f * g * g * g)));
                else
                    a = 0.5f * g * (1.0f + std::erf(g * 0.70710678118f));
                v = a * row[I + i];
            }
            out[static_cast<size_t>(m) * I + i] = v;
        }
    }
}

lru_capacity_pin_t::lru_capacity_pin_t(uint32_t capacity)
    : prev_(matmul_config_t::instance().get_lru_cache_capacity()) {
    matmul_config_t::instance().set_lru_cache_capacity(capacity);
}

lru_capacity_pin_t::~lru_capacity_pin_t() {
    matmul_config_t::instance().set_lru_cache_capacity(prev_);
}

algo_pin_t::algo_pin_t(matmul_algo_t algo)
    : prev_(matmul_config_t::instance().get_algo()) {
    // resolve_matmul_backend() latches the algo it first sees for the whole process;
    // it must latch the process setting, not the pinned one.
    (void)mm::resolve_matmul_backend();
    matmul_config_t::instance().set_algo(static_cast<int32_t>(algo));
}

algo_pin_t::~algo_pin_t() {
    matmul_config_t::instance().set_algo(prev_);
}

struct ck_route_pin_t::impl_t {
    algo_pin_t algo {matmul_algo_t::none};
    bool route_prev = matmul_config_t::instance().get_custom_kernel_route();
    moe_test_utils::AlgoEnvGuard grouped_auto {0};
    moe_test_utils::CustomKernelOverride custom {true};
    moe_test_utils::CustomKernelInt8Override custom_int8 {true};
    moe_test_utils::NTileStrategyOverride n_tile {2};
    impl_t() { matmul_config_t::instance().set_custom_kernel_route(true); }
    ~impl_t() {
        matmul_config_t::instance().set_custom_kernel_route(route_prev);
    }
};

ck_route_pin_t::ck_route_pin_t() : impl_(new impl_t) {}
ck_route_pin_t::~ck_route_pin_t() = default;

bool default_route_reachable() {
    return mm::resolve_matmul_backend() == matmul_algo_t::aocl_dlp_blocked;
}

size_t native_layouts(const void *weight) {
    return mm::native::native_weight_layouts(weight);
}

bool native_selected(const mm::matmul_params &p) {
    return mm::native::native_brgemm_selected(p);
}

ck_engagement_t::ck_engagement_t() {
    mm::test_api::s_last_group_matmul_direct_execution_path.store(
            nullptr, std::memory_order_relaxed);
    mm::test_api::s_capture_execution_path.store(
            true, std::memory_order_release);
}

ck_engagement_t::~ck_engagement_t() {
    mm::test_api::s_capture_execution_path.store(
            false, std::memory_order_release);
}

bool ck_engagement_t::engaged() const {
    const char *mode
            = mm::test_api::s_last_group_matmul_direct_execution_path.load(
                    std::memory_order_relaxed);
    return mode != nullptr && std::strstr(mode, "custom") != nullptr;
}

namespace {

enum class mode_t { band, steps, bits };

::testing::AssertionResult compare(const void *got, int ldg, const void *ref,
        int ldr, data_type_t dt, int rows, int cols, mode_t mode, float rtol,
        float abs_bound, int steps) {
    const size_t esz = dt_bytes(dt);
    size_t bad = 0;
    double worst = 0.0;
    std::ostringstream first;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const size_t ig = static_cast<size_t>(r) * ldg + c;
            const size_t ie = static_cast<size_t>(r) * ldr + c;
            const float g = load_f32(got, dt, ig);
            const float e = load_f32(ref, dt, ie);
            const float err = std::fabs(g - e);
            bool ok = false;
            if (mode == mode_t::bits) {
                ok = std::memcmp(static_cast<const uint8_t *>(got) + ig * esz,
                             static_cast<const uint8_t *>(ref) + ie * esz, esz)
                        == 0;
            } else if (mode == mode_t::steps) {
                ok = err <= static_cast<float>(steps);
            } else {
                ok = !std::isnan(g) && err <= abs_bound + rtol * std::fabs(e);
            }
            if (ok) continue;
            if (bad++ == 0)
                first << " first at (" << r << ", " << c << "): got " << g
                      << ", expected " << e;
            if (!std::isnan(err)) worst = std::max(worst, double(err));
        }
    }
    if (bad == 0) return ::testing::AssertionSuccess();
    ::testing::AssertionResult res = ::testing::AssertionFailure();
    res << bad << " of " << static_cast<size_t>(rows) * cols << " elements ";
    if (mode == mode_t::band)
        res << "outside abs " << abs_bound << " + rtol " << rtol;
    else if (mode == mode_t::steps)
        res << "more than " << steps << " steps apart";
    else
        res << "not bit-identical";
    return res << ", worst |err| " << worst << ";" << first.str();
}

} // namespace

float comparison_abs_bound(uint64_t k, float epsilon, float alpha, float beta,
        bool is_low_precision, bool is_quant, bool is_dst_u8) {
    constexpr int C = 20;
    constexpr int P = 15;
    constexpr int scale_factor = 4;
    if (is_dst_u8) return 1.0f;
    const float abs_alpha = std::fabs(alpha);
    const float abs_beta = std::fabs(beta);
    const float k_f = static_cast<float>(k);
    constexpr float kSymQuantPanelK = 2048.f;
    constexpr float kSymQuantPanelAbs = 1.5e-3f;
    const float n_sym_quant_panels
            = is_quant ? std::ceil(std::max(k_f, 1.f) / kSymQuantPanelK) : 0.f;
    float gemm_accum = 0.0f;
    if (is_quant) {
        gemm_accum = k_f + static_cast<float>(P);
    } else if (is_low_precision) {
        gemm_accum = k_f;
    } else {
        gemm_accum
                = (static_cast<float>(C)
                          + std::log2(k_f) / static_cast<float>(scale_factor))
                        * k_f
                + static_cast<float>(P);
    }
    const float beta_scale
            = (is_low_precision || is_quant) ? 1.0f : static_cast<float>(P);
    return (abs_alpha * gemm_accum + abs_beta * beta_scale) * epsilon
            + n_sym_quant_panels * kSymQuantPanelAbs;
}

::testing::AssertionResult within_band(const void *got, int ldg,
        const void *ref, int ldr, data_type_t dt, int rows, int cols,
        float rtol, float abs_bound) {
    return compare(got, ldg, ref, ldr, dt, rows, cols, mode_t::band, rtol,
            abs_bound, 0);
}

::testing::AssertionResult within_steps(const void *got, int ldg,
        const void *ref, int ldr, data_type_t dt, int rows, int cols,
        int steps) {
    const bool integer = dt == data_type_t::s8 || dt == data_type_t::u8
            || dt == data_type_t::s32;
    return compare(got, ldg, ref, ldr, dt, rows, cols,
            integer && steps > 0 ? mode_t::steps : mode_t::bits, 0.0f, 0.0f,
            steps);
}

// ── Settings read once per process ──────────────────────────────────────

namespace {

namespace nat = zendnnl::lowoha::matmul::native;

/// Set in the environment of expect_in_child's children.
constexpr const char *kChildVariable = "ZENDNNL_GTEST_NATIVE_CHILD";
/// The file of child_buffers_t::publish, for the children.
constexpr const char *kBuffersVariable = "ZENDNNL_GTEST_NATIVE_BUFFERS";

/// The temporary files that exist; a child process, which exits from the
/// middle of a test, removes them at its exit.
class temp_files_t {
public:
    static temp_files_t &instance() {
        static temp_files_t files;
        return files;
    }
    void add(const std::string &path) {
        std::lock_guard<std::mutex> lock(mutex_);
        paths_.insert(path);
    }
    void remove(const std::string &path) {
        std::lock_guard<std::mutex> lock(mutex_);
        paths_.erase(path);
        std::remove(path.c_str());
    }
    ~temp_files_t() {
        for (const std::string &p : paths_)
            std::remove(p.c_str());
    }

private:
    std::mutex mutex_;
    std::set<std::string> paths_;
};

} // namespace

nat::native_plan_source_t top_source(const nat::native_plan_t &plan) {
    for (const nat::native_plan_source_t s :
            {nat::native_plan_source_t::override,
                    nat::native_plan_source_t::table,
                    nat::native_plan_source_t::rules}) {
        if ((plan.sources & (1u << static_cast<unsigned>(s))) != 0) return s;
    }
    return nat::native_plan_source_t::heuristic;
}

std::vector<std::string> native_rule_switches() {
    std::vector<std::string> out;
    size_t count = 0;
    const nat::native_config_key_t *keys = nat::native_config_keys(count);
    for (size_t i = 0; i < count; ++i)
        if (keys[i].type == nat::native_key_type_t::int32 && keys[i].min == 0
                && keys[i].max == 1)
            out.push_back(keys[i].name);
    return out;
}

std::vector<std::string> native_enum_values(nat::native_key_type_t type) {
    switch (type) {
        case nat::native_key_type_t::schedule:
            return {"auto", "slice", "row_groups", "blocks"};
        case nat::native_key_type_t::traversal: return {"m_inner", "n_inner"};
        case nat::native_key_type_t::kc_split: return {"budget", "equal"};
        default: return {};
    }
}

std::string hw_match_text(const nat::native_hw_t &hw) {
    const nat::UarchParams plan = nat::native_plan_uarch(hw);
    const int domain = plan.l3_domain_cores > 0 ? plan.l3_domain_cores
                                                : plan.family_l3_cores;
    nlohmann::ordered_json m = nlohmann::ordered_json::object();
    const auto count = [&m](const char *key, int64_t v) {
        if (v > 0) m[key] = v;
    };
    if (!hw.vendor.empty()) m["vendor"] = hw.vendor;
    if (hw.family >= 0) m["family"] = hw.family;
    if (hw.model >= 0) m["model"] = hw.model;
    if (hw.stepping >= 0) m["stepping"] = hw.stepping;
    nlohmann::ordered_json isa = nlohmann::ordered_json::array();
    if (hw.uarch.avx512f) isa.push_back("avx512f");
    if (hw.uarch.avx512bf16) isa.push_back("avx512bf16");
    if (hw.uarch.avx512vnni) isa.push_back("avx512vnni");
    m["isa"] = isa;
    count("l1d_kib", hw.uarch.l1d_detected ? hw.uarch.l1d_bytes / 1024 : 0);
    count("l2_kib", hw.uarch.l2_detected ? hw.uarch.l2_bytes / 1024 : 0);
    count("l3_kib", hw.uarch.l3_detected ? hw.uarch.l3_domain_bytes / 1024 : 0);
    count("l3_cores", domain);
    count("smt", hw.smt);
    count("l3_threads", hw.l3_domain_threads);
    count("cores", hw.physical_cores);
    count("logical_cpus", hw.logical_cpus);
    return m.dump();
}

env_pin_t::env_pin_t(const char *name, const char *value) : name_(name) {
    const char *old = std::getenv(name);
    had_ = old != nullptr;
    if (had_) old_ = old;
    if (value != nullptr)
        setenv(name, value, 1);
    else
        unsetenv(name);
}

env_pin_t::~env_pin_t() {
    if (had_)
        setenv(name_.c_str(), old_.c_str(), 1);
    else
        unsetenv(name_.c_str());
}

void expect_in_child(const std::vector<env_setting_t> &settings,
        const std::function<int()> &check, const std::string &what) {
    testing::FLAGS_gtest_death_test_style = "threadsafe";
    std::vector<std::unique_ptr<env_pin_t>> pins;
    pins.push_back(std::make_unique<env_pin_t>(kChildVariable, "1"));
    for (const env_setting_t &s : settings)
        pins.push_back(std::make_unique<env_pin_t>(s.first, s.second));
    EXPECT_EXIT(
            std::exit(check() == 0 ? 0 : 1), ::testing::ExitedWithCode(0), "")
            << what;
}

bool in_child() {
    return std::getenv(kChildVariable) != nullptr;
}

temp_file_t::temp_file_t(const std::string &name, const std::string &bytes) {
    static std::atomic<unsigned> count {0};
    path_ = ::testing::TempDir() + "native_"
            + std::to_string(std::random_device {}()) + "_"
            + std::to_string(count.fetch_add(1)) + "_" + name;
    temp_files_t::instance().add(path_);
    std::ofstream out(path_, std::ios::binary);
    out << bytes;
}

temp_file_t::~temp_file_t() {
    temp_files_t::instance().remove(path_);
}

child_buffers_t::child_buffers_t() {
    const char *path = in_child() ? std::getenv(kBuffersVariable) : nullptr;
    if (path == nullptr) return;
    std::ifstream in(path, std::ios::binary);
    uint64_t n = 0;
    in.read(reinterpret_cast<char *>(&n), sizeof(n));
    for (uint64_t i = 0; in && i < n; ++i) {
        uint64_t bytes = 0;
        in.read(reinterpret_cast<char *>(&bytes), sizeof(bytes));
        buffer_t b = make_buffer(static_cast<size_t>(bytes));
        in.read(reinterpret_cast<char *>(b.get()),
                static_cast<std::streamsize>(bytes));
        buffers_.push_back(b);
    }
    if (!in) buffers_.clear();
}

child_buffers_t::~child_buffers_t() = default;

void child_buffers_t::publish(const std::vector<buffer_t> &buffers) {
    buffers_ = buffers;
    std::string bytes;
    const auto put = [&bytes](uint64_t v) {
        bytes.append(reinterpret_cast<const char *>(&v), sizeof(v));
    };
    put(buffers.size());
    for (const buffer_t &b : buffers) {
        put(b.bytes);
        bytes.append(reinterpret_cast<const char *>(b.get()), b.bytes);
    }
    pin_.reset();
    file_ = std::make_unique<temp_file_t>("buffers", bytes);
    pin_ = std::make_unique<env_pin_t>(kBuffersVariable, file_->path().c_str());
}

} // namespace native_brgemm_test
