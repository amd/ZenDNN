/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#include "lowoha_sdpa_ck_common.hpp"
#include "lowoha_sdpa_ck.hpp"

#include "common/config_params.hpp"
#include "common/zendnnl_global.hpp"

#include <atomic>
#include <cstdlib>

namespace zendnnl {
namespace lowoha {
namespace sdpa {
namespace ck {

namespace {

std::atomic<int> g_bf16_override {-1}, g_int8_override {-1};
std::atomic<bool> g_count {false};
std::atomic<uint64_t> g_bf16_calls {0}, g_int8_calls {0};

/// Boolean variable `name`; `fallback` when unset or not a boolean.
bool env_bool(const char *name, bool fallback) {
    bool value = fallback;
    zendnnl::common::parse_cache_bool(std::getenv(name), value);
    return value;
}

mode_t from_int(int v) {
    return v == 1 ? mode_t::gemm : mode_t::off;
}

/// Mode from the boolean variable `env`, on unless it is false; off without
/// `isa` (named `need`), with a warning only when `env` is set.
mode_t env_mode(const char *env, bool isa, const char *need, const char *on) {
    if (!env_bool(env, true)) return mode_t::off;
    if (!isa) {
        if (std::getenv(env))
            zendnnl::error_handling::apilog_warning(
                    env, " ignored: needs ", need);
        return mode_t::off;
    }
    zendnnl::error_handling::apilog_info(env, ": ", on);
    return mode_t::gemm;
}

void count(std::atomic<uint64_t> &calls) {
    if (g_count.load(std::memory_order_relaxed))
        calls.fetch_add(1, std::memory_order_relaxed);
}

} // namespace

mode_t bf16_mode() {
    static const bool isa = [] {
        const auto &pi = zendnnl::common::zendnnl_platform_info();
        return pi.get_avx512_bf16_status() && pi.get_avx512_bw_vl_status();
    }();
    const int ovr = g_bf16_override.load(std::memory_order_relaxed);
    if (ovr >= 0) return isa ? from_int(ovr) : mode_t::off;
    static const mode_t env = env_mode("ZENDNNL_SDPA_BF16_CK", isa,
            "AVX512-BF16, AVX512-BW and AVX512-VL",
            "flash BF16 GEMMs on the custom kernel");
    return env;
}

mode_t int8_mode() {
    static const bool isa = [] {
        const auto &pi = zendnnl::common::zendnnl_platform_info();
        return pi.get_avx512_vnni_status() && pi.get_avx512_bw_vl_status();
    }();
    const int ovr = g_int8_override.load(std::memory_order_relaxed);
    if (ovr >= 0) return isa ? from_int(ovr) : mode_t::off;
    static const mode_t env = env_mode("ZENDNNL_SDPA_INT8_CK", isa,
            "AVX512-VNNI, AVX512-BW and AVX512-VL",
            "flash INT8 GEMMs on the custom kernel");
    return env;
}

void count_bf16_call() {
    count(g_bf16_calls);
}

void count_int8_call() {
    count(g_int8_calls);
}

namespace test_api {
void set_bf16_mode(int value) {
    g_bf16_override.store(value, std::memory_order_relaxed);
}

void set_int8_mode(int value) {
    g_int8_override.store(value, std::memory_order_relaxed);
}

void count_calls(bool on) {
    g_bf16_calls.store(0, std::memory_order_relaxed);
    g_int8_calls.store(0, std::memory_order_relaxed);
    g_count.store(on, std::memory_order_relaxed);
}

uint64_t bf16_gemm_calls() {
    return g_bf16_calls.load(std::memory_order_relaxed);
}

uint64_t int8_gemm_calls() {
    return g_int8_calls.load(std::memory_order_relaxed);
}
} // namespace test_api

} // namespace ck
} // namespace sdpa
} // namespace lowoha
} // namespace zendnnl
