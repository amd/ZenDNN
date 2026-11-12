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

#include "gdn_lowoha.hpp"

#include <chrono>
#include <memory>
#include <numeric>
#include <random>

namespace zendnnl {
namespace benchdnn {
namespace gdn {

namespace {

// The batch description of one call: decodes first, then the prefills, one
// shuffled pool slot per sequence.
struct gdn_batch {
    std::vector<int32_t> cu_seqlens;
    std::vector<int32_t> state_indices;
    std::unique_ptr<bool[]> has_initial_state;
};

gdn_batch make_batch(const GdnConfig &cfg) {
    const int64_t B = cfg.batch();
    gdn_batch batch;
    batch.cu_seqlens.resize(B + 1);
    batch.cu_seqlens[0] = 0;
    for (int64_t i = 0; i < B; ++i) {
        const int64_t len = i < cfg.num_decodes ? 1 : cfg.prefill_len;
        batch.cu_seqlens[i + 1]
                = batch.cu_seqlens[i] + static_cast<int32_t>(len);
    }
    batch.state_indices.resize(B);
    std::iota(batch.state_indices.begin(), batch.state_indices.end(), 0);
    std::mt19937 rng(7);
    std::shuffle(batch.state_indices.begin(), batch.state_indices.end(), rng);
    batch.has_initial_state.reset(new bool[B]);
    for (int64_t i = 0; i < B; ++i) {
        batch.has_initial_state[i] = cfg.has_initial_state;
    }
    return batch;
}

void populate_gdn_params(gdn_params &p, const GdnConfig &cfg,
        const gdn_tensor_sizes &s, gdn_tensors &t, const gdn_batch &batch) {
    p.batch = cfg.batch();
    p.num_decodes = cfg.num_decodes;
    p.num_tokens = cfg.num_tokens();
    p.cu_seqlens = batch.cu_seqlens.data();
    p.has_initial_state = batch.has_initial_state.get();

    p.num_k_heads = cfg.num_k_heads;
    p.num_v_heads = cfg.num_v_heads;
    p.head_k_dim = cfg.head_k_dim;
    p.head_v_dim = cfg.head_v_dim;
    p.conv_width = cfg.conv_width;
    p.silu = true;

    const int64_t D = s.conv_dim(), HV = cfg.num_v_heads, V = cfg.head_v_dim;
    p.qkv_stride = D;
    p.a_stride = HV;
    p.b_stride = HV;
    p.out_stride_t = HV * V;
    p.out_stride_h = V;
    // SD conv pool [slot][tap][channel].
    p.conv_state_slot_stride = s.conv_taps() * D;
    p.conv_state_tap_stride = D;
    p.conv_state_dim_stride = 1;
    p.ssm_state_slot_stride = HV * cfg.head_k_dim * V;
    p.ssm_state_head_stride = cfg.head_k_dim * V;
    p.state_layout = cfg.state_layout;
    p.num_state_slots = s.slots;

    p.src_dt = cfg.src_dt;
    p.state_dt = cfg.state_dt;
    p.a_log_dt = data_type_t::f32;
    p.dt_bias_dt = cfg.src_dt;
    if (cfg.fused_norm) {
        p.norm_weight = t.norm_weight.get_raw_handle_const();
        p.norm_weight_dt = data_type_t::f32;
        p.z = t.z.get_raw_handle_const();
        p.z_stride_t = HV * V;
        p.z_stride_h = V;
    }
    p.prefill_algo = cfg.prefill_algo;
    p.kernel = cfg.kernel;
    p.num_threads = cfg.num_threads;
}

} // anonymous namespace

int gdn_lowoha_benchdnn(const std::vector<GdnConfig> &configs,
        std::vector<std::pair<GdnConfig, TimingStats>> &gdn_results,
        const global_options &options, size_t cache_size) {

    for (const auto &cfg : configs) {
        try {
            std::string reason;
            if (!isSupportedGdnConfig(cfg, reason)) {
                commonlog_error("Skipping unsupported GDN config: ", reason);
                log_benchmark_failure(cfg);
                continue;
            }

            const gdn_tensor_sizes sizes {cfg.num_k_heads, cfg.num_v_heads,
                    cfg.head_k_dim, cfg.head_v_dim, cfg.conv_width,
                    cfg.num_tokens(), cfg.batch(), cfg.src_dt, cfg.state_dt,
                    cfg.fused_norm};
            tensor_factory_t tensor_factory;
            gdn_tensors t;
            if (create_gdn_tensors(tensor_factory, sizes, t) != OK) {
                testlog_error("create_gdn_tensors failed");
                log_benchmark_failure(cfg);
                continue;
            }
            const gdn_batch batch = make_batch(cfg);

            gdn_params params;
            populate_gdn_params(params, cfg, sizes, t, batch);

            const void *qkv = t.mixed_qkv.get_raw_handle_const();
            const void *a = t.a.get_raw_handle_const();
            const void *b = t.b.get_raw_handle_const();
            const void *A_log = t.A_log.get_raw_handle_const();
            const void *dt_bias = t.dt_bias.get_raw_handle_const();
            const void *conv_w = t.conv_weight.get_raw_handle_const();
            const void *conv_b = t.conv_bias.get_raw_handle_const();
            void *conv_state = t.conv_state.get_raw_handle_unsafe();
            void *ssm_state = t.ssm_state.get_raw_handle_unsafe();
            void *out = t.output.get_raw_handle_unsafe();
            auto run = [&]() {
                return gdn_direct(qkv, a, b, A_log, dt_bias, conv_w, conv_b,
                        conv_state, ssm_state, batch.state_indices.data(), out,
                        params);
            };

            bool failed = false;
            for (int i = 0; i < cfg.warmup_iters; ++i) {
                if (run() != status_t::success) {
                    testlog_error("LOWOHA: gdn_direct failed during warm-up.");
                    failed = true;
                    break;
                }
            }

            double elapsed_ms = 0.0;
            for (int i = 0; i < cfg.iters && !failed; ++i) {
                if (options.cache_mode == CacheMode::COLD) {
                    flush_cache(cache_size);
                }
                const auto start = std::chrono::high_resolution_clock::now();
                const status_t status = run();
                const auto end = std::chrono::high_resolution_clock::now();
                if (status != status_t::success) {
                    testlog_error(
                            "LOWOHA: gdn_direct failed during benchmark "
                            "iterations.");
                    failed = true;
                    break;
                }
                elapsed_ms += std::chrono::duration<double, std::milli>(
                        end - start)
                                      .count();
            }
            if (failed) {
                log_benchmark_failure(cfg);
                continue;
            }

            TimingStats time_stats;
            time_stats.total_time_ms = elapsed_ms;
            gdn_results.emplace_back(cfg, time_stats);
        } catch (const exception_t &ex) {
            std::cout << ex.what() << std::endl;
            return NOT_OK;
        }
    }
    return OK;
}

} // namespace gdn
} // namespace benchdnn
} // namespace zendnnl
