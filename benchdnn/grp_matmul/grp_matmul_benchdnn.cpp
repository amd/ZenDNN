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

/// Group MatMul (MoE) benchdnn driver.
///
/// Input file format (CSV, one line per config):
///   num_ops, M, K, N, iters, src_dt:wei_dt:dst_dt, is_weights_const, warmup[, moe_topk[, gated_act[, N_down]]]
///
/// M can be a single int (all experts same) or colon-separated per-expert:
///   8, 4, 4096, 14336, 200, bf16:bf16:bf16, true, 50              <- plain GEMM
///   8, 4, 4096, 14336, 200, bf16:bf16:bf16, true, 50, 2           <- MoE topk=2
///   8, 4, 4096, 28672, 200, bf16:bf16:bf16, true, 50, 2, 1, 4096  <- full fused MoE block
///
/// moe_topk (optional, default 0): 0 = no MoE post-op, >0 = fused weighted-reduce.
/// gated_act (optional, default 0): 0 = off, 1 = silu_and_mul, 2 = gelu_and_mul,
///   3 = swiglu_oai_mul.  Requires N even.
/// N_down (optional, default 0): 0 = no fused down_proj, >0 = fused
///   Op1(gate+up) → activation → Op2(down_proj) with this output width.
///   K_down = N/2 (dim after activation), output = [M, N_down] per expert.
///
/// Env vars:
///   ZENDNNL_GRP_MATMUL_ALGO=0|1|2|3|4|5 - select parallel strategy
///     0=auto, 1=sequential, 2=flat_ccd_m_tile, 3=flat_ccd_n_tile, 4=multilevel, 5=per_expert
///   ZENDNNL_MATMUL_ALGO=N         - select kernel (default: aocl_dlp_blocked)

#include "grp_matmul_benchdnn.hpp"
#include "grp_matmul_utils.hpp"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "common/data_types.hpp"
#include "common/error_status.hpp"
#include "lowoha_operators/matmul/lowoha_common.hpp"
#include "lowoha_operators/matmul/lowoha_matmul.hpp"
#include "utils/benchdnn_utils.hpp"

namespace zendnnl {
namespace benchdnn {
namespace grp_matmul {

using namespace zendnnl::lowoha::matmul;
using zendnnl::common::data_type_t;
using zendnnl::common::size_of;

// ── Run one config ──────────────────────────────────────────────────────

static bool run_config(const GrpMatmulConfig &cfg, std::ostream &csv,
                       [[maybe_unused]] size_t cache_size) {
    const int n = cfg.num_ops;
    const size_t src_elem = size_of(cfg.src_dt);
    const size_t wei_elem = size_of(cfg.wei_dt);
    const size_t dst_elem = size_of(cfg.dst_dt);

    std::vector<AlignedBuffer> A(n), B(n), C(n);
    for (int i = 0; i < n; ++i) {
        int Mi = cfg.M_per_op[i];
        A[i].alloc(Mi * cfg.K * src_elem);
        B[i].alloc(cfg.K * cfg.N * wei_elem);
        C[i].alloc(Mi * cfg.N * dst_elem);
        fill_buffer(A[i].ptr, Mi * cfg.K, cfg.src_dt, 42 + i * 3);
        fill_buffer(B[i].ptr, cfg.K * cfg.N, cfg.wei_dt, 137 + i * 7);
        std::memset(C[i].ptr, 0, Mi * cfg.N * dst_elem);
    }

    std::vector<char> layout(n, 'r');
    std::vector<bool> transA(n, false), transB(n, false);
    std::vector<int> Mv(cfg.M_per_op), Nv(n, cfg.N), Kv(n, cfg.K);
    std::vector<float> alpha(n, 1.0f), beta(n, 0.0f);
    std::vector<int> lda(n), ldb(n, cfg.N), ldc(n, cfg.N);
    for (int i = 0; i < n; ++i) lda[i] = cfg.K;
    std::vector<bool> wconst(n, cfg.is_weights_const);

    std::vector<const void *> src_ptrs(n), wei_ptrs(n), bias_ptrs(n, nullptr);
    std::vector<void *> dst_ptrs(n);
    for (int i = 0; i < n; ++i) {
        src_ptrs[i] = A[i].ptr;
        wei_ptrs[i] = B[i].ptr;
        dst_ptrs[i] = C[i].ptr;
    }

    std::vector<matmul_params> params(n);
    for (int i = 0; i < n; ++i) {
        params[i].dtypes.src = cfg.src_dt;
        params[i].dtypes.wei = cfg.wei_dt;
        params[i].dtypes.dst = cfg.dst_dt;
        params[i].dtypes.bias = data_type_t::none;
    }

    // Build MoE post-op when moe_topk > 0.
    const bool moe_enabled = (cfg.moe_topk > 0);
    const int topk = cfg.moe_topk;
    int total_M = 0;
    for (int i = 0; i < n; ++i) total_M += cfg.M_per_op[i];
    const int num_tokens = moe_enabled ? (total_M / topk) : 0;
    const int num_slots = num_tokens * topk;

    std::vector<float> moe_weights;
    std::vector<const void *> moe_row_ptrs_vec;
    AlignedBuffer moe_output_buf;
    group_matmul_moe_postop_params moe;
    group_matmul_moe_postop_params *moe_ptr = nullptr;

    if (moe_enabled) {
        if (total_M % topk != 0) {
            std::cerr << "ERROR: total_M=" << total_M
                      << " not divisible by moe_topk=" << topk << std::endl;
            return false;
        }

        moe_weights.assign(static_cast<size_t>(num_slots),
                           1.f / static_cast<float>(topk));
        moe_output_buf.alloc(static_cast<size_t>(num_tokens) * cfg.N * dst_elem);

        moe_row_ptrs_vec.resize(static_cast<size_t>(num_slots));
        int slot = 0;
        for (int e = 0; e < n; ++e) {
            auto *base = static_cast<const char *>(C[e].ptr);
            for (int j = 0; j < cfg.M_per_op[e]; ++j) {
                moe_row_ptrs_vec[static_cast<size_t>(slot)] =
                    base + static_cast<size_t>(j) * cfg.N * dst_elem;
                ++slot;
            }
        }

        moe.num_tokens = num_tokens;
        moe.topk = topk;
        moe.output = moe_output_buf.ptr;
        moe.ldc_output = cfg.N;
        moe.topk_weights = moe_weights.data();
        moe.skip_weighted = false;
        moe.row_ptrs = moe_row_ptrs_vec.data();
        moe_ptr = &moe;
    }

    // Gated activation: controlled by the gated_act field in the CSV.
    grp_matmul_gated_act_params act;
    act.act = static_cast<grp_matmul_gated_act_t>(cfg.gated_act);
    grp_matmul_gated_act_params *act_ptr =
        (cfg.gated_act > 0) ? &act : nullptr;

    // Fused down_proj: when N_down > 0, build fused_moe params.
    // The MoE-semantically correct flow is Op1 → gated activation → Op2.
    // Running fused with gated_act=0 feeds Op2 the raw first-half of
    // Op1's output (K_down = N/2) — a valid numerical operation but NOT
    // equivalent to MoE.  Warn so the config author sees it, but don't
    // reject (the grp_matmul gtests exercise this path as a reference).
    const bool fused_enabled = (cfg.N_down > 0);
    if (fused_enabled && cfg.gated_act == 0) {
        std::cerr << "WARNING: N_down=" << cfg.N_down
                  << " with gated_act=0 — Op2 will read raw first-half "
                     "columns of Op1's output (not MoE-semantic).  "
                     "Set gated_act>0 for Op1 → activation → Op2.\n";
    }
    if (fused_enabled && (cfg.N & 1) != 0) {
        std::cerr << "ERROR: N=" << cfg.N
                  << " must be even when N_down > 0 (K_down = N/2).  "
                     "Skipping config.\n";
        return false;
    }
    const int dim = cfg.N / 2;

    std::vector<AlignedBuffer> B_down(n), C_down(n);
    grp_matmul_fused_moe_params fused;
    grp_matmul_fused_moe_params *fused_ptr = nullptr;

    if (fused_enabled) {
        fused.down_weight.resize(n);
        fused.N_down.resize(n, cfg.N_down);
        fused.ldb_down.resize(n, cfg.N_down);
        fused.bias_down.resize(n, nullptr);
        fused.dst_down.resize(n);
        fused.ldc_down.resize(n, cfg.N_down);

        for (int i = 0; i < n; ++i) {
            B_down[i].alloc(static_cast<size_t>(dim) * cfg.N_down * wei_elem);
            C_down[i].alloc(static_cast<size_t>(cfg.M_per_op[i]) * cfg.N_down * dst_elem);
            fill_buffer(B_down[i].ptr, static_cast<size_t>(dim) * cfg.N_down,
                        cfg.wei_dt, 200 + i * 11);
            std::memset(C_down[i].ptr, 0,
                        static_cast<size_t>(cfg.M_per_op[i]) * cfg.N_down * dst_elem);
            fused.down_weight[i] = B_down[i].ptr;
            fused.dst_down[i] = C_down[i].ptr;
        }
        fused_ptr = &fused;

        // When fused, moe_postop D = N_down (not N).
        if (moe_enabled) {
            moe.ldc_output = cfg.N_down;
            moe_output_buf.alloc(static_cast<size_t>(num_tokens) * cfg.N_down * dst_elem);
            moe.output = moe_output_buf.ptr;
            // Rebuild row_ptrs to point into C_down instead of C.
            int slot = 0;
            for (int e = 0; e < n; ++e) {
                auto *base = static_cast<const char *>(C_down[e].ptr);
                for (int j = 0; j < cfg.M_per_op[e]; ++j) {
                    moe_row_ptrs_vec[static_cast<size_t>(slot)] =
                        base + static_cast<size_t>(j) * cfg.N_down * dst_elem;
                    ++slot;
                }
            }
        }
    }

    // Warmup
    for (int w = 0; w < cfg.warmup; ++w) {
        auto st = group_matmul_direct(layout, transA, transB, Mv, Nv, Kv, alpha,
                                      src_ptrs, lda, wei_ptrs, ldb, bias_ptrs, beta,
                                      dst_ptrs, ldc, wconst, params,
                                      moe_ptr, act_ptr, fused_ptr);
        if (st != status_t::success) {
            std::cerr << "ERROR: group_matmul_direct failed during warmup"
                      << std::endl;
            return false;
        }
    }

    // Timed iterations
    double min_ms = std::numeric_limits<double>::max();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < cfg.iters; ++it) {
#if COLD_CACHE
        flush_cache(cache_size);
#endif
        auto ti0 = std::chrono::high_resolution_clock::now();
        group_matmul_direct(layout, transA, transB, Mv, Nv, Kv, alpha,
                            src_ptrs, lda, wei_ptrs, ldb, bias_ptrs, beta,
                            dst_ptrs, ldc, wconst, params,
                            moe_ptr, act_ptr, fused_ptr);
        auto ti1 = std::chrono::high_resolution_clock::now();
        double iter_ms = std::chrono::duration<double, std::milli>(ti1 - ti0).count();
        if (iter_ms < min_ms) min_ms = iter_ms;
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double avg_ms = total_ms / cfg.iters;

    double total_flops = 0;
    for (int i = 0; i < n; ++i) {
        // Op1: gate+up GEMM
        total_flops += 2.0 * cfg.M_per_op[i] * cfg.K * cfg.N;
        // Op2: down_proj GEMM (when fused)
        if (fused_enabled)
            total_flops += 2.0 * cfg.M_per_op[i] * dim * cfg.N_down;
    }
    if (moe_enabled) {
        int D_out = fused_enabled ? cfg.N_down : cfg.N;
        total_flops += 2.0 * num_tokens * D_out * topk;
    }
    double gflops_avg = (total_flops / avg_ms) * 1e-6;
    double gflops_peak = (total_flops / min_ms) * 1e-6;

    std::string dtypes = datatypeToStr(cfg.src_dt) + ":" + datatypeToStr(cfg.wei_dt)
                         + ":" + datatypeToStr(cfg.dst_dt);
    std::string m_str = format_M(cfg);

    std::string moe_str = moe_enabled ? "topk=" + std::to_string(topk) : "off";
    std::string fused_str = fused_enabled
        ? "N_down=" + std::to_string(cfg.N_down)
        : "off";

    // Console
    std::cout << std::setw(4) << n << "  "
              << std::setw(12) << m_str << "  "
              << std::setw(6) << cfg.K << "  "
              << std::setw(6) << cfg.N << "  "
              << std::setw(5) << cfg.iters << "  "
              << std::setw(6) << cfg.warmup << "  "
              << std::setw(14) << dtypes << "  "
              << std::setw(10) << moe_str << "  "
              << std::setw(12) << fused_str << "  "
              << std::fixed << std::setprecision(3)
              << std::setw(10) << avg_ms << "  "
              << std::setw(10) << min_ms << "  "
              << std::setprecision(1)
              << std::setw(9) << gflops_avg << "  "
              << std::setw(9) << gflops_peak
              << std::endl;

    // CSV
    csv << n << "," << m_str << "," << cfg.K << "," << cfg.N << ","
        << cfg.iters << "," << cfg.warmup << "," << dtypes << ","
        << (cfg.is_weights_const ? "true" : "false") << ","
        << cfg.moe_topk << "," << cfg.gated_act << "," << cfg.N_down << ","
        << std::fixed << std::setprecision(6)
        << total_ms << "," << avg_ms << "," << min_ms << ","
        << std::setprecision(2) << gflops_avg << "," << gflops_peak << "\n";
    return true;
}

// ── Entry point ─────────────────────────────────────────────────────────

int bench(const std::string &in_filename, const std::string &out_filename,
          [[maybe_unused]] size_t cache_size) {

    std::ifstream infile(in_filename);
    if (!infile.is_open()) {
        commonlog_error("Cannot open input file: ", in_filename);
        return NOT_OK;
    }

    std::vector<GrpMatmulConfig> configs;
    std::string line;
    while (std::getline(infile, line)) {
        GrpMatmulConfig cfg;
        if (parse_config(line, cfg))
            configs.push_back(cfg);
    }
    infile.close();

    if (configs.empty()) {
        commonlog_error("No valid configs in input file");
        return NOT_OK;
    }

    const char *ver_env = std::getenv("ZENDNNL_GRP_MATMUL_ALGO");
    const char *algo_env = std::getenv("ZENDNNL_MATMUL_ALGO");
    const char *omp_env = std::getenv("OMP_NUM_THREADS");

    std::ofstream csv(out_filename);
    csv << "num_ops,M,K,N,iters,warmup,dtypes,is_weights_const,moe_topk,"
           "gated_act,N_down,total_ms,avg_ms,min_ms,GFLOPS_avg,GFLOPS_peak\n";

    std::cout << "================================================================"
              << std::endl;
    std::cout << "  Group MatMul Benchmark" << std::endl;
    std::cout << "  Configs    : " << configs.size() << std::endl;
    std::cout << "  GRP_ALGO   : " << (ver_env ? ver_env : "default(1)") << std::endl;
    std::cout << "  MATMUL_ALGO: " << (algo_env ? algo_env : "default(1)") << std::endl;
    std::cout << "  Threads    : " << (omp_env ? omp_env : "default") << std::endl;
    std::cout << "================================================================"
              << std::endl;
    std::cout << " ops             M       K       N  iters warmup          dtypes"
                 "        moe        fused      avg_ms      min_ms  GFLOPS_a  GFLOPS_p" << std::endl;

    for (const auto &cfg : configs) {
        if (!run_config(cfg, csv, cache_size))
            std::cerr << "  ^ Failed, skipping." << std::endl;
    }

    csv.close();
    std::cout << "\nResults written to " << out_filename << std::endl;
    return OK;
}

} // namespace grp_matmul
} // namespace benchdnn
} // namespace zendnnl
