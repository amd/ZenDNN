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
///   num_ops, M, K, N, iters, src_dt:wei_dt:dst_dt, is_weights_const, warmup
///       [, moe_topk[, gated_act[, N_down[, use_internal_alloc[,
///          total_experts[, dynamic_quant[, compute_dt]]]]]]]
///
/// M can be a single int (all experts same) or colon-separated per-expert:
///   8, 4, 4096, 14336, 200, bf16:bf16:bf16, true, 50                    <- plain GEMM
///   8, 4, 4096, 14336, 200, bf16:bf16:bf16, true, 50, 2                 <- MoE topk=2
///   8, 4, 4096, 28672, 200, bf16:bf16:bf16, true, 50, 2, 1, 4096        <- fused (caller-alloc)
///   8, 4, 4096, 28672, 200, bf16:bf16:bf16, true, 50, 2, 1, 4096, 1     <- fused (lib-alloc)
///   4, 32, 2880, 5760,  200, bf16:bf16:bf16, true, 50, 4, 3, 2880, 1, 32 <- prepack-extras (4/32)
///   8, 4, 4096, 14336, 200, bf16:s8:bf16,    true, 50, 0, 0, 0,    0, 0, 1, s8  <- DQ-INT8 sym
///   8, 4, 4096, 14336, 200, bf16:s8:bf16,    true, 50, 0, 0, 0,    0, 0, 1, u8  <- DQ-INT8 asym
///
/// moe_topk (optional, default 0): 0 = no MoE post-op, >0 = fused weighted-reduce.
/// gated_act (optional, default 0): 0 = off, 1 = silu_and_mul, 2 = gelu_and_mul,
///   3 = swiglu_oai_mul.  Requires N even.
/// N_down (optional, default 0): 0 = no fused down_proj, >0 = fused
///   Op1(gate+up) → activation → Op2(down_proj) with this output width.
///   K_down follows the gated_act contract (mirrors `op2_k_for_act()` in
///   `group_matmul_parallel_common.hpp`):
///     * gated_act > 0 (silu_and_mul / gelu_and_mul / swiglu_oai_mul):
///         K_down = N/2 (gate × up collapses to half).  Requires N even.
///     * gated_act = 0 (no activation):
///         K_down = N (full Op1 output flows straight into Op2).
///   Op2 output = [M, N_down] per expert in either case.
/// use_internal_alloc (optional, default 0; only when N_down > 0):
///   0 = legacy mode (caller allocates dst[] for Op1 + dst_down[] for Op2);
///   1 = library mode (library allocates Op1 scratch internally, Op2
///       writes back into src[] in place — caller passes empty dst[] and
///       empty fused.dst_down).  Requires lda[i] >= N_down[i] (Op2 reuses
///       the original src row stride for its output writes).
/// total_experts (optional, default 0 → defaults to num_ops):
///   Drives the framework prepack-extras contract: when > num_ops, the
///   driver allocates `total_experts` weight buffers, fills the first
///   `num_ops` as the firing experts plus extras for the remainder, and
///   propagates `params[i].active_matmul = num_ops; total_matmul = total_experts`
///   to the dispatcher.  The library computes only the first `num_ops`
///   GEMMs and pre-warms the cache for all `total_experts` slots.
///   Mirrors the production MoE rotating-experts use case.  Rejected
///   if < num_ops.
/// dynamic_quant (optional, default 0): 0 = bf16 path, 1 = DQ-INT8 path
///   (the N-tile / custom-kernel int8 family).  Sets
///   `params[i].dynamic_quant=true` and populates the per-channel
///   `wei_scale` buffer (length N, dims = {1, N}, f32) on every active
///   expert + every prepack-extras slot.  The per-token `src_scale` is
///   left null; the library's pre-OMP source-reorder hoist allocates and
///   fills it at runtime — same contract as production callers.
///   Requires src:wei:dst == bf16:s8:bf16 and K%4==0; the parser
///   refuses configurations that don't satisfy these preconditions.
/// compute_dt (optional, default s8 when dynamic_quant=1): the int8
///   compute discriminator on `params[i].dtypes.compute`:
///     "s8" → kS8_S8_BF16_SYM (symmetric; no src zero-point)
///     "u8" → kU8_S8_BF16_ASYM (asymmetric; hoist allocates src_zp)
///   Ignored when dynamic_quant=0.
///
/// Env vars (all read by the library, not parsed by this driver):
///   ZENDNNL_GRP_MATMUL_ALGO=0|1|2|3|4|5 - select parallel strategy
///     0=auto, 1=sequential, 2=flat_ccd_m_tile, 3=flat_ccd_n_tile, 4=multilevel, 5=per_expert
///   ZENDNNL_GRP_MATMUL_PREPACK=0|1      - master switch for ahead-of-time weight prepack (default 1)
///   ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL=0|1 - in-house custom kernel for ALGO 3 (default 1)
///   ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_INT8=0|1 - DQ-INT8 sub-kernel toggle inside the master CK (default 1)
///   ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_N_TILE=N - per-thread N-tile floor override (0 = use default)
///   ZENDNNL_GRP_MATMUL_AOCL_STABLE_NTILE=0|1 - stable AOCL DLP cache key under MoE churn (default 1)
///   ZENDNNL_MATMUL_ALGO=N               - select inner kernel (default: aocl_dlp_blocked)
///   ZENDNNL_MATMUL_WEIGHT_CACHE=0|1     - global weight-reorder cache toggle (default 1)

#include "grp_matmul_benchdnn.hpp"
#include "grp_matmul_utils.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
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
#include "utils/perf_counters.hpp"

#ifdef _OPENMP
  #include <omp.h>
#endif

namespace zendnnl {
namespace benchdnn {
namespace grp_matmul {

using namespace zendnnl::lowoha::matmul;
using zendnnl::common::data_type_t;
using zendnnl::common::size_of;

// ── Run one config ──────────────────────────────────────────────────────
//
// `perf_ctrs` (nullable) is owned by the caller (`bench()`) so that the
// underlying `perf_event_open()` fds are created once per process,
// *before* any OpenMP parallel region in benchdnn.  The OMP team
// spawned during the first config's warmup then joins the inheritance
// subtree, and every subsequent config can do `reset/enable/disable/
// read` without needing to re-open (which would otherwise leave the
// already-spawned OMP pool outside the inheritance tree — see
// `doc/perf_counters.md`).  Passing `nullptr` disables perf counter
// collection for this config.

static bool run_config(const GrpMatmulConfig &cfg, std::ostream &csv,
                       const global_options &options,
                       [[maybe_unused]] size_t cache_size,
                       PerfCounterGroup *perf_ctrs) {
    if (cfg.iters <= 0) {
        std::cerr << "ERROR: iters=" << cfg.iters << " must be > 0\n";
        return false;
    }
    const int n = cfg.num_ops;
    // Total expert table size for the framework prepack-extras
    // contract.  When the input file specifies `total_experts > num_ops`,
    // benchdnn allocates weight / K / N / ldb / transB / is_weights_const
    // at `total` (firing experts in [0, n), extras in [n, total)) and
    // sets `params[i].active_matmul = n`, `params[i].total_matmul =
    // total`.  When equal (the default), behaviour is identical to
    // the legacy single-size path.  See `grp_matmul_utils.hpp` for the
    // input-format note and `lowoha_common.hpp` for the library-side
    // contract.
    const int total = cfg.total_experts;
    const size_t src_elem = size_of(cfg.src_dt);
    const size_t wei_elem = size_of(cfg.wei_dt);
    const size_t dst_elem = size_of(cfg.dst_dt);

    // Internal-alloc requires N_down > 0 (parser already guards this).
    const bool internal_alloc = (cfg.use_internal_alloc != 0);

    // Internal-alloc requires matched src/dst precision because Op2
    // writes dst-typed elements into the caller's src buffer.  The
    // library enforces this with an always-on guard; reject upfront
    // here so the benchdnn driver fails fast with a clear message
    // instead of relying on the generic library status_t::failure.
    if (internal_alloc && cfg.src_dt != cfg.dst_dt) {
        std::cerr << "ERROR: use_internal_alloc=1 requires matching src/dst "
                     "dtypes; mixed-precision callers must use legacy mode "
                     "(use_internal_alloc=0). Got src="
                  << datatypeToStr(cfg.src_dt)
                  << ", dst=" << datatypeToStr(cfg.dst_dt) << "\n";
        return false;
    }

    // src row stride.  Internal-alloc requires lda[i] >= N_down[i]
    // because Op2 writes back into src[i] using the original src row
    // stride; otherwise rows would alias.  For the typical MoE case
    // (hidden_size == K == N_down) lda = K naturally satisfies this.
    const int lda_val = internal_alloc
        ? std::max(cfg.K, cfg.N_down)
        : cfg.K;

    // src buffer footprint.  In internal-alloc mode Op2 writes one row
    // every lda_val * dst_elem bytes (NOT N_down * dst_elem — N_down
    // is just the column count written per row, the row pitch tracks
    // lda_val).  With matched precision (enforced above) src_elem ==
    // dst_elem, so this simplifies to lda_val * elem; we keep the
    // explicit max so the formula is robust if matched-precision is
    // ever relaxed.
    const size_t src_row_bytes = internal_alloc
        ? std::max(static_cast<size_t>(cfg.K) * src_elem,
                   static_cast<size_t>(lda_val) * dst_elem)
        : static_cast<size_t>(cfg.K) * src_elem;

    // Weight buffers sized to `total` so the prepack-extras tail
    // (experts in [n, total)) has valid pointers + dimensions for
    // the prepack module to walk.  Activation / dst stays at `n`
    // (only firing experts compute).
    std::vector<AlignedBuffer> A(n), B(total), C(n);
    // Pristine copy of the initial src fill.  In internal-alloc mode
    // every library call rewrites A[i] in place with the Op2 output,
    // so subsequent iterations would otherwise benchmark a different
    // workload (Op1 consuming the prior Op2 result, drifting toward
    // saturated/NaN values).  We snapshot A[] once after the initial
    // fill and `memcpy` back into A[] before every warmup / timed
    // call, outside the chrono::now() bracket, so the restore cost
    // is excluded from sum_iter_ms.  Only allocated when needed
    // (internal_alloc == true) to avoid doubling src footprint in
    // legacy mode where the restore is a no-op.
    std::vector<AlignedBuffer> A_orig;
    std::vector<size_t> A_bytes(n, 0);
    if (internal_alloc) A_orig.resize(n);
    for (int i = 0; i < n; ++i) {
        int Mi = cfg.M_per_op[i];
        A[i].alloc(static_cast<size_t>(Mi) * src_row_bytes);
        B[i].alloc(static_cast<size_t>(cfg.K) * cfg.N * wei_elem);
        // GEMM reads each src row at stride lda_val (which == K in
        // legacy mode and may > K in internal-alloc mode when N_down >
        // K).  When lda_val > K, a tightly-packed [Mi, K] fill leaves
        // GARBAGE between rows that the GEMM would later read at
        // offset r*lda_val, producing NaN/Inf.  Fill the entire
        // allocation including padding so every row has valid data at
        // the GEMM's expected row start.  In matched-precision mode
        // (the only mode internal_alloc supports) src_row_bytes ==
        // lda_val * src_elem, so the byte count divides cleanly.
        const size_t fill_elems = (internal_alloc && lda_val > cfg.K)
            ? static_cast<size_t>(Mi) * src_row_bytes / src_elem
            : static_cast<size_t>(Mi) * cfg.K;
        fill_buffer(A[i].ptr, fill_elems, cfg.src_dt, 42 + i * 3);
        fill_buffer(B[i].ptr, static_cast<size_t>(cfg.K) * cfg.N, cfg.wei_dt,
                    137 + i * 7);
        // Snapshot the pristine src for per-iter restore in internal-
        // alloc mode (see A_orig doc above).
        A_bytes[i] = static_cast<size_t>(Mi) * src_row_bytes;
        if (internal_alloc) {
            A_orig[i].alloc(A_bytes[i]);
            std::memcpy(A_orig[i].ptr, A[i].ptr, A_bytes[i]);
        }
        // Op1 dst (C[i]) only allocated in legacy mode.  In internal-
        // alloc mode the library owns the Op1 scratch.
        if (!internal_alloc) {
            C[i].alloc(static_cast<size_t>(Mi) * cfg.N * dst_elem);
            std::memset(C[i].ptr, 0,
                        static_cast<size_t>(Mi) * cfg.N * dst_elem);
        }
    }

    // Allocate + fill the prepack-extras weight tail (experts in
    // [n, total)).  These weights are NOT consumed by the matmul
    // (the dispatcher iterates only `[0, n)` for compute), but the
    // prepack module DOES walk them via the weight-side metadata
    // vectors (K / N / ldb / transB / is_weights_const sized to
    // `total`) and warms their cache entries up front.  Production
    // MoE inference under the framework `(active, total)` contract
    // routes a different `n` experts on each call but always passes
    // the full `total` weight pool — this loop replicates that
    // pattern for benchmarking.
    for (int i = n; i < total; ++i) {
        B[i].alloc(static_cast<size_t>(cfg.K) * cfg.N * wei_elem);
        fill_buffer(B[i].ptr, static_cast<size_t>(cfg.K) * cfg.N,
                    cfg.wei_dt, 137 + i * 7);
    }

    std::vector<char> layout(n, 'r');
    // transB=true matches the PyTorch / vLLM / TGI nn.Linear weight
    // convention (weight stored as [N, K] row-major; ldb = K).  This
    // is the layout real LLM frameworks pass, so benchdnn benchmarks
    // here exercise the same dispatcher / pack / kernel path as
    // production model-level callers.
    //
    // Vector sizing under the (active, total) contract:
    //
    //   * Weight-side (read by both prepack [0, total) and matmul
    //     [0, n)): weight, K, N, ldb, transB, is_weights_const sized
    //     to `total`.  Extras' values mirror the firing experts'
    //     uniform shape (cfg.K, cfg.N) so the prepack cache key is
    //     well-formed for every entry.
    //
    //   * Input/output-side (read only by matmul [0, n)): M, src,
    //     dst, bias, lda, ldc, alpha, beta, layout, transA stay
    //     sized at `n`.  The dispatcher's relaxed-size validator
    //     accepts any size >= num_ops when active_matmul > 0 (see
    //     `group_matmul_direct.cpp::size_relaxed`).
    //
    //   * params: sized at `n`; only `params[0].active_matmul` /
    //     `params[0].total_matmul` are read by the dispatcher to
    //     decide the contract.  We populate every entry for
    //     defensive symmetry.
    std::vector<bool> transA(n, false), transB(total, true);
    std::vector<int> Mv(cfg.M_per_op), Nv(total, cfg.N), Kv(total, cfg.K);
    std::vector<float> alpha(n, 1.0f), beta(n, 0.0f);
    std::vector<int> lda(n, lda_val), ldb(total, cfg.K), ldc(n, cfg.N);
    std::vector<bool> wconst(total, cfg.is_weights_const);

    std::vector<const void *> src_ptrs(n), wei_ptrs(total), bias_ptrs(n, nullptr);
    // dst_ptrs is empty in internal-alloc mode (the library detects
    // this and runs the in-place reuse path; ldc is also unused).
    std::vector<void *> dst_ptrs;
    if (!internal_alloc) {
        dst_ptrs.resize(n);
    }
    for (int i = 0; i < n; ++i) {
        src_ptrs[i] = A[i].ptr;
        if (!internal_alloc) dst_ptrs[i] = C[i].ptr;
    }
    for (int i = 0; i < total; ++i) {
        wei_ptrs[i] = B[i].ptr;
    }

    std::vector<matmul_params> params(n);
    for (int i = 0; i < n; ++i) {
        params[i].dtypes.src = cfg.src_dt;
        params[i].dtypes.wei = cfg.wei_dt;
        params[i].dtypes.dst = cfg.dst_dt;
        params[i].dtypes.bias = data_type_t::none;
        // Framework prepack-extras contract.  When `total > n`, the
        // dispatcher takes the relaxed-size validator path and the
        // prepack module walks the full `total` weight pool.  When
        // `total == n` (the default for inputs without an explicit
        // total_experts column), this still engages
        // `params[0].active_matmul > 0`, so the dispatcher uses the
        // active-prefix branch that's behaviourally equivalent to
        // legacy.
        params[i].active_matmul = static_cast<uint32_t>(n);
        params[i].total_matmul  = static_cast<uint32_t>(total);
    }

    // ── DQ-INT8 wiring ────────────────────────────────────────────────
    // When `cfg.dynamic_quant` is set (CSV column 14), populate the
    // CK DQ-INT8 contract on every active expert:
    //
    //   * `params[i].dynamic_quant = true` — flips the
    //     `resolve_variant()` truth table from the bf16 family to
    //     the int8 family; mirrored on the auto-select gate
    //     (`check_n_tile_extra` in
    //     `group_matmul_parallel_common.hpp`).
    //   * `params[i].dtypes.compute = s8` (sym) or `u8` (asym),
    //     selected by `cfg.compute_dt`.  Drives the kernel-variant
    //     resolution: s8 → kS8_S8_BF16_SYM, u8 → kU8_S8_BF16_ASYM.
    //   * `params[i].quant_params.wei_scale` populated with a per-
    //     expert per-channel f32 buffer of length `N` (dims = {1, N}).
    //     Filled from `fill_quant_scale_f32` with values clustered
    //     around 1/127, the magnitude a real symmetric-quant
    //     `max(|w|) / 127` would produce for bf16 weights in
    //     [-1, 1].  The wei_scale buffer is required by the gate;
    //     test_algos.cpp::TestGroupMatmulAutoSelectAlgo_DynamicQuant
    //     case 4 confirms the gate refuses with a null wei_scale.
    //   * `params[i].quant_params.src_scale.buff = nullptr` — the
    //     library's pre-OMP source-reorder hoist
    //     (`HoistedSrcQuant` in `n_tile/group_matmul_n_tile.cpp`)
    //     allocates and fills the per-token scale at runtime.
    //     Caller leaves the buffer null and only sets the dims +
    //     dtype so the gate sees a properly-shaped descriptor.
    //
    // wei_scale is only attached to the ACTIVE experts ([0, n)).  The
    // prepack-extras tail ([n, total)) has no `params` slot at all
    // (`params` is sized `n`); prepack warms those tail weights from the
    // `total`-sized weight-side metadata vectors (wei_ptrs / Kv / Nv /
    // ldb / transB / wconst), not from a per-expert wei_scale.  So
    // allocating/filling tail scales would just be dead memory + fill
    // time — restrict both to the active prefix.
    std::vector<AlignedBuffer> wei_scale_buf(n);
    if (cfg.dynamic_quant) {
        // Allocate + fill wei_scale for the active experts only.  Per-
        // channel along N (dims = {1, N}), f32, length `N`.  Seed
        // offsets keep the per-expert scales distinct so cache aliasing
        // on a hash-keyed prepack lookup would surface as a measurable
        // regression rather than a silent hit.
        for (int e = 0; e < n; ++e) {
            wei_scale_buf[e].alloc(static_cast<size_t>(cfg.N) * sizeof(float));
            fill_quant_scale_f32(static_cast<float *>(wei_scale_buf[e].ptr),
                                 static_cast<size_t>(cfg.N),
                                 311 + e * 13);
        }
        for (int i = 0; i < n; ++i) {
            params[i].dynamic_quant = true;
            params[i].dtypes.compute = cfg.compute_dt;
            // src_scale: per-token {M, 1}, f32; buff stays nullptr —
            // hoist-allocates at call time.
            params[i].quant_params.src_scale.buff = nullptr;
            params[i].quant_params.src_scale.dims = {cfg.M_per_op[i], 1};
            params[i].quant_params.src_scale.dt   = data_type_t::f32;
            // wei_scale: per-channel {1, N}, f32; buff is the per-
            // expert allocation above.  `matmul_quant_t::buff` is a
            // `const void *`, so the `void *` arena pointer converts
            // implicitly (no cast needed; read-only on the library side).
            params[i].quant_params.wei_scale.buff = wei_scale_buf[i].ptr;
            params[i].quant_params.wei_scale.dims =
                {1, static_cast<int64_t>(cfg.N)};
            params[i].quant_params.wei_scale.dt   = data_type_t::f32;
            // Asym: src_zp dims/dt populated but buff stays null —
            // the hoist allocates the per-token zp alongside the
            // s8/u8 reorder when `compute_dt = u8`.  For sym we
            // leave the field default-constructed (all-null), which
            // resolve_variant() reads as "no asym correction".
            if (cfg.compute_dt == data_type_t::u8) {
                params[i].quant_params.src_zp.buff = nullptr;
                params[i].quant_params.src_zp.dims = {cfg.M_per_op[i], 1};
                params[i].quant_params.src_zp.dt   = data_type_t::s32;
            }
        }
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
        if (internal_alloc && cfg.N_down <= 0) {
            std::cerr << "ERROR: internal_alloc + moe_postop requires N_down>0\n";
            return false;
        }

        moe_weights.assign(static_cast<size_t>(num_slots),
                           1.f / static_cast<float>(topk));
        moe_output_buf.alloc(static_cast<size_t>(num_tokens) * cfg.N * dst_elem);

        // Initial row_ptrs: Op1 dst layout (legacy fused-disabled path).
        // Below, when N_down > 0 (fused), they are rebuilt to the Op2
        // output layout, which lives in C_down[] (legacy) or src[]
        // (internal-alloc).
        moe_row_ptrs_vec.resize(static_cast<size_t>(num_slots));
        int slot = 0;
        for (int e = 0; e < n; ++e) {
            auto *base = !internal_alloc
                ? static_cast<const char *>(C[e].ptr)
                : static_cast<const char *>(A[e].ptr);
            const int row_stride = !internal_alloc ? cfg.N : lda_val;
            const size_t elem = !internal_alloc ? dst_elem : src_elem;
            for (int j = 0; j < cfg.M_per_op[e]; ++j) {
                moe_row_ptrs_vec[static_cast<size_t>(slot)] =
                    base + static_cast<size_t>(j) * row_stride * elem;
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
    // K_down comes from the gated_act contract — see file-header doc
    // block and `op2_k_for_act()` in
    // `group_matmul_parallel_common.hpp`.  Hardcoding `N/2` here used
    // to corrupt the act=none path (Op2 expected the full N columns
    // but got N/2 worth of weight bytes), masked only because the
    // benchdnn library calls re-validate sizes at the dispatch layer.
    const bool fused_enabled = (cfg.N_down > 0);
    const bool act_halves_k  = (cfg.gated_act > 0);
    const int  K_down        = act_halves_k ? (cfg.N / 2) : cfg.N;

    // gated_act=0 + N_down>0 IS a legal numerical config (act=none
    // path), but it's not MoE-semantic — warn the config author so a
    // typo in the CSV doesn't silently turn off the activation.
    if (fused_enabled && !act_halves_k) {
        std::cerr << "WARNING: N_down=" << cfg.N_down
                  << " with gated_act=0 — Op2 will read the FULL N="
                  << cfg.N << " columns of Op1's output (K_down=N, "
                     "act=none path).  Valid numerically, but NOT "
                     "MoE-semantic.  Set gated_act>0 for Op1 → "
                     "activation → Op2.\n";
    }
    // The "N must be even" constraint only applies when the
    // activation halves K (gate × up split).  For act=none, K_down=N
    // and N can be arbitrary.
    if (fused_enabled && act_halves_k && (cfg.N & 1) != 0) {
        std::cerr << "ERROR: N=" << cfg.N
                  << " must be even when N_down > 0 and gated_act > 0 "
                     "(K_down = N/2).  Skipping config.\n";
        return false;
    }

    // Op2 weight buffers sized to `total` (prepack-extras tail);
    // dst_down stays at `n` (Op2 only computes for firing experts).
    std::vector<AlignedBuffer> B_down(total), C_down(n);
    grp_matmul_fused_moe_params fused;
    grp_matmul_fused_moe_params *fused_ptr = nullptr;

    if (fused_enabled) {
        // Weight-side metadata sized to `total` so the Pass 2 prepack
        // module walks the full down-projection weight pool.  See the
        // outer (Op1) sizing block above for the rationale.
        //
        // `ldb_down` must satisfy the library's minimum-row-stride
        // contract (`group_matmul_fused_moe.cpp::group_matmul_fused_
        // moe_execute` validator):
        //   transB == false: ldb_down >= N_down  (row stride of an
        //                                         [K_down, N_down]
        //                                         row-major weight)
        //   transB == true:  ldb_down >= K_down  (row stride of an
        //                                         [N_down, K_down]
        //                                         row-major weight,
        //                                         i.e. the PyTorch /
        //                                         vLLM nn.Linear
        //                                         convention)
        // benchdnn currently runs every Op2 expert with the same
        // transB as Op1 (the outer `transB` vector is uniform), so
        // pulling `transB[0]` is sufficient.  Hardcoding `cfg.N_down`
        // for the transB=true case (the previous behaviour) silently
        // tripped the library validator on any shape with
        // `K_down > N_down` (e.g. Mixtral down_proj), which is a
        // common configuration; `transB=false` callers are
        // unaffected because `cfg.N_down == cfg.N_down`.
        const int ldb_down_val = transB[0] ? K_down : cfg.N_down;
        fused.down_weight.resize(total);
        fused.N_down.resize(total, cfg.N_down);
        fused.ldb_down.resize(total, ldb_down_val);
        fused.bias_down.resize(total, nullptr);
        // Op2 dst (dst_down / ldc_down) only populated in legacy mode
        // and stays at `n` (only firing experts produce output).
        // Internal-alloc leaves them empty so the library detects the
        // mode and reuses src[] for the Op2 output.
        if (!internal_alloc) {
            fused.dst_down.resize(n);
            fused.ldc_down.resize(n, cfg.N_down);
        }

        for (int i = 0; i < n; ++i) {
            B_down[i].alloc(static_cast<size_t>(K_down) * cfg.N_down * wei_elem);
            fill_buffer(B_down[i].ptr,
                        static_cast<size_t>(K_down) * cfg.N_down,
                        cfg.wei_dt, 200 + i * 11);
            fused.down_weight[i] = B_down[i].ptr;
            if (!internal_alloc) {
                C_down[i].alloc(static_cast<size_t>(cfg.M_per_op[i])
                                * cfg.N_down * dst_elem);
                std::memset(C_down[i].ptr, 0,
                            static_cast<size_t>(cfg.M_per_op[i])
                            * cfg.N_down * dst_elem);
                fused.dst_down[i] = C_down[i].ptr;
            }
        }
        // Allocate + fill the Op2 prepack-extras weight tail.
        for (int i = n; i < total; ++i) {
            B_down[i].alloc(static_cast<size_t>(K_down) * cfg.N_down * wei_elem);
            fill_buffer(B_down[i].ptr,
                        static_cast<size_t>(K_down) * cfg.N_down,
                        cfg.wei_dt, 200 + i * 11);
            fused.down_weight[i] = B_down[i].ptr;
        }
        fused_ptr = &fused;

        // When fused, moe_postop D = N_down (not N).  Rebuild row_ptrs
        // to point at the Op2 output: C_down[] in legacy mode, the
        // caller's src[] in internal-alloc mode (in-place reuse).
        if (moe_enabled) {
            moe.ldc_output = cfg.N_down;
            moe_output_buf.alloc(static_cast<size_t>(num_tokens)
                                 * cfg.N_down * dst_elem);
            moe.output = moe_output_buf.ptr;
            int slot = 0;
            for (int e = 0; e < n; ++e) {
                const char *base = !internal_alloc
                    ? static_cast<const char *>(C_down[e].ptr)
                    : static_cast<const char *>(A[e].ptr);
                // Row stride = N_down for tightly-packed C_down[];
                // = lda_val for in-place src reuse (dst_elem == src_elem
                // for matched-precision MoE, the only case the library
                // currently supports for internal-alloc + post-op).
                const int row_stride =
                    !internal_alloc ? cfg.N_down : lda_val;
                for (int j = 0; j < cfg.M_per_op[e]; ++j) {
                    moe_row_ptrs_vec[static_cast<size_t>(slot)] =
                        base + static_cast<size_t>(j) * row_stride * dst_elem;
                    ++slot;
                }
            }
        }
    }

    // ── HW perf counters ──────────────────────────────────────────────
    // The PerfCounterGroup is owned by bench() and was opened once for
    // the whole process, before any OpenMP parallel region in benchdnn.
    // Here we only `reset/enable` immediately before the timed loop and
    // `disable/read` immediately after — warmup costs (thread spawn,
    // JIT, ITLB warmup, code-cache fills) are therefore not folded into
    // the counter totals.  `use_perf` short-circuits both branches when
    // the caller didn't request counters (or open() failed in bench()).
    const bool use_perf = (perf_ctrs != nullptr);

    // ── Warmup ────────────────────────────────────────────────────────
    // Warmup primes the OMP thread team, jit caches, page tables and
    // the L1/L2/L3 hierarchy.  Crucial for stable measurements:
    // without it the first timed iter pays one-time costs (thread
    // spawn ≈ ms-class, ITLB warmups, code-cache fills) that don't
    // belong in the steady-state number.
    //
    // In internal-alloc mode the library writes Op2 output back into
    // src[i] in place, so we refill src from A_orig[] before every
    // warmup / timed call.  Without this, iter N+1 would consume
    // iter N's Op2 output (not the original input) and the run
    // wouldn't benchmark the configured workload — numbers would
    // also drift as values saturate toward Inf/NaN.  The memcpy is
    // excluded from sum_iter_ms by placing it before ti0.
    auto restore_src = [&]() {
        if (!internal_alloc) return;
        for (int i = 0; i < n; ++i)
            std::memcpy(A[i].ptr, A_orig[i].ptr, A_bytes[i]);
    };
    for (int w = 0; w < cfg.warmup; ++w) {
        restore_src();
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

    // ── Timed iterations ──────────────────────────────────────────────
    // Two timing axes:
    //   * sum_iter_ms — cumulative per-call wall time, EXCLUDES
    //     flush_cache() in COLD_CACHE mode.  This is what avg_ms is
    //     derived from (matches the matmul benchdnn convention).
    //   * total_ms    — outer wall clock, INCLUDES per-iter flush
    //     and chrono::now() overhead.  Reported separately for
    //     diagnostic / overhead-budget analysis.
    //   * min_ms      — best-of-N per-iter time, the steady-state
    //     "peak" measurement (used for GFLOPS_peak).
    //
    // Warm-cache mode (COLD_CACHE=0): sum_iter_ms ≈ total_ms minus
    // chrono overhead (~50 ns × 2 calls/iter), so avg_ms ≈ wall/iters.
    // Cold-cache mode (COLD_CACHE=1): flush_cache adds ~ms per iter;
    // sum_iter_ms isolates the kernel time we actually care about.
    double min_ms = std::numeric_limits<double>::max();
    double sum_iter_ms = 0.0;
    if (use_perf && perf_ctrs->is_available()) {
        perf_ctrs->reset();
        perf_ctrs->enable();
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < cfg.iters; ++it) {
        // Runtime cache_mode (cold/warm/hot) replaces the older
        // compile-time COLD_CACHE flag in line with the rest of
        // benchdnn.  flush_cache() is excluded from per-iter ti0/ti1
        // timing so sum_iter_ms reflects pure kernel time.
        // restore_src() is also excluded so the memcpy cost of the
        // internal-alloc warmup/restore path doesn't inflate timing.
        restore_src();
        if (options.cache_mode == CacheMode::COLD)
            flush_cache(cache_size);
        auto ti0 = std::chrono::high_resolution_clock::now();
        auto st = group_matmul_direct(layout, transA, transB, Mv, Nv, Kv, alpha,
                                      src_ptrs, lda, wei_ptrs, ldb, bias_ptrs, beta,
                                      dst_ptrs, ldc, wconst, params,
                                      moe_ptr, act_ptr, fused_ptr);
        auto ti1 = std::chrono::high_resolution_clock::now();
        if (st != status_t::success) {
            std::cerr << "ERROR: group_matmul_direct failed at iter " << it
                      << std::endl;
            return false;
        }
        double iter_ms = std::chrono::duration<double, std::milli>(ti1 - ti0).count();
        sum_iter_ms += iter_ms;
        if (iter_ms < min_ms) min_ms = iter_ms;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    if (use_perf && perf_ctrs->is_available()) {
        perf_ctrs->disable();
        perf_ctrs->read();
    }

    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double avg_ms = sum_iter_ms / cfg.iters;

    // ── GFLOPS accounting ──────────────────────────────────────────────
    // Sum of GEMM-equivalent multiply-add FLOPS across the four
    // pipeline stages.  We deliberately omit element-wise activation
    // FLOPS (Stage 2): for typical MoE shapes they are < 0.1 % of the
    // GEMM cost.  Their wall-clock IS measured (the timer wraps the
    // whole call), so reported avg_ms is honest end-to-end; only the
    // GFLOPS denominator is approximate (dominated by GEMM compute).
    double total_flops = 0;
    for (int i = 0; i < n; ++i) {
        total_flops += 2.0 * cfg.M_per_op[i] * cfg.K * cfg.N;          // Stage 1: Op1 gate+up GEMM
        if (fused_enabled)
            total_flops += 2.0 * cfg.M_per_op[i] * K_down * cfg.N_down; // Stage 3: Op2 down_proj GEMM
    }
    if (moe_enabled) {
        int D_out = fused_enabled ? cfg.N_down : cfg.N;
        total_flops += 2.0 * num_tokens * D_out * topk;             // Stage 4: weighted reduce
    }
    double gflops_avg  = (total_flops / avg_ms ) * 1e-6;            // mean throughput
    double gflops_peak = (total_flops / min_ms) * 1e-6;             // best-of-N throughput

    std::string dtypes = datatypeToStr(cfg.src_dt) + ":" + datatypeToStr(cfg.wei_dt)
                         + ":" + datatypeToStr(cfg.dst_dt);
    std::string m_str = format_M(cfg);

    std::string moe_str = moe_enabled ? "topk=" + std::to_string(topk) : "off";
    std::string fused_str;
    if (fused_enabled) {
        fused_str = "N_down=" + std::to_string(cfg.N_down);
        if (internal_alloc) fused_str += "*";  // * = library-allocated
    } else {
        fused_str = "off";
    }
    // DQ-INT8 column — surfaces the variant on the console / CSV so
    // a sweep that mixes bf16 and int8 lines is visually self-
    // describing.  "bf16" is the default (no DQ-INT8); "dq8s" / "dq8u"
    // distinguish sym vs asym so a tuning script can plot the two
    // families separately without re-reading the input file.
    std::string quant_str = "bf16";
    if (cfg.dynamic_quant) {
        quant_str = (cfg.compute_dt == data_type_t::u8) ? "dq8u" : "dq8s";
    }

    // Console
    std::cout << std::setw(4) << n << "  "
              << std::setw(12) << m_str << "  "
              << std::setw(6) << cfg.K << "  "
              << std::setw(6) << cfg.N << "  "
              << std::setw(5) << cfg.iters << "  "
              << std::setw(6) << cfg.warmup << "  "
              << std::setw(14) << dtypes << "  "
              << std::setw(5) << quant_str << "  "
              << std::setw(10) << moe_str << "  "
              << std::setw(13) << fused_str << "  "
              << std::fixed << std::setprecision(3)
              << std::setw(10) << avg_ms << "  "
              << std::setw(10) << min_ms << "  "
              << std::setprecision(1)
              << std::setw(9) << gflops_avg << "  "
              << std::setw(9) << gflops_peak
              << std::endl;

    // HW perf counter print — uses sum_iter_ms (kernel-only wall time
    // across all timed iters, restore_src + flush_cache excluded) as
    // the elapsed_sec basis for derived metrics so the per-byte /
    // per-cycle ratios reflect actual GEMM work, not the measurement
    // harness.  Mirrors the matmul driver's [PERF] line + raw dump.
    if (use_perf && perf_ctrs->is_available()) {
        const double elapsed_sec = sum_iter_ms / 1000.0;
        int nt = 1;
#ifdef _OPENMP
        nt = omp_get_max_threads();
#endif
        if (nt < 1) nt = 1;
        auto derived = perf_ctrs->derive(elapsed_sec, nt);
        printf("  [PERF]");
        perf_ctrs->print_values(derived, false);
        printf("\n");
        perf_ctrs->print_raw_counters();
    }

    // CSV (wall_ms = outer t1-t0; sum_iter_ms = ∑ per-iter kernel time;
    // gap = wall - sum_iter ≈ flush_cache + chrono overhead)
    csv << n << "," << m_str << "," << cfg.K << "," << cfg.N << ","
        << cfg.iters << "," << cfg.warmup << "," << dtypes << ","
        << quant_str << ","
        << (cfg.is_weights_const ? "true" : "false") << ","
        << cfg.moe_topk << "," << cfg.gated_act << "," << cfg.N_down << ","
        << cfg.use_internal_alloc << ","
        << std::fixed << std::setprecision(6)
        << total_ms << "," << sum_iter_ms << ","
        << avg_ms << "," << min_ms << ","
        << std::setprecision(2) << gflops_avg << "," << gflops_peak << "\n";
    return true;
}

// ── Entry point ─────────────────────────────────────────────────────────

int bench(const std::string &in_filename, const std::string &out_filename,
          const global_options &options,
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
    csv << "num_ops,M,K,N,iters,warmup,dtypes,quant,is_weights_const,moe_topk,"
           "gated_act,N_down,use_internal_alloc,wall_ms,sum_iter_ms,"
           "avg_ms,min_ms,GFLOPS_avg,GFLOPS_peak\n";

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
                 "  quant         moe          fused      avg_ms      min_ms  GFLOPS_a  GFLOPS_p"
              << std::endl;
    std::cout << "  (fused column: 'N_down=X' = caller-allocated, "
                 "'N_down=X*' = library-allocated + src-reuse;\n"
                 "   quant column: bf16 = standard bf16 path, "
                 "dq8s/dq8u = DQ-INT8 sym/asym custom kernel)"
              << std::endl;

    // ── HW perf counters: open once per process ─────────────────────
    // Lifecycle is owned here, not in run_config().  Opening before
    // the first call to run_config() guarantees the perf_event_open()
    // fds exist *before* any OpenMP parallel region in benchdnn — so
    // the OMP team that libgomp / libiomp5 spawns during the first
    // config's warmup joins the inheritance subtree of these counters
    // (`attr.inherit=1`, `pid=getpid()` in perf_counters.cpp).
    //
    // For all subsequent configs we just reset/enable/disable/read
    // around the timed loop; the same set of OMP workers stays in the
    // inheritance subtree across the whole sweep.  Without this lift,
    // each config opened fresh fds after the OMP pool already existed,
    // leaving every config after the first with no descendants in its
    // perf-event subtree.
    //
    // Residual limitation: Linux returns
    //   master_task_count + sum(counts of EXITED descendants)
    // on `read()`, i.e. *live* descendants are not walked at read
    // time.  Persistent OMP pools never exit, so even with the lift
    // the `[PERF]` line still reflects predominantly master-thread
    // events.  For true process-wide aggregation use external
    // `perf stat ./benchdnn ...` or run with `OMP_NUM_THREADS=1`.
    // See doc/perf_counters.md for the full discussion.
    PerfProfile perf_profile = parse_perf_profile(
                                 options.perf_profile_str.c_str());
    PerfCounterGroup perf_ctrs(perf_profile);
    PerfCounterGroup *perf_ctrs_ptr = nullptr;
    if (options.perf_counters && perf_ctrs.is_available()) {
        if (perf_ctrs.open()) {
            perf_ctrs_ptr = &perf_ctrs;
        } else {
            commonlog_warning("HW perf counters unavailable; continuing without them.");
        }
    }

    for (const auto &cfg : configs) {
        if (!run_config(cfg, csv, options, cache_size, perf_ctrs_ptr))
            std::cerr << "  ^ Failed, skipping." << std::endl;
    }

    csv.close();
    std::cout << "\nResults written to " << out_filename << std::endl;
    return OK;
}

} // namespace grp_matmul
} // namespace benchdnn
} // namespace zendnnl
