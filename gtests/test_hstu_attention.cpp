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

/// HSTU attention: the causal row band against the mask predicate, both
/// kernels against an FP64 transcription of `pytorch_hstu_mha` /
/// `pytorch_cached_hstu_mha`, the bitwise determinism of the tiled kernel and
/// its fast silu (ZENDNNL_HSTU_ATTENTION_FAST_SILU), padding, kernel
/// selection, the tiled planning and driver helpers, and argument
/// validation. Dynamic INT8 is in test_hstu_attention_int8.cpp.

#include "hstu_attention_test_utils.hpp"
#include "lowoha_operators/hstu_attention/common/lowoha_hstu_select.hpp"
#include "lowoha_operators/hstu_attention/tiled/lowoha_hstu_tiled_driver.hpp"

#include <omp.h>

#include <initializer_list>

namespace {

using namespace hstu_test;

using i64_list = std::initializer_list<int64_t>;

TEST(HstuAttentionMask, CausalBandMatchesPredicate) {
    for (int64_t n = 1; n <= 40; ++n)
        for (int64_t t : i64_list {-1, 0, 1, 3, n - 1, n})
            for (int64_t c : i64_list {0, 1, 3})
                for (int64_t w : i64_list {0, 1, 4, 100})
                    for (int64_t m : i64_list {0, 2}) {
                        if (t > n || t < -1) continue;
                        ha::hstu_attention_params p;
                        p.causal = true;
                        p.contextual_seq_len = c;
                        p.max_attn_len = w;
                        p.min_full_attn_seq_len = m;
                        const ha::seq_mask_t mask(p, n, t);
                        for (int64_t i = 0; i < n; ++i) {
                            const ha::row_band_t band = mask.band(i);
                            for (int64_t j = 0; j < n; ++j) {
                                const bool want = oracle_valid(
                                        n, t, c, w, m, true, i, j);
                                ASSERT_EQ(mask.valid(i, j), want)
                                        << "n=" << n << " t=" << t << " c=" << c
                                        << " w=" << w << " m=" << m
                                        << " i=" << i << " j=" << j;
                                const bool in_band
                                        = (j >= band.lo && j < band.hi)
                                        || (band.extra_diag && j == i);
                                ASSERT_EQ(in_band, want)
                                        << "n=" << n << " t=" << t << " c=" << c
                                        << " w=" << w << " m=" << m
                                        << " i=" << i << " j=" << j;
                            }
                        }
                    }
}

TEST(HstuAttentionMask, NonCausalPredicate) {
    for (int64_t n = 1; n <= 24; ++n)
        for (int64_t t : i64_list {-1, 0, 2, n})
            for (int64_t c : i64_list {0, 2, n + 1})
                for (int64_t w : i64_list {0, 1, 5})
                    for (int64_t m : i64_list {0, 3}) {
                        ha::hstu_attention_params p;
                        p.causal = false;
                        p.contextual_seq_len = c;
                        p.max_attn_len = w;
                        p.min_full_attn_seq_len = m;
                        const ha::seq_mask_t mask(p, n, t);
                        for (int64_t i = 0; i < n; ++i)
                            for (int64_t j = 0; j < n; ++j)
                                ASSERT_EQ(mask.valid(i, j),
                                        oracle_valid(
                                                n, t, c, w, m, false, i, j))
                                        << "n=" << n << " t=" << t << " c=" << c
                                        << " w=" << w << " m=" << m
                                        << " i=" << i << " j=" << j;
                    }
}

class HstuAttentionKernels : public ::testing::TestWithParam<hstu_case> {};

TEST_P(HstuAttentionKernels, MatchOracle) {
    if (env_overrides_kernel())
        GTEST_SKIP() << "ZENDNNL_HSTU_ATTENTION_KERNEL overrides the kernel";
    const hstu_case &c = GetParam();
    const hstu_problem prob(c);
    const std::vector<double> want = prob.oracle();
    const double tol
            = c.out_dt == data_type_t::f32 && c.in_dt == data_type_t::f32
            ? 1e-5
            : 1.5e-2;

    std::vector<float> got;
    bool canary = false;
    ASSERT_EQ(
            prob.run(kernel_t::reference, got, 0, &canary), status_t::success);
    EXPECT_LT(rel_error(got, want), tol) << "reference";
    EXPECT_TRUE(canary) << "reference wrote outside its rows";

    const bool tiled = c.causal && c.in_dt == data_type_t::bf16 && has_bf16();
    // The tiled kernel sizes its query tiles from the thread count.
    for (int32_t threads : {0, 1, 5, 64}) {
        ASSERT_EQ(prob.run(kernel_t::none, got, threads, &canary),
                status_t::success);
        EXPECT_LT(rel_error(got, want), tol) << "auto, threads=" << threads;
        EXPECT_TRUE(canary) << "auto wrote outside its rows";

        const status_t st = prob.run(kernel_t::tiled, got, threads, &canary);
        if (!tiled) {
            EXPECT_EQ(st, status_t::unimplemented);
            continue;
        }
        ASSERT_EQ(st, status_t::success);
        EXPECT_LT(rel_error(got, want), tol) << "tiled, threads=" << threads;
        EXPECT_TRUE(canary) << "tiled wrote outside its rows";
    }
}

INSTANTIATE_TEST_SUITE_P(Cases, HstuAttentionKernels,
        ::testing::ValuesIn(make_cases()),
        [](const ::testing::TestParamInfo<hstu_case> &info) {
            return info.param.name;
        });

/// Rows are independent and P columns outside a row's band are exact zeros,
/// so the tiled output does not depend on the tile size or thread count.
TEST(HstuAttentionTiled, DeterministicAcrossThreads) {
    if (env_overrides_kernel() || !has_bf16()) GTEST_SKIP();
    for (const hstu_case &c : make_cases()) {
        if (!c.causal || c.in_dt != data_type_t::bf16) continue;
        const hstu_problem prob(c);
        std::vector<float> want, got;
        ASSERT_EQ(prob.run(kernel_t::tiled, want, 1), status_t::success);
        for (int32_t threads : {5, 64}) {
            ASSERT_EQ(
                    prob.run(kernel_t::tiled, got, threads), status_t::success);
            EXPECT_TRUE(bitwise_equal(got, want))
                    << c.name << ", threads=" << threads;
        }
    }
}

/// Scratch is reused across calls (and output types); results must not
/// depend on what an earlier call left in it.
TEST(HstuAttentionTiled, RepeatedCallsReuseScratch) {
    if (env_overrides_kernel() || !has_bf16()) GTEST_SKIP();
    const hstu_case big = find_case("dlrm_shape_fused_uvqk");
    hstu_case delta = find_case("delta_window");
    hstu_case delta_f32 = delta;
    delta_f32.out_dt = data_type_t::f32;
    const hstu_problem big_prob(big), delta_prob(delta),
            delta_f32_prob(delta_f32);
    std::vector<float> big0, delta0, delta_f32_0, got;
    for (int32_t threads : {0, 3}) {
        ASSERT_EQ(big_prob.run(kernel_t::tiled, big0, threads),
                status_t::success);
        ASSERT_EQ(delta_prob.run(kernel_t::tiled, delta0, threads),
                status_t::success);
        ASSERT_EQ(delta_f32_prob.run(kernel_t::tiled, delta_f32_0, threads),
                status_t::success);
        for (int rep = 0; rep < 2; ++rep) {
            ASSERT_EQ(big_prob.run(kernel_t::tiled, got, threads),
                    status_t::success);
            EXPECT_TRUE(bitwise_equal(got, big0)) << "big, rep " << rep;
            ASSERT_EQ(delta_f32_prob.run(kernel_t::tiled, got, threads),
                    status_t::success);
            EXPECT_TRUE(bitwise_equal(got, delta_f32_0))
                    << "delta f32, rep " << rep;
            ASSERT_EQ(delta_prob.run(kernel_t::tiled, got, threads),
                    status_t::success);
            EXPECT_TRUE(bitwise_equal(got, delta0)) << "delta, rep " << rep;
        }
    }
}

/// Padding every sequence to max_seq_len (is_varlen = false) gives the
/// variable-length output bit for bit, on every case (delta mode stays
/// variable-length) and kernel, and leaves the canaries untouched.
TEST(HstuAttentionVarlen, PaddedMatchesVarlen) {
    if (env_overrides_kernel()) GTEST_SKIP();
    // BF16, and with INT8 the per-block scales of the padded layout.
    std::vector<std::pair<bool, bool>> quant {{false, false}};
    if (has_int8())
        quant.insert(quant.end(), {{true, false}, {true, true}, {false, true}});
    for (const hstu_case &c : make_cases()) {
        const hstu_problem prob(c);
        for (kernel_t kernel : {kernel_t::none, kernel_t::reference})
            for (int32_t threads : {1, 5})
                for (const auto &[qk, pv] : quant) {
                    // The reference kernel ignores the flags.
                    if ((qk || pv) && kernel == kernel_t::reference) continue;
                    std::vector<float> var, pad;
                    bool var_ok = false, pad_ok = false;
                    ASSERT_EQ(prob.run(kernel, var, threads, &var_ok, qk, pv),
                            status_t::success)
                            << c.name;
                    ASSERT_EQ(prob.run(kernel, pad, threads, &pad_ok, qk, pv,
                                      -1, false),
                            status_t::success)
                            << c.name;
                    EXPECT_TRUE(pad_ok) << c.name;
                    EXPECT_TRUE(bitwise_equal(var, pad))
                            << c.name << ", kernel "
                            << ha::kernel_to_string(kernel) << ", threads "
                            << threads << ", qk " << qk << ", pv " << pv;
                }
    }
}

/// The fast silu adds at most 1.2e-4 relative error to P before its BF16
/// rounding (2^-9), so the outputs keep the default path's error.
TEST(HstuAttentionFastSilu, MatchesOracle) {
    if (!has_bf16()) GTEST_SKIP();
    for (const hstu_case &c : fast_silu_cases()) {
        const hstu_problem prob(c);
        const std::vector<double> want = prob.oracle();
        std::vector<float> base, got;
        ASSERT_EQ(prob.run(kernel_t::tiled, base, 0, nullptr, false, false, 0),
                status_t::success);
        const double base_err = rel_error(base, want);
        EXPECT_LT(base_err, 1.5e-2) << c.name << ", default silu";
        for (int32_t threads : {0, 1, 5, 64}) {
            bool canary = false;
            ASSERT_EQ(prob.run(kernel_t::tiled, got, threads, &canary, false,
                              false, 1),
                    status_t::success);
            const double err = rel_error(got, want);
            EXPECT_LT(err, 1.5e-2) << c.name << ", threads=" << threads;
            EXPECT_LT(err, 1.1 * base_err + 5e-4)
                    << c.name << ", threads=" << threads
                    << ", default silu error " << base_err;
            EXPECT_TRUE(canary) << c.name << " wrote outside its rows";
        }
    }
}

TEST(HstuAttentionSelect, ChoosesKernel) {
    using ha::choose_kernel;
    const kernel_t none = kernel_t::none, tiled = kernel_t::tiled,
                   ref = kernel_t::reference;
    struct row {
        kernel_t requested;
        bool tiled_ok, quant, vnni;
        kernel_t kernel;
        bool int8;
    };
    const row rows[] = {
            {none, true, false, true, tiled, false},
            {none, true, true, true, tiled, true},
            // No VNNI: the BF16 tiled kernel.
            {none, true, true, false, tiled, false},
            {tiled, true, true, false, tiled, false},
            {none, false, true, true, ref, false},
            // A forced reference ignores the flags.
            {ref, true, true, true, ref, false},
            // A forced tiled on an unsupported call (the caller returns
            // unimplemented).
            {tiled, false, true, true, tiled, false},
    };
    for (const row &r : rows) {
        const ha::kernel_choice_t c
                = choose_kernel(r.requested, r.tiled_ok, r.quant, r.vnni);
        EXPECT_EQ(c.kernel, r.kernel)
                << int(r.requested) << r.tiled_ok << r.quant << r.vnni;
        EXPECT_EQ(c.int8, r.int8)
                << int(r.requested) << r.tiled_ok << r.quant << r.vnni;
    }
}

/// ZENDNNL_HSTU_ATTENTION_INT8_QK / _PV, _FAST_SILU and _VARLEN: unset,
/// empty or not a boolean is -1, else 0 or 1, in any case.
TEST(HstuAttentionSelect, ParsesBoolEnv) {
    for (const char *bad : {static_cast<const char *>(nullptr), "", "2", "fast",
                 "pad", "maybe"})
        EXPECT_EQ(ha::parse_bool_env(bad), -1) << (bad ? bad : "nullptr");
    for (const char *on : {"1", "true", "TRUE", "on", "ON", "yes"})
        EXPECT_EQ(ha::parse_bool_env(on), 1) << on;
    for (const char *off : {"0", "false", "Off", "no"})
        EXPECT_EQ(ha::parse_bool_env(off), 0) << off;
}

TEST(HstuAttentionSelect, Int8Environment) {
    for (bool param : {false, true}) {
        EXPECT_EQ(ha::resolve_int8_flag(-1, param), param);
        EXPECT_TRUE(ha::resolve_int8_flag(1, param));
        EXPECT_FALSE(ha::resolve_int8_flag(0, param));
    }
}

/// The flags a call runs with: QK and PV overrides are independent and win
/// in both directions; f32 q/k/v drop both; qk_dim past kInt8MaxQkDim drops
/// INT8 Q*K^T only.
TEST(HstuAttentionSelect, QuantFlags) {
    const data_type_t bf16 = data_type_t::bf16, f32 = data_type_t::f32;
    const int64_t d = 128, big = ha::kInt8MaxQkDim + 1;
    struct row_t {
        bool qk, pv;
        data_type_t dt;
        int64_t qk_dim;
        int env_qk, env_pv;
        bool want_qk, want_pv, dtype, qk_dim_drop;
    };
    const row_t rows[] = {
            {false, false, bf16, d, -1, -1, false, false, false, false},
            {true, false, bf16, d, -1, -1, true, false, false, false},
            {false, true, bf16, d, -1, -1, false, true, false, false},
            {false, false, bf16, d, 1, -1, true, false, false, false},
            {false, false, bf16, d, -1, 1, false, true, false, false},
            {true, true, bf16, d, 0, -1, false, true, false, false},
            {true, true, bf16, d, -1, 0, true, false, false, false},
            {true, true, f32, d, -1, -1, false, false, true, false},
            {false, false, f32, d, 1, 1, false, false, true, false},
            {true, true, bf16, ha::kInt8MaxQkDim, -1, -1, true, true, false,
                    false},
            {true, true, bf16, big, -1, -1, false, true, false, true},
            {false, true, bf16, big, -1, -1, false, true, false, false},
    };
    for (const row_t &r : rows) {
        const ha::quant_flags_t f = ha::resolve_quant_flags(
                r.qk, r.pv, r.dt, r.qk_dim, r.env_qk, r.env_pv);
        const std::string at = std::to_string(&r - rows);
        EXPECT_EQ(f.qk, r.want_qk) << "row " << at;
        EXPECT_EQ(f.pv, r.want_pv) << "row " << at;
        EXPECT_EQ(f.dropped_dtype, r.dtype) << "row " << at;
        EXPECT_EQ(f.dropped_qk_dim, r.qk_dim_drop) << "row " << at;
    }
}

/// The fields of a case that the work decomposition reads; `c` and `offsets`
/// must outlive the result.
ha::hstu_attention_params plan_params(
        const hstu_case &c, std::vector<int64_t> &offsets) {
    offsets.assign(c.lens.size() + 1, c.base);
    for (size_t b = 0; b < c.lens.size(); ++b)
        offsets[b + 1] = offsets[b] + c.lens[b];
    ha::hstu_attention_params p;
    p.num_seqs = c.lens.size();
    p.num_heads = c.heads;
    p.seq_offsets = offsets.data();
    p.delta_size = c.delta;
    p.causal = c.causal;
    p.num_targets = c.targets.empty() ? nullptr : c.targets.data();
    p.max_attn_len = c.window;
    p.contextual_seq_len = c.ctx;
    p.min_full_attn_seq_len = c.min_full;
    p.v_dim = c.v_dim;
    p.max_seq_len = *std::max_element(c.lens.begin(), c.lens.end()) + 5;
    return p;
}

TEST(HstuAttentionPlan, WorkItemsInSdpaOrder) {
    for (const hstu_case &c : make_cases()) {
        if (!c.causal) continue;
        std::vector<int64_t> offsets;
        const ha::hstu_attention_params p = plan_params(c, offsets);
        for (int64_t bm : {32, 64, 128}) {
            const ha::work_plan_t plan(p, bm);
            int64_t z = 0;
            for (int64_t b = 0; b < p.num_seqs; ++b) {
                const ha::seq_rows_t s = ha::seq_rows(p, b);
                for (int64_t h = 0; h < p.num_heads; ++h)
                    for (int64_t row0 = 0; row0 < s.num_q; row0 += bm, ++z) {
                        ASSERT_LT(z, plan.size()) << c.name;
                        const ha::work_item_t w = plan.item(z);
                        ASSERT_EQ(w.b, b) << c.name << " z=" << z;
                        ASSERT_EQ(w.h, h) << c.name << " z=" << z;
                        ASSERT_EQ(w.row0, row0) << c.name << " z=" << z;
                        ASSERT_EQ(w.rows, std::min(bm, s.num_q - row0))
                                << c.name << " z=" << z;
                        ASSERT_EQ(w.s.q_row, s.q_row) << c.name;
                        ASSERT_EQ(w.s.num_q, s.num_q) << c.name;
                    }
            }
            EXPECT_EQ(z, plan.size()) << c.name << " bm=" << bm;
        }
    }
}

TEST(HstuAttentionPlan, KeyBlocksCoverKeyRanges) {
    for (const hstu_case &c : make_cases()) {
        if (!c.causal) continue;
        std::vector<int64_t> offsets;
        const ha::hstu_attention_params p = plan_params(c, offsets);
        const std::vector<ha::key_range_t> need = ha::key_ranges(p);
        ASSERT_EQ(int64_t(need.size()), p.num_seqs);
        // Every valid key except a diagonal outside the row's band.
        for (int64_t b = 0; b < p.num_seqs; ++b) {
            const ha::seq_rows_t s = ha::seq_rows(p, b);
            const ha::seq_mask_t mask(p, s.len, s.targets);
            int64_t lo = s.len, hi = 0;
            for (int64_t i = s.q_pos; i < s.q_pos + s.num_q; ++i) {
                const ha::row_band_t band = mask.band(i);
                for (int64_t j = 0; j < s.len; ++j)
                    if (mask.valid(i, j) && !(band.extra_diag && j == i)) {
                        lo = std::min(lo, j);
                        hi = std::max(hi, j + 1);
                    }
            }
            if (lo >= hi) lo = hi = 0;
            EXPECT_EQ(need[b].lo, lo) << c.name << " b=" << b;
            EXPECT_EQ(need[b].hi, hi) << c.name << " b=" << b;
        }

        const ha::kv_blocks_t blocks(p, need);
        std::vector<int64_t> jobs;
        int64_t g = 0;
        for (int64_t b = 0; b < p.num_seqs; ++b) {
            const int64_t len = offsets[b + 1] - offsets[b];
            EXPECT_EQ(blocks.first_block(b), g) << c.name;
            for (int64_t k0 = 0; k0 < len; k0 += ha::kBlockN, ++g) {
                const ha::kv_block_t blk = blocks.block(g);
                EXPECT_EQ(blk.g, g) << c.name;
                EXPECT_EQ(blk.b, b) << c.name;
                EXPECT_EQ(blk.k0, k0) << c.name;
                EXPECT_EQ(blk.nb, std::min(ha::kBlockN, len - k0)) << c.name;
                if (k0 < need[b].hi && k0 + ha::kBlockN > need[b].lo)
                    jobs.push_back(g);
            }
        }
        EXPECT_EQ(blocks.count(), g) << c.name;
        EXPECT_EQ(blocks.jobs(), jobs) << c.name;
    }
}

TEST(HstuAttentionPlan, BlockHeightGivesEveryThreadTiles) {
    for (const hstu_case &c : make_cases()) {
        std::vector<int64_t> offsets;
        const ha::hstu_attention_params p = plan_params(c, offsets);
        auto tiles = [&](int64_t bm) {
            int64_t n = 0;
            for (int64_t b = 0; b < p.num_seqs; ++b)
                n += (ha::seq_rows(p, b).num_q + bm - 1) / bm;
            return n * p.num_heads;
        };
        for (int32_t threads : {1, 5, 32, 64, 256}) {
            const int64_t bm = ha::choose_block_m(p, threads);
            EXPECT_TRUE(bm == 32 || bm == 64 || bm == 128) << bm;
            const int64_t want = ha::kTilesPerThread * threads;
            if (bm > ha::kMinBlockM) { EXPECT_GE(tiles(bm), want) << c.name; }
            if (bm < ha::kMaxBlockM) {
                EXPECT_LT(tiles(2 * bm), want) << c.name;
            }
        }
    }
}

/// The driver's per-tile windows: rows and bands from prepare_rows, the key
/// blocks it visits, block_window aligned to key_align and tight, and
/// chunk_bounds equal to the bands.
template <int64_t key_align>
void check_windows(const hstu_case &c) {
    std::vector<int64_t> offsets;
    const ha::hstu_attention_params p = plan_params(c, offsets);
    ha::arena_t arena;
    constexpr int64_t width = 64;
    for (int64_t bm : {32, 64, 128}) {
        const ha::tile_pool_t pool(1, bm, 0, 0, p.v_dim, arena);
        const ha::tile_t t = pool.get(0);
        const ha::work_plan_t plan(p, bm);
        for (int64_t z = 0; z < plan.size(); ++z) {
            const ha::work_item_t w = plan.item(z);
            const ha::key_range_t keys = ha::prepare_rows(p, w, t);
            const ha::seq_mask_t mask(p, w.s.len, w.s.targets);
            for (int64_t r = 0; r < w.rows; ++r) {
                const ha::row_band_t band = mask.band(w.s.q_pos + w.row0 + r);
                ASSERT_EQ(t.bands[r].lo, band.lo) << c.name;
                ASSERT_EQ(t.bands[r].hi, band.hi) << c.name;
                ASSERT_EQ(t.bands[r].extra_diag, band.extra_diag) << c.name;
            }
            for (int64_t k0 = 0; k0 < w.s.len; k0 += ha::kBlockN) {
                const int64_t nb = std::min(ha::kBlockN, w.s.len - k0);
                int64_t ra = w.rows, rb = 0, lo = nb, hi = 0;
                for (int64_t r = 0; r < w.rows; ++r) {
                    const int64_t a = std::max(t.bands[r].lo, k0) - k0;
                    const int64_t e = std::min(t.bands[r].hi, k0 + nb) - k0;
                    if (a < e) {
                        ra = std::min(ra, r);
                        rb = r + 1;
                        lo = std::min(lo, a);
                        hi = std::max(hi, e);
                    }
                }
                ha::block_window_t bw;
                const bool any = ha::block_window<key_align>(
                        t.bands, w.rows, 7, k0, nb, bw);
                ASSERT_EQ(any, ra < rb) << c.name << " z=" << z;
                if (!any) continue;
                EXPECT_GE(k0, keys.lo / ha::kBlockN * ha::kBlockN) << c.name;
                EXPECT_LT(k0, keys.hi) << c.name;
                EXPECT_EQ(bw.g, 7);
                EXPECT_EQ(bw.k0, k0);
                EXPECT_EQ(bw.nb, nb);
                EXPECT_EQ(bw.ra, ra) << c.name;
                EXPECT_EQ(bw.rb, rb) << c.name;
                EXPECT_EQ(bw.plo, lo / key_align * key_align) << c.name;
                EXPECT_EQ(bw.phi,
                        std::min(nb,
                                (hi + key_align - 1) / key_align * key_align))
                        << c.name;
                for (int64_t c0 = bw.plo / width * width; c0 < bw.phi;
                        c0 += width) {
                    ha::chunk_bounds(
                            t.bands, ra, rb, k0 + c0, width, t.lo, t.hi);
                    for (int64_t r = ra; r < rb; ++r)
                        for (int64_t j = 0; j < width; ++j) {
                            const int64_t key = k0 + c0 + j;
                            ASSERT_EQ(j >= t.lo[r] && j < t.hi[r],
                                    key >= t.bands[r].lo && key < t.bands[r].hi)
                                    << c.name << " r=" << r << " key=" << key;
                        }
                }
            }
        }
    }
}

TEST(HstuAttentionDriver, BlockWindowsCoverBands) {
    for (const hstu_case &c : make_cases()) {
        if (!c.causal) continue;
        check_windows<2>(c);
        check_windows<4>(c);
    }
}

TEST(HstuAttentionApi, RejectsInvalidArguments) {
    const hstu_case c {"invalid", {10, 20}};
    hstu_problem prob(c);
    std::vector<float> got;
    prob.offsets[1] = 30; // second sequence length becomes negative
    EXPECT_EQ(prob.run(kernel_t::reference, got), status_t::failure);
    prob.offsets[1] = 10;
    prob.max_len = 15; // shorter than the second sequence
    EXPECT_EQ(prob.run(kernel_t::reference, got), status_t::failure);
}

/// Valid parameters for two sequences of 10 and 20 tokens, H=2, D=V=8.
struct api_fixture {
    std::vector<bfloat16_t> q, k, v, out;
    std::vector<int64_t> offsets {0, 10, 30}, targets {2, 3};
    ha::hstu_attention_params p;
    api_fixture() : q(30 * 16), k(30 * 16), v(30 * 16), out(30 * 16) {
        p.num_seqs = 2;
        p.num_heads = 2;
        p.qk_dim = p.v_dim = 8;
        p.max_seq_len = 20;
        p.seq_offsets = offsets.data();
        p.num_targets = targets.data();
        p.q_stride_t = p.k_stride_t = p.v_stride_t = p.o_stride_t = 16;
        p.q_stride_h = p.k_stride_h = p.v_stride_h = p.o_stride_h = 8;
        p.kernel = kernel_t::reference;
    }
    status_t run(const void *qp = nullptr) {
        return ha::hstu_attention_direct(
                qp ? qp : q.data(), k.data(), v.data(), out.data(), p);
    }
};

TEST(HstuAttentionApi, RejectsInvalidParameters) {
    if (env_overrides_kernel()) GTEST_SKIP();
    {
        api_fixture f;
        EXPECT_EQ(f.run(), status_t::success);
    }
    auto expect_failure = [](auto mutate, const char *what) {
        api_fixture f;
        mutate(f);
        EXPECT_EQ(f.run(), status_t::failure) << what;
    };
    expect_failure([](api_fixture &f) { f.p.seq_offsets = nullptr; },
            "null seq_offsets");
    expect_failure([](api_fixture &f) { f.p.num_seqs = -1; }, "num_seqs < 0");
    expect_failure(
            [](api_fixture &f) { f.targets[1] = 21; }, "num_targets > length");
    expect_failure(
            [](api_fixture &f) { f.targets[0] = -1; }, "num_targets < 0");
    expect_failure([](api_fixture &f) { f.offsets = {0, 20, 10}; },
            "decreasing offsets");
    expect_failure([](api_fixture &f) { f.p.k_stride_t = 0; }, "zero stride");
    expect_failure([](api_fixture &f) { f.p.qkv_dt = data_type_t::s8; },
            "unsupported dtype");
    expect_failure([](api_fixture &f) { f.p.o_stride_h = 4; },
            "overlapping output heads");
    expect_failure([](api_fixture &f) { f.p.o_stride_t = 8; },
            "overlapping output rows");
    static const float scale[30] = {};
    expect_failure([](api_fixture &f) {
        f.p.delta_size = 4;
        f.p.attn_scale_per_token = scale;
    }, "per-token scale in delta mode");
    {
        api_fixture f;
        EXPECT_EQ(f.run(nullptr), status_t::success);
        f.p.num_seqs = 0;
        EXPECT_EQ(f.run(), status_t::success) << "no sequences";
    }
}

TEST(HstuAttentionApi, KeepsTheCallersThreadCount) {
    if (env_overrides_kernel()) GTEST_SKIP();
    const int before = omp_get_max_threads();
    for (kernel_t kernel : {kernel_t::reference, kernel_t::tiled}) {
        if (kernel == kernel_t::tiled && !has_bf16()) continue;
        for (bool quant : {false, true})
            for (int32_t nt : {0, 2}) {
                omp_set_num_threads(3);
                api_fixture f;
                f.p.kernel = kernel;
                f.p.num_threads = nt;
                f.p.is_qk_quant = f.p.is_pv_quant = quant;
                EXPECT_EQ(f.run(), status_t::success);
                EXPECT_EQ(omp_get_max_threads(), 3)
                        << "kernel " << static_cast<int>(kernel) << " nt " << nt
                        << " quant " << quant;
            }
    }
    omp_set_num_threads(before);
}

TEST(HstuAttentionApi, DeltaLongerThanSequenceFails) {
    hstu_case c {"delta_too_long", {10, 4}};
    c.delta = 5;
    const hstu_problem prob(c);
    std::vector<float> got;
    EXPECT_EQ(prob.run(kernel_t::reference, got), status_t::failure);
}

} // namespace
