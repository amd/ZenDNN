/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

/// WHO MAY MUTATE AN int8 WEIGHT BUFFER, AND WHEN.
///
/// `test_wei_buffer_capacity.cpp` covers the SIZE question -- does the
/// blocked image fit what the caller declared -- as a pure function.  That
/// is necessary and nowhere near sufficient: every bug this file guards
/// passed those assertions, because each one got the size right and the
/// TIMING or the PERMISSION wrong.  An in-place reorder is only correct
/// when three things hold at once, and the size gate is the last of them:
///
///   1. the process weight-cache mode permits it (WC=2, not WC=1/0);
///   2. every OTHER layout has already been packed from the raw bytes;
///   3. the blocked image fits the declared capacity.
///
/// The warmer is where (1) and (2) are decided, so the warmer is where
/// they have to be tested.  These are deliberately unit-level: a wrong
/// answer here surfaces end-to-end only as slightly-off logits or as tens
/// of GB of extra RSS, neither of which a correctness suite would catch
/// and both of which cost hours to bisect.
///
/// Each test names the regression it exists for.

#include <gtest/gtest.h>

#include <cstring>
#include <vector>

#include <omp.h>

#include "moe_test_utils.hpp"

#include "lowoha_operators/matmul/backends/aocl/aocl_kernel.hpp"
#include "lowoha_operators/matmul/group_matmul/prepack/prepack_aocl_dlp.hpp"

namespace {

namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;
using zendnnl::common::bfloat16_t;
using zendnnl::common::data_type_t;
using zendnnl::common::matmul_config_t;
using zendnnl::error_handling::status_t;

constexpr int kExperts = 4;
constexpr int kK = 128;
constexpr int kN = 128;

/// Per-expert s8 weights over-allocated with trailing slack, the way a
/// framework that opts into the capacity protocol lays them out.  The slack
/// must be PER EXPERT: in a packed [E, K, N] tensor the bytes after expert
/// `i` belong to expert `i+1`, so a single tail pad would have the reorder
/// overwrite its neighbour.  Separate allocations here make an overrun a
/// heap error rather than silent cross-expert corruption.
struct PaddedWeights {
    std::vector<std::vector<int8_t>> banks;
    std::vector<const void *> ptrs;
    std::vector<std::vector<int8_t>> pristine;
    std::vector<int> K, N, ldb;
    std::vector<bool> transB;

    /// Generous enough to cover the blocked image plus its int32
    /// compensation row and the 64-byte overscan rounding.
    static size_t capacity_bytes() {
        return static_cast<size_t>(kK) * kN + static_cast<size_t>(kN) * 4
                + 4096;
    }

    PaddedWeights() {
        banks.resize(kExperts);
        pristine.resize(kExperts);
        ptrs.resize(kExperts);
        for (int e = 0; e < kExperts; ++e) {
            banks[e].resize(capacity_bytes());
            // Varied, non-constant content: a reorder of constant bytes can
            // land back on the same image and hide a mutation.
            for (size_t i = 0; i < banks[e].size(); ++i) {
                banks[e][i]
                        = static_cast<int8_t>(((i * 31) + e * 17) % 251 - 125);
            }
            pristine[e] = banks[e];
            ptrs[e] = banks[e].data();
        }
        K.assign(kExperts, kK);
        N.assign(kExperts, kN);
        // Contiguous: the in-place write-back is a flat memcpy, so a strided
        // view is refused outright and would make the test vacuous.
        ldb.assign(kExperts, kN);
        transB.assign(kExperts, false);
    }

    bool any_mutated() const {
        for (int e = 0; e < kExperts; ++e) {
            if (std::memcmp(
                        banks[e].data(), pristine[e].data(), pristine[e].size())
                    != 0) {
                return true;
            }
        }
        return false;
    }
};

/// Restores the two pieces of process-wide state these tests move, so a
/// failure cannot leak a sticky mode into an unrelated suite.
class Int8InPlaceWarm : public ::testing::Test {
protected:
    void SetUp() override {
        reset_grp_matmul_caches();
        prev_wc_ = matmul_config_t::instance().get_weight_cache();
        prev_mixed_ = matmul_config_t::instance().get_grp_auto_mixed_inplace();
    }
    void TearDown() override {
        matmul_config_t::instance().set_weight_cache(prev_wc_);
        matmul_config_t::instance().set_grp_auto_mixed_inplace(prev_mixed_);
        reset_grp_matmul_caches();
    }

    static void set_mode(int32_t wc, bool mixed) {
        matmul_config_t::instance().set_weight_cache(wc);
        matmul_config_t::instance().set_grp_auto_mixed_inplace(mixed);
    }

    static prepack::aocl_dlp::AoclDlpPackProbeStats warm(
            PaddedWeights &w, size_t capacity, bool allow_inplace) {
        prepack::aocl_dlp::AoclDlpPackProbeStats st;
        EXPECT_EQ(prepack::aocl_dlp::warm_pack_all_aocl_dlp_experts_sym_quant(
                          w.ptrs, w.K, w.N, w.ldb, w.transB,
                          /*is_weights_const=*/std::vector<bool> {},
                          /*total_count=*/kExperts, data_type_t::s8, st,
                          /*group_size=*/0, capacity, allow_inplace),
                status_t::success);
        return st;
    }

private:
    int32_t prev_wc_ = 2;
    bool prev_mixed_ = false;
};

// ── (1) the mode must be honoured ───────────────────────────────────────

TEST_F(Int8InPlaceWarm, ExplicitOutOfPlaceModeLeavesWeightsRaw) {
    // WC=1 is the one knob an operator reaches for to say "do not touch my
    // weights".  A warmer that derived its cache type from the declared
    // capacity alone ignored it and mutated anyway -- the caller asked for
    // out-of-place and silently got an in-place rewrite.
    PaddedWeights w;
    set_mode(/*wc=*/1, /*mixed=*/false);

    warm(w, PaddedWeights::capacity_bytes(), /*allow_inplace=*/true);

    EXPECT_FALSE(w.any_mutated())
            << "ZENDNNL_MATMUL_WEIGHT_CACHE=1 must keep the caller's weights "
               "raw no matter how much capacity was declared";
}

TEST_F(Int8InPlaceWarm, PinnedAlgoSkipsTheWarmEntirely) {
    // WC=2 with the mixed flag clear IS a pinned ALGO: cross-warm has already
    // returned early, so there is exactly one layout and nothing to
    // pre-populate ahead of a mutation.  Warming there is actively harmful --
    // it materialises an out-of-place copy of every expert that the runtime
    // then HITS, which both costs a full extra copy and denies the lazy
    // first-call in-place pack a pinned ALGO is entitled to.
    PaddedWeights w;
    set_mode(/*wc=*/2, /*mixed=*/false);

    const auto st = warm(w, PaddedWeights::capacity_bytes(),
            /*allow_inplace=*/true);

    EXPECT_EQ(st.total_attempted, 0)
            << "a pinned ALGO must skip the full-weight warm; anything warmed "
               "here is a resident copy the runtime did not need";
    EXPECT_EQ(st.packed_ok, 0);
    EXPECT_FALSE(w.any_mutated());
}

// ── (2) the ordering contract ───────────────────────────────────────────

TEST_F(Int8InPlaceWarm, RefusedInPlaceLeavesWeightsRawUnderMixedMode) {
    // `allow_inplace=false` is how a caller says "I cannot yet prove every
    // other layout was packed from the raw bytes".  Honouring it is what
    // keeps the decode layout from being snapshotted out of an
    // already-reordered buffer -- the failure mode there is wrong decode
    // output with no error raised, so this must hold even though the mode
    // and the capacity both permit mutation.
    PaddedWeights w;
    set_mode(/*wc=*/2, /*mixed=*/true);

    warm(w, PaddedWeights::capacity_bytes(), /*allow_inplace=*/false);

    EXPECT_FALSE(w.any_mutated())
            << "a warm that was refused in-place must not rewrite the weights";
}

TEST_F(Int8InPlaceWarm, PermittedInPlaceMutatesUnderMixedMode) {
    // The positive case: mode permits, ordering permits, capacity covers.
    // Without this the three negative tests above are satisfiable by a
    // warmer that simply never goes in place, which would quietly give back
    // the whole memory saving the capacity protocol exists for.
    PaddedWeights w;
    set_mode(/*wc=*/2, /*mixed=*/true);

    const auto st = warm(w, PaddedWeights::capacity_bytes(),
            /*allow_inplace=*/true);

    if (st.packed_ok == 0) {
        GTEST_SKIP() << "AOCL DLP sym-quant reorder unavailable in this build";
    }
    EXPECT_TRUE(w.any_mutated())
            << "declared capacity + mixed mode + permission must reorder into "
               "the caller's buffer; staying out-of-place here silently costs "
               "a full extra copy of every expert";
}

// ── (3) back-compat: no declaration, no mutation ────────────────────────

TEST_F(Int8InPlaceWarm, UndeclaredCapacityNeverMutates) {
    // The int8 blocked image is N*4 bytes larger than the logical weight, so
    // a caller that declared nothing has not made room for it.  Every
    // framework that has not opted in must keep the historical behaviour on
    // a rebuild.
    PaddedWeights w;
    set_mode(/*wc=*/2, /*mixed=*/true);

    warm(w, /*capacity=*/0, /*allow_inplace=*/true);

    EXPECT_FALSE(w.any_mutated())
            << "an undeclared buffer must be assumed to be exactly K*N";
}

// ── (4) going in place must not change the answer ───────────────────────

#if ZENDNNL_DEPENDS_AOCLDLP
/// THE correctness question for this feature, and the one nothing else asks.
///
/// Every other test here checks WHETHER a mutation happens.  None checks that
/// the result is still right afterwards -- so a reorder that goes in place and
/// writes a subtly different image would pass the whole suite while silently
/// corrupting inference.  The three bugs already found on this branch were all
/// of that shape: correct size, correct gate, wrong outcome.
///
/// In-place and out-of-place run the IDENTICAL reorder and differ only in
/// where the blocked bytes land, so the two images must be BITWISE equal.
/// That is far stronger than any tolerance and needs no reference kernel: if
/// the bytes match, every consumer of them agrees by construction.
TEST_F(Int8InPlaceWarm, InPlaceBlockedImageIsByteIdenticalToOutOfPlace) {
    // Two independent copies of one weight.  Independent because the in-place
    // run destroys its input, and distinct pointers also keep the two runs in
    // separate LRU slots so neither can answer from the other's cache.
    std::vector<int8_t> pristine(static_cast<size_t>(kK) * kN);
    for (size_t i = 0; i < pristine.size(); ++i) {
        pristine[i] = static_cast<int8_t>((i * 37) % 251 - 125);
    }
    std::vector<int8_t> exact(
            pristine); // exactly k*n -> must stay out-of-place
    std::vector<int8_t> padded(PaddedWeights::capacity_bytes());
    std::memcpy(padded.data(), pristine.data(), pristine.size());

    dlp_metadata_t meta = {};
    dlp_quant_op_t bq = {};
    bq.quant_op_kind = DLP_QUANT_OP_QUANTIZE;
    bq.group_size = kK;
    meta.b_quant_op = &bq;

    const size_t req = aocl_get_reorder_buf_size_s8s8s32os32_sym_quant(
            'r', 'n', 'B', kK, kN, &meta);
    ASSERT_GT(req, 0u);

    const auto run = [&](std::vector<int8_t> &buf, int32_t wc, bool mixed,
                             size_t capacity) -> void * {
        set_mode(wc, mixed);
        void *out = nullptr;
        Key_matmul key(/*transB=*/false, kK, kN, /*ldb=*/kN, buf.data(),
                static_cast<uint32_t>(
                        zendnnl::common::matmul_algo_t::aocl_dlp_blocked),
                std::hash<int64_t> {}(static_cast<int64_t>(kK)));
        EXPECT_TRUE(reorderAndCacheWeightsSymQuant<int8_t>(key, buf.data(), out,
                kK, kN, /*ldb=*/kN, /*order=*/'r', /*trans=*/'n',
                /*mem_format_b=*/'n',
                aocl_get_reorder_buf_size_s8s8s32os32_sym_quant,
                aocl_reorder_s8s8s32os32_sym_quant, &meta, wc, capacity));
        return out;
    };

    void *oop = run(exact, /*wc=*/1, /*mixed=*/false, /*capacity=*/0);
    ASSERT_NE(oop, nullptr);
    ASSERT_NE(oop, exact.data()) << "WC=1 must leave the caller's buffer alone";

    void *ip = run(
            padded, /*wc=*/2, /*mixed=*/true, PaddedWeights::capacity_bytes());
    ASSERT_EQ(ip, padded.data())
            << "declared capacity + mixed mode must reorder INTO the caller's "
               "buffer; if this fails the comparison below is vacuous because "
               "no in-place reorder happened";

    EXPECT_EQ(std::memcmp(oop, ip, req), 0)
            << "the in-place blocked image differs from the out-of-place one "
               "over "
            << req
            << " bytes -- same reorder, same input, different destination, so "
               "any difference is a bug in the in-place write-back";

    // The slack past the blocked image is the caller's; the reorder must not
    // have run past what it declared it needed.
    for (size_t i = req; i < padded.size(); ++i) {
        ASSERT_EQ(padded[i], 0)
                << "in-place reorder wrote past its blocked image at byte " << i
                << " of " << padded.size() << " (declared need was " << req
                << ") -- in a packed [E, ...] tensor those bytes are the next "
                   "expert's weights";
    }
}
#endif

// ── (5) the framework's pad must cover what the backend actually asks ───

#if ZENDNNL_DEPENDS_AOCLDLP
/// A framework cannot see AOCL's blocked size; it can only guess from the
/// layout it knows about -- the logical weight plus one int32 per output
/// column.  zentorch's `_pad_experts_for_inplace` guesses exactly that,
/// rounded to a cache line with one spare:
///
///     pad = roundup64(N * 4) + 64
///
/// If AOCL's real requirement exceeds `K*N + pad` for any shape, the gate
/// refuses and the whole opt-in is silently inert -- the framework pays the
/// padding and gets none of the saving, with no diagnostic beyond an
/// apilog line that blames "blocked size != plain size".
///
/// This is a cross-repo invariant with no compile-time coupling, so it can
/// only be caught by asserting it.  Failing here means the guess is wrong
/// and the backend needs to publish the number instead.
TEST_F(Int8InPlaceWarm, FrameworkPadFormulaCoversAoclRequirement) {
    struct Shape {
        int k, n;
        const char *what;
    };
    // Per-expert MoE projection shapes typical of a W8A8 decode, plus the
    // small shape the rest of this file uses.
    const Shape shapes[] = {
            {2048, 1024, "w13 half"},
            {2048, 256, "narrow"},
            {512, 2048, "w2"},
            {4096, 2048, "wide"},
            {kK, kN, "unit-test shape"},
    };

    for (const auto &s : shapes) {
        dlp_metadata_t meta = {};
        dlp_quant_op_t bq = {};
        bq.quant_op_kind = DLP_QUANT_OP_QUANTIZE;
        bq.group_size = s.k;
        meta.b_quant_op = &bq;

        const size_t req = aocl_get_reorder_buf_size_s8s8s32os32_sym_quant(
                'r', 'n', 'B', s.k, s.n, &meta);
        const size_t reorder_size = (req + 63u) & ~static_cast<size_t>(63u);
        const size_t plain = static_cast<size_t>(s.k) * s.n;
        const size_t framework_pad
                = ((static_cast<size_t>(s.n) * 4u + 63u) / 64u) * 64u + 64u;
        const size_t declared = plain + framework_pad;

        EXPECT_GE(declared, reorder_size)
                << s.what << " K=" << s.k << " N=" << s.n << ": AOCL needs "
                << reorder_size << " B (raw " << req
                << ") but a framework padding by roundup64(N*4)+64 declares "
                << declared << " B -- short by " << (reorder_size - declared)
                << " B, so the in-place gate refuses and the opt-in is inert";
    }
}
#endif

// ── (6) a column TILE is not the buffer the capacity describes ─────────

#if ZENDNNL_DEPENDS_AOCLDLP
/// ALGO 3 splits each expert's N across the team and hands the AOCL backend a
/// COLUMN SLICE of the weight: base `weight[e] + col_start * ldb`, `n =
/// n_tile`, `ldb` unchanged.  `do_tile` re-anchors every other per-tile field
/// to that slice (weight scale, bias, dst); the declared capacity has to be
/// re-anchored too, because it describes the WHOLE expert weight and there is
/// no capacity that describes a slice.
///
/// Inherited unchanged it is not merely over-permissive, it is an overrun the
/// size gate cannot see.  A tile's blocked image needs `K*n_tile + 4*n_tile`
/// bytes, which the parent's capacity trivially covers, so `wei_inplace_fits`
/// says yes and the write-back puts `4*n_tile` bytes PAST the tile's own
/// columns -- on top of the rows the NEXT tile has still to read, or has
/// already blocked.  Whichever tile touches the shared bytes second loses, and
/// thread scheduling picks it, so the corruption is silent and varies run to
/// run.  Only a TRANSPOSED weight is exposed (a `[N, K]` tile is contiguous,
/// so it clears the contiguity gate) -- which is the layout MoE weights
/// actually arrive in.
///
/// The observable: the same call with the capacity UNDECLARED reorders every
/// tile out-of-place from raw weights.  Both runs execute the identical GEMM
/// over identical inputs with an identical N-split, so their outputs must be
/// BITWISE equal -- no tolerance to relax and no reference kernel needed.
TEST_F(Int8InPlaceWarm, ColumnTileMustNotSpendTheWholeWeightCapacity) {
    using zendnnl::lowoha::matmul::group_matmul_direct;
    using zendnnl::lowoha::matmul::matmul_params;

    constexpr int kE = 2;
    constexpr int kMrows = 8;
    constexpr int kKt = 256;
    constexpr int kNt = 512;
    constexpr int kThreads = 64;

    // A single tile IS the whole weight, and then the slice question never
    // arises and this test passes vacuously.  State the requirement instead of
    // assuming it: `aocl_stable_n_thr` is the planner's own tile count.
    ASSERT_GE(zendnnl::lowoha::matmul::aocl_stable_n_thr(kThreads, kNt), 2)
            << "this shape/team must split N into at least two column tiles, "
               "otherwise nothing here exercises a sliced weight";

    const size_t plain = static_cast<size_t>(kKt) * kNt;
    // What a framework that opts in allocates per expert: the logical weight
    // plus room for the blocked image's compensation row and overscan.
    const size_t cap = plain + static_cast<size_t>(kNt) * 4u + 4096u;

    // transB: each expert's weight is stored [N, K] with ldb = K, so a column
    // tile is a CONTIGUOUS row range -- the layout that clears the in-place
    // contiguity gate.
    const auto make_banks = [&](std::vector<std::vector<int8_t>> &banks) {
        banks.assign(kE, std::vector<int8_t>(cap, 0));
        for (int e = 0; e < kE; ++e) {
            for (size_t i = 0; i < plain; ++i) {
                banks[e][i]
                        = static_cast<int8_t>(((i * 31) + e * 17) % 251 - 125);
            }
        }
    };

    std::vector<int8_t> src(static_cast<size_t>(kMrows) * kKt);
    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = static_cast<int8_t>((i * 13) % 127 - 63);
    }
    std::vector<float> src_scale(kMrows, 0.01f); // per-token {M, 1}
    std::vector<float> wei_scale(kNt, 0.02f); // per-channel {1, N}

    const auto run
            = [&](std::vector<std::vector<int8_t>> &banks, size_t capacity,
                      std::vector<std::vector<bfloat16_t>> &dst_out) {
        // Pointer-keyed LRUs: without the reset the second run would HIT the
        // first run's blocked entries and never reorder at all.
        reset_grp_matmul_caches();
        set_mode(/*wc=*/2, /*mixed=*/false);

        std::vector<const void *> wptr(kE), sptr(kE, src.data()),
                bptr(kE, nullptr);
        std::vector<void *> dptr(kE);
        dst_out.assign(kE,
                std::vector<bfloat16_t>(
                        static_cast<size_t>(kMrows) * kNt, bfloat16_t(0.0f)));
        for (int e = 0; e < kE; ++e) {
            wptr[e] = banks[e].data();
            dptr[e] = dst_out[e].data();
        }

        std::vector<matmul_params> params(kE);
        for (auto &p : params) {
            p.dtypes.src = data_type_t::s8;
            p.dtypes.wei = data_type_t::s8;
            p.dtypes.dst = data_type_t::bf16;
            p.dtypes.compute = data_type_t::s8;
            p.dynamic_quant = false;
            p.quant_params.src_scale.buff = src_scale.data();
            p.quant_params.src_scale.dt = data_type_t::f32;
            p.quant_params.src_scale.dims = {kMrows, 1};
            p.quant_params.wei_scale.buff = wei_scale.data();
            p.quant_params.wei_scale.dt = data_type_t::f32;
            p.quant_params.wei_scale.dims = {1, kNt};
            p.weight_cache_type = 2;
            p.num_threads = kThreads;
            p.wei_buffer_capacity_bytes = capacity;
        }

        const std::vector<char> layout(kE, 'r');
        const std::vector<bool> transA(kE, false), transB(kE, true),
                iwc(kE, true);
        const std::vector<float> alpha(kE, 1.0f), beta(kE, 0.0f);
        const std::vector<int> Ms(kE, kMrows), Ns(kE, kNt), Ks(kE, kKt),
                lda(kE, kKt), ldb(kE, kKt), ldc(kE, kNt);

        ASSERT_EQ(group_matmul_direct(layout, transA, transB, Ms, Ns, Ks, alpha,
                          sptr, lda, wptr, ldb, bptr, beta, dptr, ldc, iwc,
                          params),
                status_t::success);
    };

    // ALGO 3 is the only executor that column-splits; the int8 CK packs whole
    // weights, so turn it off to reach the AOCL DLP per-tile sym-quant path.
    moe_test_utils::AlgoEnvGuard algo_guard(3);
    moe_test_utils::CustomKernelInt8Override ck_int8(false);
    const int prev_omp = omp_get_max_threads();
    omp_set_num_threads(kThreads);

    std::vector<std::vector<int8_t>> wei_ref, wei_cap;
    make_banks(wei_ref);
    make_banks(wei_cap);
    std::vector<std::vector<bfloat16_t>> dst_ref, dst_cap;

    run(wei_ref, /*capacity=*/0, dst_ref);
    run(wei_cap, cap, dst_cap);
    omp_set_num_threads(prev_omp);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());

    for (int e = 0; e < kE; ++e) {
        EXPECT_EQ(std::memcmp(dst_cap[e].data(), dst_ref[e].data(),
                          dst_ref[e].size() * sizeof(bfloat16_t)),
                0)
                << "expert " << e
                << ": declaring a weight-buffer capacity changed the ALGO 3 "
                   "output.  The declaration is a promise about free space, "
                   "not a licence to reorder a column tile into the next "
                   "tile's rows";
    }

    // The mechanism, stated directly: the capacity run was never entitled to
    // touch the weights, because no tile owns the buffer the caller described.
    // `wei_ref` is the pristine image -- an undeclared int8 weight can never
    // reorder in place.
    for (int e = 0; e < kE; ++e) {
        EXPECT_EQ(std::memcmp(wei_cap[e].data(), wei_ref[e].data(), cap), 0)
                << "expert " << e
                << ": a column tile reordered into the caller's weight buffer "
                   "on the strength of a capacity declared for the whole "
                   "weight";
    }
}
#endif

// ── (7) a cross-warm that packed nothing has proved nothing ────────────

#if ZENDNNL_DEPENDS_AOCLDLP
/// The prompt prepack runs `cross_warm` FIRST so the decode layout is
/// snapshotted from raw weights, then mutates those weights in place.  The
/// test between the two is what makes the order mean anything, and "no expert
/// was skipped" does not: a warm that attempted NOTHING satisfies it.
///
/// `warm_pack_all_custom_kernel_experts` returns SUCCESS with every counter
/// zero when the host lacks the family's ISA, while `cross_warm` records the
/// regime it chose regardless.  Read as complete, that empty arena licenses
/// the prompt mutation — and decode, forced onto the AOCL fallback by the very
/// ISA gap that emptied the warm, then reorders from mutated bytes.  The ALGO
/// 3 arm of `cross_warm` already refuses on this ground (`primary_attempted`);
/// the prompt arm has the same exposure and needs the same test.
///
/// Driven through the FP16 family because its ISA (AVX-512-FP16) is the one a
/// Zen host does not have, which makes the zero-attempt warm reachable here
/// rather than hypothetical.  What is asserted is the GUARD, not FP16: on an
/// empty cross-warm the process must fall back to out-of-place rather than
/// keep the in-place mode it can no longer justify.
TEST_F(Int8InPlaceWarm, EmptyCrossWarmMustNotLicenseTheInPlacePrompt) {
    namespace ck = zendnnl::lowoha::matmul::custom_kernel;
    if (ck::avx512f16_available()) {
        GTEST_SKIP() << "host HAS AVX-512-FP16, so the FP16 cross-warm packs "
                        "for real and cannot stand in for an empty one";
    }
    if (zendnnl::lowoha::matmul::get_grp_matmul_algo() != 0) {
        GTEST_SKIP() << "cross-warm only runs under AUTO scheduling";
    }

    constexpr int kE = 2;
    constexpr int kKf = 32;
    constexpr int kNf = 64;

    std::vector<std::vector<zendnnl::common::float16_t>> banks(kE,
            std::vector<zendnnl::common::float16_t>(
                    static_cast<size_t>(kKf) * kNf,
                    zendnnl::common::float16_t(0.25f)));
    std::vector<const void *> wptr(kE);
    for (int e = 0; e < kE; ++e) {
        wptr[e] = banks[e].data();
    }
    const std::vector<int> K(kE, kKf), N(kE, kNf), ldb(kE, kNf), M(kE, 4);
    const std::vector<bool> transB(kE, false), iwc(kE, true);

    prepack::PrepackParams pp;
    pp.weight = &wptr;
    pp.K = &K;
    pp.N = &N;
    pp.ldb = &ldb;
    pp.transB = &transB;
    pp.M = &M;
    pp.is_weights_const = &iwc;
    pp.num_ops_total = kE;
    pp.num_ops_active = kE;
    pp.src_dtype = data_type_t::f16;
    pp.wei_dtype = data_type_t::f16;
    pp.dst_dtype = data_type_t::f16;
    pp.custom_kernel_on = true;
    pp.num_threads = 8;
    pp.nr_align = 1;
    // The opt-in: any declared capacity puts this weight on the mixed
    // in-place path, which is what orders cross-warm ahead of the primary.
    pp.wei_buffer_capacity_bytes
            = static_cast<size_t>(kKf) * kNf * sizeof(uint16_t) + 4096u;

    moe_test_utils::CustomKernelOverride ck_master(true);
    moe_test_utils::CustomKernelF16Override ck_f16(true);

    reset_grp_matmul_caches();
    set_mode(/*wc=*/2, /*mixed=*/true);

    prepack::prepack_for_algo_1(pp);

    EXPECT_EQ(matmul_config_t::instance().get_weight_cache(), 1)
            << "cross-warm packed nothing (no AVX-512-FP16, so the CK warm "
               "returned success with every counter zero) yet the process "
               "stayed in the in-place mode.  Nothing pre-warmed the decode "
               "layout, so the next prompt reorder would mutate weights the "
               "decode fallback still has to read raw";
    EXPECT_FALSE(matmul_config_t::instance().get_grp_auto_mixed_inplace())
            << "the mixed-in-place flag must be cleared alongside the "
               "weight-cache downgrade; leaving it set keeps the CK runtime "
               "on its out-of-place branch for a mode that no longer exists";
}
#endif

// ── (8) the decode phase must not read weights the prompt destroyed ────

#if ZENDNNL_DEPENDS_AOCLDLP
/// THE end-to-end question, and the one every test above stops short of.
///
/// The tests so far are unit-level on purpose: they ask whether the warmer
/// mutates, and whether the mutated image is the right one.  None of them runs
/// the sequence production actually runs -- a prompt call that mutates, then a
/// DECODE call on the same weights through the AUTO scheduler -- and none of
/// them looks at the decode NUMBERS.  So a decode layout that misses its cache
/// and re-derives itself from the already-reordered buffer passes the whole
/// file while returning garbage logits.
///
/// The reference is the same decode call with the capacity UNDECLARED: an
/// undeclared int8 weight can never reorder in place (test 3 above), so its
/// decode output is by construction computed from PRISTINE weights.  Both legs
/// execute the identical GEMM over identical inputs, so the outputs must be
/// BITWISE equal -- no tolerance, and no reference kernel to disagree with.
///
/// Swept over the two decode regimes, because they fail for different reasons
/// and cross-warm pre-warms only one of them per call: CK ON puts decode on the
/// custom-kernel pack arena, CK OFF puts it on the AOCL DLP per-tile sym-quant
/// reorder.  Swept over a thread-count change too: the per-tile cache key
/// embeds `n_tile`, which is derived from the team size, while the warm latch
/// keys on `weight_pool_fingerprint` -- which deliberately OMITS num_threads
/// (prepack.cpp) so that a prompt and a decode call on one weight pool contend
/// on one latch.  A second call with a different team therefore finds the pool
/// "already warmed", skips the prepack, and asks for per-tile keys nothing
/// warmed.
TEST_F(Int8InPlaceWarm, AutoPromptThenDecodeMatchesPristineReference) {
    using zendnnl::lowoha::matmul::group_matmul_direct;
    using zendnnl::lowoha::matmul::matmul_params;

    constexpr int kE = 16; // enough experts that decode picks ALGO 3
    constexpr int kKe = 256;
    constexpr int kNe = 512;
    constexpr int kPromptM = 32;
    constexpr int kDecodeM = 1;

    const size_t plain = static_cast<size_t>(kKe) * kNe;
    // What a framework that opts in allocates: the logical weight plus room
    // for the blocked image's compensation row and 64-byte overscan.
    const size_t cap = plain + static_cast<size_t>(kNe) * 4u + 4096u;

    const auto make_banks = [&](std::vector<std::vector<int8_t>> &banks) {
        banks.assign(kE, std::vector<int8_t>(cap, 0));
        for (int e = 0; e < kE; ++e) {
            for (size_t i = 0; i < plain; ++i) {
                banks[e][i]
                        = static_cast<int8_t>(((i * 31) + e * 17) % 251 - 125);
            }
        }
    };

    std::vector<int8_t> src_prompt(static_cast<size_t>(kPromptM) * kKe);
    for (size_t i = 0; i < src_prompt.size(); ++i) {
        src_prompt[i] = static_cast<int8_t>((i * 13) % 127 - 63);
    }
    std::vector<int8_t> src_decode(static_cast<size_t>(kDecodeM) * kKe);
    for (size_t i = 0; i < src_decode.size(); ++i) {
        src_decode[i] = static_cast<int8_t>((i * 7) % 127 - 63);
    }
    std::vector<float> src_scale(kPromptM, 0.01f); // per-token {M, 1}
    std::vector<float> wei_scale(kNe, 0.02f); // per-channel {1, N}

    // One `group_matmul_direct` call.  `threads` goes through
    // `params[0].num_threads`, which is what `resolve_num_threads` reads, so
    // the team size is a property of the call rather than of the process.
    const auto phase
            = [&](std::vector<std::vector<int8_t>> &banks, size_t capacity,
                      int Mrows, const int8_t *srcp, int threads,
                      std::vector<std::vector<bfloat16_t>> &dst_out) {
        std::vector<const void *> wptr(kE), sptr(kE, srcp), bptr(kE, nullptr);
        std::vector<void *> dptr(kE);
        dst_out.assign(kE,
                std::vector<bfloat16_t>(
                        static_cast<size_t>(Mrows) * kNe, bfloat16_t(0.0f)));
        for (int e = 0; e < kE; ++e) {
            wptr[e] = banks[e].data();
            dptr[e] = dst_out[e].data();
        }

        std::vector<matmul_params> params(kE);
        for (auto &p : params) {
            p.dtypes.src = data_type_t::s8;
            p.dtypes.wei = data_type_t::s8;
            p.dtypes.dst = data_type_t::bf16;
            p.dtypes.compute = data_type_t::s8;
            p.dynamic_quant = false;
            p.quant_params.src_scale.buff = src_scale.data();
            p.quant_params.src_scale.dt = data_type_t::f32;
            p.quant_params.src_scale.dims = {Mrows, 1};
            p.quant_params.wei_scale.buff = wei_scale.data();
            p.quant_params.wei_scale.dt = data_type_t::f32;
            p.quant_params.wei_scale.dims = {1, kNe};
            p.weight_cache_type = 2;
            p.num_threads = threads;
            p.wei_buffer_capacity_bytes = capacity;
        }

        const std::vector<char> layout(kE, 'r');
        // transB: each expert is stored [N, K] with ldb = K, so the weight is
        // contiguous -- the layout that clears the in-place contiguity gate.
        const std::vector<bool> transA(kE, false), transB(kE, true),
                iwc(kE, true);
        const std::vector<float> alpha(kE, 1.0f), beta(kE, 0.0f);
        const std::vector<int> Ms(kE, Mrows), Ns(kE, kNe), Ks(kE, kKe),
                lda(kE, kKe), ldb(kE, kKe), ldc(kE, kNe);

        ASSERT_EQ(group_matmul_direct(layout, transA, transB, Ms, Ns, Ks, alpha,
                          sptr, lda, wptr, ldb, bptr, beta, dptr, ldc, iwc,
                          params),
                status_t::success);
    };

    struct Regime {
        bool ck_int8;
        int prompt_threads;
        int decode_threads;
        const char *what;
    };
    // `aocl_stable_n_thr(nt, N) = max(1, nt / target_slots)`, so 32 and 64
    // threads give DIFFERENT per-tile splits; 32/32 is the control.
    const Regime regimes[] = {
            {true, 32, 32, "CK decode, one team"},
            {false, 32, 32, "AOCL per-tile decode, one team"},
            {false, 32, 64, "AOCL per-tile decode, team grew 32 -> 64"},
    };

    // AUTO is what makes the prompt and the decode take DIFFERENT scheduling
    // ALGOs, which is the whole reason cross-warm and the in-place ordering
    // exist.  A pinned ALGO would make this test vacuous.
    moe_test_utils::AlgoEnvGuard algo_guard(0);

    int non_vacuous = 0;
    for (const auto &r : regimes) {
        SCOPED_TRACE(r.what);
        moe_test_utils::CustomKernelInt8Override ck_int8(r.ck_int8);

        // Reference leg: capacity UNDECLARED, so no int8 reorder anywhere may
        // go in place and the decode output is computed from pristine weights.
        std::vector<std::vector<int8_t>> wei_ref;
        std::vector<std::vector<bfloat16_t>> dst_ref_prompt, dst_ref_decode;
        make_banks(wei_ref);
        reset_grp_matmul_caches();
        set_mode(/*wc=*/2, /*mixed=*/false); // let the dispatcher decide
        phase(wei_ref, /*capacity=*/0, kPromptM, src_prompt.data(),
                r.prompt_threads, dst_ref_prompt);
        phase(wei_ref, /*capacity=*/0, kDecodeM, src_decode.data(),
                r.decode_threads, dst_ref_decode);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());

        std::vector<std::vector<int8_t>> wei_cap;
        std::vector<std::vector<bfloat16_t>> dst_cap_prompt, dst_cap_decode;
        make_banks(wei_cap);
        reset_grp_matmul_caches();
        set_mode(/*wc=*/2, /*mixed=*/false);
        phase(wei_cap, cap, kPromptM, src_prompt.data(), r.prompt_threads,
                dst_cap_prompt);
        // State the precondition rather than assuming it: if the prompt did
        // NOT mutate, every comparison below is satisfied trivially and this
        // test proves nothing.
        bool mutated = false;
        for (int e = 0; e < kE && !mutated; ++e) {
            mutated = std::memcmp(wei_cap[e].data(), wei_ref[e].data(), cap)
                    != 0;
        }
        if (!mutated) {
            // Not a failure: refusing to mutate is the correct answer whenever
            // the prepack cannot prove the decode layout was captured from raw
            // bytes.  Nothing to compare, so move on -- but if EVERY regime
            // declines, the whole test proved nothing and says so below.
            continue;
        }
        ++non_vacuous;
        phase(wei_cap, cap, kDecodeM, src_decode.data(), r.decode_threads,
                dst_cap_decode);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());

        for (int e = 0; e < kE; ++e) {
            EXPECT_EQ(std::memcmp(dst_cap_decode[e].data(),
                              dst_ref_decode[e].data(),
                              dst_ref_decode[e].size() * sizeof(bfloat16_t)),
                    0)
                    << "expert " << e
                    << ": the DECODE output changed once the prompt was "
                       "allowed to reorder into the caller's weight buffer.  "
                       "Decode re-derived its layout from bytes the prompt had "
                       "already overwritten";
            EXPECT_EQ(std::memcmp(dst_cap_prompt[e].data(),
                              dst_ref_prompt[e].data(),
                              dst_ref_prompt[e].size() * sizeof(bfloat16_t)),
                    0)
                    << "expert " << e
                    << ": the PROMPT output itself changed, so the in-place "
                       "reorder does not produce the same blocked image as the "
                       "out-of-place one";
        }
    }

    if (non_vacuous == 0) {
        GTEST_SKIP() << "no configuration reordered in place, so nothing here "
                        "compared a decode output against pristine weights";
    }
}
#endif

} // namespace
