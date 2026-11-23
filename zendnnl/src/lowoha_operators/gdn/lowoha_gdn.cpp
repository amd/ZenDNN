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

#include "lowoha_gdn.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <numeric>
#include <omp.h>
#include <sstream>
#include <vector>
#include "common/logging.hpp"
#include "common/zendnnl_global.hpp"
#include "lowoha_operators/common/omp_thread_control.hpp"
#include "lowoha_operators/common/operator_instrumentation.hpp"
#include "lowoha_operators/gdn/lowoha_gdn_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace gdn {

namespace {

// Automatic selection takes the chunked prefill once the prefill sequences of
// a call average this many tokens. Its GEMMs run FP32 FMA for f16 and f32
// sources, half the work per instruction of vdpbf16ps (bf16), so the
// crossover with the recurrent form moves up.
int64_t chunked_min_avg_tokens(data_type_t src_dt) {
    return src_dt == data_type_t::bf16   ? 16
            : src_dt == data_type_t::f16 ? 32
                                         : 24;
}

// Share of a chunk's time, for a whole head row, spent on what does not
// depend on the value columns: the q/k conv and norm, the gates, K K^T,
// Q K^T, the decays, T and W. Measured with benchdnn on one Turin core with
// Qwen3.6 shapes (K = V = 128; 0.44 bf16, 0.30 f16 and f32): bf16
// operands halve the cost of the products, so the rest weighs more.
double chunk_key_share(data_type_t src_dt) {
    return src_dt == data_type_t::bf16 ? 0.44 : 0.30;
}

// More value blocks must shorten the estimate by this fraction, which leaves
// out the extra reads of q and k and the per-unit setup.
constexpr double kChunkBlockMinGain = 0.05;

bool cpu_has_native_isa() {
    static const bool ok = []() {
        const auto &pi = zendnnl::common::zendnnl_platform_info();
        return pi.get_avx512f_status() && pi.get_avx512_bw_vl_status()
                && pi.get_avx512_bf16_status();
    }();
    return ok;
}

// ZENDNNL_GDN_KERNEL=native|reference overrides params.kernel.
gdn_kernel_t env_kernel_override() {
    static const gdn_kernel_t k = []() {
        const char *env = std::getenv("ZENDNNL_GDN_KERNEL");
        if (env == nullptr || *env == '\0') { return gdn_kernel_t::none; }
        if (std::strcmp(env, "native") == 0) { return gdn_kernel_t::native; }
        if (std::strcmp(env, "reference") == 0) {
            return gdn_kernel_t::reference;
        }
        log_warning("gdn: ignoring ZENDNNL_GDN_KERNEL=", env,
                " (expected native or reference)");
        return gdn_kernel_t::none;
    }();
    return k;
}

// ZENDNNL_GDN_PREFILL=recurrent|chunked overrides params.prefill_algo.
gdn_prefill_algo_t env_prefill_override() {
    static const gdn_prefill_algo_t algo = []() {
        const char *env = std::getenv("ZENDNNL_GDN_PREFILL");
        if (env == nullptr || *env == '\0') {
            return gdn_prefill_algo_t::automatic;
        }
        if (std::strcmp(env, "recurrent") == 0) {
            return gdn_prefill_algo_t::recurrent;
        }
        if (std::strcmp(env, "chunked") == 0) {
            return gdn_prefill_algo_t::chunked;
        }
        log_warning("gdn: ignoring ZENDNNL_GDN_PREFILL=", env,
                " (expected recurrent or chunked)");
        return gdn_prefill_algo_t::automatic;
    }();
    return algo;
}

// A field left at 0 passes, so default params ask about the CPU alone.
bool native_supports(const gdn_params &p) {
    const auto head_ok = [](int64_t d) {
        return d >= 0 && d % 16 == 0 && d <= kMaxHeadDim;
    };
    return cpu_has_native_isa() && head_ok(p.head_k_dim)
            && head_ok(p.head_v_dim) && p.conv_width >= 0
            && p.conv_width <= kMaxConvWidth;
}

gdn_kernel_t select_kernel(const gdn_params &p) {
    gdn_kernel_t k = env_kernel_override();
    if (k == gdn_kernel_t::none) { k = p.kernel; }
    if (k == gdn_kernel_t::none) {
        k = native_supports(p) ? gdn_kernel_t::native : gdn_kernel_t::reference;
    }
    if (k == gdn_kernel_t::native && !native_supports(p)) {
        apilog_warning(
                "gdn: native kernel does not support this shape / "
                "CPU, using reference");
        k = gdn_kernel_t::reference;
    }
    return k;
}

// Prefill algorithm of the native kernel for np prefill sequences holding nt
// tokens. The chunked form needs head dims that are multiples of 32 and, with
// a fused norm, the rounded core output (normalised in a pass over the rows);
// calls that do not qualify run the recurrent form.
gdn_prefill_algo_t select_prefill_algo(
        const gdn_params &p, int64_t np, int64_t nt) {
    gdn_prefill_algo_t algo = env_prefill_override();
    if (algo == gdn_prefill_algo_t::automatic) { algo = p.prefill_algo; }
    if (p.head_k_dim % 32 != 0 || p.head_v_dim % 32 != 0
            || (p.norm_weight != nullptr && !p.round_core_out)) {
        if (algo == gdn_prefill_algo_t::chunked) {
            apilog_warning(
                    "gdn: chunked prefill needs head dims that are multiples "
                    "of 32 (and round_core_out with a fused norm), using "
                    "recurrent");
        }
        return gdn_prefill_algo_t::recurrent;
    }
    if (algo != gdn_prefill_algo_t::automatic) { return algo; }
    return nt >= chunked_min_avg_tokens(p.src_dt) * np
            ? gdn_prefill_algo_t::chunked
            : gdn_prefill_algo_t::recurrent;
}

// cu_seqlens must start at 0, never decrease, give every decode one token and
// end at num_tokens; without it every sequence is one token.
status_t validate_offsets(const gdn_params &p) {
    const int32_t *cu = p.cu_seqlens;
    if (cu == nullptr) {
        if (p.num_tokens != p.batch) {
            log_error("gdn_direct: num_tokens (", p.num_tokens,
                    ") must equal batch (", p.batch,
                    ") when cu_seqlens is null");
            return status_t::failure;
        }
        return status_t::success;
    }
    if (cu[0] != 0) {
        log_error("gdn_direct: cu_seqlens[0] must be 0, got ", cu[0]);
        return status_t::failure;
    }
    for (int64_t i = 0; i < p.batch; ++i) {
        const int64_t len
                = static_cast<int64_t>(cu[i + 1]) - static_cast<int64_t>(cu[i]);
        if (len < 0) {
            log_error(
                    "gdn_direct: cu_seqlens must be non-decreasing "
                    "(sequence ",
                    i, ")");
            return status_t::failure;
        }
        if (i < p.num_decodes && len != 1) {
            log_error("gdn_direct: decode sequence ", i, " has ", len,
                    " tokens; cu_seqlens[num_decodes] must equal "
                    "num_decodes");
            return status_t::failure;
        }
    }
    if (cu[p.batch] != p.num_tokens) {
        log_error("gdn_direct: num_tokens (", p.num_tokens,
                ") must equal cu_seqlens[batch] (", cu[p.batch], ")");
        return status_t::failure;
    }
    return status_t::success;
}

// Whether a prefill of the call continues from its slot. An empty one reads
// nothing, whatever its flag. Runs after validate_offsets.
bool reads_initial_state(const gdn_params &p, const int32_t *state_indices) {
    if (p.has_initial_state == nullptr) { return false; }
    const int32_t *cu = p.cu_seqlens;
    for (int64_t i = p.num_decodes; i < p.batch; ++i) {
        const bool has_tokens = cu == nullptr || cu[i + 1] > cu[i];
        if (has_tokens && p.has_initial_state[i]
                && state_indices[i] != p.pad_slot_id) {
            return true;
        }
    }
    return false;
}

// Speculative verify (see gdn_params::spec_state_indices). Runs after
// validate_offsets.
status_t validate_spec(const gdn_params &p) {
    constexpr const char *op = "gdn_direct";
    if (p.spec_state_stride < 1 || !p.num_accepted_tokens || !p.cu_seqlens) {
        log_error(op,
                ": speculative verify needs spec_state_stride >= 1, "
                "num_accepted_tokens and cu_seqlens");
        return status_t::failure;
    }
    if (p.num_decodes != 0 || !p.update_state || p.separate_qkv()
            || p.precomputed_gates()) {
        log_error(op,
                ": speculative verify takes no decodes, update_state = true, "
                "packed rows and raw gates");
        return status_t::failure;
    }
    if (p.conv_width == 1) {
        log_error(op, ": speculative verify needs conv_width 0 or at least 2");
        return status_t::failure;
    }
    const int64_t SS = p.spec_state_stride;
    for (int64_t i = 0; i < p.batch; ++i) {
        const int64_t len = static_cast<int64_t>(p.cu_seqlens[i + 1])
                - static_cast<int64_t>(p.cu_seqlens[i]);
        const int32_t *slots = p.spec_state_indices + i * SS;
        if (len > SS) {
            log_error(op, ": sequence ", i, " has ", len,
                    " tokens, more than spec_state_stride (", SS, ")");
            return status_t::failure;
        }
        if (len == 0 || slots[0] == p.pad_slot_id) { continue; }
        const int32_t acc = p.num_accepted_tokens[i];
        if (acc < 1 || acc > SS) {
            log_error(op, ": num_accepted_tokens[", i, "] = ", acc,
                    " is outside [1, ", SS, "]");
            return status_t::failure;
        }
        // The slot the state comes from, then those the tokens write.
        for (int64_t t = -1; t < len; ++t) {
            const int64_t col = t < 0 ? acc - 1 : t;
            const int32_t s = slots[col];
            if (s == p.pad_slot_id || s < 0
                    || (p.num_state_slots > 0 && s >= p.num_state_slots)) {
                log_error(op, ": spec_state_indices[", i, "][", col, "] = ", s,
                        " is not a slot");
                return status_t::failure;
            }
        }
    }
    return status_t::success;
}

status_t validate_params(const gdn_params &p, const void *mixed_qkv,
        const void *a, const void *b, const void *A_log, const void *dt_bias,
        const void *conv_weight, const void *conv_state, const void *ssm_state,
        const int32_t *state_indices, const void *output) {
    constexpr const char *op = "gdn_direct";
    if (p.batch < 0 || p.num_k_heads <= 0 || p.num_v_heads <= 0
            || p.head_k_dim <= 0 || p.head_v_dim <= 0) {
        log_error(op, ": invalid shape");
        return status_t::failure;
    }
    if (p.num_v_heads % p.num_k_heads != 0) {
        log_error(op, ": num_v_heads must be a multiple of num_k_heads");
        return status_t::failure;
    }
    if (p.num_decodes < 0 || p.num_decodes > p.batch) {
        log_error(op, ": num_decodes (", p.num_decodes,
                ") must lie in [0, batch]");
        return status_t::failure;
    }
    if (!is_gdn_float(p.src_dt) || !is_gdn_float(p.state_dt)
            || !is_gdn_float(p.a_log_dt) || !is_gdn_float(p.dt_bias_dt)
            || !is_gdn_float(p.a_type()) || !is_gdn_float(p.b_type())) {
        log_error(op, ": data types must be f32, bf16 or f16");
        return status_t::failure;
    }
    if (p.gate_input != gdn_gate_input_t::raw
            && p.gate_input != gdn_gate_input_t::precomputed) {
        log_error(op, ": invalid gate_input");
        return status_t::failure;
    }
    if (p.allow_neg_eigval && p.precomputed_gates()) {
        log_error(op,
                ": allow_neg_eigval takes raw gates; precomputed gates give "
                "beta itself");
        return status_t::failure;
    }
    if (validate_offsets(p) != status_t::success) { return status_t::failure; }
    if (p.batch == 0) { return status_t::success; }
    if (p.spec_verify() && validate_spec(p) != status_t::success) {
        return status_t::failure;
    }
    if (!p.update_state && p.num_decodes > 0) {
        log_error(op,
                ": update_state = false needs a prefill-only call "
                "(num_decodes = 0)");
        return status_t::failure;
    }
    if ((p.separate_qkv() || p.precomputed_gates()) && p.num_decodes > 0) {
        log_error(op,
                ": separate q / k / v and precomputed gates need a "
                "prefill-only call (num_decodes = 0)");
        return status_t::failure;
    }
    if (p.separate_qkv() && (!p.k || !p.v || p.conv_width != 0)) {
        log_error(op, ": separate q / k / v need k, v and conv_width = 0");
        return status_t::failure;
    }
    // Precomputed gates read neither A_log nor dt_bias, and speculative
    // verify takes its slots from spec_state_indices.
    if (!mixed_qkv || !a || !b || (!state_indices && !p.spec_verify())
            || !output || (!p.precomputed_gates() && (!A_log || !dt_bias))) {
        log_error(op, ": null tensor pointer");
        return status_t::failure;
    }
    // The pools may be null when the call neither reads nor writes them.
    const bool pools = p.update_state || reads_initial_state(p, state_indices);
    if (pools && !ssm_state) {
        log_error(op, ": null tensor pointer");
        return status_t::failure;
    }
    if (p.conv_width < 0
            || (p.conv_width > 0 && (!conv_weight || (pools && !conv_state)))) {
        log_error(op, ": conv_width > 0 needs conv_weight and conv_state");
        return status_t::failure;
    }
    if (p.num_threads < 0) {
        log_error(op, ": num_threads must be >= 0");
        return status_t::failure;
    }
    if (p.state_layout != gdn_state_layout_t::kv
            && p.state_layout != gdn_state_layout_t::vk) {
        log_error(op, ": invalid state_layout");
        return status_t::failure;
    }
    if (p.prefill_algo != gdn_prefill_algo_t::automatic
            && p.prefill_algo != gdn_prefill_algo_t::recurrent
            && p.prefill_algo != gdn_prefill_algo_t::chunked) {
        log_error(op, ": invalid prefill_algo");
        return status_t::failure;
    }
    const int64_t HV = p.num_v_heads, H = p.num_k_heads;
    const int64_t K = p.head_k_dim, V = p.head_v_dim;
    // A token's heads must not overlap each other or the next token.
    const auto rows_ok = [](int64_t st, int64_t sh, int64_t heads, int64_t d) {
        return sh >= d && st >= heads * sh;
    };
    const bool qkv_ok = p.separate_qkv()
            ? rows_ok(p.qkv_stride, p.q_stride_h, H, K)
                    && rows_ok(p.k_stride_t, p.k_stride_h, H, K)
                    && rows_ok(p.v_stride_t, p.v_stride_h, HV, V)
            : p.qkv_stride >= p.conv_dim();
    if (!qkv_ok || p.a_stride < HV || p.b_stride < HV
            || p.out_stride_h < p.head_v_dim
            || p.out_stride_t < HV * p.out_stride_h) {
        log_error(op, ": row strides smaller than the packed widths");
        return status_t::failure;
    }
    if (p.ssm_state_head_stride < p.head_k_dim * p.head_v_dim
            || p.ssm_state_slot_stride < HV * p.ssm_state_head_stride) {
        log_error(op, ": ssm_state strides overlap heads or slots");
        return status_t::failure;
    }
    if (p.conv_width > 1) {
        // Taps x channels of one slot must not overlap each other or the
        // next slot, in either the SD (dim contiguous) or DS layout.
        const int64_t taps = p.conv_taps(), D = p.conv_dim();
        const int64_t ts = p.conv_state_tap_stride;
        const int64_t ds = p.conv_state_dim_stride;
        const bool disjoint
                = ts > 0 && ds > 0 && (ts >= D * ds || ds >= taps * ts);
        if (!disjoint
                || p.conv_state_slot_stride
                        < (taps - 1) * ts + (D - 1) * ds + 1) {
            log_error(
                    op, ": conv_state strides overlap taps, channels or slots");
            return status_t::failure;
        }
    }
    if (p.norm_weight != nullptr
            && (p.z == nullptr || !is_gdn_float(p.norm_weight_dt)
                    || p.z_stride_h < p.head_v_dim
                    || p.z_stride_t < HV * p.z_stride_h
                    || (p.norm_gate != gdn_norm_gate_t::silu
                            && p.norm_gate != gdn_norm_gate_t::sigmoid))) {
        log_error(op,
                ": fused norm needs z with valid strides, a float "
                "norm_weight_dt and a silu or sigmoid gate");
        return status_t::failure;
    }
    if (p.num_state_slots > 0 && !p.spec_verify()) {
        for (int64_t i = 0; i < p.batch; ++i) {
            const int32_t s = state_indices[i];
            if (s != p.pad_slot_id && (s < 0 || s >= p.num_state_slots)) {
                log_error(op, ": state_indices[", i, "] = ", s,
                        " is outside [0, num_state_slots)");
                return status_t::failure;
            }
        }
    }
    return status_t::success;
}

void to_f32(
        const void *src, data_type_t dt, int64_t n, std::vector<float> &dst) {
    dst.resize(n);
    for (int64_t i = 0; i < n; ++i) {
        dst[i] = load_f32(src, dt, i);
    }
}

// split: that of a chunked prefill over `pairs` (sequence, value head)
// pairs, or null.
void log_call(const gdn_params &p, gdn_kernel_t kernel,
        const char *prefill_algo, const gdn_chunk_split_t *split, int64_t pairs,
        bool is_profile, zendnnl::profile::profiler_t &profiler) {
    std::ostringstream ss;
    ss << "LOWOHA gdn_direct: batch=" << p.batch
       << ", num_decodes=" << p.num_decodes << ", num_tokens=" << p.num_tokens
       << ", varlen=" << (p.cu_seqlens ? "true" : "false")
       << ", num_k_heads=" << p.num_k_heads << ", num_v_heads=" << p.num_v_heads
       << ", head_k_dim=" << p.head_k_dim << ", head_v_dim=" << p.head_v_dim
       << ", conv_width=" << p.conv_width << ", state_layout="
       << (p.state_layout == gdn_state_layout_t::kv ? "kv" : "vk")
       << ", fused_norm=" << (p.norm_weight ? "true" : "false")
       << ", update_state=" << (p.update_state ? "true" : "false")
       << ", spec_state_stride="
       << (p.spec_verify() ? p.spec_state_stride : int64_t(0))
       << ", qkv=" << (p.separate_qkv() ? "separate" : "packed")
       << ", gates=" << (p.precomputed_gates() ? "precomputed" : "raw")
       << (p.allow_neg_eigval ? ", allow_neg_eigval=true" : "")
       << ", src_dt=" << dtype_info(p.src_dt)
       << ", state_dt=" << dtype_info(p.state_dt)
       << ", kernel=" << kernel_to_string(kernel)
       << ", prefill_algo=" << prefill_algo;
    if (split != nullptr) {
        const int64_t n_split = split->blocks > 1
                ? pairs - std::min(split->first_split, pairs)
                : 0;
        ss << ", chunk_v_blocks=" << split->blocks
           << ", chunk_split_pairs=" << n_split << "/" << pairs;
    }
    apilog_info(ss.str());
    if (is_profile) {
        profilelog_verbose(ss.str(), ", time=", profiler.tbp_elapsedtime(),
                profiler.get_res_str());
    }
}

} // namespace

// A unit of a c-chunk sequence costs c for a whole head and
// c * (key + (1 - key) / blocks) for a block, key = chunk_key_share: every
// block recomputes the part of a chunk that does not depend on the value
// columns. The units run in order on schedule(dynamic, 1), each on the first
// thread that is free; that schedule is simulated for whole heads, for the
// heads of the last round split (so that round fills the threads) and for
// every head split, in 2 and in 4 blocks.
gdn_chunk_split_t gdn_chunk_split(const gdn_params &p,
        const int32_t *state_indices, const int32_t *offsets, int64_t s0,
        int64_t s1, int32_t nthr) {
    const int64_t HV = p.num_v_heads;
    const int64_t pairs = HV * (s1 - s0);
    const gdn_chunk_split_t whole {1, pairs};
    // Chunks per sequence, 0 for those that do not run.
    static thread_local std::vector<int64_t> chunks;
    chunks.assign(static_cast<size_t>(s1 - s0), 0);
    int64_t active = 0;
    for (int64_t si = s0; si < s1; ++si) {
        const int64_t len = static_cast<int64_t>(offsets[si + 1]) - offsets[si];
        if (state_indices[si] == p.pad_slot_id || len <= 0) { continue; }
        chunks[si - s0] = (len + kChunkLen - 1) / kChunkLen;
        active += HV;
    }
    // Beyond 8 heads per thread the threads stay busy as they are.
    if (nthr <= 1 || active == 0 || active > 8 * static_cast<int64_t>(nthr)) {
        return whole;
    }
    const double key = chunk_key_share(p.src_dt);
    // busy: min-heap of the threads' finish times.
    static thread_local std::vector<double> busy;
    const auto finish = [&](const gdn_chunk_split_t &s) {
        const double narrow = key + (1.0 - key) / static_cast<double>(s.blocks);
        busy.assign(static_cast<size_t>(nthr), 0.0);
        double end = 0.0;
        for (int64_t pr = 0; pr < pairs; ++pr) {
            const double c = static_cast<double>(chunks[pr / HV]);
            if (c == 0.0) { continue; }
            const bool split = pr >= s.first_split;
            for (int64_t j = 0; j < (split ? s.blocks : 1); ++j) {
                std::pop_heap(busy.begin(), busy.end(), std::greater<double>());
                busy.back() += split ? c * narrow : c;
                end = std::max(end, busy.back());
                std::push_heap(
                        busy.begin(), busy.end(), std::greater<double>());
            }
        }
        return end;
    };
    // The last active % nthr running pairs make up the last round.
    int64_t tail_first = pairs;
    for (int64_t pr = pairs - 1, n = active % nthr; pr >= 0 && n > 0; --pr) {
        if (chunks[pr / HV] > 0) {
            tail_first = pr;
            --n;
        }
    }
    gdn_chunk_split_t best = whole;
    double best_time = finish(whole);
    for (int64_t b = 2; b <= 4 && p.head_v_dim % (32 * b) == 0; b *= 2) {
        for (const int64_t first : {tail_first, int64_t(0)}) {
            if (first >= pairs) { continue; }
            const double t = finish({b, first});
            if (t < (1.0 - kChunkBlockMinGain) * best_time) {
                best = {b, first};
                best_time = t;
            }
        }
    }
    return best;
}

status_t gdn_direct(const void *mixed_qkv, const void *a, const void *b,
        const void *A_log, const void *dt_bias, const void *conv_weight,
        const void *conv_bias, void *conv_state, void *ssm_state,
        const int32_t *state_indices, void *output, gdn_params &params) {
    zendnnl::profile::profiler_t profiler;
    const bool is_profile = is_profile_enabled();
    if (is_profile) { profiler.tbp_start(); }

    status_t st = op_instrumentation::validate([&]() {
        return validate_params(params, mixed_qkv, a, b, A_log, dt_bias,
                conv_weight, conv_state, ssm_state, state_indices, output);
    });
    if (st != status_t::success) { return st; }
    if (params.batch == 0) { return status_t::success; }

    static thread_local std::vector<float> a_log_f, dt_bias_f;
    const int64_t n_gate = params.precomputed_gates() ? 0 : params.num_v_heads;
    to_f32(A_log, params.a_log_dt, n_gate, a_log_f);
    to_f32(dt_bias, params.dt_bias_dt, n_gate, dt_bias_f);
    const gdn_tensors_t io {mixed_qkv, a, b, a_log_f.data(), dt_bias_f.data(),
            conv_weight, conv_bias, conv_state, ssm_state, state_indices,
            output};

    const int64_t nd = params.num_decodes, ns = params.batch;
    // Prefill offsets; without cu_seqlens sequence i is row i.
    const int32_t *offsets = params.cu_seqlens;
    static thread_local std::vector<int32_t> unit_offsets;
    if (offsets == nullptr && nd < ns) {
        unit_offsets.resize(static_cast<size_t>(ns + 1));
        std::iota(unit_offsets.begin(), unit_offsets.end(), 0);
        offsets = unit_offsets.data();
    }

    const gdn_kernel_t kernel = select_kernel(params);
    const bool native = kernel == gdn_kernel_t::native;
    if (params.spec_verify()) {
        st = native ? gdn_verify_native(io, ns, offsets, params)
                    : gdn_verify_reference(io, ns, offsets, params);
        if (st != status_t::success) { return st; }
        if (is_profile) { profiler.tbp_stop(); }
        if (apilog_info_enabled() || is_profile) {
            log_call(params, kernel, "none", nullptr, 0, is_profile, profiler);
        }
        return status_t::success;
    }
    gdn_prefill_algo_t algo = gdn_prefill_algo_t::recurrent;
    gdn_chunk_split_t split {1, 0};
    if (native && nd < ns) {
        algo = select_prefill_algo(params, ns - nd, offsets[ns] - offsets[nd]);
        if (algo == gdn_prefill_algo_t::chunked) {
            split = gdn_chunk_split(params, state_indices, offsets, nd, ns,
                    resolve_num_threads(
                            params.num_threads, omp_get_max_threads()));
        }
    }

    if (nd > 0) {
        st = native ? gdn_decode_native(io, nd, params)
                    : gdn_decode_reference(io, nd, params);
        if (st != status_t::success) { return st; }
    }
    if (nd < ns) {
        st = native
                ? gdn_prefill_native(io, nd, ns, offsets, algo, split, params)
                : gdn_prefill_reference(io, nd, ns, offsets, params);
        if (st != status_t::success) { return st; }
    }

    if (is_profile) { profiler.tbp_stop(); }
    if (apilog_info_enabled() || is_profile) {
        const bool chunked
                = native && nd < ns && algo == gdn_prefill_algo_t::chunked;
        log_call(params, kernel,
                nd < ns ? prefill_algo_to_string(algo) : "none",
                chunked ? &split : nullptr, (ns - nd) * params.num_v_heads,
                is_profile, profiler);
    }
    return status_t::success;
}

bool gdn_native_supported(const gdn_params &params) {
    return native_supports(params);
}

} // namespace gdn
} // namespace lowoha
} // namespace zendnnl
