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

#include <algorithm>
#include <vector>
#include "lowoha_operators/gdn/lowoha_gdn_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace gdn {

namespace {

inline const char *byte_ptr(const void *p) {
    return static_cast<const char *>(p);
}

inline void *conv_state_elem(void *conv_state, const gdn_params &p,
        int64_t slot, int64_t tap, int64_t c) {
    const int64_t off = slot * p.conv_state_slot_stride
            + tap * p.conv_state_tap_stride + c * p.conv_state_dim_stride;
    return static_cast<char *>(conv_state) + off * dt_size(p.src_dt);
}

inline int64_t state_offset(const gdn_params &p, int64_t k, int64_t v) {
    return p.state_layout == gdn_state_layout_t::kv ? k * p.head_v_dim + v
                                                    : v * p.head_k_dim + k;
}

// Row r of separate q, k and v as one q | k | v row in FP32.
void separate_row(const gdn_tensors_t &io, int64_t r, std::vector<float> &x,
        const gdn_params &p) {
    const int64_t H = p.num_k_heads, HV = p.num_v_heads;
    const int64_t K = p.head_k_dim, V = p.head_v_dim;
    for (int64_t h = 0; h < H; ++h) {
        for (int64_t i = 0; i < K; ++i) {
            x[h * K + i] = load_f32(io.mixed_qkv, p.src_dt,
                    r * p.qkv_stride + h * p.q_stride_h + i);
            x[H * K + h * K + i] = load_f32(
                    p.k, p.src_dt, r * p.k_stride_t + h * p.k_stride_h + i);
        }
    }
    for (int64_t hv = 0; hv < HV; ++hv) {
        for (int64_t j = 0; j < V; ++j) {
            x[2 * H * K + hv * V + j] = load_f32(
                    p.v, p.src_dt, r * p.v_stride_t + hv * p.v_stride_h + j);
        }
    }
}

// One recurrence step for every value head of one token. x holds the
// convolved q | k | v row in FP32.
void recurrent_step(const std::vector<float> &x, const void *a_row,
        const void *b_row, const float *A_log, const float *dt_bias,
        float *state_slot /* [HV][K*V] fp32 scratch */, const void *z_row,
        void *out_row, const gdn_params &p) {
    const int64_t H = p.num_k_heads, HV = p.num_v_heads;
    const int64_t K = p.head_k_dim, V = p.head_v_dim;
    const int64_t r = HV / H;
    const float scale = p.effective_scale();
    std::vector<float> q(K), k(K), kv(V), o(V);

    for (int64_t hv = 0; hv < HV; ++hv) {
        const int64_t h = hv / r;
        for (int64_t i = 0; i < K; ++i) {
            q[i] = x[h * K + i];
            k[i] = x[H * K + h * K + i];
        }
        const float *v = x.data() + 2 * H * K + hv * V;
        if (p.use_qk_l2norm) {
            float sq = 0.0f, sk = 0.0f;
            for (int64_t i = 0; i < K; ++i) {
                sq += q[i] * q[i];
                sk += k[i] * k[i];
            }
            const float rq = 1.0f / std::sqrt(sq + p.l2norm_eps);
            const float rk = 1.0f / std::sqrt(sk + p.l2norm_eps);
            for (int64_t i = 0; i < K; ++i) {
                q[i] *= rq;
                k[i] *= rk;
            }
        }
        for (int64_t i = 0; i < K; ++i) {
            q[i] *= scale;
        }

        const float av = load_f32(a_row, p.a_type(), hv);
        const float bv = load_f32(b_row, p.b_type(), hv);
        // Precomputed gates: a holds g, b holds beta.
        float g = av, beta = bv;
        if (!p.precomputed_gates()) {
            g = -std::exp(A_log[hv])
                    * softplus_f32(av + dt_bias[hv], p.softplus_threshold);
            beta = p.beta_scale() * sigmoid_f32(bv);
        }
        const float eg = std::exp(g);

        float *S = state_slot + hv * K * V;
        for (int64_t j = 0; j < V; ++j) {
            float acc = 0.0f;
            for (int64_t i = 0; i < K; ++i) {
                acc += S[state_offset(p, i, j)] * eg * k[i];
            }
            kv[j] = (v[j] - acc) * beta;
        }
        for (int64_t j = 0; j < V; ++j) {
            float acc = 0.0f;
            for (int64_t i = 0; i < K; ++i) {
                float &s = S[state_offset(p, i, j)];
                s = s * eg + k[i] * kv[j];
                acc += s * q[i];
            }
            o[j] = acc;
        }

        if (p.norm_weight != nullptr) {
            float ss = 0.0f;
            for (int64_t j = 0; j < V; ++j) {
                if (p.round_core_out) { o[j] = round_to(o[j], p.src_dt); }
                ss += o[j] * o[j];
            }
            const float rs
                    = 1.0f / std::sqrt(ss / static_cast<float>(V) + p.norm_eps);
            const char *z_head
                    = byte_ptr(z_row) + hv * p.z_stride_h * dt_size(p.src_dt);
            for (int64_t j = 0; j < V; ++j) {
                const float w = load_f32(p.norm_weight, p.norm_weight_dt, j);
                const float zv = load_f32(z_head, p.src_dt, j);
                o[j] = o[j] * rs * w * norm_gate_f32(zv, p.norm_gate);
            }
        }
        char *out_head = static_cast<char *>(out_row)
                + hv * p.out_stride_h * dt_size(p.src_dt);
        for (int64_t j = 0; j < V; ++j) {
            store_f32(out_head, p.src_dt, j, o[j]);
        }
    }
}

void load_state(
        const void *ssm_state, int64_t slot, float *dst, const gdn_params &p) {
    const int64_t KV = p.head_k_dim * p.head_v_dim;
    for (int64_t hv = 0; hv < p.num_v_heads; ++hv) {
        const int64_t base
                = slot * p.ssm_state_slot_stride + hv * p.ssm_state_head_stride;
        for (int64_t i = 0; i < KV; ++i) {
            dst[hv * KV + i] = load_f32(ssm_state, p.state_dt, base + i);
        }
    }
}

void store_state(
        void *ssm_state, int64_t slot, const float *src, const gdn_params &p) {
    const int64_t KV = p.head_k_dim * p.head_v_dim;
    for (int64_t hv = 0; hv < p.num_v_heads; ++hv) {
        const int64_t base
                = slot * p.ssm_state_slot_stride + hv * p.ssm_state_head_stride;
        for (int64_t i = 0; i < KV; ++i) {
            store_f32(ssm_state, p.state_dt, base + i, src[hv * KV + i]);
        }
    }
}

// Convolve one token. window[j] for j < W - 1 is the history (oldest first)
// and the token itself is the last tap.
float conv_channel(const float *window, const void *conv_weight,
        const void *conv_bias, int64_t c, const gdn_params &p) {
    const int64_t W = p.conv_width;
    float acc = conv_bias ? load_f32(conv_bias, p.src_dt, c) : 0.0f;
    for (int64_t j = 0; j < W; ++j) {
        acc += load_f32(conv_weight, p.src_dt, c * W + j) * window[j];
    }
    if (p.silu) { acc = silu_f32(acc); }
    return p.round_conv_out ? round_to(acc, p.src_dt) : acc;
}

} // namespace

status_t gdn_decode_reference(
        const gdn_tensors_t &io, int64_t n, const gdn_params &p) {
    const int64_t D = p.conv_dim();
    const int64_t W = p.conv_width;
    const int64_t es = dt_size(p.src_dt);
    std::vector<float> x(D), window(W > 0 ? W : 1);
    std::vector<float> S(p.num_v_heads * p.head_k_dim * p.head_v_dim);

    for (int64_t bi = 0; bi < n; ++bi) {
        const int32_t slot = io.state_indices[bi];
        char *out_row
                = static_cast<char *>(io.output) + bi * p.out_stride_t * es;
        if (slot == p.pad_slot_id) {
            for (int64_t hv = 0; hv < p.num_v_heads; ++hv) {
                for (int64_t j = 0; j < p.head_v_dim; ++j) {
                    store_f32(out_row + hv * p.out_stride_h * es, p.src_dt, j,
                            0.0f);
                }
            }
            continue;
        }
        const char *in_row = byte_ptr(io.mixed_qkv) + bi * p.qkv_stride * es;
        for (int64_t c = 0; c < D; ++c) {
            const float xin = load_f32(in_row, p.src_dt, c);
            if (W == 0) {
                x[c] = xin;
                continue;
            }
            for (int64_t j = 0; j < W - 1; ++j) {
                window[j] = load_f32(
                        conv_state_elem(io.conv_state, p, slot, j, c), p.src_dt,
                        0);
            }
            window[W - 1] = xin;
            x[c] = conv_channel(
                    window.data(), io.conv_weight, io.conv_bias, c, p);
            for (int64_t j = 0; j < W - 1; ++j) {
                store_f32(conv_state_elem(io.conv_state, p, slot, j, c),
                        p.src_dt, 0, window[j + 1]);
            }
        }

        load_state(io.ssm_state, slot, S.data(), p);
        const void *z_row = p.z ? static_cast<const void *>(byte_ptr(p.z)
                                          + bi * p.z_stride_t * es)
                                : nullptr;
        recurrent_step(x, byte_ptr(io.a) + bi * p.a_stride * es,
                byte_ptr(io.b) + bi * p.b_stride * es, io.A_log, io.dt_bias,
                S.data(), z_row, out_row, p);
        store_state(io.ssm_state, slot, S.data(), p);
    }
    return status_t::success;
}

status_t gdn_verify_reference(const gdn_tensors_t &io, int64_t n,
        const int32_t *offsets, const gdn_params &p) {
    const int64_t D = p.conv_dim();
    const int64_t W = p.conv_width;
    const int64_t es = dt_size(p.src_dt);
    const int64_t L = p.conv_taps();
    std::vector<float> x(D), window(W > 0 ? W : 1);
    std::vector<float> S(p.num_v_heads * p.head_k_dim * p.head_v_dim);
    // hist[c][j]: the W - 1 most recent raw inputs of channel c, oldest first.
    std::vector<float> hist(W > 1 ? D * (W - 1) : 1);
    std::vector<float> keep(L > 0 ? L : 1);

    for (int64_t bi = 0; bi < n; ++bi) {
        const int64_t r0 = offsets[bi], r1 = offsets[bi + 1];
        const int32_t *slots = p.spec_state_indices + bi * p.spec_state_stride;
        if (r1 <= r0) { continue; }
        if (slots[0] == p.pad_slot_id) {
            for (int64_t r = r0; r < r1; ++r) {
                char *out_row = static_cast<char *>(io.output)
                        + r * p.out_stride_t * es;
                for (int64_t hv = 0; hv < p.num_v_heads; ++hv) {
                    for (int64_t j = 0; j < p.head_v_dim; ++j) {
                        store_f32(out_row + hv * p.out_stride_h * es, p.src_dt,
                                j, 0.0f);
                    }
                }
            }
            continue;
        }
        const int64_t a = p.num_accepted_tokens[bi];
        for (int64_t c = 0; W > 1 && c < D; ++c) {
            for (int64_t j = 0; j < W - 1; ++j) {
                hist[c * (W - 1) + j]
                        = load_f32(conv_state_elem(io.conv_state, p, slots[0],
                                           a - 1 + j, c),
                                p.src_dt, 0);
            }
        }
        load_state(io.ssm_state, slots[a - 1], S.data(), p);

        for (int64_t r = r0; r < r1; ++r) {
            const char *in_row = byte_ptr(io.mixed_qkv) + r * p.qkv_stride * es;
            for (int64_t c = 0; c < D; ++c) {
                const float xin = load_f32(in_row, p.src_dt, c);
                if (W == 0) {
                    x[c] = xin;
                    continue;
                }
                float *h = hist.data() + c * (W - 1);
                for (int64_t j = 0; j < W - 1; ++j) {
                    window[j] = h[j];
                }
                window[W - 1] = xin;
                x[c] = conv_channel(
                        window.data(), io.conv_weight, io.conv_bias, c, p);
                for (int64_t j = 0; j + 1 < W - 1; ++j) {
                    h[j] = h[j + 1];
                }
                if (W > 1) { h[W - 2] = xin; }
            }
            char *out_row
                    = static_cast<char *>(io.output) + r * p.out_stride_t * es;
            const void *z_row = p.z ? static_cast<const void *>(byte_ptr(p.z)
                                              + r * p.z_stride_t * es)
                                    : nullptr;
            recurrent_step(x, byte_ptr(io.a) + r * p.a_stride * es,
                    byte_ptr(io.b) + r * p.b_stride * es, io.A_log, io.dt_bias,
                    S.data(), z_row, out_row, p);
            // The next token starts from the stored state, as a decode does.
            store_state(io.ssm_state, slots[r - r0], S.data(), p);
            load_state(io.ssm_state, slots[r - r0], S.data(), p);
        }

        const int64_t q = r1 - r0;
        for (int64_t c = 0; W > 1 && c < D; ++c) {
            for (int64_t j = 0; j < W - 2; ++j) {
                keep[j] = load_f32(
                        conv_state_elem(io.conv_state, p, slots[0], a + j, c),
                        p.src_dt, 0);
            }
            for (int64_t j = 0; j < W - 2; ++j) {
                store_f32(conv_state_elem(io.conv_state, p, slots[0], j, c),
                        p.src_dt, 0, keep[j]);
            }
            for (int64_t t = 0; t < q; ++t) {
                const char *in_row
                        = byte_ptr(io.mixed_qkv) + (r0 + t) * p.qkv_stride * es;
                store_f32(conv_state_elem(
                                  io.conv_state, p, slots[0], W - 2 + t, c),
                        p.src_dt, 0, load_f32(in_row, p.src_dt, c));
            }
        }
    }
    return status_t::success;
}

status_t gdn_prefill_reference(const gdn_tensors_t &io, int64_t s0, int64_t s1,
        const int32_t *offsets, const gdn_params &p) {
    const int64_t D = p.conv_dim();
    const int64_t W = p.conv_width;
    const int64_t es = dt_size(p.src_dt);
    std::vector<float> x(D);
    std::vector<float> S(p.num_v_heads * p.head_k_dim * p.head_v_dim);
    // hist[c][j]: the W - 1 most recent raw inputs of channel c, oldest first.
    std::vector<float> hist(W > 1 ? D * (W - 1) : 1);
    std::vector<float> window(W > 0 ? W : 1);

    for (int64_t si = s0; si < s1; ++si) {
        const int32_t slot = io.state_indices[si];
        const int64_t r0 = offsets[si], r1 = offsets[si + 1];
        if (slot == p.pad_slot_id || r1 <= r0) { continue; }
        const bool init
                = p.has_initial_state != nullptr && p.has_initial_state[si];

        if (init) {
            load_state(io.ssm_state, slot, S.data(), p);
        } else {
            std::fill(S.begin(), S.end(), 0.0f);
        }
        if (W > 1) {
            for (int64_t c = 0; c < D; ++c) {
                for (int64_t j = 0; j < W - 1; ++j) {
                    hist[c * (W - 1) + j] = init
                            ? load_f32(conv_state_elem(
                                               io.conv_state, p, slot, j, c),
                                      p.src_dt, 0)
                            : 0.0f;
                }
            }
        }

        for (int64_t r = r0; r < r1; ++r) {
            const char *in_row = byte_ptr(io.mixed_qkv) + r * p.qkv_stride * es;
            if (p.separate_qkv()) { separate_row(io, r, x, p); }
            for (int64_t c = 0; c < D; ++c) {
                const float xin = p.separate_qkv()
                        ? x[c]
                        : load_f32(in_row, p.src_dt, c);
                if (W == 0) {
                    x[c] = xin;
                    continue;
                }
                float *h = hist.data() + c * (W - 1);
                for (int64_t j = 0; j < W - 1; ++j) {
                    window[j] = h[j];
                }
                window[W - 1] = xin;
                x[c] = conv_channel(
                        window.data(), io.conv_weight, io.conv_bias, c, p);
                for (int64_t j = 0; j + 1 < W - 1; ++j) {
                    h[j] = h[j + 1];
                }
                if (W > 1) { h[W - 2] = xin; }
            }
            char *out_row
                    = static_cast<char *>(io.output) + r * p.out_stride_t * es;
            const void *z_row = p.z ? static_cast<const void *>(byte_ptr(p.z)
                                              + r * p.z_stride_t * es)
                                    : nullptr;
            recurrent_step(x,
                    byte_ptr(io.a) + r * p.a_stride * dt_size(p.a_type()),
                    byte_ptr(io.b) + r * p.b_stride * dt_size(p.b_type()),
                    io.A_log, io.dt_bias, S.data(), z_row, out_row, p);
        }

        if (!p.update_state) { continue; }
        store_state(io.ssm_state, slot, S.data(), p);
        if (W > 1) {
            for (int64_t c = 0; c < D; ++c) {
                for (int64_t j = 0; j < W - 1; ++j) {
                    store_f32(conv_state_elem(io.conv_state, p, slot, j, c),
                            p.src_dt, 0, hist[c * (W - 1) + j]);
                }
            }
        }
    }
    return status_t::success;
}

} // namespace gdn
} // namespace lowoha
} // namespace zendnnl
