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

/// Grouped calls (`group_matmul_direct` with two or more operations), served
/// as `ntile_flat_parallel::execute_grouped`: their admission, their call
/// sites, the grouped front, and the call.

#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/ntile_flat_parallel.hpp"

#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <vector>
#include <shared_mutex>

#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/api/api_common.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/engine/engine.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/engine/scratch.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/engine/source.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/formats/formats.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/pack/call_sites.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/pack/scales.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/pack/store.hpp"
#include "lowoha_operators/matmul/lowoha_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ntile_flat_parallel {
namespace detail {

namespace {

/// How an operation's source scales cover its rows.
enum class source_scales { per_row, per_tensor, per_group, unsupported };

/// The form of source scales `dims` for `M` rows against a weight of `groups`
/// scale groups: `{M, 1}` (or none) per row, `{1}` per tensor, `{M, groups}`
/// per (row, group).
source_scales source_form(
        const std::vector<int64_t> &dims, const int M, const int groups) {
    if (dims.empty()) return source_scales::per_row;
    const int64_t cols = dims.size() >= 2 ? dims.back() : 1;
    if (cols <= 0) return source_scales::unsupported;
    const int64_t rows = element_count(dims) / cols;
    if (cols == 1) {
        return rows == M    ? source_scales::per_row
                : rows == 1 ? source_scales::per_tensor
                            : source_scales::unsupported;
    }
    return groups > 1 && cols == groups && rows == M
            ? source_scales::per_group
            : source_scales::unsupported;
}

/// A grouped call's arguments, as `execute_grouped` takes them.
struct grouped_args {
    int num_ops;
    const std::vector<char> &layout;
    const std::vector<bool> &transA;
    const std::vector<bool> &transB;
    const std::vector<int> &M;
    const std::vector<int> &N;
    const std::vector<int> &K;
    const std::vector<float> &alpha;
    const std::vector<const void *> &src;
    const std::vector<int> &lda;
    const std::vector<const void *> &weight;
    const std::vector<int> &ldb;
    const std::vector<const void *> &bias;
    const std::vector<float> &beta;
    const std::vector<void *> &dst;
    const std::vector<int> &ldc;
    const std::vector<bool> &is_weights_const;
    const std::vector<matmul_params> &params;
    grp_matmul_gated_act_t act;
    const grp_matmul_fused_moe_params *fused;
    const group_matmul_moe_postop_params *reduce;
    int num_threads;
};

/// A grouped call as it runs.
struct grouped_spec_t {
    /// The first operation with rows: the others must match it.
    int first = -1;
    data_type_t wei = data_type_t::none;
    data_type_t src = data_type_t::none;
    data_type_t dst = data_type_t::none;
    /// The call's format.
    const nr32::format_desc *fmt = nullptr;
    /// The caller's rows are s8 already, with their scales.
    bool src_s8 = false;
    bool u8_src = false;
    nr32::epilogue ep = nr32::epilogue::plain;
    nr32::weight_desc w13, w2;
    scale_form_t scales13, scales2;
    /// Op1's source scales: `per_group`, one per (row, W13 group), or
    /// `per_row` (each caller's s8 operation may give one for all its rows).
    source_scales source = source_scales::per_row;
    /// Op2's source scales are one per (row, W2 group): Op1's are per group
    /// and W2 has several groups.
    bool per_group2 = false;
    /// Op1 writes the library's rows instead of `dst`.
    bool op1_internal = false;
    /// Op2 writes the caller's source rows instead of `dst_down`.
    bool op2_internal = false;
    /// Expert weights the call packs: its operations' and the prepack
    /// extras'.
    int num_packs = 0;
};

/// Every operation with rows has a pointer in `ptrs`, all null or none.
bool all_null(const std::vector<void *> &ptrs, const grouped_args &a) {
    for (int e = 0; e < a.num_ops; ++e) {
        if (a.M[e] > 0 && e < static_cast<int>(ptrs.size())
                && ptrs[e] != nullptr) {
            return false;
        }
    }
    return true;
}

/// Null when N-tile flat parallel runs the grouped call `a` with rows, `s` then
/// describing it; else why not.
const char *check_grouped(const grouped_args &a, grouped_spec_t &s) {
    const int n = a.num_ops;
    const matmul_params &p0 = a.params[s.first];
    s.wei = p0.dtypes.wei;
    s.src = p0.dtypes.src;
    s.dst = p0.dtypes.dst;
    // The format whose kernels read the rows the call gives (16-bit rows as
    // they are, or the caller's s8 rows), or the rows it quantizes to.  Its
    // sources are bf16 rows the call quantizes to its kernels' s8, where its
    // kernels take them the caller's s8 rows, or its own 16-bit rows.
    const auto &ws0 = p0.quant_params.wei_scale;
    const nr32::format_desc *fmt = nr32::format_for(s.wei,
            p0.dynamic_quant ? p0.dtypes.compute : s.src,
            nr32::scale_enc_of(ws0.buff, ws0.dt));
    if (fmt == nullptr || s.dst != fmt->dst) {
        return "unsupported_types";
    } else if (fmt->src != data_type_t::s8) {
        if (s.src != fmt->src || p0.dynamic_quant) return "unsupported_types";
    } else if (s.src == data_type_t::s8 && !p0.dynamic_quant) {
        if (!fmt->s8_rows) return "unsupported_types";
        s.src_s8 = true;
    } else if (s.src != data_type_t::bf16 || !p0.dynamic_quant
            || p0.dtypes.compute != fmt->src) {
        return "unsupported_types";
    }
    s.fmt = fmt;
    s.ep = epilogue_of(a.act);
    const int K = a.K[s.first], N = a.N[s.first];
    if (nr32::gated(s.ep) && N % 2 != 0) return "shape";
    const int out13 = nr32::gated(s.ep) ? N / 2 : N;
    s.w13.fmt = fmt;
    s.w13.K = K;
    s.w13.N = N;
    s.w13.trans = a.transB[s.first];
    s.w13.ldb = a.ldb[s.first];
    if (fmt->scaled()) {
        if (!scale_form(ws0.dims, ws0.dt, *fmt, K, N, s.scales13))
            return "unsupported_weight_scale";
        s.w13.group_size = pack_group_size(*fmt, K, s.scales13);
        s.source = source_form(p0.quant_params.src_scale.dims, a.M[s.first],
                static_cast<int>(s.scales13.groups));
    }
    s.op1_internal = a.fused != nullptr && all_null(a.dst, a);
    for (int e = 0; e < n; ++e) {
        if (a.M[e] == 0) continue;
        const matmul_params &p = a.params[e];
        const auto &qp = p.quant_params;
        if (a.K[e] != K || a.N[e] != N || a.ldb[e] != s.w13.ldb
                || a.transB[e] != s.w13.trans || p.dtypes.wei != s.wei
                || p.dtypes.src != s.src || p.dtypes.dst != s.dst
                || p.dtypes.compute != p0.dtypes.compute
                || p.dynamic_quant != p0.dynamic_quant) {
            return "non_uniform_operations";
        }
        if ((a.layout[e] != 'r' && a.layout[e] != 'R') || a.transA[e]
                || a.alpha[e] != 1.0f || a.beta[e] != 0.0f) {
            return "unsupported_operation";
        }
        if (!a.is_weights_const[e]) return "non_constant_weights";
        if (effective_weight_cache_type(p.weight_cache_type) < 1)
            return "weight_cache_disabled";
        if (!p.postop_.empty()) return "post_ops";
        if (p.mem_format_b != 'n' || p.packing.pack_format_b != 0)
            return "prepacked_weights";
        if (qp.wei_zp.buff != nullptr || qp.src_zp.buff != nullptr
                || qp.dst_scale.buff != nullptr || qp.dst_zp.buff != nullptr) {
            return "unsupported_quantization";
        }
        if (a.src[e] == nullptr || a.weight[e] == nullptr || a.lda[e] < K)
            return "shape";
        if (a.bias[e] != nullptr
                && bias_kind_of(p.dtypes.bias) == nr32::bias_kind::none) {
            return "unsupported_bias";
        }
        if (!s.op1_internal
                && (static_cast<int>(a.dst.size()) <= e || a.dst[e] == nullptr
                        || static_cast<int>(a.ldc.size()) <= e
                        || a.ldc[e] < out13)) {
            return "shape";
        }
        if (!fmt->scaled()) continue;
        if (qp.wei_scale.buff == nullptr || qp.wei_scale.dt != ws0.dt
                || qp.wei_scale.dims != ws0.dims) {
            return "unsupported_weight_scale";
        }
        // Every operation's source scales group as the first's: per group,
        // or per row (one row's per-tensor scale is its per-row one).  Only
        // the caller's s8 rows come with one scale for all of them.
        source_scales form = source_form(
                qp.src_scale.dims, a.M[e], static_cast<int>(s.scales13.groups));
        if (form == source_scales::per_tensor && !s.src_s8)
            form = source_scales::unsupported;
        const bool per_group = form == source_scales::per_group;
        if (form == source_scales::unsupported
                || per_group != (s.source == source_scales::per_group)
                || (qp.src_scale.buff != nullptr
                        && !scale_type(qp.src_scale.dt))
                || (s.src_s8 && qp.src_scale.buff == nullptr)) {
            return "unsupported_source_scale";
        }
    }
    if (s.source == source_scales::per_tensor)
        s.source = source_scales::per_row;
    if (a.fused != nullptr) {
        const grp_matmul_fused_moe_params &f = *a.fused;
        const size_t ops = static_cast<size_t>(n);
        if (f.down_weight.size() < ops || f.N_down.size() < ops
                || f.ldb_down.size() < ops) {
            return "shape";
        }
        s.w2.fmt = s.fmt;
        s.w2.K = out13;
        s.w2.N = f.N_down[s.first];
        s.w2.trans = s.w13.trans;
        s.w2.ldb = f.ldb_down[s.first];
        if (s.fmt->scaled()) {
            if (f.down_scale.size() < ops) return "unsupported_weight_scale";
            const auto &ds0 = f.down_scale[s.first];
            if (!scale_form(
                        ds0.dims, ds0.dt, *s.fmt, s.w2.K, s.w2.N, s.scales2))
                return "unsupported_weight_scale";
            s.w2.group_size = pack_group_size(*s.fmt, s.w2.K, s.scales2);
            s.per_group2 = s.source == source_scales::per_group
                    && s.scales2.groups > 1;
        }
        s.op2_internal = all_null(f.dst_down, a);
        if (s.op2_internal && s.src != s.dst) return "unsupported_types";
        for (int e = 0; e < n; ++e) {
            if (a.M[e] == 0) continue;
            if (f.N_down[e] != s.w2.N || f.ldb_down[e] != s.w2.ldb)
                return "non_uniform_operations";
            if (f.down_weight[e] == nullptr) return "shape";
            if (e < static_cast<int>(f.down_zp.size())
                    && f.down_zp[e].buff != nullptr) {
                return "unsupported_quantization";
            }
            const bool scaled = e < static_cast<int>(f.down_scale.size())
                    && f.down_scale[e].buff != nullptr;
            if (s.fmt->scaled() ? !scaled || f.down_scale[e].dt != s.scales2.dt
                                    || f.down_scale[e].dims != s.scales2.dims
                                : scaled) {
                return "unsupported_weight_scale";
            }
            if (e < static_cast<int>(f.bias_down.size())
                    && f.bias_down[e] != nullptr
                    && bias_kind_of(f.bias_dt_down) == nr32::bias_kind::none) {
                return "unsupported_bias";
            }
            if (s.op2_internal ? a.lda[e] < s.w2.N
                               : static_cast<int>(f.dst_down.size()) <= e
                                    || f.dst_down[e] == nullptr
                                    || static_cast<int>(f.ldc_down.size()) <= e
                                    || f.ldc_down[e] < s.w2.N) {
                return "shape";
            }
        }
    }
    s.u8_src = u8_src_pays(*s.fmt, api_kind::grouped);
    s.num_packs = n;
    const matmul_params &lead = a.params[0];
    if (lead.active_matmul > 0 && lead.total_matmul > lead.active_matmul) {
        s.num_packs = static_cast<int>(
                std::min({static_cast<size_t>(lead.total_matmul),
                        a.weight.size(), a.K.size(), a.N.size(), a.ldb.size(),
                        a.transB.size(), a.is_weights_const.size()}));
    }
    return nullptr;
}

/// Whether the call packs expert `e`'s W13 weight: an operation with rows
/// does; another expert does when it matches them, an expert past the call's
/// `params` taken to be of the call's types.
bool packs_w13(const grouped_args &a, const grouped_spec_t &s, const int e) {
    if (a.weight[e] == nullptr) return false;
    if (e < a.num_ops && a.M[e] > 0) return true;
    if (a.K[e] != s.w13.K || a.N[e] != s.w13.N || a.ldb[e] != s.w13.ldb
            || a.transB[e] != s.w13.trans || !a.is_weights_const[e]) {
        return false;
    }
    if (e >= static_cast<int>(a.params.size())) return true;
    const matmul_params &p = a.params[e];
    return p.dtypes.wei == s.wei && p.mem_format_b == 'n'
            && p.packing.pack_format_b == 0;
}

/// Expert `e`'s W13 weight scales when the call gives them in its operations'
/// form, else null (always for 16-bit weights).
const void *scales_w13(
        const grouped_args &a, const grouped_spec_t &s, const int e) {
    if (!s.fmt->scaled() || e >= static_cast<int>(a.params.size()))
        return nullptr;
    const auto &q = a.params[e].quant_params.wei_scale;
    return q.dt == s.scales13.dt && q.dims == s.scales13.dims ? q.buff
                                                              : nullptr;
}

/// `packs_w13` for expert `e`'s W2 weight.
bool packs_w2(const grouped_args &a, const grouped_spec_t &s, const int e) {
    const grp_matmul_fused_moe_params &f = *a.fused;
    if (e >= static_cast<int>(f.down_weight.size())
            || f.down_weight[e] == nullptr) {
        return false;
    }
    if (e < a.num_ops && a.M[e] > 0) return true;
    return packs_w13(a, s, e) && e < static_cast<int>(f.N_down.size())
            && e < static_cast<int>(f.ldb_down.size()) && f.N_down[e] == s.w2.N
            && f.ldb_down[e] == s.w2.ldb;
}

/// `scales_w13` for expert `e`'s W2 weight.
const void *scales_w2(
        const grouped_args &a, const grouped_spec_t &s, const int e) {
    const grp_matmul_fused_moe_params &f = *a.fused;
    if (!s.fmt->scaled() || e >= static_cast<int>(f.down_scale.size()))
        return nullptr;
    const auto &q = f.down_scale[e];
    return q.dt == s.scales2.dt && q.dims == s.scales2.dims ? q.buff : nullptr;
}

/// A fingerprint of a projection's pool: the weights its call names, the
/// operations' and the prepack extras', whatever their order and whichever
/// have rows, and their shape and types.
size_t extras_fingerprint(const nr32::weight_desc &desc,
        const scale_form_t &form, const std::vector<const void *> &actives,
        const std::vector<const void *> &extras) {
    size_t sum = 0;
    for (const auto *pool : {&actives, &extras}) {
        for (const void *w : *pool) {
            if (w != nullptr) sum += pointer_hash(w);
        }
    }
    size_t h = sum;
    const auto mix = [&h](size_t v) { h = (h ^ v) * 0x9E3779B97F4A7C15ULL; };
    for (const int v : {nr32::format_index(*desc.fmt), desc.K, desc.N, desc.ldb,
                 desc.group_size, static_cast<int>(desc.trans),
                 static_cast<int>(form.dt), static_cast<int>(form.groups)}) {
        mix(static_cast<size_t>(v));
    }
    return h;
}

/// The packs of a grouped call's operations with rows (null for the others),
/// from `find_packs` for a call that began at flush count `since`;
/// `weight(e)` and `scales(e)` give expert e's weight (null when the call
/// doesn't pack it) and weight scales.  The call's other experts, its prepack
/// extras, are packed the first time a call names them.
template <typename Weight, typename Scales>
const char *call_packs(const grouped_args &a, const grouped_spec_t &s,
        const nr32::weight_desc &desc, const scale_form_t &form,
        const Weight &weight, const Scales &scales, const uint64_t since,
        std::vector<expert_pack> &out) {
    std::vector<const void *> w(s.num_packs, nullptr), sc(s.num_packs, nullptr);
    std::vector<const void *> extra_w, extra_sc;
    for (int e = 0; e < s.num_packs; ++e) {
        const void *const p = weight(e);
        if (p == nullptr) continue;
        if (e < a.num_ops && a.M[e] > 0) {
            w[e] = p;
            sc[e] = scales(e);
        } else {
            extra_w.push_back(p);
            extra_sc.push_back(scales(e));
        }
    }
    const char *unserved
            = find_packs(desc, form, w, sc, since, a.num_threads, out);
    if (unserved != nullptr) return unserved;
    for (int e = 0; e < s.num_packs; ++e) {
        if (w[e] != nullptr && out[e] == nullptr)
            return "unsupported_weight_scale";
    }
    if (extra_w.empty()) return nullptr;
    const size_t fingerprint = extras_fingerprint(desc, form, w, extra_w);
    if (extras_warmed(fingerprint, since)) return nullptr;
    std::vector<expert_pack> extras;
    unserved = find_packs(
            desc, form, extra_w, extra_sc, since, a.num_threads, extras);
    if (unserved != nullptr) return unserved;
    mark_extras_warmed(fingerprint, since);
    return nullptr;
}

/// Expert `e`'s W13 and W2 weights, null where the call doesn't pack them.
const void *weight13(const grouped_args &a, const grouped_spec_t &s, int e) {
    return packs_w13(a, s, e) ? a.weight[e] : nullptr;
}
const void *weight2(const grouped_args &a, const grouped_spec_t &s, int e) {
    return a.fused != nullptr && packs_w2(a, s, e) ? a.fused->down_weight[e]
                                                   : nullptr;
}

/// The call's site: the weights it names, whatever their order, their
/// shapes and scale forms, and what its tile plans depend on.
size_t site_key(const grouped_args &a, const grouped_spec_t &s) {
    const size_t downs = a.fused != nullptr ? a.fused->down_weight.size() : 0;
    size_t h = 0;
    for (int e = 0; e < s.num_packs; ++e) {
        const void *const down = static_cast<size_t>(e) < downs
                ? a.fused->down_weight[e]
                : nullptr;
        h += pointer_hash(a.weight[e])
                ^ (pointer_hash(down) * 0x9E3779B97F4A7C15ULL);
    }
    const auto mix = [&h](size_t v) { h = (h ^ v) * 0x9E3779B97F4A7C15ULL; };
    for (const nr32::weight_desc *d : {&s.w13, &s.w2}) {
        mix(weight_key_hash_t {}(key_of(*d, nullptr)));
    }
    for (const scale_form_t *f : {&s.scales13, &s.scales2}) {
        mix(static_cast<size_t>(f->dt));
        mix(static_cast<size_t>(f->groups));
        mix(static_cast<size_t>(f->per_tensor));
    }
    mix(static_cast<size_t>(s.ep));
    mix(static_cast<size_t>(s.u8_src));
    mix(static_cast<size_t>(s.source == source_scales::per_group));
    mix(static_cast<size_t>(s.num_packs));
    return h;
}

/// The packs of the call's operations with rows, from `site`, into the
/// scratch; false when the site lacks one or holds it with other scales.
/// `check_grouped` has given every such operation its scales in the call's
/// forms.
bool site_packs(const call_site_t &site, const grouped_args &a,
        const grouped_spec_t &s, scratch_t &S) {
    const bool fused = a.fused != nullptr;
    S.packs13.assign(static_cast<size_t>(s.num_packs), nullptr);
    if (fused) S.packs2.assign(static_cast<size_t>(s.num_packs), nullptr);
    for (int e = 0; e < a.num_ops; ++e) {
        if (a.M[e] <= 0) continue;
        const call_site_t::slot_t *p = find_slot(site.w13, a.weight[e]);
        if (p == nullptr || p->pack == nullptr
                || (s.fmt->scaled()
                        && !(p->scales
                                == scale_id_of(
                                        a.params[e].quant_params.wei_scale.buff,
                                        s.scales13)))) {
            return false;
        }
        S.packs13[e] = p->pack;
        if (!fused) continue;
        p = find_slot(site.w2, a.fused->down_weight[e]);
        if (p == nullptr || p->pack == nullptr
                || (s.fmt->scaled()
                        && !(p->scales
                                == scale_id_of(a.fused->down_scale[e].buff,
                                        s.scales2)))) {
            return false;
        }
        S.packs2[e] = p->pack;
    }
    return true;
}

/// A site of the packs the store holds for the call's weights, and of the
/// call's layouts and tile plans.
std::shared_ptr<const call_site_t> make_site(const grouped_args &a,
        const grouped_spec_t &s, const nr32::pack_layout &l13,
        const nr32::pack_layout &l2, const nr32::tile_plan &p13,
        const nr32::tile_plan &p2) {
    auto site = std::make_shared<call_site_t>();
    site->l13 = l13;
    site->l2 = l2;
    site->p13 = p13;
    site->p2 = p2;
    size_t slots = 8;
    while (slots < 2 * static_cast<size_t>(s.num_packs))
        slots *= 2;
    site->w13.resize(slots);
    if (a.fused != nullptr) site->w2.resize(slots);
    const auto add
            = [&](std::vector<call_site_t::slot_t> &table,
                      const nr32::weight_desc &desc, const auto &weight_of) {
        std::vector<const void *> weights(static_cast<size_t>(s.num_packs));
        for (int e = 0; e < s.num_packs; ++e)
            weights[e] = weight_of(e);
        const std::vector<weight_packs_t> held = stored_packs(desc, weights);
        const size_t mask = table.size() - 1;
        for (size_t e = 0; e < weights.size(); ++e) {
            const void *const weight = weights[e];
            const weight_packs_t &w = held[e];
            if (weight == nullptr || w.bare == nullptr) continue;
            const expert_pack &pack = s.fmt->scaled() ? w.scaled : w.bare;
            for (size_t i = pointer_hash(weight) & mask;; i = (i + 1) & mask) {
                call_site_t::slot_t &slot = table[i];
                if (slot.weight != nullptr && slot.weight != weight) continue;
                slot.weight = weight;
                slot.scales = w.scales;
                slot.pack = pack.get();
                if (pack != nullptr) site->keep.push_back(pack);
                break;
            }
        }
    };
    add(site->w13, s.w13, [&](const int e) { return weight13(a, s, e); });
    if (a.fused != nullptr)
        add(site->w2, s.w2, [&](const int e) { return weight2(a, s, e); });
    return site;
}

/// A grouped call's rows: operation e's are the caller's, `lda[e]` apart.
/// A W13 that reads s8 rows reads each row in the scratch, quantized there
/// (or copied from the caller's s8 rows) before any unit; another reads the
/// caller's rows in place.  Op1 writes the caller's `dst`, or the scratch's
/// rows when the call keeps no Op1 output; Op2 writes `dst_down`, or the
/// caller's source rows.
struct grouped_front {
    scratch_t &S;
    const grouped_args &a;
    const grouped_spec_t &s;
    int K, out13, K2;
    source_spec src13, src2;

    /// Row `row` of the call, of operation e, in Op1's output.
    uint16_t *out13_row(const int e, const int row) const {
        return s.op1_internal
                ? S.inter.data() + static_cast<size_t>(row) * out13
                : static_cast<uint16_t *>(a.dst[e])
                        + static_cast<size_t>(row - S.offset[e]) * a.ldc[e];
    }

    int out13_ld(const int e) const {
        return s.op1_internal ? out13 : a.ldc[e];
    }

    void prologue() const {
        if (!src13.s8) return;
        const int n = a.num_ops;
        const int *const offset = S.offset.data();
        const int total = offset[n];
        const int groups = src13.groups;
#pragma omp for schedule(static)
        for (int g = 0; g < total; ++g) {
            const int e = static_cast<int>(
                                  std::upper_bound(offset, offset + n + 1, g)
                                  - offset)
                    - 1;
            const int j = g - offset[e];
            const auto &qs = a.params[e].quant_params.src_scale;
            if (s.src_s8) {
                copy_s8_row(src13,
                        static_cast<const int8_t *>(a.src[e])
                                + static_cast<size_t>(j) * a.lda[e],
                        K, S.src13, g);
                float *const row_scale = S.src13.row_scales(src13, g);
                const bool one = element_count(qs.dims) == 1;
                for (int i = 0; i < groups; ++i) {
                    row_scale[i] = scale_at(qs.buff, qs.dt,
                            one ? 0 : static_cast<size_t>(j) * groups + i);
                }
            } else {
                quantize_row(src13,
                        static_cast<const uint16_t *>(a.src[e])
                                + static_cast<size_t>(j) * a.lda[e],
                        K, S.src13, g);
                const float *const row_scale = S.src13.row_scales(src13, g);
                for (int i = 0;
                        qs.buff != nullptr && !qs.dims.empty() && i < groups;
                        ++i) {
                    store_scale(const_cast<void *>(qs.buff), qs.dt,
                            static_cast<size_t>(j) * groups + i, row_scale[i]);
                }
            }
        }
    }

    void before_w13(int) const {}

    tile_io w13(const row_block_t &b) const {
        const int e = b.e;
        tile_io io;
        if (src13.s8) {
            io.src = S.src13.view(src13, b.row, K);
        } else {
            io.src.rows = static_cast<const uint16_t *>(a.src[e])
                    + static_cast<size_t>(b.row - S.offset[e]) * a.lda[e];
            io.src.lda = a.lda[e];
        }
        io.bias = a.bias[e];
        io.bias_t = a.bias[e] != nullptr ? bias_kind_of(a.params[e].dtypes.bias)
                                         : nr32::bias_kind::none;
        io.dst = out13_row(e, b.row);
        io.ldc = out13_ld(e);
        return io;
    }

    tile_io w2(const row_block_t &b) const {
        const grp_matmul_fused_moe_params &f = *a.fused;
        const int e = b.e, j = b.row - S.offset[e];
        tile_io io;
        if (src2.s8) {
            io.src = S.src2.view(src2, b.row, K2);
        } else {
            io.src.rows = out13_row(e, b.row);
            io.src.lda = out13_ld(e);
        }
        io.bias = e < static_cast<int>(f.bias_down.size()) ? f.bias_down[e]
                                                           : nullptr;
        io.bias_t = io.bias != nullptr ? bias_kind_of(f.bias_dt_down)
                                       : nr32::bias_kind::none;
        io.dst = s.op2_internal
                ? static_cast<uint16_t *>(const_cast<void *>(a.src[e]))
                        + static_cast<size_t>(j) * a.lda[e]
                : static_cast<uint16_t *>(f.dst_down[e])
                        + static_cast<size_t>(j) * f.ldc_down[e];
        io.ldc = s.op2_internal ? a.lda[e] : f.ldc_down[e];
        return io;
    }

    const uint16_t *op1_row(const int e, const int row) const {
        return out13_row(e, row);
    }
};

status_t run_grouped(
        const grouped_args &a, const grouped_spec_t &s, const uint64_t since) {
    const int n = a.num_ops, num_threads = a.num_threads;
    const bool fused = a.fused != nullptr;

    // The call's packs and plans: from its site, else from the store and
    // planned now, which then make the site for the site's next calls.
    scratch_t &S = scratch();
    const size_t key = site_key(a, s);
    uint64_t epoch = 0;
    std::shared_ptr<const call_site_t> site = find_site(key, epoch);
    std::vector<expert_pack> packs13, packs2;
    if (site == nullptr || !site_packs(*site, a, s, S)) {
        // Whether this CPU runs both projections, before anything is packed.
        if (!nr32::packable(s.w13) || (fused && !nr32::packable(s.w2)))
            return decline("unsupported_weight");
        const nr32::pack_layout l13 = nr32::layout_of(s.w13);
        const nr32::pack_layout l2
                = fused ? nr32::layout_of(s.w2) : nr32::pack_layout {};
        nr32::tile_plan p13, p2;
        const nr32::plan_budget budget = budget_for(*s.fmt, api_kind::grouped);
        if (!nr32::make_plan(l13, s.ep, s.u8_src, p13,
                    s.source == source_scales::per_group, budget)
                || (fused
                        && !nr32::make_plan(l2, nr32::epilogue::plain, s.u8_src,
                                p2, s.per_group2, budget))) {
            return decline("no_kernel");
        }
        const char *unserved
                = call_packs(a, s, s.w13, s.scales13, [&](const int e) {
            return weight13(a, s, e);
        }, [&](const int e) { return scales_w13(a, s, e); }, since, packs13);
        if (unserved != nullptr) return refuse(unserved);
        if (fused) {
            unserved = call_packs(a, s, s.w2, s.scales2, [&](const int e) {
                return weight2(a, s, e);
            }, [&](const int e) { return scales_w2(a, s, e); }, since, packs2);
            if (unserved != nullptr) return refuse(unserved);
        }
        S.packs13.assign(static_cast<size_t>(s.num_packs), nullptr);
        S.packs2.assign(packs2.size(), nullptr);
        for (size_t e = 0; e < packs13.size(); ++e)
            S.packs13[e] = packs13[e].get();
        for (size_t e = 0; e < packs2.size(); ++e)
            S.packs2[e] = packs2[e].get();
        site = make_site(a, s, l13, l2, p13, p2);
        publish_site(key, site, epoch, since);
    }
    const nr32::pack_layout &l13 = site->l13, &l2 = site->l2;
    const nr32::tile_plan &p13 = site->p13, &p2 = site->p2;
    const source_spec src13 = source_of(l13, p13);
    const source_spec src2 = fused ? source_of(l2, p2) : source_spec {};

    // Operation e owns rows [offset[e], offset[e+1]) of the call.
    S.offset.assign(static_cast<size_t>(n) + 1, 0);
    for (int e = 0; e < n; ++e)
        S.offset[e + 1] = S.offset[e] + a.M[e];
    const int total = S.offset[n];
    const int K = s.w13.K, out13 = p13.out_cols, K2 = out13;
    if (src13.s8) S.src13.grow(src13, static_cast<size_t>(total), K);
    if (src2.s8) S.src2.grow(src2, static_cast<size_t>(total), K2);
    if (s.op1_internal) grow(S.inter, static_cast<size_t>(total) * out13);

    const grouped_front front {S, a, s, K, out13, K2, src13, src2};
    engine_t en;
    en.n = n;
    en.rows = a.M.data();
    en.fmt = s.fmt;
    en.api = api_kind::grouped;
    en.fused = fused;
    en.wei_bytes = s.fmt->elem_bits / 8.0f;
    en.K13 = K;
    en.K2 = K2;
    en.src13 = src13;
    en.src2 = src2;
    en.l13 = &l13;
    en.l2 = &l2;
    en.p13 = &p13;
    en.p2 = &p2;
    en.packs13 = S.packs13.data();
    en.packs2 = S.packs2.data();
    en.reduce = a.reduce;
    en.reduce_cols = fused ? s.w2.N : out13;
    en.reduce_f16 = s.dst == data_type_t::f16;
    en.num_threads = num_threads;
    const status_t status = run_engine(en, front);
    if (test_api::s_capture_ntile_flat_parallel.load(
                std::memory_order_relaxed)) {
        test_api::s_ntile_flat_parallel_grouped_completed.fetch_add(
                1, std::memory_order_relaxed);
        if (s.u8_src) {
            test_api::s_ntile_flat_parallel_u8_src_calls.fetch_add(
                    1, std::memory_order_relaxed);
        }
    }
    return status;
}

} // namespace
} // namespace detail

status_t execute_grouped(const size_t num_ops, const std::vector<char> &layout,
        const std::vector<bool> &transA, const std::vector<bool> &transB,
        const std::vector<int> &M, const std::vector<int> &N,
        const std::vector<int> &K, const std::vector<float> &alpha,
        const std::vector<const void *> &src, const std::vector<int> &lda,
        const std::vector<const void *> &weight, const std::vector<int> &ldb,
        const std::vector<const void *> &bias, const std::vector<float> &beta,
        const std::vector<void *> &dst, const std::vector<int> &ldc,
        const std::vector<bool> &is_weights_const,
        const std::vector<matmul_params> &params,
        const grp_matmul_gated_act_t act,
        const grp_matmul_fused_moe_params *fused,
        const group_matmul_moe_postop_params *reduce, const int num_threads,
        const uint64_t since) {
    const detail::grouped_args a {static_cast<int>(num_ops), layout, transA,
            transB, M, N, K, alpha, src, lda, weight, ldb, bias, beta, dst, ldc,
            is_weights_const, params, act, fused, reduce, num_threads};
    if (a.num_ops < 2 || num_threads <= 0) return detail::decline("shape");
    detail::grouped_spec_t s;
    for (int e = 0; e < a.num_ops; ++e) {
        if (M[e] < 0) return detail::decline("shape");
        if (M[e] > 0 && s.first < 0) s.first = e;
    }
    // The weighted reduce sums rows of the call's last output.
    const int lead = std::max(s.first, 0);
    if (static_cast<size_t>(lead) >= N.size()
            || static_cast<size_t>(lead) >= params.size()
            || (fused != nullptr
                    && static_cast<size_t>(lead) >= fused->N_down.size())) {
        return detail::decline("shape");
    }
    const int reduce_cols = fused != nullptr ? fused->N_down[lead]
                                             : op2_k_for_act(N[lead], act);
    if (reduce != nullptr) {
        const status_t valid = validate_group_matmul_moe_postop(
                reduce, reduce_cols, params[lead].dtypes.dst);
        if (valid != status_t::success) return valid;
    }
    if (s.first < 0) {
        return reduce == nullptr
                ? status_t::success
                : group_matmul_moe_postop_execute(reduce, reduce_cols,
                          num_threads, params[lead].dtypes.dst);
    }
    const char *unserved = detail::check_grouped(a, s);
    if (unserved != nullptr) return detail::decline(unserved);
    try {
        return detail::run_grouped(a, s, since);
    } catch (const std::length_error &) {
        return status_t::memory_bad_size;
    } catch (const std::bad_alloc &) { return status_t::memory_bad_storage; }
}

} // namespace ntile_flat_parallel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
