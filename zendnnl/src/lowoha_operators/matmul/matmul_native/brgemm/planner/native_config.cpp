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

#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_config.hpp"

#include <cctype>
#include <charconv>
#include <cstddef>
#include <iterator>
#include <limits>
#include <variant>
#include <system_error>
#include <type_traits>

#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_tuning_keys.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

std::string_view native_text_trim(std::string_view s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front())))
        s.remove_prefix(1);
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())))
        s.remove_suffix(1);
    return s;
}

bool native_text_iequals(std::string_view a, std::string_view b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::tolower(static_cast<unsigned char>(a[i]))
                != std::tolower(static_cast<unsigned char>(b[i])))
            return false;
    }
    return true;
}

bool native_text_int64(std::string_view s, int64_t &v) {
    if (s.empty()) return false;
    const char *last = s.data() + s.size();
    const std::from_chars_result r = std::from_chars(s.data(), last, v);
    return r.ec == std::errc() && r.ptr == last;
}

bool native_text_bool(std::string_view s, bool &v) {
    if (s == "1" || native_text_iequals(s, "true")
            || native_text_iequals(s, "on")) {
        v = true;
        return true;
    }
    if (s == "0" || native_text_iequals(s, "false")
            || native_text_iequals(s, "off")) {
        v = false;
        return true;
    }
    return false;
}

namespace {

using key_type = native_key_type_t;

constexpr int64_t kBig20 = int64_t {1} << 20;
constexpr int64_t kBig30 = int64_t {1} << 30;
constexpr int64_t kBig40 = int64_t {1} << 40;

// name, type, min, max, step, pow2; a key that not every INT8 scheme applies
// lists the schemes that do
constexpr native_config_key_t kKeys[] = {
        {"nr", key_type::int32, 16, 128, 1, true},
        {"mr", key_type::int32, 1, 12, 1, false},
        {"mb", key_type::int32, 1, 8192, 1, false},
        {"kc", key_type::int32, 2, kBig30, 1, false},
        {"col_groups", key_type::int32, 0, 1024, 1, false},
        {"schedule", key_type::schedule, 0, 0, 1, false},
        {"traversal", key_type::traversal, 0, 0, 1, false},
        {"unit_tiles", key_type::int32, 1, 4096, 1, false},
        {"threads_cap", key_type::int32, 1, 4096, 1, false},
        {"min_tiles_per_thread", key_type::int32, 1, 64, 1, false},
        {"row_unit", key_type::int32, 1, 4096, 1, false},
        {"tile_rows", key_type::int32, 1, 12, 1, false},
        {"tall_rows", key_type::int32, 1, 12, 1, false},
        {"block_rows", key_type::int32, 1, 8192, 1, false},
        {"min_tile_cols", key_type::int32, 16, 32, 1, true},
        {"row_split", key_type::boolean, 0, 1, 1, false},
        {"min_pass_k", key_type::int32, 2, kBig20, 2, false,
                kNativeKeyPerChannel},
        {"bf16_l2_pct", key_type::int32, 1, 1000, 1, false},
        {"kc_split", key_type::kc_split, 0, 0, 1, false},
        {"l3_pct", key_type::int32, 1, 1000, 1, false},
        {"domain_threads", key_type::int32, 1, 1024, 1, false},
        {"int8_mr", key_type::int32, 1, 12, 1, false},
        {"int8_cw", key_type::cw_list, 0, 0, 1, false},
        {"int8_l2_pct", key_type::int32, 1, 1000, 1, false,
                kNativeKeyPerChannel},
        {"int8_kc", key_type::int32, 4, kBig20, 4, false, kNativeKeyPerChannel},
        {"int4_l1_pct", key_type::int32, 0, 1000, 1, false, kNativeKeyPerGroup},
        {"direct_store", key_type::boolean, 0, 1, 1, false},
        // The bias then follows the dequantization instead of fusing with
        // it in one FMA.
        {"bias_in_dequant", key_type::boolean, 0, 1, 1, false,
                kNativeKeyPerChannel},
        {"copy_block", key_type::int32, 1, 4096, 1, false},
        {"bf16_copy_par_min", key_type::int64, 0, kBig40, 1, false},
        {"int8_copy_par_min", key_type::int64, 0, kBig40, 1, false},
        {"makespan", key_type::int32, 0, 1, 1, false},
        {"min_gain_pct", key_type::int32, 0, 100, 1, false},
        {"idle_pct", key_type::int32, 1, 100, 1, false},
        {"decode_m", key_type::int32, 0, kBig30, 1, false},
        {"l1_pass", key_type::int32, 0, 1, 1, false, kNativeKeyPerChannel},
        {"l1_pct", key_type::int32, 1, 1000, 1, false, kNativeKeyPerChannel},
        {"l1_pass_max_mb", key_type::int32, 1, 8192, 1, false,
                kNativeKeyPerChannel},
        {"traffic_groups", key_type::int32, 0, 1, 1, false},
        {"bf16_alias", key_type::int32, 0, 1, 1, false},
        {"bf16_alias_mr", key_type::int32, 1, 12, 1, false},
        {"bf16_narrow_n", key_type::int32, 0, 1, 1, false},
        {"bf16_narrow_n_min_m", key_type::int32, 1, kBig30, 1, false},
        {"f16_wide_min_m", key_type::int32, 1, kBig30, 1, false},
        {"blocks", key_type::int32, 0, 1, 1, false},
        {"bf16_blocks_min_m", key_type::int32, 1, kBig30, 1, false},
        {"int8_blocks_min_m", key_type::int32, 1, kBig30, 1, false},
        // The cost model's switch, margin and the inputs of its price of a
        // tile slice also apply to the per-group tile of an s4 weight.
        {"cost_model", key_type::int32, 0, 1, 1, false},
        {"cm_gain_pct", key_type::int32, 0, 100, 1, false},
        {"cm_short_rows", key_type::int32, 1, 12, 1, false},
        {"cm_fma_pipes", key_type::int32, 0, 16, 1, false},
        {"cm_fma_lat", key_type::int32, 1, 64, 1, false},
        {"cm_call_cycles", key_type::int32, 0, kBig20, 1, false},
        {"cm_tall_pct", key_type::int32, -1, 100, 1, false},
        {"cm_l2_bytes", key_type::int32, 1, 4096, 1, false},
        {"cm_l3_cycles", key_type::int32, 0, kBig20, 1, false},
        {"cm_mem_cycles", key_type::int32, 0, kBig20, 1, false},
        {"cm_prefetch_kib", key_type::int32, 0, kBig20, 1, false},
        {"cm_overlap_pct", key_type::int32, 0, 100, 1, false},
        {"cm_reuse_pct", key_type::int32, 1, 1000, 1, false,
                kNativeKeyPerChannel},
        {"cm_stream_cycles", key_type::int32, 0, kBig20, 1, false},
        {"cm_sync_tiles", key_type::int32, 0, kBig20, 1, false},
        {"cm_group_slab_pct", key_type::int32, 1, 100000, 1, false,
                kNativeKeyPerChannel},
        {"cm_group_l3_pct", key_type::int32, 1, 100000, 1, false,
                kNativeKeyPerChannel},
        {"cm_group_skew_pct", key_type::int32, 100, 100000, 1, false,
                kNativeKeyPerChannel},
        {"cm_group_rows_pct", key_type::int32, 100, 200, 1, false,
                kNativeKeyPerChannel},
        {"cm_drift_kib", key_type::int32, 0, kBig30, 1, false},
        {"cm_pass_kib", key_type::int32, 0, kBig20, 1, false},
        {"cm_pass_calls", key_type::int32, 1, 4096, 1, false},
        {"cm_block_kib", key_type::int32, 0, kBig30, 1, false},
        {"cm_int8_l1_pct", key_type::int32, 0, 1000, 1, false,
                kNativeKeyPerChannel},
        {"cm_int4_min_m", key_type::int32, 0, kBig30, 1, false,
                kNativeKeyPerGroup},
};
constexpr size_t kKeyCount = sizeof(kKeys) / sizeof(kKeys[0]);

template <typename T>
struct binding_t {
    std::optional<T> native_config_t::*cfg;
    T heuristic_params_t::*params;
};

using field_t = std::variant<binding_t<int>, binding_t<int64_t>,
        binding_t<bool>, binding_t<uint8_t>, binding_t<native_schedule_t>,
        binding_t<native_traversal_t>, binding_t<native_kc_split_t>>;

template <typename T>
field_t bind(
        std::optional<T> native_config_t::*cfg, T heuristic_params_t::*params) {
    return binding_t<T> {cfg, params};
}

/// The fields of kKeys, in the same order.
const field_t *fields() {
    using c = native_config_t;
    using p = heuristic_params_t;
    static const field_t k[] = {
            bind(&c::nr, &p::nr),
            bind(&c::mr, &p::mr),
            bind(&c::mb, &p::mb),
            bind(&c::kc, &p::kc),
            bind(&c::col_groups, &p::col_groups),
            bind(&c::schedule, &p::schedule),
            bind(&c::traversal, &p::traversal),
            bind(&c::unit_tiles, &p::unit_tiles),
            bind(&c::threads_cap, &p::threads_cap),
            bind(&c::min_tiles_per_thread, &p::min_tiles_per_thread),
            bind(&c::row_unit, &p::row_unit),
            bind(&c::tile_rows, &p::tile_rows),
            bind(&c::tall_rows, &p::tall_rows),
            bind(&c::block_rows, &p::block_rows),
            bind(&c::min_tile_cols, &p::min_tile_cols),
            bind(&c::row_split, &p::row_split),
            bind(&c::min_pass_k, &p::min_pass_k),
            bind(&c::bf16_l2_pct, &p::bf16_l2_pct),
            bind(&c::kc_split, &p::kc_split),
            bind(&c::l3_pct, &p::l3_pct),
            bind(&c::domain_threads, &p::domain_threads),
            bind(&c::int8_mr, &p::int8_mr),
            bind(&c::int8_cw_mask, &p::int8_cw_mask),
            bind(&c::int8_l2_pct, &p::int8_l2_pct),
            bind(&c::int8_kc, &p::int8_kc),
            bind(&c::int4_l1_pct, &p::int4_l1_pct),
            bind(&c::direct_store, &p::direct_store),
            bind(&c::bias_in_dequant, &p::bias_in_dequant),
            bind(&c::copy_block, &p::copy_block),
            bind(&c::bf16_copy_par_min, &p::bf16_copy_par_min),
            bind(&c::int8_copy_par_min, &p::int8_copy_par_min),
            bind(&c::makespan, &p::makespan),
            bind(&c::min_gain_pct, &p::min_gain_pct),
            bind(&c::idle_pct, &p::idle_pct),
            bind(&c::decode_m, &p::decode_m),
            bind(&c::l1_pass, &p::l1_pass),
            bind(&c::l1_pct, &p::l1_pct),
            bind(&c::l1_pass_max_mb, &p::l1_pass_max_mb),
            bind(&c::traffic_groups, &p::traffic_groups),
            bind(&c::bf16_alias, &p::bf16_alias),
            bind(&c::bf16_alias_mr, &p::bf16_alias_mr),
            bind(&c::bf16_narrow_n, &p::bf16_narrow_n),
            bind(&c::bf16_narrow_n_min_m, &p::bf16_narrow_n_min_m),
            bind(&c::f16_wide_min_m, &p::f16_wide_min_m),
            bind(&c::blocks, &p::blocks),
            bind(&c::bf16_blocks_min_m, &p::bf16_blocks_min_m),
            bind(&c::int8_blocks_min_m, &p::int8_blocks_min_m),
            bind(&c::cost_model, &p::cost_model),
            bind(&c::cm_gain_pct, &p::cm_gain_pct),
            bind(&c::cm_short_rows, &p::cm_short_rows),
            bind(&c::cm_fma_pipes, &p::cm_fma_pipes),
            bind(&c::cm_fma_lat, &p::cm_fma_lat),
            bind(&c::cm_call_cycles, &p::cm_call_cycles),
            bind(&c::cm_tall_pct, &p::cm_tall_pct),
            bind(&c::cm_l2_bytes, &p::cm_l2_bytes),
            bind(&c::cm_l3_cycles, &p::cm_l3_cycles),
            bind(&c::cm_mem_cycles, &p::cm_mem_cycles),
            bind(&c::cm_prefetch_kib, &p::cm_prefetch_kib),
            bind(&c::cm_overlap_pct, &p::cm_overlap_pct),
            bind(&c::cm_reuse_pct, &p::cm_reuse_pct),
            bind(&c::cm_stream_cycles, &p::cm_stream_cycles),
            bind(&c::cm_sync_tiles, &p::cm_sync_tiles),
            bind(&c::cm_group_slab_pct, &p::cm_group_slab_pct),
            bind(&c::cm_group_l3_pct, &p::cm_group_l3_pct),
            bind(&c::cm_group_skew_pct, &p::cm_group_skew_pct),
            bind(&c::cm_group_rows_pct, &p::cm_group_rows_pct),
            bind(&c::cm_drift_kib, &p::cm_drift_kib),
            bind(&c::cm_pass_kib, &p::cm_pass_kib),
            bind(&c::cm_pass_calls, &p::cm_pass_calls),
            bind(&c::cm_block_kib, &p::cm_block_kib),
            bind(&c::cm_int8_l1_pct, &p::cm_int8_l1_pct),
            bind(&c::cm_int4_min_m, &p::cm_int4_min_m),
    };
    static_assert(sizeof(k) / sizeof(k[0]) == kKeyCount, "one field per key");
    return k;
}

const char *const kScheduleNames[] = {"auto", "slice", "row_groups", "blocks"};
const char *const kTraversalNames[] = {"m_inner", "n_inner"};
const char *const kKcSplitNames[] = {"budget", "equal"};

/// "32/16" (any order, any subset) as a width mask.
bool parse_cw(std::string_view s, uint8_t &mask) {
    mask = 0;
    for (;;) {
        const size_t slash = s.find('/');
        int64_t w = 0;
        if (!native_text_int64(native_text_trim(s.substr(0, slash)), w))
            return false;
        if (w == 16)
            mask = static_cast<uint8_t>(mask | kNativeInt8Cw16);
        else if (w == 32)
            mask = static_cast<uint8_t>(mask | kNativeInt8Cw32);
        else
            return false;
        if (slash == std::string_view::npos) return true;
        s.remove_prefix(slash + 1);
    }
}

template <typename E, size_t N>
bool parse_enum(std::string_view s, const char *const (&names)[N], E &v) {
    for (size_t i = 0; i < N; ++i) {
        if (native_text_iequals(s, names[i])) {
            v = static_cast<E>(i);
            return true;
        }
    }
    return false;
}

template <typename E, size_t N>
bool enum_ok(E v, const char *const (&)[N]) {
    return static_cast<size_t>(v) < N;
}

template <typename E, size_t N>
const char *enum_name(E v, const char *const (&names)[N]) {
    return enum_ok(v, names) ? names[static_cast<size_t>(v)] : "?";
}

template <typename T>
bool parse_value(std::string_view s, T &v) {
    if constexpr (std::is_same_v<T, bool>) {
        return native_text_bool(s, v);
    } else if constexpr (std::is_same_v<T, uint8_t>) {
        return parse_cw(s, v);
    } else if constexpr (std::is_same_v<T, native_schedule_t>) {
        return parse_enum(s, kScheduleNames, v);
    } else if constexpr (std::is_same_v<T, native_traversal_t>) {
        return parse_enum(s, kTraversalNames, v);
    } else if constexpr (std::is_same_v<T, native_kc_split_t>) {
        return parse_enum(s, kKcSplitNames, v);
    } else if constexpr (std::is_same_v<T, int>) {
        int64_t x = 0;
        if (!native_text_int64(s, x) || x < std::numeric_limits<int>::min()
                || x > std::numeric_limits<int>::max())
            return false;
        v = static_cast<int>(x);
        return true;
    } else {
        return native_text_int64(s, v);
    }
}

template <typename T>
std::string format_value(const T &v) {
    if constexpr (std::is_same_v<T, bool>) {
        return v ? "1" : "0";
    } else if constexpr (std::is_same_v<T, uint8_t>) {
        std::string s;
        for (const int w : {32, 16}) {
            const int bit = w == 32 ? kNativeInt8Cw32 : kNativeInt8Cw16;
            if ((v & bit) == 0) continue;
            if (!s.empty()) s += '/';
            s += std::to_string(w);
        }
        return s;
    } else if constexpr (std::is_same_v<T, native_schedule_t>) {
        return enum_name(v, kScheduleNames);
    } else if constexpr (std::is_same_v<T, native_traversal_t>) {
        return enum_name(v, kTraversalNames);
    } else if constexpr (std::is_same_v<T, native_kc_split_t>) {
        return enum_name(v, kKcSplitNames);
    } else {
        return std::to_string(v);
    }
}

std::string range_text(const native_config_key_t &key, int64_t v) {
    std::string s = std::string(key.name) + "=" + std::to_string(v)
            + " is not one of ";
    if (key.pow2)
        s += "the powers of two in ";
    else if (key.step > 1)
        s += "the multiples of " + std::to_string(key.step) + " in ";
    return s + std::to_string(key.min) + " .. " + std::to_string(key.max);
}

template <typename T>
bool check_value(const native_config_key_t &key, const T &v, std::string &why) {
    if constexpr (std::is_same_v<T, bool>) {
        return true;
    } else if constexpr (std::is_same_v<T, uint8_t>) {
        if ((v & kNativeInt8Cw16) != 0 && (v & ~kNativeInt8CwAll) == 0)
            return true;
        why = "int8_cw must list 16, and only 16 or 32";
        return false;
    } else if constexpr (std::is_same_v<T, native_schedule_t>) {
        if (enum_ok(v, kScheduleNames)) return true;
        why = "bad schedule";
        return false;
    } else if constexpr (std::is_same_v<T, native_traversal_t>) {
        if (enum_ok(v, kTraversalNames)) return true;
        why = "bad traversal";
        return false;
    } else if constexpr (std::is_same_v<T, native_kc_split_t>) {
        if (enum_ok(v, kKcSplitNames)) return true;
        why = "bad kc_split";
        return false;
    } else {
        const int64_t x = v;
        if (x >= key.min && x <= key.max && x % key.step == 0
                && (!key.pow2 || (x & (x - 1)) == 0))
            return true;
        why = range_text(key, x);
        return false;
    }
}

template <typename T>
bool parse_into(std::optional<T> &slot, std::string_view value) {
    T v {};
    if (!parse_value(value, v)) return false;
    slot = v;
    return true;
}

/// Parses one "key=value" into cfg; a key may appear once.
bool parse_setting(
        std::string_view token, native_config_t &cfg, std::string &why) {
    const size_t eq = token.find('=');
    if (eq == std::string_view::npos) {
        why = "'" + std::string(token) + "' is not key=value";
        return false;
    }
    const std::string_view key = native_text_trim(token.substr(0, eq));
    const std::string_view value = native_text_trim(token.substr(eq + 1));
    size_t i = 0;
    while (i < kKeyCount && !native_text_iequals(key, kKeys[i].name))
        ++i;
    if (i == kKeyCount) {
        why = "unknown key '" + std::string(key) + "'";
        return false;
    }
    bool twice = false;
    bool parsed = true;
    std::visit([&](auto b) {
        auto &slot = cfg.*b.cfg;
        if (slot)
            twice = true;
        else
            parsed = parse_into(slot, value);
    }, fields()[i]);
    if (twice) {
        why = "key '" + std::string(kKeys[i].name) + "' given twice";
        return false;
    }
    if (!parsed) {
        why = "bad value '" + std::string(value) + "' for " + kKeys[i].name;
        return false;
    }
    return true;
}

} // namespace

const native_config_key_t *native_config_keys(size_t &count) {
    count = kKeyCount;
    return kKeys;
}

bool native_config_t::empty() const {
    for (size_t i = 0; i < kKeyCount; ++i) {
        bool set = false;
        std::visit(
                [&](auto b) { set = (this->*b.cfg).has_value(); }, fields()[i]);
        if (set) return false;
    }
    return true;
}

bool native_config_t::operator==(const native_config_t &o) const {
    for (size_t i = 0; i < kKeyCount; ++i) {
        bool same = true;
        std::visit([&](auto b) { same = (this->*b.cfg) == (o.*b.cfg); },
                fields()[i]);
        if (!same) return false;
    }
    return true;
}

bool parse_native_config(
        std::string_view text, native_config_t &cfg, std::string &why) {
    cfg = native_config_t {};
    text = native_text_trim(text);
    if (text.empty()) return true;
    for (;;) {
        const size_t comma = text.find(',');
        if (!parse_setting(native_text_trim(text.substr(0, comma)), cfg, why))
            return false;
        if (comma == std::string_view::npos) break;
        text.remove_prefix(comma + 1);
    }
    return check_native_config(cfg, why);
}

std::string format_native_config(const native_config_t &cfg) {
    std::string out;
    for (size_t i = 0; i < kKeyCount; ++i) {
        std::visit([&](auto b) {
            const auto &v = cfg.*b.cfg;
            if (!v) return;
            if (!out.empty()) out += ',';
            out += kKeys[i].name;
            out += '=';
            out += format_value(*v);
        }, fields()[i]);
    }
    return out;
}

void merge_native_config(
        native_config_t &base, const native_config_t &overlay) {
    for (size_t i = 0; i < kKeyCount; ++i) {
        std::visit([&](auto b) {
            if (overlay.*b.cfg) base.*b.cfg = overlay.*b.cfg;
        }, fields()[i]);
    }
    if (overlay.schedule || overlay.col_groups) {
        base.schedule = overlay.schedule;
        base.col_groups = overlay.col_groups;
    }
}

bool check_native_config(const native_config_t &cfg, std::string &why) {
    for (size_t i = 0; i < kKeyCount; ++i) {
        bool ok = true;
        std::visit([&](auto b) {
            const auto &v = cfg.*b.cfg;
            if (v) ok = check_value(kKeys[i], *v, why);
        }, fields()[i]);
        if (!ok) return false;
    }
    if (cfg.schedule && cfg.col_groups) {
        if (*cfg.schedule == native_schedule_t::slice && *cfg.col_groups > 0) {
            why = "schedule=slice with col_groups > 0";
            return false;
        }
        if (*cfg.schedule == native_schedule_t::row_groups
                && *cfg.col_groups == 0) {
            why = "schedule=row_groups with col_groups=0";
            return false;
        }
        if (*cfg.schedule == native_schedule_t::blocks) {
            why = "schedule=blocks with col_groups";
            return false;
        }
    }
    return true;
}

heuristic_params_t apply_native_config(
        heuristic_params_t params, const native_config_t &cfg) {
    for (size_t i = 0; i < kKeyCount; ++i) {
        std::visit([&](auto b) {
            if (cfg.*b.cfg) params.*b.params = *(cfg.*b.cfg);
        }, fields()[i]);
    }
    return params;
}

native_config_t native_config_for_scheme(const native_config_t &cfg,
        native_int8_scheme_t scheme, std::string *dropped) {
    native_config_t out = cfg;
    for (size_t i = 0; i < kKeyCount; ++i) {
        if (kKeys[i].applies_to(scheme)) continue;
        std::visit([&](auto b) {
            auto &v = out.*b.cfg;
            if (!v) return;
            v.reset();
            if (dropped == nullptr) return;
            if (!dropped->empty()) *dropped += ',';
            *dropped += kKeys[i].name;
        }, fields()[i]);
    }
    return out;
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
