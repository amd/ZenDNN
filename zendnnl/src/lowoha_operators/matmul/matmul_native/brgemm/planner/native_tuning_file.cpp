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

#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_tuning_file.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <utility>
#include <vector>
#include <string_view>
#include <system_error>
#include <unordered_map>

#include "common/zendnnl_global.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_tuning_keys.hpp"
#include "nlohmann/json.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

using zendnnl::error_handling::apilog_info;
using zendnnl::error_handling::apilog_info_enabled;
using zendnnl::error_handling::apilog_verbose;
using zendnnl::error_handling::apilog_verbose_enabled;
using zendnnl::error_handling::apilog_warning;

namespace {

using json = nlohmann::json;

constexpr const char *kConfigVariable = "ZENDNNL_MATMUL_NATIVE_CONFIG";
constexpr const char *kTuningFormat = kNativeTuningFormat;
constexpr int64_t kTuningVersion = kNativeTuningVersion;
constexpr size_t kMaxClauses = 64;
constexpr uint64_t kMaxFileBytes = uint64_t {64} << 20;
constexpr size_t kMaxEntries = 65536;
constexpr size_t kMaxEntryWarnings = 16;
constexpr size_t kMaxTreeNodes = 65535;
constexpr int kMaxTreeDepth = 64;
/// Arrays and objects nested deeper make the whole file ignored.
constexpr int kMaxJsonNesting = 128;
/// More values make the whole file ignored: they bound the parsed document
/// (tens of bytes each), whatever the text size.
constexpr size_t kMaxJsonValues = size_t {1} << 21;
constexpr size_t kMaxOrderedRules = 4096;
constexpr size_t kMaxConditions = 32;

constexpr auto &kMatchKeys = kNativeMatchKeys;
constexpr int kNumericKeys = kNativeNumericMatchKeys;
constexpr int kNamedKeys = kNativeNamedMatchKeys;
constexpr int kFlagKeys = 3;
static_assert(std::size(kMatchKeys) == kNumericKeys + kNamedKeys + kFlagKeys,
        "every match key has a kind");

constexpr auto &kDtypeNames = kNativeDtypeNames;
const data_type_t kDtypeTypes[]
        = {data_type_t::bf16, data_type_t::s8, data_type_t::f32};
constexpr auto &kSrcNames = kNativeSrcNames;
const data_type_t kSrcTypes[] = {
        data_type_t::bf16, data_type_t::f32, data_type_t::u8, data_type_t::s8};
constexpr auto &kDstNames = kNativeDstNames;
const data_type_t kDstTypes[] = {data_type_t::f32, data_type_t::bf16,
        data_type_t::f16, data_type_t::u8, data_type_t::s8, data_type_t::s32};
const int kDstBytes[] = {4, 2, 2, 1, 1, 4};
constexpr auto &kEpilogueNames = kNativeEpilogueNames;
static_assert(std::size(kDtypeTypes) == std::size(kDtypeNames)
                && std::size(kSrcTypes) == std::size(kSrcNames)
                && std::size(kDstTypes) == std::size(kDstNames)
                && std::size(kDstBytes) == std::size(kDstNames),
        "a type per name");

constexpr auto &kOps = kNativeRuleOps;

enum feature_t : int {
    f_m,
    f_n,
    f_k,
    f_threads,
    f_pack_nr,
    f_bf16,
    f_dynamic,
    f_trans_a,
    f_has_bias,
    f_epilogue,
    f_dst_bytes,
    f_esz,
    f_a_bytes,
    f_b_bytes,
    f_col_tiles,
    f_m_per_thread,
    f_n_per_thread,
    f_flops_per_thread,
    f_l2,
    f_l3,
    f_l3_threads,
    f_smt,
    f_cores,
    f_a_l3,
    f_b_l3,
    f_slab_l2,
    kFeatureCount
};
constexpr auto &kFeatureNames = kNativeRuleFeatures;
static_assert(std::size(kFeatureNames) == kFeatureCount, "a name per feature");

struct names_t {
    const char *const *names;
    size_t count;
};

names_t named_key_values(int named) {
    switch (named) {
        case 0: return {kDtypeNames, std::size(kDtypeNames)};
        case 1: return {kSrcNames, std::size(kSrcNames)};
        case 2: return {kDstNames, std::size(kDstNames)};
        default: return {kEpilogueNames, std::size(kEpilogueNames)};
    }
}

template <size_t N>
int type_index(const data_type_t (&types)[N], data_type_t dt) {
    for (size_t i = 0; i < N; ++i) {
        if (types[i] == dt) return static_cast<int>(i);
    }
    return -1;
}

template <size_t N>
int name_index(const char *const (&names)[N], const json &v) {
    if (!v.is_string()) return -1;
    const std::string &s = v.get_ref<const std::string &>();
    for (size_t i = 0; i < N; ++i) {
        if (s == names[i]) return static_cast<int>(i);
    }
    return -1;
}

int match_key(std::string_view name, bool any_case) {
    for (size_t i = 0; i < std::size(kMatchKeys); ++i) {
        if (any_case ? native_text_iequals(name, kMatchKeys[i])
                     : name == kMatchKeys[i])
            return static_cast<int>(i);
    }
    return -1;
}

/// Bit of name among the values of named key named; 0 when it is none.
uint8_t name_bit(int named, std::string_view name) {
    const names_t v = named_key_values(named);
    for (size_t i = 0; i < v.count; ++i) {
        if (native_text_iequals(name, v.names[i]))
            return static_cast<uint8_t>(1u << i);
    }
    return 0;
}

/// s for a warning: at most 64 characters, '?' for one that is not
/// printable.
std::string name_text(const std::string &s) {
    std::string t = s.size() > 64 ? s.substr(0, 64) + "..." : s;
    for (char &c : t) {
        if (!std::isprint(static_cast<unsigned char>(c))) c = '?';
    }
    return t;
}

std::string id_text(const std::string &id) {
    return id.empty() ? std::string() : " (id " + name_text(id) + ")";
}

/// v for a warning: scalars as JSON, arrays and objects by their size only
/// (dump recurses once per nesting level, and the text is unbounded).
std::string value_text(const json &v) {
    if (v.is_array()) return "<array of " + std::to_string(v.size()) + ">";
    if (v.is_object())
        return "<object of " + std::to_string(v.size()) + " keys>";
    if (v.is_string() && v.get_ref<const std::string &>().size() > 64)
        return "<string of "
                + std::to_string(v.get_ref<const std::string &>().size())
                + " characters>";
    return v.dump();
}

/// The deepest nesting of arrays and objects in text and an upper bound of
/// its values (opening brackets and commas, plus one), outside strings.
struct json_shape_t {
    int depth = 0;
    size_t values = 1;
};

json_shape_t json_shape(std::string_view text) {
    json_shape_t shape;
    int depth = 0;
    bool in_string = false, escaped = false;
    for (const char c : text) {
        if (in_string) {
            if (escaped)
                escaped = false;
            else if (c == '\\')
                escaped = true;
            else if (c == '"')
                in_string = false;
        } else if (c == '"') {
            in_string = true;
        } else if (c == '[' || c == '{') {
            shape.depth = std::max(shape.depth, ++depth);
            ++shape.values;
        } else if (c == ']' || c == '}') {
            --depth;
        } else if (c == ',') {
            ++shape.values;
        }
    }
    return shape;
}

// ---------------------------------------------------------------------------
// Problem matching

struct range_t {
    int64_t lo = 0;
    int64_t hi = 0;
};

/// The filters of an override clause or a table entry; an absent key matches
/// any value.
struct problem_match_t {
    std::vector<range_t> numeric[kNumericKeys]; ///< any range holds
    uint8_t named[kNamedKeys] = {}; ///< masks of names; 0: any
    int8_t flags[kFlagKeys] = {-1, -1, -1}; ///< -1: any
    uint32_t given = 0; ///< bit per match key read

    bool holds(const native_problem_t &p) const noexcept;
};

int64_t numeric_value(const native_problem_t &p, int key) {
    switch (key) {
        case 0: return p.M;
        case 1: return p.N;
        case 2: return p.K;
        case 3: return p.threads;
        default: return p.pack_nr;
    }
}

int named_value(const native_problem_t &p, int named) {
    switch (named) {
        case 0: return type_index(kDtypeTypes, p.wei_dt);
        case 1: return type_index(kSrcTypes, p.src_dt);
        case 2: return type_index(kDstTypes, p.dst_dt);
        default: return static_cast<int>(p.epilogue);
    }
}

bool flag_value(const native_problem_t &p, int flag) {
    return flag == 0 ? p.trans_a : (flag == 1 ? p.dynamic_quant : p.has_bias);
}

bool problem_match_t::holds(const native_problem_t &p) const noexcept {
    for (int i = 0; i < kNumericKeys; ++i) {
        if (numeric[i].empty()) continue;
        const int64_t v = numeric_value(p, i);
        bool any = false;
        for (const range_t &r : numeric[i]) {
            if (v >= r.lo && v <= r.hi) {
                any = true;
                break;
            }
        }
        if (!any) return false;
    }
    for (int i = 0; i < kNamedKeys; ++i) {
        if (named[i] == 0) continue;
        const int v = named_value(p, i);
        if (v < 0 || (named[i] & (1u << v)) == 0) return false;
    }
    for (int i = 0; i < kFlagKeys; ++i) {
        if (flags[i] >= 0 && (flags[i] != 0) != flag_value(p, i)) return false;
    }
    return true;
}

struct matched_entry_t {
    problem_match_t match;
    native_config_entry_t entry;
};

// ---------------------------------------------------------------------------
// ZENDNNL_MATMUL_NATIVE_CONFIG

/// Calls f(piece) for the pieces of s between delimiters; false when f does.
template <typename F>
bool for_each_piece(std::string_view s, char delimiter, F &&f) {
    for (;;) {
        const size_t at = s.find(delimiter);
        if (!f(s.substr(0, at))) return false;
        if (at == std::string_view::npos) return true;
        s.remove_prefix(at + 1);
    }
}

/// "a", "a-b" or "a-" (a or more): non-negative and inclusive.
bool parse_range(std::string_view s, range_t &r) {
    const auto digit = [](std::string_view t) {
        return !t.empty() && std::isdigit(static_cast<unsigned char>(t[0]));
    };
    s = native_text_trim(s);
    const size_t dash = s.find('-');
    const std::string_view lo = native_text_trim(s.substr(0, dash));
    if (!digit(lo) || !native_text_int64(lo, r.lo)) return false;
    r.hi = r.lo;
    if (dash == std::string_view::npos) return true;
    const std::string_view hi = native_text_trim(s.substr(dash + 1));
    if (hi.empty()) {
        r.hi = std::numeric_limits<int64_t>::max();
        return true;
    }
    return digit(hi) && native_text_int64(hi, r.hi) && r.lo <= r.hi;
}

bool parse_filter(
        std::string_view token, problem_match_t &m, std::string &why) {
    const size_t eq = token.find('=');
    if (eq == std::string_view::npos) {
        why = "filter '" + std::string(token) + "' is not key=value";
        return false;
    }
    const std::string_view key = native_text_trim(token.substr(0, eq));
    const std::string_view value = native_text_trim(token.substr(eq + 1));
    const int i = match_key(key, true);
    if (i < 0) {
        why = "unknown filter '" + std::string(key) + "'";
        return false;
    }
    if (((m.given >> i) & 1u) != 0) {
        why = "filter '" + std::string(kMatchKeys[i]) + "' given twice";
        return false;
    }
    m.given |= 1u << i;
    bool ok = true;
    if (i < kNumericKeys) {
        range_t r;
        ok = parse_range(value, r);
        if (ok) m.numeric[i].push_back(r);
    } else if (i < kNumericKeys + kNamedKeys) {
        const int named = i - kNumericKeys;
        ok = for_each_piece(value, '/', [&](std::string_view name) {
            const uint8_t bit = name_bit(named, native_text_trim(name));
            m.named[named] = static_cast<uint8_t>(m.named[named] | bit);
            return bit != 0;
        });
    } else {
        bool b = false;
        ok = native_text_bool(value, b);
        m.flags[i - kNumericKeys - kNamedKeys] = b ? 1 : 0;
    }
    if (!ok)
        why = "bad value '" + std::string(value) + "' for filter "
                + kMatchKeys[i];
    return ok;
}

/// clause = [ filters ":" ] [ settings ].
bool parse_clause(
        std::string_view clause, matched_entry_t &c, std::string &why) {
    std::string_view settings = clause;
    const size_t colon = clause.find(':');
    if (colon != std::string_view::npos) {
        const std::string_view filters
                = native_text_trim(clause.substr(0, colon));
        settings = clause.substr(colon + 1);
        if (filters.empty()) {
            why = "no filter before ':'";
            return false;
        }
        if (!for_each_piece(filters, ',', [&](std::string_view f) {
            return parse_filter(native_text_trim(f), c.match, why);
        }))
            return false;
    }
    return parse_native_config(settings, c.entry.config, why);
}

class override_source_t final : public native_config_source_t {
public:
    const native_config_entry_t *lookup(
            const native_problem_t &p) const noexcept override {
        for (const auto &c : clauses) {
            if (c->match.holds(p)) return &c->entry;
        }
        return nullptr;
    }

    std::vector<std::unique_ptr<matched_entry_t>> clauses;
};

// ---------------------------------------------------------------------------
// Tuning file: values

/// An integer JSON value that fits int64_t.
bool json_int(const json &v, int64_t &out) {
    if (v.is_number_unsigned()) {
        const uint64_t u = v.get<uint64_t>();
        if (u > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
            return false;
        out = static_cast<int64_t>(u);
        return true;
    }
    if (!v.is_number_integer()) return false;
    out = v.get<int64_t>();
    return true;
}

/// A number, an array of numbers (any of them) or {"min", "max"} (inclusive,
/// either bound may be omitted); integers only.
bool json_ranges(const json &v, std::vector<range_t> &out) {
    int64_t x = 0;
    if (json_int(v, x)) {
        out.push_back({x, x});
        return true;
    }
    if (v.is_array()) {
        for (const json &e : v) {
            if (!json_int(e, x)) return false;
            out.push_back({x, x});
        }
        return !v.empty();
    }
    if (!v.is_object() || v.empty()) return false;
    range_t r {std::numeric_limits<int64_t>::min(),
            std::numeric_limits<int64_t>::max()};
    for (auto it = v.begin(); it != v.end(); ++it) {
        int64_t b = 0;
        if (!json_int(it.value(), b)) return false;
        if (it.key() == "min")
            r.lo = b;
        else if (it.key() == "max")
            r.hi = b;
        else
            return false;
    }
    out.push_back(r);
    return r.lo <= r.hi;
}

/// A name or an array of names (any of them).
bool json_names(const json &v, int named, uint8_t &mask) {
    const auto add = [&](const json &s) {
        const uint8_t bit = s.is_string()
                ? name_bit(named, s.get_ref<const std::string &>())
                : 0;
        mask = static_cast<uint8_t>(mask | bit);
        return bit != 0;
    };
    if (v.is_string()) return add(v);
    if (!v.is_array() || v.empty()) return false;
    for (const json &s : v) {
        if (!add(s)) return false;
    }
    return true;
}

bool json_match(const json &m, problem_match_t &out, std::string &why) {
    if (!m.is_object()) {
        why = "match is not an object";
        return false;
    }
    for (auto it = m.begin(); it != m.end(); ++it) {
        const int i = match_key(it.key(), false);
        if (i < 0) {
            why = "unknown match key '" + name_text(it.key()) + "'";
            return false;
        }
        const json &v = it.value();
        bool ok = false;
        if (i < kNumericKeys) {
            ok = json_ranges(v, out.numeric[i]);
        } else if (i < kNumericKeys + kNamedKeys) {
            ok = json_names(v, i - kNumericKeys, out.named[i - kNumericKeys]);
        } else if (v.is_boolean()) {
            ok = true;
            out.flags[i - kNumericKeys - kNamedKeys] = v.get<bool>() ? 1 : 0;
        }
        if (!ok) {
            why = "bad match value " + name_text(it.key()) + "="
                    + value_text(v);
            return false;
        }
    }
    return true;
}

/// v in the key=value text form of key; false on a wrong JSON type.
bool json_value_text(
        const native_config_key_t &key, const json &v, std::string &text) {
    switch (key.type) {
        case native_key_type_t::int32:
        case native_key_type_t::int64:
            if (!v.is_number_integer()) return false;
            text = v.dump();
            return true;
        case native_key_type_t::boolean:
            if (!v.is_boolean()) return false;
            text = v.get<bool>() ? "1" : "0";
            return true;
        case native_key_type_t::cw_list:
            if (!v.is_array() || v.empty()) return false;
            text.clear();
            for (const json &w : v) {
                if (!w.is_number_integer()) return false;
                if (!text.empty()) text += '/';
                text += w.dump();
            }
            return true;
        default: {
            if (!v.is_string()) return false;
            // Value names are lower-case letters, digits and '_': nothing
            // that separates the text form's tokens.
            const std::string &s = v.get_ref<const std::string &>();
            for (const char c : s) {
                const unsigned char u = static_cast<unsigned char>(c);
                if (!std::islower(u) && !std::isdigit(u) && c != '_')
                    return false;
            }
            text = s;
            return !s.empty();
        }
    }
}

bool json_config(const json &v, native_config_t &cfg, std::string &why) {
    if (!v.is_object()) {
        why = "config is not an object";
        return false;
    }
    size_t count = 0;
    const native_config_key_t *keys = native_config_keys(count);
    std::string text;
    for (auto it = v.begin(); it != v.end(); ++it) {
        const native_config_key_t *key = nullptr;
        for (size_t i = 0; i < count && key == nullptr; ++i) {
            if (it.key() == keys[i].name) key = &keys[i];
        }
        if (key == nullptr) {
            why = "unknown config key '" + name_text(it.key()) + "'";
            return false;
        }
        std::string value;
        if (!json_value_text(*key, it.value(), value)) {
            why = "config " + name_text(it.key()) + "=" + value_text(it.value())
                    + " has the wrong JSON type";
            return false;
        }
        if (!text.empty()) text += ',';
        text += it.key();
        text += '=';
        text += value;
    }
    return parse_native_config(text, cfg, why);
}

/// {"id", "match", "config"}, each optional.
bool json_entry(const json &e, matched_entry_t &out, std::string &id,
        std::string &why) {
    if (!e.is_object()) {
        why = "not an object";
        return false;
    }
    const auto idv = e.find("id");
    if (idv != e.end()) {
        if (!idv->is_string()) {
            why = "id is not a string";
            return false;
        }
        id = idv->get<std::string>();
    }
    for (auto it = e.begin(); it != e.end(); ++it) {
        if (it.key() == "id") continue;
        if (it.key() == "match") {
            if (!json_match(it.value(), out.match, why)) return false;
        } else if (it.key() == "config") {
            if (!json_config(it.value(), out.entry.config, why)) return false;
        } else {
            why = "unknown entry key '" + name_text(it.key()) + "'";
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Tuning file: hardware

bool hw_has_isa(const native_hw_t &hw, const std::string &name) {
    if (name == "avx512f") return hw.uarch.avx512f;
    if (name == "avx512bf16") return hw.uarch.avx512bf16;
    if (name == "avx512vnni") return hw.uarch.avx512vnni;
    return false;
}

// Each hardware key is a field of the CPU, checked in this order.  Numbers
// take a value, an array of values or {"min", "max"}; vendor a string or an
// array of strings; isa the features the CPU must all have.
enum hw_key_t : int {
    hk_vendor,
    hk_family,
    hk_model,
    hk_stepping,
    hk_isa,
    hk_l1d_kib,
    hk_l2_kib,
    hk_l3_kib,
    hk_l3_cores,
    hk_smt,
    hk_l3_threads,
    hk_cores,
    hk_logical_cpus,
    kHwKeyCount
};
constexpr auto &kHwKeys = kNativeHardwareKeys;
static_assert(std::size(kHwKeys) == kHwKeyCount, "a name per hardware key");
constexpr auto &kIsaNames = kNativeIsaNames;

int hw_key(const std::string &name) {
    for (int k = 0; k < kHwKeyCount; ++k) {
        if (name == kHwKeys[k]) return k;
    }
    return -1;
}

/// The physical cores of the L3 domain the plan uses on hw.
int plan_domain_cores(const native_hw_t &hw) {
    const UarchParams u = native_plan_uarch(hw);
    return u.l3_domain_cores > 0 ? u.l3_domain_cores : u.family_l3_cores;
}

/// hw's value of numeric key k; -1 when hw does not report it.
int64_t hw_field(const native_hw_t &hw, int k) {
    int64_t v = 0;
    switch (k) {
        case hk_family: return hw.family >= 0 ? hw.family : -1;
        case hk_model: return hw.model >= 0 ? hw.model : -1;
        case hk_stepping: return hw.stepping >= 0 ? hw.stepping : -1;
        case hk_l1d_kib:
            v = hw.uarch.l1d_detected ? hw.uarch.l1d_bytes / 1024 : 0;
            break;
        case hk_l2_kib:
            v = hw.uarch.l2_detected ? hw.uarch.l2_bytes / 1024 : 0;
            break;
        case hk_l3_kib:
            v = hw.uarch.l3_detected ? hw.uarch.l3_domain_bytes / 1024 : 0;
            break;
        case hk_l3_cores: v = plan_domain_cores(hw); break;
        case hk_smt: v = hw.smt; break;
        case hk_l3_threads: v = hw.l3_domain_threads; break;
        case hk_cores: v = hw.physical_cores; break;
        case hk_logical_cpus: v = hw.logical_cpus; break;
        default: break;
    }
    return v > 0 ? v : -1;
}

struct hw_condition_t {
    int key = 0;
    std::vector<range_t> ranges; ///< numbers: any holds
    std::vector<std::string> names; ///< vendor: any is it; isa: all present
    std::string text; ///< the file's value, for messages
};

/// v as JSON when it is an array or object of at most eight short scalars,
/// else value_text(v).
std::string short_text(const json &v) {
    if (!v.is_array() && !v.is_object()) return value_text(v);
    if (v.size() > 8) return value_text(v);
    for (auto it = v.begin(); it != v.end(); ++it) {
        const bool long_key = v.is_object() && it.key().size() > 64;
        const bool long_string = it->is_string()
                && it->get_ref<const std::string &>().size() > 64;
        if (long_key || long_string || it->is_array() || it->is_object())
            return value_text(v);
    }
    return v.dump();
}

/// A string or a non-empty array of strings.
bool json_strings(const json &v, std::vector<std::string> &out) {
    if (v.is_string()) {
        out.push_back(v.get<std::string>());
        return true;
    }
    if (!v.is_array() || v.empty()) return false;
    for (const json &s : v) {
        if (!s.is_string()) return false;
        out.push_back(s.get<std::string>());
    }
    return true;
}

/// The conditions of a hardware object in check order; false with why set at
/// an unknown key, a malformed value or an unknown ISA name.
bool parse_hardware(
        const json &h, std::vector<hw_condition_t> &out, std::string &why) {
    if (!h.is_object()) {
        why = "hardware is not an object";
        return false;
    }
    const json *given[kHwKeyCount] = {};
    for (auto it = h.begin(); it != h.end(); ++it) {
        const int k = hw_key(it.key());
        if (k < 0) {
            why = "unknown hardware key '" + name_text(it.key()) + "'";
            return false;
        }
        given[k] = &it.value();
    }
    for (int k = 0; k < kHwKeyCount; ++k) {
        if (given[k] == nullptr) continue;
        const json &v = *given[k];
        hw_condition_t c;
        c.key = k;
        c.text = short_text(v);
        bool ok = false;
        if (k == hk_vendor) {
            ok = json_strings(v, c.names);
        } else if (k == hk_isa) {
            ok = (v.is_array() && v.empty()) || json_strings(v, c.names);
            for (const std::string &f : c.names) {
                if (ok
                        && std::find(std::begin(kIsaNames), std::end(kIsaNames),
                                   f)
                                == std::end(kIsaNames)) {
                    why = "hardware.isa names " + value_text(json(f))
                            + ", which this build does not know";
                    return false;
                }
            }
        } else {
            ok = json_ranges(v, c.ranges);
        }
        if (!ok) {
            why = std::string("hardware.") + kHwKeys[k] + " has a bad value "
                    + c.text;
            return false;
        }
        out.push_back(std::move(c));
    }
    return true;
}

/// The first condition hw fails, as why; true when every one holds.
bool hardware_holds(const std::vector<hw_condition_t> &conditions,
        const native_hw_t &hw, std::string &why) {
    for (const hw_condition_t &c : conditions) {
        const std::string key = std::string("hardware.") + kHwKeys[c.key];
        if (c.key == hk_isa) {
            for (const std::string &f : c.names) {
                if (!hw_has_isa(hw, f)) {
                    why = key + " lists \"" + f + "\", which this CPU lacks";
                    return false;
                }
            }
            continue;
        }
        const bool vendor = c.key == hk_vendor;
        const int64_t x = vendor ? 0 : hw_field(hw, c.key);
        if (vendor ? hw.vendor.empty() : x < 0) {
            why = key + " cannot be checked: this CPU does not report it";
            return false;
        }
        bool holds = false;
        if (vendor) {
            holds = std::find(c.names.begin(), c.names.end(), hw.vendor)
                    != c.names.end();
        } else {
            for (const range_t &r : c.ranges)
                holds = holds || (x >= r.lo && x <= r.hi);
        }
        if (!holds) {
            why = key + "=" + c.text + " does not hold "
                    + (vendor ? "\"" + hw.vendor + "\"" : std::to_string(x));
            return false;
        }
    }
    return true;
}

/// The features of kIsaNames hw has.
std::vector<std::string> hw_isa(const native_hw_t &hw) {
    std::vector<std::string> isa;
    for (const char *f : kIsaNames) {
        if (hw_has_isa(hw, f)) isa.push_back(f);
    }
    return isa;
}

/// hw's value of every hardware key, "?" where it reports none.
std::string hardware_text(const native_hw_t &hw) {
    std::string s;
    for (int k = 0; k < kHwKeyCount; ++k) {
        if (k > 0) s += ' ';
        s += kHwKeys[k];
        s += '=';
        if (k == hk_vendor) {
            s += hw.vendor.empty() ? "?" : "\"" + hw.vendor + "\"";
        } else if (k == hk_isa) {
            std::string isa;
            for (const std::string &f : hw_isa(hw)) {
                if (!isa.empty()) isa += '+';
                isa += f;
            }
            s += isa.empty() ? "none" : isa;
        } else {
            const int64_t x = hw_field(hw, k);
            s += x < 0 ? "?" : std::to_string(x);
        }
    }
    return s;
}

// ---------------------------------------------------------------------------
// Tuning file: table

class table_source_t final : public native_config_source_t {
public:
    const native_config_entry_t *lookup(
            const native_problem_t &p) const noexcept override {
        const std::vector<uint32_t> *bucket = nullptr;
        const int d = type_index(kDtypeTypes, p.wei_dt);
        if (d >= 0 && !indexed_.empty()) {
            const auto it = indexed_.find(index_key(d, p.N, p.K));
            if (it != indexed_.end()) bucket = &it->second;
        }
        // Both lists hold positions in file order: walk them merged.
        const size_t na = bucket != nullptr ? bucket->size() : 0;
        size_t a = 0;
        size_t b = 0;
        while (a < na || b < others_.size()) {
            const bool from_bucket = b == others_.size()
                    || (a < na && (*bucket)[a] < others_[b]);
            const uint32_t at = from_bucket ? (*bucket)[a++] : others_[b++];
            const matched_entry_t &e = *entries_[at];
            if (e.match.holds(p)) return &e.entry;
        }
        return nullptr;
    }

    /// Entries with one dtype and an exact n and k are indexed by them.
    void add(std::unique_ptr<matched_entry_t> e) {
        const problem_match_t &m = e->match;
        const auto exact = [](const std::vector<range_t> &v, int64_t &x) {
            if (v.size() != 1 || v[0].lo != v[0].hi || v[0].lo < 0
                    || v[0].lo > std::numeric_limits<int>::max())
                return false;
            x = v[0].lo;
            return true;
        };
        const uint32_t at = static_cast<uint32_t>(entries_.size());
        int64_t n = 0;
        int64_t k = 0;
        // The one dtype of a mask of a single name: its index.
        const uint8_t d = m.named[0];
        const int dtype = d == 1 ? 0 : (d == 2 ? 1 : (d == 4 ? 2 : -1));
        if (dtype >= 0 && exact(m.numeric[1], n) && exact(m.numeric[2], k))
            indexed_[index_key(dtype, n, k)].push_back(at);
        else
            others_.push_back(at);
        entries_.push_back(std::move(e));
    }

    size_t size() const { return entries_.size(); }

private:
    /// n and k below 2^31, dtype below 4.
    static uint64_t index_key(int dtype, int64_t n, int64_t k) {
        return (static_cast<uint64_t>(n) << 33)
                | (static_cast<uint64_t>(k) << 2)
                | static_cast<uint64_t>(dtype);
    }

    std::vector<std::unique_ptr<matched_entry_t>> entries_;
    std::unordered_map<uint64_t, std::vector<uint32_t>> indexed_;
    std::vector<uint32_t> others_;
};

// ---------------------------------------------------------------------------
// Tuning file: rules

struct tree_node_t {
    int feature = -1; ///< -1: a leaf
    double le = 0.0;
    uint32_t then_node = 0;
    uint32_t else_node = 0;
    const native_config_entry_t *leaf = nullptr; ///< nullptr: no opinion
};

struct condition_t {
    int feature = 0;
    int op = 0;
    double value = 0.0;
};

struct ordered_rule_t {
    uint32_t first = 0; ///< conditions first .. first + count
    uint32_t count = 0;
    const native_config_entry_t *leaf = nullptr; ///< nullptr: no opinion
};

bool condition_holds(const condition_t &c, const double *f) {
    const double x = f[c.feature];
    switch (c.op) {
        case 0: return x < c.value;
        case 1: return x <= c.value;
        case 2: return x > c.value;
        case 3: return x >= c.value;
        case 4: return x == c.value;
        default: return x != c.value;
    }
}

/// The hardware facts the features read.
struct feature_hw_t {
    double l2 = 0.0;
    double l3 = 0.0;
    double l3_threads = 0.0;
    double smt = 0.0;
    double cores = 0.0;
};

void problem_features(
        const native_problem_t &p, const feature_hw_t &hw, double *f) {
    const double esz = p.f32() ? 4.0 : (p.bf16() ? 2.0 : 1.0);
    const double m = p.M, n = p.N, k = p.K;
    const double threads = p.threads, pack_nr = p.pack_nr;
    const int dst = type_index(kDstTypes, p.dst_dt);
    f[f_m] = m;
    f[f_n] = n;
    f[f_k] = k;
    f[f_threads] = threads;
    f[f_pack_nr] = pack_nr;
    f[f_bf16] = p.bf16() ? 1.0 : 0.0;
    f[f_dynamic] = p.dynamic_quant ? 1.0 : 0.0;
    f[f_trans_a] = p.trans_a ? 1.0 : 0.0;
    f[f_has_bias] = p.has_bias ? 1.0 : 0.0;
    f[f_epilogue] = static_cast<double>(p.epilogue);
    f[f_dst_bytes] = dst >= 0 ? kDstBytes[dst] : 0.0;
    f[f_esz] = esz;
    f[f_a_bytes] = m * k * esz;
    f[f_b_bytes] = n * k * esz;
    f[f_col_tiles] = pack_nr != 0.0 ? n / pack_nr : 0.0;
    f[f_m_per_thread] = threads != 0.0 ? m / threads : 0.0;
    f[f_n_per_thread] = threads != 0.0 ? n / threads : 0.0;
    f[f_flops_per_thread] = threads != 0.0 ? 2.0 * m * n * k / threads : 0.0;
    f[f_l2] = hw.l2;
    f[f_l3] = hw.l3;
    f[f_l3_threads] = hw.l3_threads;
    f[f_smt] = hw.smt;
    f[f_cores] = hw.cores;
    f[f_a_l3] = hw.l3 != 0.0 ? f[f_a_bytes] / hw.l3 : 0.0;
    f[f_b_l3] = hw.l3 != 0.0 ? f[f_b_bytes] / hw.l3 : 0.0;
    f[f_slab_l2] = hw.l2 != 0.0 ? pack_nr * k * esz / hw.l2 : 0.0;
}

class rules_source_t final : public native_config_source_t {
public:
    explicit rules_source_t(const native_hw_t &hw) {
        hw_.l2 = hw.uarch.l2_bytes;
        hw_.l3 = hw.uarch.l3_detected ? hw.uarch.l3_domain_bytes : 0;
        hw_.l3_threads = hw.l3_domain_threads;
        hw_.smt = hw.smt;
        hw_.cores = hw.physical_cores;
    }

    const native_config_entry_t *lookup(
            const native_problem_t &p) const noexcept override {
        double f[kFeatureCount];
        problem_features(p, hw_, f);
        if (tree) {
            uint32_t i = 0;
            while (nodes[i].feature >= 0) {
                const tree_node_t &n = nodes[i];
                i = f[n.feature] <= n.le ? n.then_node : n.else_node;
            }
            return nodes[i].leaf;
        }
        for (const ordered_rule_t &r : rules) {
            bool all = true;
            for (uint32_t c = 0; all && c < r.count; ++c)
                all = condition_holds(conditions[r.first + c], f);
            if (all) return r.leaf;
        }
        return nullptr;
    }

    bool tree = true;
    std::vector<tree_node_t> nodes; ///< nodes[0] is the root
    std::vector<condition_t> conditions;
    std::vector<ordered_rule_t> rules;
    std::vector<std::unique_ptr<native_config_entry_t>> leaves;

private:
    feature_hw_t hw_;
};

class rules_parser_t {
public:
    rules_parser_t(rules_source_t &out, const std::string &label)
        : out_(out), label_(label) {}

    /// The index of node n of a tree, or -1 with why set.
    int64_t node(const json &n, int depth) {
        if (depth > kMaxTreeDepth) {
            why = "the tree is deeper than " + std::to_string(kMaxTreeDepth);
            return -1;
        }
        if (out_.nodes.size() >= kMaxTreeNodes) {
            why = "the tree has more than " + std::to_string(kMaxTreeNodes)
                    + " nodes";
            return -1;
        }
        if (!n.is_object()) {
            why = "a tree node is not an object";
            return -1;
        }
        const size_t self = out_.nodes.size();
        out_.nodes.emplace_back();
        const auto feature = n.find("feature");
        if (feature == n.end()) {
            const auto config = n.find("config");
            const auto id = n.find("id");
            const size_t keys = id != n.end() ? 2 : 1;
            if (config == n.end() || n.size() != keys
                    || (id != n.end() && !id->is_string())) {
                why = "a leaf is not {\"config\"} with an optional \"id\"";
                return -1;
            }
            out_.nodes[self].leaf = leaf(*config,
                    id != n.end() ? id->get<std::string>() : std::string(),
                    self);
            return static_cast<int64_t>(self);
        }
        const int f = name_index(kFeatureNames, *feature);
        if (f < 0) {
            why = "unknown feature " + value_text(*feature);
            return -1;
        }
        const auto le = n.find("le");
        const auto then_node = n.find("then");
        const auto else_node = n.find("else");
        if (n.size() != 4 || le == n.end() || !le->is_number()
                || then_node == n.end() || else_node == n.end()) {
            why = "an inner node is not {\"feature\", \"le\", \"then\", "
                  "\"else\"}";
            return -1;
        }
        const int64_t t = node(*then_node, depth + 1);
        if (t < 0) return -1;
        const int64_t e = node(*else_node, depth + 1);
        if (e < 0) return -1;
        tree_node_t &x = out_.nodes[self];
        x.feature = f;
        x.le = le->get<double>();
        x.then_node = static_cast<uint32_t>(t);
        x.else_node = static_cast<uint32_t>(e);
        return static_cast<int64_t>(self);
    }

    bool ordered(const json &list) {
        if (!list.is_array()) {
            why = "the ordered list is not an array";
            return false;
        }
        if (list.size() > kMaxOrderedRules) {
            why = "more than " + std::to_string(kMaxOrderedRules)
                    + " ordered rules";
            return false;
        }
        for (size_t i = 0; i < list.size(); ++i) {
            const json &r = list[i];
            const std::string where = "rule " + std::to_string(i);
            if (!r.is_object()) {
                why = where + " is not an object";
                return false;
            }
            const auto when = r.find("when");
            const auto config = r.find("config");
            const auto id = r.find("id");
            const size_t keys = (when != r.end() ? 1 : 0)
                    + (config != r.end() ? 1 : 0) + (id != r.end() ? 1 : 0);
            if (config == r.end() || keys != r.size()
                    || (id != r.end() && !id->is_string())) {
                why = where + " is not {\"when\", \"config\"} with an "
                              "optional \"id\"";
                return false;
            }
            ordered_rule_t rule;
            rule.first = static_cast<uint32_t>(out_.conditions.size());
            if (when != r.end()) {
                if (!when->is_array() || when->size() > kMaxConditions) {
                    why = where + " has no array of at most "
                            + std::to_string(kMaxConditions) + " conditions";
                    return false;
                }
                for (const json &c : *when) {
                    const bool shaped = c.is_array() && c.size() == 3;
                    const int f = shaped ? name_index(kFeatureNames, c[0]) : -1;
                    const int op = shaped ? name_index(kOps, c[1]) : -1;
                    if (!shaped || f < 0 || op < 0 || !c[2].is_number()) {
                        why = where + " has a bad condition " + value_text(c);
                        return false;
                    }
                    out_.conditions.push_back({f, op, c[2].get<double>()});
                }
            }
            rule.count = static_cast<uint32_t>(
                    out_.conditions.size() - rule.first);
            rule.leaf = leaf(*config,
                    id != r.end() ? id->get<std::string>() : std::string(), i);
            out_.rules.push_back(rule);
        }
        return true;
    }

    std::string why;
    size_t emptied = 0; ///< leaves whose config was bad

private:
    /// The entry of a leaf's config; nullptr for an empty or a bad config
    /// (emptied; the first kMaxEntryWarnings warn).
    const native_config_entry_t *leaf(
            const json &config, const std::string &id, size_t position) {
        std::string name = label_
                + (out_.tree ? " rules leaf " : " rules rule ")
                + std::to_string(position) + id_text(id);
        native_config_t cfg;
        std::string bad;
        if (!json_config(config, cfg, bad)) {
            if (emptied++ < kMaxEntryWarnings)
                apilog_warning("native BRGEMM: ", name, " emptied: ", bad);
            return nullptr;
        }
        if (cfg.empty()) return nullptr;
        auto e = std::make_unique<native_config_entry_t>();
        e->config = cfg;
        e->label = std::move(name);
        out_.leaves.push_back(std::move(e));
        return out_.leaves.back().get();
    }

    rules_source_t &out_;
    const std::string &label_;
};

/// The rules section; nullptr when it is dropped or has no opinion anywhere.
std::unique_ptr<rules_source_t> parse_rules(const json &r,
        const native_hw_t &hw, const std::string &label, std::string &kind) {
    auto out = std::make_unique<rules_source_t>(hw);
    rules_parser_t parser(*out, label);
    bool ok = false;
    const auto k = r.find("kind");
    if (!r.is_object() || k == r.end() || !k->is_string()) {
        parser.why = "rules are not an object with a kind";
    } else if (k->get_ref<const std::string &>() == "tree") {
        const auto root = r.find("root");
        out->tree = true;
        if (root == r.end() || r.size() != 2)
            parser.why = "tree rules are not {\"kind\", \"root\"}";
        else
            ok = parser.node(*root, 1) >= 0;
    } else if (k->get_ref<const std::string &>() == "ordered") {
        const auto list = r.find("list");
        out->tree = false;
        if (list == r.end() || r.size() != 2)
            parser.why = "ordered rules are not {\"kind\", \"list\"}";
        else
            ok = parser.ordered(*list);
    } else {
        parser.why = "rules kind " + value_text(*k)
                + " is neither tree nor ordered";
    }
    if (parser.emptied > kMaxEntryWarnings)
        apilog_warning("native BRGEMM: tuning file ", label, ": ",
                parser.emptied - kMaxEntryWarnings,
                " more rule leaves emptied");
    if (!ok) {
        apilog_warning("native BRGEMM: tuning file ", label,
                " rules dropped: ", parser.why);
        return nullptr;
    }
    kind = out->tree ? "tree" : "ordered";
    if (out->leaves.empty()) return nullptr;
    return out;
}

const char *const kTopKeys[]
        = {"format", "version", "hardware", "meta", "entries", "rules"};

native_tuning_sources_t parse_tuning(std::string_view text,
        const native_hw_t &hw, const std::string &label) {
    native_tuning_sources_t out;
    const auto ignore = [&label](const std::string &why) {
        apilog_warning("native BRGEMM: tuning file ", label, " ignored: ", why);
    };
    if (text.size() > kMaxFileBytes) {
        ignore("it is larger than 64 MiB");
        return out;
    }
    const json_shape_t shape = json_shape(text);
    if (shape.depth > kMaxJsonNesting) {
        ignore("it nests arrays and objects deeper than "
                + std::to_string(kMaxJsonNesting) + " levels");
        return out;
    }
    if (shape.values > kMaxJsonValues) {
        ignore("it has more than " + std::to_string(kMaxJsonValues)
                + " values");
        return out;
    }
    const json doc = json::parse(text.begin(), text.end(), nullptr, false);
    if (doc.is_discarded() || !doc.is_object()) {
        ignore("it is not a JSON object");
        return out;
    }
    const auto format = doc.find("format");
    if (format == doc.end() || !format->is_string()
            || format->get_ref<const std::string &>() != kTuningFormat) {
        ignore(std::string("its format is not ") + kTuningFormat);
        return out;
    }
    const auto version = doc.find("version");
    int64_t v = 0;
    if (version == doc.end() || !json_int(*version, v) || v != kTuningVersion) {
        ignore("its version is not " + std::to_string(kTuningVersion)
                + ", the version this build reads");
        return out;
    }
    size_t unknown = 0;
    for (auto it = doc.begin(); it != doc.end(); ++it) {
        if (std::find(std::begin(kTopKeys), std::end(kTopKeys), it.key())
                        == std::end(kTopKeys)
                && unknown++ < kMaxEntryWarnings)
            apilog_warning("native BRGEMM: tuning file ", label,
                    ": unknown key '", name_text(it.key()), "' ignored");
    }
    if (unknown > kMaxEntryWarnings)
        apilog_warning("native BRGEMM: tuning file ", label, ": ",
                unknown - kMaxEntryWarnings, " more unknown keys ignored");
    std::string why;
    const auto hardware = doc.find("hardware");
    if (hardware != doc.end()) {
        std::vector<hw_condition_t> conditions;
        if (!parse_hardware(*hardware, conditions, why)) {
            ignore(why);
            return out;
        }
        if (!hardware_holds(conditions, hw, why)) {
            ignore("it does not apply to this CPU: " + why);
            if (apilog_verbose_enabled())
                apilog_verbose("native BRGEMM: tuning file ", label,
                        ": this CPU: ", hardware_text(hw));
            return out;
        }
    }

    auto table = std::make_unique<table_source_t>();
    size_t dropped = 0;
    const auto entries = doc.find("entries");
    if (entries != doc.end() && !entries->is_array()) {
        apilog_warning("native BRGEMM: tuning file ", label,
                ": entries is not an array; no entry read");
    } else if (entries != doc.end()) {
        const size_t n = std::min(entries->size(), kMaxEntries);
        if (entries->size() > kMaxEntries)
            apilog_warning("native BRGEMM: tuning file ", label,
                    ": only the first ", kMaxEntries, " of ", entries->size(),
                    " entries read");
        for (size_t i = 0; i < n; ++i) {
            auto e = std::make_unique<matched_entry_t>();
            std::string id;
            if (!json_entry((*entries)[i], *e, id, why)) {
                if (dropped++ < kMaxEntryWarnings)
                    apilog_warning("native BRGEMM: tuning file ", label,
                            " entry ", i, id_text(id), " dropped: ", why);
                continue;
            }
            e->entry.label
                    = label + " entry " + std::to_string(i) + id_text(id);
            table->add(std::move(e));
        }
        if (dropped > kMaxEntryWarnings)
            apilog_warning("native BRGEMM: tuning file ", label, ": ",
                    dropped - kMaxEntryWarnings, " more entries dropped");
    }

    std::string kind;
    std::unique_ptr<rules_source_t> rules;
    const auto r = doc.find("rules");
    if (r != doc.end()) rules = parse_rules(*r, hw, label, kind);

    if (apilog_info_enabled()) {
        std::string what = "no rules";
        if (rules != nullptr)
            what = kind + " rules of "
                    + std::to_string(rules->tree ? rules->nodes.size()
                                                 : rules->rules.size())
                    + (rules->tree ? " nodes" : " rules");
        apilog_info("native BRGEMM: tuning file ", label, ": ", table->size(),
                " entries kept, ", dropped, " dropped; ", what);
    }
    if (table->size() > 0) out.table = std::move(table);
    out.rules = std::move(rules);
    return out;
}

} // namespace

std::unique_ptr<native_config_source_t> make_native_override_source(
        std::string_view text) {
    auto source = std::make_unique<override_source_t>();
    size_t position = 0;
    size_t extra = 0;
    for_each_piece(text, ';', [&](std::string_view piece) {
        const std::string_view clause = native_text_trim(piece);
        if (clause.empty()) return true;
        if (++position > kMaxClauses) {
            ++extra;
            return true;
        }
        auto c = std::make_unique<matched_entry_t>();
        std::string why;
        if (!parse_clause(clause, *c, why)) {
            apilog_warning("native BRGEMM: ", kConfigVariable, " clause ",
                    position, " '", clause, "' dropped: ", why);
            return true;
        }
        c->entry.label = std::string(kConfigVariable) + " clause "
                + std::to_string(position) + " (" + std::string(clause) + ")";
        source->clauses.push_back(std::move(c));
        return true;
    });
    if (extra > 0)
        apilog_warning("native BRGEMM: ", kConfigVariable, " has more than ",
                kMaxClauses, " clauses; the last ", extra, " dropped");
    if (source->clauses.empty()) return nullptr;
    return source;
}

native_tuning_sources_t load_native_tuning_file(
        const std::string &path, const native_hw_t &hw) {
    try {
        // Opening a FIFO or a device can block the first native call.
        std::error_code ec;
        if (!std::filesystem::is_regular_file(path, ec)) {
            apilog_warning("native BRGEMM: tuning file ", path,
                    " ignored: it cannot be read");
            return {};
        }
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        const std::streamoff size
                = f ? static_cast<std::streamoff>(f.tellg()) : -1;
        if (size >= 0 && static_cast<uint64_t>(size) > kMaxFileBytes) {
            apilog_warning("native BRGEMM: tuning file ", path,
                    " ignored: it is larger than 64 MiB");
            return {};
        }
        std::string text(size > 0 ? static_cast<size_t>(size) : 0, '\0');
        if (size > 0) {
            f.seekg(0, std::ios::beg);
            f.read(&text[0], size);
        }
        if (size < 0 || !f) {
            apilog_warning("native BRGEMM: tuning file ", path,
                    " ignored: it cannot be read");
            return {};
        }
        return parse_tuning(text, hw, path);
    } catch (...) {
        apilog_warning("native BRGEMM: tuning file ", path,
                " ignored: it could not be loaded");
        return {};
    }
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
