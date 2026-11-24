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

#include "ntile_flat_parallel_config.hpp"

#include <algorithm>
#include <atomic>
#include <bitset>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <utility>

#include "lowoha_operators/matmul/group_matmul/common/grp_matmul_machine.hpp"
#include "lowoha_operators/matmul/lowoha_common.hpp"
#include "nr32_format.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ntile_flat_parallel {

namespace {

constexpr const char *kEnv = "ZENDNNL_GRP_MATMUL_NTILE_FLAT_PARALLEL_CONFIG";
/// N-tile flat parallel's formats, then every other type.
constexpr int kTypes = nr32::kFormatCount;
constexpr int kPhases = 2;
constexpr int kApis = 2;

int type_index(const data_type_t wei) {
    return nr32::format_index(wei);
}

/// The index of the format named `v`, or -1.
int format_named(const std::string &v) {
    for (int i = 0; i < nr32::kFormatCount; ++i)
        if (v == nr32::kFormats[i].name) return i;
    return -1;
}

enum class setting : uint8_t {
    units,
    prompt_units,
    sweep_rows,
    unit_order,
    heavy_light,
    l2_frac,
    subtile_rows,
    u8_src,
    gather_rows,
    wait_spins,
    scratch_keep_mib,
    call_sites,
    warmed_pools,
    count
};
enum class scope : uint8_t { call, plan, limit };

/// How a setting's value reads: an integer or a fraction in [lo, hi], or a
/// choice among `choices` (separated by '|'; the index is the value).
struct setting_info {
    const char *name;
    setting id;
    scope where;
    double lo;
    double hi;
    bool integer;
    const char *choices;
};

const setting_info kSettings[] = {
        {"units", setting::units, scope::call, 1, 256, true, nullptr},
        {"prompt_units", setting::prompt_units, scope::call, 1, 256, true,
                nullptr},
        {"sweep_rows", setting::sweep_rows, scope::call, 1, 1 << 20, true,
                nullptr},
        {"unit_order", setting::unit_order, scope::call, 0, 0, true,
                "auto|queue|own"},
        {"heavy_light", setting::heavy_light, scope::call, 0, 0, true,
                "off|on"},
        {"l2_frac", setting::l2_frac, scope::plan, 0.01, 1.0, false, nullptr},
        {"subtile_rows", setting::subtile_rows, scope::plan, 1, 1024, true,
                nullptr},
        {"u8_src", setting::u8_src, scope::plan, 0, 0, true, "auto|off|on"},
        {"gather_rows", setting::gather_rows, scope::limit, 1, 4096, true,
                nullptr},
        {"wait_spins", setting::wait_spins, scope::limit, 0, 1 << 24, true,
                nullptr},
        {"scratch_keep_mib", setting::scratch_keep_mib, scope::limit, 0,
                1 << 20, true, nullptr},
        {"call_sites", setting::call_sites, scope::limit, 1, 1 << 20, true,
                nullptr},
        {"warmed_pools", setting::warmed_pools, scope::limit, 1, 1 << 20, true,
                nullptr},
};

const setting_info *find_setting(const std::string &name) {
    for (const setting_info &s : kSettings)
        if (name == s.name) return &s;
    return nullptr;
}

struct clause_t {
    /// -1 for any; else the phase (0 decode, 1 prompt), the type index or the
    /// API.
    int phase = -1;
    int type = -1;
    int api = -1;
    std::vector<std::pair<setting, double>> sets;
};

struct config_t {
    std::string text;
    std::vector<clause_t> clauses;
    std::vector<std::string> warnings;
    call_knobs call[kPhases][kTypes + 1][kApis];
    plan_knobs plan[kTypes + 1][kApis];
    limits_t limits;
};

std::string trim(const std::string &s) {
    const size_t b = s.find_first_not_of(" \t");
    if (b == std::string::npos) return {};
    const size_t e = s.find_last_not_of(" \t");
    return s.substr(b, e - b + 1);
}

std::vector<std::string> split(const std::string &s, const char sep) {
    std::vector<std::string> out;
    size_t start = 0;
    for (;;) {
        const size_t end = s.find(sep, start);
        out.push_back(trim(s.substr(start, end - start)));
        if (end == std::string::npos) return out;
        start = end + 1;
    }
}

/// The index of `v` among `choices`, or -1.
int choice_of(const char *choices, const std::string &v) {
    const std::vector<std::string> names = split(choices, '|');
    for (size_t i = 0; i < names.size(); ++i)
        if (names[i] == v) return static_cast<int>(i);
    return -1;
}

bool parse_value(const setting_info &s, const std::string &v, double &out) {
    if (s.choices != nullptr) {
        const int c = choice_of(s.choices, v);
        if (c < 0) return false;
        out = c;
        return true;
    }
    if (v.empty()) return false;
    errno = 0;
    char *end = nullptr;
    const double d = std::strtod(v.c_str(), &end);
    if (errno != 0 || end == v.c_str() || *end != '\0') return false;
    // In range before any integer conversion: nan, inf and 1e30 are not.
    if (!std::isfinite(d) || d < s.lo || d > s.hi) return false;
    if (s.integer && d != std::trunc(d)) return false;
    out = d;
    return true;
}

/// Reads one clause into `c`, or returns why it is bad.
std::string parse_clause(const std::string &text, clause_t &c) {
    std::string filters, sets = text;
    const size_t colon = text.find(':');
    if (colon != std::string::npos) {
        filters = trim(text.substr(0, colon));
        sets = trim(text.substr(colon + 1));
    }
    if (!filters.empty()) {
        for (const std::string &f : split(filters, ',')) {
            const size_t eq = f.find('=');
            if (eq == std::string::npos) return "filter '" + f + "' has no '='";
            const std::string k = trim(f.substr(0, eq));
            const std::string v = trim(f.substr(eq + 1));
            int *slot = nullptr;
            int value = -1;
            if (k == "phase") {
                slot = &c.phase;
                value = choice_of("decode|prompt", v);
            } else if (k == "dtype") {
                slot = &c.type;
                value = format_named(v);
            } else if (k == "api") {
                slot = &c.api;
                value = choice_of("routed|grouped", v);
            } else {
                return "unknown filter '" + k + "'";
            }
            if (value < 0) return "filter " + k + " takes no '" + v + "'";
            if (*slot >= 0) return "filter " + k + " given twice";
            *slot = value;
        }
    }
    if (sets.empty()) return "no settings";
    std::bitset<static_cast<size_t>(setting::count)> seen;
    for (const std::string &a : split(sets, ',')) {
        const size_t eq = a.find('=');
        if (eq == std::string::npos) return "setting '" + a + "' has no '='";
        const std::string k = trim(a.substr(0, eq));
        const std::string v = trim(a.substr(eq + 1));
        const setting_info *s = find_setting(k);
        if (s == nullptr) return "unknown setting '" + k + "'";
        if (seen[static_cast<size_t>(s->id)]) return k + " given twice";
        seen[static_cast<size_t>(s->id)] = true;
        double value = 0;
        if (!parse_value(*s, v, value)) return k + " takes no '" + v + "'";
        if (s->where == scope::plan && c.phase >= 0)
            return k + " is a plan setting and takes no phase filter";
        if (s->id == setting::u8_src && c.type >= 0
                && !nr32::kFormats[c.type].u8_src) {
            return k + " takes only the dtypes with u8-source kernels (s8)";
        }
        if (s->where == scope::limit
                && (c.phase >= 0 || c.type >= 0 || c.api >= 0)) {
            return k + " is a process limit and takes no filter";
        }
        c.sets.emplace_back(s->id, value);
    }
    return {};
}

bool matches(
        const clause_t &c, const int phase, const int type, const int api) {
    return (c.phase < 0 || c.phase == phase) && (c.type < 0 || c.type == type)
            && (c.api < 0 || c.api == api);
}

void apply_call(call_knobs &k, const setting s, const double v) {
    switch (s) {
        case setting::units: k.units = static_cast<int>(v); break;
        case setting::prompt_units: k.prompt_units = static_cast<int>(v); break;
        case setting::sweep_rows: k.sweep_rows = static_cast<int>(v); break;
        case setting::unit_order:
            k.order = static_cast<unit_order>(static_cast<int>(v));
            break;
        case setting::heavy_light: k.heavy_light = v != 0; break;
        default: break;
    }
}

void apply_plan(plan_knobs &k, const setting s, const double v) {
    switch (s) {
        case setting::l2_frac: k.l2_frac = v; break;
        case setting::subtile_rows: k.subtile_rows = static_cast<int>(v); break;
        case setting::u8_src:
            k.u8_src = static_cast<switch3>(static_cast<int>(v));
            break;
        default: break;
    }
}

void apply_limit(limits_t &k, const setting s, const double v) {
    switch (s) {
        case setting::gather_rows: k.gather_rows = static_cast<int>(v); break;
        case setting::wait_spins: k.wait_spins = static_cast<int>(v); break;
        case setting::scratch_keep_mib:
            k.scratch_keep_bytes = static_cast<size_t>(v) << 20;
            break;
        case setting::call_sites: k.call_sites = static_cast<size_t>(v); break;
        case setting::warmed_pools:
            k.warmed_pools = static_cast<size_t>(v);
            break;
        default: break;
    }
}

scope scope_of(const setting s) {
    for (const setting_info &i : kSettings)
        if (i.id == s) return i.where;
    return scope::call;
}

config_t build(const char *text) {
    config_t cfg;
    cfg.text = text != nullptr ? text : "";
    for (const std::string &raw : split(cfg.text, ';')) {
        if (raw.empty()) continue;
        clause_t c;
        const std::string why = parse_clause(raw, c);
        if (!why.empty()) {
            cfg.warnings.push_back(std::string(kEnv) + ": dropped clause '"
                    + raw + "': " + why);
            continue;
        }
        cfg.clauses.push_back(std::move(c));
    }
    // For each setting, the first clause whose filters hold and that sets it.
    using done_t = std::bitset<static_cast<size_t>(setting::count)>;
    for (int t = 0; t <= kTypes; ++t) {
        for (int a = 0; a < kApis; ++a) {
            for (int p = 0; p < kPhases; ++p) {
                done_t done;
                for (const clause_t &c : cfg.clauses) {
                    if (!matches(c, p, t, a)) continue;
                    for (const auto &[s, v] : c.sets) {
                        if (scope_of(s) != scope::call
                                || done[static_cast<size_t>(s)]) {
                            continue;
                        }
                        done[static_cast<size_t>(s)] = true;
                        apply_call(cfg.call[p][t][a], s, v);
                    }
                }
            }
            done_t done;
            for (const clause_t &c : cfg.clauses) {
                if (c.phase >= 0 || !matches(c, 0, t, a)) continue;
                for (const auto &[s, v] : c.sets) {
                    if (scope_of(s) != scope::plan
                            || done[static_cast<size_t>(s)]) {
                        continue;
                    }
                    done[static_cast<size_t>(s)] = true;
                    apply_plan(cfg.plan[t][a], s, v);
                }
            }
        }
    }
    done_t done;
    for (const clause_t &c : cfg.clauses) {
        if (c.phase >= 0 || c.type >= 0 || c.api >= 0) continue;
        for (const auto &[s, v] : c.sets) {
            if (scope_of(s) != scope::limit || done[static_cast<size_t>(s)])
                continue;
            done[static_cast<size_t>(s)] = true;
            apply_limit(cfg.limits, s, v);
        }
    }
    return cfg;
}

void log_config(const config_t &cfg) {
    if (apilog_info_enabled()) {
        apilog_info("[GRP_MATMUL.NTILE_FLAT_PARALLEL.CONFIG] machine: ",
                grp_machine::describe(
                        grp_machine::machine(), grp_machine::dot_rates()));
        if (!cfg.text.empty()) {
            apilog_info("[GRP_MATMUL.NTILE_FLAT_PARALLEL.CONFIG] ", kEnv, "='",
                    cfg.text, "': ", cfg.clauses.size(), " clause(s)");
        }
    }
    if (!cfg.warnings.empty() && apilog_warning_enabled()) {
        for (const std::string &w : cfg.warnings)
            apilog_warning("[GRP_MATMUL.NTILE_FLAT_PARALLEL.CONFIG] ", w);
    }
}

std::atomic<const config_t *> g_config {nullptr};

std::mutex &config_mutex() {
    static auto *mutex = new std::mutex;
    return *mutex;
}

const config_t &config() {
    if (const config_t *c = g_config.load(std::memory_order_acquire)) return *c;
    std::lock_guard<std::mutex> lock(config_mutex());
    if (const config_t *c = g_config.load(std::memory_order_relaxed)) return *c;
    // Never destroyed: calls may still read it while static objects are
    // destroyed.
    const config_t *c = new config_t(build(std::getenv(kEnv)));
    log_config(*c);
    g_config.store(c, std::memory_order_release);
    return *c;
}

} // namespace

const call_knobs &call_knobs_for(const grp_matmul_phase phase,
        const data_type_t wei, const api_kind api) {
    const int p = phase == grp_matmul_phase::prompt ? 1 : 0;
    return config().call[p][type_index(wei)][static_cast<int>(api)];
}

const plan_knobs &plan_knobs_for(const data_type_t wei, const api_kind api) {
    return config().plan[type_index(wei)][static_cast<int>(api)];
}

const limits_t &limits() {
    return config().limits;
}

int64_t l2_budget_bytes(const data_type_t wei, const api_kind api) {
    const grp_machine::machine_t &m = grp_machine::machine();
    const double frac = plan_knobs_for(wei, api).l2_frac;
    if (frac <= 0) return m.l2_budget_bytes;
    return std::max<int64_t>(static_cast<int64_t>(m.l2_plan_bytes * frac), 1);
}

bool u8_src(const data_type_t wei, const api_kind api) {
    const nr32::format_desc *f = nr32::format_of(wei);
    if (f == nullptr || !f->u8_src) return false;
    const switch3 s = plan_knobs_for(wei, api).u8_src;
    return s == switch3::on
            || (s == switch3::automatic
                    && grp_machine::dot_rates().u8_src_pays);
}

} // namespace ntile_flat_parallel

void test_api::set_ntile_flat_parallel_config(const char *text) {
    using namespace ntile_flat_parallel;
    std::lock_guard<std::mutex> lock(config_mutex());
    // The previous config is kept: a call may still read it.
    const config_t *c
            = new config_t(build(text != nullptr ? text : std::getenv(kEnv)));
    log_config(*c);
    g_config.store(c, std::memory_order_release);
}

std::vector<std::string> test_api::ntile_flat_parallel_config_warnings() {
    return ntile_flat_parallel::config().warnings;
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
