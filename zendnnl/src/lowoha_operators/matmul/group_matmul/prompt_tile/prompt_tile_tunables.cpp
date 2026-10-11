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

#include "prompt_tile_tunables.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iterator>

#include "../group_matmul_parallel_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace prompt_tile {

namespace {

constexpr int kMaxProfileEntries = 64;
/// Range keys, in `ProfileEntry::lo` order; `fam` is key 5.
constexpr const char *kRangeKeys[5] = {"K", "N", "M", "Mmax", "T"};
constexpr int kFamKey = 5;

std::string trim(const std::string &s) {
    const size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    return s.substr(b, s.find_last_not_of(" \t\r\n") - b + 1);
}

std::vector<std::string> split(const std::string &s, char sep) {
    std::vector<std::string> out;
    for (size_t b = 0;;) {
        const size_t e = s.find(sep, b);
        out.push_back(trim(s.substr(b, e == std::string::npos ? e : e - b)));
        if (e == std::string::npos) return out;
        b = e + 1;
    }
}

/// `name=value` into its trimmed halves; false without an '='.
bool split_pair(const std::string &kv, std::string &name, std::string &value) {
    const size_t eq = kv.find('=');
    if (eq == std::string::npos) return false;
    name = trim(kv.substr(0, eq));
    value = trim(kv.substr(eq + 1));
    return true;
}

bool parse_count(const std::string &s, int64_t &out) {
    if (s.empty() || s.size() > 18
            || s.find_first_not_of("0123456789") != std::string::npos)
        return false;
    out = 0;
    for (const char c : s)
        out = out * 10 + (c - '0');
    return true;
}

/// `v`, `lo-hi`, `lo-` or `-hi`.
bool parse_range(const std::string &s, int64_t &lo, int64_t &hi) {
    const size_t dash = s.find('-');
    if (dash == std::string::npos) {
        if (!parse_count(s, lo)) return false;
        hi = lo;
        return true;
    }
    const std::string a = trim(s.substr(0, dash));
    const std::string b = trim(s.substr(dash + 1));
    lo = 0;
    hi = ProfileEntry::kAny;
    return !(a.empty() && b.empty()) && (a.empty() || parse_count(a, lo))
            && (b.empty() || parse_count(b, hi)) && lo <= hi;
}

bool parse_keys(const std::string &text, ProfileEntry &e) {
    uint32_t seen = 0;
    for (const std::string &kv : split(text, ',')) {
        std::string name, value;
        if (!split_pair(kv, name, value)) return false;
        int id = name == "fam" ? kFamKey : -1;
        for (int d = 0; d < 5; ++d)
            if (name == kRangeKeys[d]) id = d;
        if (id < 0 || ((seen >> id) & 1u)) return false;
        seen |= 1u << id;
        if (id != kFamKey) {
            if (!parse_range(value, e.lo[id], e.hi[id])) return false;
            continue;
        }
        for (int f = 0; f < kNumFamilies; ++f)
            if (value == kFamilies[f].name) e.fam = f;
        if (e.fam < 0) return false;
    }
    return true;
}

bool parse_knobs(const std::string &text, ProfileEntry &e) {
    for (const std::string &kv : split(text, ',')) {
        std::string name, value;
        if (!split_pair(kv, name, value)) return false;
        int id = -1, v = 0;
        for (int k = 0; k < kNumKnobs; ++k)
            if (name == kKnobs[k].name) id = k;
        if (id < 0 || ((e.set >> id) & 1u) || kKnobs[id].scope != Scope::kShape
                || !parse_knob_value(static_cast<Knob>(id), value.c_str(), v))
            return false;
        e.set |= 1u << id;
        e.val.v[id] = v;
    }
    return true;
}

void apply(Tunables &t, const ProfileEntry &e) {
    for (int k = 0; k < kNumKnobs; ++k)
        if ((e.set >> k) & 1u) t.v[k] = e.val.v[k];
}

struct Inputs {
    ProfileEntry env;
    Profile env_profile;
};

const Inputs &inputs() {
    static const Inputs in = [] {
        Inputs r;
        std::string err;
        for (int k = 0; k < kNumKnobs; ++k) {
            const Knob id = static_cast<Knob>(k);
            int v = 0;
            if (parse_knob_value(id, std::getenv(env_name(id).c_str()), v)) {
                r.env.set |= 1u << k;
                r.env.val.v[k] = v;
            }
        }
        const char *p = std::getenv("ZENDNNL_GRP_MATMUL_PROMPT_TILE_PROFILE");
        if (p != nullptr && !load_profile(p, r.env_profile, &err))
            apilog_warning(
                    "[GRP_MATMUL.PTILE] "
                    "ZENDNNL_GRP_MATMUL_PROMPT_TILE_PROFILE ignored: ",
                    err);
        return r;
    }();
    return in;
}

int override_of(Knob k) {
    return test_api::s_override[k].load(std::memory_order_relaxed);
}

} // namespace

std::string env_name(Knob k) {
    std::string s = "ZENDNNL_GRP_MATMUL_PROMPT_TILE";
    if (k == kEnable) return s;
    s += '_';
    for (const char *c = kKnobs[k].name; *c != '\0'; ++c)
        s += static_cast<char>(std::toupper(static_cast<unsigned char>(*c)));
    return s;
}

bool parse_knob_value(Knob k, const char *text, int &out) {
    int v = 0;
    if (!parse_env_int_strict(text, v) || v < kKnobs[k].lo || v > kKnobs[k].hi)
        return false;
    out = v;
    return true;
}

bool parse_profile(const std::string &text, Profile &out, std::string *err) {
    out.clear();
    Profile p;
    for (const std::string &entry : split(text, ';')) {
        if (entry.empty()) continue;
        ProfileEntry e;
        const size_t colon = entry.find(':');
        const bool full = static_cast<int>(p.size()) == kMaxProfileEntries;
        if (!full && colon != std::string::npos
                && (colon == 0 || parse_keys(entry.substr(0, colon), e))
                && parse_knobs(entry.substr(colon + 1), e)) {
            p.push_back(e);
            continue;
        }
        if (err != nullptr)
            *err = full ? "more than 64 entries" : "bad entry '" + entry + "'";
        return false;
    }
    out = std::move(p);
    return true;
}

bool load_profile(const char *value, Profile &out, std::string *err) {
    if (value == nullptr || value[0] != '@')
        return parse_profile(value != nullptr ? value : "", out, err);
    std::ifstream f(value + 1);
    if (!f) {
        out.clear();
        if (err != nullptr) *err = std::string("cannot read ") + (value + 1);
        return false;
    }
    std::string s {std::istreambuf_iterator<char>(f),
            std::istreambuf_iterator<char>()};
    std::replace(s.begin(), s.end(), '\n', ';');
    return parse_profile(s, out, err);
}

int match_profile(const Profile &p, const ShapeKey &key) {
    const int64_t v[5] = {key.K, key.N, key.M, key.Mmax, key.T};
    for (size_t i = 0; i < p.size(); ++i) {
        const ProfileEntry &e = p[i];
        if (e.fam >= 0 && e.fam != static_cast<int>(key.fam)) continue;
        int d = 0;
        while (d < 5 && v[d] >= e.lo[d] && v[d] <= e.hi[d])
            ++d;
        if (d == 5) return static_cast<int>(i);
    }
    return -1;
}

Resolved resolve_with(const ShapeKey &key, const ProfileEntry &env,
        const Profile &env_profile) {
    Resolved r;
    r.t = defaults();
    apply(r.t, env);
    r.env_entry = match_profile(env_profile, key);
    if (r.env_entry >= 0) apply(r.t, env_profile[r.env_entry]);
    return r;
}

Resolved resolve(const ShapeKey &key) {
    const Inputs &in = inputs();
    Resolved r = resolve_with(key, in.env, in.env_profile);
    for (int k = 0; k < kNumKnobs; ++k) {
        const int o = override_of(static_cast<Knob>(k));
        if (o != 0) r.t.v[k] = o - 1;
    }
    for (const Knob k : {kGrab, kMinTiles, kNsL2Pct, kMbL2Pct})
        r.t[k] = std::max(r.t[k], 1);
    return r;
}

int global_value(Knob k) {
    const int o = override_of(k);
    if (o != 0) return o - 1;
    const ProfileEntry &env = inputs().env;
    return ((env.set >> k) & 1u) ? env.val[k] : kKnobs[k].def;
}

void sanitize(Tunables &t, int row_align, int min_mr) {
    const int a = std::max(row_align, 1);
    const int lo = (std::max(t[kMbMin], min_mr) + a - 1) / a * a;
    t[kMbMin] = lo;
    t[kMbMax] = std::max(t[kMbMax] / a * a, lo);
}

} // namespace prompt_tile
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
