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

/// @file prompt_tile_tunables.hpp
/// @brief Tunables of the ALGO 3 prompt tile executor, and the CK facts of
///        each dtype family it plans for.
///
/// A call's knobs resolve in this order, later steps winning: the knob's
/// default (`KnobDef::def`), its env var (an invalid or out-of-range value
/// is ignored), the first matching env profile entry, then gtest overrides.
/// The executor then fits `mb_min` / `mb_max` to the call's row alignment
/// (`sanitize`).  Env vars and profiles are read once.
///
/// Knobs, as `ZENDNNL_GRP_MATMUL_PROMPT_TILE[_NAME]` (default):
///   (none)     family bitmask: 1 BF16, 2 INT8, 4 W4A8; 0 = off (1)
///   TEAM       0 = teams from sysfs L3 domains, n = contiguous teams of n (0)
///   SUBTEAM    max threads per sub-team of an L3 domain, 0 = whole (4)
///   MB         forced M-block rows, rounded up to the row alignment (0 = auto)
///   NS         forced N-slice columns, a multiple of the pack NR (0 = auto)
///   NS_L2_PCT  % of L2 an auto N-slice's B fills (100)
///   MB_L2_PCT  % of L2 an auto M-block's A fills (50)
///   MB_MIN     smallest auto M-block (16)
///   MB_MAX     largest auto M-block (512)
///   RB         B-stream cost of a tile, in rows (32)
///   GRAB       tiles per queue grab (1)
///   ORDER      0 N-slice major, 1 M-block major, 2 per expert by A vs B (2)
///   STEAL      1 drains other teams' queues when idle (1)
///   MIN_TILES  tiles per thread before auto sizes shrink (8)
///   L3SB       1 splits heavy experts into L3-resident superblocks (1)
///   SB_L3_PCT  % of a team's L3 a superblock's A fills (50)
///   BM_L3_PCT  % of a team's L3 an expert's B may fill and still be walked
///              M-block major (50)
///   SPAN_PCT   % of one team's work above which an expert is walked M-block
///              major (50)
/// The first three are env-only, so no profile changes whether the executor
/// engages.
///
/// Profile: `ZENDNNL_GRP_MATMUL_PROMPT_TILE_PROFILE`, inline or `@/path` (one
/// entry per line).  `;`-separated `keys:knobs` entries, e.g.
///   fam=bf16,K=2880,N=5760,M=8192-:mb=64,ns=320; K=2048:rb=48
/// Keys: `fam` (bf16, int8, w4a8), `K` and `N` of the first active expert,
/// `M` (active rows summed), `Mmax` (tallest expert), `T` (threads); each
/// takes `v`, `lo-hi`, `lo-` or `-hi`, and an omitted key matches anything.
/// Knobs use the lower-case names above.  The first matching entry wins; at
/// most 64 entries; any error rejects the whole profile with one warning.
///
/// Recipes: expert parallel `mb=65536,ns=1048576`; M only `ns=1048576`;
/// N only `mb=65536`; fully dynamic `TEAM=<threads>` (one shared queue);
/// static only `steal=0`.

#ifndef ZENDNNL_GROUP_MATMUL_PROMPT_TILE_TUNABLES_HPP
#define ZENDNNL_GROUP_MATMUL_PROMPT_TILE_TUNABLES_HPP

#include <array>
#include <atomic>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace prompt_tile {

enum class Family : int { kBf16, kInt8, kW4a8 };
inline constexpr int kNumFamilies = 3;

/// What the planner needs to know about a family's CK microkernel.
struct FamilyFacts {
    const char *name;
    int a_bytes; ///< Per A element (INT8 and W4A8 read the hoisted s8 / u8 A).
    int wei_bits; ///< Per packed weight element.
    /// Fewest rows an MR call may take unless its tile spans the expert's M.
    /// The CK splits `rows` into `ceil(rows / max_mr)` near-equal calls, and
    /// BF16 sums K on two accumulator sets at MR <= 3, a different fp32
    /// order.  INT8 (int32) and W4A8 (fixed group order) keep bits at any MR.
    int min_mr;
    int row_align; ///< M-block row alignment; 0 = the call's CK max MR.
    bool kblock; ///< The CK may K-block; the executor stands down if so.
};

inline constexpr FamilyFacts kFamilies[kNumFamilies] = {
        {"bf16", 2, 16, 4, 8, true},
        {"int8", 1, 8, 1, 0, false},
        {"w4a8", 1, 4, 1, 0, false},
};

constexpr const FamilyFacts &facts(Family f) {
    return kFamilies[static_cast<int>(f)];
}

/// Knob ids, in `kKnobs` order.
enum Knob : int {
    kEnable,
    kTeam,
    kSubteam,
    kMb,
    kNs,
    kNsL2Pct,
    kMbL2Pct,
    kMbMin,
    kMbMax,
    kRb,
    kGrab,
    kOrder,
    kSteal,
    kMinTiles,
    kL3sb,
    kSbL3Pct,
    kBmL3Pct,
    kSpanPct,
    kNumKnobs
};

/// `kGlobal` knobs come from the env only; profile entries may set `kShape`.
enum class Scope : int { kGlobal, kShape };

struct KnobDef {
    Knob id;
    const char *name; ///< Profile and log name; upper-cased, the env suffix.
    int lo, hi;
    int def;
    Scope scope;
};

inline constexpr KnobDef kKnobs[kNumKnobs] = {
        {kEnable, "enable", 0, 7, 1, Scope::kGlobal},
        {kTeam, "team", 0, 4096, 0, Scope::kGlobal},
        {kSubteam, "subteam", 0, 4096, 4, Scope::kGlobal},
        {kMb, "mb", 0, 1 << 16, 0, Scope::kShape},
        {kNs, "ns", 0, 1 << 20, 0, Scope::kShape},
        {kNsL2Pct, "ns_l2_pct", 1, 400, 100, Scope::kShape},
        {kMbL2Pct, "mb_l2_pct", 1, 400, 50, Scope::kShape},
        {kMbMin, "mb_min", 1, 1 << 16, 16, Scope::kShape},
        {kMbMax, "mb_max", 1, 1 << 16, 512, Scope::kShape},
        {kRb, "rb", 0, 4096, 32, Scope::kShape},
        {kGrab, "grab", 1, 64, 1, Scope::kShape},
        {kOrder, "order", 0, 2, 2, Scope::kShape},
        {kSteal, "steal", 0, 1, 1, Scope::kShape},
        {kMinTiles, "min_tiles", 1, 4096, 8, Scope::kShape},
        {kL3sb, "l3sb", 0, 1, 1, Scope::kShape},
        {kSbL3Pct, "sb_l3_pct", 1, 400, 50, Scope::kShape},
        {kBmL3Pct, "bm_l3_pct", 1, 400, 50, Scope::kShape},
        {kSpanPct, "span_pct", 1, 1000, 50, Scope::kShape},
};

constexpr bool knob_rows_follow_ids() {
    for (int i = 0; i < kNumKnobs; ++i)
        if (kKnobs[i].id != i) return false;
    return true;
}
static_assert(knob_rows_follow_ids(), "kKnobs rows must follow the Knob ids");
static_assert(kNumKnobs <= 32, "knob sets are 32-bit masks");

/// Knob values, indexed by `Knob`.
struct Tunables {
    std::array<int, kNumKnobs> v;
    constexpr int operator[](Knob k) const { return v[k]; }
    constexpr int &operator[](Knob k) { return v[k]; }
};

constexpr Tunables defaults() {
    Tunables t {};
    for (int i = 0; i < kNumKnobs; ++i)
        t.v[i] = kKnobs[i].def;
    return t;
}

/// What profile entries match on.
struct ShapeKey {
    Family fam = Family::kBf16;
    int64_t K = 0; ///< K and N of the first active expert.
    int64_t N = 0;
    int64_t M = 0; ///< Active rows, summed.
    int64_t Mmax = 0; ///< Rows of the tallest expert.
    int64_t T = 0; ///< Threads.
};

/// Inclusive ranges on the key, and the knobs the entry sets.  The knob env
/// vars form an entry that matches every call.
struct ProfileEntry {
    static constexpr int64_t kAny = std::numeric_limits<int64_t>::max();
    int fam = -1; ///< -1 matches every family.
    int64_t lo[5] = {0, 0, 0, 0, 0}; ///< K, N, M, Mmax, T.
    int64_t hi[5] = {kAny, kAny, kAny, kAny, kAny};
    uint32_t set = 0; ///< Bit `k`: the entry sets knob `k`.
    Tunables val {};
};
using Profile = std::vector<ProfileEntry>;

/// A call's knobs, and the env profile entry it matched (-1 = none).
struct Resolved {
    Tunables t {};
    int env_entry = -1;
};

std::string env_name(Knob k);

/// False unless `text` is an integer in the knob's range.
bool parse_knob_value(Knob k, const char *text, int &out);

/// On error returns false, leaves `out` empty and names the bad entry in
/// `*err`.
bool parse_profile(
        const std::string &text, Profile &out, std::string *err = nullptr);

/// `parse_profile` of `value`, or of the file named after a leading '@'.
bool load_profile(const char *value, Profile &out, std::string *err = nullptr);

/// Index of the first entry of `p` that matches `key`, or -1.
int match_profile(const Profile &p, const ShapeKey &key);

/// The precedence up to the env profile, over explicit inputs.
Resolved resolve_with(const ShapeKey &key, const ProfileEntry &env,
        const Profile &env_profile);

/// The whole precedence, over this process's env and profiles.
Resolved resolve(const ShapeKey &key);

/// Value of a `Scope::kGlobal` knob.
int global_value(Knob k);

inline bool enabled(Family f) {
    return ((global_value(kEnable) >> static_cast<int>(f)) & 1) != 0;
}

/// Rounds `mb_min` up and `mb_max` down to `row_align`, keeping
/// `min_mr <= mb_min <= mb_max`.
void sanitize(Tunables &t, int row_align, int min_mr);

namespace test_api {
/// Per-knob gtest override, stored as value + 1 (0 = none).
inline std::atomic<int> s_override[kNumKnobs] {};
/// Calls served by the prompt tile executor (engagement probe for tests).
inline std::atomic<long> s_calls_served {0};

class ScopedOverride {
public:
    ScopedOverride(Knob k, int v)
        : k_(k)
        , prev_(s_override[k].exchange(v + 1, std::memory_order_relaxed)) {}
    ~ScopedOverride() {
        s_override[k_].store(prev_, std::memory_order_relaxed);
    }
    ScopedOverride(const ScopedOverride &) = delete;
    ScopedOverride &operator=(const ScopedOverride &) = delete;

private:
    Knob k_;
    int prev_;
};
} // namespace test_api

} // namespace prompt_tile
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_PROMPT_TILE_TUNABLES_HPP
