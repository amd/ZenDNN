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

#include "prompt_tile_topology.hpp"

#include <omp.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <tuple>

#if defined(__linux__)
#include <sched.h>
#endif

#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace prompt_tile {

namespace {

int fallback_team_size() {
    const int ccx = native::detect_uarch().ccx_cores;
    return ccx > 1 ? ccx : 8;
}

#if defined(__linux__)
bool read_line(const std::string &path, std::string &out) {
    std::ifstream f(path);
    if (!f) return false;
    std::getline(f, out);
    return !out.empty();
}

/// sysfs cache sizes look like "1024K" or "32M".
int64_t parse_cache_size(const std::string &s) {
    char *end = nullptr;
    const long long v = std::strtoll(s.c_str(), &end, 10);
    if (end == s.c_str() || v <= 0) return 0;
    int64_t scale = 1;
    if (*end == 'K' || *end == 'k') scale = 1024;
    if (*end == 'M' || *end == 'm') scale = int64_t(1024) * 1024;
    if (*end == 'G' || *end == 'g') scale = int64_t(1024) * 1024 * 1024;
    return static_cast<int64_t>(v) * scale;
}

/// Directory of `cpu`'s unified cache at `level`, or "" if absent.
std::string cache_dir(int cpu, int level) {
    const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu)
            + "/cache/index";
    for (int i = 0; i < 8; ++i) {
        const std::string dir = base + std::to_string(i);
        std::string lvl, type;
        if (!read_line(dir + "/level", lvl)) break;
        if (std::atoi(lvl.c_str()) != level) continue;
        if (read_line(dir + "/type", type) && type == "Instruction") continue;
        return dir;
    }
    return std::string();
}

/// Stable key of the L3 that `cpu` belongs to: sysfs `id` when present,
/// otherwise the `shared_cpu_list` string; "" when unknown.
std::string l3_key(int cpu) {
    if (cpu < 0) return std::string();
    const std::string dir = cache_dir(cpu, 3);
    if (dir.empty()) return std::string();
    std::string key;
    if (read_line(dir + "/id", key)) return "id:" + key;
    if (read_line(dir + "/shared_cpu_list", key)) return "cpus:" + key;
    return std::string();
}

int64_t cache_bytes(int cpu, int level) {
    const std::string dir = cache_dir(cpu, level);
    std::string size;
    if (dir.empty() || !read_line(dir + "/size", size)) return 0;
    return parse_cache_size(size);
}
#endif

void build_steal_order(TeamMap &m) {
    const int n = m.n_teams;
    m.steal_order.clear();
    m.steal_order.reserve(static_cast<size_t>(n) * std::max(n - 1, 0));
    for (int t = 0; t < n; ++t) {
        for (const bool same : {true, false}) {
            for (int d = 1; d < n; ++d) {
                const int v = (t + d) % n;
                if ((m.domain_of_team[v] == m.domain_of_team[t]) == same)
                    m.steal_order.push_back(v);
            }
        }
    }
}

std::vector<int> contiguous_domains(int num_threads, int size) {
    std::vector<int> d(std::max(num_threads, 1));
    const int s = std::max(size, 1);
    for (size_t t = 0; t < d.size(); ++t)
        d[t] = static_cast<int>(t) / s;
    return d;
}

TeamMap detect(int num_threads, int team_override, int subteam, bool nested) {
    const auto &ua = native::detect_uarch();
    int64_t l2 = ua.l2_bytes > 0 ? ua.l2_bytes : (int64_t(1) << 20);
    int64_t l3 = ua.l3_bytes_per_ccd > 0 ? ua.l3_bytes_per_ccd
                                         : (int64_t(32) << 20);
    if (team_override > 0)
        return make_contiguous_team_map(num_threads, team_override, l2, l3);
    const auto fallback = [&]() {
        return make_domain_team_map(
                contiguous_domains(num_threads, fallback_team_size()), subteam,
                l2, l3);
    };
    if (nested || num_threads <= 1) return fallback();
#if defined(__linux__)
    std::vector<int> cpu(num_threads, -1);
#pragma omp parallel num_threads(num_threads)
    {
        if (omp_get_num_threads() == num_threads)
            cpu[omp_get_thread_num()] = sched_getcpu();
    }
    std::vector<int> domain(num_threads, 0);
    std::map<std::string, int> domain_id;
    for (int t = 0; t < num_threads; ++t) {
        const std::string key = l3_key(cpu[t]);
        if (key.empty()) return fallback();
        domain[t] = domain_id.emplace(key, static_cast<int>(domain_id.size()))
                            .first->second;
    }
    const int64_t sys_l2 = cache_bytes(cpu[0], 2);
    const int64_t sys_l3 = cache_bytes(cpu[0], 3);
    TeamMap m = make_domain_team_map(domain, subteam, sys_l2 > 0 ? sys_l2 : l2,
            sys_l3 > 0 ? sys_l3 : l3);
    m.from_l3 = true;
    return m;
#else
    return fallback();
#endif
}

} // namespace

TeamMap make_contiguous_team_map(
        int num_threads, int team_size, int64_t l2_bytes, int64_t l3_bytes) {
    TeamMap m;
    m.num_threads = std::max(num_threads, 1);
    const int ts = std::clamp(team_size, 1, m.num_threads);
    m.n_teams = (m.num_threads + ts - 1) / ts;
    m.team_of.resize(m.num_threads);
    m.team_size.assign(m.n_teams, 0);
    for (int t = 0; t < m.num_threads; ++t) {
        m.team_of[t] = t / ts;
        ++m.team_size[t / ts];
    }
    m.domain_of_team.resize(m.n_teams);
    std::iota(m.domain_of_team.begin(), m.domain_of_team.end(), 0);
    build_steal_order(m);
    m.l2_bytes = l2_bytes;
    m.l3_bytes = l3_bytes;
    m.from_l3 = false;
    return m;
}

TeamMap make_domain_team_map(const std::vector<int> &domain_of_thread,
        int subteam, int64_t l2_bytes, int64_t l3_bytes) {
    TeamMap m;
    m.num_threads = static_cast<int>(domain_of_thread.size());
    m.team_of.assign(m.num_threads, 0);
    m.l2_bytes = l2_bytes;
    m.l3_bytes = l3_bytes;
    m.from_l3 = false;

    // Member threads of each domain, domains numbered by first appearance.
    std::map<int, int> dense;
    std::vector<std::vector<int>> members;
    for (int t = 0; t < m.num_threads; ++t) {
        const auto it = dense.emplace(
                domain_of_thread[t], static_cast<int>(members.size()));
        if (it.second) members.emplace_back();
        members[it.first->second].push_back(t);
    }
    for (size_t d = 0; d < members.size(); ++d) {
        const int s = static_cast<int>(members[d].size());
        const int k = subteam > 0 ? (s + subteam - 1) / subteam : 1;
        const int base = m.n_teams;
        m.team_size.resize(base + k, 0);
        m.domain_of_team.resize(base + k, static_cast<int>(d));
        for (int i = 0; i < s; ++i) {
            m.team_of[members[d][i]] = base + i * k / s;
            ++m.team_size[base + i * k / s];
        }
        m.n_teams += k;
    }
    build_steal_order(m);
    return m;
}

const TeamMap &get_team_map(int num_threads, int team_override, int subteam) {
    using Key = std::tuple<int, int, int, bool>;
    const bool nested = omp_in_parallel() != 0;
    const Key key(num_threads, team_override, subteam, nested);
    // Maps are never evicted, so a thread may keep its last hit.
    static thread_local Key last_key;
    static thread_local const TeamMap *last = nullptr;
    if (last != nullptr && key == last_key) return *last;

    static std::mutex mu;
    static std::map<Key, std::unique_ptr<TeamMap>> cache;
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(key);
    if (it == cache.end()) {
        it = cache.emplace(key,
                          std::make_unique<TeamMap>(detect(
                                  num_threads, team_override, subteam, nested)))
                     .first;
    }
    last_key = key;
    last = it->second.get();
    return *last;
}

} // namespace prompt_tile
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
