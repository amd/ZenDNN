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

#include "gdn_utils.hpp"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace zendnnl {
namespace benchdnn {
namespace gdn {

namespace {

std::string lower(std::string v) {
    std::transform(v.begin(), v.end(), v.begin(), ::tolower);
    return v;
}

bool parseBool(const std::string &raw) {
    const std::string v = lower(raw);
    if (v == "true" || v == "1") { return true; }
    if (v == "false" || v == "0") { return false; }
    throw std::invalid_argument(
            "invalid bool '" + raw + "' (use true/false or 1/0)");
}

gdn_state_layout_t parseLayout(const std::string &raw) {
    const std::string v = lower(raw);
    if (v == "kv") { return gdn_state_layout_t::kv; }
    if (v == "vk") { return gdn_state_layout_t::vk; }
    throw std::invalid_argument(
            "invalid state_layout '" + raw + "' (use kv or vk)");
}

gdn_prefill_algo_t parseAlgo(const std::string &raw) {
    const std::string v = lower(raw);
    if (v == "auto" || v == "automatic") {
        return gdn_prefill_algo_t::automatic;
    }
    if (v == "recurrent") { return gdn_prefill_algo_t::recurrent; }
    if (v == "chunked") { return gdn_prefill_algo_t::chunked; }
    throw std::invalid_argument("invalid prefill_algo '" + raw
            + "' (use auto, recurrent or chunked)");
}

gdn_kernel_t parseKernel(const std::string &raw) {
    const std::string v = lower(raw);
    if (v == "auto" || v == "none") { return gdn_kernel_t::none; }
    if (v == "native") { return gdn_kernel_t::native; }
    if (v == "reference") { return gdn_kernel_t::reference; }
    throw std::invalid_argument(
            "invalid kernel '" + raw + "' (use auto, native or reference)");
}

std::string layoutToStr(gdn_state_layout_t l) {
    return l == gdn_state_layout_t::vk ? "vk" : "kv";
}

std::string algoToStr(gdn_prefill_algo_t a) {
    return a == gdn_prefill_algo_t::automatic
            ? "auto"
            : zendnnl::lowoha::gdn::prefill_algo_to_string(a);
}

std::string kernelToStr(gdn_kernel_t k) {
    return k == gdn_kernel_t::none ? "auto"
                                   : zendnnl::lowoha::gdn::kernel_to_string(k);
}

std::string boolToStr(bool b) {
    return b ? "true" : "false";
}

bool isGdnFloat(data_type_t dt) {
    return dt == data_type_t::f32 || dt == data_type_t::bf16
            || dt == data_type_t::f16;
}

int64_t dtSize(data_type_t dt) {
    return dt == data_type_t::f32 ? 4 : 2;
}

} // anonymous namespace

bool isSupportedGdnConfig(const GdnConfig &cfg, std::string &reason) {
    // Mirrors the checks of gdn_direct (lowoha_gdn.cpp).
    if (cfg.num_k_heads <= 0 || cfg.num_v_heads <= 0 || cfg.head_k_dim <= 0
            || cfg.head_v_dim <= 0) {
        reason = "num_k_heads, num_v_heads, head_k_dim and head_v_dim must "
                 "all be > 0";
        return false;
    }
    if (cfg.num_v_heads % cfg.num_k_heads != 0) {
        reason = "num_v_heads must be a multiple of num_k_heads";
        return false;
    }
    if (cfg.conv_width < 0) {
        reason = "conv_width must be >= 0 (0 = pre-convolved input)";
        return false;
    }
    if (cfg.num_decodes < 0 || cfg.num_prefills < 0 || cfg.prefill_len < 0) {
        reason = "num_decodes, num_prefills and prefill_len must be >= 0";
        return false;
    }
    if (cfg.batch() == 0) {
        reason = "the batch needs at least one decode or prefill sequence";
        return false;
    }
    if (cfg.num_prefills > 0 && cfg.prefill_len == 0) {
        reason = "prefill_len must be > 0 when num_prefills > 0";
        return false;
    }
    if (cfg.num_tokens() > std::numeric_limits<int32_t>::max()) {
        reason = "num_tokens must fit the int32 cu_seqlens";
        return false;
    }
    if (!isGdnFloat(cfg.src_dt) || !isGdnFloat(cfg.state_dt)) {
        reason = "src_dt and state_dt must be f32, bf16 or f16";
        return false;
    }
    if (cfg.num_threads < 0) {
        reason = "num_threads must be >= 0 (0 = auto)";
        return false;
    }
    if (cfg.iters <= 0) {
        reason = "iters must be > 0";
        return false;
    }
    if (cfg.warmup_iters < 0) {
        reason = "warmup_iters must be >= 0";
        return false;
    }
    return true;
}

void inputFileParser(std::ifstream &infile, std::vector<GdnConfig> &configs,
        const global_options &options) {
    std::string line;
    while (std::getline(infile, line)) {
        if (line.empty() || line[0] == '#') { continue; }

        auto fields = split(line, ',');
        if (fields.size() < GDN_REQUIRED_FIELD_COUNT) {
            commonlog_error("Invalid line (expected at least ",
                    GDN_REQUIRED_FIELD_COUNT,
                    " fields): "
                    "[model, num_k_heads, num_v_heads, head_k_dim, "
                    "head_v_dim, conv_width, num_decodes, num_prefills, "
                    "prefill_len, (iters, warmup_iters, src_dt, state_dt, "
                    "has_initial_state, state_layout, fused_norm, "
                    "prefill_algo, kernel, num_threads)]");
            continue;
        }

        GdnConfig cfg;
        try {
            size_t id = 0;
            cfg.model = fields[id++];
            cfg.num_k_heads = std::stoll(fields[id++]);
            cfg.num_v_heads = std::stoll(fields[id++]);
            cfg.head_k_dim = std::stoll(fields[id++]);
            cfg.head_v_dim = std::stoll(fields[id++]);
            cfg.conv_width = std::stoll(fields[id++]);
            cfg.num_decodes = std::stoll(fields[id++]);
            cfg.num_prefills = std::stoll(fields[id++]);
            cfg.prefill_len = std::stoll(fields[id++]);

            auto has = [&](size_t i) {
                return i < fields.size() && !fields[i].empty();
            };
            cfg.iters = has(id) ? std::stoi(fields[id]) : options.iters;
            id++;
            cfg.warmup_iters = has(id)          ? std::stoi(fields[id])
                    : options.warmup_iters >= 0 ? options.warmup_iters
                                                : cfg.iters / 5;
            id++;
            cfg.src_dt = has(id) ? strToDatatype(fields[id])
                    : options.gdn_src_dt != data_type_t::none
                    ? options.gdn_src_dt
                    : data_type_t::bf16;
            id++;
            cfg.state_dt = has(id) ? strToDatatype(fields[id])
                    : options.gdn_state_dt != data_type_t::none
                    ? options.gdn_state_dt
                    : data_type_t::f32;
            id++;
            cfg.has_initial_state = has(id) ? parseBool(fields[id]) : true;
            id++;
            cfg.state_layout = has(id) ? parseLayout(fields[id])
                                       : gdn_state_layout_t::kv;
            id++;
            cfg.fused_norm = has(id) ? parseBool(fields[id]) : false;
            id++;
            cfg.prefill_algo = has(id) ? parseAlgo(fields[id])
                                       : gdn_prefill_algo_t::automatic;
            id++;
            cfg.kernel = has(id) ? parseKernel(fields[id]) : gdn_kernel_t::none;
            id++;
            cfg.num_threads
                    = has(id) ? std::stoi(fields[id]) : options.num_threads;
            id++;

            std::string reason;
            if (!isSupportedGdnConfig(cfg, reason)) {
                commonlog_error("Skipping unsupported GDN config (", cfg.model,
                        "): ", reason);
                continue;
            }
            configs.push_back(cfg);
        } catch (const std::exception &e) {
            commonlog_error("Failed to parse line '", line, "': ", e.what());
            continue;
        }
    }
}

void log_benchmark_failure(const GdnConfig &cfg) {
    testlog_error("Benchmark failed for GDN config: model=", cfg.model,
            ", num_k_heads=", cfg.num_k_heads,
            ", num_v_heads=", cfg.num_v_heads, ", head_k_dim=", cfg.head_k_dim,
            ", head_v_dim=", cfg.head_v_dim, ", conv_width=", cfg.conv_width,
            ", num_decodes=", cfg.num_decodes,
            ", num_prefills=", cfg.num_prefills,
            ", prefill_len=", cfg.prefill_len,
            ", src_dt=", datatypeToStr(cfg.src_dt),
            ", state_dt=", datatypeToStr(cfg.state_dt),
            ", has_initial_state=", boolToStr(cfg.has_initial_state),
            ", state_layout=", layoutToStr(cfg.state_layout),
            ", fused_norm=", boolToStr(cfg.fused_norm),
            ", prefill_algo=", algoToStr(cfg.prefill_algo),
            ", kernel=", kernelToStr(cfg.kernel),
            ", num_threads=", cfg.num_threads, ", iters=", cfg.iters,
            ", warmup_iters=", cfg.warmup_iters);
}

namespace {

std::vector<std::string> buildHeaders() {
    return {"Model", "K_Heads", "V_Heads", "K_Dim", "V_Dim", "Conv_W",
            "Decodes", "Prefills", "Prefill_Len", "Tokens", "Src_DT",
            "State_DT", "Init_State", "Layout", "Fused_Norm", "Prefill_Algo",
            "Kernel", "Num_Threads", "Iters", "Warmup_Iters", "Total_time(ms)",
            "Avg_time(ms)", "State_GB/s"};
}

std::string fixed(double v, int digits) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(digits) << v;
    return ss.str();
}

// SSM state bytes one call moves: every slot is written once and read once,
// except the slots of prefills that start from zero states.
double stateBytes(const GdnConfig &cfg) {
    const double slot = static_cast<double>(cfg.num_v_heads) * cfg.head_k_dim
            * cfg.head_v_dim * dtSize(cfg.state_dt);
    const int64_t reads
            = cfg.num_decodes + (cfg.has_initial_state ? cfg.num_prefills : 0);
    return slot * static_cast<double>(cfg.batch() + reads);
}

std::vector<std::string> buildRow(
        const GdnConfig &cfg, const TimingStats &stat) {
    const double avg = cfg.iters > 0 ? stat.total_time_ms / cfg.iters : 0.0;
    return {cfg.model, std::to_string(cfg.num_k_heads),
            std::to_string(cfg.num_v_heads), std::to_string(cfg.head_k_dim),
            std::to_string(cfg.head_v_dim), std::to_string(cfg.conv_width),
            std::to_string(cfg.num_decodes), std::to_string(cfg.num_prefills),
            std::to_string(cfg.prefill_len), std::to_string(cfg.num_tokens()),
            datatypeToStr(cfg.src_dt), datatypeToStr(cfg.state_dt),
            boolToStr(cfg.has_initial_state), layoutToStr(cfg.state_layout),
            boolToStr(cfg.fused_norm), algoToStr(cfg.prefill_algo),
            kernelToStr(cfg.kernel),
            cfg.num_threads == 0 ? "auto" : std::to_string(cfg.num_threads),
            std::to_string(cfg.iters), std::to_string(cfg.warmup_iters),
            fixed(stat.total_time_ms, 2), fixed(avg, 6),
            fixed(avg > 0.0 ? stateBytes(cfg) / (avg * 1e6) : 0.0, 2)};
}

} // anonymous namespace

void print_results(std::vector<std::pair<GdnConfig, TimingStats>> &results,
        std::ostream &outfile) {
    const auto headers = buildHeaders();
    std::vector<size_t> col_widths(headers.size());
    for (size_t i = 0; i < headers.size(); ++i) {
        col_widths[i] = headers[i].size() + 2;
    }
    std::vector<std::vector<std::string>> rows;
    rows.reserve(results.size());
    for (const auto &result : results) {
        auto row = buildRow(result.first, result.second);
        for (size_t i = 0; i < row.size() && i < col_widths.size(); ++i) {
            col_widths[i] = std::max(col_widths[i], row[i].size() + 2);
        }
        rows.push_back(std::move(row));
    }
    auto print_row = [&](const std::vector<std::string> &row) {
        for (size_t i = 0; i < row.size(); ++i) {
            outfile << std::setw(col_widths[i]) << row[i];
        }
        outfile << std::endl;
    };
    outfile << std::left;
    print_row(headers);
    for (const auto &r : rows) {
        print_row(r);
    }
}

void log_results(std::vector<std::pair<GdnConfig, TimingStats>> &results,
        std::ostream &outfile) {
    const auto headers = buildHeaders();
    for (size_t i = 0; i < headers.size(); ++i) {
        outfile << headers[i] << ((i + 1 < headers.size()) ? ", " : "\n");
    }
    for (const auto &result : results) {
        const auto row = buildRow(result.first, result.second);
        for (size_t i = 0; i < row.size(); ++i) {
            outfile << row[i] << ((i + 1 < row.size()) ? ", " : "\n");
        }
    }
}

} // namespace gdn
} // namespace benchdnn
} // namespace zendnnl
