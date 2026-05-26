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

#include "sdpa_utils.hpp"
// `sdpa_utils.hpp` only forward-declares `qkv_layout_t` (to avoid an include
// cycle); this .cpp uses the enumerators directly, so it needs the full
// definition from `sdpa_tensor_factory.hpp`.
#include "sdpa_tensor_factory.hpp"

#include <algorithm>  // std::transform (lines 34, 63)
#include <iomanip>
#include <sstream>
#include <stdexcept>  // std::invalid_argument (strToQkvLayout, line 70)

namespace zendnnl {
namespace benchdnn {
namespace sdpa {

namespace {

bool parseBoolFlag(const std::string &raw, bool &out) {
  std::string v = raw;
  std::transform(v.begin(), v.end(), v.begin(), ::tolower);
  if (v == "true" || v == "1") {
    out = true;
    return true;
  }
  if (v == "false" || v == "0") {
    out = false;
    return true;
  }
  return false;
}

std::string maskNdimsToStr(int n) {
  switch (n) {
  case 0: return "none";
  case 2: return "2D";
  case 4: return "4D";
  default: return std::to_string(n);
  }
}

std::string boolToStr(bool b) {
  return b ? "true" : "false";
}

} // anonymous namespace

qkv_layout_t strToQkvLayout(const std::string &str) {
  std::string v = str;
  std::transform(v.begin(), v.end(), v.begin(), ::tolower);
  if (v == "bhsd") {
    return qkv_layout_t::bhsd;
  }
  if (v == "bshd") {
    return qkv_layout_t::bshd;
  }
  throw std::invalid_argument(
    "Unknown qkv_layout '" + str + "'. Use 'bhsd' or 'bshd'.");
}

std::string qkvLayoutToStr(qkv_layout_t layout) {
  switch (layout) {
  case qkv_layout_t::bhsd: return "bhsd";
  case qkv_layout_t::bshd: return "bshd";
  default:                 return "unknown";
  }
}

bool isSupportedSdpaConfig(const SdpaConfig &cfg, std::string &reason) {
  // Mirror checks in lowoha_flash_sdpa_utils.cpp::validate_flash_sdpa_inputs.
  if (cfg.batch <= 0 || cfg.num_heads <= 0 || cfg.seq_len <= 0 ||
      cfg.head_dim <= 0) {
    reason = "batch, num_heads, seq_len and head_dim must all be > 0";
    return false;
  }
  if (cfg.kv_seq_len < 0) {
    reason = "kv_seq_len must be >= 0 (0 means same as seq_len)";
    return false;
  }
  if (cfg.qkv_dt != data_type_t::f32 && cfg.qkv_dt != data_type_t::bf16 &&
      cfg.qkv_dt != data_type_t::f16) {
    reason = "qkv_dt must be f32, bf16 or f16";
    return false;
  }
  if (cfg.dropout_p != 0.0) {
    reason = "dropout_p must be 0.0 (only zero dropout is supported)";
    return false;
  }
  if (cfg.mask_ndims != 0 && cfg.mask_ndims != 2 && cfg.mask_ndims != 4) {
    reason = "mask_ndims must be 0, 2, or 4";
    return false;
  }
  if (cfg.mask_ndims > 0) {
    if (cfg.qkv_dt == data_type_t::f32 && cfg.mask_dt != data_type_t::f32) {
      reason = "mask_dt must be f32 when qkv_dt is f32";
      return false;
    }
    if (cfg.qkv_dt == data_type_t::bf16 &&
        cfg.mask_dt != data_type_t::f32 &&
        cfg.mask_dt != data_type_t::bf16) {
      reason = "mask_dt must be f32 or bf16 when qkv_dt is bf16";
      return false;
    }
    if (cfg.qkv_dt == data_type_t::f16 &&
        cfg.mask_dt != data_type_t::f32 &&
        cfg.mask_dt != data_type_t::f16) {
      reason = "mask_dt must be f32 or f16 when qkv_dt is f16";
      return false;
    }
  }
  if (cfg.out_dt != data_type_t::none && cfg.out_dt != cfg.qkv_dt) {
    reason = "out_dt must equal qkv_dt (or 'none' to default to qkv_dt)";
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

void inputFileParser(std::ifstream &infile, std::vector<SdpaConfig> &configs) {
  std::string line;
  while (std::getline(infile, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }

    auto fields = split(line, ',');
    if (fields.size() < SDPA_REQUIRED_FIELD_COUNT) {
      commonlog_error(
        "Invalid line (expected at least ", SDPA_REQUIRED_FIELD_COUNT, " fields): "
        "[batch, num_heads, seq_len, kv_seq_len, head_dim, qkv_dt, "
        "is_causal, mask_ndims, mask_dt, iters, "
        "(warmup_iters, scale, num_threads, out_dt, qkv_layout)]");
      continue;
    }

    SdpaConfig cfg;
    try {
      int id = 0;
      cfg.batch      = std::stoll(fields[id++]);
      cfg.num_heads  = std::stoll(fields[id++]);
      cfg.seq_len    = std::stoll(fields[id++]);
      cfg.kv_seq_len = std::stoll(fields[id++]);
      cfg.head_dim   = std::stoll(fields[id++]);

      if (fields[id].empty()) {
        commonlog_error("qkv_dt is required (f32, bf16 or f16).");
        continue;
      }
      cfg.qkv_dt = strToDatatype(fields[id++]);

      bool causal = false;
      if (!parseBoolFlag(fields[id], causal)) {
        commonlog_error("is_causal: invalid value '", fields[id],
                        "'. Use true/false or 1/0.");
        continue;
      }
      cfg.is_causal = causal;
      id++;

      if (fields[id].empty()) {
        commonlog_error("mask_ndims is required (0, 2, or 4).");
        continue;
      }
      cfg.mask_ndims = std::stoi(fields[id++]);

      if (cfg.mask_ndims > 0) {
        if (fields[id].empty() || fields[id] == "none") {
          commonlog_error("mask_dt is required when mask_ndims > 0.");
          continue;
        }
        cfg.mask_dt = strToDatatype(fields[id]);
      }
      else {
        cfg.mask_dt = data_type_t::none;
      }
      id++;

      if (fields[id].empty()) {
        commonlog_error("iters is required.");
        continue;
      }
      cfg.iters = std::stoi(fields[id++]);

      // Optional trailing fields: warmup_iters, scale, num_threads, out_dt.
      cfg.warmup_iters = (id < (int)fields.size() && !fields[id].empty()) ?
                         std::stoi(fields[id]) :
                         static_cast<int>(0.2 * cfg.iters);
      id++;

      cfg.scale = (id < (int)fields.size() && !fields[id].empty()) ?
                  std::stod(fields[id]) : 0.0;
      id++;

      cfg.num_threads = (id < (int)fields.size() && !fields[id].empty()) ?
                        std::stoi(fields[id]) : 0;
      id++;

      cfg.out_dt = (id < (int)fields.size() && !fields[id].empty()) ?
                   strToOptionalDatatype(fields[id]) : data_type_t::none;
      id++;

      // Optional 15th field: qkv_layout. Defaults to bhsd for backwards
      // compatibility with all pre-existing input files.
      cfg.qkv_layout = (id < (int)fields.size() && !fields[id].empty()) ?
                      strToQkvLayout(fields[id]) : qkv_layout_t::bhsd;
      id++;

      cfg.dropout_p = 0.0;

      std::string reason;
      if (!isSupportedSdpaConfig(cfg, reason)) {
        commonlog_error("Skipping unsupported SDPA config: ", reason);
        continue;
      }

      configs.push_back(cfg);
    }
    catch (const std::exception &e) {
      commonlog_error("Failed to parse line '", line, "': ", e.what());
      continue;
    }
  }
}

void inputCommandLineParser(std::vector<SdpaConfig> &configs,
                            const global_options &options) {
  SdpaConfig cfg;
  try {
    cfg.batch      = static_cast<int64_t>(options.bs);
    cfg.num_heads  = options.num_heads;
    cfg.seq_len    = options.seq_len;
    cfg.kv_seq_len = options.kv_seq_len;
    cfg.head_dim   = options.head_dim;
    cfg.qkv_dt     = options.sdt;
    cfg.is_causal  = options.is_causal;
    cfg.mask_ndims = options.mask_ndims;
    cfg.mask_dt    = options.mask_dt;
    cfg.iters      = options.iters;
    cfg.warmup_iters = options.warmup_iters < 0 ?
                       static_cast<int>(0.2 * cfg.iters) : options.warmup_iters;
    cfg.scale      = options.scale;
    cfg.num_threads = options.num_threads;
    cfg.out_dt     = options.out_dt;
    cfg.qkv_layout = strToQkvLayout(options.qkv_layout);
    cfg.dropout_p  = 0.0;

    std::string reason;
    if (!isSupportedSdpaConfig(cfg, reason)) {
      commonlog_error("Skipping unsupported SDPA config: ", reason);
      return;
    }

    configs.push_back(cfg);
  }
  catch (const std::exception &e) {
    commonlog_error("CLI parse error: ", e.what());
  }
}

void inputModelFileParser(std::ifstream &infile,
                          std::vector<SdpaConfig> &configs,
                          const global_options &options) {
  std::string line;
  while (std::getline(infile, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }

    auto fields = split(line, ',');
    // Expected: ModelName, batch, num_heads, seq_len, kv_seq_len, head_dim
    if (fields.size() < 6) {
      commonlog_error(
        "Invalid model file line (expected 6 fields): "
        "[ModelName, batch, num_heads, seq_len, kv_seq_len, head_dim] - got: ", line);
      continue;
    }

    SdpaConfig cfg;
    try {
      int id = 0;
      cfg.modelName  = fields[id++];
      cfg.batch      = std::stoll(fields[id++]);
      cfg.num_heads  = std::stoll(fields[id++]);
      cfg.seq_len    = std::stoll(fields[id++]);
      cfg.kv_seq_len = std::stoll(fields[id++]);
      cfg.head_dim   = std::stoll(fields[id++]);

      // Defaults from CLI options.
      cfg.qkv_dt     = options.sdt;
      cfg.is_causal  = options.is_causal;
      cfg.mask_ndims = options.mask_ndims;
      cfg.mask_dt    = options.mask_dt;
      cfg.iters      = options.iters;
      cfg.warmup_iters = options.warmup_iters < 0 ?
                         static_cast<int>(0.2 * cfg.iters) : options.warmup_iters;
      cfg.scale      = options.scale;
      cfg.num_threads = options.num_threads;
      cfg.out_dt     = options.out_dt;
      cfg.qkv_layout = strToQkvLayout(options.qkv_layout);
      cfg.dropout_p  = 0.0;

      std::string reason;
      if (!isSupportedSdpaConfig(cfg, reason)) {
        commonlog_error("Skipping unsupported SDPA model config (", cfg.modelName,
                        "): ", reason);
        continue;
      }

      configs.push_back(cfg);
    }
    catch (const std::exception &e) {
      commonlog_error("Failed to parse model line '", line, "': ", e.what());
      continue;
    }
  }
}

void log_benchmark_failure(const SdpaConfig &cfg) {
  testlog_error(
    "Benchmark failed for SDPA config: "
    "batch=", cfg.batch, ", num_heads=", cfg.num_heads,
    ", seq_len=", cfg.seq_len, ", kv_seq_len=", cfg.kv_seq_len,
    ", head_dim=", cfg.head_dim,
    ", qkv_dt=", datatypeToStr(cfg.qkv_dt),
    ", qkv_layout=", qkvLayoutToStr(cfg.qkv_layout),
    ", is_causal=", boolToStr(cfg.is_causal),
    ", mask_ndims=", cfg.mask_ndims,
    ", mask_dt=", datatypeToStr(cfg.mask_dt),
    ", out_dt=", datatypeToStr(cfg.out_dt),
    ", scale=", cfg.scale,
    ", num_threads=", cfg.num_threads,
    ", iters=", cfg.iters,
    ", warmup_iters=", cfg.warmup_iters);
}

namespace {

std::vector<std::string> buildHeaders(const InputMode inputMode) {
  std::vector<std::string> headers;
  if (inputMode == InputMode::MODEL) {
    headers.push_back("Model_Name");
  }
  headers.insert(headers.end(), {
    "Batch", "Num_Heads", "Seq_Len", "KV_Seq_Len", "Head_Dim",
    "QKV_DT", "QKV_Layout", "Is_Causal", "Mask_Ndims", "Mask_DT", "Out_DT",
    "Scale", "Num_Threads", "Iters", "Warmup_Iters",
    "Total_time(ms)", "Avg_time(ms)"
  });
  return headers;
}

std::vector<std::string> buildRow(const SdpaConfig &cfg,
                                  const TimingStats &stat,
                                  const InputMode inputMode) {
  std::vector<std::string> row;
  if (inputMode == InputMode::MODEL) {
    row.push_back(cfg.modelName);
  }
  row.push_back(std::to_string(cfg.batch));
  row.push_back(std::to_string(cfg.num_heads));
  row.push_back(std::to_string(cfg.seq_len));
  row.push_back(std::to_string(cfg.kv_seq_len));
  row.push_back(std::to_string(cfg.head_dim));
  row.push_back(datatypeToStr(cfg.qkv_dt));
  row.push_back(qkvLayoutToStr(cfg.qkv_layout));
  row.push_back(boolToStr(cfg.is_causal));
  row.push_back(maskNdimsToStr(cfg.mask_ndims));
  row.push_back(datatypeToStr(cfg.mask_dt));
  row.push_back(datatypeToStr(cfg.out_dt));
  {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6) << cfg.scale;
    row.push_back(ss.str());
  }
  row.push_back((cfg.num_threads == 0) ? "auto"
                : std::to_string(cfg.num_threads));
  row.push_back(std::to_string(cfg.iters));
  row.push_back(std::to_string(cfg.warmup_iters));
  {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2) << stat.total_time_ms;
    row.push_back(ss.str());
  }
  {
    double avg = (cfg.iters > 0) ? (stat.total_time_ms / cfg.iters) : 0.0;
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6) << avg;
    row.push_back(ss.str());
  }
  return row;
}

} // anonymous namespace

void print_results(std::vector<std::pair<SdpaConfig, TimingStats>>
                   &sdpa_results, std::ostream &outfile,
                   const InputMode inputMode) {

  auto headers = buildHeaders(inputMode);
  std::vector<size_t> col_widths(headers.size());
  for (size_t i = 0; i < headers.size(); ++i) {
    col_widths[i] = headers[i].size() + 2;
  }

  // First pass: compute column widths.
  std::vector<std::vector<std::string>> rows;
  rows.reserve(sdpa_results.size());
  for (const auto &result : sdpa_results) {
    auto row = buildRow(result.first, result.second, inputMode);
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

void log_results(std::vector<std::pair<SdpaConfig, TimingStats>>
                 &sdpa_results, std::ostream &outfile,
                 const InputMode inputMode) {

  auto headers = buildHeaders(inputMode);
  for (size_t i = 0; i < headers.size(); ++i) {
    outfile << headers[i];
    outfile << ((i + 1 < headers.size()) ? ", " : "\n");
  }

  for (const auto &result : sdpa_results) {
    auto row = buildRow(result.first, result.second, inputMode);
    for (size_t i = 0; i < row.size(); ++i) {
      outfile << row[i];
      outfile << ((i + 1 < row.size()) ? ", " : "\n");
    }
  }
}

} // namespace sdpa
} // namespace benchdnn
} // namespace zendnnl
