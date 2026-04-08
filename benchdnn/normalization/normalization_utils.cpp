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

#include "normalization_utils.hpp"

namespace zendnnl {
namespace benchdnn {
namespace normalization {

std::string strToNormType(const std::string &str) {
  std::string lower = str;
  std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
  if (lower == "layer_norm" || lower == "layernorm") {
    return "layer_norm";
  }
  if (lower == "batch_norm" || lower == "batchnorm") {
    return "batch_norm";
  }
  if (lower == "rms_norm" || lower == "rmsnorm") {
    return "rms_norm";
  }
  if (lower == "fused_add_rms_norm" || lower == "fusedaddrmsnorm") {
    return "fused_add_rms_norm";
  }
  return "";
}

std::vector<uint64_t> parseShape(const std::string &shape_str) {
  std::vector<uint64_t> dims;
  std::stringstream ss(shape_str);
  std::string token;
  while (std::getline(ss, token, 'x')) {
    if (!token.empty()) {
      dims.push_back(std::stoull(token));
    }
  }
  return dims;
}

static std::string shapeToStr(const std::vector<uint64_t> &shape) {
  std::ostringstream ss;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) {
      ss << "x";
    }
    ss << shape[i];
  }
  return ss.str();
}

uint64_t compute_norm_size(const NormalizationConfig &cfg) {
  uint64_t norm_size = 1;
  int ndims = static_cast<int>(cfg.shape.size());
  for (int i = ndims - cfg.norm_ndims; i < ndims; ++i) {
    norm_size *= cfg.shape[i];
  }
  return norm_size;
}

uint64_t compute_num_channels(const NormalizationConfig &cfg) {
  if (cfg.shape.size() >= 2) {
    return cfg.shape[1];
  }
  return 0;
}

norm_type_t strToLowohaType(const std::string &norm_type) {
  if (norm_type == "layer_norm") {
    return norm_type_t::LAYER_NORM;
  }
  if (norm_type == "batch_norm") {
    return norm_type_t::BATCH_NORM;
  }
  if (norm_type == "rms_norm") {
    return norm_type_t::RMS_NORM;
  }
  if (norm_type == "fused_add_rms_norm") {
    return norm_type_t::FUSED_ADD_RMS_NORM;
  }
  return norm_type_t::NONE;
}

norm_algo_t strToLowohaAlgo(const std::string &algo) {
  if (algo == "dynamic_dispatch") {
    return norm_algo_t::dynamic_dispatch;
  }
  if (algo == "reference") {
    return norm_algo_t::reference;
  }
  return norm_algo_t::none;
}

void inputParser(std::ifstream &infile,
                 std::vector<NormalizationConfig> &configs) {
  std::string line;

  while (std::getline(infile, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }

    auto fields = split(line, ',');
    if (fields.size() < NORM_REQUIRED_FIELD_COUNT) {
      commonlog_error(
        "Invalid line (expected at least ", NORM_REQUIRED_FIELD_COUNT, " fields): "
        "[norm_type, shape, norm_ndims, src_dt:dst_dt, epsilon, use_scale, "
        "use_shift, iters, warmup_iters (opt), gamma_dt (opt), beta_dt (opt), "
        "algorithm (opt), num_threads (opt), isInplace (opt)]");
      continue;
    }

    NormalizationConfig cfg;
    try {
      int id = 0;

      cfg.norm_type = strToNormType(fields[id]);
      if (cfg.norm_type.empty()) {
        commonlog_error("Unknown norm_type: ", fields[id],
                        ". Supported: layer_norm, batch_norm, rms_norm, fused_add_rms_norm");
        continue;
      }
      id++;

      cfg.shape = parseShape(fields[id]);
      if (cfg.shape.empty()) {
        commonlog_error("Invalid shape: ", fields[id],
                        ". Expected format: dim0xdim1x... (e.g., 2x4096)");
        continue;
      }

      int ndims = static_cast<int>(cfg.shape.size());
      constexpr int NORM_MAX_NDIMS = 5;

      if (ndims > NORM_MAX_NDIMS) {
        commonlog_error("Shape has ", ndims, " dimensions, but normalization supports "
                        "at most ", NORM_MAX_NDIMS, "D tensors. Got shape: ", fields[id]);
        continue;
      }

      if (cfg.norm_type == "batch_norm" && ndims < 2) {
        commonlog_error("batch_norm requires shape with >= 2 dimensions (N, C, ...), got ",
                        ndims, "D shape: ", fields[id]);
        continue;
      }
      id++;

      cfg.norm_ndims = std::stoi(fields[id++]);

      if (cfg.norm_type == "batch_norm") {
        if (cfg.norm_ndims != 0) {
          commonlog_warning("batch_norm uses norm_ndims=0 (channel-wise normalization). "
                            "Overriding provided norm_ndims=", cfg.norm_ndims, " to 0.");
          cfg.norm_ndims = 0;
        }
      }
      else {
        if (cfg.norm_ndims < 1 || cfg.norm_ndims > ndims) {
          commonlog_error("norm_ndims must be in range [1, ", ndims,
                          "] for ", cfg.norm_type, ", got ", cfg.norm_ndims);
          continue;
        }
      }

      auto dt = split(fields[id++], ':');
      if (dt.size() >= 2) {
        cfg.src_dt = strToDatatype(dt[0]);
        cfg.dst_dt = strToDatatype(dt[1]);
      }
      else if (dt.size() == 1) {
        cfg.src_dt = strToDatatype(dt[0]);
        cfg.dst_dt = cfg.src_dt;
        commonlog_warning("Only one data type specified. Using same type for src and dst.");
      }
      else {
        cfg.src_dt = data_type_t::f32;
        cfg.dst_dt = data_type_t::f32;
        commonlog_warning("No data types specified. Defaulting to f32.");
      }

      cfg.epsilon = std::stof(fields[id++]);

      std::string scale_flag = fields[id++];
      std::transform(scale_flag.begin(), scale_flag.end(), scale_flag.begin(),
                     ::tolower);
      cfg.use_scale = (scale_flag == "true" || scale_flag == "1");

      std::string shift_flag = fields[id++];
      std::transform(shift_flag.begin(), shift_flag.end(), shift_flag.begin(),
                     ::tolower);
      cfg.use_shift = (shift_flag == "true" || shift_flag == "1");

      cfg.iters = std::stoi(fields[id++]);

      // Optional fields: check fields.size() > id before accessing
      cfg.warmup_iters = (fields.size() > static_cast<size_t>(id) &&
                          !fields[id].empty()) ?
                         std::stoi(fields[id]) :
                         static_cast<int>(0.2 * cfg.iters);
      id++;

      cfg.gamma_dt = (fields.size() > static_cast<size_t>(id) &&
                      !fields[id].empty()) ?
                     strToDatatype(fields[id]) : data_type_t::f32;
      id++;

      cfg.beta_dt = (fields.size() > static_cast<size_t>(id) && !fields[id].empty()) ?
                    strToDatatype(fields[id]) : data_type_t::f32;
      id++;

      if (fields.size() > static_cast<size_t>(id) && !fields[id].empty()) {
        std::string algo = fields[id];
        std::transform(algo.begin(), algo.end(), algo.begin(), ::tolower);
        cfg.algorithm = algo;
      }
      else {
        cfg.algorithm = "none";
      }
      id++;

      cfg.num_threads = (fields.size() > static_cast<size_t>(id) &&
                         !fields[id].empty()) ?
                        std::stoi(fields[id]) : 0;
      id++;

      if (fields.size() > static_cast<size_t>(id) && !fields[id].empty()) {
        std::string inplace_flag = fields[id];
        std::transform(inplace_flag.begin(), inplace_flag.end(),
                       inplace_flag.begin(), ::tolower);
        cfg.isInplace = !(inplace_flag == "false" || inplace_flag == "0");
      }
      else {
        cfg.isInplace = true;
      }
      id++;

      if (cfg.isInplace && cfg.src_dt != cfg.dst_dt) {
        commonlog_warning("In-place normalization requires src_dt == dst_dt, but got ",
                          datatypeToStr(cfg.src_dt), " != ", datatypeToStr(cfg.dst_dt),
                          ". Falling back to out-of-place.");
        cfg.isInplace = false;
      }

      configs.push_back(cfg);
    }
    catch (const std::exception &e) {
      commonlog_error(e.what());
      continue;
    }
  }
}

void log_benchmark_failure(const NormalizationConfig &cfg) {
  testlog_error("Benchmark failed for ", cfg.norm_type, ", ",
                shapeToStr(cfg.shape), ", ", cfg.norm_ndims, ", ",
                datatypeToStr(cfg.src_dt), ":", datatypeToStr(cfg.dst_dt), ", ",
                cfg.epsilon, ", ", cfg.use_scale, ", ", cfg.use_shift, ", ",
                cfg.iters, ", ", cfg.warmup_iters, ", ",
                datatypeToStr(cfg.gamma_dt), ", ", datatypeToStr(cfg.beta_dt), ", ",
                cfg.algorithm, ", ",
                "num_threads=", cfg.num_threads, ", ",
                "isInplace=", cfg.isInplace);
}

void print_results(std::vector<std::pair<NormalizationConfig, TimingStats>>
                   &normalization_results, std::ostream &outfile) {
  std::vector<std::string> headers = {
    "Norm_Type", "Shape", "Norm_Ndims", "Data_Type", "Epsilon",
    "Use_Scale", "Use_Shift", "Iterations", "Warmup_Iters",
    "Gamma_DT", "Beta_DT", "Algorithm", "Num_Threads", "Inplace",
    "Total_time(ms) (all iters)", "Avg_time(ms)"
  };
  std::vector<size_t> col_widths(headers.size());
  for (size_t i = 0; i < headers.size(); ++i) {
    col_widths[i] = headers[i].size() + 2;
  }

  for (const auto &result : normalization_results) {
    const auto &config = result.first;
    const auto &stat = result.second;
    int col = 0;
    col_widths[col] = std::max(col_widths[col], config.norm_type.size() + 2);
    col++;
    col_widths[col] = std::max(col_widths[col],
                               shapeToStr(config.shape).size() + 2);
    col++;
    col_widths[col] = std::max(col_widths[col],
                               std::to_string(config.norm_ndims).size() + 2);
    col++;
    std::string dt_str = datatypeToStr(config.src_dt) + ":" +
                         datatypeToStr(config.dst_dt);
    col_widths[col] = std::max(col_widths[col], dt_str.size() + 2);
    col++;
    std::ostringstream eps_ss;
    eps_ss << config.epsilon;
    col_widths[col] = std::max(col_widths[col], eps_ss.str().size() + 2);
    col++;
    col_widths[col] = std::max(col_widths[col],
                               std::string(config.use_scale ? "true" : "false").size() + 2);
    col++;
    col_widths[col] = std::max(col_widths[col],
                               std::string(config.use_shift ? "true" : "false").size() + 2);
    col++;
    col_widths[col] = std::max(col_widths[col],
                               std::to_string(config.iters).size() + 2);
    col++;
    col_widths[col] = std::max(col_widths[col],
                               std::to_string(config.warmup_iters).size() + 2);
    col++;
    col_widths[col] = std::max(col_widths[col],
                               datatypeToStr(config.gamma_dt).size() + 2);
    col++;
    col_widths[col] = std::max(col_widths[col],
                               datatypeToStr(config.beta_dt).size() + 2);
    col++;
    col_widths[col] = std::max(col_widths[col], config.algorithm.size() + 2);
    col++;
    std::string threads_str = (config.num_threads == 0) ?
                              "auto" : std::to_string(config.num_threads);
    col_widths[col] = std::max(col_widths[col], threads_str.size() + 2);
    col++;
    col_widths[col] = std::max(col_widths[col],
                               std::string(config.isInplace ? "true" : "false").size() + 2);
    col++;
    std::ostringstream total_time_ss;
    total_time_ss << std::fixed << std::setprecision(2) << stat.total_time_ms;
    col_widths[col] = std::max(col_widths[col], total_time_ss.str().size() + 2);
    col++;
    double avg_time = (config.iters > 0) ?
                      stat.total_time_ms / config.iters : 0.0;
    std::ostringstream avg_time_ss;
    avg_time_ss << std::fixed << std::setprecision(6) << avg_time;
    col_widths[col] = std::max(col_widths[col], avg_time_ss.str().size() + 2);
    col++;
  }

  auto print_row = [&](const std::vector<std::string> &row) {
    for (size_t i = 0; i < row.size(); ++i) {
      outfile << std::setw(col_widths[i]) << row[i];
    }
    outfile << std::endl;
  };

  outfile << std::fixed << std::setprecision(2);
  outfile << std::left;
  print_row(headers);

  for (const auto &result : normalization_results) {
    const auto &config = result.first;
    const auto &stat = result.second;
    std::vector<std::string> row;
    row.push_back(config.norm_type);
    row.push_back(shapeToStr(config.shape));
    row.push_back(std::to_string(config.norm_ndims));
    row.push_back(datatypeToStr(config.src_dt) + ":" +
                  datatypeToStr(config.dst_dt));
    std::ostringstream eps_ss;
    eps_ss << config.epsilon;
    row.push_back(eps_ss.str());
    row.push_back(config.use_scale ? "true" : "false");
    row.push_back(config.use_shift ? "true" : "false");
    row.push_back(std::to_string(config.iters));
    row.push_back(std::to_string(config.warmup_iters));
    row.push_back(datatypeToStr(config.gamma_dt));
    row.push_back(datatypeToStr(config.beta_dt));
    row.push_back(config.algorithm);
    row.push_back((config.num_threads == 0) ?
                  "auto" : std::to_string(config.num_threads));
    row.push_back(config.isInplace ? "true" : "false");
    std::ostringstream total_time_ss;
    total_time_ss << std::fixed << std::setprecision(2) << stat.total_time_ms;
    row.push_back(total_time_ss.str());
    double avg_time = (config.iters > 0) ?
                      stat.total_time_ms / config.iters : 0.0;
    std::ostringstream avg_time_ss;
    avg_time_ss << std::fixed << std::setprecision(6) << avg_time;
    row.push_back(avg_time_ss.str());
    print_row(row);
  }
}

void log_results(std::vector<std::pair<NormalizationConfig, TimingStats>>
                 &normalization_results, std::ostream &outfile) {
  outfile <<
          "Norm_Type, Shape, Norm_Ndims, Data_Type, Epsilon, "
          "Use_Scale, Use_Shift, Iterations, Warmup_Iters, "
          "Gamma_DT, Beta_DT, Algorithm, Num_Threads, Inplace, "
          "Total_time(ms) (all iters), Avg_time(ms)" << std::endl;

  for (const auto &result : normalization_results) {
    const auto &config = result.first;
    const auto &stat = result.second;
    outfile <<
            config.norm_type << ", " <<
            shapeToStr(config.shape) << ", " <<
            config.norm_ndims << ", " <<
            datatypeToStr(config.src_dt) << ":" << datatypeToStr(config.dst_dt) << ", " <<
            config.epsilon << ", " <<
            (config.use_scale ? "true" : "false") << ", " <<
            (config.use_shift ? "true" : "false") << ", " <<
            config.iters << ", " <<
            config.warmup_iters << ", " <<
            datatypeToStr(config.gamma_dt) << ", " <<
            datatypeToStr(config.beta_dt) << ", " <<
            config.algorithm << ", " <<
            config.num_threads << ", " <<
            (config.isInplace ? "true" : "false") << ", " <<
            stat.total_time_ms << ", " <<
            ((config.iters > 0) ? stat.total_time_ms / config.iters : 0.0) <<
            std::endl;
  }
}

} // namespace normalization
} // namespace benchdnn
} // namespace zendnnl
