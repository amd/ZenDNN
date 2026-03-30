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

#include "reorder_utils.hpp"

namespace zendnnl {
namespace benchdnn {
namespace reorder {

reorder_algo_t strToReorderAlgo(const std::string &str) {
  std::string lower = str;
  std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
  if (lower == "dt") {
    return reorder_algo_t::DT;
  }
  if (lower == "native") {
    return reorder_algo_t::native;
  }
  if (lower == "reference") {
    return reorder_algo_t::reference;
  }
  commonlog_warning("Unknown algo '", str, "'. Defaulting to DT.");
  return reorder_algo_t::DT;
}

std::string reorderAlgoToStr(reorder_algo_t algo) {
  switch (algo) {
  case reorder_algo_t::DT:
    return "DT";
  case reorder_algo_t::native:
    return "native";
  case reorder_algo_t::reference:
    return "reference";
  default:
    return "DT";
  }
}

void inputParser(std::ifstream &infile, std::vector<ReorderConfig> &configs,
                 bool is_lowoha) {
  std::string line;

  while (std::getline(infile, line)) {
    if (line.empty()) {
      continue;
    }

    auto fields = split(line, ',');
    ReorderConfig cfg;

    if (!is_lowoha) {
      // --- Regular reorder format ---
      // rows, cols, iterations, dtype, kernel_name, isInplace [, warmup_iters]
      if (fields.size() < 6) {
        commonlog_error(
          "Invalid line (expected 6 fields): [rows, cols, iterations, dtype, "
          "kernel name, isInplace, warmup_iters (optional)]");
        continue;
      }
      try {
        cfg.rows = std::stoi(fields[0]);
        cfg.cols = std::stoi(fields[1]);
        cfg.iters = std::stoi(fields[2]);
        if (static_cast<int>(cfg.rows) <= 0 || static_cast<int>(cfg.cols) <= 0 ||
            cfg.iters <= 0) {
          commonlog_error("Invalid rows/cols/iters (must be > 0), skipping input line");
          continue;
        }
        cfg.dt = strToDatatype(fields[3]);
        if (fields[4].empty()) {
          commonlog_warning("No kernel name specified. Defaulting to 'aocl'.");
          cfg.kernel_name = "aocl";
        }
        else {
          cfg.kernel_name = fields[4];
        }
        cfg.isInplace = (fields[5] == "true");
        if (fields.size() >= 7) {
          cfg.warmup_iters = std::stoi(fields[6]);
        }
        else {
          cfg.warmup_iters = 0.2 * cfg.iters;
        }
        configs.push_back(cfg);
      }
      catch (const std::exception &e) {
        commonlog_error(e.what());
        continue;
      }
    }
    else {
      // --- LOWOHA reorder format ---
      // batch_size, rows, cols, iters, src_dtype, dst_dtype, algo,
      // scale_granularity, group_size, dynamic_quant [, num_threads] [, warmup_iters]
      if (fields.size() < 10) {
        commonlog_error(
          "Invalid LOWOHA line (expected >= 10 fields): [batch_size, rows, cols, "
          "iters, src_dtype, dst_dtype, algo, scale_granularity, group_size, "
          "dynamic_quant, num_threads (optional), warmup_iters (optional)]");
        continue;
      }
      try {
        cfg.batch_size = std::stoi(fields[0]);
        cfg.rows = std::stoi(fields[1]);
        cfg.cols = std::stoi(fields[2]);
        cfg.iters = std::stoi(fields[3]);
        if (static_cast<int>(cfg.batch_size) <= 0|| static_cast<int>(cfg.rows) <= 0 ||
            static_cast<int>(cfg.cols) <= 0 || cfg.iters <= 0) {
          commonlog_error("Invalid batch_size/rows/cols/iters (must be > 0), skipping input line");
          continue;
        }
        cfg.src_dtype = strToDatatype(fields[4]);
        cfg.dst_dtype = strToDatatype(fields[5]);
        if (fields[6].empty()) {
          commonlog_warning("No algo specified. Defaulting to 'DT'.");
          cfg.algo = "DT";
        }
        else {
          cfg.algo = fields[6];
        }
        cfg.scale_granularity = fields[7];
        int group_size_val = std::stoi(fields[8]);
        if (group_size_val < 0) {
          commonlog_error("Invalid group_size (must be >= 0), skipping input line");
          continue;
        }
        cfg.group_size = group_size_val;
        std::string dq = fields[9];
        std::transform(dq.begin(), dq.end(), dq.begin(), ::tolower);
        cfg.dynamic_quant = (dq == "true" || dq == "1");

        if (fields.size() >= 11) {
          int num_threads_val = std::stoi(fields[10]);
          if (num_threads_val < 0) {
            commonlog_error("Invalid num_threads (must be >= 0), skipping input line");
            continue;
          }
          cfg.num_threads = num_threads_val;
        }
        else {
          cfg.num_threads = 0;
        }
        if (fields.size() >= 12) {
          cfg.warmup_iters = std::stoi(fields[11]);
        }
        else {
          cfg.warmup_iters = 0.2 * cfg.iters;
        }
        configs.push_back(cfg);
      }
      catch (const std::exception &e) {
        commonlog_error(e.what());
        continue;
      }
    }
  }
}

void log_benchmark_failure(const ReorderConfig &cfg, bool is_lowoha) {
  if (!is_lowoha) {
    testlog_error("Benchmark failed for ", cfg.rows, ", ", cfg.cols, ", ",
                  cfg.iters, ", ", datatypeToStr(cfg.dt),
                  ", ", cfg.kernel_name, ", ", cfg.isInplace);
  }
  else {
    testlog_error("LOWOHA Benchmark failed for batch_size=", cfg.batch_size,
                  ", rows=", cfg.rows, ", cols=", cfg.cols,
                  ", iters=", cfg.iters,
                  ", src_dtype=", datatypeToStr(cfg.src_dtype),
                  ", dst_dtype=", datatypeToStr(cfg.dst_dtype),
                  ", algo=", cfg.algo,
                  ", scale_granularity=", cfg.scale_granularity,
                  ", group_size=", cfg.group_size,
                  ", dynamic_quant=", cfg.dynamic_quant);
  }
}

void print_results(std::vector<std::pair<ReorderConfig, TimingStats>>
                   &reorder_results, std::ostream &outfile, const bool isLOWOHA) {

  std::vector<std::string> headers;
  if (!isLOWOHA) {
    headers = {
      "Rows", "Cols", "Iterations", "Data_type", "Kernel_Name", "In-place", "Warmup_iters", "Total_time(ms)", "Avg_time(ms)"
    };
#if MEASURE_INDIVIDUAL_TIMINGS
    headers.push_back("Ctx_Creation(ms_%)");
    headers.push_back("Op_Creation(ms_%)");
    headers.push_back("Op_Execution(ms_%)");
    headers.push_back("Others(ms_%)");
#endif
  }
  else {
    headers = {
      "Batch_size", "Rows", "Cols", "Iterations", "Src_dtype", "Dst_dtype", "Algo", "Scale_granularity", "Group_size", "Dynamic_quant",
      "Num_threads", "Warmup_iters", "Total_time(ms)", "Avg_time(ms)"
    };
  }

  std::vector<size_t> col_widths(headers.size());
  for (size_t i = 0; i < headers.size(); ++i) {
    col_widths[i] = headers[i].size() + 2;
  }

  for (const auto &result : reorder_results) {
    const auto &config = result.first;
    const auto &stat = result.second;

    if (!isLOWOHA) {
      int col = 0;
      col_widths[col] = std::max(col_widths[col],
                                 std::to_string(config.rows).size() + 2);
      col++;
      col_widths[col] = std::max(col_widths[col],
                                 std::to_string(config.cols).size() + 2);
      col++;
      col_widths[col] = std::max(col_widths[col],
                                 std::to_string(config.iters).size() + 2);
      col++;
      col_widths[col] = std::max(col_widths[col],
                                 datatypeToStr(config.dt).size() + 2);
      col++;
      col_widths[col] = std::max(col_widths[col],
                                 config.kernel_name.size() + 2);
      col++;
      col_widths[col] = std::max(col_widths[col],
                                 std::to_string(config.isInplace).size() + 2);
      col++;
      col_widths[col] = std::max(col_widths[col],
                                 std::to_string(config.warmup_iters).size() + 2);
      col++;
      std::ostringstream total_time_ss;
      total_time_ss << std::fixed << std::setprecision(2) << stat.total_time_ms;
      col_widths[col] = std::max(col_widths[col],
                                 total_time_ss.str().size() + 2);
      col++;
      std::ostringstream avg_time_ss;
      avg_time_ss << std::fixed << std::setprecision(6)
                  << (stat.total_time_ms / config.iters);
      col_widths[col] = std::max(col_widths[col],
                                 avg_time_ss.str().size() + 2);
#if MEASURE_INDIVIDUAL_TIMINGS
      col++;
      std::ostringstream ctx_str, op_create_str, op_exec_str, other_str;
      double ctx_pct = (stat.context_creation_ms / stat.total_time_ms) * 100;
      double op_create_pct = (stat.operator_creation_ms / stat.total_time_ms) * 100;
      double op_exec_pct = (stat.operator_execution_ms / stat.total_time_ms) * 100;
      double other_pct = (stat.other_ms / stat.total_time_ms) * 100;
      ctx_str << std::fixed << std::setprecision(2)
              << stat.context_creation_ms << " (" << ctx_pct << " %)";
      op_create_str << std::fixed << std::setprecision(2)
                    << stat.operator_creation_ms << " (" << op_create_pct << " %)";
      op_exec_str << std::fixed << std::setprecision(2)
                  << stat.operator_execution_ms << " (" << op_exec_pct << " %)";
      other_str << std::fixed << std::setprecision(2)
                << stat.other_ms << " (" << other_pct << " %)";
      col_widths[col] = std::max(col_widths[col], ctx_str.str().size() + 2);
      col++;
      col_widths[col] = std::max(col_widths[col], op_create_str.str().size() + 2);
      col++;
      col_widths[col] = std::max(col_widths[col], op_exec_str.str().size() + 2);
      col++;
      col_widths[col] = std::max(col_widths[col], other_str.str().size() + 2);
#endif
    }
    else {
      int col = 0;
      col_widths[col] = std::max(col_widths[col],
                                 std::to_string(config.batch_size).size() + 2);
      col++;
      col_widths[col] = std::max(col_widths[col],
                                 std::to_string(config.rows).size() + 2);
      col++;
      col_widths[col] = std::max(col_widths[col],
                                 std::to_string(config.cols).size() + 2);
      col++;
      col_widths[col] = std::max(col_widths[col],
                                 std::to_string(config.iters).size() + 2);
      col++;
      col_widths[col] = std::max(col_widths[col],
                                 datatypeToStr(config.src_dtype).size() + 2);
      col++;
      col_widths[col] = std::max(col_widths[col],
                                 datatypeToStr(config.dst_dtype).size() + 2);
      col++;
      col_widths[col] = std::max(col_widths[col], config.algo.size() + 2);
      col++;
      col_widths[col] = std::max(col_widths[col],
                                 config.scale_granularity.size() + 2);
      col++;
      col_widths[col] = std::max(col_widths[col],
                                 std::to_string(config.group_size).size() + 2);
      col++;
      col_widths[col] = std::max(col_widths[col],
                                 std::to_string(config.dynamic_quant).size() + 2);
      col++;
      col_widths[col] = std::max(col_widths[col],
                                 std::to_string(config.num_threads).size() + 2);
      col++;
      col_widths[col] = std::max(col_widths[col],
                                 std::to_string(config.warmup_iters).size() + 2);
      col++;
      std::ostringstream total_time_ss;
      total_time_ss << std::fixed << std::setprecision(2) << stat.total_time_ms;
      col_widths[col] = std::max(col_widths[col], total_time_ss.str().size() + 2);
      col++;
      std::ostringstream avg_time_ss;
      avg_time_ss << std::fixed << std::setprecision(6)
                  << (stat.total_time_ms / config.iters);
      col_widths[col] = std::max(col_widths[col], avg_time_ss.str().size() + 2);
    }
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

  for (const auto &result : reorder_results) {
    const auto &config = result.first;
    const auto &stat = result.second;
    std::vector<std::string> row;

    if (!isLOWOHA) {
      row.push_back(std::to_string(config.rows));
      row.push_back(std::to_string(config.cols));
      row.push_back(std::to_string(config.iters));
      row.push_back(datatypeToStr(config.dt));
      row.push_back(config.kernel_name);
      row.push_back(std::to_string(config.isInplace));
      row.push_back(std::to_string(config.warmup_iters));
      std::ostringstream total_time_ss;
      total_time_ss << std::fixed << std::setprecision(2) << stat.total_time_ms;
      row.push_back(total_time_ss.str());
      std::ostringstream avg_time_ss;
      avg_time_ss << std::fixed << std::setprecision(6)
                  << (stat.total_time_ms / config.iters);
      row.push_back(avg_time_ss.str());
#if MEASURE_INDIVIDUAL_TIMINGS
      std::ostringstream ctx_str, op_create_str, op_exec_str, other_str;
      double ctx_pct = (stat.context_creation_ms / stat.total_time_ms) * 100;
      double op_create_pct = (stat.operator_creation_ms / stat.total_time_ms) * 100;
      double op_exec_pct = (stat.operator_execution_ms / stat.total_time_ms) * 100;
      double other_pct = (stat.other_ms / stat.total_time_ms) * 100;
      ctx_str << std::fixed << std::setprecision(2)
              << stat.context_creation_ms << " (" << ctx_pct << " %)";
      op_create_str << std::fixed << std::setprecision(2)
                    << stat.operator_creation_ms << " (" << op_create_pct << " %)";
      op_exec_str << std::fixed << std::setprecision(2)
                  << stat.operator_execution_ms << " (" << op_exec_pct << " %)";
      other_str << std::fixed << std::setprecision(2)
                << stat.other_ms << " (" << other_pct << " %)";
      row.push_back(ctx_str.str());
      row.push_back(op_create_str.str());
      row.push_back(op_exec_str.str());
      row.push_back(other_str.str());
#endif
    }
    else {
      row.push_back(std::to_string(config.batch_size));
      row.push_back(std::to_string(config.rows));
      row.push_back(std::to_string(config.cols));
      row.push_back(std::to_string(config.iters));
      row.push_back(datatypeToStr(config.src_dtype));
      row.push_back(datatypeToStr(config.dst_dtype));
      row.push_back(config.algo);
      row.push_back(config.scale_granularity);
      row.push_back(std::to_string(config.group_size));
      row.push_back(std::to_string(config.dynamic_quant));
      row.push_back(std::to_string(config.num_threads));
      row.push_back(std::to_string(config.warmup_iters));
      std::ostringstream total_time_ss;
      total_time_ss << std::fixed << std::setprecision(2) << stat.total_time_ms;
      row.push_back(total_time_ss.str());
      std::ostringstream avg_time_ss;
      avg_time_ss << std::fixed << std::setprecision(6)
                  << (stat.total_time_ms / config.iters);
      row.push_back(avg_time_ss.str());
    }
    print_row(row);
  }
}

void log_results(std::vector<std::pair<ReorderConfig, TimingStats>>
                 &reorder_results, std::ostream &outfile, const bool isLOWOHA) {

  outfile << std::fixed << std::setprecision(2);

  if (!isLOWOHA) {
    outfile <<
            "Rows, Cols, Iterations, Data type, Kernel name, In-place, Warmup iterations, Total time (ms), Avg time (ms)";
#if MEASURE_INDIVIDUAL_TIMINGS
    outfile <<
            ", Context Creation Time (ms & %), Operator Creation Time (ms & %), Operator Execution Time (ms & %), Others (ms & %)";
#endif
    outfile << std::endl;

    for (const auto &result : reorder_results) {
      const auto &config = result.first;
      const auto &stat = result.second;
      outfile <<
              config.rows << ", " <<
              config.cols << ", " <<
              config.iters << ", " <<
              datatypeToStr(config.dt) << ", " <<
              config.kernel_name << ", " <<
              config.isInplace << ", " <<
              config.warmup_iters << ", " <<
              stat.total_time_ms << ", " <<
              (stat.total_time_ms / config.iters);
#if MEASURE_INDIVIDUAL_TIMINGS
      double ctx_pct = (stat.context_creation_ms / stat.total_time_ms) * 100;
      double op_create_pct = (stat.operator_creation_ms / stat.total_time_ms) * 100;
      double op_exec_pct = (stat.operator_execution_ms / stat.total_time_ms) * 100;
      double other_pct = (stat.other_ms / stat.total_time_ms) * 100;
      outfile << ", " <<
              stat.context_creation_ms << " (" << ctx_pct << " %), " <<
              stat.operator_creation_ms << " (" << op_create_pct << " %), " <<
              stat.operator_execution_ms << " (" << op_exec_pct << " %), " <<
              stat.other_ms << " (" << other_pct << " %)";
#endif
      outfile << std::endl;
    }
  }
  else {
    outfile <<
            "Batch size, Rows, Cols, Iterations, Src dtype, Dst dtype, Algo, Scale granularity, Group size, Dynamic quant, Num threads, Warmup iterations, Total time (ms), Avg time (ms)"
            << std::endl;

    for (const auto &result : reorder_results) {
      const auto &config = result.first;
      const auto &stat = result.second;
      outfile <<
              config.batch_size << ", " <<
              config.rows << ", " <<
              config.cols << ", " <<
              config.iters << ", " <<
              datatypeToStr(config.src_dtype) << ", " <<
              datatypeToStr(config.dst_dtype) << ", " <<
              config.algo << ", " <<
              config.scale_granularity << ", " <<
              config.group_size << ", " <<
              config.dynamic_quant << ", " <<
              config.num_threads << ", " <<
              config.warmup_iters << ", " <<
              stat.total_time_ms << ", " <<
              (stat.total_time_ms / config.iters) << std::endl;
    }
  }
}

} // namespace reorder
} // namespace benchdnn
} // namespace zendnnl
