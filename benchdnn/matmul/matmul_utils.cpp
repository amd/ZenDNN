/********************************************************************************
# * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "matmul_utils.hpp"

namespace zendnnl {
namespace benchdnn {
namespace matmul {

void inputFileParser(std::ifstream &infile, std::vector<MatmulConfig> &configs,
                     bool &isPipeline, const global_options &options) {
  std::string line;

  // Parse each line of the input file into a MatmulConfig object
  while (std::getline(infile, line)) {
    if (line.empty()) {
      continue;
    }

    // Split the line into fields and validate
    auto fields = split(line, ',');
    int fields_size = fields.size();
    // If last field starts with a digit, decrease fields_size
    if (!fields.empty() && std::isdigit(fields.back()[0])) {
      fields_size--;
    }
    int expected_fields_cnt = options.ndims + 1 + MATMUL_EXTRA_INPUT_FIELD_COUNT;
    if (fields_size != expected_fields_cnt) {
      commonlog_error(
        "Invalid line (expected ", expected_fields_cnt, " fields): [",
        (options.ndims > 2) ? "bs, " : "",
        "m, k, n, iterations, input_dtype:weights_dtype:output_dtype, isBiasEnabled, bias_dtype, postOp, ",
        "kernel name, isTransA, isTransB, warmup iterations (optional)]");
      continue;
    }
    MatmulConfig cfg;
    try {
      int id = 0;
      if (options.ndims > 2) {
        if (fields[id].empty() || std::stoi(fields[id]) <= 0) {
          commonlog_error("BS value cannot be empty or <= 0. Please provide a valid number.");
          continue;
        }
        cfg.bs = std::stoi(fields[id++]);
      }
      else {
        cfg.bs = 1;
      }
      if (fields[id].empty() || std::stoi(fields[id]) <= 0) {
        commonlog_error("M value cannot be empty or <= 0. Please provide a valid number.");
        continue;
      }
      cfg.m = std::stoi(fields[id++]);
      if (fields[id].empty() || std::stoi(fields[id]) <= 0) {
        commonlog_error("K value cannot be empty or <= 0. Please provide a valid number.");
        continue;
      }
      cfg.k = std::stoi(fields[id++]);
      if (fields[id].empty()) {
        commonlog_error("N values cannot be empty. Please provide a valid number.");
        continue;
      }
      auto n_values = split(fields[id++], ':');
      for (const auto &n : n_values) {
        if (n.empty() || std::stoi(n) <= 0) {
          commonlog_error("One of the n values is empty or <= 0. Please provide a valid value.");
          continue;
        }
        cfg.n_values.push_back(std::stoi(n));
      }
      // Set isPipeline to true if more than one n value is present
      isPipeline = (n_values.size() > 1) ? true : isPipeline;
      if (fields[id].empty()) {
        commonlog_error("Field for iterations is empty. Please provide a value.");
        continue;
      }
      cfg.iters = std::stoi(fields[id++]);
      // Parse data types (input:weights:output)
      auto dt = split(fields[id++], ':');
      if (fields[id - 1].size() > 0) {
        auto i = 0;
        for (; i < dt.size(); i++) {
          cfg.dt.push_back(strToDatatype(dt[i]));
        }
        for (; i < 3; i++) {
          cfg.dt.push_back(data_type_t::f32);
        }
        if (dt.size() < 3) {
          commonlog_warning("Less than 3 data types specified. Defaulting missing types to f32.");
        }
      }
      else {
        cfg.dt.push_back(data_type_t::f32);
        cfg.dt.push_back(data_type_t::f32);
        cfg.dt.push_back(data_type_t::f32);
        commonlog_warning("No data types specified. Defaulting all to f32.");
      }
      // Parse bias flag and bias data type
      if (fields[id].empty()) {
        commonlog_error("Field for isBiasEnabled is empty. Please provide a value.");
        continue;
      }
      std::string bias_flag = fields[id];
      std::transform(bias_flag.begin(), bias_flag.end(), bias_flag.begin(),
                     ::tolower);
      if (bias_flag == "true" || bias_flag == "1") {
        cfg.isBiasEnabled = true;
      }
      else if (bias_flag == "false" || bias_flag == "0") {
        cfg.isBiasEnabled = false;
      }
      else {
        commonlog_error("Invalid value for isBiasEnabled. Use true/false or 1/0.");
        continue;
      }
      id++;
      if (cfg.isBiasEnabled) {
        if (!fields[id++].empty()) {
          cfg.bias_dt = strToDatatype(fields[id - 1]);
        }
        else {
          commonlog_warning("No data type specified for bias. Defaulting it to f32.");
          cfg.bias_dt = data_type_t::f32;
        }
      }
      else {
        id++; // Skip bias data type field if bias is not enabled
      }
      // Parse post-operations (e.g., relu, gelu, binary ops)
      auto postOps = split(fields[id++], ':');
      auto binary_post_op_pos = 0;
      for (auto i = 0; i < postOps.size(); i++) {
        if (!postOps[i].empty()) {
          cfg.post_ops.push_back(strToPostOps(postOps[i]));
          // Track positions of binary post-operations
          if (postOps[i].find("binary_") == 0) {
            cfg.binary_post_ops_pos.push_back(binary_post_op_pos);
          }
          binary_post_op_pos++;
        }
      }
      // Parse kernel name (default to 'aocl_blis' if empty)
      if (fields[id++].empty()) {
        commonlog_warning("No kernel name specified. Defaulting to 'aocl_blis'.");
        cfg.kernel_name = "aocl_blis";
      }
      else {
        cfg.kernel_name = fields[id - 1];
        if (options.ndims > 2 && (cfg.kernel_name != "aocl_blis" &&
                                  cfg.kernel_name != "onednn")) {
          commonlog_error("For BMM, kernel name should be 'aocl_blis'.");
          continue;
        }
      }

      std::string transA_flag = fields[id];
      std::transform(transA_flag.begin(), transA_flag.end(), transA_flag.begin(),
                     ::tolower);
      if (transA_flag == "true" || transA_flag == "1") {
        cfg.isTransA = true;
      }
      else if (transA_flag == "false" || transA_flag == "0") {
        cfg.isTransA = false;
      }
      else {
        commonlog_error("Invalid value for isTransA. Use true/false or 1/0.");
        continue;
      }
      id++;

      std::string transB_flag = fields[id];
      std::transform(transB_flag.begin(), transB_flag.end(), transB_flag.begin(),
                     ::tolower);
      if (transB_flag == "true" || transB_flag == "1") {
        cfg.isTransB = true;
      }
      else if (transB_flag == "false" || transB_flag == "0") {
        cfg.isTransB = false;
      }
      else {
        commonlog_error("Invalid value for isTransB. Use true/false or 1/0.");
        continue;
      }
      id++;
      // Parse warmup iterations if provided, otherwise use 20% of main iterations
      if (id < fields.size() && !(fields[id].empty())) {
        cfg.warmup_iters = std::stoi(fields[id]);
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

void inputModelFileParser(std::ifstream &infile,
                          std::vector<MatmulConfig> &configs, bool &isPipeline,
                          const global_options &options) {
  std::string line;

  // Parse each line of the input file into a MatmulConfig object
  while (std::getline(infile, line)) {
    if (line.empty()) {
      continue;
    }

    // Split the line into fields and validate
    auto fields = split(line, ',');
    int fields_size = fields.size();
    int expected_fields_cnt = options.ndims + 1; // modelname, [bs], [m], k, n
    if (fields_size > 2) {
      auto to_lower = [](const std::string &s) {
        std::string out = s;
        std::transform(out.begin(), out.end(), out.begin(), ::tolower);
        return out;
      };
      auto is_bool_str = [](const std::string &s) {
        return s == "true" || s == "1" || s == "false" || s == "0";
      };
      auto is_number_str = [](const std::string &s) {
        return !s.empty() && std::all_of(s.begin(), s.end(), ::isdigit);
      };
      std::string last = to_lower(fields[fields_size - 1]);
      std::string prev = (fields_size > 1) ? fields[fields_size - 2] : "";

      if (is_bool_str(last)) {
        // If previous is not a number, treat as trailing [postOp, isBiasEnabled] and ignore both
        if (!is_number_str(prev)) {
          fields_size -= 2;
        }
        else if (!(last == "1" || last == "0")) {
          // If last is not 1/0 and previous is a number, error
          commonlog_error(
            "Invalid input: ", line, " (expected ", expected_fields_cnt, " fields): <",
            "ModelName, ", (options.ndims > 2) ? "bs, " : "",
            "M, K, N, [postOp, isBiasEnabled]>");
          continue;
        }
        // else: last is 1/0 and previous is a number, do nothing
      }
      else if (!is_number_str(last)) {
        // If last is not a number, treat as trailing postOp and ignore it
        fields_size--;
      }
      // else: last is a number, do nothing
    }

    if (fields_size < expected_fields_cnt) {
      commonlog_error(
        "Invalid input: ", line, " (expected ", expected_fields_cnt, " fields): <",
        "ModelName, ", (options.ndims > 2) ? "bs, " : "",
        "M, K, N, [postOp, isBiasEnabled]>");
      continue;
    }
    MatmulConfig cfg;
    try {
      int id = 0;
      cfg.modelName = fields[id++];
      cfg.bs = 1;
      // Handle different field counts based on ndims
      if (fields_size == 3) {
        if (options.m <= 0) {
          commonlog_error("M value cannot be <= 0. Please provide a valid number.");
          continue;
        }
        cfg.m = options.m;
      }
      else if (fields_size == 4) {
        if (fields[id].empty() || std::stoi(fields[id]) <= 0) {
          commonlog_error("M value cannot be empty or <= 0. Please provide a valid number.");
          continue;
        }
        else {
          cfg.m = std::stoi(fields[id++]);
        }
      }
      else if (fields_size == 5) {
        if (fields[id].empty()) {
          commonlog_error("Field for bs is empty. Please provide a value.");
          continue;
        }
        else {
          cfg.bs = std::stoi(fields[id++]);
        }
        if (fields[id].empty()) {
          commonlog_error("Field for m is empty. Please provide a value.");
          continue;
        }
        else {
          cfg.m = std::stoi(fields[id++]);
        }
      }
      if (fields[id].empty()) {
        commonlog_error("Field for k is empty. Please provide a value.");
        continue;
      }
      else {
        cfg.k = std::stoi(fields[id++]);
      }
      if (fields[id].empty()) {
        commonlog_error("Field for n is empty. Please provide a value.");
        continue;
      }
      auto n_values = split(fields[id++], ':');
      for (const auto &n : n_values) {
        if (n.empty()) {
          commonlog_error("One of the n values is empty. Please provide a value.");
          continue;
        }
        cfg.n_values.push_back(std::stoi(n));
      }

      // Set isPipeline to true if more than one n value is present
      isPipeline = (n_values.size() > 1) ? true : isPipeline;

      if (id < fields.size() && !(fields[id].empty())) {
        auto postOps = split(fields[id], ':');
        auto binary_post_op_pos = 0;
        for (auto i = 0; i < postOps.size(); i++) {
          if (!postOps[i].empty()) {
            cfg.post_ops.push_back(strToPostOps(postOps[i]));
            // Track positions of binary post-operations
            if (postOps[i].find("binary_") == 0) {
              cfg.binary_post_ops_pos.push_back(binary_post_op_pos);
            }
            binary_post_op_pos++;
          }
        }
      }
      id++;

      if (id < fields.size() && !(fields[id].empty())) {
        std::string bias_flag = fields[id];
        std::transform(bias_flag.begin(), bias_flag.end(), bias_flag.begin(),
                       ::tolower);
        if (bias_flag == "true" || bias_flag == "1") {
          cfg.isBiasEnabled = true;
        }
        else if (bias_flag == "false" || bias_flag == "0") {
          cfg.isBiasEnabled = false;
        }
        else {
          commonlog_error("Invalid value for isBiasEnabled. Use true/false or 1/0.");
          continue;
        }
      }
      else {
        cfg.isBiasEnabled = false;
      }

      cfg.iters = options.iters;
      cfg.dt.push_back(options.sdt);
      cfg.dt.push_back(options.wdt);
      cfg.dt.push_back(options.ddt);
      cfg.kernel_name = options.kernel_name;
      cfg.bias_dt = options.bias_dt;
      cfg.isTransA = options.isTransA;
      cfg.isTransB = options.isTransB;
      cfg.warmup_iters = options.warmup_iters;

      configs.push_back(cfg);
    }
    catch (const std::exception &e) {
      commonlog_error(e.what());
      continue;
    }
  }
}

void inputCommandLineParser(std::vector<MatmulConfig> &configs,
                            bool &isPipeline,
                            const global_options &options) {
  MatmulConfig cfg;
  try {
    if (options.ndims > 2) {
      cfg.bs = options.bs;
    }
    else {
      cfg.bs = 1;
    }
    cfg.m = options.m;
    cfg.k = options.k;
    if (options.n_values.size() > 1) {
      isPipeline = true;
    }
    for (const auto &n : options.n_values) {
      cfg.n_values.push_back(n);
    }
    cfg.isBiasEnabled = options.isBiasEnabled;
    cfg.iters = options.iters;
    cfg.dt.push_back(options.sdt);
    cfg.dt.push_back(options.wdt);
    cfg.dt.push_back(options.ddt);
    cfg.kernel_name = options.kernel_name;
    cfg.bias_dt = options.bias_dt;
    cfg.isTransA = options.isTransA;
    cfg.isTransB = options.isTransB;
    cfg.warmup_iters = options.warmup_iters;

    configs.push_back(cfg);
  }
  catch (const std::exception &e) {
    commonlog_error(e.what());
  }
}

void log_benchmark_failure(const MatmulConfig &cfg) {
  std::string post_op = "";
  if (!cfg.post_ops.empty()) {
    for (auto j = 0; j < cfg.post_ops.size(); j++) {
      post_op += (j > 0 ? ":" : "") + postOpsToStr(cfg.post_ops[j]);
    }
  }
  std::string n_values = "";
  for (auto i = 0; i < cfg.n_values.size(); i++) {
    n_values += (i > 0 ? ":" : "") + std::to_string(cfg.n_values[i]);
  }
  commonlog_error("Benchmark failed for ", cfg.m, ", ", cfg.k, ", ",
                  n_values, ", ", datatypeToStr(cfg.dt[0]), ":",
                  datatypeToStr(cfg.dt[1]), ":", datatypeToStr(cfg.dt[2]), ", ",
                  cfg.isBiasEnabled, ", ", (cfg.isBiasEnabled ? datatypeToStr(cfg.bias_dt) :""),
                  ", ", post_op, ", ", cfg.kernel_name, ", ", cfg.warmup_iters);
}

void print_matmul_execution_summary(const MatmulConfig &cfg,
                                    const std::vector<TimingStats> &time_stats_layer,
                                    const global_options &options) {
  std::string post_op = "";
  if (!cfg.post_ops.empty()) {
    for (auto j = 0; j < cfg.post_ops.size(); j++) {
      post_op += (j > 0 ? ":" : "") + postOpsToStr(cfg.post_ops[j]);
    }
  }
  std::string n_values = "";
  double total_time = 0.0;
  for (auto i = 0; i < cfg.n_values.size(); i++) {
    n_values += (i > 0 ? ":" : "") + std::to_string(cfg.n_values[i]);
    total_time += time_stats_layer[i].total_time_ms;
  }
  if (options.ndims > 2) {
    std::cout << cfg.bs << ", ";
  }
  std::cout << cfg.m << ", " << cfg.k << ", " << n_values << ", "
            << cfg.iters << ", "
            << datatypeToStr(cfg.dt[0]) << ":" << datatypeToStr(cfg.dt[1])
            << ":" << datatypeToStr(cfg.dt[2]) << ", "
            << cfg.isBiasEnabled << ", "
            << (cfg.isBiasEnabled ? datatypeToStr(cfg.bias_dt) : "")
            << ", " << post_op << ", " << cfg.kernel_name << ", "
            << cfg.isTransA << ", "
            << cfg.isTransB << ", "
            << cfg.warmup_iters << ", "
            << total_time << std::endl;
}

void write_each_config_result(const MatmulConfig &config,
                              const std::vector<TimingStats> &stat, std::ostream &outfile,
                              const bool isLOWOHA,
                              int layer_num,
                              double percentage, bool isPipeline) {

  size_t m = config.m;
  size_t k = (layer_num == 0) ? config.k : config.n_values[layer_num - 1];
  size_t n = config.n_values[layer_num];
  size_t bs = config.bs;
  double gops = (2 * bs * m * k * n * 0.000000001);
  double gflops_val = (gops / (stat[layer_num].total_time_ms / config.iters)) *
                      1000;
  outfile << m << ", " << k << ", " << n;
  outfile << ", " << config.iters << ", "
          << datatypeToStr(config.dt[0]) << ":"
          << datatypeToStr(config.dt[1]) << ":"
          << datatypeToStr(config.dt[2]) << ", "
          << config.isBiasEnabled << ", "
          << (config.isBiasEnabled ? datatypeToStr(config.bias_dt) : "") << ", ";
  if (!config.post_ops.empty()) {
    outfile << postOpsToStr(config.post_ops[0]);
    for (size_t j = 1; j < config.post_ops.size(); ++j) {
      outfile << ":" << postOpsToStr(config.post_ops[j]);
    }
  }
  outfile << ", ";
  outfile << config.kernel_name << ", "
          << config.isTransA << ", "
          << config.isTransB << ", "
          << config.warmup_iters << ", " << stat[layer_num].total_time_ms
          << ", " << gflops_val;
  if (isPipeline) {
    outfile << ", " << percentage;
  }
#if MEASURE_INDIVIDUAL_TIMINGS
  if (!isLOWOHA) {
    double ctx_creation_percentage = (stat[layer_num].context_creation_ms /
                                      stat[layer_num].total_time_ms) * 100;
    double op_creation_percentage = (stat[layer_num].operator_creation_ms /
                                     stat[layer_num].total_time_ms) * 100;
    double op_execution_percentage = (stat[layer_num].operator_execution_ms /
                                      stat[layer_num].total_time_ms) * 100;
    outfile << ", "
            << stat[layer_num].context_creation_ms << " ("
            << ctx_creation_percentage << " %), "
            << stat[layer_num].operator_creation_ms << " ("
            << op_creation_percentage << " %), "
            << stat[layer_num].operator_execution_ms << " ("
            << op_execution_percentage << " %)";
  }
#endif
  outfile << std::endl;
}

void cal_column_width(const MatmulConfig &config,
                      const std::vector<TimingStats> &stat, std::vector<size_t> &col_widths,
                      int st_index, const bool isLOWOHA, int layer_num, double percentage,
                      bool isPipeline) {
  size_t m = config.m;
  size_t k = ((layer_num == 0) ? config.k : config.n_values[layer_num - 1]);
  size_t n = config.n_values[layer_num];
  size_t bs = config.bs;
  double gops = (2 * bs * m * k * n * 0.000000001);
  double gflops_val = (gops / (stat[layer_num].total_time_ms / config.iters)) *
                      1000;
  int col = st_index;
  col_widths[col++] = std::max(col_widths[col], std::to_string(m).size() + 2);
  col_widths[col++] = std::max(col_widths[col], std::to_string(k).size() + 2);
  col_widths[col++] = std::max(col_widths[col], std::to_string(n).size() + 2);
  col_widths[col++] = std::max(col_widths[col],
                               std::to_string(config.iters).size() + 2);
  std::string dt_str = datatypeToStr(config.dt[0]) + ":" + datatypeToStr(
                         config.dt[1]) + ":" + datatypeToStr(config.dt[2]);
  col_widths[col++] = std::max(col_widths[col], dt_str.size() + 2);
  col_widths[col++] = std::max(col_widths[col],
                               std::to_string(config.isBiasEnabled).size() + 2);
  std::string bias_dt_str = config.isBiasEnabled ? datatypeToStr(
                              config.bias_dt) : "";
  col_widths[col++] = std::max(col_widths[col], bias_dt_str.size() + 2);
  std::string postop_str;
  if (!config.post_ops.empty()) {
    postop_str += postOpsToStr(config.post_ops[0]);
    for (size_t j = 1; j < config.post_ops.size(); ++j) {
      postop_str += ":" + postOpsToStr(config.post_ops[j]);
    }
  }
  col_widths[col++] = std::max(col_widths[col], postop_str.size() + 2);
  col_widths[col++] = std::max(col_widths[col], config.kernel_name.size() + 2);
  col_widths[col++] = std::max(col_widths[col],
                               std::to_string(config.isTransA).size() + 2);
  col_widths[col++] = std::max(col_widths[col],
                               std::to_string(config.isTransB).size() + 2);
  col_widths[col++] = std::max(col_widths[col],
                               std::to_string(config.warmup_iters).size() + 2);
  col_widths[col++] = std::max(col_widths[col],
                               std::to_string((int)stat[0].total_time_ms).size() + 2);
  std::ostringstream gflops_ss;
  gflops_ss << std::fixed << std::setprecision(2) << gflops_val;
  col_widths[col++] = std::max(col_widths[col], gflops_ss.str().size() + 2);
  if (isPipeline) {
    std::ostringstream perc_ss;
    perc_ss << std::fixed << std::setprecision(2) << percentage << " %";
    col_widths[col++] = std::max(col_widths[col], perc_ss.str().size() + 2);
  }
#if MEASURE_INDIVIDUAL_TIMINGS
  if (!isLOWOHA) {
    std::ostringstream ctx_str, op_create_str, op_exec_str;
    double ctx_creation_percentage = (stat[0].context_creation_ms /
                                      stat[0].total_time_ms) * 100;
    double op_creation_percentage = (stat[0].operator_creation_ms /
                                     stat[0].total_time_ms) * 100;
    double op_execution_percentage = (stat[0].operator_execution_ms /
                                      stat[0].total_time_ms) * 100;
    ctx_str << std::fixed << std::setprecision(2)
            << stat[0].context_creation_ms << " (" << ctx_creation_percentage << " %)";
    op_create_str << std::fixed << std::setprecision(2)
                  << stat[0].operator_creation_ms << " (" << op_creation_percentage << " %)";
    op_exec_str << std::fixed << std::setprecision(2)
                << stat[0].operator_execution_ms << " (" << op_execution_percentage << " %)";
    col_widths[col++] = std::max(col_widths[col], ctx_str.str().size() + 2);
    col_widths[col++] = std::max(col_widths[col], op_create_str.str().size() + 2);
    col_widths[col++] = std::max(col_widths[col], op_exec_str.str().size() + 2);
  }
#endif
}

void fill_row(const MatmulConfig &config,
              const std::vector<TimingStats> &stat, std::vector<std::string> &row,
              const bool isLOWOHA, int layer_num, double percentage,
              bool isPipeline) {
  size_t m = config.m;
  size_t k = ((layer_num == 0) ? config.k : config.n_values[layer_num - 1]);
  size_t n = config.n_values[layer_num];
  size_t bs = config.bs;
  double gops = (2 * bs * m * k * n * 0.000000001);
  double gflops_val = ((gops / (stat[layer_num].total_time_ms / config.iters)) *
                       1000);
  row.push_back(std::to_string(m));
  row.push_back(std::to_string(k));
  row.push_back(std::to_string(n));
  row.push_back(std::to_string(config.iters));
  row.push_back(datatypeToStr(config.dt[0]) + ":" +
                datatypeToStr(config.dt[1]) + ":" + datatypeToStr(config.dt[2]));
  row.push_back(std::to_string(config.isBiasEnabled));
  row.push_back(config.isBiasEnabled ? datatypeToStr(config.bias_dt) : "");
  std::string postop_str;
  if (!config.post_ops.empty()) {
    postop_str += postOpsToStr(config.post_ops[0]);
    for (size_t j = 1; j < config.post_ops.size(); ++j) {
      postop_str += ":" + postOpsToStr(config.post_ops[j]);
    }
  }
  row.push_back(postop_str);
  row.push_back(config.kernel_name);
  row.push_back(std::to_string(config.isTransA));
  row.push_back(std::to_string(config.isTransB));
  row.push_back(std::to_string(config.warmup_iters));
  std::ostringstream total_time_ss;
  total_time_ss << std::fixed << std::setprecision(2) <<
                stat[layer_num].total_time_ms;
  row.push_back(total_time_ss.str());
  std::ostringstream gflops_ss;
  gflops_ss << std::fixed << std::setprecision(2) << gflops_val;
  row.push_back(gflops_ss.str());
  if (isPipeline) {
    std::ostringstream perc_ss;
    perc_ss << std::fixed << std::setprecision(2) << percentage << " %";
    row.push_back(perc_ss.str());
  }
#if MEASURE_INDIVIDUAL_TIMINGS
  if (!isLOWOHA) {
    std::ostringstream ctx_str, op_create_str, op_exec_str;
    double ctx_creation_percentage = (stat[layer_num].context_creation_ms /
                                      stat[layer_num].total_time_ms) * 100;
    double op_creation_percentage = (stat[layer_num].operator_creation_ms /
                                     stat[layer_num].total_time_ms) * 100;
    double op_execution_percentage = (stat[layer_num].operator_execution_ms /
                                      stat[layer_num].total_time_ms) * 100;
    ctx_str << std::fixed << std::setprecision(2)
            << stat[layer_num].context_creation_ms << " ("
            << ctx_creation_percentage << " %)";
    op_create_str << std::fixed << std::setprecision(2)
                  << stat[layer_num].operator_creation_ms << " ("
                  << op_creation_percentage << " %)";
    op_exec_str << std::fixed << std::setprecision(2)
                << stat[layer_num].operator_execution_ms << " ("
                << op_execution_percentage << " %)";
    row.push_back(ctx_str.str());
    row.push_back(op_create_str.str());
    row.push_back(op_exec_str.str());
  }
#endif
}

void log_pipeline_results(
  std::vector<std::pair<MatmulConfig, std::vector<TimingStats>>> &matmul_results,
  std::ostream &outfile, const global_options &options,
  const InputMode inputMode) {

  outfile << std::fixed << std::setprecision(2);
  outfile << "Layer Number, ";
  if (inputMode == InputMode::MODEL) {
    outfile << "Model Name, ";
  }
  if (options.ndims > 2) {
    outfile << "BS, ";
  }
  outfile <<
          "M, K, N, Iterations, Data type, Bias Enabled, Bias Data type, Post Operation, Kernel name, isTransA, isTransB, Warmup iterations, Total time (ms) (all iters), GFLOPS, % of Total";
#if MEASURE_INDIVIDUAL_TIMINGS
  outfile <<
          ", Context Creation (ms & %), Operator Creation (ms & %), Operator Execution (ms & %)";
#endif
  outfile << std::endl;

  // Write results to CSV for each configuration
  for (const auto &result : matmul_results) {
    const MatmulConfig &config = result.first;
    const std::vector<TimingStats> &stat = result.second;
    double total_time = 0.0;
    for (auto i = 0; i < config.n_values.size(); i++) {
      total_time += stat[i].total_time_ms;
    }
    outfile << "Summary, ";
    if (inputMode == InputMode::MODEL) {
      outfile << config.modelName << ", ";
    }
    if (options.ndims > 2) {
      outfile << config.bs << ", ";
    }
    outfile << config.m << ", " << config.k << ", ";
    // Output N values separated by ':'
    if (!config.n_values.empty()) {
      outfile << config.n_values[0];
      for (size_t i = 1; i < config.n_values.size(); ++i) {
        outfile << ":" << config.n_values[i];
      }
    }
    outfile << ", " <<
            config.iters << ", " <<
            datatypeToStr(config.dt[0]) << ":" <<
            datatypeToStr(config.dt[1]) << ":" <<
            datatypeToStr(config.dt[2]) << ", " <<
            config.isBiasEnabled << ", " <<
            (config.isBiasEnabled ? datatypeToStr(config.bias_dt) : "") << ", ";
    if (!config.post_ops.empty()) {
      outfile << postOpsToStr(config.post_ops[0]);
      for (size_t j = 1; j < config.post_ops.size(); ++j) {
        outfile << ":" << postOpsToStr(config.post_ops[j]);
      }
    }
    outfile << ", ";
    outfile << config.kernel_name << ", "
            << config.isTransA << ", "
            << config. isTransB << ", "
            << config.warmup_iters << ", " << total_time;
    outfile << std::endl;

    for (auto i = 0; i < stat.size(); i++) {
      double percentage = (stat[i].total_time_ms / total_time) * 100;
      outfile << "Layer " << i << ", ";
      if (inputMode == InputMode::MODEL) {
        outfile << config.modelName << ", ";
      }
      if (options.ndims > 2) {
        outfile << config.bs << ", ";
      }
      write_each_config_result(config, stat, outfile, false, i, percentage, true);
    }
  }
}

void print_pipeline_results(
  std::vector<std::pair<MatmulConfig, std::vector<TimingStats>>> &matmul_results,
  std::ostream &outfile, const global_options &options,
  const InputMode inputMode) {

  // Dynamic column widths calculation
  std::vector<std::string> headers = { "Layer" };
  if (inputMode == InputMode::MODEL) {
    headers.push_back("Model Name");
  }
  if (options.ndims > 2) {
    headers.push_back("BS");
  }
  headers.insert(headers.end(), {
    "M", "K", "N", "Iters", "Data_type", "Bias_Enabled", "Bias_dt", "PostOp", "Kernel_Name",
    "isTransA", "isTransB",
    "Warmup_iters", "Total_time(ms, all iters)", "GFLOPS", "%_of_Total"
  });
#if MEASURE_INDIVIDUAL_TIMINGS
  headers.push_back("Ctx_Creation(ms_%)");
  headers.push_back("Op_Creation(ms_%)");
  headers.push_back("Op_Execution(ms_%)");
#endif
  std::vector<size_t> col_widths(headers.size());
  // Initialize with header lengths
  for (size_t i = 0; i < headers.size(); ++i) {
    col_widths[i] = headers[i].size() + 2;
  }
  // Compute max width for each column based on all data rows
  for (const auto &result : matmul_results) {
    const MatmulConfig &config = result.first;
    const std::vector<TimingStats> &stat = result.second;
    double total_time = 0.0;
    for (auto i = 0; i < config.n_values.size(); i++) {
      total_time += stat[i].total_time_ms;
    }
    // Update column widths for summary and per-layer rows
    int col = 0;
    col_widths[col++] = std::max(col_widths[col],
                                 std::string("Summary").size() + 2);
    if (inputMode == InputMode::MODEL) {
      col_widths[col++] = std::max(col_widths[col],
                                   std::string("Model Name").size() + 2);
    }
    if (options.ndims > 2) {
      col_widths[col++] = std::max(col_widths[col],
                                   std::string("BS").size() + 2);
    }
    col_widths[col++] = std::max(col_widths[col],
                                 std::to_string(config.m).size() + 2);
    col_widths[col++] = std::max(col_widths[col],
                                 std::to_string(config.k).size() + 2);
    // N field (colon separated)
    std::string n_str;
    if (!config.n_values.empty()) {
      n_str += std::to_string(config.n_values[0]);
      for (size_t i = 1; i < config.n_values.size(); ++i) {
        n_str += ":" + std::to_string(config.n_values[i]);
      }
    }
    col_widths[col++] = std::max(col_widths[col], n_str.size() + 2);
    col_widths[col++] = std::max(col_widths[col],
                                 std::to_string(config.iters).size() + 2);
    std::string dt_str = datatypeToStr(config.dt[0]) + ":" + datatypeToStr(
                           config.dt[1]) + ":" + datatypeToStr(config.dt[2]);
    col_widths[col++] = std::max(col_widths[col], dt_str.size() + 2);
    col_widths[col++] = std::max(col_widths[col],
                                 std::to_string(config.isBiasEnabled).size() + 2);
    std::string bias_dt_str = config.isBiasEnabled ? datatypeToStr(
                                config.bias_dt) : "";
    col_widths[col++] = std::max(col_widths[col], bias_dt_str.size() + 2);
    std::string postop_str;
    if (!config.post_ops.empty()) {
      postop_str += postOpsToStr(config.post_ops[0]);
      for (size_t j = 1; j < config.post_ops.size(); ++j) {
        postop_str += ":" + postOpsToStr(config.post_ops[j]);
      }
    }
    col_widths[col++] = std::max(col_widths[col], postop_str.size() + 2);
    col_widths[col++] = std::max(col_widths[col], config.kernel_name.size() + 2);
    col_widths[col++] = std::max(col_widths[col],
                                 std::to_string(config.isTransA).size() + 2);
    col_widths[col++] = std::max(col_widths[col],
                                 std::to_string(config.isTransB).size() + 2);
    col_widths[col++] = std::max(col_widths[col],
                                 std::to_string(config.warmup_iters).size() + 2);
    col_widths[col++] = std::max(col_widths[col],
                                 std::to_string((int)total_time).size() + 2);
    col_widths[col++] = std::max(col_widths[col], std::string("GFLOPS").size() + 2);
    col_widths[col++] = std::max(col_widths[col],
                                 std::string("%_of_Total").size() + 2);
#if MEASURE_INDIVIDUAL_TIMINGS
    col_widths[col++] = std::max(col_widths[col],
                                 std::string("Ctx_Creation(ms_%)").size() + 2);
    col_widths[col++] = std::max(col_widths[col],
                                 std::string("Op_Creation(ms_%)").size() + 2);
    col_widths[col++] = std::max(col_widths[col],
                                 std::string("Op_Execution(ms_%)").size() + 2);
#endif
    // Per-layer rows
    for (auto i = 0; i < stat.size(); i++) {
      double percentage = (stat[i].total_time_ms / total_time) * 100;
      int st_index = 0;

      std::string layer_str = "Layer_" + std::to_string(i);
      col_widths[st_index] = std::max(col_widths[st_index], layer_str.size() + 2);
      st_index++;
      if (inputMode == InputMode::MODEL) {
        col_widths[st_index] = std::max(col_widths[st_index],
                                        std::string("Model Name").size() + 2);
        st_index++;
      }
      if (options.ndims > 2) {
        col_widths[st_index] = std::max(col_widths[st_index],
                                        std::string("BS").size() + 2);
        st_index++;
      }
      cal_column_width(config, stat, col_widths, st_index, false, i, percentage,
                       true);
    }
  }

  // Helper lambda to print a row
  auto print_row = [&](const std::vector<std::string> &row) {
    for (size_t i = 0; i < row.size(); ++i) {
      outfile << std::setw(col_widths[i]) << row[i];
    }
    outfile << std::endl;
  };

  // Print table header
  outfile << std::fixed << std::setprecision(2);
  outfile << std::left;
  print_row(headers);

  // Print summary and per-layer rows for each configuration
  for (const auto &result : matmul_results) {
    const MatmulConfig &config = result.first;
    const std::vector<TimingStats> &stat = result.second;
    double total_time = 0.0;
    for (auto i = 0; i < config.n_values.size(); i++) {
      total_time += stat[i].total_time_ms;
    }
    // Summary row (aggregated for the pipeline)
    std::vector<std::string> summary_row;
    summary_row.push_back("Summary");
    if (inputMode == InputMode::MODEL) {
      summary_row.push_back(config.modelName);
    }
    if (options.ndims > 2) {
      summary_row.push_back(std::to_string(config.bs));
    }
    summary_row.push_back(std::to_string(config.m));
    summary_row.push_back(std::to_string(config.k));
    // N values as colon separated string
    std::string n_str;
    if (!config.n_values.empty()) {
      n_str += std::to_string(config.n_values[0]);
      for (size_t i = 1; i < config.n_values.size(); ++i) {
        n_str += ":" + std::to_string(config.n_values[i]);
      }
    }
    summary_row.push_back(n_str);
    summary_row.push_back(std::to_string(config.iters));
    summary_row.push_back(datatypeToStr(config.dt[0]) + ":" +
                          datatypeToStr(config.dt[1]) + ":" + datatypeToStr(config.dt[2]));
    summary_row.push_back(std::to_string(config.isBiasEnabled));
    summary_row.push_back(config.isBiasEnabled ?
                          datatypeToStr(config.bias_dt) : "");
    std::string postop_str;
    if (!config.post_ops.empty()) {
      postop_str += postOpsToStr(config.post_ops[0]);
      for (size_t j = 1; j < config.post_ops.size(); ++j) {
        postop_str += ":" + postOpsToStr(config.post_ops[j]);
      }
    }
    summary_row.push_back(postop_str);
    summary_row.push_back(config.kernel_name);
    summary_row.push_back(std::to_string(config.isTransA));
    summary_row.push_back(std::to_string(config.isTransB));
    summary_row.push_back(std::to_string(config.warmup_iters));
    std::ostringstream total_time_oss;
    total_time_oss << std::fixed << std::setprecision(2) << total_time;
    summary_row.push_back(total_time_oss.str());
    summary_row.push_back("");
    summary_row.push_back("");
#if MEASURE_INDIVIDUAL_TIMINGS
    summary_row.push_back("");
    summary_row.push_back("");
    summary_row.push_back("");
#endif
    print_row(summary_row);

    // Per-layer rows (detailed timing for each layer in the pipeline)
    for (auto i = 0; i < stat.size(); i++) {
      double percentage = (stat[i].total_time_ms / total_time) * 100;
      std::vector<std::string> layer_row;
      layer_row.push_back("Layer_" + std::to_string(i));
      if (inputMode == InputMode::MODEL) {
        layer_row.push_back(config.modelName);
      }
      if (options.ndims > 2) {
        layer_row.push_back(std::to_string(config.bs));
      }
      fill_row(config, stat, layer_row, false, i, percentage, true);
      print_row(layer_row);
    }
  }
}

void log_results(
  std::vector<std::pair<MatmulConfig, std::vector<TimingStats>>> &matmul_results,
  std::ostream &outfile,  const global_options &options, const bool isLOWOHA,
  const InputMode inputMode) {

  outfile << std::fixed << std::setprecision(2);
  if (inputMode == InputMode::MODEL) {
    outfile << "Model_Name, ";
  }
  if (options.ndims > 2) {
    outfile << "BS, ";
  }
  outfile <<
          "M, K, N, Iterations, Data type, Bias Enabled, Bias Data type, Post Operation, Kernel name, isTransA, isTransB, Warmup iterations, Total time (ms) (all iters), GFLOPS";
#if MEASURE_INDIVIDUAL_TIMINGS
  if (!isLOWOHA) {
    outfile <<
            ", Context Creation (ms & %), Operator Creation (ms & %), Operator Execution (ms & %)";
  }
#endif
  outfile << std::endl;

  // Write results to CSV for each configuration
  for (const auto &result : matmul_results) {
    const MatmulConfig &config = result.first;
    const std::vector<TimingStats> &stat = result.second;
    if (inputMode == InputMode::MODEL) {
      outfile << config.modelName << ", ";
    }
    if (options.ndims > 2) {
      outfile << config.bs << ", ";
    }
    write_each_config_result(config, stat, outfile, isLOWOHA);
  }
}

void print_results(
  std::vector<std::pair<MatmulConfig, std::vector<TimingStats>>> &matmul_results,
  std::ostream &outfile, const global_options &options, const bool isLOWOHA,
  const InputMode inputMode) {

  // Dynamic column widths calculation
  std::vector<std::string> headers;
  if (inputMode == InputMode::MODEL) {
    headers.insert(headers.begin(), "Model_Name");
  }
  if (options.ndims > 2) {
    headers.insert(headers.end(), "BS");
  }
  headers.insert(headers.end(), {
    "M", "K", "N", "Iters", "Data_type", "Bias_Enabled", "Bias_dt", "PostOp", "Kernel_Name",
    "isTransA", "isTransB",
    "Warmup_iters", "Total_time(ms, all iters)", "GFLOPS"
  });
#if MEASURE_INDIVIDUAL_TIMINGS
  if (!isLOWOHA) {
    headers.push_back("Ctx_Creation(ms_%)");
    headers.push_back("Op_Creation(ms_%)");
    headers.push_back("Op_Execution(ms_%)");
  }
#endif
  std::vector<size_t> col_widths(headers.size());
  // Initialize with header lengths
  for (size_t i = 0; i < headers.size(); ++i) {
    col_widths[i] = headers[i].size() + 2;
  }
  // Compute max width for each column based on all data rows
  for (const auto &result : matmul_results) {
    const MatmulConfig &config = result.first;
    const std::vector<TimingStats> &stat = result.second;
    // Column index offset for dynamic table formatting.
    int st_index = 0;
    if (inputMode == InputMode::MODEL) {
      col_widths[st_index] = std::max(col_widths[st_index],
                                      config.modelName.size() + 2);
      st_index++;
    }
    // For BMM (ndims > 2), the first column is batch size (BS), so st_index is incremented.
    if (options.ndims > 2) {
      col_widths[st_index] = std::max(col_widths[st_index],
                                      std::to_string(config.bs).size() + 2);
      st_index++;
    }
    cal_column_width(config, stat, col_widths, st_index, isLOWOHA);
  }

  // Helper lambda to print a row for the table
  auto print_row = [&](const std::vector<std::string> &row) {
    for (size_t i = 0; i < row.size(); ++i) {
      outfile << std::setw(col_widths[i]) << row[i];
    }
    outfile << std::endl;
  };

  // Print table header
  outfile << std::fixed << std::setprecision(2);
  outfile << std::left;
  print_row(headers);

  // Print each result row for every configuration
  for (const auto &result : matmul_results) {
    const MatmulConfig &config = result.first;
    const std::vector<TimingStats> &stat = result.second;
    std::vector<std::string> row;
    if (inputMode == InputMode::MODEL) {
      row.push_back(config.modelName);
    }
    if (options.ndims > 2) {
      row.push_back(std::to_string(config.bs));
    }
    fill_row(config, stat, row, isLOWOHA);
    print_row(row);
  }
}

} // namespace matmul
} // namespace benchdnn
} // namespace zendnnl