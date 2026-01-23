/********************************************************************************
# * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "embag_utils.hpp"

namespace zendnnl {
namespace benchdnn {
namespace embag {

embag_algo_t strToEmbagalgo(const std::string &algo_str) {
  if (algo_str == "sum") {
    return embag_algo_t::sum;
  }
  if (algo_str == "mean") {
    return embag_algo_t::mean;
  }
  if (algo_str == "max") {
    return embag_algo_t::max;
  }
  throw std::invalid_argument("Unknown algorithm: " + algo_str);
}

std::string embagalgoToStr(embag_algo_t algo) {
  switch (algo) {
  case embag_algo_t::sum:
    return "sum";
  case embag_algo_t::mean:
    return "mean";
  case embag_algo_t::max:
    return "max";
  default:
    throw std::invalid_argument("Unknown embag_algo_t value");
  }
}

void inputParser(std::ifstream &infile, std::vector<EmbagConfig> &configs) {
  std::string line;

  // Parse each line of the input file into a EmbagConfig object
  while (std::getline(infile, line)) {
    if (line.empty()) {
      continue;
    }

    // Split the line into fields and validate
    auto fields = split(line, ',');
    if (fields.size() < 12) {
      commonlog_error(
        "Invalid line (expected 12 fields): [num_embeddings, embedding_dims, num_bags, num_indices, algo, iterations, dtype, fp16_scale_bias, padding_index, include_last_offset, is_weights, scatter_stride, warmup_iters (optional)]");
      continue;
    }
    EmbagConfig cfg;
    try {
      cfg.num_embeddings = std::stoi(fields[0]);
      cfg.embedding_dims = std::stoi(fields[1]);
      cfg.num_bags = std::stoi(fields[2]);
      cfg.num_indices = std::stoi(fields[3]);
      cfg.algo = strToEmbagalgo(fields[4]);
      cfg.iters = std::stoi(fields[5]);
      auto dt = split(fields[6], ':');
      if (fields[6].size() > 0) {
        auto i = 0;
        for (; i < dt.size(); i++) {
          cfg.dt.push_back(strToDatatype(dt[i]));
        }
        for (; i < 2; i++) {
          cfg.dt.push_back(data_type_t::f32);
        }
        if (dt.size() < 2) {
          commonlog_warning("Less than 2 data types specified. Defaulting missing types to f32.");
        }
      }
      else {
        cfg.dt.push_back(data_type_t::f32);
        cfg.dt.push_back(data_type_t::f32);
        commonlog_warning("No data types specified. Defaulting all to f32.");
      }
      if (!fields[7].empty()) {
        std::string fp16_scale_bias_flag = fields[7];
        std::transform(fp16_scale_bias_flag.begin(), fp16_scale_bias_flag.end(),
                       fp16_scale_bias_flag.begin(),::tolower);
        if (fp16_scale_bias_flag == "true" || fp16_scale_bias_flag == "1") {
          cfg.fp16_scale_bias = true;
        }
        else if (fp16_scale_bias_flag == "false" || fp16_scale_bias_flag == "0") {
          cfg.fp16_scale_bias = false;
        }
        else {
          commonlog_error("Invalid value for fp16_scale_bias. Use true/false or 1/0.");
          continue;
        }
      }
      cfg.padding_index = std::stoi(fields[8]);
      if (fields[9].empty()) {
        commonlog_error("Field for include_last_offset is empty. Please provide a value.");
        continue;
      }
      std::string offset_flag = fields[9];
      std::transform(offset_flag.begin(), offset_flag.end(), offset_flag.begin(),
                     ::tolower);
      if (offset_flag == "true" || offset_flag == "1") {
        cfg.include_last_offset = true;
      }
      else if (offset_flag == "false" || offset_flag == "0") {
        cfg.include_last_offset = false;
      }
      else {
        commonlog_error("Include calue for include_last_offset. Use true/false or 1/0.");
        continue;
      }
      std::string weights_flag = fields[10];
      std::transform(weights_flag.begin(), weights_flag.end(), weights_flag.begin(),
                     ::tolower);
      if (weights_flag == "true" || weights_flag == "1") {
        cfg.is_weights = true;
      }
      else if (weights_flag == "false" || weights_flag == "0") {
        cfg.is_weights = false;
      }
      else {
        commonlog_error("Include calue for is_weights. Use true/false or 1/0.");
        continue;
      }
      cfg.scatter_stride = std::stoi(fields[11]);
      if (fields.size() == 13) {
        cfg.warmup_iters = std::stoi(fields[12]);
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

void log_benchmark_failure(const EmbagConfig &cfg) {
  testlog_error("Benchmark failed for ", cfg.num_embeddings, ", ",
                cfg.embedding_dims, ", ", cfg.num_bags, ", ", cfg.num_indices, ", ",
                embagalgoToStr(cfg.algo), ", ", cfg.iters, ", ", datatypeToStr(cfg.dt[0]), ", ",
                datatypeToStr(cfg.dt[1]), ", ", cfg.fp16_scale_bias, ", ",
                cfg.padding_index, ", ", cfg.include_last_offset, ", ", cfg.is_weights, ", ",
                cfg.scatter_stride, ", ", cfg.warmup_iters);
}

void print_results(std::vector<std::pair<EmbagConfig, TimingStats>>
                   &embag_results, std::ostream &outfile, const bool isLOWOHA) {
  std::vector<std::string> headers = {
    "Num_Embeddings", "Embedding_Dims", "Num_Bags", "Num_Indices", "Algo", "Iterations", "Data_type", "Fp16_Scale_Bias", "Padding_Index", "Include_Last_Offset", "Is_Weights", "Scatter_Stride", "Warmup_iters", "Total_time(ms) (all iters)"
  };
#if MEASURE_INDIVIDUAL_TIMINGS
  // Only add individual timing headers for non-LOWOHA mode
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
  for (const auto &result : embag_results) {
    const auto &config = result.first;
    const auto &stat = result.second;
    col_widths[0] = std::max(col_widths[0],
                             std::to_string(config.num_embeddings).size() + 2);
    col_widths[1] = std::max(col_widths[1],
                             std::to_string(config.embedding_dims).size() + 2);
    col_widths[2] = std::max(col_widths[2],
                             std::to_string(config.num_bags).size() + 2);
    col_widths[3] = std::max(col_widths[3],
                             std::to_string(config.num_indices).size() + 2);
    std::string algo_str = embagalgoToStr(config.algo);
    col_widths[4] = std::max(col_widths[4], algo_str.size() + 2);
    col_widths[5] = std::max(col_widths[5],
                             std::to_string(config.iters).size() + 2);
    std::string dt_str = datatypeToStr(config.dt[0]) + ":" + datatypeToStr(
                           config.dt[1]);
    col_widths[6] = std::max(col_widths[6], dt_str.size() + 2);
    std::string fp16_scale_bias = (config.dt[0] == data_type_t::s8 ||
                                   config.dt[0] == data_type_t::s4 ||
                                   config.dt[0] == data_type_t::u4) ?
                                  std::to_string(config.fp16_scale_bias) : "";
    col_widths[7] = std::max(col_widths[7], fp16_scale_bias.size() + 2);
    col_widths[8] = std::max(col_widths[8],
                             std::to_string(config.padding_index).size() + 2);
    col_widths[9] = std::max(col_widths[9],
                             std::to_string(config.include_last_offset).size() + 2);
    col_widths[10] = std::max(col_widths[10],
                              std::to_string(config.is_weights).size() + 2);
    col_widths[11] = std::max(col_widths[11],
                              std::to_string(config.scatter_stride).size() + 2);
    col_widths[12] = std::max(col_widths[12],
                              std::to_string(config.warmup_iters).size() + 2);
    std::ostringstream total_time_ss;
    total_time_ss << std::fixed << std::setprecision(2) << stat.total_time_ms;
    col_widths[13] = std::max(col_widths[13], total_time_ss.str().size() + 2);
#if MEASURE_INDIVIDUAL_TIMINGS
    // Only compute individual timing widths for non-LOWOHA mode
    if (!isLOWOHA) {
      std::ostringstream ctx_str, op_create_str, op_exec_str;
      double ctx_creation_percentage = (stat.context_creation_ms / stat.total_time_ms)
                                       * 100;
      double op_creation_percentage = (stat.operator_creation_ms / stat.total_time_ms)
                                      * 100;
      double op_execution_percentage = (stat.operator_execution_ms /
                                        stat.total_time_ms) * 100;
      ctx_str << std::fixed << std::setprecision(2)
              << stat.context_creation_ms << " (" << ctx_creation_percentage << " %)";
      op_create_str << std::fixed << std::setprecision(2)
                    << stat.operator_creation_ms << " (" << op_creation_percentage << " %)";
      op_exec_str << std::fixed << std::setprecision(2)
                  << stat.operator_execution_ms << " (" << op_execution_percentage << " %)";
      col_widths[14] = std::max(col_widths[14], ctx_str.str().size() + 2);
      col_widths[15] = std::max(col_widths[15], op_create_str.str().size() + 2);
      col_widths[16] = std::max(col_widths[16], op_exec_str.str().size() + 2);
    }
#endif
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
  for (const auto &result : embag_results) {
    const auto &config = result.first;
    const auto &stat = result.second;
    std::vector<std::string> row;
    row.push_back(std::to_string(config.num_embeddings));
    row.push_back(std::to_string(config.embedding_dims));
    row.push_back(std::to_string(config.num_bags));
    row.push_back(std::to_string(config.num_indices));
    row.push_back(embagalgoToStr(config.algo));
    row.push_back(std::to_string(config.iters));
    row.push_back(datatypeToStr(config.dt[0]) + ":" + datatypeToStr(config.dt[1]));
    row.push_back((config.dt[0] == data_type_t::s8 ||
                   config.dt[0] == data_type_t::s4 ||
                   config.dt[0] == data_type_t::u4) ?
                  std::to_string(config.fp16_scale_bias) : "");
    row.push_back(std::to_string(config.padding_index));
    row.push_back(std::to_string(config.include_last_offset));
    row.push_back(std::to_string(config.is_weights));
    row.push_back(std::to_string(config.scatter_stride));
    row.push_back(std::to_string(config.warmup_iters));
    std::ostringstream total_time_ss;
    total_time_ss << std::fixed << std::setprecision(2) << stat.total_time_ms;
    row.push_back(total_time_ss.str());
#if MEASURE_INDIVIDUAL_TIMINGS
    // Only add individual timing data for non-LOWOHA mode
    if (!isLOWOHA) {
      std::ostringstream ctx_str, op_create_str, op_exec_str;
      double ctx_creation_percentage = (stat.context_creation_ms / stat.total_time_ms)
                                       * 100;
      double op_creation_percentage = (stat.operator_creation_ms / stat.total_time_ms)
                                      * 100;
      double op_execution_percentage = (stat.operator_execution_ms /
                                        stat.total_time_ms) * 100;
      ctx_str << std::fixed << std::setprecision(2)
              << stat.context_creation_ms << " (" << ctx_creation_percentage << " %)";
      op_create_str << std::fixed << std::setprecision(2)
                    << stat.operator_creation_ms << " (" << op_creation_percentage << " %)";
      op_exec_str << std::fixed << std::setprecision(2)
                  << stat.operator_execution_ms << " (" << op_execution_percentage << " %)";
      row.push_back(ctx_str.str());
      row.push_back(op_create_str.str());
      row.push_back(op_exec_str.str());
    }
#endif
    print_row(row);
  }
}

void log_results(std::vector<std::pair<EmbagConfig, TimingStats>>
                 &embag_results, std::ostream &outfile, const bool isLOWOHA) {
  // Write CSV header
  outfile <<
          "Num_Embeddings, Embedding_Dims, Num_Bags, Num_Indices, Algo, Iterations, Data_type, Fp16_Scale_Bias, Padding_Index, Include_Last_Offset, Is_Weights, Scatter_Stride, Warmup_iters, Total_time(ms) (all iters)";
#if MEASURE_INDIVIDUAL_TIMINGS
  // Only add individual timing headers for non-LOWOHA mode
  if (!isLOWOHA) {
    outfile <<
            ", Context Creation Time (ms & %), Operator Creation Time (ms & %), Operator Execution Time (ms & %)";
  }
#endif
  outfile << std::endl;

  // Write results to CSV for each configuration
  for (const auto &result : embag_results) {
    const auto &config = result.first;
    const auto &stat = result.second;
    outfile <<
            config.num_embeddings << ", " <<
            config.embedding_dims << ", " <<
            config.num_bags << ", " <<
            config.num_indices << ", " <<
            embagalgoToStr(config.algo) << ", " <<
            config.iters << ", " <<
            datatypeToStr(config.dt[0]) << ":" << datatypeToStr(config.dt[1]) << ", " <<
            ((config.dt[0] == data_type_t::s8 ||
              config.dt[0] == data_type_t::s4 ||
              config.dt[0] == data_type_t::u4) ?
             std::to_string(config.fp16_scale_bias) : "") << ", " <<
            config.padding_index << ", " <<
            config.include_last_offset << ", " <<
            config.is_weights << ", " <<
            config.scatter_stride << ", " <<
            config.warmup_iters << ", " <<
            stat.total_time_ms;
#if MEASURE_INDIVIDUAL_TIMINGS
    // Only add individual timing data for non-LOWOHA mode
    if (!isLOWOHA) {
      double ctx_creation_percentage = (stat.context_creation_ms / stat.total_time_ms)
                                       * 100;
      double op_creation_percentage = (stat.operator_creation_ms / stat.total_time_ms)
                                      * 100;
      double op_execution_percentage = (stat.operator_execution_ms /
                                        stat.total_time_ms) * 100;
      outfile << ", " <<
              stat.context_creation_ms << " (" << ctx_creation_percentage << " %), " <<
              stat.operator_creation_ms << " (" << op_creation_percentage << " %), " <<
              stat.operator_execution_ms << " (" << op_execution_percentage << " %)";
    }
#endif
    outfile << std::endl;
  }
}

} // namespace embag
} // namespace benchdnn
} // namespace zendnnl