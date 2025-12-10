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

#include "benchdnn_utils.hpp"

namespace zendnnl {
namespace benchdnn {

volatile unsigned long global_sum = 0;

using namespace zendnnl::interface;

void trim(std::string &str) {
  str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());
}

std::vector<std::string> split(const std::string &str, char delimiter) {
  std::vector<std::string> tokens;
  size_t start = 0, end;

  while ((end = str.find(delimiter, start)) != std::string::npos) {
    std::string token = str.substr(start, end - start);
    trim(token);
    tokens.emplace_back(token);  // include empty token
    start = end + 1;
  }

  std:: string token = str.substr(start);
  trim(token);
  tokens.emplace_back(token);  // last token (even if empty)
  return tokens;
}

data_type_t strToDatatype(const std::string &str) {
  if (str == "f32") {
    return data_type_t::f32;
  }
  if (str == "f16") {
    return data_type_t::f16;
  }
  if (str == "bf16") {
    return data_type_t::bf16;
  }
  if (str == "s32") {
    return data_type_t::s32;
  }
  if (str == "s16") {
    return data_type_t::s16;
  }
  if (str == "s8") {
    return data_type_t::s8;
  }
  if (str == "s4") {
    return data_type_t::s4;
  }
  if (str == "u32") {
    return data_type_t::u32;
  }
  if (str == "u16") {
    return data_type_t::u16;
  }
  if (str == "u8") {
    return data_type_t::u8;
  }
  if (str == "u4") {
    return data_type_t::u4;
  }
  throw std::invalid_argument("Unknown data type string '" + str + "'");
}

std::string datatypeToStr(data_type_t dt) {
  switch (dt) {
  case data_type_t::f32:
    return "f32";
  case data_type_t::f16:
    return "f16";
  case data_type_t::bf16:
    return "bf16";
  case data_type_t::s32:
    return "s32";
  case data_type_t::s16:
    return "s16";
  case data_type_t::s8:
    return "s8";
  case data_type_t::s4:
    return "s4";
  case data_type_t::u32:
    return "u32";
  case data_type_t::u16:
    return "u16";
  case data_type_t::u8:
    return "u8";
  case data_type_t::u4:
    return "u4";
  default:
    return "unknown";
  }
}

post_op_type_t strToPostOps(const std::string &str) {
  if (str == "elu") {
    return post_op_type_t::elu;
  }
  if (str == "relu") {
    return post_op_type_t::relu;
  }
  if (str == "leaky_relu") {
    return post_op_type_t::leaky_relu;
  }
  if (str == "gelu_tanh") {
    return post_op_type_t::gelu_tanh;
  }
  if (str == "gelu_erf") {
    return post_op_type_t::gelu_erf;
  }
  if (str == "sigmoid") {
    return post_op_type_t::sigmoid;
  }
  if (str == "swish") {
    return post_op_type_t::swish;
  }
  if (str == "tanh") {
    return post_op_type_t::tanh;
  }
  if (str == "softmax") {
    return post_op_type_t::softmax;
  }
  if (str == "pooling") {
    return post_op_type_t::pooling;
  }
  if (str == "square") {
    return post_op_type_t::square;
  }
  if (str == "abs") {
    return post_op_type_t::abs;
  }
  if (str == "sqrt") {
    return post_op_type_t::sqrt;
  }
  if (str == "exp") {
    return post_op_type_t::exp;
  }
  if (str == "log") {
    return post_op_type_t::log;
  }
  if (str == "clip") {
    return post_op_type_t::clip;
  }
  if (str == "binary_add") {
    return post_op_type_t::binary_add;
  }
  if (str == "binary_mul") {
    return post_op_type_t::binary_mul;
  }
  throw std::invalid_argument("Unknown post-op string '" + str + "'");
}

std::string postOpsToStr(post_op_type_t post_op) {
  switch (post_op) {
  case post_op_type_t::elu:
    return "elu";
  case post_op_type_t::relu:
    return "relu";
  case post_op_type_t::leaky_relu:
    return "leaky_relu";
  case post_op_type_t::gelu_tanh:
    return "gelu_tanh";
  case post_op_type_t::gelu_erf:
    return "gelu_erf";
  case post_op_type_t::sigmoid:
    return "sigmoid";
  case post_op_type_t::swish:
    return "swish";
  case post_op_type_t::tanh:
    return "tanh";
  case post_op_type_t::softmax:
    return "softmax";
  case post_op_type_t::pooling:
    return "pooling";
  case post_op_type_t::square:
    return "square";
  case post_op_type_t::abs:
    return "abs";
  case post_op_type_t::sqrt:
    return "sqrt";
  case post_op_type_t::exp:
    return "exp";
  case post_op_type_t::log:
    return "log";
  case post_op_type_t::clip:
    return "clip";
  case post_op_type_t::binary_add:
    return "binary_add";
  case post_op_type_t::binary_mul:
    return "binary_mul";
  default:
    return "relu";
  }
}

#if COLD_CACHE
void flush_cache(std::vector<char> &buffer) {
  unsigned long sum = 0;
  for (size_t i = 0; i < CACHE_SIZE; i += CACHE_LINE_SIZE) {
    buffer[i]++;
    global_sum += buffer[i];
    _mm_clflush(&buffer[i]);
  }
}
#endif

int parseCLArgs(benchdnn::global_options &options, std::string arg) {
  if (arg.find("--ndims=") == 0) {
    if (arg.substr(8).empty()) {
      commonlog_error("ndims value cannot be empty. Please provide a valid number.");
      return NOT_OK;
    }
    options.ndims = std::stoi(arg.substr(8));
  }
  else if (arg.find("--bs=") == 0) {
    if (arg.substr(5).empty() || std::stoi(arg.substr(5)) <= 0) {
      commonlog_error("bs value cannot be empty or <= 0. Please provide a valid number.");
      return NOT_OK;
    }
    options.bs = std::stoi(arg.substr(5));
  }
  else if (arg.find("--m=") == 0) {
    if (arg.substr(4).empty() || std::stoi(arg.substr(4)) <= 0) {
      commonlog_error("M value cannot be empty or <= 0. Please provide a valid number.");
      return NOT_OK;
    }
    options.m = std::stoi(arg.substr(4));
  }
  else if (arg.find("--k=") == 0) {
    if (arg.substr(4).empty() || std::stoi(arg.substr(4)) <= 0) {
      commonlog_error("K value cannot be empty or <= 0. Please provide a valid number.");
      return NOT_OK;
    }
    options.k = std::stoi(arg.substr(4));
  }
  else if (arg.find("--n=") == 0) {
    std::string n_values_str = arg.substr(4);
    if (n_values_str.empty()) {
      commonlog_error("N values cannot be empty. Please provide valid numbers.");
      return NOT_OK;
    }
    auto n_values = split(n_values_str, ':');
    for (const auto &n : n_values) {
      if (n.empty() || std::stoi(n) <= 0) {
        commonlog_error("One of the n values is empty or <= 0. Please provide a valid value.");
        return NOT_OK;
      }
      options.n_values.push_back(std::stoi(n));
    }
  }
  else if (arg.find("--bias=") == 0) {
    std::string bias_str = arg.substr(7);
    if (bias_str.empty()) {
      commonlog_error("Bias value cannot be empty. Please provide true/false or 1/0.");
      return NOT_OK;
    }
    std::transform(bias_str.begin(), bias_str.end(), bias_str.begin(), ::tolower);
    if (bias_str == "true" || bias_str == "1") {
      options.isBiasEnabled = true;
    }
    else if (bias_str == "false" || bias_str == "0") {
      options.isBiasEnabled = false;
    }
    else {
      commonlog_error("Invalid value for bias. Use true/false or 1/0.");
      return NOT_OK;
    }
  }
  else if (arg.find("--post_ops=") == 0) {
    std::string post_ops_str = arg.substr(11);
    if (post_ops_str.empty()) {
      commonlog_error("Post-ops string cannot be empty. Please provide valid post-ops.");
      return NOT_OK;
    }
    auto post_ops_vec = split(post_ops_str, ':');
    for (const auto &post_op_str : post_ops_vec) {
      options.post_ops.push_back(strToPostOps(post_op_str));
    }
  }
  else if (arg.find("--iters=") == 0) {
    if (arg.substr(8).empty()) {
      commonlog_error("Iterations value cannot be empty. Please provide a valid number.");
      return NOT_OK;
    }
    options.iters = std::stoi(arg.substr(8));
  }
  else if (arg.find("--sdt=") == 0) {
    if (arg.substr(6).empty()) {
      commonlog_error("Source data type cannot be empty. Please provide a valid data type.");
      return NOT_OK;
    }
    options.sdt = strToDatatype(arg.substr(6));
  }
  else if (arg.find("--wdt=") == 0) {
    if (arg.substr(6).empty()) {
      commonlog_error("Weights data type cannot be empty. Please provide a valid data type.");
      return NOT_OK;
    }
    options.wdt = strToDatatype(arg.substr(6));
  }
  else if (arg.find("--ddt=") == 0) {
    if (arg.substr(6).empty()) {
      commonlog_error("Destination data type cannot be empty. Please provide a valid data type.");
      return NOT_OK;
    }
    options.ddt = strToDatatype(arg.substr(6));
  }
  else if (arg.find("--kernel_name=") == 0) {
    if (arg.substr(14).empty()) {
      commonlog_error("Kernel name cannot be empty. Please provide a valid kernel name.");
      return NOT_OK;
    }
    options.kernel_name = arg.substr(14);
  }
  else if (arg.find("--bias_dt=") == 0) {
    if (arg.substr(10).empty()) {
      commonlog_error("Bias data type cannot be empty. Please provide a valid data type.");
      return NOT_OK;
    }
    options.bias_dt = strToDatatype(arg.substr(10));
  }
  else if (arg.find("--isTransA=") == 0) {
    if (arg.substr(11).empty()) {
      commonlog_error("isTransA value cannot be empty. Please provide true/false or 1/0.");
      return NOT_OK;
    }
    options.isTransA = (arg.substr(11) == "true");
  }
  else if (arg.find("--isTransB=") == 0) {
    if (arg.substr(11).empty()) {
      commonlog_error("isTransB value cannot be empty. Please provide true/false or 1/0.");
      return NOT_OK;
    }
    options.isTransB = (arg.substr(11) == "true");
  }
  else if (arg.find("--alpha=") == 0) {
    if (arg.substr(8).empty()) {
      commonlog_error("Alpha value cannot be empty. Please provide a valid number.");
      return NOT_OK;
    }
    options.alpha = std::stof(arg.substr(8));
  }
  else if (arg.find("--beta=") == 0) {
    if (arg.substr(7).empty()) {
      commonlog_error("Beta value cannot be empty. Please provide a valid number.");
      return NOT_OK;
    }
    options.beta = std::stof(arg.substr(7));
  }
  else if (arg.find("--warmup_iters=") == 0) {
    if (arg.substr(15).empty()) {
      commonlog_error("Warmup iterations value cannot be empty. Please provide a valid number.");
      return NOT_OK;
    }
    options.warmup_iters = std::stoi(arg.substr(15));
  }
  else {
    commonlog_error("Unknown argument: ", arg);
    return NOT_OK;
  }
  return OK;
}

} // namespace benchdnn
} // namespace zendnnl