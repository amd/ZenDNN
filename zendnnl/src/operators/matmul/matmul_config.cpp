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

#include "matmul_config.hpp"

namespace zendnnl {
namespace ops {

void matmul_config_t::set_default_config() {
  // Set default configuration for matmul
  uint32_t matmul_algo = static_cast<int>(matmul_algo_t::none);
  set_algo(matmul_algo);
}

status_t matmul_config_t::set_user_config(json config_json) {
  // Set user-defined configuration for matmul from json
  auto runtime_variables_json = config_json["runtime_variables"];
  if (runtime_variables_json.empty()) {
    return status_t::failure;
  }
  // get matmul_algo
  uint32_t matmul_algo = static_cast<int>(matmul_algo_t::none);
  auto matmul_json = runtime_variables_json["matmul"];
  if (! matmul_json.empty()) {
    auto matmul_algo_json = matmul_json["kernel"];
    if (! matmul_algo_json.empty()) {
      auto matmul_algo_str = matmul_algo_json.template get<std::string>();
      if (! matmul_algo_str.empty()) {
        matmul_algo = static_cast<int>(str_to_matmul_algo(matmul_algo_str));
      }
    }
  }
  set_algo(matmul_algo);
  return status_t::success;
}

void matmul_config_t::set_env_config() {
  // Set environment variables configuration for matmul
  char *algo_env = std::getenv("ZENDNNL_MATMUL_ALGO");
  uint32_t matmul_algo = static_cast<int>(matmul_algo_t::none);
  if (algo_env) {
    uint32_t algo = std::stoi(algo_env);
    if (algo < uint32_t(matmul_algo_t::algo_count)) {
      matmul_algo = static_cast<int>(matmul_algo_t(algo));
    }
    else {
      matmul_algo = static_cast<int>(matmul_algo_t::algo_count);
    }
  }
  set_algo(matmul_algo);
}

void matmul_config_t::set_algo(uint32_t algo) {
  matmul_algo = algo;
}

uint32_t matmul_config_t::get_algo() {
  return matmul_algo;
}

matmul_config_t &matmul_config_t::instance() {
  // The static local variable 'instance' is created on first call
  // and reused for all subsequent calls, providing global access
  // to configuration.
  static matmul_config_t instance;
  return instance;
}

matmul_algo_t matmul_config_t::str_to_matmul_algo(std::string algo) {
  //transform algo to lower case.
  std::transform(algo.begin(), algo.end(), algo.begin(), [](unsigned char c) {
    return std::tolower(c);
  });

  if (algo == "none") {
    return matmul_algo_t::none;
  }
  else if (algo == "aocl_blis") {
    return matmul_algo_t::aocl_blis;
  }
  else if (algo == "aocl_blis_blocked") {
    return matmul_algo_t::aocl_blis_blocked;
  }
  else if ((algo == "onednn") || (algo == "onednn_blocked")) {
    return matmul_algo_t::onednn;
  }
  else if (algo == "reference") {
    return matmul_algo_t::reference;
  }

  return matmul_algo_t::algo_count;
}

}
}