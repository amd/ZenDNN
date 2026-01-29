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

#include "matmul_config.hpp"

namespace zendnnl {
namespace ops {

void matmul_config_t::set_default_config() {
  // Set default configuration for matmul
  int32_t matmul_algo = static_cast<int32_t>(matmul_algo_t::none);
  set_algo(matmul_algo);
  set_bmm_algo(matmul_algo);
  set_weight_cache(1);
  set_zp_comp_cache(true);  // Enable ZP compensation caching by default
}

status_t matmul_config_t::set_user_config(json config_json) {
  // Set user-defined configuration for matmul from json
  auto runtime_variables_json = config_json["runtime_variables"];
  if (runtime_variables_json.empty()) {
    return status_t::failure;
  }
  // get matmul_algo
  int32_t matmul_algo = static_cast<int32_t>(matmul_algo_t::none);
  int32_t bmm_algo = static_cast<int32_t>(matmul_algo_t::none);
  int32_t matmul_weight_cache = 1;
  bool zp_comp_cache_enabled = true;  // Default enabled
  uint32_t lru_cache_capacity = std::numeric_limits<uint32_t>::max();
  auto matmul_json = runtime_variables_json["matmul"];
  if (! matmul_json.empty()) {
    auto matmul_algo_json = matmul_json["mm_kernel"];
    if (! matmul_algo_json.empty()) {
      auto matmul_algo_str = matmul_algo_json.template get<std::string>();
      if (! matmul_algo_str.empty()) {
        matmul_algo = static_cast<int32_t>(str_to_matmul_algo(matmul_algo_str));
      }
    }
    // Read MM partitioner setting from JSON
    auto mm_partitioner_json = matmul_json["mm_partitioner_enabled"];
    if (! mm_partitioner_json.empty()) {
      auto mm_partitioner_str = mm_partitioner_json.template get<std::string>();
      if (! mm_partitioner_str.empty()) {
        std::string value_str = mm_partitioner_str;
        std::transform(value_str.begin(), value_str.end(), value_str.begin(),
        [](unsigned char c) {
          return std::tolower(c);
        });
        mm_partitioner_enabled = (value_str == "true" || value_str == "1" ||
                                  value_str == "on");
      }
    }
    // Read tile sizes from JSON
    auto tile_m_json = matmul_json["tile_m"];
    if (! tile_m_json.empty()) {
      tile_m = tile_m_json.template get<int32_t>();
      if (tile_m <= 0) {
        tile_m = 0;
      }
    }
    auto tile_n_json = matmul_json["tile_n"];
    if (! tile_n_json.empty()) {
      tile_n = tile_n_json.template get<int32_t>();
      if (tile_n <= 0) {
        tile_n = 0;
      }
    }
    auto bmm_algo_json = matmul_json["bmm_kernel"];
    if (! bmm_algo_json.empty()) {
      auto bmm_algo_str = bmm_algo_json.template get<std::string>();
      if (! bmm_algo_str.empty()) {
        bmm_algo = static_cast<int32_t>(str_to_matmul_algo(bmm_algo_str));
      }
    }

    // If bmm_kernel is not specified or set to "none", fallback to matmul algo
    if (bmm_algo_json.empty() ||
        bmm_algo == static_cast<int32_t>(matmul_algo_t::none)) {
      bmm_algo = matmul_algo;
    }

    auto matmul_weight_cache_json = matmul_json["weight_cache"];
    if (! matmul_weight_cache_json.empty()) {
      auto matmul_weight_cache_str = matmul_weight_cache_json.template
                                     get<std::string>();
      if (! matmul_weight_cache_str.empty()) {
        matmul_weight_cache = (matmul_weight_cache_str == "0") ? 0 : 1;
      }
    }
    // Read ZP compensation cache setting from JSON
    auto zp_comp_cache_json = matmul_json["zp_comp_cache"];
    if (! zp_comp_cache_json.empty()) {
      auto zp_comp_cache_str = zp_comp_cache_json.template get<std::string>();
      if (! zp_comp_cache_str.empty()) {
        zp_comp_cache_enabled = (zp_comp_cache_str != "0");
      }
    }
  }
  // Read LRU cache capacity setting from JSON
  auto lru_cache_json = config_json["lru_cache"];
  if (!lru_cache_json.empty()) {
    //get log levels of each log
    auto capacity_json = lru_cache_json["capacity"];
    if (!capacity_json.empty()) {
      lru_cache_capacity = capacity_json.get<uint32_t>();
    }
  }
  set_algo(matmul_algo);
  set_bmm_algo(bmm_algo);
  set_weight_cache(matmul_weight_cache);
  set_zp_comp_cache(zp_comp_cache_enabled);
  set_lru_cache_capacity(lru_cache_capacity);
  set_mm_partitioner_enabled(mm_partitioner_enabled);
  set_tile_m(tile_m);
  set_tile_n(tile_n);

  return status_t::success;
}

void matmul_config_t::set_env_config() {
  // Set environment variables configuration for matmul
  char *algo_env = std::getenv("ZENDNNL_MATMUL_ALGO");
  int32_t matmul_algo = static_cast<int32_t>(matmul_algo_t::none);
  if (algo_env) {
    std::string algoStr(algo_env);
    std::transform(algoStr.begin(), algoStr.end(), algoStr.begin(),
    [](unsigned char c) {
      return std::tolower(c);
    });
    if (algoStr == "auto") {
      matmul_algo = static_cast<int32_t>(matmul_algo_t::auto_tuner);
    }
    else {
      try {
        int32_t algo = std::stoi(algoStr);
        if (algo > static_cast<int32_t>(matmul_algo_t::none) &&
            algo < static_cast<int32_t>(matmul_algo_t::algo_count)) {
          matmul_algo = static_cast<int32_t>(matmul_algo_t(algo));
        }
        else {
          matmul_algo = static_cast<int32_t>(matmul_algo_t::algo_count);
        }
      }
      catch (const std::invalid_argument &e) {
        matmul_algo = static_cast<int32_t>(matmul_algo_t::algo_count);
      }
      catch (const std::out_of_range &e) {
        matmul_algo = static_cast<int32_t>(matmul_algo_t::algo_count);
      }
    }
  }
  set_algo(matmul_algo);
  char *bmm_algo_env = std::getenv("ZENDNNL_BMM_ALGO");
  int32_t bmm_algo = matmul_algo;  // Default to matmul_algo
  if (bmm_algo_env) {
    std::string bmm_algoStr(bmm_algo_env);
    std::transform(bmm_algoStr.begin(), bmm_algoStr.end(), bmm_algoStr.begin(),
    [](unsigned char c) {
      return std::tolower(c);
    });
    if (bmm_algoStr == "auto") {
      bmm_algo = static_cast<int32_t>(matmul_algo_t::auto_tuner);
    }
    else {
      try {
        int32_t algo = std::stoi(bmm_algoStr);
        if (algo > static_cast<int32_t>(matmul_algo_t::none) &&
            algo < static_cast<int32_t>(matmul_algo_t::algo_count)) {
          bmm_algo = static_cast<int32_t>(matmul_algo_t(algo));
        }
        else {
          bmm_algo = static_cast<int32_t>(matmul_algo_t::algo_count);
        }
      }
      catch (const std::invalid_argument &e) {
        bmm_algo = static_cast<int32_t>(matmul_algo_t::algo_count);
      }
      catch (const std::out_of_range &e) {
        bmm_algo = static_cast<int32_t>(matmul_algo_t::algo_count);
      }
    }
  }

  // If ZENDNNL_BMM_ALGO is set to -1 (none), fallback to matmul_algo
  if (bmm_algo == static_cast<int32_t>(matmul_algo_t::none)) {
    bmm_algo = matmul_algo;
  }

  set_bmm_algo(bmm_algo);
  char *weight_cache_env = std::getenv("ZENDNNL_MATMUL_WEIGHT_CACHE");
  [[maybe_unused]] int32_t matmul_weight_cache = 1;
  if (weight_cache_env) {
    int32_t weight_cache = std::stoi(weight_cache_env);
    if (weight_cache == 0 || weight_cache == 1) {
      matmul_weight_cache = weight_cache;
    }
  }
  set_weight_cache(matmul_weight_cache);

  // Read ZP compensation cache setting from environment
  char *zp_comp_cache_env = std::getenv("ZENDNNL_ZP_COMP_CACHE");
  bool zp_comp_cache_enabled = true;  // Default enabled
  if (zp_comp_cache_env) {
    zp_comp_cache_enabled = (std::stoi(zp_comp_cache_env) != 0);
  }
  set_zp_comp_cache(zp_comp_cache_enabled);

  char *lru_cache_capacity_env = std::getenv("ZENDNNL_LRU_CACHE_CAPACITY");
  uint32_t lru_cache_capacity = std::numeric_limits<uint32_t>::max();
  if (lru_cache_capacity_env) {
    lru_cache_capacity = std::stoi(lru_cache_capacity_env);
  }
  set_lru_cache_capacity(lru_cache_capacity);

  // Read MM partitioner enable setting from environment
  char *mm_partitioner_env = std::getenv("ZENDNN_ENABLE_MM_PARTITIONER");
  bool mm_partitioner_enabled = false;  // Default disabled
  if (mm_partitioner_env) {
    std::string value_str(mm_partitioner_env);
    std::transform(value_str.begin(), value_str.end(), value_str.begin(),
    [](unsigned char c) {
      return std::tolower(c);
    });

    if (value_str == "true" || value_str == "1" || value_str == "on") {
      mm_partitioner_enabled = true;
    }
    else if (value_str == "false" || value_str == "0" || value_str == "off") {
      mm_partitioner_enabled = false;
    }
  }
  set_mm_partitioner_enabled(mm_partitioner_enabled);

  // Read tile size settings from environment
  char *tile_m_env = std::getenv("ZENDNN_MM_TILE_M");
  int32_t tile_m = 0;
  if (tile_m_env) {
    try {
      int32_t value = std::stoi(tile_m_env);
      if (value > 0) {
        tile_m = value;
      }
    }
    catch (const std::exception &e) {
      // Keep default value on error
    }
  }
  set_tile_m(tile_m);

  char *tile_n_env = std::getenv("ZENDNN_MM_TILE_N");
  int32_t tile_n = 0;
  if (tile_n_env) {
    try {
      int32_t value = std::stoi(tile_n_env);
      if (value > 0) {
        tile_n = value;
      }
    }
    catch (const std::exception &e) {
      // Keep default value on error
    }
  }
  set_tile_n(tile_n);
}

void matmul_config_t::set_algo(int32_t algo) {
  matmul_algo = algo;
}

int32_t matmul_config_t::get_algo() {
  return matmul_algo;
}

void matmul_config_t::set_bmm_algo(int32_t algo) {
  bmm_algo = algo;
}

int32_t matmul_config_t::get_bmm_algo() {
  return bmm_algo;
}

void matmul_config_t::set_weight_cache(int32_t weight_cache) {
  matmul_weight_cache = weight_cache;
}

int32_t matmul_config_t::get_weight_cache() {
  return matmul_weight_cache;
}

void matmul_config_t::set_zp_comp_cache(bool comp_cache) {
  zp_comp_cache = comp_cache;
}

bool matmul_config_t::get_zp_comp_cache() {
  return zp_comp_cache;
}

void matmul_config_t::set_lru_cache_capacity(uint32_t capacity) {
  lru_cache_capacity = capacity;
}

uint32_t matmul_config_t::get_lru_cache_capacity() {
  return lru_cache_capacity;
}
void matmul_config_t::set_mm_partitioner_enabled(bool enabled) {
  mm_partitioner_enabled = enabled;
}

bool matmul_config_t::get_mm_partitioner_enabled() {
  return mm_partitioner_enabled;
}

void matmul_config_t::set_tile_m(int32_t size) {
  tile_m = size;
}

int32_t matmul_config_t::get_tile_m() {
  return tile_m;
}

void matmul_config_t::set_tile_n(int32_t size) {
  tile_n = size;
}

int32_t matmul_config_t::get_tile_n() {
  return tile_n;
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
  else if (algo == "dynamic_dispatch") {
    return matmul_algo_t::dynamic_dispatch;
  }
  else if (algo == "aocl_dlp") {
    return matmul_algo_t::aocl_dlp;
  }
  else if (algo == "aocl_dlp_blocked") {
    return matmul_algo_t::aocl_dlp_blocked;
  }
  else if (algo == "onednn") {
    return matmul_algo_t::onednn;
  }
  else if (algo == "onednn_blocked") {
    return matmul_algo_t::onednn_blocked;
  }
  else if (algo == "libxsmm") {
    return matmul_algo_t::libxsmm;
  }
  else if (algo == "libxsmm_blocked") {
    return matmul_algo_t::libxsmm_blocked;
  }
  else if (algo == "reference") {
    return matmul_algo_t::reference;
  }
  else if (algo == "batched_sgemm") {
    return matmul_algo_t::batched_sgemm;
  }
  else if (algo == "auto_tuner") {
    return matmul_algo_t::auto_tuner;
  }

  return matmul_algo_t::algo_count;
}

}
}