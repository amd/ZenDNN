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

#include "embag_config.hpp"

namespace zendnnl {
namespace ops {

void embag_config_t::set_default_config() {
  // Set default configuration for embag
  int32_t embag_kernel = static_cast<int32_t>(embag_kernel_t::none);
  set_kernel(embag_kernel);
}

status_t embag_config_t::set_user_config(json config_json) {
  // Set user-defined configuration for embag from json
  auto runtime_variables_json = config_json["runtime_variables"];
  if (runtime_variables_json.empty()) {
    return status_t::failure;
  }

  int32_t embag_kernel = static_cast<int32_t>(embag_kernel_t::none);
  auto embag_json = runtime_variables_json["embag"];
  if (!embag_json.empty()) {
    auto embag_kernel_json = embag_json["kernel"];
    if (!embag_kernel_json.empty()) {
      auto embag_kernel_str = embag_kernel_json.template get<std::string>();
      if (!embag_kernel_str.empty()) {
        embag_kernel = static_cast<int32_t>(str_to_embag_kernel(embag_kernel_str));
      }
    }
  }

  set_kernel(embag_kernel);
  return status_t::success;
}

void embag_config_t::set_env_config() {
  // Set environment variables configuration for embag
  char *kernel_env = std::getenv("ZENDNNL_EMBAG_ALGO");
  int32_t embag_kernel = static_cast<int32_t>(embag_kernel_t::none);
  if (kernel_env) {
    try {
      int32_t kernel = std::stoi(kernel_env);
      if (kernel > static_cast<int32_t>(embag_kernel_t::none) &&
          kernel < static_cast<int32_t>(embag_kernel_t::kernel_count)) {
        embag_kernel = static_cast<int32_t>(embag_kernel_t(kernel));
      }
      else {
        embag_kernel = static_cast<int32_t>(embag_kernel_t::kernel_count);
      }
    }
    catch (const std::invalid_argument &e) {
      embag_kernel = static_cast<int32_t>(embag_kernel_t::kernel_count);
    }
    catch (const std::out_of_range &e) {
      embag_kernel = static_cast<int32_t>(embag_kernel_t::kernel_count);
    }
  }
  set_kernel(embag_kernel);
}

void embag_config_t::set_kernel(int32_t kernel) {
  embag_kernel = static_cast<embag_kernel_t>(kernel);
}

int32_t embag_config_t::get_kernel() {
  return static_cast<int32_t>(embag_kernel);
}

embag_config_t &embag_config_t::instance() {
  // The static local variable 'instance' is created on first call
  // and reused for all subsequent calls, providing global access
  // to configuration.
  static embag_config_t instance;
  return instance;
}

embag_kernel_t embag_config_t::str_to_embag_kernel(std::string kernel) {
  //transform kernel to lower case.
  std::transform(kernel.begin(), kernel.end(),
  kernel.begin(), [](unsigned char c) {
    return std::tolower(c);
  });

  if (kernel == "none") {
    return embag_kernel_t::none;
  }
  else if (kernel == "auto_tuner") {
    return embag_kernel_t::auto_tuner;
  }
  else if (kernel == "native") {
    return embag_kernel_t::native;
  }
  else if (kernel == "fbgemm") {
    return embag_kernel_t::fbgemm;
  }
  else if (kernel == "reference") {
    return embag_kernel_t::reference;
  }

  return embag_kernel_t::kernel_count;
}

}
}

