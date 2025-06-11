/********************************************************************************
# * Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef _CONFIG_MANAGER_HPP_
#define _CONFIG_MANAGER_HPP_

#include <iostream>
#include <fstream>
#include <string>
#include <cctype>
#include <cstdlib>
#include <algorithm>
#include "nlohmann/json.hpp"
#include "common/error_status.hpp"
#include "common/config_params.hpp"

namespace zendnnl {
namespace common {

using namespace zendnnl::error_handling;
using json = nlohmann::json;

/** @class config_manager_t
 *  @brief A class to hold multi-dimensional data.
 *
 *  Configuraion manager is owned by @c zendnnl_global_block_t. Though
 *  this class is not a singleton, only one configuration manager owned
 *  by @c zendnnl_global_block_t will be used.
 *
 *  ZenDNNL depends on many runtime configurable parameters. As an example
 *  logging level of various logs is a configurable parameter that can be
 *  provided at runtime. This class receives such configurable parameters
 *  and sets them.
 *
 *  These configurable user parameters can be provided in the following ways
 *
 *  - Default parameters : This class sets defaults for user parameters.
 *  - Using JSON file    : The parameters can be provided using a JSON file.
 *                         The library finds this JSON file using an
 *                         environment variable ZENDNNL_CONFIG_FILE.
 *  - Using env vars     : Another way to provide user parameters is
 *                         by providing appropriate environment variables.
 *
 *  The way config_manager_t sets user parameters is as follows. It sets
 *  default parameters. It then looks if ZENDNNL_CONFIG_FILE is pointing
 *  to a JSON config file, and tries to read configuration from this JSON
 *  file. If JSON file is not provided, or it fails to read JSON file, it
 *  tries to set user parameters from environment variables.
 */
class config_manager_t final {
public:
  /** @name Configure
   */
  /**@{*/
  /** @brief Configure ZenDNNL
   *
   *  Configures the library.
   */
  void                        config();
  /**@}*/

  /** @name Get Configurations
   */
  /**@{*/
  /** @brief Get logger configuration
   *
   *  @return Logger configuration.
   */
  const config_logger_t&      get_logger_config() const;
  /**@}*/

private:

  /** @brief Parse a JSON file.
   *
   *  Parse a JSON file and get object in @c config_json.
   *  @param file_name_ : JSON file to be parsed.
   *  @return success if successful.
   */
  status_t          parse(std::string file_name_);

  /** @brief Set default configuration.
   */
  void              set_default_config();

  /** @brief Set user configuration from JSON file.
   */
  void              set_user_config();

  /** @brief Set config using evnironment variables.
   */
  void              set_env_config();

  /** @brief Set default logger config.
   *
   * @return success.
   */
  status_t          set_default_logger_config();

  /** @brief Set logger config from JSON file.
   *
   * @return success.
   */
  status_t          set_user_logger_config();

  /** @brief Set logger config from environment variables.
   *
   * @return success.
   */
  status_t          set_env_logger_config();

  json              config_json;    /**< JSON object read from
                                       config file */
  config_logger_t   config_logger;  /**< Logger config */
};


} //common
} //zendnnl

#endif
