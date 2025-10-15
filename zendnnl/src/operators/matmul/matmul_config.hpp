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

#ifndef _MATMUL_CONFIG_HPP_
#define _MATMUL_CONFIG_HPP_

#include <algorithm>
#include <string>

#include "operators/common/operator_config.hpp"

namespace zendnnl {
namespace ops {

/** @enum matmul_algo_t
 *  @brief defines different algo levels.
 *
 * Defines all available matmul backends.
 */
enum class matmul_algo_t : uint32_t {
  dynamic_dispatch = 0,  /*!< Dynamic dispatch */
  aocl_blis = 1,         /*!< AOCL */
  aocl_blis_blocked = 2, /*!< Blocked AOCL */
  onednn = 3,            /*!< OneDNN */
  onednn_blocked = 4,    /*!< Blocked OneDNN */
  libxsmm = 5,           /*!< LIBXSMM */
  reference = 6,         /*!< Reference */
  algo_count             /*!< Algo count */
};

/**
* @class matmul_config_t
* @brief config for @c matmul_operator_t.
*
* This class encapsulates all configuration parameters and methods
* required to control the behavior of the MatMul operator.
* It supports setting default, user, and environment-based configurations,
* and provides a singleton instance for global access.
*
* Usage:
* - Use @c instance() to access the singleton configuration object.
* - Use @c set_default_config(), @c set_user_config(), and @c set_env_config()
*   to initialize configuration from different sources.
*
* Example:
* @code
* matmul_config_t &config = matmul_config_t::instance();
* config.set_default_config();
* @endcode
*
* @sa matmul_operator_t
*/
class matmul_config_t final : public op_config_t {
 public:
  void set_default_config() override;
  status_t set_user_config(json config_json) override;
  void set_env_config() override;

  /** @brief Sets matmul algo.
  *
  * @param algo The MatMul algorithm to set.
  */
  void set_algo(uint32_t algo);

  /** @brief Get matmul algo.
   *
   * @return matmul algo.
   */
  uint32_t get_algo();

  /** @brief Returns the singleton instance of matmul_config_t.
  *
  *  This method ensures only one instance of matmul_config_t exists
  *  throughout the program lifetime.
  *
  *  @return Reference to the singleton matmul_config_t instance.
  **/
  static matmul_config_t &instance();

  /** @brief Convert from string to matmul_algo.
  *
  *  @param str_ : string contains matmul algo name.
  *  @return matmul algo for appropriate string.
  *          matmul_algo_t::algo_count if string is not
  *          appropriate.
  */
  matmul_algo_t str_to_matmul_algo(std::string algo);

 private:
  /**
  * @brief Private constructor for singleton pattern.
  *
  * The constructor is private to prevent direct instantiation of the class.
  * Use the @c instance() method to access the single global instance.
  */
  matmul_config_t() = default;

  uint32_t matmul_algo;       /**< Matmul runtime algorithm. */
};

}
}

#endif