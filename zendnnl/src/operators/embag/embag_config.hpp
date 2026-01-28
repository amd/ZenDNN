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

#ifndef _EMBAG_CONFIG_HPP_
#define _EMBAG_CONFIG_HPP_

#include <algorithm>
#include <string>
#include "operators/common/operator_config.hpp"

namespace zendnnl {
namespace ops {

/** @enum embag_kernel_t
 *  @brief defines different kernel levels.
 *
 * Defines all available embedding bag kernel backends.
 */
enum class embag_kernel_t : int32_t {
  none = -1,             /*!< No kernel selected */
  auto_tuner = 0,        /*!< TODO: Auto-tuner kernel */
  native = 1,            /*!< Native kernel */
  fbgemm = 2,            /*!< FBGEMM kernel */
  reference = 3,         /*!< Reference kernel */
  kernel_count           /*!< Kernel count */
};

/** @enum eb_thread_algo_t
 *  @brief Defines different threading algorithms for group embedding bag operations.
 *
 * These algorithms control how work is distributed across threads when
 * processing multiple embedding tables in parallel.
 */
enum class eb_thread_algo_t : int32_t {
  batch_threaded = 0,      /*!< Sequential tables with batch-level threading */
  table_threaded = 1,      /*!< Thread-per-table parallelism */
  ccd_threaded = 2,        /*!< CCD-aware threading with nested parallelism */
  hybrid_threaded = 3,     /*!< Hybrid threading when tables < threads */
  thread_algo_count        /*!< Thread algorithm count */
};

/**
* @class embag_config_t
* @brief config for @c embag_operator_t.
*
* This class encapsulates all configuration parameters and methods
* required to control the behavior of the Embedding Bag operator.
* It supports setting default, user, and environment-based configurations,
* and provides a singleton instance for global access.
*
* Usage:
* - Use @c instance() to access the singleton configuration object.
* - Use @c set_default_config() to set default configuration.
* - Use @c set_user_config() to set configuration from JSON file.
* - Use @c set_env_config() to initialize configuration from environment variables.
*
* Example:
* @code
* embag_config_t &config = embag_config_t::instance();
* config.set_default_config();
* config.set_user_config(config_json);
* config.set_env_config();
* @endcode
*
* @sa embag_operator_t
*/
class embag_config_t final : public op_config_t {
 public:
  void set_default_config() override;
  status_t set_user_config(json config_json) override;
  void set_env_config() override;

  /** @brief Sets embedding bag kernel.
  *
  * @param kernel The Embedding Bag kernel to set.
  */
  void set_kernel(int32_t kernel);

  /** @brief Get embedding bag kernel.
   *
   * @return embedding bag kernel.
   */
  int32_t get_kernel();

  /** @brief Sets thread algorithm for group embedding bag.
  *
  * @param algo The thread algorithm to set.
  */
  void set_thread_algo(int32_t algo);

  /** @brief Get thread algorithm for group embedding bag.
   *
   * @return thread algorithm.
   */
  eb_thread_algo_t get_thread_algo();

  static embag_config_t &instance();

  /** @brief Convert from string to embag_kernel.
  *
  *  @param str_ : string contains embag kernel name.
  *  @return embag kernel for appropriate string.
  *          embag_kernel_t::kernel_count if string is not
  *          appropriate.
  */
  embag_kernel_t str_to_embag_kernel(std::string kernel);

  /** @brief Convert from string to thread algorithm.
  *
  *  @param str_ : string contains thread algo name.
  *  @return thread algorithm for appropriate string.
  */
  eb_thread_algo_t str_to_thread_algo(std::string algo);

 private:
  /**
  * @brief Private constructor for singleton pattern.
  *
  * The constructor is private to prevent direct instantiation of the class.
  * Use the @c instance() method to access the single global instance.
  */
  embag_config_t() = default;

  embag_kernel_t embag_kernel;  /**< Embag runtime kernel. */
  eb_thread_algo_t thread_algo; /**< Thread algorithm. */
};

}
}

#endif

