/*******************************************************************************
 * Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/
#ifndef  _ZENDNNL_GLOBAL_BLOCK_HPP_
#define  _ZENDNNL_GLOBAL_BLOCK_HPP_

#include <cstdint>
#include <mutex>
#include "common/error_status.hpp"
#include "common/zendnnl_exceptions.hpp"
#include "common/platform_info.hpp"
#include "common/logging.hpp"

namespace zendnnl {
namespace common {

using namespace zendnnl::error_handling;

/** @class zendnnl_global_block_t
 *  @brief A singleton class for all ZenDNNL persistent data structures.
 *
 *  Once initialized this singleton remains persistent across all the calls to
 *  the ZenDNNL library. This class owns the folowing data
 *  - Platform info
 *     contains the information about runtime platform, for
 *     example, ISA supported, number of cores etc. This information can be
 *     used to enable platform specific operator kernels, obtain operator
 *     level parallelism and compare performance with roofline performance.
 *  - Logger
 *     a threadsafe logger.
 *  - Configuration manager
 *     a configuration manager that accepts a runtime configuration from user
 *     and configures ZenDNNL.
 *  - Persistent caches
 *     caches to store any other persistent data like reordered tensors,
 *     JIT generated kernels etc.
 */
class zendnnl_global_block_t {
public:

  /** @brief Get pointer to the singleton
   *
   * Returns a pointer to the siggleton object, creating the object if
   * it is not already created. This call is thread-safe.
   *
   * @return A pointer to the singleton object.
   **/
  static zendnnl_global_block_t* get();

  /** @brief Get platform info
   * @return A reference to platform info object.
   */
  platform_info_t& get_platform_info();

  /** @brief Get logger
   * @return A reference to logger.
   */
  logger_t&        get_logger();

private:
  /** @brief private constructor
   *
   * Made private to prevent multiple instances creation.
   */
  zendnnl_global_block_t();

  /** @brief deleted copy constuctor
   *
   * Deleted to create multiple instances creation.
   */
  zendnnl_global_block_t(const zendnnl_global_block_t&) = delete;

  /** @brief deleted copy assignment
   *
   * Deleted to prevent multiple instances creation.
   */
  zendnnl_global_block_t& operator=(const zendnnl_global_block_t&) = delete;

  static std::mutex              instance_mutex; /*!< mutex for thread safety */
  static zendnnl_global_block_t* instance; /*!< singleton instance pointer */

  platform_info_t                platform_info; /*!< platform info */
  logger_t                       logger; /*!< logger */
};

}//common

// namespace interface{
// using zendnnl::common::zendnnl_global_block_t;
// }//interface

}//zendnnl

#endif
