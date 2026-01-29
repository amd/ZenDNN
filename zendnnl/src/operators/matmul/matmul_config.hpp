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
enum class matmul_algo_t : int32_t {
  none = -1,             /*!< No algorithm selected */
  dynamic_dispatch = 0,  /*!< Dynamic dispatch */
  aocl_dlp_blocked = 1,  /*!< Blocked AOCL */
  onednn_blocked = 2,    /*!< Blocked OneDNN */
  libxsmm_blocked = 3,   /*!< Blocked LIBXSMM */
  aocl_dlp  = 4,         /*!< AOCL */
  onednn = 5,            /*!< OneDNN */
  libxsmm = 6,           /*!< LIBXSMM */
  batched_sgemm = 7,     /*!< Batched SGEMM */
  auto_tuner = 8,        /*!< Auto Tuner */
  reference = 9,         /*!< Reference */
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
  void set_algo(int32_t algo);

  /** @brief Get matmul algo.
   *
   * @return matmul algo.
   */
  int32_t get_algo();

  /** @brief Sets bmm algo.
  *
  * @param algo The BMM algorithm to set.
  */
  void set_bmm_algo(int32_t algo);

  /** @brief Get bmm algo.
   *
   * @return bmm algo.
   */
  int32_t get_bmm_algo();

  /** @brief Sets matmul_weight_cache.
  *
  * @param weight_cache The matmul_weight_cache type to set.
  */
  void set_weight_cache(int32_t weight_cache);

  /** @brief Get matmul_weight_cache.
   *
   * @return matmul_weight_cache.
   */
  int32_t get_weight_cache();

  /** @brief Sets zp_comp_cache enable flag.
  *
  * @param enable Whether to enable zero-point compensation caching.
  */
  void set_zp_comp_cache(bool enable);

  /** @brief Get zp_comp_cache enable flag.
   *
   * @return true if ZP compensation caching is enabled.
   */
  bool get_zp_comp_cache();

  /** @brief Sets lru_cache_capacity.
  *
  * @param capacity The LRU cache capacity to set.
  */
  void set_lru_cache_capacity(uint32_t capacity);

  /** @brief Get lru_cache_capacity.
   *
   * @return lru_cache_capacity.
   */
  uint32_t get_lru_cache_capacity();

  /** @brief Sets mm_partitioner_enabled flag.
  *
  * @param enabled Whether to enable MM partitioner.
  */
  void set_mm_partitioner_enabled(bool enabled);

  /** @brief Get mm_partitioner_enabled flag.
   *
   * @return true if MM partitioner is enabled.
   */
  bool get_mm_partitioner_enabled();

  /** @brief Sets tile_m size.
  *
  * @param size The tile size for M dimension.
  */
  void set_tile_m(int32_t size);

  /** @brief Get tile_m size.
   *
   * @return tile_m size.
   */
  int32_t get_tile_m();

  /** @brief Sets tile_n size.
  *
  * @param size The tile size for N dimension.
  */
  void set_tile_n(int32_t size);

  /** @brief Get tile_n size.
   *
   * @return tile_n size.
   */
  int32_t get_tile_n();

  /** @brief Sets tile_k size.
  *
  * @param size The tile size for K dimension.
  */
  void set_tile_k(int32_t size);

  /** @brief Get tile_k size.
   *
   * @return tile_k size.
   */
  int32_t get_tile_k();

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

  int32_t matmul_algo;         /**< Matmul runtime algorithm. */
  int32_t bmm_algo;            /**< Batched Matmul algorithm. */
  int32_t matmul_weight_cache; /**< Matmul weight cache type. */
  bool zp_comp_cache;          /**< Enable zero-point compensation caching. */
  uint32_t lru_cache_capacity; /**< LRU cache capacity. */
  bool mm_partitioner_enabled;      /**< Enable MM partitioner. */
  int32_t tile_m;                   /**< Tile size for M dimension. */
  int32_t tile_n;                   /**< Tile size for N dimension. */
};

}
}

#endif