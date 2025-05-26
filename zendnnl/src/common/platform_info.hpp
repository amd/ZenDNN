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
#ifndef _PLATFORM_INFO_HPP_
#define _PLATFORM_INFO_HPP_

#include <string>
#include <cstdint>
#include "common/error_status.hpp"
#include "common/zendnnl_exceptions.hpp"
#include "Au/Cpuid/CacheInfo.hh"
#include "Au/Cpuid/X86Cpu.hh"

namespace zendnnl {
namespace common {

using namespace Au;
using namespace zendnnl::error_handling;

/** @class platform_info_t
 *  @brief A class to query and hold all platform information.
 *
 *  Generally an object of this class is appended to @c zendnnl_global_block_t.
 *  It is populated when @c zendnnl_global_block_t gets initialized.
 */

class platform_info_t final {
public:
  /** @name Constructors, Destructors and Assignments */
  /**@{*/
  /** @brief Default constrcutor */
  platform_info_t();
  /**@}*/

  /** @name Interface */
  /**@{*/
  /** @brief Populate platform info
   * @return status_t::success
   */
  status_t populate();

  /** @brief Get avx2 status
   *  @return true if platform supports avx2 else false.
   */
  bool get_avx2_status() const;

  /** @brief Get avx512 status
   *  @return true if platform supports avx512 else false.
   */
  bool get_avx512f_status() const;

  /** @brief Get isa version
   *  @return isa version.
   */
  uint32_t get_isa_version() const;

  /** @brief Get cpu family
   *  @return cpu family.
   */
  uint32_t get_cpu_family() const;

  /** @brief Get cpu model
   *  @return cpu model.
   */
  uint32_t get_cpu_model() const;

  /** @brief Get cpu vendor
   *  @return cpu vendor.
   */
  uint32_t get_cpu_vendor() const;

  /** @brief Get cpu uarch
   *  @return cpu uarch.
   */
  uint32_t get_cpu_uarch() const;
  /**@}*/

private:
  bool          is_avx2;
  bool          is_avx512f;
  uint32_t      isa_version;
  uint32_t      cpu_family;
  uint32_t      cpu_model;
  uint32_t      cpu_vendor;
  uint32_t      cpu_uarch;
};

} //common
} //zendnnl
#endif
