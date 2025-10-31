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
#ifndef _ERROR_STATUS_HPP_
#define _ERROR_STATUS_HPP_

#include <cstdint>

namespace zendnnl {
/** @namespace zendnnl::error_handling
 *  @brief A namespace for classes, functions, variables and enums for error handling.
 *
 *  This namespace contains error code enumerations, exception classes and logger
 *  classes needed to detect, propagate, and act upon errors generated in ZenDNNL.
 */
namespace error_handling {

/** @enum status_t
 *  @brief Error status of a function, an object or a class.
 */
enum class status_t : int32_t {
  success                  = 1,  /*!< Success. Also used to represent a valid object */
  failure                  = 0,  /*!< Unknown failure */
  unimplemented            = -1, /*!< Unimplemented function,kernel or feature */
  bad_hash_object          = -2, /*!< Bad or ill formed hashable object */
  op_bad_context           = -3, /*!< Bad operator context */
  op_bad_io                = -4, /*!< Bad input-output to operator */
  op_bad_forced_kernel     = -5, /*!< Bad or inconsistant forced kernel */
  utils_bad_module_name    = -6, /*!< Bad module name */
  utils_bad_dynamic_module = -7, /*!< Bad dynamic module */
  memory_bad_size          = -8, /*!< bad tensor size */
  memory_bad_aligned_size  = -9, /*!< bad tensor aligned size */
  memory_bad_order         = -10, /*!< bad tensor axis order */
  memory_bad_stride        = -11, /*!< bad tensor stride */
  memory_bad_base          = -12, /*!< bad tensor base */
  memory_bad_index         = -13, /*!< bad tensor index */
  memory_bad_storage       = -14, /*!< bad tensor index */
  config_bad_json_file     = -15, /*!< bad json file */
  memory_bad_meta          = -16,  /*!< bad quantization input to tensor */
  memory_bad_quant         = -17  /*!< bad quantization input to tensor */

};

} //error_handling

namespace interface {
using status_t = zendnnl::error_handling::status_t;
} //export

} //zendnnl
#endif
