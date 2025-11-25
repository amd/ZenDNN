/********************************************************************************
# * Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef _ZENDNNL_API_MACROS_HPP_
#define _ZENDNNL_API_MACROS_HPP_

#include "common/zendnnl_object.hpp"
#include "common/hashable_object.hpp"
#include "common/zendnnl_global.hpp"

#define CHECK_AND_THROW_EXCEPTION(OBJ)                                  \
  do {                                                                  \
    if (OBJ.get_last_status()                                           \
        != zendnnl::error_handling::status_t::success) {                \
      std::string message = "Invalid object ";                          \
      message += OBJ.get_name();                                        \
      throw zendnnl::error_handling::exception_t(message);              \
    }                                                                   \
  }  while(0);                                                          \

/** @def CHECK_AND_LOG_ERROR(object_)
 *  @brief supporting macro to check the status of an object and log error
 */

#define CHECK_AND_LOG_ERROR(object_)                                    \
  do {                                                                  \
    if (OBJ.get_last_status()                                           \
        != zendnnl::error_handling::status_t::success) {                \
      std::string message = "Invalid object ";                          \
      message += OBJ.get_name();                                        \
      apilog_error(message);                                            \
    }                                                                   \
  }  while(0);                                                          \

#endif
