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
#ifndef _ZENDNNL_EXCEPTIONS_HPP_
#define _ZENDNNL_EXCEPTIONS_HPP_

#include <iostream>
#include <exception>
#include <stdexcept>
#include <string>

/** @def EXCEPTION(message)
 *  @brief Throw exception with message.
 */
#define   EXCEPTION(message)                    \
  do                                            \
    throw exception_t(message);                 \
  while(0)

/** @def EXCEPTION_WITH_LOC(message)
 *  @brief Throw exception with message, also give file and line
 *  where exception occurred.
 */
#define   EXCEPTION_WITH_LOC(message)               \
  do                                                \
    throw exception_t{__FILE__, __LINE__, message}; \
  while(0)

namespace zendnnl {
namespace error_handling {

/** @class exception_t
 *  @brief base ZenDNNL exception class.
 *
 * This class can be used as an exception class, and also can be used as a
 * base class for other exception classes in ZenDNNL.
 */
class exception_t : public std::exception {
public:
  /** @name Constructors, Destructors and Assignments
   */
  /**@{*/
  /**
   * @brief Should rarely be used. Use EXECPTION macro instead.
   */
  exception_t(std::string err_msg);

  /**
   * @brief Should rarely be used. Use EXECPTION macro instead.
   */
  exception_t(const char* err_msg);

  /**
   * @brief Should rarely be used. Use EXECPTION_WITH_LOC macro instead.
   */
  exception_t(const char* file, int line, std::string err_msg);

  /**
   * @brief Should rarely be used. Use EXECPTION_WITH_LOC macro instead.
   */
  exception_t(const char* file, int line, const char* err_msg);
  /**@}*/

  /** @name Interface
   */
  /**@{*/
  /**
   * @brief Error message
   * @return A pointer to error messsage.
   */
  const char* what() const noexcept override;
  /**@}*/
private:
  std::string  what_msg; /*!< Error message string. */
};

} //error_handling

namespace interface {
using exception_t = zendnnl::error_handling::exception_t;
} //export

} //zendnnl
#endif
