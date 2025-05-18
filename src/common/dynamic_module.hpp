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
#ifndef _DYNAMIC_MODULE_HPP_
#define _DYNAMIC_MODULE_HPP_

#ifdef __linux__
#include <dlfcn.h>
#endif

#include <string>
#include "zendnnl_global.hpp"

namespace zendnnl {
namespace common {

using namespace zendnnl::error_handling;

/** @class dynamic_module_t
 *  @brief A wrapper class to load dynamic modules using linux dlopen().
 *
 * This class is a wrapper class that loads a dynamic module using linux
 * dlopen() call at runtime, and exteract a symbol from it. A typical use
 * of this is to create different modules for different machine ISA, and
 * load an appropriate module by detecting runtime machine ISA.
 *
 * Please see [the dlopen() documentation](https://man7.org/linux/man-pages/man3/dlopen.3.html)
 * for more details.
 */
class dynamic_module_t {
public:
  /** @name Constructors, Destructors and Assignments
   */
  /**@{*/
  /**
   * @brief Default constructor.
   */
  dynamic_module_t();
  /**
   * @brief Default destructor.
   *
   * Closes any open module.
   */
  ~dynamic_module_t();
  /**@}*/

  /** @name Interface
   */
  /**@{*/
  /**
   * @brief Set the name of the module to be loaded.
   *
   * Module name will be prefixed with "lib" and suffixed with ".so".
   * For example if module name "op_kernel_avx512" is given, then
   * a module file "libop_kernel_avx512.so" will be searched and loaded.
   * @param module_name_ : module to be loaded.
   * @return A reference to self.
   */
  dynamic_module_t& set_name(std::string module_name_);

  /**
   * @brief Get the name of the mdoule.
   * @return name of the module set wth set_name(), else an empty string.
   */
  std::string       get_name() const;

  /**
   * @brief Load the module to memory.
   *
   * @se If dlopen() fails for any reason, an exception is thrown with dlerror(),
   *     else module_handle is loaded with the module.
   * @return success if module is loaded, else utils_bad_module_name.
   */
  status_t          load();

  /**
   * @brief Unload module from the memory.
   * @se module_handle is set to nullptr.
   */
  void              unload();

  /**
   * @brief Get a symbol from the module.
   *
   * A symbol is generally a function name to be executed.
   * @se If dlsym() fails for some reason, an exception is thrown.
   * @return A pointer to the symbol.
   */
  void*             get_symbol(std::string symbol_name_);
  /**@}*/

private:
  void*       module_handle; /*!< module handle to the mdoule in memory */
  std::string module_name; /*!< name of the module to be loaded */
};

} //common
} //zendnnl
#endif
