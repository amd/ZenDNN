/********************************************************************************
# * Copyright (c) 2025-2028 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef _OPERATOR_IMPL_HPP_
#define _OPERATOR_IMPL_HPP_

#include <cstdint>
#include <cstddef>
#include <string>
#include <memory>
#include <optional>

#include "common/zendnnl_global.hpp"
#include "common/platform_info.hpp"
#include "common/dynamic_module.hpp"
#include "common/hashable_object.hpp"
#include "memory/tensor.hpp"
#include "operators/common/operator_context.hpp"
#include "operators/common/operator_kernel.hpp"

namespace zendnnl {
namespace ops {

using namespace zendnnl::common;
using namespace zendnnl::profile;
using namespace zendnnl::error_handling;

template<typename OP_CONTEXT_T>
class operator_impl_t : public hashable_object_t {
public:
  /** @brief Parent type */
  using   parent_type                =  hashable_object_t;
  /** @brief Context type */
  using   context_type               =  OP_CONTEXT_T;
  /** @brief Kernel type */
  using   kernel_type                =  op_kernel_t<OP_CONTEXT_T>;
  /** @brief Shared pointer to kernels */
  using   kernel_sptr_type           =  std::shared_ptr<kernel_type>;
  /** @brief A map type from strings to tensors */
  using   tensor_map_type            =  std::map<std::string, tensor_t>;
  /** @brief Kernel handle type */
  using   create_kernel_handle_type  =  kernel_sptr_type(*)();
  /** @brief dynamic module handle type */
  using   dynamic_module_sptr_type  =  std::shared_ptr<dynamic_module_t>;

public:
  /** @brief default constructor */
  operator_impl_t();

  /** @brief virtual destructor */
  virtual ~operator_impl_t() = default;

public:
  /** @brief set context */
  void set_context(const context_type& context_);

  /** @brief get context */
  const context_type&  get_context() const;

  /** @brief set an input tensor. */
  void set_input(const std::string& key_, const tensor_t& input_tensor_);

  /** @brief set an input tensor. */
  std::optional<tensor_t> get_input(const std::string& key_) const;

  /** @brief set an output tensor. */
  void set_output(const std::string& key_, const tensor_t& input_tensor_);

  /** @brief get an output tensor. */
  std::optional<tensor_t> get_output(const std::string& key_) const;

  /** @brief set forced kernel. */
  void set_forced_kernel(const std::string& forced_kernel_name_);

  /** @brief get forced kernel. */
  std::string get_forced_kernel() const;

  /** @brief set validation. */
  void set_validation(bool validation_flag_);

  /** @brief get validation. */
  bool get_validation() const;

  /** @brief create */
  void create();

  /** @brief execute an operator */
  virtual status_t    execute();

protected:
  /** @brief Load dynamic module
   *
   * This is useful only if the operator is loading
   * kernel modules at runtime. please see @c dynamic_module_t further.
   * @se Raises an exception if module is not found on the disk.
   * @param module_ : module name.
   * @return status_t::success for success.
   */
  status_t load_module(const std::string& module_);

  /** @brief Load a kernel from a dynamic module.
   *
   * This is useful only if the operator is loading
   * kernel modules at runtime. The module should already have been loaded
   * with @c load_module(). please see @c dynamic_module_t further.
   * @se Raises an exception if the kernel is not found in the module.
   * @param module_ : module name.
   * @return status_t::success for success.
   */
  status_t load_kernel(const std::string& symbol_);

  /** @brief Validate input and output tensors.
   *
   * Basic validation consists of checking if all inputs and outputs are
   * valid tensors. Any other validation can be implemented by overriding.
   * @return status_t::success if successful, else status_t::failure.
   */
  virtual status_t    validate();

  /** @brief Validate forced kernel.
   *
   * Basic validation includes only validating that forced kernel is
   * empty. If a forced kernel is given, it returns failure and subsequently
   * execute will fail. Derived operators need to override this and implement
   * their own validation.
   * @return status_t::success if successful, else status_t::failure.
   */
  virtual status_t    validate_forced_kernel();

  /** @brief Select a kernel for execution.
   *
   * Kernel selection depends on multiple factors like input/parameters
   * quantization, machine ISA, problem size or backend library. An
   * operator overrides this to provide its own implementation.
   * @se A kernel is selected for execution.
   * @return status_t::success if successful, else status_t::failure.
   */
  virtual status_t    kernel_factory()  = 0;

  /** @brief Returns operator information. */
  virtual std::string op_create_info();

  /** @brief Returns operator information. */
  virtual std::string op_execute_info();

  /** @brief override reset */
  void reset() override;

  /** @brief override hash */
  std::size_t hash() override;

protected:
  tensor_map_type           inputs; /**< Input tensors. */
  tensor_map_type           outputs; /**< Output tensors. */
  context_type              context; /**< Operator context. */
  kernel_sptr_type          kernel; /**< Kernel ptr for execution. */
  dynamic_module_sptr_type  dynamic_module; /**< To load dynamic modules. */
  std::string               forced_kernel; /**< forced kernel name */
  platform_info_t           platform_info; /**< HW platform info. */
  bool                      validation_flag; /**< To validate io or not */
};

/** implementation */
template<typename OP_CONTEXT_T>
operator_impl_t<OP_CONTEXT_T>::operator_impl_t():
  inputs{}, outputs{}, context{},
  kernel{nullptr}, dynamic_module{std::make_shared<dynamic_module_t>()},
  forced_kernel{}, validation_flag{true} {

  platform_info = zendnnl_platform_info();
}

template<typename OP_CONTEXT_T>
void operator_impl_t<OP_CONTEXT_T>::set_context(const context_type& context_) {
  if (status != status_t::success) {
    context = context_;
  }
}

template<typename OP_CONTEXT_T>
const typename operator_impl_t<OP_CONTEXT_T>::context_type&
operator_impl_t<OP_CONTEXT_T>::get_context()  const {
  return context;
}

template<typename OP_CONTEXT_T>
void operator_impl_t<OP_CONTEXT_T>::set_input(const std::string& key_,
                                              const tensor_t& input_tensor_) {
  if (status == status_t::success) {
    inputs[key_] = input_tensor_;
  }
}

template<typename OP_CONTEXT_T>
std::optional<tensor_t>
operator_impl_t<OP_CONTEXT_T>::get_input(const std::string& key_) const {
  if (status == status_t::success) {
    auto search_ptr = inputs.find(key_);
    if (search_ptr != inputs.end()) {
      return search_ptr->second;
    }
  }

  return std::nullopt;
}

template<typename OP_CONTEXT_T>
void operator_impl_t<OP_CONTEXT_T>::set_output(const std::string& key_,
                                              const tensor_t& output_tensor_) {
  if (status == status_t::success) {
    outputs[key_] = output_tensor_;
  }
}

template<typename OP_CONTEXT_T>
std::optional<tensor_t>
operator_impl_t<OP_CONTEXT_T>::get_output(const std::string& key_) const {
  if (status == status_t::success) {
    auto search_ptr = outputs.find(key_);
    if (search_ptr != outputs.end()) {
      return search_ptr->second;
    }
  }

  return std::nullopt;
}

template<typename OP_CONTEXT_T>
void operator_impl_t<OP_CONTEXT_T>::set_forced_kernel(const std::string& forced_kernel_) {
  if (status == status_t::success) {
    forced_kernel = forced_kernel_;
  }
}

template<typename OP_CONTEXT_T>
std::string operator_impl_t<OP_CONTEXT_T>::get_forced_kernel() const {
  return forced_kernel;
}

template<typename OP_CONTEXT_T>
void operator_impl_t<OP_CONTEXT_T>::set_validation(bool validation_flag_) {
  if (status != status_t::success) {
    validation_flag = validation_flag_;
  }
}

template<typename OP_CONTEXT_T>
bool operator_impl_t<OP_CONTEXT_T>::get_validation() const {
    return validation_flag;
}

template<typename OP_CONTEXT_T>
void operator_impl_t<OP_CONTEXT_T>::create() {

  /* return if alread created */
  if (status == status_t::success)
    return;

  /* enable time based profiling */
  profiler_t profiler;
  bool profile_enabled = is_profile_enabled();

  if (profile_enabled) {
    profiler.tbp_start();
  }

  /* check if the context is proper */
  if (! context.check()) {
    log_error("operator <", obj_name, ">:bad context");
    status = status_t::op_bad_context;
    return;
  }

  /* check for operator name */
  if (is_unnamed_object()) {
    apilog_warning("An unnamed operator is being created.");
  }

  /* check for validation flag */
  if (! validation_flag) {
    apilog_warning("<", obj_name, "> validation is off. this can be error-prone.");
  }

  /* make the operator complete */
  status = status_t::success;

  /* generate hash */
  hash();

  /* print to api log */
  if (apilog_verbose_enabled()) {
    apilog_verbose(op_create_info());
  }

  /* print profiler info */
  if (profile_enabled) {
    profiler.tbp_stop();

    if(profilelog_verbose_enabled()) {
      profilelog_verbose(op_create_info(),
                         ",:time:",
                         profiler.tbp_elapsedtime(),
                         profiler.get_res_str());
    }
  }
}

template<typename OP_CONTEXT_T>
status_t operator_impl_t<OP_CONTEXT_T>::execute() {
  /* enable time based profiling */
  profiler_t profiler;
  bool profile_enabled = is_profile_enabled();
  if (profile_enabled) {
      profiler.tbp_start();
    }

  /* check if the operator is a valid operator */
  if (status != status_t::success) {
    apilog_error("<", obj_name, "> bad operator.");
    return status;
  }

  /* check if io is valid */
  if (validation_flag) {
    if (validate() != status_t::success) {
      apilog_error("<", obj_name, "> bad input or output");
      return status_t::op_bad_io;
    }

    /* sanity chck forced kernel */
    if (validate_forced_kernel() != status_t::success) {
      apilog_error("<", obj_name, "> bad or inconsistent forced kernel");
      return status_t::op_bad_forced_kernel;
    }
  }

  /* get and execute kernel */
  std::string execution_info = std::string();
  try {
    /* kernel factory assigns a kernel */
    if (kernel_factory() != status_t::success) {
      apilog_error("<", obj_name, "> failed to generate kernel");
      return status_t::failure;
    }

    /* execute kernel */
    if (kernel->execute(context, inputs, outputs) != status_t::success) {
      apilog_error("<", obj_name, "> kernel execution failed");
      return status_t::failure;
    }

    if (apilog_verbose_enabled()) {
      execution_info = op_execute_info();
      apilog_verbose(execution_info);
    }

    /* cleanup after execution */
    kernel.reset();
    inputs.clear();
    outputs.clear();
  }
  catch (const exception_t &ex) {
    EXCEPTION_WITH_LOC(ex.what());
  }

  if (profile_enabled) {
      profiler.tbp_stop();
      if (profilelog_verbose_enabled()) {
        profilelog_verbose(execution_info,
                           ",time:",
                           profiler.tbp_elapsedtime(),
                           profiler.get_res_str());
      }
  }

  return status_t::success;
}

template<typename OP_CONTEXT_T>
status_t operator_impl_t<OP_CONTEXT_T>::load_module(const std::string& module_) {
  try {
    if ((*dynamic_module).set_name(module_).load() != status_t::success) {
      EXCEPTION_WITH_LOC("dynamic module load failed.");
    }
  }
  catch (const exception_t &ex) {
    EXCEPTION_WITH_LOC(ex.what());
  }

  return status_t::success;
}

template<typename OP_CONTEXT_T>
status_t operator_impl_t<OP_CONTEXT_T>::load_kernel(const std::string& symbol_) {
  try {
    create_kernel_handle_type create_kernel_handle =
      reinterpret_cast<create_kernel_handle_type>(dynamic_module->get_symbol(
            symbol_));

    if (! create_kernel_handle) {
      EXCEPTION_WITH_LOC("dynamic symbol load returned null.");
    }

    kernel = create_kernel_handle();
  }
  catch (const exception_t &ex) {
    EXCEPTION_WITH_LOC(ex.what());
  }

  return status_t::success;
}

template<typename OP_CONTEXT_T>
status_t operator_impl_t<OP_CONTEXT_T>::validate() {

  for (const auto& [k, v] : inputs) {
    std::optional<tensor_t> t = v;
    if (t && !(t->check())) {
      return status_t::failure;
    }
  }

  for (const auto& [k, v] : outputs) {
    std::optional<tensor_t> t = v;
    if (t && !(t->check())) {
      return status_t::failure;
    }
  }

  return status_t::success;
}

template<typename OP_CONTEXT_T>
status_t operator_impl_t<OP_CONTEXT_T>::validate_forced_kernel() {
  if (! forced_kernel.empty()) {
    return status_t::failure;
  }

  return status_t::success;
}

template<typename OP_CONTEXT_T>
std::string operator_impl_t<OP_CONTEXT_T>::op_create_info() {
  return std::string();
}

template<typename OP_CONTEXT_T>
std::string operator_impl_t<OP_CONTEXT_T>::op_execute_info() {
  return std::string();
}

template<typename OP_CONTEXT_T>
void operator_impl_t<OP_CONTEXT_T>::reset() {
  parent_type::reset();

  inputs.clear();
  outputs.clear();

  context         = context_type{};
  kernel          = nullptr;
  dynamic_module  = std::make_shared<dynamic_module_t>();
  forced_kernel   = std::string();
  validation_flag = true;
}

template<typename OP_CONTEXT_T>
std::size_t operator_impl_t<OP_CONTEXT_T>::hash() {
  if (status == status_t::success) {
    if (! hash_key) {
      hash_key = context.hash();
    }
  }

  return hash_key;
}

} //ops
} //zendnnl

#endif
