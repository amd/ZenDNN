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
#ifndef _OPERATOR_HPP_
#define _OPERATOR_HPP_

#include <cstdint>
#include <cstddef>
#include <string>
#include <memory>

#include "common/zendnnl_global.hpp"
#include "common/platform_info.hpp"
#include "common/dynamic_module.hpp"
#include "common/hash_object.hpp"
#include "memory/tensor.hpp"
#include "operators/common/operator_context.hpp"
#include "operators/common/operator_kernel.hpp"

// static_assert(std::is_base_of_v<op_context_base_t<OP_CONTEXT_T>, OP_CONTEXT_T>,
//               "OP_CONTEXT_T should be derived from op_context_base_t");

namespace zendnnl {
namespace ops {

using namespace zendnnl::common;
using namespace zendnnl::profile;
using namespace zendnnl::error_handling;

/** @class operator_t
 *  @brief A class to implement computation of a node of a computational graph.
 *
 *  An operator in a computational graph is a node that takes input tensors,
 *  performs some computations on them and produces output tensors. An operator
 *  can implement some simple computation on input tensors (add, concat...),
 *  computations involving other tensors acting as operator parameters (matrix
 *  multiplication with weight and bias as operator parameters), or a complex
 *  subgraph (attention layer in LLMs).
 *
 *  operator parameter tensors (for example weight and bias), and any other
 *  parameters (for example element-wise post-ops, or embedding bag algorithm
 *  like add, mean, or max), needed to implement operator computation are called
 *  operator context. An operator context is derived from @c op_context_t.
 *
 *  An operator receives its context at its creation. It validates its context,
 *  and preprocesses parameters for any acceletated computation.
 *
 *  An operator can have multiple implementations of its computation. These
 *  implementations are called operator kernels. Operator kernels are derived
 *  from @c op_kernel_t. Kernel selection depends on the following
 *
 *  1. Different computations or backends for different problem sizes.
 *  2. Input/output quantization level.
 *  3. Machine ISA/Heterogeneous compute requirements.
 *
 *  A kernel preferably consists of optimized code only for computation. Any
 *  decision that can be taken at kernel selection level or before that need not
 *  be part of a kernel. For example a kernel preferably should not have code
 *  that takes different paths depending on machine ISA.
 *
 * @todo Move dynamic module loading from operator level to ZenDNNL level. ZenDNNL
 * should sense machine ISA at initialization and load appropriate module for
 * all operators.
 *
 * @todo Add support for reference kernel and user overriding kernel selection.
 */
template<typename OP_T, typename OP_CONTEXT_T>
class operator_t : public hash_object_t {
 public:
  /** @brief Parent type */
  using   parent_type                =  hash_object_t;
  /** @brief Operator type */
  using   operator_type              =  OP_T;
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

  /** @brief Virtual destructor
   *
   *  Virtual since this class acts as virtual base class.
   */
  virtual ~operator_t() = default;

  /**@name Create
   */
  /**@{*/
  /** @brief Set operator context.
   *
   * operator parameter tensors (for example weight and bias), and any other
   * parameters (for example element-wise post-ops, or embedding bag algorithm
   * like add, mean, or max), needed to implement operator computation are called
   * operator context. An operator context is derived from @c op_context_t.
   *
   * Operator context need to be given before operator creation, else creation will
   * fail.
   * @param context_ : operator context.
   * @return A reference to self.
   */
  OP_T         &set_context(const context_type &context_);

  /** @brief Get operator context.
   *
   * Please see @ set_context() for context description.
   * @todo make this function const.
   * @return Operator context.
   */
  context_type  get_context();

  /** @brief Create an operator
   *
   * Operator creation follows setting up the context using @c set_context().
   * Creation validates the context, pre-processes the context (like
   * reordering tensors for accelerated computations) and creates the object.
   *
   * Derived operators can either add more functionality or override it.
   * @se operator last status is set to status_t::success in case of successful
   * creation. The status can be checked using @c hash_object_t.check().
   * @return A reference to self.
   */
  virtual OP_T       &create();
  /**@}*/

  /**@name Execute
   */
  /**@{*/
  /** @brief Set an input tensor.
   *
   * An operator is expected to identify an input tensor by an <em> operator
   * given key </em>. Missing a mandatory input will fail input-output
   * validation and return status_t::op_bad_io error. All mandatory inputs
   * and outputs need to be given before @c execute().
   */
  OP_T         &set_input(std::string key_, tensor_t &input_tensor_);

  /** @brief Get an input tensor.
   *
   * Returns an input tensor mapped to a key. If no tensor is present
   * return empty optional. If an input is optional, then no tensor may be
   * mapped to its key, so an optional is returned.
   * @param key_ : input tensor key.
   * @return (optional) input tensor.
   */
  std::optional<tensor_t> get_input(std::string key_);

  /** @brief Set forced kernel.
   *
   * A forced kernel is a kernel enforced by the user. If this kernel is
   * consistent with the inputs and outputs, else status_t::bad_forced_kernel
   * will be returned.
   */
  OP_T         &set_forced_kernel(std::string forced_kernel_name_);

  /** @brief Get forced kernel.
   *
   * Returns forced kernel name.
   * @return forced kernel name.
   */
  std::string get_forced_kernel();

  /** @brief Set an output tensor.
   *
   * An operator is expected to identify an output tensor by an <em> operator
   * given key </em>. Missing a mandatory input will fail input-output
   * validation and return status_t::op_bad_io error. All mandatory inputs
   * and outputs need to be given before @c execute().
   */
  OP_T         &set_output(std::string key_, tensor_t &input_tensor_);
  std::optional<tensor_t> get_output(std::string key_);

  /** @brief Execute an operator
   *
   * Operator execution follows setting up all input and output tensors
   * using @c set_input() and @c set_output(). It validates inputs and
   * outputs, selects appropriate kernel, and executes the selected kernel.
   *
   * @return Exexution status. Returns status_t::success on success.
   */
  virtual status_t    execute();
  /**@}*/

  /** @name Profiling and Diagnostics
   */
  /**@{*/
  /** @brief Set name.
   *
   * Name is relevant only for object identification in logging, profiling
   * and diagnostics. Default is "unknown operator".
   * @param name_ : object name.
   * @return A reference to self.
   */
  OP_T         &set_name(std::string name_);

  /** @brief Get name.
   * @return Object name.
   */
  std::string   get_name();
  /**@}*/

  std::size_t   hash() override;

 protected:
  /** @brief Default constructor.
   *
   * ZenDNNL follows the convension of making constructors protected (or private),
   * where the class need to serve as a virtual base class, and no object of
   * the class should be created.
   */
  operator_t();

  /** @brief Load dynamic module
   *
   * This is useful only if the operator is loading
   * kernel modules at runtime. please see @c dynamic_module_t further.
   * @se Raises an exception if module is not found on the disk.
   * @param module_ : module name.
   * @return status_t::success for success.
   */
  status_t load_module(std::string module_);

  /** @brief Load a kernel from a dynamic module.
   *
   * This is useful only if the operator is loading
   * kernel modules at runtime. The module should already have been loaded
   * with @c load_module(). please see @c dynamic_module_t further.
   * @se Raises an exception if the kernel is not found in the module.
   * @param module_ : module name.
   * @return status_t::success for success.
   */
  status_t load_kernel(std::string symbol_);

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

  /** @brief Returns operator information.
   *
   * Returns a string containing opearator meta data.
   * This includes operator name, context information, input and
   * output tensor information. This is used for logging and profiling.
   * @return std:string containing operator information.
   */
  virtual std::string operator_info();

  //data
  tensor_map_type                      inputs; /**< Input tensors. */
  tensor_map_type                      outputs; /**< Output tensors. */
  context_type                         context; /**< Operator context. */
  std::string                          name; /**< Name for diagnostic purpose */
  kernel_sptr_type                     kernel; /**< Pointer to the kernel chosen
                                                  for execution. */
  std::shared_ptr<dynamic_module_t>
  dynamic_module; /**< To load dynamic modules. */
  std::string                          forced_kernel;
  platform_info_t                      platform_info; /**< HW platform info. */
};

//implementation
template<typename OP_T, typename OP_CONTEXT_T>
operator_t<OP_T, OP_CONTEXT_T>::operator_t():
  context{}, name{"unknown operator"}, kernel{nullptr},
  dynamic_module{std::make_shared<dynamic_module_t>()},
  forced_kernel{} {
  platform_info = zendnnl_platform_info();
}

template<typename OP_T, typename OP_CONTEXT_T>
OP_T &operator_t<OP_T, OP_CONTEXT_T>::set_context(const OP_CONTEXT_T
    &context_) {
  LOG_DEBUG_INFO("<", name, "> Setting context for operator_t");
  if (status != status_t::success) {
    context = context_;
  }

  return dynamic_cast<OP_T &>(*this);
}

template<typename OP_T, typename OP_CONTEXT_T>
OP_CONTEXT_T operator_t<OP_T, OP_CONTEXT_T>::get_context() {
  LOG_DEBUG_INFO("<", name, "> Getting context for operator_t");
  return context;
}

template<typename OP_T, typename OP_CONTEXT_T>
OP_T &operator_t<OP_T, OP_CONTEXT_T>::set_name(std::string name_) {
  LOG_DEBUG_INFO("<", name, "> Setting name for operator_t as ", name_);
  name = name_;
  return dynamic_cast<OP_T &>(*this);
}

template<typename OP_T, typename OP_CONTEXT_T>
std::string operator_t<OP_T, OP_CONTEXT_T>::get_name() {
  LOG_DEBUG_INFO("<", name, "> Getting name for operator_t");
  return name;
}

template<typename OP_T, typename OP_CONTEXT_T>
OP_T &operator_t<OP_T, OP_CONTEXT_T>::create() {
  LOG_DEBUG_INFO("<", name, "> Creating operator");
  apilog_info("Operator create - ",name);

  if (status != status_t::success) {
    if (! context.check()) {
      log_error("operator <", name, "> : bad context");
      status =  status_t::op_bad_context;
      return static_cast<OP_T &>(*this);
    }

    // make operator complete
    status = status_t::success;
    hash();
  }
  return dynamic_cast<OP_T &>(*this);
}

template<typename OP_T, typename OP_CONTEXT_T>
status_t operator_t<OP_T, OP_CONTEXT_T>::execute() {
  LOG_DEBUG_INFO("<",name, "> Executing operator");
  apilog_info("Operator execute - ",name,",",operator_info());

  try {
    // check if pre_processing is successful
    if (status != status_t::success) {
      log_error("<", name, "> bad object");
      return status;
    }

    //sanity check on io
    if (validate() != status_t::success) {
      log_error("<", name, "> bad input or output");
      return status_t::op_bad_io;
    }

    //sanity chck forced kernel
    if (validate_forced_kernel() != status_t::success) {
      log_error("<", name, "> bad or inconsistent forced kernel");
      return status_t::op_bad_forced_kernel;
    }

    //kernel factory assigns a kernel
    if (kernel_factory() != status_t::success) {
      apilog_error("<", name, "> failed to generate kernel");
      return status_t::failure;
    }

    // create a profiler instance
    profiler_t obj;

    //start the timer
    obj.tbp_start();

    //execute kernel
    if (kernel->execute(context, inputs, outputs) != status_t::success) {
      log_error("<", name, "> kernel execution failed");
      return status_t::failure;
    }

    //stop the timer
    obj.tbp_stop();

    profilelog_info("Operator execute - ",name,",",operator_info(),
                    ",time:",obj.tbp_elapsedtime(),obj.get_res_str());

    //cleanup
    kernel.reset();
    inputs.clear();
    outputs.clear();

  }
  catch (const exception_t &ex) {
    EXCEPTION_WITH_LOC(ex.what());
  }

  return status_t::success;
}

template<typename OP_T, typename OP_CONTEXT_T>
OP_T &operator_t<OP_T, OP_CONTEXT_T>::set_input(std::string key_,
    tensor_t &input_tensor_) {
  LOG_DEBUG_INFO("<", name, "> Setting input tensor for ", key_,
                 " for operator_t");
  if (status == status_t::success) {
    inputs[key_] = input_tensor_;
  }

  return dynamic_cast<OP_T &>(*this);
}

template<typename OP_T, typename OP_CONTEXT_T>
std::optional<tensor_t> operator_t<OP_T, OP_CONTEXT_T>::get_input(
  std::string key_) {
  LOG_DEBUG_INFO("<", name, "> Getting input tensor for ", key_,
                 " for operator_t");
  if (status == status_t::success) {
    for (const auto& [k, v] : inputs) {
      if (k == key_) {
        return v;
      }
    }
  }

  return std::nullopt;
}

template<typename OP_T, typename OP_CONTEXT_T>
OP_T &operator_t<OP_T, OP_CONTEXT_T>::set_output(std::string key_,
    tensor_t &output_tensor_) {
  LOG_DEBUG_INFO("<", name, "> Setting output tensor for ", key_,
                 " for operator_t");
  if (status == status_t::success) {
    outputs[key_] = output_tensor_;
  }

  return dynamic_cast<OP_T &>(*this);
}

template<typename OP_T, typename OP_CONTEXT_T>
std::optional<tensor_t> operator_t<OP_T, OP_CONTEXT_T>::get_output(
  std::string key_) {
  LOG_DEBUG_INFO("<", name, "> Getting output tensor for ", key_,
                 " for operator_t");
  if (status == status_t::success) {
    for (const auto& [k, v] : outputs) {
      if (k == key_) {
        return v;
      }
    }
  }

  return std::nullopt;
}

template<typename OP_T, typename OP_CONTEXT_T>
OP_T &operator_t<OP_T, OP_CONTEXT_T>::set_forced_kernel(
  std::string forced_kernel_) {
  LOG_DEBUG_INFO("<", name, "> Setting forced kernel operaor_t");
  if (status == status_t::success) {
    forced_kernel = forced_kernel_;
  }

  return dynamic_cast<OP_T &>(*this);
}

template<typename OP_T, typename OP_CONTEXT_T>
std::string operator_t<OP_T, OP_CONTEXT_T>::get_forced_kernel() {
  LOG_DEBUG_INFO("<", name, "> Getting forced kernel for operator_t");
  if (status == status_t::success) {
    return forced_kernel;
  }

  return std::string();
}

template<typename OP_T, typename OP_CONTEXT_T>
std::size_t operator_t<OP_T, OP_CONTEXT_T>::hash() {
  LOG_DEBUG_INFO("<", name, "> Getting hash for operator_t");
  if (status == status_t::success) {
    if (hash_key) {
      return hash_key;
    }
    hash_key =  context.hash();
  }

  return hash_key;
}

template<typename OP_T, typename OP_CONTEXT_T>
status_t operator_t<OP_T, OP_CONTEXT_T>::load_module(std::string module_) {
  LOG_DEBUG_INFO("<", name, "> Loading dynamic module operator_t");
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

template<typename OP_T, typename OP_CONTEXT_T>
status_t operator_t<OP_T, OP_CONTEXT_T>::load_kernel(std::string symbol_) {
  LOG_DEBUG_INFO("<", name, "> Loading dynamic kernel operator_t");
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

template<typename OP_T, typename OP_CONTEXT_T>
status_t operator_t<OP_T, OP_CONTEXT_T>::validate() {
  LOG_DEBUG_INFO("<", name, "> Validating operator_t");
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

template<typename OP_T, typename OP_CONTEXT_T>
status_t operator_t<OP_T, OP_CONTEXT_T>::validate_forced_kernel() {
  LOG_DEBUG_INFO("<", name, "> Validating forced kernel operator_t");
  if (! forced_kernel.empty()) {
    return status_t::failure;
  }

  return status_t::success;
}

template<typename OP_T, typename OP_CONTEXT_T>
std::string operator_t<OP_T, OP_CONTEXT_T>::operator_info() {
  LOG_DEBUG_INFO("Getting operator info");
  return "";
}

} //namespace ops
} //namespace zendnnl

#endif
