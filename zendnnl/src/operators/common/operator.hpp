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
#ifndef _OPERATOR_HPP_
#define _OPERATOR_HPP_

#include <cstdint>
#include <cstddef>
#include <string>
#include <memory>

#include "memory/tensor.hpp"
#include "common/zendnnl_global.hpp"
#include "common/zendnnl_api_object.hpp"
#include "operators/common/operator_impl.hpp"

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
template<typename SELF_T,
         typename OP_CONTEXT_T,
         typename OP_IMPL_T= operator_impl_t<OP_CONTEXT_T>>
class operator_t : public api_object_t<SELF_T, OP_IMPL_T> {
 public:
  /** @brief Operator type */
  using   self_type =  SELF_T;
  /** @brief Context type */
  using   context_type =  OP_CONTEXT_T;
  /** @brief Parent type */
  using   parent_type =  api_object_t<SELF_T, OP_IMPL_T>;
  /** @brief implementation type */
  using  impl_type = typename parent_type::impl_type;
  /** @brief implementation type */
  using  impl_sptr_type = typename parent_type::impl_sptr_type;

public:
  /** @brief Virtual destructor */
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
  self_type&  set_context(const context_type& context_);

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
  self_type& create() override;
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
  self_type& set_input(const std::string& key_, const tensor_t& input_tensor_);

  /** @brief Get an input tensor.
   *
   * Returns an input tensor mapped to a key. If no tensor is present
   * return empty optional. If an input is optional, then no tensor may be
   * mapped to its key, so an optional is returned.
   * @param key_ : input tensor key.
   * @return (optional) input tensor.
   */
  std::optional<tensor_t> get_input(const std::string& key_) const;

  /** @brief Set an output tensor.
   *
   * An operator is expected to identify an output tensor by an <em> operator
   * given key </em>. Missing a mandatory input will fail input-output
   * validation and return status_t::op_bad_io error. All mandatory inputs
   * and outputs need to be given before @c execute().
   */
  self_type& set_output(const std::string& key_, const tensor_t& input_tensor_);

  /** @brief Set an output tensor.*/
  std::optional<tensor_t> get_output(const std::string& key_) const;

  /** @brief Set forced kernel.
   *
   * A forced kernel is a kernel enforced by the user. If this kernel is
   * consistent with the inputs and outputs, else status_t::bad_forced_kernel
   * will be returned.
   */
  self_type& set_forced_kernel(const std::string& forced_kernel_);

  /** @brief Get forced kernel.
   *
   * Returns forced kernel name.
   * @return forced kernel name.
   */
  std::string get_forced_kernel() const;

  /** @brief set validation. */
  self_type& set_validation(bool validation_flag_);

  /** @brief get validation. */
  bool get_validation() const;

  /** @brief Execute an operator
   *
   * Operator execution follows setting up all input and output tensors
   * using @c set_input() and @c set_output(). It validates inputs and
   * outputs, selects appropriate kernel, and executes the selected kernel.
   *
   * @return Exexution status. Returns status_t::success on success.
   */
  virtual status_t execute();
  /**@}*/

protected:
  /** @brief Default constructor.
   *
   * ZenDNNL follows the convension of making constructors protected (or private),
   * where the class need to serve as a virtual base class, and no object of
   * the class should be created.
   */
  operator_t() = default;
};

/* implementation */
template<typename SELF_T, typename OP_CONTEXT_T, typename OP_IMPL_T>
typename operator_t<SELF_T, OP_CONTEXT_T, OP_IMPL_T>::self_type&
operator_t<SELF_T, OP_CONTEXT_T, OP_IMPL_T>::set_context(const context_type& context_) {
  /* check the cache before context copy */
  this->hash_key  = context_.get_hash();
  if (this->hash_key) {
    auto cached_value = zendnnl_lru_cache().get_value(this->hash_key);
    if (cached_value) {
      this->impl = std::static_pointer_cast<impl_type>(cached_value.value());
    }
    else {
      this->hash_key = 0;
      this->impl->set_context(context_);
    }
  }
  else {
    this->impl->set_context(context_);
  }

  return dynamic_cast<self_type&>(*this);
}

template<typename SELF_T, typename OP_CONTEXT_T, typename OP_IMPL_T>
typename operator_t<SELF_T, OP_CONTEXT_T, OP_IMPL_T>::context_type
operator_t<SELF_T, OP_CONTEXT_T, OP_IMPL_T>::get_context() {
  return this->impl->get_context();
}

template<typename SELF_T, typename OP_CONTEXT_T, typename OP_IMPL_T>
typename operator_t<SELF_T, OP_CONTEXT_T, OP_IMPL_T>::self_type&
operator_t<SELF_T, OP_CONTEXT_T, OP_IMPL_T>::create() {
  if (! this->hash_key) {
    this->impl->create();

    if (this->cache_flag) {
      if (! this->impl->is_bad_object()) {
      auto hash_key         = this->impl->get_hash();
      auto type_erased_impl = std::static_pointer_cast<void>(this->impl);
      zendnnl_lru_cache().insert(hash_key, type_erased_impl);
      }
      else {
        apilog_error("<", this->get_name(), "> unable to cache a bad object.");
      }
    }
  }
  else {
    apilog_info("< ", this->get_name(), " > operator loaded from cache.");
  }

  return dynamic_cast<self_type&>(*this);
}

template<typename SELF_T, typename OP_CONTEXT_T, typename OP_IMPL_T>
typename operator_t<SELF_T, OP_CONTEXT_T, OP_IMPL_T>::self_type&
operator_t<SELF_T, OP_CONTEXT_T, OP_IMPL_T>::set_input(const std::string& key_,
                                            const tensor_t& input_tensor_) {
  this->impl->set_input(key_, input_tensor_);
  return dynamic_cast<self_type&>(*this);
}

template<typename SELF_T, typename OP_CONTEXT_T, typename OP_IMPL_T>
std::optional<tensor_t>
operator_t<SELF_T, OP_CONTEXT_T, OP_IMPL_T>::get_input(const std::string& key_) const {
  return this->impl->get_input(key_);
}

template<typename SELF_T, typename OP_CONTEXT_T, typename OP_IMPL_T>
typename operator_t<SELF_T, OP_CONTEXT_T, OP_IMPL_T>::self_type&
operator_t<SELF_T, OP_CONTEXT_T, OP_IMPL_T>::set_output(const std::string& key_,
                                             const tensor_t& output_tensor_) {
  this->impl->set_output(key_, output_tensor_);
  return dynamic_cast<self_type&>(*this);
}

template<typename SELF_T, typename OP_CONTEXT_T, typename OP_IMPL_T>
std::optional<tensor_t>
operator_t<SELF_T, OP_CONTEXT_T, OP_IMPL_T>::get_output(const std::string& key_) const {
  return this->impl->get_output(key_);
}

template<typename SELF_T, typename OP_CONTEXT_T, typename OP_IMPL_T>
typename operator_t<SELF_T, OP_CONTEXT_T, OP_IMPL_T>::self_type&
operator_t<SELF_T, OP_CONTEXT_T, OP_IMPL_T>::set_forced_kernel(const std::string& forced_kernel_) {
  this->impl->set_forced_kernel(forced_kernel_);
  return dynamic_cast<self_type&>(*this);
}

template<typename SELF_T, typename OP_CONTEXT_T, typename OP_IMPL_T>
std::string operator_t<SELF_T, OP_CONTEXT_T, OP_IMPL_T>::get_forced_kernel() const {
  return this->impl->get_forced_kernel();
}

template<typename SELF_T, typename OP_CONTEXT_T, typename OP_IMPL_T>
typename operator_t<SELF_T, OP_CONTEXT_T, OP_IMPL_T>::self_type&
operator_t<SELF_T, OP_CONTEXT_T, OP_IMPL_T>::set_validation(bool validation_flag_) {
  this->impl->set_validation(validation_flag_);
  return dynamic_cast<self_type&>(*this);
}

template<typename SELF_T, typename OP_CONTEXT_T, typename OP_IMPL_T>
bool operator_t<SELF_T, OP_CONTEXT_T, OP_IMPL_T>::get_validation() const {
  return this->impl->get_validation();
}

template<typename SELF_T, typename OP_CONTEXT_T, typename OP_IMPL_T>
status_t operator_t<SELF_T, OP_CONTEXT_T, OP_IMPL_T>::execute() {
  return this->impl->execute();
}

} //namespace ops
} //namespace zendnnl

#endif
