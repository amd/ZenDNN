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
#ifndef _ZENAI_OPERATOR_CONTEXT_HPP_
#define _ZENAI_OPERATOR_CONTEXT_HPP_

#include <vector>
#include <map>
#include <cstdint>
#include <string>
#include <optional>
#include <functional>

#include "common/zendnnl_global.hpp"
#include "common/hash_object.hpp"
#include "memory/tensor.hpp"
#include "operators/common/post_op.hpp"

namespace zendnnl {
/** @namespace zendnnl::ops
 *  @brief A namespace for all classes, functions, variables and enums related
 *  to operators.
 */
namespace ops {

using namespace zendnnl::memory;
using namespace zendnnl::common;

/** @class op_context_t
 *  @brief A base class for operator context.
 *
 *  Given an operator, its parameter tensors (for example weight and bias),
 *  and any other parameters (for example element-wise post-ops, or embedding bag
 *  algorithm like add, mean, or max), needed to implement operator computation
 *  are called operator context of the operator.
 *
 *  Since operator context is copied into operator, and may be further to kernels,
 *  a derived context need to take care that it should not have raw pointers, or
 *  raw memory allocations that are shared across copies. If a raw shared memory
 *  allocation happens, then memory management becomes problematic. Any shared
 *  allocation need to be wrapped in shared smart pointer.
 *
 *  @todo Remove the requirement that set_opst_op() need to be given in order.
 *  Provide index of the post op with it.
 */
template<typename OP_CONTEXT_T>
class op_context_t : public hash_object_t {
 public:
  /** @brief Parent type. */
  using   parent_type       = hash_object_t;

  /** @brief Derived context type as template parameter. */
  using   context_type      = OP_CONTEXT_T;

  /** @brief A map type for parameter tensors. */
  using   params_map_type   = std::map<std::string, tensor_t>;

  /** @brief A vector type for post ops. */
  using   post_op_vec_type  = std::vector<post_op_t>;

  /** @brief Virtual destructor
   *
   *  Virtual since this class acts as virtual base class.
   */
  virtual ~op_context_t();

  /** @name Create */
  /**@{*/
  /** @brief Set parameter tensor.
   * An operator is expected to identify an parameter tensor by an <em> operator
   * given key </em>. Parameters are validated using @c validate().
   * Missing a mandatory parameter should result in creation failure.
   * All mandatory parameters need to be given before @c create().
   * @param key_ : key given by operator for parameter identification.
   * @param param_tensor_ : parameter tensor.
   * @reurn A referene to self.
   */
  OP_CONTEXT_T &set_param(std::string key_, tensor_t &param_tensor_);

  /** @brief Get parameter tensor.
   *
   * Please see @c set_param() for detailed description.
   * @param key_ : key given by operator for parameter identification.
   * @return (optional) parameter tensor.
   */
  std::optional<tensor_t> get_param(std::string key_) const;

  /** @brief Set post op.
   *
   * Post op refers to a localized computation (mostly elementwise
   * non-linear operators) that can be performed on the output of an
   * operator when the output data is still "hot" in cache. A post_op
   * can be fused with an operator. multiple post-ops can follow an
   * operator.
   *
   * This is a function that need to be given in order in which post-op
   * are needed to be computed.
   * @param post_op_ : post_op.
   * @return A reference to self.
   */
  OP_CONTEXT_T &set_post_op(post_op_t &post_op_);

  /** @brief Get post op.
   *
   * Please see @c set_post_op() for further description.
   * @param i_ : index to post op vector.
   * @return Post_op at the given index.
   */
  post_op_t     get_post_op(uint32_t i_) const;

  /** @brief Get post op.
   *
   * Please see @c set_post_op() for further description.
   * @return Post_op vector.
   */
  std::vector<post_op_t> get_post_op() const;

  /** @brief Get post op count.
   *
   * Please see @c set_post_op() for further description.
   * @return Post_op count.
   */
  uint32_t      get_post_op_count() const;

  /** @brief Create an operator context
   *
   * Operator creation follows giving all mandatory parameters using
   * @c set_param() and post-ops using @c set_post_op().
   * Creation validates the parameters and creates the object. Please
   * see @c validate() for further description.
   *
   * Derived operators can either add more functionality or override it.
   * @se operator last status is set to status_t::success in case of successful
   * creation. The status can be checked using @c hash_object_t.check().
   * @return A reference to self.
   */
  virtual OP_CONTEXT_T &create();
  /**@}*/

  /** @name Execution Context */
  /**@{*/
  /** @brief Set core count.
   *
   * Set the number of cores for operator execution. This is unused.
   * @return A reference to self.
   */
  OP_CONTEXT_T &set_core_count(uint32_t count);

  /** @brief Get core count.
   *
   * Get the number of cores for operator execution. This is unused.
   * @return A reference to self.
   */
  uint32_t get_core_count() const;

  //set and get core binding
  OP_CONTEXT_T &set_core_binding(std::vector<uint32_t> cores);
  std::vector<uint32_t> get_core_binding() const;
  /**@}*/

  /** @brief Generate object hash.
   *
   * Hash generated by an object uniquely identifies the object, therefore hash is
   * generated by taking all the paramaters that uniquely identify a context.
   *
   * Only a valid object returns a hash. Invalid object hash is set to zero.
   * @return Object hash.
   */
  std::size_t   hash() override;

 protected:
  /** @brief Default constructor.
   *
   * ZenDNNL follows the convension of making constructors protected (or private),
   * where the class need to serve as a virtual base class, and no object of
   * the class should be created.
   */
  op_context_t();

  /** @brief Validate context parameters.
   *
   * Basic validation consists of checking if all parameters are
   * valid tensors. Any other validation can be implemented by overriding.
   * @return status_t::success if successful, else status_t::failure.
   */
  virtual status_t validate();

  /** @brief Preprocess.
   * Do any required preprocessing with parameters. Generally this preprocessing
   * requires reordering of parameter tensors, or post-op creations.
   * @return status_t::success if successful, else status_t::failure.
   */
  virtual status_t    preprocess();

  std::map<std::string, tensor_t> params; /**< operator parameters */
  std::vector<post_op_t> post_ops; /**< operator post ops vector */
  uint32_t core_count; /**< operator core count */
  std::vector<uint32_t> core_binding; /**< operator cores vector */
  uint32_t binary_add_count; /**< element-wise addition count */
  uint32_t binary_mul_count; /**< element-wise multiplication count */
  uint32_t activation_count; /**< activation count */
};

//implementation
template<typename OP_CONTEXT_T>
op_context_t<OP_CONTEXT_T>::op_context_t():
  params{}, post_ops{}, core_count{1}, core_binding{},
  binary_add_count{0}, binary_mul_count{0}, activation_count{0} {
}

template<typename OP_CONTEXT_T>
op_context_t<OP_CONTEXT_T>::~op_context_t() {
};

template<typename OP_CONTEXT_T>
OP_CONTEXT_T &op_context_t<OP_CONTEXT_T>::create() {
  LOG_DEBUG_INFO("Creating op_context_t");
  if (validate() != status_t::success) {
    status = status_t::failure;
    return static_cast<OP_CONTEXT_T &>(*this);
  }

  if (preprocess() != status_t::success) {
    status = status_t::failure;
    return static_cast<OP_CONTEXT_T &>(*this);
  }

  status = status_t::success;
  hash();
  return static_cast<OP_CONTEXT_T &>(*this);
};

template<typename OP_CONTEXT_T>
OP_CONTEXT_T &op_context_t<OP_CONTEXT_T>::set_param(std::string key,
    tensor_t &param_tensor) {
  LOG_DEBUG_INFO("Setting param op_context_t");
  params[key] = param_tensor;
  hash_key    = 0;

  return static_cast<OP_CONTEXT_T &>(*this);
}

template<typename OP_CONTEXT_T>
std::optional<tensor_t> op_context_t<OP_CONTEXT_T>::get_param(
  std::string key) const {
  LOG_DEBUG_INFO("Getting param op_context_t");
  for (const auto& [k, v] : params) {
    if (k == key) {
      return v;
    }
  }
  return std::nullopt;
}

template<typename OP_CONTEXT_T>
OP_CONTEXT_T &op_context_t<OP_CONTEXT_T>::set_post_op(post_op_t &post_op_) {
  LOG_DEBUG_INFO("Setting post-op op_context_t");
  if (post_op_.type == post_op_type_t::binary_add){
    post_op_.binary_add_params.tensor_name += std::to_string(binary_add_count);
    binary_add_count++;
  }
  else if (post_op_.type == post_op_type_t::binary_mul) {
    post_op_.binary_mul_params.tensor_name += std::to_string(binary_mul_count);
    binary_mul_count++;
  }
  else {
    activation_count++;
  }
  post_ops.push_back(post_op_);
  hash_key = 0;
  return static_cast<OP_CONTEXT_T &>(*this);
}

template<typename OP_CONTEXT_T>
post_op_t op_context_t<OP_CONTEXT_T>::get_post_op(uint32_t i_) const {
  LOG_DEBUG_INFO("Getting post-op op_context_t");
  try {
    return post_ops.at(i_);
  }
  catch (const std::out_of_range &ex) {
    EXCEPTION_WITH_LOC("operator post_op out of range encountered.");
  }
}

template<typename OP_CONTEXT_T>
std::vector<post_op_t> op_context_t<OP_CONTEXT_T>::get_post_op() const {
  LOG_DEBUG_INFO("Getting post-op vector op_context_t");
  return post_ops;
}

template<typename OP_CONTEXT_T>
uint32_t op_context_t<OP_CONTEXT_T>::get_post_op_count() const {
  LOG_DEBUG_INFO("Getting post-op count for op_context_t");
  return post_ops.size();
}

template<typename OP_CONTEXT_T>
OP_CONTEXT_T &op_context_t<OP_CONTEXT_T>::set_core_count(uint32_t count) {
  LOG_DEBUG_INFO("Setting core count for op_context_t");
  core_count = count;

  return static_cast<OP_CONTEXT_T &>(*this);
}

template<typename OP_CONTEXT_T>
uint32_t op_context_t<OP_CONTEXT_T>::get_core_count() const {
  LOG_DEBUG_INFO("Getting core count for op_context_t");
  return core_count;
}

template<typename OP_CONTEXT_T>
OP_CONTEXT_T &op_context_t<OP_CONTEXT_T>::set_core_binding(
  std::vector<uint32_t> cores) {
  LOG_DEBUG_INFO("Setting core binding for op_context_t");
  core_binding = cores;
  core_count   = core_binding.size();

  return static_cast<OP_CONTEXT_T &>(*this);
}

template<typename OP_CONTEXT_T>
std::vector<uint32_t> op_context_t<OP_CONTEXT_T>::get_core_binding() const {
  LOG_DEBUG_INFO("Getting core binding for op_context_t");
  return core_binding;
}

template<typename OP_CONTEXT_T>
status_t op_context_t<OP_CONTEXT_T>::validate() {
  LOG_DEBUG_INFO("Validating op_context_t");
  for (const auto& [k, v] : params) {
    std::optional<tensor_t> t = v;
    if (t && !(t->check())) {
      return status_t::failure;
    }
  }
  return status_t::success;
}

template<typename OP_CONTEXT_T>
status_t op_context_t<OP_CONTEXT_T>::preprocess() {
  LOG_DEBUG_INFO("Preprocessing for op_context_t");
  return status_t::success;
}

template<typename OP_CONTEXT_T>
std::size_t op_context_t<OP_CONTEXT_T>::hash() {
  LOG_DEBUG_INFO("Creating hash for op_context_t");
  if (status == status_t::success) {
    if (hash_key) {
      return hash_key;
    }

    for (const auto& [k, v] : params) {
      tensor_t temp = static_cast<tensor_t>(v);
      hash_key = hash_combine(hash_key, temp);
    }

    for (const auto &v : post_ops) {
      hash_key = hash_combine(hash_key, uint32_t(v.type));
    }
  }

  return hash_key;
}

} //namespace ops
} //namespace zendnnl
#endif
