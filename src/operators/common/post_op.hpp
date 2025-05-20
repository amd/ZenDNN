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
#ifndef _POST_OP_HPP_
#define _POST_OP_HPP_

#include <string>
#include <initializer_list>
#include "common/zendnnl_global.hpp"

namespace zendnnl {
namespace ops {

// a post_op refers to a localized operation on tensor data. this localized operattion involves
// a single tensor element or a small neighborhood around an element. it is understood that such
// a neighborhood is "hot" and already in L1 cache or registers. such a localized operation can be
// easily fused with an operator preseeding it (for example a matmul op).
// there could by multiple such post-ops following a major operator and all of them could be fused
// in pipeline fashion with the major operator.

/** @enum post_op_ptye_t
 *  @brief Supported post_op types.
 */
enum class post_op_type_t {
  elu, /*!< eltwise elu*/
  relu,/*!< eltwise relu */
  leaky_relu,/*!< eltwise leaky relu */
  gelu_tanh,/*!< eltwise gelu_tanh */
  gelu_erf,/*!< eltwise gelu_erf */
  sigmoid,/*!< sigmoid */
  swish,/*!< swish */
  tanh,/*!< tanh */
  softmax,/*!< softmax */
  pooling,/*!< pooling */
  square,/*!< eltwise square */
  abs,/*!< eltwise abs */
  sqrt,/*!< eltwise sqrt */
  exp,/*!< eltwise exp */
  log,/*!< eltwise log */
  clip,/*!< eltwise clip */
  binary_add,/*!< eltwise add with another tensor */
  binary_mul/*!< eltwise mul with another tensor */
};

/** @struct elu_params_t
 *  @brief elu parameters.
 */
struct elu_params_t {
  float alpha;
};

/** @struct leaky_relu_params_t
 *  @brief Leaky ReLU parameters.
 */
struct leaky_relu_params_t {
  float nslope;
};

/** @struct swish_params_t
 *  @brief swish parameters.
 */
struct swish_params_t {
  float scale;
};

/** @struct clip_params_t
 *  @brief Clip parameters.
 */
struct clip_params_t {
  float lower;
  float upper;
};

/** @struct binary_add_params_t
 *  @brief Eltwise add with another tensor parameters.
 */
struct binary_add_params_t {
  float scale;
  std::string tensor_name;
};

/** @struct binary_mul_params_t
 *  @brief Eltwise add with another tensor parameters.
 */
struct binary_mul_params_t {
  float scale;
  std::string tensor_name;
};

/** @struct post_op_t
 *  @brief Post Op Struct
 *
 * Post op refers to a localized computation (mostly elementwise
 * non-linear operators) that can be performed on the output of an
 * operator when the output data is still "hot" in cache. A post_op
 * can be fused with an operator. multiple post-ops can follow an
 * operator.
 */
struct post_op_t {
  /** @brief Constructor for a type without parameters */
  post_op_t(post_op_type_t type_);
  /** @brief Constructor for elu */
  post_op_t(elu_params_t params_);
  /** @brief Constructor for leaky ReLU */
  post_op_t(leaky_relu_params_t params_);
  /** @brief Constructor for swish */
  post_op_t(swish_params_t params_);
  /** @brief Constructor for clip */
  post_op_t(clip_params_t params_);
  /** @brief Constructor for binary add with a tensor */
  post_op_t(binary_add_params_t params_);
  /** @brief Constructor for binary mul with a tensor */
  post_op_t(binary_mul_params_t params_);

  post_op_type_t      type;
  elu_params_t        elu_params;
  swish_params_t      swish_params;
  leaky_relu_params_t leaky_relu_params;
  clip_params_t       clip_params;
  binary_add_params_t binary_add_params;
  binary_mul_params_t binary_mul_params;
};

} //namespace ops

namespace interface {
using post_op_type_t      = zendnnl::ops::post_op_type_t;
using elu_params_t        = zendnnl::ops::elu_params_t;
using leaky_relu_params_t = zendnnl::ops::leaky_relu_params_t;
using swish_params_t      = zendnnl::ops::swish_params_t;
using clip_params_t       = zendnnl::ops::clip_params_t;
using binary_add_params_t = zendnnl::ops::binary_add_params_t;
using binary_mul_params_t = zendnnl::ops::binary_mul_params_t;
using post_op_t           = zendnnl::ops::post_op_t;
} //interface
} //namespace zendnnl
#endif
