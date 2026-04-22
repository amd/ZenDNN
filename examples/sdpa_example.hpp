/********************************************************************************
# * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef _SDPA_ENCODER_EXAMPLE_HPP_
#define _SDPA_ENCODER_EXAMPLE_HPP_

#include "zendnnl.hpp"
#include "example_utils.hpp"

#define  OK          (0)
#define  NOT_OK      (1)

// SDPA dimensions
#define  BS  2
#define  NUM_HEADS   8
#define  SEQ_LEN     64
#define  HEAD_DIM    64

namespace zendnnl {
namespace examples {

/**
 * @brief SDPA example using the high-level encoder context/operator API.
 *
 * Creates Q/K/V tensors via tensor_factory, builds an sdpa_encoder_context_t
 * and sdpa_encoder_operator_t, then executes the SDPA operation.
 */
int sdpa_example();

/**
 * @brief SDPA example calling sdpa_direct with raw buffers and sdpa_params.
 *
 * Allocates contiguous BHSD float buffers for Q/K/V/output, populates
 * sdpa_params (dimensions, strides, dtypes, scale), and invokes
 * sdpa_direct directly.
 */
int sdpa_direct_example();
} //examples
} //zendnnl

#endif