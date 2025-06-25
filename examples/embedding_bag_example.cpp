
/********************************************************************************
# * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <random>
#include <set>
#include "embedding_bag_example.hpp"

namespace zendnnl {
namespace examples {

using namespace zendnnl::interface;

// Generate random indices
std::vector<uint32_t> generate_random_indices(size_t count,
    uint32_t min_val = 0, uint32_t max_val = 99) {
  std::vector<uint32_t> indices(count);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint32_t> dis(min_val, max_val);
  for (size_t i = 0; i < count; ++i) {
    indices[i] = dis(gen);
  }

  return indices;
}

// Generate offsets
std::vector<uint32_t> generate_offsets(size_t batch_size) {
  std::vector<uint32_t> offsets(batch_size, 0);
  for (size_t i = 1; i < batch_size; ++i) {
    offsets[i] = offsets[i-1] + 2;
  }

  return offsets;
}

std::vector<uint32_t> indices = generate_random_indices(INDICES_SIZE);
std::vector<uint32_t> offsets = generate_offsets(EMB_BATCH_SIZE);

int embedding_bag_f32_kernel_example() {

  try {
    status_t status;
    tensor_factory_t tensor_factory;

    auto table = tensor_factory.uniform_tensor({EMB_ROW, EMB_DIM},
                 data_type_t::f32,
                 1.0, "table");

    //define embedding bag context
    auto embedding_bag_context = embag_context_t()
                                 .set_param("table", table)
                                 .set_algo(embag_algo_t::sum)
                                 .create();

    if (! embedding_bag_context.check()) {
      testlog_error("embedding bag context creation failed");
      return NOT_OK;
    }

    //define embedding bag operator
    auto embedding_bag_operator = embag_operator_t()
                                  .set_name("embedding_bag_f32")
                                  .set_context(embedding_bag_context)
                                  .create();

    if (! embedding_bag_operator.check()) {
      testlog_error(" operator ", embedding_bag_operator.get_name(),
                    " creation failed.");
      return NOT_OK;
    }

    auto indices_tensor = tensor_factory.non_uniform_tensor({indices.size()},
                          data_type_t::s32,
                          indices, "indices");

    auto offsets_tensor = tensor_factory.non_uniform_tensor({EMB_BATCH_SIZE},
                          data_type_t::s32,
                          offsets, "offsets");

    auto output_tensor = tensor_factory.zero_tensor({EMB_BATCH_SIZE, EMB_DIM},
                         data_type_t::f32, "output");

    status = embedding_bag_operator
             .set_input("indices", indices_tensor)
             .set_input("offsets", offsets_tensor)
             .set_output("output", output_tensor)
             .execute();

    if (status == status_t::success) {
      testlog_info("<",embedding_bag_operator.get_name(),">",
                   " operator execution successful.");
    }
    else {
      testlog_error("<",embedding_bag_operator.get_name(),">",
                    " operator execution failed.");
      return NOT_OK;
    }

  }
  catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

int embedding_bag_f32_forced_ref_kernel_example() {

  try {
    status_t status;
    tensor_factory_t tensor_factory;

    auto table = tensor_factory.uniform_tensor({EMB_ROW, EMB_DIM},
                 data_type_t::f32,
                 1.0, "table");

    //define embedding bag context
    auto embedding_bag_context = embag_context_t()
                                 .set_param("table", table)
                                 .set_algo(embag_algo_t::sum)
                                 .create();

    if (! embedding_bag_context.check()) {
      testlog_error("embedding bag context creation failed");
      return NOT_OK;
    }

    //define embedding bag operator
    auto embedding_bag_operator = embag_operator_t()
                                  .set_name("embedding_bag_forced_ref_operator")
                                  .set_context(embedding_bag_context)
                                  .create();

    if (! embedding_bag_operator.check()) {
      testlog_error(" operator ", embedding_bag_operator.get_name(),
                    " creation failed.");
      return NOT_OK;
    }

    auto indices_tensor = tensor_factory.non_uniform_tensor({indices.size()},
                          data_type_t::s32,
                          indices, "indices");

    auto offsets_tensor = tensor_factory.non_uniform_tensor({EMB_BATCH_SIZE},
                          data_type_t::s32,
                          offsets, "offsets");

    auto output_tensor = tensor_factory.zero_tensor({EMB_BATCH_SIZE, EMB_DIM},
                         data_type_t::f32, "output");

    status = embedding_bag_operator
             .set_input("indices", indices_tensor)
             .set_input("offsets", offsets_tensor)
             .set_output("output", output_tensor)
             .set_forced_kernel("reference")
             .execute();

    if (status == status_t::success) {
      testlog_info("<",embedding_bag_operator.get_name(),">",
                   " operator execution successful.");
    }
    else {
      testlog_error("<",embedding_bag_operator.get_name(),">",
                    " operator execution failed.");
      return NOT_OK;
    }

  }
  catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

} // namespace examples
} // namespace zendnnl
