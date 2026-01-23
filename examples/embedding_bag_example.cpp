
/********************************************************************************
# * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

// Generate random indices
std::vector<int64_t> generate_random_indices(size_t count,
    int64_t min_val = 0, int64_t max_val = 99) {
  std::vector<int64_t> indices(count);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int64_t> dis(min_val, max_val);
  for (size_t i = 0; i < count; ++i) {
    indices[i] = dis(gen);
  }

  return indices;
}

// Generate offsets
std::vector<int64_t> generate_offsets(size_t batch_size) {
  std::vector<int64_t> offsets(batch_size, 0);
  for (size_t i = 1; i < batch_size; ++i) {
    offsets[i] = offsets[i-1] + 2;
  }

  return offsets;
}

std::vector<int64_t> indices = generate_random_indices(INDICES_SIZE);
std::vector<int64_t> offsets = generate_offsets(EMB_BATCH_SIZE);

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

    if (embedding_bag_operator.is_bad_object()) {
      testlog_error(" operator ", embedding_bag_operator.get_name(),
                    " creation failed.");
      return NOT_OK;
    }

    auto indices_tensor = tensor_factory.non_uniform_tensor({indices.size()},
                          data_type_t::s64,
                          indices, "indices");

    auto offsets_tensor = tensor_factory.non_uniform_tensor({EMB_BATCH_SIZE},
                          data_type_t::s64,
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

    if (embedding_bag_operator.is_bad_object()) {
      testlog_error(" operator ", embedding_bag_operator.get_name(),
                    " creation failed.");
      return NOT_OK;
    }

    auto indices_tensor = tensor_factory.non_uniform_tensor({indices.size()},
                          data_type_t::s64,
                          indices, "indices");

    auto offsets_tensor = tensor_factory.non_uniform_tensor({EMB_BATCH_SIZE},
                          data_type_t::s64,
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

// This example demonstrates creating an embedding lookup operator for f32 data type
// using the same underlying embedding bag operator infrastructure.
// Unlike embedding bag operations, this performs direct index-to-embedding lookups
// without offsets or reduction operations (no sum/mean/max aggregation).
int embedding_f32_kernel_example() {

  try {
    status_t status;
    tensor_factory_t tensor_factory;

    auto table = tensor_factory.uniform_tensor({EMB_ROW, EMB_DIM},
                 data_type_t::f32,
                 1.0, "table");

    //define embedding context
    auto embedding_context = embag_context_t()
                             .set_param("table", table)
                             .create();

    if (! embedding_context.check()) {
      testlog_error("embedding context creation failed");
      return NOT_OK;
    }

    //define embedding operator
    auto embedding_operator = embag_operator_t()
                              .set_name("embedding_f32")
                              .set_context(embedding_context)
                              .create();

    if (embedding_operator.is_bad_object()) {
      testlog_error(" operator ", embedding_operator.get_name(),
                    " creation failed.");
      return NOT_OK;
    }

    auto indices_tensor = tensor_factory.non_uniform_tensor({indices.size()},
                          data_type_t::s64,
                          indices, "indices");

    auto output_tensor = tensor_factory.zero_tensor({indices.size(), EMB_DIM},
                         data_type_t::f32, "output");

    status = embedding_operator
             .set_input("indices", indices_tensor)
             .set_output("output", output_tensor)
             .execute();

    if (status == status_t::success) {
      testlog_info("<",embedding_operator.get_name(),">",
                   " operator execution successful.");
    }
    else {
      testlog_error("<",embedding_operator.get_name(),">",
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

int embedding_bag_u4_kernel_example() {
  try {
    status_t status;
    tensor_factory_t tensor_factory;

    auto table = tensor_factory.quantized_embedding_tensor_random({EMB_ROW, EMB_DIM},
                 data_type_t::u4, "table", true);

    //define embedding bag context
    auto embedding_bag_context = embag_context_t()
                                 .set_param("table", table)
                                 .set_algo(embag_algo_t::sum)
                                 .set_fp16_scale_bias(true)
                                 .create();

    if (! embedding_bag_context.check()) {
      testlog_error("embedding bag context creation failed");
      return NOT_OK;
    }

    //define embedding bag operator
    auto embedding_bag_operator = embag_operator_t()
                                  .set_name("embedding_bag_int4_operator")
                                  .set_context(embedding_bag_context)
                                  .create();

    if (embedding_bag_operator.is_bad_object()) {
      testlog_error(" operator ", embedding_bag_operator.get_name(),
                    " creation failed.");
      return NOT_OK;
    }

    auto indices_tensor = tensor_factory.non_uniform_tensor({indices.size()},
                          data_type_t::s64, indices, "indices");

    auto offsets_tensor = tensor_factory.non_uniform_tensor({EMB_BATCH_SIZE},
                          data_type_t::s64, offsets, "offsets");

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
    //free this table pointer after use
    free(table.get_raw_handle_unsafe());
    table.reset();
  }
  catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

int embedding_bag_u4_ref_kernel_example() {
  try {
    status_t status;
    tensor_factory_t tensor_factory;

    auto table = tensor_factory.quantized_embedding_tensor_random({EMB_ROW, EMB_DIM},
                 data_type_t::u4, "table", true);

    //define embedding bag context
    auto embedding_bag_context = embag_context_t()
                                 .set_param("table", table)
                                 .set_algo(embag_algo_t::sum)
                                 .set_fp16_scale_bias(true)
                                 .create();

    if (! embedding_bag_context.check()) {
      testlog_error("embedding bag context creation failed");
      return NOT_OK;
    }

    //define embedding bag operator
    auto embedding_bag_operator = embag_operator_t()
                                  .set_name("embedding_bag_int4_ref")
                                  .set_context(embedding_bag_context)
                                  .create();

    if (embedding_bag_operator.is_bad_object()) {
      testlog_error(" operator ", embedding_bag_operator.get_name(),
                    " creation failed.");
      return NOT_OK;
    }

    auto indices_tensor = tensor_factory.non_uniform_tensor({indices.size()},
                          data_type_t::s64, indices, "indices");

    auto offsets_tensor = tensor_factory.non_uniform_tensor({EMB_BATCH_SIZE},
                          data_type_t::s64, offsets, "offsets");

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

  //free this table pointer after use
  free(table.get_raw_handle_unsafe());
  table.reset();
  }
  catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

// This example demonstrates the group_embedding_bag_direct API which
// performs multiple embedding bag operations in a single call.
// Each operation has its own embedding table, indices, offsets, and output.
int group_embedding_bag_direct_example() {

  try {
    // Number of embedding tables to process
    constexpr size_t NUM_TABLES = 3;

    // Define dimensions for each embedding table
    // Each table can have different dimensions
    const std::vector<size_t> num_embeddings = {100, 150, 80};
    const std::vector<size_t> embedding_dims = {16, 32, 24};
    const std::vector<size_t> num_indices_per_table = {10, 15, 8};
    const std::vector<size_t> num_bags_per_table = {5, 5, 4};

    // Allocate embedding tables
    std::vector<std::vector<float>> tables_data(NUM_TABLES);
    std::vector<const void*> tables(NUM_TABLES);
    for (size_t i = 0; i < NUM_TABLES; ++i) {
      tables_data[i].resize(num_embeddings[i] * embedding_dims[i], 1.0f);
      // Initialize with some pattern for verification
      for (size_t j = 0; j < tables_data[i].size(); ++j) {
        tables_data[i][j] = static_cast<float>(i + 1) * 0.1f;
      }
      tables[i] = tables_data[i].data();
    }

    // Generate indices for each table
    std::vector<std::vector<int64_t>> indices_data(NUM_TABLES);
    std::vector<const void*> indices_ptrs(NUM_TABLES);
    std::random_device rd;
    std::mt19937 gen(rd());
    for (size_t i = 0; i < NUM_TABLES; ++i) {
      indices_data[i].resize(num_indices_per_table[i]);
      std::uniform_int_distribution<int64_t> dis(0, num_embeddings[i] - 1);
      for (size_t j = 0; j < num_indices_per_table[i]; ++j) {
        indices_data[i][j] = dis(gen);
      }
      indices_ptrs[i] = indices_data[i].data();
    }

    // Generate offsets for each table
    std::vector<std::vector<int64_t>> offsets_data(NUM_TABLES);
    std::vector<const void*> offsets_ptrs(NUM_TABLES);
    for (size_t i = 0; i < NUM_TABLES; ++i) {
      offsets_data[i].resize(num_bags_per_table[i]);
      int64_t offset = 0;
      size_t indices_per_bag = num_indices_per_table[i] / num_bags_per_table[i];
      for (size_t j = 0; j < num_bags_per_table[i]; ++j) {
        offsets_data[i][j] = offset;
        offset += indices_per_bag;
      }
      offsets_ptrs[i] = offsets_data[i].data();
    }

    // Allocate output buffers
    std::vector<std::vector<float>> outputs_data(NUM_TABLES);
    std::vector<void*> outputs(NUM_TABLES);
    for (size_t i = 0; i < NUM_TABLES; ++i) {
      outputs_data[i].resize(num_bags_per_table[i] * embedding_dims[i], 0.0f);
      outputs[i] = outputs_data[i].data();
    }

    // No weights for this example
    std::vector<const float*> weights(NUM_TABLES, nullptr);

    // Setup parameters for each embedding bag operation
    std::vector<embag_params_t> params(NUM_TABLES);
    for (size_t i = 0; i < NUM_TABLES; ++i) {
      params[i].dtypes.table = data_type_t::f32;
      params[i].dtypes.output = data_type_t::f32;
      params[i].dtypes.indices = data_type_t::s64;
      params[i].dtypes.offsets = data_type_t::s64;
      params[i].algo = embag_algo_t::sum;
      params[i].num_embeddings = num_embeddings[i];
      params[i].embedding_dim = embedding_dims[i];
      params[i].num_indices = num_indices_per_table[i];
      params[i].num_bags = num_bags_per_table[i];
      params[i].is_weights = false;
      params[i].include_last_offset = false;
      params[i].padding_idx = -1;
    }

    // Execute group embedding bag
    status_t status = group_embedding_bag_direct(
        tables, indices_ptrs, offsets_ptrs, weights, outputs, params);

    if (status == status_t::success) {
      testlog_info("<group_embedding_bag_direct>",
                   " executed successfully for ", NUM_TABLES, " tables.");
    }
    else {
      testlog_error("<group_embedding_bag_direct>",
                    " execution failed.");
      return NOT_OK;
    }

  }
  catch (const std::exception &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

} // namespace examples
} // namespace zendnnl
