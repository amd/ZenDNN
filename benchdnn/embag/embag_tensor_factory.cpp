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

#include "embag_tensor_factory.hpp"

namespace zendnnl {
namespace benchdnn {
namespace embag {

int create_table_tensor(tensor_factory_t &tensor_factory, EmbagConfig cfg,
                        tensor_t &table) {
  if (cfg.dt[0] == data_type_t::f32 ||
      cfg.dt[0] == data_type_t::bf16) {
    table = tensor_factory.uniform_dist_tensor({cfg.num_embeddings, cfg.embedding_dims},cfg.dt[0],
            2.0f, "table_tensor");
  }
  else {
    table = tensor_factory.quantized_embedding_tensor_random({cfg.num_embeddings, cfg.embedding_dims},
            cfg.dt[0], "table_tensor", cfg.fp16_scale_bias);
  }

  return OK;
}

int create_indices_tensor(tensor_factory_t &tensor_factory, EmbagConfig cfg,
                          tensor_t &indices) {

  indices = tensor_factory.random_indices_tensor({cfg.num_indices},
            cfg.num_embeddings);

  return OK;
}

int create_offsets_tensor(tensor_factory_t &tensor_factory, EmbagConfig cfg,
                          tensor_t &offsets) {

  uint64_t offsets_size = cfg.include_last_offset ? cfg.num_bags + 1 :
                          cfg.num_bags;
  offsets = tensor_factory.random_offsets_tensor({offsets_size}, cfg.num_indices,
            cfg.include_last_offset);

  return OK;
}

int create_weights_tensor(tensor_factory_t &tensor_factory, EmbagConfig cfg,
                          tensor_t &weights) {

  if (cfg.is_weights) {
    weights = tensor_factory.uniform_dist_tensor({cfg.num_indices},
              data_type_t::f32, 2.0f, "weights_tensor");
  }
  else {
    weights = tensor_t();
  }

  return OK;
}

int create_output_tensor(tensor_factory_t &tensor_factory, EmbagConfig cfg,
                         tensor_t &output) {

  output = tensor_factory.zero_tensor({cfg.num_bags, cfg.embedding_dims},
                                      cfg.dt[1], "output_tensor");

  return OK;
}

} // namespace embag
} // namespace benchdnn
} // namespace zendnnl