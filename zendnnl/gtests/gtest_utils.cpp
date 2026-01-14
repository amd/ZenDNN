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

#include "gtest_utils.hpp"

MatmulType::MatmulType(uint32_t test_index, uint32_t total_tests) {
  matmul_m   = MATMUL_SIZE_START + rand() % MATMUL_SIZE_END;
  matmul_k   = MATMUL_SIZE_START + rand() % MATMUL_SIZE_END;
  matmul_n   = MATMUL_SIZE_START + rand() % MATMUL_SIZE_END;
  transA     = rand() % 2;
  transB     = rand() % 2;
  // Post-op selection based on command-line input or random selection
  if (!cmd_post_op.empty()) {
    po_type = strToPostOps(cmd_post_op);
  }
  else {
    po_type = post_op_arr[rand() % (po_size + 1)];
  }

  // Use std::mt19937 for random float generation
  std::mt19937 gen(rand());
  std::uniform_real_distribution<float> dist(0.0, 10.0);
  alpha    = dist(gen);
  beta     = dist(gen);

  if (!cmd_lowoha.empty()) {
    use_LOWOHA = (cmd_lowoha == "true") || (cmd_lowoha == "1");
  }
  matmul_config_t &matmul_config = matmul_config_t::instance();
  int32_t algo_ = matmul_config.get_algo();
  algo = static_cast<matmul_algo_t>(algo_);

  if (algo == matmul_algo_t::none) {
    // Algorithm configuration based on command-line input or random selection
    if (!cmd_backend.empty()) {
      // Handle oneDNN dependency check for command-line backends
#if ZENDNNL_DEPENDS_ONEDNN
      algo = strToAlgo(cmd_backend);
#else
      // Fallback to AOCL DLP if oneDNN backends requested but not available
      algo = (cmd_backend == "onednn" || cmd_backend == "onednn_blocked")
             ? matmul_algo_t::aocl_dlp
             : strToAlgo(cmd_backend);
#endif
      // Configure algorithm-specific parameters and LOWOHA settings
      if (algo == matmul_algo_t::libxsmm || algo == matmul_algo_t::libxsmm_blocked) {
        alpha = 1.0f;
        beta  = rand() % 2;
        use_LOWOHA = true;
        //ToDo: Need to support silu, gelu_tanh.
        if (po_type == post_op_type_t::swish || po_type == post_op_type_t::gelu_tanh) {
          po_type = post_op_type_t::none;
        }
      }
      else {
        use_LOWOHA = rand() % 2;
      }
    }
    else {
      // Random algorithm selection
      int algo_range_max = 6; // 6 algorithms in total
      std::uniform_int_distribution<int> algo_dist(1, algo_range_max);
      algo = static_cast<matmul_algo_t>(algo_dist(gen));
      if (!ZENDNNL_DEPENDS_ONEDNN && (algo == matmul_algo_t::onednn ||
                                      algo == matmul_algo_t::onednn_blocked)) {
        algo = matmul_algo_t::aocl_dlp;
      }

      // If no lowoha argument is provided, automatically partition tests into three
      if (cmd_lowoha.empty()) {
        if (algo == matmul_algo_t::libxsmm || algo == matmul_algo_t::libxsmm_blocked) {
          algo = matmul_algo_t::aocl_dlp;
        }
        // Control LOWOHA and LIBXSMM based on test index
        // First third: both off, second third: LOWOHA on LIBXSMM off, last third: both on
        uint32_t third = total_tests / TEST_PARTITIONS;

        use_LOWOHA = (test_index >= third);
        if (test_index >= 2 * third) {
          alpha = 1.0f;
          beta = rand() % 2;
          algo = (rand() % 2) ? matmul_algo_t::libxsmm : matmul_algo_t::libxsmm_blocked;
          if (!ZENDNNL_DEPENDS_LIBXSMM) {
            algo = matmul_algo_t::aocl_dlp;
          }
          // ToDo: Add support for other postops. Currently disabling gelu_tanh, swish.
          if (po_type == post_op_type_t::swish || po_type == post_op_type_t::gelu_tanh) {
            po_type = post_op_type_t::none;
          }
        }
      }
      // If lowoha argument is explicitly set to true
      else if (use_LOWOHA) {
        if (algo == matmul_algo_t::libxsmm || algo == matmul_algo_t::libxsmm_blocked) {
          alpha = 1.0f;
          beta = rand() % 2;
          algo = (rand() % 2) ? matmul_algo_t::libxsmm : matmul_algo_t::libxsmm_blocked;
          if (!ZENDNNL_DEPENDS_LIBXSMM) {
            algo = matmul_algo_t::aocl_dlp;
          }
          // ToDo: Add support for other postops. Currently disabling gelu_tanh, swish.
          if (po_type == post_op_type_t::swish || po_type == post_op_type_t::gelu_tanh) {
            po_type = post_op_type_t::none;
          }
        }
      }
      // If lowoha argument is explicitly set to false
      else {
        if (algo == matmul_algo_t::libxsmm || algo == matmul_algo_t::libxsmm_blocked) {
          algo = matmul_algo_t::aocl_dlp;
        }
      }
    }
  }
  else {
    if (algo == matmul_algo_t::libxsmm ||
        algo == matmul_algo_t::libxsmm_blocked) {
      alpha    = 1.0f;
      beta     = rand() % 2;
      use_LOWOHA = true;
      if (po_type == post_op_type_t::swish || po_type == post_op_type_t::gelu_tanh) {
        po_type = post_op_type_t::none;
      }
    }
    else {
      use_LOWOHA = rand() % 2;
    }
  }
  source_dtype = rand() % 2 == 0 ? data_type_t::s8 : data_type_t::u8;
  output_dtype = dtype_arr[rand() % dtype_size];
  weight_granularity = rand() % 2 == 0 ? quant_granularity_t::tensor :
                       quant_granularity_t::channel;
}

// EmbagType constructor
EmbagType::EmbagType() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_int_distribution<int> algo_dist(1,
      3);  // sum=1, mean=2, max=3

  num_embeddings = 128 + std::rand() % 2048;
  embedding_dim = 16 + std::rand() % 512;
  num_indices = 128 + std::rand() % 2048;
  num_bags = 16 + std::rand() % 512;
  algo = static_cast<embag_algo_t>(algo_dist(gen));
  padding_index = -1;
  include_last_offset = std::rand() % 2;
  is_weights = std::rand() % 2;
  indices_dtype = rand() % 2 == 0 ? data_type_t::s32 : data_type_t::s64;
  offsets_dtype = indices_dtype;
  fp16_scale_bias = std::rand() % 2;
  strided = std::rand() % 2;
  use_LOWOHA = std::rand() % 2;
}

// EmbeddingType constructor
EmbeddingType::EmbeddingType() {
  num_embeddings = 128 + std::rand() % 2048;
  embedding_dim = 16 + std::rand() % 512;
  num_indices = 128 + std::rand() % 2048;
  padding_index = -1;
  is_weights = std::rand() % 2;
  indices_dtype = rand() % 2 == 0 ? data_type_t::s32 : data_type_t::s64;
  fp16_scale_bias = std::rand() % 2;
  strided = std::rand() % 2;
  use_LOWOHA = std::rand() % 2;
}

BatchMatmulType::BatchMatmulType(uint32_t test_index, uint32_t total_tests) {
  batch_size = BATCH_START + rand() % BATCH_END;
  mat = MatmulType(test_index, total_tests);
}

ReorderType::ReorderType(uint32_t test_index, uint32_t total_tests) {
  inplace_reorder = rand() % 2;
  mat = MatmulType(test_index, total_tests);
}

bool is_binary_postop(post_op_type_t post_op) {
  return post_op == post_op_type_t::binary_add ||
         post_op == post_op_type_t::binary_mul;
}

tensor_t tensor_factory_t::zero_tensor(const std::vector<index_type> size_,
                                       data_type dtype_, tensor_t scale, tensor_t zp,
                                       bool strided) {

  auto ztensor = tensor_t()
                 .set_name("zero tensor")
                 .set_size(size_)
                 .set_data_type(dtype_);
  if (strided) {
    uint64_t x = size_[1] + rand() % 50;
    ztensor.set_stride({x, 1});
    ztensor.set_aligned_size({size_[0], x});
    ztensor.set_storage();
  }
  else {
    ztensor.set_storage();
  }

  if (scale.get_nelem() != 0) {
    ztensor.set_quant_scale(scale);
  }
  if (zp.get_nelem() != 0) {
    ztensor.set_quant_zero_point(zp);
  }
  ztensor.create();

  if (! ztensor.check()) {
    log_warning("tensor creation of ", ztensor.get_name(), " failed.");
  }
  else {
    auto  buf_size = ztensor.get_buffer_sz_bytes();
    void *buf_ptr  = ztensor.get_raw_handle_unsafe();
    std::memset(buf_ptr, 0, buf_size);
  }
  return ztensor;
}

tensor_t tensor_factory_t::uniform_dist_tensor(const std::vector<index_type>
    size_, data_type dtype_, float val,
    bool trans, tensor_t scale, tensor_t zp) {
  auto udtensor = tensor_t()
                  .set_name("uniform distributed tensor")
                  .set_size(size_)
                  .set_data_type(dtype_);

  auto tensor_dim = udtensor.get_dim();
  if (trans && tensor_dim >=2) {
    std::string tag;
    for (size_t i=0; i<tensor_dim; ++i) {
      tag += 'a' + i;
    }
    std::swap(tag[size_.size() - 2], tag[size_.size() - 1]);
    udtensor.set_order(tag);
  }

  udtensor.set_storage();

  if (scale.get_nelem() != 0) {
    udtensor.set_quant_scale(scale);
  }
  if (zp.get_nelem() != 0) {
    udtensor.set_quant_zero_point(zp);
  }
  udtensor.create();

  if (! udtensor.check()) {
    log_warning("tensor creation of ", udtensor.get_name(), " failed.");
  }
  else {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist2(-1.0 * val, 1.0 * val);

    auto  buf_nelem  = udtensor.get_nelem();
    void *buf_vptr   = udtensor.get_raw_handle_unsafe();

    if (dtype_ == data_type::f32) {
      float *buf_ptr = static_cast<float *>(buf_vptr);
      std::generate(buf_ptr, buf_ptr+buf_nelem, [&] {return dist2(gen);});
    }
    else if (dtype_ == data_type::bf16) {
      bfloat16_t *buf_ptr = static_cast<bfloat16_t *>(buf_vptr);
      std::generate(buf_ptr, buf_ptr+buf_nelem, [&] {return bfloat16_t(dist2(gen));});
    }
    else if (dtype_ == data_type::s8) {
      std::uniform_int_distribution<int> dist_s8(-1*val, val);
      int8_t *buf_ptr = static_cast<int8_t *>(buf_vptr);
      std::generate(buf_ptr, buf_ptr + buf_nelem, [&] { return static_cast<int8_t>(dist_s8(gen)); });
    }
    else if (dtype_ == data_type::u8) {
      std::uniform_int_distribution<int> dist_u8(0, val);
      uint8_t *buf_ptr = static_cast<uint8_t *>(buf_vptr);
      std::generate(buf_ptr, buf_ptr + buf_nelem, [&] { return static_cast<uint8_t>(dist_u8(gen)); });
    }
    else if (dtype_ == data_type::s32) {
      std::uniform_int_distribution<int> dist_s32(-1 * val, val);
      int32_t *buf_ptr = static_cast<int32_t *>(buf_vptr);
      std::generate(buf_ptr, buf_ptr + buf_nelem, [&] { return static_cast<int32_t>(dist_s32(gen)); });
    }
    else if (dtype_ == data_type::s4) {
      // S4 is packed: 2 x 4-bit values per byte, range [-8, 7]
      // buf_nelem is the number of S4 elements, stored in buf_nelem/2 bytes
      std::uniform_int_distribution<int> dist_s4(-8, 7);  // S4 range: -8 to 7
      int8_t *buf_ptr = static_cast<int8_t *>(buf_vptr);
      size_t num_bytes = (buf_nelem + 1) / 2;  // Round up for odd number of elements
      for (size_t i = 0; i < num_bytes; ++i) {
        int8_t low_nibble = static_cast<int8_t>(dist_s4(gen)) & 0x0F;
        int8_t high_nibble = static_cast<int8_t>(dist_s4(gen)) & 0x0F;
        buf_ptr[i] = low_nibble | (high_nibble << 4);
      }
    }
    else {
      log_warning("tensor ", udtensor.get_name(), " unsupported data type.");
    }
  }
  return udtensor;
}

tensor_t tensor_factory_t::uniform_tensor(const std::vector<index_type> size_,
    data_type dtype_, float val_,
    std::string tensor_name_, bool trans,
    tensor_t scale, tensor_t zp) {

  auto utensor = tensor_t()
                 .set_name(tensor_name_)
                 .set_size(size_)
                 .set_data_type(dtype_);

  auto tensor_dim = utensor.get_dim();
  if (trans && tensor_dim >= 2) {
    std::string tag;
    for (size_t i = 0; i < tensor_dim; ++i) {
      tag += 'a' + i;
    }
    std::swap(tag[size_.size() - 2], tag[size_.size() - 1]);
    utensor.set_order(tag);
  }

  utensor.set_storage();

  if (scale.get_nelem() != 0) {
    utensor.set_quant_scale(scale);
  }
  if (zp.get_nelem() != 0) {
    utensor.set_quant_zero_point(zp);
  }
  utensor.create();

  if (!utensor.check()) {
    log_warning("tensor creation of ", utensor.get_name(), " failed.");
  }
  else {
    auto  buf_nelem = utensor.get_nelem();
    void *buf_vptr  = utensor.get_raw_handle_unsafe();

    if (dtype_ == data_type::f32) {
      float *buf_ptr = static_cast<float *>(buf_vptr);
      for (index_type i = 0; i < buf_nelem; ++i) {
        buf_ptr[i] = val_;
      }
    }
    else if (dtype_ == data_type::bf16) {
      bfloat16_t *buf_ptr = static_cast<bfloat16_t *>(buf_vptr);
      for (index_type i = 0; i < buf_nelem; ++i) {
        buf_ptr[i] = bfloat16_t(val_);
      }
    }
    else if (dtype_ == data_type::s8) {
      int8_t *buf_ptr = static_cast<int8_t *>(buf_vptr);
      for (index_type i = 0; i < buf_nelem; ++i) {
        buf_ptr[i] = static_cast<int8_t>(val_);
      }
    }
    else if (dtype_ == data_type::u8) {
      uint8_t *buf_ptr = static_cast<uint8_t *>(buf_vptr);
      for (index_type i = 0; i < buf_nelem; ++i) {
        buf_ptr[i] = static_cast<uint8_t>(val_);
      }
    }
    else if (dtype_ == data_type::s32) {
      int32_t *buf_ptr = static_cast<int32_t *>(buf_vptr);
      for (index_type i = 0; i < buf_nelem; ++i) {
        buf_ptr[i] = static_cast<int32_t>(val_);
      }
    }
    else if (dtype_ == data_type::s4) {
      // S4 is packed: 2 x 4-bit values per byte, range [-8, 7]
      int8_t s4_val = static_cast<int8_t>(val_) & 0x0F;
      int8_t packed_val = s4_val | (s4_val << 4);  // Same value in both nibbles
      int8_t *buf_ptr = static_cast<int8_t *>(buf_vptr);
      size_t num_bytes = (buf_nelem + 1) / 2;
      for (size_t i = 0; i < num_bytes; ++i) {
        buf_ptr[i] = packed_val;
      }
    }
    else {
      log_warning("tensor ", utensor.get_name(), " unsupported data type.");
    }
  }
  return utensor;
}

tensor_t tensor_factory_t::uniform_dist_strided_tensor(const
    std::vector<index_type> size_, const std::vector<index_type> aligned_size_,
    data_type dtype_, float range_, bool trans, tensor_t scale, tensor_t zp) {
  auto udstensor = tensor_t()
                   .set_name("uniform distributed strided tensor")
                   .set_size(size_)
                   .set_data_type(dtype_)
                   .set_aligned_size(aligned_size_)
                   .set_storage();
  if (scale.get_nelem() != 0) {
    udstensor.set_quant_scale(scale);
  }
  if (zp.get_nelem() != 0) {
    udstensor.set_quant_zero_point(zp);
  }
  udstensor.create();

  if (! udstensor.check()) {
    log_warning("tensor creation of ", udstensor.get_name(), " failed.");
  }
  else {
    std::mt19937 gen(100);
    std::uniform_real_distribution<float> dist(-1.0 * range_, 1.0 * range_);

    auto  buf_nelem   = aligned_size_[0];
    for (size_t i = 1; i < aligned_size_.size(); i++) {
      buf_nelem *= aligned_size_[i];
    }
    void *buf_vptr = udstensor.get_raw_handle_unsafe();

    if (dtype_ == data_type::f32) {
      float *buf_ptr = static_cast<float *>(buf_vptr);
      std::generate(buf_ptr, buf_ptr + buf_nelem, [&] {return dist(gen);});
    }
    else if (dtype_ == data_type::bf16) {
      bfloat16_t *buf_ptr = static_cast<bfloat16_t *>(buf_vptr);
      std::generate(buf_ptr, buf_ptr + buf_nelem, [&] {return bfloat16_t(dist(gen));});
    }
    else if (dtype_ == data_type::s8) {
      int8_t *buf_ptr = static_cast<int8_t *>(buf_vptr);
      std::generate(buf_ptr, buf_ptr + buf_nelem, [&] { return static_cast<int8_t>(dist(gen)); });
    }
    else if (dtype_ == data_type::u8) {
      uint8_t *buf_ptr = static_cast<uint8_t *>(buf_vptr);
      std::generate(buf_ptr, buf_ptr + buf_nelem, [&] { return static_cast<uint8_t>(dist(gen)); });
    }
    else {
      log_warning("tensor ", udstensor.get_name(), " unsupported data type.");
    }
  }
  return udstensor;
}

tensor_t tensor_factory_t::blocked_tensor(const std::vector<index_type> size_,
    data_type dtype_,
    float val) {

  auto btensor = tensor_t()
                 .set_name("blocked tensor")
                 .set_size(size_)
                 .set_data_type(dtype_)
                 .set_layout(tensor_layout_t::blocked)
                 .set_storage()
                 .create();

  if (! btensor.check()) {
    log_warning("tensor creation of ", btensor.get_name(), " failed.");
  }
  else {
    std::mt19937 gen(100);
    std::uniform_real_distribution<float> dist(-1.0 * val, 1.0 * val);

    auto  buf_nelem  = btensor.get_nelem();
    void *buf_vptr   = btensor.get_raw_handle_unsafe();

    if (dtype_ == data_type::f32) {
      float *buf_ptr = static_cast<float *>(buf_vptr);
      std::generate(buf_ptr, buf_ptr+buf_nelem, [&] {return dist(gen);});
    }
    else if (dtype_ == data_type::bf16) {
      bfloat16_t *buf_ptr = static_cast<bfloat16_t *>(buf_vptr);
      std::generate(buf_ptr, buf_ptr+buf_nelem, [&] {return bfloat16_t(dist(gen));});
    }
    else if (dtype_ == data_type::s8) {
      int8_t *buf_ptr = static_cast<int8_t *>(buf_vptr);
      std::generate(buf_ptr, buf_ptr+buf_nelem, [&] {return int8_t(dist(gen));});
    }
    else {
      log_warning("tensor ", btensor.get_name(), " unsupported data type.");
    }
  }

  return btensor;
}

tensor_t tensor_factory_t::copy_tensor(const std::vector<index_type> size_,
                                       data_type dtype_, StorageParam param,
                                       bool trans, bool is_blocked) {
  auto ctensor = tensor_t()
                 .set_size(size_)
                 .set_data_type(dtype_);

  auto tensor_dim = ctensor.get_dim();
  if (trans && tensor_dim >=2) {
    std::string tag;
    for (size_t i=0; i<tensor_dim; ++i) {
      tag += 'a' + i;
    }
    std::swap(tag[size_.size() - 2], tag[size_.size() - 1]);
    ctensor.set_order(tag);
  }

  if (is_blocked) {
    ctensor.set_name("blocked tensor");
    ctensor.set_layout(tensor_layout_t::blocked);
  }
  else {
    ctensor.set_name("uniform distributed tensor");
  }

  if (std::holds_alternative<std::pair<size_t, void *>>(param)) {
    auto [reorder_size, reorder_buff] = std::get<std::pair<size_t, void *>>(param);
    ctensor.set_storage(reorder_buff, reorder_size);
  }
  else if (std::holds_alternative<tensor_t>(param)) {
    tensor_t input_tensor = std::get<tensor_t>(param);
    ctensor.set_storage(input_tensor);
  }
  ctensor.create();

  if (! ctensor.check()) {
    log_warning("tensor creation of ", ctensor.get_name(), " failed.");
  }
  return ctensor;
}

// Extended tensor factory implementations
tensor_t tensor_factory_t::random_indices_tensor(const std::vector<index_type>
    size_, uint64_t num_embeddings_,
    data_type_t indices_dtype_) {
  auto indices_tensor = tensor_t()
                        .set_name("indices_tensor")
                        .set_size(size_)
                        .set_data_type(indices_dtype_)
                        .set_storage()
                        .create();
  void *data = indices_tensor.get_raw_handle_unsafe();
  int num_indices = size_[0];

  std::random_device rd;
  std::mt19937 gen(rd());

  if (indices_dtype_ == data_type_t::s32) {
    std::uniform_int_distribution<int32_t> dist(0,
        static_cast<int32_t>(num_embeddings_ - 1));

    for (int i = 0; i < num_indices; ++i) {
      static_cast<int32_t *>(data)[i] = dist(gen);
    }
  }
  else if (indices_dtype_ == data_type_t::s64) {
    std::uniform_int_distribution<int64_t> dist(0,
        static_cast<int64_t>(num_embeddings_ - 1));

    for (int i = 0; i < num_indices; ++i) {
      static_cast<int64_t *>(data)[i] = dist(gen);
    }
  }
  else {
    log_warning("tensor ", indices_tensor.get_name(), " unsupported data type.");
  }

  return indices_tensor;
}

tensor_t tensor_factory_t::random_offsets_tensor(const std::vector<index_type>
    size_, uint64_t num_indices_,
    data_type_t offsets_dtype_,
    bool include_last_offset_) {
  auto tensor = tensor_t()
                .set_name("offsets_tensor")
                .set_size(size_)
                .set_data_type(offsets_dtype_)
                .set_storage()
                .create();
  void *data = tensor.get_raw_handle_unsafe();

  int num_offsets = size_[0];
  if (include_last_offset_) {
    num_offsets--;
  }

  if (offsets_dtype_ == data_type_t::s32) {
    for (int i = 0; i < num_offsets; ++i) {
      static_cast<int32_t *>(data)[i] = (i * num_indices_) / num_offsets;
    }

    if (include_last_offset_) {
      static_cast<int32_t *>(data)[num_offsets] = num_indices_;
    }
  }
  else if (offsets_dtype_ == data_type_t::s64) {
    for (int i = 0; i < num_offsets; ++i) {
      static_cast<int64_t *>(data)[i] = (i * num_indices_) / num_offsets;
    }

    if (include_last_offset_) {
      static_cast<int64_t *>(data)[num_offsets] = num_indices_;
    }
  }
  else {
    log_warning("tensor ", tensor.get_name(), " unsupported data type.");
  }

  return tensor;
}

// Convert float32 to float16 (stored as uint16_t)
uint16_t float_to_half(float f) {
  uint32_t x;
  std::memcpy(&x, &f, sizeof(x));

  uint32_t sign = (x >> 31) & 0x1;
  int32_t exponent = ((x >> 23) & 0xFF) - 127 + 15;
  uint32_t mantissa = (x >> 13) & 0x3FF;

  if (exponent <= 0) {
    if (exponent < -10) {
      return static_cast<uint16_t>(sign << 15);
    }
    mantissa = (x & 0x7FFFFF) | 0x800000;
    mantissa >>= (1 - exponent + 13);
    return static_cast<uint16_t>((sign << 15) | mantissa);
  }
  else if (exponent >= 31) {
    return static_cast<uint16_t>((sign << 15) | (0x1F << 10));
  }

  return static_cast<uint16_t>((sign << 15) | (exponent << 10) | mantissa);
}

tensor_t tensor_factory_t::quantized_embedding_tensor_random(
  const std::vector<index_type> size_,
  data_type dtype_,
  std::string tensor_name_,
  bool fp16_scale_bias,
  float scale_min,
  float scale_max,
  int8_t zp_min,
  int8_t zp_max) {

  const int num_embeddings = size_[0];
  const int embedding_dim = size_[1];
  const int quantized_size = (dtype_ == data_type_t::s4 ||
                              dtype_ == data_type_t::u4) ?
                             (embedding_dim + 1) / 2 :
                             embedding_dim;
  const int row_size = quantized_size + (fp16_scale_bias ? 4 : 8);

  uint64_t num_bytes = static_cast<uint64_t>(num_embeddings) *
                       static_cast<uint64_t>(row_size) * sizeof(uint8_t);

  void *raw_buffer = malloc(num_bytes);

  if (!raw_buffer) {
    log_warning("malloc failed for ", num_bytes, " bytes");
    return tensor_t();
  }
  std::memset(raw_buffer, 0, num_bytes);

  auto qtensor = tensor_t()
                 .set_name(tensor_name_)
                 .set_size({static_cast<size_t>(num_embeddings), static_cast<size_t>(embedding_dim)})
                 .set_data_type(dtype_)
                 .set_storage(raw_buffer, num_bytes - (fp16_scale_bias ? 4 : 8))
                 .create();
  if (! qtensor.check()) {
    log_warning("tensor creation of ", qtensor.get_name(), " failed.");
    std::free(raw_buffer);
  }
  else {
    int8_t *input = static_cast<int8_t *>(raw_buffer);

    // Random generators
    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int> dist_s4(-8, 7);
    std::uniform_int_distribution<int> dist_u4(0, 15);
    std::uniform_int_distribution<int> dist_s8(-128, 127);
    std::uniform_real_distribution<float> scale_dist(scale_min, scale_max);
    std::uniform_int_distribution<int8_t> zp_dist(zp_min, zp_max);

    for (int i = 0; i < num_embeddings; ++i) {
      const size_t row_base = i * row_size;
      float scale = scale_dist(gen);
      float zp = static_cast<float>(zp_dist(gen));

      if (dtype_ == data_type_t::s4) {
        std::memset(input + row_base, 0, quantized_size);
        for (int j = 0; j < embedding_dim; ++j) {
          int8_t qval = dist_s4(gen);
          int byte_idx = j/2;
          if (j % 2 == 0) {
            input[row_base + byte_idx] = (qval & 0x0F);
          }
          else {
            input[row_base + byte_idx] &= 0x0F;
            input[row_base + byte_idx] |= (qval & 0x0F) << 4;
          }
        }
      }
      else if (dtype_ == data_type_t::u4) {
        std::memset(input + row_base, 0, quantized_size);
        for (int j = 0; j < embedding_dim; ++j) {
          uint8_t qval = dist_u4(gen);
          int byte_idx = j/2;
          if (j % 2 == 0) {
            input[row_base + byte_idx] = (qval & 0x0F);
          }
          else {
            input[row_base + byte_idx] &= 0x0F;
            input[row_base + byte_idx] |= (qval & 0x0F) << 4;
          }
        }
      }
      else {
        for (int j = 0; j < embedding_dim; ++j) {
          int8_t qval = dist_s8(gen);
          input[row_base + j] = qval;
        }
      }

      // Append scale and zp
      if (fp16_scale_bias) {
        uint16_t scale_fp16 = float_to_half(scale);
        uint16_t zp_fp16 = float_to_half(zp);
        std::memcpy(&input[row_base + quantized_size], &scale_fp16,
                    sizeof(uint16_t));
        std::memcpy(&input[row_base + quantized_size + 2], &zp_fp16,
                    sizeof(uint16_t));
      }
      else {
        std::memcpy(&input[row_base + quantized_size], &scale, sizeof(float));
        std::memcpy(&input[row_base + quantized_size + 4], &zp, sizeof(float));
      }
    }
  }
  return qtensor;
}

tensor_t tensor_factory_t::inverse_tensor(const tensor_t &input_tensor) {
  auto inv_tensor = tensor_t()
                    .set_size(input_tensor.get_size())
                    .set_data_type(input_tensor.get_data_type())
                    .set_storage()
                    .create();

  float *inv_scale_ptr = static_cast<float *>(inv_tensor.get_raw_handle_unsafe());

  uint64_t num_rows = input_tensor.get_size()[0];
  uint64_t num_cols = input_tensor.get_size()[1];

  for (uint64_t i = 0; i < num_rows; ++i) {
    for (uint64_t j = 0; j < num_cols; ++j) {
      float scale_value = input_tensor.at({i, j});
      uint64_t idx = i * num_cols + j;
      inv_scale_ptr[idx] = 1.0f / scale_value;
    }
  }

  return inv_tensor;
}

void Parser::operator()(const int &argc, char *argv[], int64_t &seed,
                        uint32_t &tests, std::string &po, std::string &backend, std::string &lowoha) {
  for (int i=1; i<argc; ++i) {
    std::string arg = argv[i];
    if (arg.rfind("--",0)==0 && arg.find("gtest")==std::string::npos && i+1<argc) {
      std::string key = arg.substr(2);
      umap[key] = argv[++i];
    }
  }
  read_from_umap("seed", seed);
  read_from_umap("test", tests);
  read_from_umap("postop", po);
  read_from_umap("backend", backend);
  read_from_umap("lowoha", lowoha);
  return;
}

void Parser::read_from_umap(const std::string &key, int64_t &num) {
  if (umap.count(key)) {
    std::string val = umap.at(key);
    if (isInteger(val)) {
      try {
        num = static_cast<int64_t>(stol(val));
      }
      catch (const std::out_of_range &e) {
        log_info("Out-of-range argument for ", key,
                 ", so using default value i.e. timestamp.");
      }
    }
    else {
      log_info("Invalid argument for ", key,
               ", so using default value i.e. timestamp.");
    }
  }
  else {
    log_info("No argument for ", key, ", so using default value i.e. timestamp.");
  }
}

void Parser::read_from_umap(const std::string &key, uint32_t &num) {
  if (umap.count(key)) {
    std::string val = umap.at(key);
    if (isInteger(val) && val[0] != '-') {
      try {
        num = static_cast<uint32_t>(stoul(val));
      }
      catch (const std::out_of_range &e) {
        log_info("Out-of-range argument for ", key,
                 ", so using default value i.e. 1000.");
      }
    }
    else {
      log_info("Invalid argument for ", key, ", so using default value i.e. 1000.");
    }
  }
  else {
    log_info("No argument for ", key, ", so using default value i.e. 1000.");
  }
}

void Parser::read_from_umap(const std::string &key, std::string &num) {
  if (umap.count(key)) {
    num = umap.at(key);
  }
  else {
    log_info("No argument for ", key,
             ", so using the random ", key, " from supported list.");
  }
}

bool Parser::isInteger(const std::string &s) {
  if (s.empty()) {
    return false;
  }
  size_t i = 0;
  if (s[0] == '+' || s[0] == '-') {
    i = 1;
  }
  for (; i < s.size(); ++i) {
    if (!isdigit(s[i])) {
      return false;
    }
  }
  return true;
}

matmul_algo_t strToAlgo(std::string str) {
  if (str == "aocl_dlp") {
    return matmul_algo_t::aocl_dlp;
  }
  if (str == "aocl_dlp_blocked") {
    return matmul_algo_t::aocl_dlp_blocked;
  }
  if (str == "onednn") {
    return matmul_algo_t::onednn;
  }
  if (str == "onednn_blocked") {
    return matmul_algo_t::onednn_blocked;
  }
  if (str == "libxsmm") {
    return matmul_algo_t::libxsmm;
  }
  if (str == "libxsmm_blocked") {
    return matmul_algo_t::libxsmm_blocked;
  }
  return matmul_algo_t::none;
}

std::string algoToStr(matmul_algo_t algo) {
  switch (algo) {
  case matmul_algo_t::aocl_dlp:
    return "aocl_dlp";
  case matmul_algo_t::aocl_dlp_blocked:
    return "aocl_dlp_blocked";
  case matmul_algo_t::onednn:
    return "onednn";
  case matmul_algo_t::onednn_blocked:
    return "onednn_blocked";
  case matmul_algo_t::libxsmm:
    return "libxsmm";
  case matmul_algo_t::libxsmm_blocked:
    return "libxsmm_blocked";
  default:
    return "none";
  }
}

post_op_type_t strToPostOps(const std::string &str) {
  if (str == "relu") {
    return post_op_type_t::relu;
  }
  if (str == "gelu_tanh") {
    return post_op_type_t::gelu_tanh;
  }
  if (str == "gelu_erf") {
    return post_op_type_t::gelu_erf;
  }
  if (str == "sigmoid") {
    return post_op_type_t::sigmoid;
  }
  if (str == "swish") {
    return post_op_type_t::swish;
  }
  if (str == "tanh") {
    return post_op_type_t::tanh;
  }
  if (str == "binary_add") {
    return post_op_type_t::binary_add;
  }
  if (str == "binary_mul") {
    return post_op_type_t::binary_mul;
  }
  return post_op_type_t::none;
}

std::string postOpsToStr(post_op_type_t post_op) {
  switch (post_op) {
  case post_op_type_t::relu:
    return "relu";
  case post_op_type_t::gelu_tanh:
    return "gelu_tanh";
  case post_op_type_t::gelu_erf:
    return "gelu_erf";
  case post_op_type_t::sigmoid:
    return "sigmoid";
  case post_op_type_t::swish:
    return "swish";
  case post_op_type_t::tanh:
    return "tanh";
  case post_op_type_t::binary_add:
    return "binary_add";
  case post_op_type_t::binary_mul:
    return "binary_mul";
  default:
    return "none";
  }
}

status_t matmul_kernel_test(tensor_t &input_tensor, tensor_t &weight_tensor,
                            tensor_t &bias_tensor, tensor_t &output_tensor,
                            post_op_type_t po_type, tensor_t &binary_tensor, bool use_LOWOHA,
                            matmul_algo_t algo,
                            float alpha,
                            float beta) {
  try {

    if (use_LOWOHA) {
      try {
        // Validate input tensors
        if (!input_tensor.check() || !weight_tensor.check() || !output_tensor.check()) {
          log_error("LOWOHA: Invalid tensor state detected");
          return status_t::failure;
        }
        auto input_dim              = input_tensor.get_dim();
        auto weight_dim             = weight_tensor.get_dim();
        auto output_dim             = output_tensor.get_dim();
        if (input_dim < 2 || input_dim > 3 || weight_dim < 2 || weight_dim > 3 ||
            output_dim < 2 || output_dim > 3 ||
            !((input_dim == weight_dim && output_dim == input_dim) ||
              (input_dim == 2 && weight_dim == 3 && output_dim == 3) ||
              (input_dim == 3 && weight_dim == 2 && output_dim == 3))) {
          log_error("LOWOHA: Invalid tensor dimensions - Input dim:", input_dim,
                    " Weight dim:", weight_dim, " Output dim:", output_dim);
          return status_t::failure;
        }
        if (input_tensor.get_size(input_dim - 2) != output_tensor.get_size(
              output_dim - 2) ||
            input_tensor.get_size(input_dim - 1) != weight_tensor.get_size(
              weight_dim - 2) ||
            weight_tensor.get_size(weight_dim - 1) != output_tensor.get_size(
              output_dim - 1)) {
          log_error("LOWOHA: Mismatched tensor dimensions - Input sizes: [",
                    input_tensor.get_size(input_dim - 2),
                    ", ", input_tensor.get_size(input_dim - 1), "], Weight sizes: [",
                    weight_tensor.get_size(weight_dim - 2),
                    ", ", weight_tensor.get_size(weight_dim - 1), "], Output sizes: [",
                    output_tensor.get_size(output_dim - 2),
                    ", ", output_tensor.get_size(output_dim - 1), "]");
          return status_t::failure;
        }
        bool transA       = (input_dim == 2)  ? (input_tensor.get_order() ==
                            "ba") : (input_tensor.get_order() == "acb");
        bool transB       = (weight_dim == 2) ? (weight_tensor.get_order() ==
                            "ba") : (weight_tensor.get_order() == "acb");

        const int   lda             = transA ?
                                      input_tensor.get_stride(input_dim-1) :
                                      input_tensor.get_stride(input_dim-2);
        const int   ldb             = transB ?
                                      weight_tensor.get_stride(weight_dim-1):
                                      weight_tensor.get_stride(weight_dim-2);
        const int   ldc             = output_tensor.get_stride(output_dim-2);

        // Extract tensor dimensions
        const int batchA            = (input_dim==3) ? input_tensor.get_size(
                                        input_dim-3) : 1;
        const int batchB            = (weight_dim==3) ? weight_tensor.get_size(
                                        weight_dim-3) : 1;
        const int batchC            = (output_dim==3) ? output_tensor.get_size(
                                        output_dim-3) : 1;

        const int M                 = output_tensor.get_size(output_dim-2);
        const int K                 = input_tensor.get_size(input_dim-1);
        const int N                 = output_tensor.get_size(output_dim-1);
        // Validate dimensions
        if (M == 0 || K == 0 || N == 0) {
          log_error("LOWOHA: Invalid tensor dimensions - M:", M, " K:", K, " N:", N);
          return status_t::failure;
        }
        if (std::max(batchA, batchB) != batchC) {
          log_error("Invalid output batch size");
          return status_t::failure;
        }

        // Get tensor data pointers
        void *A_data = input_tensor.get_raw_handle_unsafe();
        void *B_data = weight_tensor.get_raw_handle_unsafe();
        void *C_data = output_tensor.get_raw_handle_unsafe();
        void *bias_data = bias_tensor.get_raw_handle_unsafe();

        //TODO: For LIBXSMM matmul, bias is not supported currently due to accuracy issues
        if ((algo == matmul_algo_t::libxsmm ||
             algo == matmul_algo_t::libxsmm_blocked) &&
            output_tensor.get_data_type() == data_type_t::bf16) {
          bias_data = nullptr;
        }

        // Validate data pointers
        if (!A_data || !B_data || !C_data) {
          log_error("LOWOHA: Null data pointer detected");
          return status_t::failure;
        }

        // Get data types
        data_type_t src_data_type = input_tensor.get_data_type();
        data_type_t wei_data_type = weight_tensor.get_data_type();
        data_type_t out_data_type = output_tensor.get_data_type();
        data_type_t bias_data_type = bias_tensor.get_data_type();
        matmul_data_types matmul_dtypes;
        matmul_dtypes.src = src_data_type;
        matmul_dtypes.wei = wei_data_type;
        matmul_dtypes.dst = out_data_type;
        matmul_dtypes.bias = bias_data_type;
        matmul_dtypes.compute = data_type_t::none;

        // Validate data types
        if (src_data_type != data_type_t::f32 && src_data_type != data_type_t::bf16 &&
            src_data_type != data_type_t::u8 && src_data_type != data_type_t::s8) {
          log_error("LOWOHA: Unsupported source data type");
          return status_t::failure;
        }
        if (out_data_type != data_type_t::f32 && out_data_type != data_type_t::bf16 &&
            out_data_type != data_type_t::u8 && out_data_type != data_type_t::s8 &&
            out_data_type != data_type_t::s32) {
          log_error("LOWOHA: Unsupported output data type");
          return status_t::failure;
        }

        // Check if this is WOQ (Weight-Only Quantization): BF16 src + S4 weights
        bool is_woq = (src_data_type == data_type_t::bf16 &&
                       wei_data_type == data_type_t::s4);

        // Check if this is INT8 quantization
        bool is_int8 = (src_data_type == data_type_t::u8 ||
                        src_data_type == data_type_t::s8) &&
                       wei_data_type == data_type_t::s8;

        log_info("LOWOHA: Calling matmul_direct with batchA:", batchA, " batchB:",
                 batchB, " M:", M, " N:", N, " K:", K,
                 " alpha:", alpha, " beta:", beta, " is_woq:", is_woq, " is_int8:", is_int8);

        // Extract batch strides from tensors if they have batch dimension (3D)
        // Batch strides are in elements, not bytes
        size_t batch_stride_src = static_cast<size_t>(-1);
        size_t batch_stride_wei = static_cast<size_t>(-1);
        size_t batch_stride_dst = static_cast<size_t>(-1);

        if (input_dim == 3) {
          // For 3D input tensor, get stride of batch dimension (dimension 0) in elements
          batch_stride_src = input_tensor.get_stride(0);
        }
        if (weight_dim == 3) {
          // For 3D weight tensor, get stride of batch dimension (dimension 0) in elements
          batch_stride_wei = weight_tensor.get_stride(0);
        }
        if (output_dim == 3) {
          // For 3D output tensor, get stride of batch dimension (dimension 0) in elements
          batch_stride_dst = output_tensor.get_stride(0);
        }

        // Create lowoha_post_op structure
        matmul_params params;
        params.lowoha_algo = algo;
        params.dtypes = matmul_dtypes;
        params.num_threads = 0; // Use default (omp_get_max_threads)

        // For WOQ: Extract quantization parameters from weight tensor
        if (is_woq) {
          // Extract weight scale
          const void *scale_buff = weight_tensor.get_quant_scale_raw_handle_const();
          params.quant_params.wei_scale.buff = scale_buff;
          params.quant_params.wei_scale.dt = weight_tensor.get_quant_scale_data_type();
          auto scale_size = weight_tensor.get_quant_scale_size();
          params.quant_params.wei_scale.dims.assign(scale_size.begin(), scale_size.end());
          log_info("LOWOHA WOQ: Weight scale extracted, dims: [",
                   params.quant_params.wei_scale.dims.size() > 0 ?
                   params.quant_params.wei_scale.dims[0] : 0,
                   params.quant_params.wei_scale.dims.size() > 1 ?
                   params.quant_params.wei_scale.dims[1] : 0, "]");


          // Extract weight zero point (if asymmetric quantization)
          if (weight_tensor.get_quant_subtype() == quant_subtype_t::asymmetric) {
            const void *zp_buff = weight_tensor.get_quant_zero_raw_handle_const();
            if (zp_buff) {
              params.quant_params.wei_zp.buff = zp_buff;
              params.quant_params.wei_zp.dt = weight_tensor.get_quant_zero_data_type();
              auto zp_size = weight_tensor.get_quant_zero_size();
              params.quant_params.wei_zp.dims.assign(zp_size.begin(), zp_size.end());
              log_info("LOWOHA WOQ: Weight zero point extracted");
            }
          }
        }

        // For INT8: Extract quantization parameters from all tensors
        if (is_int8) {
          // Extract source scale
          if (input_tensor.is_quantized()) {
            const void *src_scale_buff = input_tensor.get_quant_scale_raw_handle_const();
            if (src_scale_buff) {
              params.quant_params.src_scale.buff = src_scale_buff;
              params.quant_params.src_scale.dt = input_tensor.get_quant_scale_data_type();
              auto src_scale_size = input_tensor.get_quant_scale_size();
              params.quant_params.src_scale.dims.assign(src_scale_size.begin(),
                  src_scale_size.end());
              log_info("LOWOHA INT8: Source scale extracted");
            }
            // Extract source zero point (for asymmetric quantization)
            if (input_tensor.get_quant_subtype() == quant_subtype_t::asymmetric) {
              const void *src_zp_buff = input_tensor.get_quant_zero_raw_handle_const();
              if (src_zp_buff) {
                params.quant_params.src_zp.buff = src_zp_buff;
                params.quant_params.src_zp.dt = input_tensor.get_quant_zero_data_type();
                auto src_zp_size = input_tensor.get_quant_zero_size();
                params.quant_params.src_zp.dims.assign(src_zp_size.begin(), src_zp_size.end());
                log_info("LOWOHA INT8: Source zero-point extracted");
              }
            }
          }

          // Extract weight scale
          if (weight_tensor.is_quantized()) {
            const void *wei_scale_buff = weight_tensor.get_quant_scale_raw_handle_const();
            if (wei_scale_buff) {
              params.quant_params.wei_scale.buff = wei_scale_buff;
              params.quant_params.wei_scale.dt = weight_tensor.get_quant_scale_data_type();
              auto wei_scale_size = weight_tensor.get_quant_scale_size();
              params.quant_params.wei_scale.dims.assign(wei_scale_size.begin(),
                  wei_scale_size.end());
              log_info("LOWOHA INT8: Weight scale extracted");
            }
            // Extract weight zero point (for asymmetric quantization)
            if (weight_tensor.get_quant_subtype() == quant_subtype_t::asymmetric) {
              const void *wei_zp_buff = weight_tensor.get_quant_zero_raw_handle_const();
              if (wei_zp_buff) {
                params.quant_params.wei_zp.buff = wei_zp_buff;
                params.quant_params.wei_zp.dt = weight_tensor.get_quant_zero_data_type();
                auto wei_zp_size = weight_tensor.get_quant_zero_size();
                params.quant_params.wei_zp.dims.assign(wei_zp_size.begin(), wei_zp_size.end());
                log_info("LOWOHA INT8: Weight zero-point extracted");
              }
            }
          }

          // Extract destination scale and zero-point
          if (output_tensor.is_quantized()) {
            const void *dst_scale_buff = output_tensor.get_quant_scale_raw_handle_const();
            if (dst_scale_buff) {
              params.quant_params.dst_scale.buff = dst_scale_buff;
              params.quant_params.dst_scale.dt = output_tensor.get_quant_scale_data_type();
              auto dst_scale_size = output_tensor.get_quant_scale_size();
              params.quant_params.dst_scale.dims.assign(dst_scale_size.begin(),
                  dst_scale_size.end());
              log_info("LOWOHA INT8: Destination scale extracted");
            }
            // Extract destination zero point (for asymmetric quantization)
            if (output_tensor.get_quant_subtype() == quant_subtype_t::asymmetric) {
              const void *dst_zp_buff = output_tensor.get_quant_zero_raw_handle_const();
              if (dst_zp_buff) {
                params.quant_params.dst_zp.buff = dst_zp_buff;
                params.quant_params.dst_zp.dt = output_tensor.get_quant_zero_data_type();
                auto dst_zp_size = output_tensor.get_quant_zero_size();
                params.quant_params.dst_zp.dims.assign(dst_zp_size.begin(), dst_zp_size.end());
                log_info("LOWOHA INT8: Destination zero-point extracted");
              }
            }
          }
        }

        // Create batch_params structure
        matmul_batch_params_t batch_params;
        batch_params.Batch_A = batchA;
        batch_params.Batch_B = batchB;
        batch_params.batch_stride_src = batch_stride_src;
        batch_params.batch_stride_wei = batch_stride_wei;
        batch_params.batch_stride_dst = batch_stride_dst;

        // Add post-ops based on po_type
        if (po_type != post_op_type_t::none) {
          matmul_post_op postop_item;
          postop_item.po_type = po_type;

          // For binary operations, set the buffer to binary_tensor
          if (po_type == post_op_type_t::binary_add ||
              po_type == post_op_type_t::binary_mul) {
            postop_item.buff = binary_tensor.get_raw_handle_unsafe();
            postop_item.dtype = binary_tensor.get_data_type();
            auto binary_tensor_dims = binary_tensor.get_size();
            postop_item.dims.assign(binary_tensor_dims.begin(), binary_tensor_dims.end());
          }
          else {
            postop_item.buff = nullptr; // For element-wise operations
            postop_item.dtype = out_data_type;
          }

          params.postop_.push_back(postop_item);
        }
        status_t status = matmul_direct(
                            'r',  // layout: row-major
                            transA, transB,
                            static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                            alpha, A_data, lda, B_data, ldb, bias_data,  // No bias
                            beta, C_data, ldc, (is_woq || is_int8) ? true : rand() % 2 == 0 ? true : false,
                            batch_params, params);
        if (status != status_t::success) {
          log_error("LOWOHA matmul_direct execution failed.");
          return status_t::failure;
        }
      }
      catch (const std::exception &e) {
        log_error("LOWOHA matmul_direct execution failed: ", e.what());
        return status_t::failure;
      }
      catch (...) {
        log_error("LOWOHA matmul_direct execution failed with unknown exception");
        return status_t::failure;
      }
    }
    else {
      // default postop relu
      post_op_t post_op = post_op_t{post_op_type_t::relu};
      // postop update according to the post_op_type_t enum value
      if (po_type != post_op_type_t::none) {
        post_op = post_op_t{po_type};
      }
      weight_tensor.set_name("weights");
      bias_tensor.set_name("bias");

      //define matmul context
      matmul_context_t matmul_context = matmul_context_t()
                                        .set_param("weights", weight_tensor)
                                        .set_param("bias", bias_tensor)
                                        .set_alpha(alpha)
                                        .set_beta(beta);
      if (po_type != post_op_type_t::none) {
        matmul_context = matmul_context.set_post_op(post_op).create();
      }
      else {
        matmul_context = matmul_context.create();//No Postop case
      }

      //define matmul operator
      matmul_operator_t matmul_operator = matmul_operator_t()
                                          .set_name("matmul_operator")
                                          .set_context(matmul_context)
                                          .create();

      if (matmul_operator.is_bad_object()) {
        log_error("operator ", matmul_operator.get_name(), " creation failed.");
        return status_t::failure;
      }

      input_tensor.set_name("matmul_input");
      output_tensor.set_name("matmul_output");
      // Set binary tensor for binary postops
      if (po_type != post_op_type_t::none) {
        if (po_type == post_op_type_t::binary_add) {
          matmul_operator.set_input(post_op.binary_add_params.tensor_name, binary_tensor);
        }
        else if (po_type == post_op_type_t::binary_mul) {
          matmul_operator.set_input(post_op.binary_mul_params.tensor_name, binary_tensor);
        }
      }
      matmul_operator.set_input("matmul_input", input_tensor)
      .set_output("matmul_output", output_tensor);
      if (algo != matmul_algo_t::none) {
        matmul_operator.set_forced_kernel(algoToStr(algo));
      }
      status_t status = matmul_operator.execute();

      if (status != status_t::success) {
        log_info("operator ", matmul_operator.get_name(), " execution failed.");
        return status_t::failure;
      }

    }

  }
  catch (const exception_t &ex) {
    log_verbose(ex.what());
    return status_t::failure;
  }
  return status_t::success;
}

status_t matmul_forced_ref_kernel_test(tensor_t &input_tensor,
                                       tensor_t &weight_tensor,
                                       tensor_t &bias_tensor, tensor_t &output_tensor,
                                       post_op_type_t po_type, tensor_t &binary_tensor, bool use_LOWOHA,
                                       matmul_algo_t algo,
                                       float alpha,
                                       float beta) {
  try {
    // Default postop relu
    post_op_t post_op = post_op_t{post_op_type_t::relu};
    // postop update according to the post_op_type_t enum value
    if (po_type != post_op_type_t::none) {
      post_op = post_op_t{po_type};
    }
    weight_tensor.set_name("weights");
    bias_tensor.set_name("bias");

    //define matmul context
    matmul_context_t matmul_context = matmul_context_t()
                                      .set_param("weights", weight_tensor)
                                      .set_alpha(alpha)
                                      .set_beta(beta);

    //TODO: For LIBXSMM matmul, bias is not supported currently due to accuracy issues
    if (!((algo == matmul_algo_t::libxsmm ||
           algo == matmul_algo_t::libxsmm_blocked) &&
          output_tensor.get_data_type() == data_type_t::bf16)) {
      matmul_context = matmul_context.set_param("bias", bias_tensor);
    }


    if (po_type != post_op_type_t::none) {
      matmul_context = matmul_context.set_post_op(post_op).create();
    }
    else {
      matmul_context = matmul_context.create(); //No postop case
    }

    //define matmul operator
    matmul_operator_t matmul_operator = matmul_operator_t()
                                        .set_name("matmul_forced_ref_operator")
                                        .set_context(matmul_context)
                                        .create();

    if (matmul_operator.is_bad_object()) {
      log_error("operator ", matmul_operator.get_name(), " creation failed.");
      return status_t::failure;
    }
    input_tensor.set_name("matmul_input");
    output_tensor.set_name("matmul_output");

    if (po_type != post_op_type_t::none) {
      if (po_type == post_op_type_t::binary_add) {
        matmul_operator.set_input(post_op.binary_add_params.tensor_name, binary_tensor);
      }
      else if (po_type == post_op_type_t::binary_mul) {
        // Set binary tensor for binary postops
        matmul_operator.set_input(post_op.binary_mul_params.tensor_name, binary_tensor);
      }
    }
    status_t status = matmul_operator.set_input("matmul_input", input_tensor)
                      .set_output("matmul_output", output_tensor)
                      .set_forced_kernel("reference")
                      .execute();

    if (status != status_t::success) {
      log_info("operator ", matmul_operator.get_name(), " execution failed.");
      return status_t::failure;
    }
  }
  catch (const exception_t &ex) {
    log_verbose(ex.what());
    return status_t::failure;
  }
  return status_t::success;
}

std::pair<tensor_t, status_t> reorder_kernel_test(tensor_t &input_tensor,
    bool inplace_reorder, void **weights, data_type_t source_dtype) {
  try {
    tensor_factory_t tensor_factory;
    status_t status;

    input_tensor.set_name("reorder_input");
    data_type_t dtype           = input_tensor.get_data_type();

    // Reorder context creation with backend aocl.
    reorder_context_t reorder_context = reorder_context_t()
                                        .set_algo_format("aocl");

    // Set Input source dtype if the weights dtype is s8
    if ((dtype == data_type_t::s8) && (source_dtype == data_type_t::s8 ||
                                       source_dtype == data_type_t::u8)) {
      reorder_context.set_source_dtype(source_dtype);
    }
    reorder_context.create();

    uint64_t rows               = input_tensor.get_size(0);
    uint64_t cols               = input_tensor.get_size(1);
    tensor_t output_tensor{};  // Initialize to avoid UNINIT issue

    bool memory_reorder         = (!(input_tensor.get_layout() | uint16_t(
                                       tensor_layout_t::contiguous)) ||
                                   (input_tensor.get_layout() & uint16_t(tensor_layout_t::aligned)));
    bool memory_unreorder       = (input_tensor.get_layout() & uint16_t(
                                     tensor_layout_t::blocked));
    bool trans                  = (input_tensor.get_order() == "ba") ? true : false;

    if (memory_reorder) {
      // Reorder operator creation with name, context and input.
      reorder_operator_t reorder_operator = reorder_operator_t()
                                            .set_name("reorder_operator")
                                            .set_context(reorder_context)
                                            .create()
                                            .set_input("reorder_input", input_tensor);

      if (reorder_operator.is_bad_object()) {
        log_error("operator ", reorder_operator.get_name(), " creation failed.");
        return std::make_pair(tensor_t(), status_t::failure);
      }

      // Compute the reorder size
      size_t reorder_size         = reorder_operator.get_reorder_size();
      // Extract the input buffer size
      size_t input_buffer_size    = input_tensor.get_buffer_sz_bytes();

      if (inplace_reorder) {
        // InPlace reorder works when reorder size is equal to input buffer size.
        if (reorder_size != input_buffer_size) {
          log_info("Inplace reorder is not possible for given input");
          return std::make_pair(tensor_t(), status_t::unimplemented);
        }
        else {
          // Assign input_tensor to buffer_params as a tensor_t variant
          StorageParam buffer_params = std::move(input_tensor);

          // Output Tensor creation with separate view for input tensor
          output_tensor = tensor_factory.copy_tensor({rows, cols},
                          dtype,
                          buffer_params, trans, true);
          output_tensor.set_name("reorder_output");
        }
      }
      else {
        // create a buffer with reorderd size
        size_t alignment = 64;
        reorder_size = get_aligned_size(alignment, reorder_size);
        *weights = aligned_alloc(alignment, reorder_size);

        if (*weights == nullptr) {
          log_info("weights can not have align allocation");
          return std::make_pair(tensor_t(), status_t::failure);
        }

        // Create a Pair of storage params [reorder size and reorder weights] and
        // use it in tensor creation
        StorageParam buffer_params = std::make_pair(reorder_size, *weights);

        // Create output tensor with blocked layout.
        output_tensor = tensor_factory.copy_tensor({rows, cols},
                        dtype,
                        buffer_params, trans, true);
        output_tensor.set_name("reorder_output");
      }
      // Reorder operator execution.
      status = reorder_operator
               .set_output("reorder_output", output_tensor)
               .execute();

      if (status != status_t::success) {
        log_info("operator ", reorder_operator.get_name(), " execution failed.");
        return std::make_pair(tensor_t(), status_t::failure);
      }
      else {
        log_info("operator ", reorder_operator.get_name(), " execution successful.");
        return std::make_pair(output_tensor, status_t::success);
      }
    }
    else {
      // reorder operator creation with name, context and input.
      reorder_operator_t reorder_operator = reorder_operator_t()
                                            .set_name("reorder_operator")
                                            .set_context(reorder_context)
                                            .create()
                                            .set_input("reorder_input", input_tensor);

      if (reorder_operator.is_bad_object()) {
        log_error("operator ", reorder_operator.get_name(), " creation failed.");
        return std::make_pair(tensor_t(), status_t::failure);
      }

      // Inplace reorder
      if (inplace_reorder) {
        StorageParam buffer_params = std::move(input_tensor);

        output_tensor = tensor_factory.copy_tensor({rows, cols},
                        dtype,
                        buffer_params, trans, false);
        output_tensor.set_name("reorder_output");
      }
      else if (memory_unreorder) {
        // Compute the output buffer size for reorder
        auto reorder_size = reorder_operator.get_reorder_size();

        // create a buffer with reorderd size
        size_t alignment = 64;
        reorder_size = get_aligned_size(alignment, reorder_size);
        *weights = aligned_alloc(alignment, reorder_size);

        if (*weights == nullptr) {
          log_info("weights can not have align allocation");
          return std::make_pair(tensor_t(), status_t::failure);
        }

        // Create a Pair of storage params [reorder size and reorder weights] and
        // use it in tensor creation
        StorageParam buffer_params = std::make_pair(reorder_size, *weights);

        // Create output tensor with contiguous layout.
        output_tensor = tensor_factory.copy_tensor({rows, cols},
                        dtype,
                        buffer_params, trans, false);
        output_tensor.set_name("reorder_output");
      }

      // Reorder operator execution.
      status = reorder_operator
               .set_output("reorder_output", output_tensor)
               .execute();

      if (status != status_t::success) {
        log_info("operator ", reorder_operator.get_name(), " execution failed.");
        return std::make_pair(tensor_t(), status_t::failure);
      }
      else {
        log_info("operator ", reorder_operator.get_name(), " execution successful.");
        return std::make_pair(output_tensor, status_t::success);
      }
    }
  }
  catch (const exception_t &ex) {
    log_verbose(ex.what());
    return std::make_pair(tensor_t(), status_t::failure);
  }
}

status_t embag_kernel_test(tensor_t &table_tensor,
                           tensor_t &indices_tensor,
                           tensor_t &offsets_tensor,
                           tensor_t &weights_tensor,
                           tensor_t &output_tensor,
                           embag_algo_t algo,
                           int64_t padding_index,
                           bool include_last_offset,
                           bool is_weights,
                           bool fp16_scale_bias,
                           bool use_LOWOHA) {
  try {
    status_t status;

    if (use_LOWOHA) {
      // LOWOHA path - use embedding_bag_direct API
      try {
        // Validate input tensors
        if (!table_tensor.check() || !indices_tensor.check() ||
            !offsets_tensor.check() || !output_tensor.check()) {
          log_error("LOWOHA embag: Invalid tensor state detected");
          return status_t::failure;
        }

        // Get raw data pointers
        void *table_data = table_tensor.get_raw_handle_unsafe();
        void *indices_data = indices_tensor.get_raw_handle_unsafe();
        void *offsets_data = offsets_tensor.get_raw_handle_unsafe();
        float *weights_data = is_weights ? (float *)
                              weights_tensor.get_raw_handle_unsafe() : nullptr;
        void *output_data = output_tensor.get_raw_handle_unsafe();

        if (!table_data || !indices_data || !offsets_data || !output_data) {
          log_error("LOWOHA embag: Null data pointer detected");
          return status_t::failure;
        }

        // Build embag_params_t structure
        embag_params_t params;

        // Set data types
        params.dtypes.table = table_tensor.get_data_type();
        params.dtypes.output = output_tensor.get_data_type();
        params.dtypes.indices = indices_tensor.get_data_type();
        params.dtypes.offsets = offsets_tensor.get_data_type();

        // Use algo directly (embag_algo_t is aliased to ops::embag_algo_t)
        params.algo = algo;

        // Set dimensions
        params.num_embeddings = table_tensor.get_size(0);
        params.embedding_dim = table_tensor.get_size(1);
        params.num_indices = indices_tensor.get_size(0);
        params.num_bags = include_last_offset ?
                          offsets_tensor.get_size(0) - 1 :
                          offsets_tensor.get_size(0);
        params.is_weights = is_weights;
        params.include_last_offset = include_last_offset;
        params.padding_idx = padding_index;
        params.num_threads = 0;  // Use default (omp_get_max_threads)
        params.fp16_scale_bias = fp16_scale_bias;
        params.dst_stride = output_tensor.get_stride()[0];

        log_info("LOWOHA embag: Calling embedding_bag_direct with "
                 "num_embeddings=", params.num_embeddings,
                 ", embedding_dim=", params.embedding_dim,
                 ", num_indices=", params.num_indices,
                 ", num_bags=", params.num_bags);

        // Call LOWOHA embedding_bag_direct API
        status = embedding_bag_direct(
                   table_data,
                   indices_data,
                   offsets_data,
                   weights_data,
                   output_data,
                   params);

        if (status != status_t::success) {
          log_error("LOWOHA embedding_bag_direct execution failed.");
          return status_t::failure;
        }
      }
      catch (const std::exception &e) {
        log_error("LOWOHA embedding_bag_direct execution failed: ", e.what());
        return status_t::failure;
      }
      catch (...) {
        log_error("LOWOHA embedding_bag_direct execution failed with unknown exception");
        return status_t::failure;
      }
    }
    else {
      // Regular operator API path
      //define embag context
      embag_context_t embedding_bag_context = embag_context_t()
                                              .set_param("table", table_tensor)
                                              .set_algo(algo)
                                              .set_padding_index(padding_index)
                                              .set_include_last_offset(include_last_offset)
                                              .set_is_weights(is_weights);
      if (table_tensor.get_data_type() == data_type_t::s8 ||
          table_tensor.get_data_type() == data_type_t::s4 ||
          table_tensor.get_data_type() == data_type_t::u4) {
        embedding_bag_context.set_fp16_scale_bias(fp16_scale_bias);
        embedding_bag_context.create();
      }
      else {
        embedding_bag_context.create();
      }

      //define embedding bag operator
      embag_operator_t embedding_bag_operator = embag_operator_t()
          .set_name("embedding_bag")
          .set_context(embedding_bag_context)
          .create();

      if (embedding_bag_operator.is_bad_object()) {
        testlog_error(" operator ", embedding_bag_operator.get_name(),
                      " creation failed.");
        return status_t::failure;
      }

      if (is_weights) {
        // Execute operator
        status = embedding_bag_operator
                 .set_input("indices", indices_tensor)
                 .set_input("weights", weights_tensor)
                 .set_input("offsets", offsets_tensor)
                 .set_output("output", output_tensor)
                 .execute();
      }
      else {
        status = embedding_bag_operator
                 .set_input("indices", indices_tensor)
                 .set_input("offsets", offsets_tensor)
                 .set_output("output", output_tensor)
                 .execute();
      }

      if (status != status_t::success) {
        log_info("operator ", embedding_bag_operator.get_name(), " execution failed.");
        return status_t::failure;
      }
    }
  }
  catch (const exception_t &ex) {
    log_verbose(ex.what());
    return status_t::failure;
  }
  return status_t::success;
}

status_t embag_forced_ref_kernel_test(tensor_t &table_tensor,
                                      tensor_t &indices_tensor,
                                      tensor_t &offsets_tensor,
                                      tensor_t &weights_tensor,
                                      tensor_t &output_tensor,
                                      embag_algo_t algo,
                                      int64_t padding_index,
                                      bool include_last_offset,
                                      bool is_weights,
                                      bool fp16_scale_bias) {
  try {
    status_t status;

    //define embag context
    embag_context_t embedding_bag_context = embag_context_t()
                                            .set_param("table", table_tensor)
                                            .set_algo(algo)
                                            .set_padding_index(padding_index)
                                            .set_include_last_offset(include_last_offset)
                                            .set_is_weights(is_weights);
    if (table_tensor.get_data_type() == data_type_t::s8 ||
        table_tensor.get_data_type() == data_type_t::s4 ||
        table_tensor.get_data_type() == data_type_t::u4) {
      embedding_bag_context.set_fp16_scale_bias(fp16_scale_bias);
      embedding_bag_context.create();
    }
    else {
      embedding_bag_context.create();
    }

    //define embedding bag operator
    embag_operator_t embedding_bag_operator = embag_operator_t()
        .set_name("ref_embedding_bag")
        .set_context(embedding_bag_context)
        .create();

    if (embedding_bag_operator.is_bad_object()) {
      testlog_error(" operator ", embedding_bag_operator.get_name(),
                    " creation failed.");
      return status_t::failure;
    }

    if (is_weights) {
      // Execute operator
      status = embedding_bag_operator
               .set_input("indices", indices_tensor)
               .set_input("weights", weights_tensor)
               .set_input("offsets", offsets_tensor)
               .set_output("output", output_tensor)
               .set_forced_kernel("reference")
               .execute();
    }
    else {
      // Execute operator
      status = embedding_bag_operator
               .set_input("indices", indices_tensor)
               .set_input("offsets", offsets_tensor)
               .set_output("output", output_tensor)
               .set_forced_kernel("reference")
               .execute();

    }

    if (status != status_t::success) {
      log_info("operator ", embedding_bag_operator.get_name(), " execution failed.");
      return status_t::failure;
    }
  }
  catch (const exception_t &ex) {
    log_verbose(ex.what());
    return status_t::failure;
  }
  return status_t::success;
}

status_t embedding_kernel_test(tensor_t &table_tensor,
                               tensor_t &indices_tensor,
                               tensor_t &weights_tensor,
                               tensor_t &output_tensor,
                               int64_t padding_index,
                               bool is_weights,
                               bool fp16_scale_bias,
                               bool use_LOWOHA) {
  try {
    status_t status;

    if (use_LOWOHA) {
      // LOWOHA path - use embedding_direct API
      try {
        // Validate input tensors
        if (!table_tensor.check() || !indices_tensor.check() ||
            !output_tensor.check()) {
          log_error("LOWOHA embedding: Invalid tensor state detected");
          return status_t::failure;
        }

        // Get raw data pointers
        void *table_data = table_tensor.get_raw_handle_unsafe();
        void *indices_data = indices_tensor.get_raw_handle_unsafe();
        float *weights_data = is_weights ? (float *)
                              weights_tensor.get_raw_handle_unsafe() : nullptr;
        void *output_data = output_tensor.get_raw_handle_unsafe();

        if (!table_data || !indices_data || !output_data) {
          log_error("LOWOHA embedding: Null data pointer detected");
          return status_t::failure;
        }

        // Build embag_params_t structure
        embag_params_t params;

        // Set data types
        params.dtypes.table = table_tensor.get_data_type();
        params.dtypes.output = output_tensor.get_data_type();
        params.dtypes.indices = indices_tensor.get_data_type();

        // Embedding uses algo = none (no reduction)
        params.algo = embag_algo_t::none;

        // Set dimensions
        params.num_embeddings = table_tensor.get_size(0);
        params.embedding_dim = table_tensor.get_size(1);
        params.num_indices = indices_tensor.get_size(0);
        params.is_weights = is_weights;
        params.padding_idx = padding_index;
        params.num_threads = 0;  // Use default (omp_get_max_threads)
        params.fp16_scale_bias = fp16_scale_bias;
        params.dst_stride = output_tensor.get_stride()[0];

        log_info("LOWOHA embedding: Calling embedding_direct with "
                 "num_embeddings=", params.num_embeddings,
                 ", embedding_dim=", params.embedding_dim,
                 ", num_indices=", params.num_indices);

        // Call LOWOHA embedding_direct API
        status = embedding_direct(
                   table_data,
                   indices_data,
                   weights_data,
                   output_data,
                   params);

        if (status != status_t::success) {
          log_error("LOWOHA embedding_direct execution failed.");
          return status_t::failure;
        }
      }
      catch (const std::exception &e) {
        log_error("LOWOHA embedding_direct execution failed: ", e.what());
        return status_t::failure;
      }
      catch (...) {
        log_error("LOWOHA embedding_direct execution failed with unknown exception");
        return status_t::failure;
      }
    }
    else {
      // Regular operator API path
      //define embedding context
      embag_context_t embedding_context = embag_context_t()
                                          .set_param("table", table_tensor)
                                          .set_padding_index(padding_index)
                                          .set_is_weights(is_weights);
      if (table_tensor.get_data_type() == data_type_t::s8 ||
          table_tensor.get_data_type() == data_type_t::s4 ||
          table_tensor.get_data_type() == data_type_t::u4) {
        embedding_context.set_fp16_scale_bias(fp16_scale_bias);
        embedding_context.create();
      }
      else {
        embedding_context.create();
      }
      //define embedding operator
      embag_operator_t embedding_operator = embag_operator_t()
                                            .set_name("embedding_bag")
                                            .set_context(embedding_context)
                                            .create();

      if (embedding_operator.is_bad_object()) {
        testlog_error(" operator ", embedding_operator.get_name(),
                      " creation failed.");
        return status_t::failure;
      }

      if (is_weights) {
        // Execute operator
        status = embedding_operator
                 .set_input("indices", indices_tensor)
                 .set_input("weights", weights_tensor)
                 .set_output("output", output_tensor)
                 .execute();
      }
      else {
        status = embedding_operator
                 .set_input("indices", indices_tensor)
                 .set_output("output", output_tensor)
                 .execute();
      }

      if (status != status_t::success) {
        log_info("operator ", embedding_operator.get_name(), " execution failed.");
        return status_t::failure;
      }
    }
  }
  catch (const exception_t &ex) {
    log_verbose(ex.what());
    return status_t::failure;
  }
  return status_t::success;
}

status_t embedding_forced_ref_kernel_test(tensor_t &table_tensor,
    tensor_t &indices_tensor,
    tensor_t &weights_tensor,
    tensor_t &output_tensor,
    int64_t padding_index,
    bool is_weights,
    bool fp16_scale_bias) {
  try {
    status_t status;

    //define embedding context
    embag_context_t embedding_context = embag_context_t()
                                        .set_param("table", table_tensor)
                                        .set_padding_index(padding_index)
                                        .set_is_weights(is_weights);
    if (table_tensor.get_data_type() == data_type_t::s8 ||
        table_tensor.get_data_type() == data_type_t::s4 ||
        table_tensor.get_data_type() == data_type_t::u4) {
      embedding_context.set_fp16_scale_bias(fp16_scale_bias);
      embedding_context.create();
    }
    else {
      embedding_context.create();
    }

    //define embedding operator
    embag_operator_t embedding_operator = embag_operator_t()
                                          .set_name("ref_embedding_bag")
                                          .set_context(embedding_context)
                                          .create();

    if (embedding_operator.is_bad_object()) {
      testlog_error(" operator ", embedding_operator.get_name(),
                    " creation failed.");
      return status_t::failure;
    }

    if (is_weights) {
      // Execute operator
      status = embedding_operator
               .set_input("indices", indices_tensor)
               .set_input("weights", weights_tensor)
               .set_output("output", output_tensor)
               .set_forced_kernel("reference")
               .execute();
    }
    else {
      // Execute operator
      status = embedding_operator
               .set_input("indices", indices_tensor)
               .set_output("output", output_tensor)
               .set_forced_kernel("reference")
               .execute();
    }

    if (status != status_t::success) {
      log_info("operator ", embedding_operator.get_name(), " execution failed.");
      return status_t::failure;
    }
  }
  catch (const exception_t &ex) {
    log_verbose(ex.what());
    return status_t::failure;
  }
  return status_t::success;
}

void compare_tensor_2D(tensor_t &output_tensor, tensor_t &output_tensor_ref,
                       uint64_t m,
                       uint64_t n, const float tol, bool &is_comparison_successful) {
  const float atol = tol;
  const float rtol = tol * 10;
  #pragma omp parallel for collapse(2)
  for (uint64_t i=0; i<m; ++i) {
    for (uint64_t j=0; j<n; ++j) {
      if (is_comparison_successful) {
        float actual_val = output_tensor.at({i,j});
        float ref_val = output_tensor_ref.at({i,j});

        float abs_err = fabs(ref_val - actual_val);

        if (abs_err > (atol + rtol * fabs(ref_val))) {
          log_verbose("actual(",i,",",j,"): ",actual_val," , ref(",i,",",j,"): ",ref_val);
          is_comparison_successful = false;
        }
      }
    }
  }
  return;
}

void compare_tensor_2D_matrix(tensor_t &output_tensor,
                              tensor_t &output_tensor_ref,
                              uint64_t m,
                              uint64_t n,
                              uint64_t k,
                              const float rtol,
                              const float epsilon,
                              bool &is_comparison_successful,
                              bool enable_f32_relaxation,
                              bool is_woq) {
  constexpr int C = 20; // Margin for F32 tolerance
  //ToDo: Add P value according to the postop currently, same value is used for all.
  constexpr int P = 15; // Post-op accumulation margin
  constexpr int scale_factor = 4; // scale factor

#if ENABLE_F32_RELAXATION
  enable_f32_relaxation = true;
#endif


  // Accumulation-based absolute bound
  // abs_bound = (C*k+P)*epsilon
  const float abs_bound =
    (output_tensor.get_data_type() == data_type_t::bf16) || is_woq
    ? (k * epsilon)
    : (((C + log2(k) / scale_factor) * k + P) * epsilon);

  // F32 zero-reference handling tolerances (controlled by bool flag) for libxsmm backends
  constexpr float ABS_ZERO_TOL_F32 = 8e-4f;
  constexpr float ZERO_REF_THRESH = 1e-6f;
  constexpr float F32_EPS_SLACK = 2e-4f;

  const bool is_f32 = output_tensor.get_data_type() == data_type_t::f32;

  log_verbose("abs_bound: ", abs_bound);

  #pragma omp parallel for collapse(2)
  for (uint64_t i = 0; i < m; ++i) {
    for (uint64_t j = 0; j < n; ++j) {
      if (is_comparison_successful) {
        float actual_val = output_tensor.at({i, j});
        float ref_val    = output_tensor_ref.at({i, j});
        float abs_err    = fabs(ref_val - actual_val);

        float allowed_err;
        if (enable_f32_relaxation && is_f32) {
          if (fabs(ref_val) < ZERO_REF_THRESH) {
            // Zero-reference F32 path
            allowed_err = std::max(abs_bound, ABS_ZERO_TOL_F32) + F32_EPS_SLACK;
          }
          else {
            // Normal F32 path with small slack
            allowed_err = abs_bound + rtol * fabs(ref_val) + F32_EPS_SLACK;
          }
        }
        else {
          // Default path
          allowed_err = abs_bound + rtol * fabs(ref_val);
        }

        if (abs_err > allowed_err) {
          log_verbose("actual(", i, ",", j, "): ", actual_val,
                      " , ref(", i, ",", j, "): ", ref_val);
          log_verbose("abs_error: ", abs_err,
                      " , allowed_err: ", allowed_err,
                      " , abs_bound: ", abs_bound);
          is_comparison_successful = false;
        }
      }
    }
  }
}
void compare_tensor_3D_matrix(tensor_t &output_tensor,
                              tensor_t &output_tensor_ref,
                              uint64_t batch_size,
                              uint64_t m,
                              uint64_t n,
                              uint64_t k,
                              const float rtol,
                              const float epsilon,
                              bool &is_comparison_successful,
                              bool enable_f32_relaxation) {
  constexpr int C = 20; // Margin for F32 tolerance
  //ToDo: Add P value according to the postop currently, same value is used for all.
  constexpr int P = 15; // Post-op accumulation margin
  constexpr int scale_factor = 4; // scale factor

#if ENABLE_F32_RELAXATION
  enable_f32_relaxation = true;
#endif


  // Accumulation-based absolute bound
  //float abs_bound = ((20 + log2(k)/4) * k + 15) * epsilon;
  //(C*K+P)*epsilon
  const float abs_bound =
    (output_tensor.get_data_type() == data_type_t::bf16)
    ? (k * epsilon)
    : (((C + log2(k) / scale_factor) * k + P) * epsilon);

  // F32 zero-reference handling tolerances (controlled by bool flag) for libxsmm backends
  constexpr float ABS_ZERO_TOL_F32 = 8e-4f;
  constexpr float ZERO_REF_THRESH = 1e-6f;
  constexpr float F32_EPS_SLACK = 2e-4f;

  const bool is_f32 = output_tensor.get_data_type() == data_type_t::f32;

  log_verbose("abs_bound: ", abs_bound);

  #pragma omp parallel for collapse(3)
  for (uint64_t bs = 0; bs < batch_size; ++bs) {
    for (uint64_t i = 0; i < m; ++i) {
      for (uint64_t j = 0; j < n; ++j) {
        if (is_comparison_successful) {
          float actual_val = output_tensor.at({bs, i, j});
          float ref_val    = output_tensor_ref.at({bs, i, j});
          float abs_err    = fabs(ref_val - actual_val);

          float allowed_err;
          if (enable_f32_relaxation && is_f32) {
            if (fabs(ref_val) < ZERO_REF_THRESH) {
              // Zero-reference F32 path
              allowed_err = std::max(abs_bound, ABS_ZERO_TOL_F32) + F32_EPS_SLACK;
            }
            else {
              // Normal F32 path with small slack
              allowed_err = abs_bound + rtol * fabs(ref_val) + F32_EPS_SLACK;
            }
          }
          else {
            // Default path
            allowed_err = abs_bound + rtol * fabs(ref_val);
          }

          if (abs_err > allowed_err) {
            log_verbose("actual(", bs, ",", i, ",", j, "): ", actual_val,
                        " , ref(", bs, ",", i, ",", j, "): ", ref_val);
            log_verbose("abs_error: ", abs_err,
                        " , allowed_err: ", allowed_err,
                        " , abs_bound: ", abs_bound);
            is_comparison_successful = false;
          }
        }
      }
    }
  }
}
size_t get_aligned_size(size_t alignment, size_t size_) {
  return ((size_ + alignment - 1) & ~(alignment - 1));
}
