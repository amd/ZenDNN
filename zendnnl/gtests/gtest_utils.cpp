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

#include "gtest_utils.hpp"

MatmulType::MatmulType() {
  matmul_m   = MATMUL_SIZE_START + rand() % MATMUL_SIZE_END;
  matmul_k   = MATMUL_SIZE_START + rand() % MATMUL_SIZE_END;
  matmul_n   = MATMUL_SIZE_START + rand() % MATMUL_SIZE_END;
  transA     = rand() % 2;
  transB     = rand() % 2;
  po_index   = rand() % (po_size + 1);

  // Use std::random_device and std::mt19937 for random float generation
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(0.0, 10.0);
  alpha    = dist(gen);
  beta     = dist(gen);
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
  scatter_stride = -1;
}

BatchMatmulType::BatchMatmulType() {
  batch_size = BATCH_START + rand() % BATCH_END;
}

ReorderType::ReorderType() {
  inplace_reorder = rand() % 2;
  source_dtype    = rand() % 2 == 0 ? data_type_t::s8 : data_type_t::u8;
}

bool is_binary_postop(const std::string post_op) {
  return post_op == "binary_add" || post_op == "binary_mul";
}

tensor_t tensor_factory_t::zero_tensor(const std::vector<index_type> size_,
                                       data_type dtype_) {

  auto ztensor = tensor_t()
                 .set_name("zero tensor")
                 .set_size(size_)
                 .set_data_type(dtype_)
                 .set_storage()
                 .create();

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
    size_,
    data_type dtype_, float val,
    bool trans) {
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

  udtensor.set_storage().create();

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
      int8_t *buf_ptr = static_cast<int8_t *>(buf_vptr);
      std::generate(buf_ptr, buf_ptr+buf_nelem, [&] {return int8_t(dist2(gen));});
    }
    else {
      log_warning("tensor ", udtensor.get_name(), " unsupported data type.");
    }
  }
  return udtensor;
}

tensor_t tensor_factory_t::uniform_tensor(const std::vector<index_type> size_,
    data_type dtype_, float val_,
    std::string tensor_name_) {

  auto utensor = tensor_t()
                 .set_name(tensor_name_)
                 .set_size(size_)
                 .set_data_type(dtype_)
                 .set_storage()
                 .create();

  if (! utensor.check()) {
    log_warning("tensor creation of ", utensor.get_name(), " failed.");
  }
  else {
    auto  buf_nelem  = utensor.get_nelem();
    void *buf_vptr   = utensor.get_raw_handle_unsafe();

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
    else {
      log_warning("tensor ", utensor.get_name(), " unsupported data type.");
    }
  }
  return utensor;
}

tensor_t tensor_factory_t::uniform_dist_strided_tensor(const
    std::vector<index_type> size_, const std::vector<index_type> aligned_size_,
    data_type dtype_, float range_, bool trans) {
  auto udstensor = tensor_t()
                   .set_name("uniform distributed strided tensor")
                   .set_size(size_)
                   .set_data_type(dtype_)
                   .set_aligned_size(aligned_size_)
                   .set_storage()
                   .create();

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
      std::generate(buf_ptr, buf_ptr + buf_nelem, [&] {return int8_t(dist(gen));});
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
    size_,
    uint64_t num_embeddings) {
  auto indices_tensor = tensor_t()
                        .set_name("indices_tensor")
                        .set_size(size_)
                        .set_data_type(data_type_t::s32)
                        .set_storage()
                        .create();
  void *data = indices_tensor.get_raw_handle_unsafe();

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint32_t> dist(0,
      static_cast<uint32_t>(num_embeddings - 1));

  int num_indices = size_[0];

  for (int i = 0; i < num_indices; ++i) {
    static_cast<uint32_t *>(data)[i] = dist(gen);
  }

  return indices_tensor;
}

tensor_t tensor_factory_t::random_offsets_tensor(const std::vector<index_type>
    size_,
    uint64_t num_indices,
    bool include_last_offset) {
  auto tensor = tensor_t()
                .set_name("offsets_tensor")
                .set_size(size_)
                .set_data_type(data_type_t::s32)
                .set_storage()
                .create();
  void *data = tensor.get_raw_handle_unsafe();

  int num_offsets = size_[0];
  if (include_last_offset) {
    num_offsets--;
  }

  for (int i = 0; i < num_offsets; ++i) {
    static_cast<uint32_t *>(data)[i] = (i * num_indices) / num_offsets;
  }

  if (include_last_offset) {
    static_cast<uint32_t *>(data)[num_offsets] = num_indices;
  }

  return tensor;
}

void Parser::operator()(const int &argc, char *argv[], int &seed,
                        uint32_t &tests, std::string &po) {
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
  return;
}

void Parser::read_from_umap(const std::string &key, int &num) {
  if (umap.count(key)) {
    std::string val = umap.at(key);
    if (isInteger(val)) {
      try {
        num = stoi(val);
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
             ", so using the random postop from supported list.");
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

status_t matmul_kernel_test(tensor_t &input_tensor, tensor_t &weight_tensor,
                            tensor_t &bias, tensor_t &output_tensor,
                            uint32_t index, tensor_t &binary_tensor, float alpha, float beta) {
  try {

    if (LOWOHA && index == po_size) {
      try {
        // Validate input tensors
        if (!input_tensor.check() || !weight_tensor.check() || !output_tensor.check()) {
          log_error("LOWOHA: Invalid tensor state detected");
          return status_t::failure;
        }
        auto input_dim              = input_tensor.get_dim();
        auto weight_dim             = weight_tensor.get_dim();
        auto output_dim             = output_tensor.get_dim();
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

        const int M                 = output_tensor.get_size(output_dim-2);
        const int K                 = input_tensor.get_size(input_dim-1);
        const int N                 = output_tensor.get_size(output_dim-1);
        // Validate dimensions
        if (M == 0 || K == 0 || N == 0) {
          log_error("LOWOHA: Invalid tensor dimensions - M:", M, " K:", K, " N:", N);
          return status_t::failure;
        }

        // Get tensor data pointers (no bias needed)
        void *A_data = input_tensor.get_raw_handle_unsafe();
        void *B_data = weight_tensor.get_raw_handle_unsafe();
        void *C_data = output_tensor.get_raw_handle_unsafe();

        // Validate data pointers (no bias validation needed)
        if (!A_data || !B_data || !C_data) {
          log_error("LOWOHA: Null data pointer detected");
          return status_t::failure;
        }

        // Get data types
        data_type_t src_data_type = input_tensor.get_data_type();
        data_type_t wei_data_type = weight_tensor.get_data_type();
        data_type_t out_data_type = output_tensor.get_data_type();
        data_type_t bias_data_type = bias.get_data_type();
        data_types matmul_dtypes;
        matmul_dtypes.src = src_data_type;
        matmul_dtypes.wei = wei_data_type;
        matmul_dtypes.dst = out_data_type;
        matmul_dtypes.bias = bias_data_type;
        matmul_dtypes.compute = data_type_t::none;

        // Validate data types
        if (src_data_type != data_type_t::f32 && src_data_type != data_type_t::bf16) {
          log_error("LOWOHA: Unsupported source data type");
          return status_t::failure;
        }
        if (out_data_type != data_type_t::f32 && out_data_type != data_type_t::bf16) {
          log_error("LOWOHA: Unsupported output data type");
          return status_t::failure;
        }

        log_info("LOWOHA: Calling matmul_direct with batchA:", batchA, " batchB:",
                 batchB, " M:", M, " N:", N, " K:", K,
                 " alpha:", alpha, " beta:", beta);

        lowoha_post_op postop;

        status_t status = matmul_direct(
                            A_data, B_data, C_data, nullptr,  // No bias
                            alpha, beta,
                            static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                            transA, transB,
                            lda, ldb, ldc,
                            matmul_dtypes, postop,
                            lowoha_quantization_params_t(),
                            batchA, batchB  // Batch_A, Batch_B
                          );
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
      post_op_t post_op = post_op_t{po_arr[0].second};
      // postop update according to the index
      if (index != po_size && index != 0) post_op = post_op_t{po_arr[index].second};
      weight_tensor.set_name("weights");
      bias.set_name("bias");

      //define matmul context
      auto matmul_context = matmul_context_t()
                            .set_param("weights", weight_tensor)
                            .set_param("bias", bias)
                            .set_alpha(alpha)
                            .set_beta(beta);
      if (index != po_size) {
        matmul_context = matmul_context.set_post_op(post_op).create();
      }
      else {
        matmul_context = matmul_context.create();//No Postop case
      }

      //define matmul operator
      auto matmul_operator = matmul_operator_t()
                             .set_name("matmul_operator")
                             .set_context(matmul_context)
                             .create();

      if (! matmul_operator.check()) {
        log_error("operator ", matmul_operator.get_name(), " creation failed.");
        return status_t::failure;
      }

      input_tensor.set_name("matmul_input");
      output_tensor.set_name("matmul_output");
      // Set binary tensor for binary postops
      if (index < po_size) {
        if (po_arr[index].second == post_op_type_t::binary_add) {
          matmul_operator.set_input(post_op.binary_add_params.tensor_name, binary_tensor);
        }
        else if (po_arr[index].second == post_op_type_t::binary_mul) {
          matmul_operator.set_input(post_op.binary_mul_params.tensor_name, binary_tensor);
        }
      }
      status_t status = matmul_operator
                        .set_input("matmul_input", input_tensor)
                        .set_output("matmul_output", output_tensor)
                        .execute();

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
                                       tensor_t &weights,
                                       tensor_t &bias, tensor_t &output_tensor,
                                       uint32_t index, tensor_t &binary_tensor, float alpha, float beta) {
  try {
    // Default postop relu
    post_op_t post_op = post_op_t{po_arr[0].second};
    // postop update according to the index
    if (index != po_size && index != 0) post_op = post_op_t{po_arr[index].second};
    weights.set_name("weights");
    bias.set_name("bias");

    //define matmul context
    auto matmul_context = matmul_context_t()
                          .set_param("weights", weights)
                          .set_param("bias", bias)
                          .set_alpha(alpha)
                          .set_beta(beta);
    if (index != po_size) {
      matmul_context = matmul_context.set_post_op(post_op).create();
    }
    else {
      matmul_context = matmul_context.create(); //No postop case
    }

    //define matmul operator
    auto matmul_operator = matmul_operator_t()
                           .set_name("matmul_forced_ref_operator")
                           .set_context(matmul_context)
                           .create();

    if (! matmul_operator.check()) {
      log_error("operator ", matmul_operator.get_name(), " creation failed.");
      return status_t::failure;
    }
    input_tensor.set_name("matmul_input");
    output_tensor.set_name("matmul_output");

    if (index < po_size) {
      if (po_arr[index].second == post_op_type_t::binary_add) {
        matmul_operator.set_input(post_op.binary_add_params.tensor_name, binary_tensor);
      }
      else if (po_arr[index].second == post_op_type_t::binary_mul) {
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
    auto reorder_context = reorder_context_t()
                           .set_algo_format("aocl");

    // Set Input source dtype if the weights dtype is s8
    if ((dtype == data_type_t::s8) && (source_dtype == data_type_t::s8 ||
                                       source_dtype == data_type_t::u8)) {
      reorder_context.set_source_dtype(source_dtype);
    }
    reorder_context.create();

    uint64_t rows               = input_tensor.get_size(0);
    uint64_t cols               = input_tensor.get_size(1);
    tensor_t output_tensor;

    bool memory_reorder         = (!(input_tensor.get_layout() | uint8_t(
                                       tensor_layout_t::contiguous)) ||
                                   (input_tensor.get_layout() & uint8_t(tensor_layout_t::aligned)));
    bool memory_unreorder       = (input_tensor.get_layout() & uint8_t(
                                     tensor_layout_t::blocked));
    bool trans                  = (input_tensor.get_order() == "ba") ? true : false;

    if (memory_reorder) {
      // Reorder operator creation with name, context and input.
      auto reorder_operator = reorder_operator_t()
                              .set_name("reorder_operator")
                              .set_context(reorder_context)
                              .create()
                              .set_input("reorder_input", input_tensor);

      if (! reorder_operator.check()) {
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
          return std::make_pair(input_tensor, status_t::unimplemented);
        }
        else {
          // Assign input_tensor to buffer_params as a tensor_t variant
          StorageParam buffer_params = input_tensor;

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
          return std::make_pair(input_tensor, status_t::failure);
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
        return std::make_pair(input_tensor, status_t::failure);
      }
      else {
        log_info("operator ", reorder_operator.get_name(), " execution successful.");
        return std::make_pair(output_tensor, status_t::success);
      }
    }
    else {
      // reorder operator creation with name, context and input.
      auto reorder_operator = reorder_operator_t()
                              .set_name("reorder_operator")
                              .set_context(reorder_context)
                              .create()
                              .set_input("reorder_input", input_tensor);

      if (! reorder_operator.check()) {
        log_error("operator ", reorder_operator.get_name(), " creation failed.");
        return std::make_pair(tensor_t(), status_t::failure);
      }

      // Inplace reorder
      if (inplace_reorder) {
        StorageParam buffer_params = input_tensor;

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
          return std::make_pair(input_tensor, status_t::failure);
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
        return std::make_pair(input_tensor, status_t::failure);
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
                           int64_t scatter_stride) {
  try {
    status_t status;

    //define embag context
    auto embedding_bag_context = embag_context_t()
                                 .set_param("table", table_tensor)
                                 .set_algo(algo)
                                 .set_padding_index(padding_index)
                                 .set_include_last_offset(include_last_offset)
                                 .set_is_weights(is_weights)
                                 .set_scatter_stride(scatter_stride)
                                 .create();

    //define embedding bag operator
    auto embedding_bag_operator = embag_operator_t()
                                  .set_name("embedding_bag")
                                  .set_context(embedding_bag_context)
                                  .create();

    if (! embedding_bag_operator.check()) {
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
                                      int64_t scatter_stride) {
  try {
    status_t status;

    //define embag context
    auto embedding_bag_context = embag_context_t()
                                 .set_param("table", table_tensor)
                                 .set_algo(algo)
                                 .set_padding_index(padding_index)
                                 .set_include_last_offset(include_last_offset)
                                 .set_is_weights(is_weights)
                                 .set_scatter_stride(scatter_stride)
                                 .create();

    //define embedding bag operator
    auto embedding_bag_operator = embag_operator_t()
                                  .set_name("ref_embedding_bag")
                                  .set_context(embedding_bag_context)
                                  .create();

    if (! embedding_bag_operator.check()) {
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
                              tensor_t &output_tensor_ref, uint64_t m,
                              uint64_t n, uint64_t k, const float rtol,
                              const float epsilon, bool &is_comparison_successful) {
  constexpr int C = 20; //Margin for F32:: tolerance
  //ToDo: Add P value according to the postop currently, same value is used for all.
  constexpr int P = 15; //to handle postop accumulation error
  constexpr int scale_factor = 4; // scale factor
  // abs_bound = (C*k+P)*epsilon
  const float abs_bound = (output_tensor.get_data_type() == data_type_t::bf16)
                          ? (k * epsilon)
                          : (((C + log2(k)/scale_factor) * k + P) * epsilon);
  log_verbose("abs_bound: ", abs_bound);
  #pragma omp parallel for collapse(2)
  for (uint64_t i=0; i<m; ++i) {
    for (uint64_t j=0; j<n; ++j) {
      if (is_comparison_successful) {
        float actual_val = output_tensor.at({i,j});
        float ref_val = output_tensor_ref.at({i,j});
        float abs_err = fabs(ref_val - actual_val);
        if (abs_err > abs_bound + rtol * fabs(ref_val)) {
          log_verbose("actual(",i,",",j,"): ",actual_val," , ref(",i,",",j,"): ",ref_val);
          log_verbose("abs_error: ", abs_err, " ,rtol* fabs(ref): ",
                      rtol * fabs(ref_val)) ;
          is_comparison_successful = false;
        }
      }
    }
  }
  return;
}

void compare_tensor_3D_matrix(tensor_t &output_tensor,
                              tensor_t &output_tensor_ref, uint64_t batch_size,
                              uint64_t m, uint64_t n, uint64_t k, const float rtol,
                              const float epsilon, bool &is_comparison_successful) {
  constexpr int C = 20; //Margin for F32:: tolerance
  //ToDo: Add P value according to the postop currently, same value is used for all.
  constexpr int P = 15; //to handle postop accumulation error
  constexpr int scale_factor = 4; // scale factor
  //float abs_bound = ((20 + log2(k)/4) * k + 15) * epsilon; //(C*K+P)*epsilon
  const float abs_bound = (output_tensor.get_data_type() == data_type_t::bf16)
                          ? (k * epsilon)
                          : (((C + log2(k)/scale_factor) * k + P) * epsilon);
  log_verbose("abs_bound: ", abs_bound);
  #pragma omp parallel for collapse(3)
  for (uint64_t bs=0; bs<batch_size; ++bs) {
    for (uint64_t i=0; i<m; ++i) {
      for (uint64_t j=0; j<n; ++j) {
        if (is_comparison_successful) {
          float actual_val = output_tensor.at({bs,i,j});
          float ref_val = output_tensor_ref.at({bs,i,j});
          float abs_err = fabs(ref_val - actual_val);
          if (abs_err > abs_bound + rtol * fabs(ref_val)) {
            log_verbose("actual(",bs,",",i,",",j,"): ",actual_val," , ref(",bs,",",i,",",j,
                        "): ",ref_val);
            log_verbose("abs_error: ", abs_err, " ,rtol* fabs(ref): ",
                        rtol * fabs(ref_val)) ;
            is_comparison_successful = false;
          }
        }
      }
    }
  }
  return;
}

size_t get_aligned_size(size_t alignment, size_t size_) {
  return ((size_ + alignment - 1) & ~(alignment - 1));
}
