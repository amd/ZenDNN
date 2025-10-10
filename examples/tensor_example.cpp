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

#include <memory>
#include <iostream>
#include <optional>
#include <cstdint>

#include "tensor_example.hpp"

namespace zendnnl {
namespace examples {

using namespace zendnnl::interface;

int tensor_unaligned_allocation_example() {
  testlog_info("Tensor unaligned memory allocation example");
  auto tensor =  tensor_t()
                 .set_name("unaligned_tensor")
                 .set_size({MATMUL_ROWS, MATMUL_COLS})
                 .set_storage()
                 .create();

  if (tensor.check()) {
    testlog_info("Tensor creation of ", tensor.get_name(), " successful.");
    testlog_verbose(tensor.get_name(),
                " elements :", tensor.get_nelem(),
                " buffer size :", tensor.get_buffer_sz_bytes(),
                " raw ptr :", reinterpret_cast<std::uintptr_t>(tensor.get_raw_handle_unsafe()));
  }
  else {
    testlog_error("Tensor creation of ", tensor.get_name(), " failed!");
    return NOT_OK;
  }

  return OK;
}

int tensor_aligned_allocation_example() {
  testlog_info("Tensor aligned memory allocation example");
  auto tensor =  tensor_t()
                 .set_name("aligned_tensor")
                 .set_size({MATMUL_ROWS, MATMUL_COLS})
                 .set_storage(ALIGNMENT_BOUNDARY)
                 .create();

  if (tensor.check()) {
    testlog_info("Tensor creation of ", tensor.get_name(), " successful.");
    testlog_verbose(tensor.get_name(),
                " elements :", tensor.get_nelem(),
                " buffer size :", tensor.get_buffer_sz_bytes(),
                " raw ptr :", reinterpret_cast<std::uintptr_t>(tensor.get_raw_handle_unsafe()));
  }
  else {
    testlog_error("Tensor creation of ", tensor.get_name(), " failed!");
    return NOT_OK;
  }

  return OK;
}

int tensor_strided_aligned_allocation_example() {
  testlog_info("Tensor strided aligned memory allocation example");
  auto tensor =  tensor_t()
                 .set_name("aligned_strided_tensor")
                 .set_aligned_size({MATMUL_ROWS, MATMUL_STRIDE_COLS})
                 .set_size({MATMUL_ROWS, MATMUL_COLS})
                 .set_storage(ALIGNMENT_BOUNDARY)
                 .create();

  if (tensor.check()) {
    testlog_info("Tensor creation of ", tensor.get_name(), " successful.");
    testlog_verbose(tensor.get_name(),
                " elements :", tensor.get_nelem(),
                " buffer size :", tensor.get_buffer_sz_bytes(),
                " raw ptr :", reinterpret_cast<std::uintptr_t>(tensor.get_raw_handle_unsafe()));
  }
  else {
    testlog_error("Tensor creation of ", tensor.get_name(), " failed!");
    return NOT_OK;
  }

  return OK;
}

int tensor_copy_and_compare_example() {
  testlog_info("Tensor copy and compare example");
  auto tensor =  tensor_t()
                 .set_name("contiguous_tensor")
                 .set_size({MATMUL_ROWS, MATMUL_COLS})
                 .set_data_type(data_type_t::bf16)
                 .set_storage()
                 .create();

  if (tensor.check()) {
    testlog_verbose("Tensor creation of ", tensor.get_name(), " successful");
    testlog_verbose(tensor.get_name(),
                " hash :", tensor.hash(),
                " elements :", tensor.get_nelem(),
                " buffer size :", tensor.get_buffer_sz_bytes(),
                " raw ptr :", reinterpret_cast<std::uintptr_t>(tensor.get_raw_handle_unsafe()));
  }
  else {
    testlog_error("Tensor creation of ", tensor.get_name(), " failed!");
    return NOT_OK;
  }

  auto copy_tensor = tensor;
  copy_tensor.set_name("copied_bf16_tensor");
  testlog_info("copied ", tensor.get_name(), " to ", copy_tensor.get_name());

  testlog_info("comparing ", tensor.get_name(), " with ", copy_tensor.get_name());
  if (copy_tensor == tensor) {
    testlog_info("tensor copy of ", copy_tensor.get_name(),
             " from ", tensor.get_name(), " is successful");
    testlog_verbose(tensor.get_name(), " storage count : ", tensor.get_storage_count());
  }
  else {
    testlog_error("tensor copy failed with original hash:", tensor.hash(),
              " copied hash:", copy_tensor.hash(), " mismatch.");

    return NOT_OK;
  }

  return OK;
}

int tensor_move_and_refcount_example() {
  testlog_info("Tensor move and refcount example.");
  auto tensor =  tensor_t()
                 .set_name("contiguous_tensor")
                 .set_size({MATMUL_ROWS, MATMUL_COLS})
                 .set_data_type(data_type_t::s8)
                 .set_storage()
                 .create();

  if (tensor.check()) {
    testlog_verbose("Tensor creation of ", tensor.get_name(), " successful");
    testlog_verbose(tensor.get_name(),
                " hash :", tensor.hash(),
                " elements :", tensor.get_nelem(),
                " buffer size :", tensor.get_buffer_sz_bytes(),
                " raw ptr :", reinterpret_cast<std::uintptr_t>(tensor.get_raw_handle_unsafe()));
  }
  else {
    testlog_error("Tensor creation of ", tensor.get_name(), " failed!");
    return NOT_OK;
  }

  // Store the original name before moving
  std::string original_name = tensor.get_name();

  auto move_tensor = std::move(tensor);
  move_tensor.set_name("move_s8_tensor");
  testlog_info("moved ", original_name, " to ", move_tensor.get_name());

  // After move, tensor is in a valid but unspecified state
  // We can only check if move_tensor is valid, not compare with moved tensor
  if (move_tensor.check()) {
    testlog_info("tensor move of ", original_name,
             " to ", move_tensor.get_name(), " is successful");
    testlog_info(move_tensor.get_name(),
             " hash :", move_tensor.hash(),
             " storage count : ", move_tensor.get_storage_count());
  }
  else {
    testlog_error("move tensor failed.");
    return NOT_OK;
  }

  return OK;
}

int tensor_constness_example() {
  testlog_info("Tensor constness example");

  //use tensor factor to create a uniform tensor
  tensor_factory_t tensor_factory;
  auto utensor = tensor_factory.uniform_tensor({MATMUL_DEPTH, MATMUL_ROWS, MATMUL_COLS},
                 data_type_t::f32,
                 2.5, "tensor");

  //make the tensor const
  utensor.set_const(true);

  //try to grab its raw pointer
  try {
    void *ptr = utensor.get_raw_handle_unsafe();
    testlog_info("raw pointer of const tensor", ptr);
  }
  catch (const exception_t &ex) {
    testlog_info("caught exception of attempt to get raw pointer of a const tensor.");
    testlog_verbose(ex.what());
  }

  //try to get const handle
  try {
    const float *const_ptr = static_cast<const float *>
                             (utensor.get_raw_handle_const());

    //try to modify data
    //const_ptr[2] = 3.0;

    //read the data
    testlog_info("tensor ", utensor.get_name(), "flat index = 2 : value = ",
             const_ptr[2]);

    //at() works
    testlog_info(utensor.get_name(), "[2,2,2] = ", utensor.at({2,2,2}));
  }
  catch (const exception_t &ex) {
    testlog_error(ex.what());
    return NOT_OK;
  }

  return OK;
}

int tensor_create_alike_example() {
  testlog_info("Tensor create alike example");

  //use tensor factory to create a uniform tensor
  tensor_factory_t tensor_factory;
  auto utensor = tensor_factory.uniform_tensor({MATMUL_DEPTH, MATMUL_ROWS, MATMUL_COLS},
                 data_type_t::f32,
                 2.5, "tensor");

  //make the tensor const
  auto tensor_option = utensor.get_tensor_option();

  try {
    tensor_t atensor = tensor_t()
                       .set_tensor_option(tensor_option)
                       .set_storage()
                       .create();

    //check few options
    if ((utensor.get_size() == atensor.get_size()) &&
        (utensor.get_data_type() == atensor.get_data_type()) &&
        (utensor.get_layout() == atensor.get_layout())) {
      testlog_info("Created a tensor with same options");
    }
    else {
      testlog_error("Failed to create tensor with same options");
      return NOT_OK;
    }

  }
  catch (const exception_t &ex) {
    testlog_error(ex.what());
    return NOT_OK;
  }

  return OK;
}

int tensor_broadcast_example() {
  testlog_info("Tensor broadcast example");

  {
    //create a 3D tensor broadcasted along depth
    auto depth_tensor = tensor_t()
      .set_name("depth_broadcast_tensor")
      .set_size({10,4,6})
      .set_stride({0,6,1})
      .set_data_type(data_type_t::f32)
      .set_storage()
      .create();

    if (! depth_tensor.check()) {
      testlog_error("Creation of ", depth_tensor.get_name(), " failed.");
      return NOT_OK;
    }
    else {
      testlog_info("Created detphwise broadcast tensor of size {10,6,4}, stride {0,6,1}");
    }

    auto      nelem   = depth_tensor.get_nelem();
    float*    buf_ptr = static_cast<float*>(depth_tensor.get_raw_handle_unsafe());
    for (uint32_t i = 0; i < nelem; ++i)
      buf_ptr[i] = i+1;

    testlog_info(depth_tensor.get_name()," has ", nelem, " elements.");

    //print same row and col element at differet depths
    testlog_info("Printing elements with same row, col but only depth changing...");
    uint32_t r = 2; uint32_t c = 3;
    for (uint32_t d = 0; d < 5; ++d) {
      auto val = depth_tensor.at({d,r,c});
      testlog_info(depth_tensor.get_name(), "[", d, ",", r, ",", c,"] = ", val);
    }
  }

  {
    //create a 3D tensor broadcasted along row
    auto row_tensor = tensor_t()
      .set_name("row_broadcast_tensor")
      .set_size({10,4,6})
      .set_stride({6,0,1})
      .set_data_type(data_type_t::f32)
      .set_storage()
      .create();

    if (! row_tensor.check()) {
      testlog_error("Creation of ", row_tensor.get_name(), " failed.");
      return NOT_OK;
    }
    else {
      testlog_info("Created row-wise broadcast tensor of size {10,4,6}, stride {6,0,1}");
    }

    auto      nelem   = row_tensor.get_nelem();
    float*    buf_ptr = static_cast<float*>(row_tensor.get_raw_handle_unsafe());
    for (uint32_t i = 0; i < nelem; ++i)
      buf_ptr[i] = i+1;

    testlog_info(row_tensor.get_name()," has ", nelem, " elements.");

    //print same row and col element at differet depths
    testlog_info("Printing elements with same depth, col but only row changing...");
    uint32_t d = 4; uint32_t c = 3;
    for (uint32_t r = 0; r < 4; ++r) {
      auto val = row_tensor.at({d,r,c});
      testlog_info(row_tensor.get_name(), "[", d, ",", r, ",", c,"] = ", val);
    }
  }

  {
    //create a 4D tensor with 'ac' subtensor broadcasted along 'bd' axes
    auto bd_tensor = tensor_t()
      .set_name("bd_broadcast_tensor")
      .set_size({10,5,4,6})
      .set_stride({4,0,1,0})
      .set_data_type(data_type_t::f32)
      .set_storage()
      .create();

    if (! bd_tensor.check()) {
      testlog_error("Creation of ", bd_tensor.get_name(), " failed.");
      return NOT_OK;
    }
    else {
      testlog_info("Created bd-wise broadcast tensor of size {10,5,4,6}, stride {4,0,1,0}");
    }

    auto      nelem   = bd_tensor.get_nelem();
    float*    buf_ptr = static_cast<float*>(bd_tensor.get_raw_handle_unsafe());
    for (uint32_t i = 0; i < nelem; ++i)
      buf_ptr[i] = i+1;

    testlog_info(bd_tensor.get_name()," has ", nelem, " elements.");

    //print same row and col element at differet depths
    testlog_info("Printing elements with same a,c, only b,d changing...");
    uint32_t a = 2; uint32_t c = 1;
    for (uint32_t b = 0; b < 5; ++b) {
      for (uint32_t d = 0; d < 6; ++d) {
        auto val = bd_tensor.at({a,b,c,d});
        testlog_info(bd_tensor.get_name(), "[", a, ",", b, ",", c, ',', d, "] = ", val);
      }
    }
  }

  return OK;
}

int tensor_axes_permutation_example() {

  //create and linearly populate a 4D tensor. stride [120,24,6,1]
  auto orig_tensor = tensor_t()
    .set_name("orig_tensor")
    .set_size({10,5,4,6})
    .set_data_type(data_type_t::f32)
    .set_storage()
    .create();

  if (! orig_tensor.check()) {
    testlog_error("Creation of ", orig_tensor.get_name(), " failed.");
    return NOT_OK;
  }

  auto      nelem   = orig_tensor.get_nelem();
  float*    buf_ptr = static_cast<float*>(orig_tensor.get_raw_handle_unsafe());
  for (uint32_t i = 0; i < nelem; ++i)
    buf_ptr[i] = i+1;

  //access an element
  tensor_t::index_vec_type index = {2,3,1,4};
  testlog_info(orig_tensor.get_name(), " [2,3,1,4] = ", orig_tensor.at(index));

  //create another tensor with permuted axes, but same buffer
  auto permuted_tensor = tensor_t()
    .set_name("permuted_tensor")
    .set_size({10,5,6,4})
    .set_order("abdc")
    .set_data_type(data_type_t::f32)
    .set_storage(orig_tensor)
    .create();

  //access same element with permuted index
  index = {2,3,4,1};
  testlog_info(permuted_tensor.get_name(), " [2,3,4,1] = ", permuted_tensor.at(index));

  //give strides in place of order
  auto stride_tensor = tensor_t()
    .set_name("stride_tensor")
    .set_size({10,5,6,4})
    .set_stride({120,24,1,6})
    .set_data_type(data_type_t::f32)
    .set_storage(orig_tensor)
    .create();

  //access same element with permuted index
  index = {2,3,4,1};
  testlog_info(stride_tensor.get_name(), " [2,3,4,1] = ", stride_tensor.at(index));

  return OK;
}

int tensor_quantization_example() {
  testlog_info("Quantizated tensor creation example");

  //use tensor factory to create a uniform tensor
  tensor_factory_t tensor_factory;

  //get a uniformly distributed tensor
  auto udtensor = tensor_factory.uniform_dist_strided_tensor({MATMUL_ROWS, MATMUL_COLS},
                                                             {MATMUL_ROWS, MATMUL_COLS},
                                                             data_type_t::f32,
                                                             1.0,
                                                             "udtensor");

  //get a scale tensor for row-wise channel quantization
  float scale  = 1.0/127.0;

  auto scales  = tensor_factory.uniform_tensor({MATMUL_ROWS, 1},
                                               data_type_t::f32,
                                               scale, "scale tensor");

  auto qtensor = tensor_t()
    .set_name("quantized tensor")
    .set_size({MATMUL_ROWS, MATMUL_COLS})
    .set_data_type(data_type_t::s8)
    .set_quant_scale(scales)
    .set_storage()
    .create();

  if (! qtensor.check() ) {
    testlog_error("tensor creation of ", qtensor.get_name(), " failed");
    return NOT_OK;
  }

  //quantize the tensor
  const float*  udhandle = (const float *)udtensor.get_raw_handle_const();
  int8_t*       qhandle  = (int8_t *)qtensor.get_raw_handle_unsafe();
  const float*  shandle  = (const float *)qtensor.get_quant_scale_raw_handle_const();

  for (uint32_t r = 0; r < MATMUL_ROWS; ++r) {
    float scale = shandle[r];

    for (uint32_t c = 0; c < MATMUL_COLS; ++c) {
      auto udoffset = udtensor.compute_offset({r,c});
      auto qoffset  = qtensor.compute_offset({r,c});

      qhandle[qoffset] = int8_t(udhandle[udoffset]/scale);
    }
  }

  //query quantization parameters
  auto quant_type = qtensor.get_quant_type();
  if (quant_type == quant_type_t::uniform)
    testlog_info(qtensor.get_name()," quant type : uniform");
  else
    testlog_info(qtensor.get_name(), " quant type : nonuniform");

  auto quant_subtype = qtensor.get_quant_subtype();
  if (quant_subtype == quant_subtype_t::symmetric)
    testlog_info(qtensor.get_name()," quant subtype : symmetric");
  else
    testlog_info(qtensor.get_name(), " quant subtype : asymmetric");

  //dequantize the tensor
  qtensor.set_const(true);

  auto dqtensor = tensor_t()
    .set_name("dequantized tensor")
    .set_size({MATMUL_ROWS, MATMUL_COLS})
    .set_data_type(data_type_t::f32)
    .set_storage()
    .create();

  if (! dqtensor.check() ) {
    testlog_error("tensor creation of ", dqtensor.get_name(), " failed");
    return NOT_OK;
  }

  float*  dqhandle       = (float *)dqtensor.get_raw_handle_unsafe();
  const int8_t* qchandle = (const int8_t *)qtensor.get_raw_handle_const();
  shandle                = (const float *)qtensor.get_quant_scale_raw_handle_const();

  for (uint32_t r = 0; r < MATMUL_ROWS; ++r) {
    float scale = shandle[r];

    for (uint32_t c = 0; c < MATMUL_COLS; ++c) {
      auto dqoffset = dqtensor.compute_offset({r,c});
      auto qoffset  = qtensor.compute_offset({r,c});

      dqhandle[dqoffset] = qchandle[qoffset]*scale;
    }
  }

  //display results
  for (uint32_t r = 0; r < MATMUL_ROWS; ++r) {
    for (uint32_t c = 0; c < MATMUL_COLS; ++c) {
      auto udval  = udtensor.at({r,c});
      auto dqval  = dqtensor.at({r,c});

      testlog_info("orig[",r,",",c,"]=",udval," dq[",r,",",c,"]=",dqval);
    }
  }

  return OK;
}

}//examples
}//zendnnl
