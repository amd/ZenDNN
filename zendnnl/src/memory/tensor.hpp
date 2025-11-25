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
#ifndef _TENSOR_HPP_
#define _TENSOR_HPP_

#include <cstdlib>
#include <iostream>
#include <cstdint>
#include <vector>
#include <memory>
#include <string>
#include <algorithm>

#include "common/zendnnl_global.hpp"
#include "common/hash_object.hpp"
#include "common/data_types.hpp"
#include "tensor_options.hpp"
#include "tensor_quant.hpp"
#include "tensor_storage.hpp"

namespace zendnnl {
/** @namespace zendnnl::memory
 *  @brief A namespace to contain all memory management related classes, enumerations,
 *  variables and functions.
 */
namespace memory {

using namespace zendnnl::common;
using namespace zendnnl::error_handling;

/** @class tensor_t
 *  @brief A class to hold multi-dimensional data.
 *
 * A tensor object consists of the following kind of data,
 * 1. Tensor meta data, like dimensions, sizes, stride and data type
 *    of the data. Meta data is stored in @c tensor_option_t.
 * 2. Tensor quantization data, like scale and zero point. Quantization data
 *    is stored in @c tensor_quant_t.
 * 3. A memory buffer for tensor data. This buffer is in @c tensor_storage_t. A
 *    tensor storage may be shared by multiple tensors.
 *
 * An object of this class can be created by calling its constructor and
 * chaining with it functions that set its meta data, quant data and memory buffers
 * (functions prefixed with "set_"), and finally calling @c create(). A @c tensor_t
 * object once created is immutable, in the sense neither its meta data, nor
 * quant data, nor tensor storage can be changed. Data in @c tensor_storage_t
 * can be changed by getting a raw pointer to its memory buffer.
 *
 * For example a bfloat16 tensor of size 5x10 can be created as follows
 *
 * @code
 * auto bfloat_tensor = tensor_t()
 *                         .set_size({5,10})
 *                         .set_data_type(data_type_t::bf16)
 *                         .create()
 * @endcode
 *
 * @c create() validates the tensor parameters provided by the tensor creation chain,
 * creates the object and sets a flag to mark the object valid. Functions to set
 * tensor parameters could be given in any orderin the creation chain. If tensor parameters
 * are found to be inconsistent or inadequate for tensor creation, it marks the object
 * as invalid object. Weather a tensor is valid can be checked by
 * @c hash_object_t::check() function of its parent class.
 *
 * @code
 * if (bfloat_tensor.check())
 *    // valid object
 * else
 *    // invalid object
 * @endcode
 *
 * Memory buffer in the @c tensor_storege_t can be acquired by the following ways,
 * 1. by memory allocation, any allocated memory is released once no tensor points to it,
 * 2. by borrowing from a deep learning framework tensor, or
 * 3. by referring to memory buffer of another tensor_t object.
 *
 * @sa tensor_option_t, tensor_quant_t, tensor_layout_t, tensor_storage_t, hash_object_t.
 */

class tensor_t final : public hash_object_t {
public:
  /** @brief Parent type */
  using   parent_type       = hash_object_t;

  /** @brief A shared pointer type to tensot storage */
  using   storage_sptr_type = std::shared_ptr<tensor_storage_t>;

  /** @brief Index type */
  using   index_type        = tensor_option_t::index_type;

  /** @brief Index vector type */
  using   index_vec_type    = tensor_option_t::index_vec_type;

  /** @name Constructors, Destructors and Assignment
   */
  /**@{*/
  /** @brief Default constuctor */
  tensor_t();

  /** @brief Copy constuctor */
  tensor_t(const tensor_t& tensor_)            = default;

  /** @brief Copy assignment */
  tensor_t& operator=(const tensor_t& tensor_) = default;

  /** @brief Move constructor */
  tensor_t(tensor_t&& other_);

  /** @brief Move assignment */
  tensor_t& operator=(tensor_t&& other_);
  /**@}*/

  /** @name Tensor Dimensions
   */
  /**@{*/
  /** @brief Set tensor size.
   *
   * Tensor creation will fail if size is not set.
   * @param size_ : a vector of tensor sizes.
   * @return A reference to self.
   */

  tensor_t& set_size(index_vec_type size_);

  /** @brief Get tensor size.
   * @return Size vector.
   */
  index_vec_type get_size() const;

  /** @brief Get tensor size at an index.
   * @param index_ : size vector index.
   * @return Size at index_.
   */
  index_type get_size(uint32_t index_) const;

  /** @brief Get tensor dimensions.
   * @return Tensor dimensions.
   */
  uint32_t get_dim() const;

  /** @brief Set aligned size for aligned tensor
   *
   * aligned size defines alignment in each dimension, and may be different than
   * tensor size. These may be used to align one dimension to a boundary,
   * or may be used to access a sub-tensor.
   *
   * For example an fp32 tensor with size 5x10, but stride size 5x16 will have
   * each row aligned at 64 byte boundary. Similarly if a tensor-A of size 5x5
   * and stride 5x10 is created by pointing to a buffer of another tensor-B of
   * size 5x10, then tensor-A accesses only left 5x5 sub-tensor of tensor-B.
   *
   * If stride size is not set then stride size is made equal to size of the
   * tensor.
   *
   * @param  aligned_size_ : a vector of aligned sizes.
   * @return A reference to self.
   */
  tensor_t& set_aligned_size(index_vec_type aligned_size_);

  /** @brief Get aligned size of the tensor.
   * @return Aligned size vector.
   */
  index_vec_type get_aligned_size() const;

  /** @brief Get tensor aligned size at an index.
   * @param index_ : aligned size vector index.
   * @return Aligned size at index_.
   */
  index_type get_aligned_size(uint32_t index_) const;

  /** @brief Set tensor base index.
   *
   * Base index refers to the index of element to be treated as "zeroth" tensor
   * element. Default base index is an all zero index.
   *
   * Base index together with @c set_aligned_size() can be used to access a sub-tensor.
   * Consider a tenor-A of size 5x10. Another tensor-B, that shares memory
   * with tensor-A, of size 5x5, aligned size 5x10, and base index 5 will access right 5x5
   * sub-tensor of tensor-A.
   *
   * @param base_ : base index.
   * @return A reference to self.
   */
  tensor_t& set_base_index(index_vec_type base_);

  /** @brief Get tensor base index.
   *
   * Please see @c set_base_index() for base index description.
   * @return Base index.
   */
  index_vec_type get_base_index() const;

  /** @brief Set stride.
   *
   *  Stride decides the access pattern of a tensor and can be used for a sub-tensor
   *  broadcast along any axis. For example a tensor of size (5,4,3), and stride
   *  (0,3,1) is broadcasting a 4x3 tensor along depth.
   */
  tensor_t& set_stride(index_vec_type stride_);

  /** @brief Get stride vector.
   *
   *  Stride decides the access pattern of a tensor. Please see @c set_stride() for
   *  for further details.
   *
   *  @return Stride vector.
   */
  index_vec_type get_stride() const;

  /** @brief Get stride at an index.
   *
   *  Stride decides the access pattern of a tensor. Please see @c set_stride() for
   *  for further details.
   *
   *  @param index_ : index of stride vector.
   *  @return Stride at given index.
   */
  index_type get_stride(uint32_t index_) const;

  /**@}*/

  /** @name Tensor Quantization
   */
  /**@{*/

  /** @brief set tensor scale */
  tensor_t& set_quant_scale(const tensor_t& quant_scale_);

  /** @brief set tensor zero */
  tensor_t& set_quant_zero_point(const tensor_t& quant_zero_);

  /** @brief check if it is a quantized tensor */
  bool is_quantized() const;

  /** @brief get quant type */
  quant_type_t get_quant_type() const;

  /** @brief get quant subtype */
  quant_subtype_t get_quant_subtype() const;

  /** @brief get quant scale size */
  index_vec_type get_quant_scale_size() const;

  /** @brief get quant scale stride */
  index_vec_type get_quant_scale_stride() const;

  /** @brief get quant scale stride */
  index_vec_type get_quant_scale_block_size() const;

  /** @brief compute scale offset */
  uint64_t compute_quant_scale_offset(const index_vec_type& index_) const;

  /** @brief get quant scale data type */
  data_type_t get_quant_scale_data_type() const;

  /** @brief get quant scale raw handle */
  const void* get_quant_scale_raw_handle_const() const;

  /** @brief get quant scale raw handle */
  const void* get_quant_scale_raw_handle_const(const index_vec_type& index_) const;

  /** @brief get quant scale size */
  index_vec_type get_quant_zero_size() const;

  /** @brief get quant scale stride */
  index_vec_type get_quant_zero_stride() const;

  /** @brief get quant scale stride */
  index_vec_type get_quant_zero_block_size() const;

  /** @brief compute quant zero offset */
  uint64_t compute_quant_zero_offset(const index_vec_type& index_) const;

  /** @brief get quant zero data type */
  data_type_t get_quant_zero_data_type() const;

  /** @brief get quant scale data type */
  const void* get_quant_zero_raw_handle_const() const;

  /** @brief get quant scale data type */
  const void* get_quant_zero_raw_handle_const(const index_vec_type& index_) const;

  /**@}*/

  /** @name DataType, Format, Order, Constness
   */
  /**@{*/
  /** @brief Set tensor data type.
   *
   * Default data type is data_type_t::f32.
   *
   * @sa @c data_type_t enum for supported data types.
   * @param data_type_ : data type.
   * @return A reference to self.
   */
  tensor_t& set_data_type(data_type_t data_type_);

  /** @brief Get tensor data type.
   *
   * @sa @c data_type_t enum for supported data types.
   * @return The tensor data type.
   */
  data_type_t get_data_type() const;

  /** @brief Set tensor layout.
   *
   * Tensor layout refers to how tensor data is layed out in
   * the tensor memory (contiguous, blocked or strided etc.).
   * @sa @c tensor_layout_t enum for supported layouts.
   * @param layout_: tensor layout.
   * @return A reference to self.
   */
  tensor_t& set_layout(tensor_layout_t layout_);

  /** @brief Get tensor layout.
   * @sa @c tensor_layout_t enum for suppported layouts.
   * @return Tensor layout.
   */
  uint16_t get_layout() const;

  /** @brief Set tensor channel order.
   *
   * Tensor channel order refers to channel order like NCHW or NHCW.
   * Channel order is generally DNN dependent (NCHW does not make sense if
   * it is not a CNN for processing image data), therefor it is represented
   * as a string. DNN can set this string as their context requires it.
   *
   * Tensor order for a 2D tensor in case of MatMul can be used by library
   * to compute in transposed or non-transposed manner.
   * If the order is set as "ab" then tensor is treated non-transposed.
   * If the order is set as "ba" then tensor is treated transposed.
   *
   * There is no default channel order.
   * @param order_: tensor channel order.
   * @return A reference to self.
   */
  tensor_t& set_order(std::string order_);

  /** @brief Get tensor order.
   * @return Tensor order.
   */
  std::string get_order() const;

  /** @brief Set tensor options
   * @param tensor_option_ : tensor option to set.
   * @return A reference to self.
   */
  tensor_t& set_tensor_option(const tensor_option_t& option_);

  /** @brief Get tensor options
   * @return Tensor option.
   */
  tensor_option_t& get_tensor_option();

  /** @brief Set tensor to be const.
   *
   * Only const raw pointer of a const tensor can be taken so that the data
   * pointed by it can not be modified.
   *
   * @param constness_: tensor constness.
   * @return A reference to self.
   */
  tensor_t& set_const(bool constness_);

  /** @brief Get tensor constness.
   * @return Tensor constness.
   */
  bool get_const() const;


  /**@}*/

  /** @name Profiling and Diagnostics
   */
  /**@{*/
  /** @brief Set tensor name.
   *
   * Name can be used to identify the tensor for profiling and disgnostic
   * purposes.
   *
   * ZenDNNL does not check for uniqueness of names.
   *
   * Default is "unknown tensor".
   * @param name_ : The tensor name.
   * @return A reference to self. This function can be chained to create a tensor.
   */
  tensor_t& set_name(std::string name_);

  /** @brief Get tensor name
   * @return The tensor name.
   */
  std::string get_name() const;

  /** @brief Compute offset corresponding to an index.
   *
   * Given an N dim tensor with index_ {i1, i2,...,iN}, and stride
   * {s1,s2,...,sN}, the offset is given by sum(ik*sk).
   *
   * @param index_ : an index for which offset is required.
   * @return offset of the index.
   */
  uint64_t compute_offset(const index_vec_type& index_) const;

  /** @brief Get tensor element.
   *
   * Tensor element of any other type is either dequantized, or converted to
   * float.
   * @param index_ : element index.
   * @return Dequantized or float converted element.
   */
  float at(const index_vec_type& index_) const;


  /**@}*/

  /** @name Storage
   */
  /**@{*/
  /** @brief Get element count.
   *
   * For example a 3x5x10 tensor will have 150 elements.
   *
   * @return Element count.
   */
  uint64_t get_nelem() const;
  tensor_t &set_nelem(uint64_t nelem_);

  /** @brief Get tensor buffer size in bytes.
   *
   * Byte size of tensor buffer depends on the data type (f32, bf16...),
   * tensor layout(contiguous, strided...) and the tensor size, set either by
   * @c set_size(), or @c set_stride_size().
   *
   * For example, a 5x10 contigenous tensor of bf16 type will have 5*10*2 = 100 bytes.
   * A 5x10 strided tensor with 5x16 strides of f32 type will have 5*16*4 = 320 bytes.
   *
   * @return Tensor buffer size in bytes.
   */
  uint64_t get_buffer_sz_bytes() const;

  /** @brief Get count of tensors sharing same storage.
   *
   * Tensor storage can be shared by multiple tensors. This is useful either
   * to provide a different view of the tensor, or a sub-tensor.
   *
   * @return Count of tensors sharing same storage.
   */
  uint32_t get_storage_count()   const;

  /** @brief Get the raw hande to tensor memory buffer.
   *
   * Getting raw handle to the memory buffer is generally unsafe, however this function is
   * provided for faster access to the tensor data. Also many low level routines like
   * AOCL require raw memory buffer.
   *
   * This function gives raw pointer to zeroth tensor element.
   *
   * @return Raw pointer to the memory buffer.
   */
  void* get_raw_handle_unsafe() const;

  /** @brief Get the raw hande to tensor memory buffer.
   *
   * Getting raw handle to the memory buffer is generally unsafe, however this function is
   * provided for faster access to the tensor data. Also many low level routines like
   * AOCL require raw memory buffer.
   *
   * This function gives raw pointer to the tensor element pointed by index_.
   *
   * @return Raw pointer to the memory buffer.
   */
  void* get_raw_handle_unsafe(const index_vec_type& index_) const;

  /** @brief Get a const raw handle to tensor memory buffer.
   *
   * Const raw handle can be used to read but not modify tensor buffer.
   * This function gives raw pointer to zeroth tensor element.
   *
   * @return Const raw pointer to the memory buffer.
   */
  const void* get_raw_handle_const() const;

  /** @brief Get a const raw handle to tensor memory buffer.
   *
   * Const raw handle can be used to read but not modify tensor buffer.
   * This function gives raw pointer to the tensor element pointed by index_.
   *
   * @return Const raw pointer to the memory buffer.
   */
  const void* get_raw_handle_const(const index_vec_type& index_) const;

  /** @brief Allocate unaligned storage to the tensor.
   *
   * Memory to be allocated is calculated based on tensor data type, tensor format, and
   * tensor size, or tensor stride size.
   *
   * @se Memory is allocated in tensor storage. An exception is raised if memory
   * allocation fails.
   * @sa get_buffer_sz_byte() for further description on how buffer size is calculated.
   * @return A reference to self.
   */
  tensor_t& set_storage();

  /** @brief Allocate aligned storage to the tensor.
   *
   * Memory to be allocated is calculated based on tensor data type, tensor format, and
   * tensor size, or tensor stride size.
   *
   * @se Memory is allocated in tensor storage at the alignment boundary given.
   * An exception is raised if memory allocation fails.
   * @sa get_buffer_sz_byte() for further description on how buffer size is calculated.
   * @return A reference to self.
   * @param aligned_to_ : memory boundary the buffer need to be aligned to.
   * @return A reference to self.
   */
  tensor_t& set_storage(uint32_t aligned_to_);
  tensor_t& set_storage(uint32_t aligned_to_, uint64_t nelem_);
  /** @brief Borrow memory buffer from another raw pointer.
   *
   * Needed to borrow tensor buffer from a deep learning framework.
   * @param raw_ptr_ : raw pointer to a memory buffer.
   * @param sz_bytes_ : buffer size in bytes.
   * @return A reference to self.
   */
  tensor_t& set_storage(void* raw_ptr_, uint64_t sz_bytes_);

  /** @brief Share tensor storage from another tensor.
   *
   * @param other_ : Tensor to share storage from.
   * @return A reference to self.
   */
  tensor_t& set_storage(const tensor_t& other_);
  /**@}**/

  /** @name Create, Reset and Hash
   */
  /**@{*/

  /** @brief Create a tensor.
   *
   * A tensor is created by default tensor constuctor, and chaining it
   * with functions to set its meta data, quant data and storage
   * (functions prefixed with "set_"), and finally chaining it with @c create().
   *
   * For example a bfloat16 tensor of size 5x10 can be created as follows
   *
   * @code
   * auto bfloat_tensor = tensor_t()
   *                         .set_size({5,10})
   *                         .set_data_type(data_type_t::bf16)
   *                         .create()
   * @endcode
   *
   * @c create() validates tensor parameters provided by the tensor creation chain,
   * creates the object and sets a flag to mark the object valid. If tensor parameters
   * are found to be inconsistent or inadequate for tensor creation, it marks the object
   * as invalid object. Weather a tensor is valid can be checked by
   * @c hash_object_t::check() function of its parent class.
   *
   * @code
   * if (bfloat_tensor.check())
   *    // valid object
   * else
   *    // invalid object
   * @endcode
   *
   * If object creation is successful it makes the tensor parameters immutable. An object
   * can be reset by calling @c reset().
   *
   * @return A reference to self.
   */
  tensor_t& create();

  /** @brief Reset the tensor.
   *
   * Reset all meta data, quant data and tensor storage. If storage is allocated by the
   * libary, free the storage. Reset the hash to zero.
   */
  void reset();

  /** @brief Generate object hash.
   *
   * Hash generated by an object uniquely identifies the object, therefore hash is
   * generated by taking all the paramaters that uniquely identify a tensor.
   *
   * Only a valid object returns a hash. Invalid object hash is set to zero.
   *
   * Tensor hash combines hash from @c tensor_option_t, @c tensor_quant_t and
   * @c tensor_storage_t.
   *
   * @return Object hash.
   */
  std::size_t hash() override;
  /**@}*/

  /** @brief Returns tensor information.
   *
   * Returns a string containing tensor meta data like size, stride size
   * and data type. This is used for logging and profiling.
   * @return std::string containing tensor information.
   */
  std::string tensor_info();

protected:
  /** @brief Sanity check on size
   *
   */
  status_t size_sanity_check() const;

  /** @brief Check if size and stride_size are consistent.
   *
   * If size and stride_size both are given, checks if they are of same size,
   * and stride_size is at least equal to size.
   *
   * @se If size and stride_size are inconsistent, sets object to bad_hash_object.
   */
  status_t aligned_size_sanity_check();

  /** @brief Order sanity check
   */
  status_t order_sanity_check();

  /** @brief Compute tensor strides.
   *
   * Compute default stride either from stride_size. If both size and
   * stride_size are given using set_size() and set_stride_size() respectively,
   * stride_sanity_check() checks if they are consistent,
   * else stride_size is set equal to size.
   *
   * @se option.stride, option.nelem and option.strided_nelem are computed.
   */
  void set_default_stride_with_order();
  void set_default_stride_without_order();

  /** @brief sanity check on stride.
   *
   */
  status_t stride_sanity_check_with_order();
  status_t stride_sanity_check_without_order();

  /** @brief Set default base index.
   *
   * If no base index is given using @c set_base(), default base index is set
   * to all zero.
   * @se option.base and option.base_offset set to zero.
   */
  void set_default_base();

  /** @brief Check if the given index is within the bounds of the tensor size.
   *
   * @return success if the index is within bounds, failure otherwise.
   */
  status_t index_sanity_check(const index_vec_type& index_) const;

  /** @brief Check if the given index is within the bounds of the tensor size.
   *
   * @return success if the index is within bounds, failure otherwise.
   */
  index_vec_type permute_axes_order(const index_vec_type& in_vec_,
                                    bool order_to_default) ;

  /** @brief Validate meta data is consistent and sufficient to create object.
   *
   * @se If meta data is inconsistent or insufficient, sets object to bad_hash_object.
   */
  status_t validate_meta_info();

  /** @brief Compute quant block size
   *
   * @return  quant block size for given size
   */
  index_vec_type compute_quant_block_size(const index_vec_type& size_);

  /** @brief Compute quant stride
   *
   * @return  quant stride for given size
   */
  index_vec_type compute_quant_stride(const index_vec_type& size_);

  /** @brief Validate quant scale.
   *
   * @return memory_bad_quant in case of failure.
   */
  status_t validate_quant_scale();

  /** @brief Validate quant zero point.
   *
   * @return memory_bad_quant in case of failure.
   */
  status_t validate_quant_zero();

  /** @brief Validate scale tensor.
   *
   * @se if scale tensor is not proper, set status to memory_bad_quant.
   */
  status_t validate_quant_info();

private:
  tensor_option_t                option; /**< Tensor meta data. See @c tensor_option_t
                                            for further description. */
  std::optional<tensor_quant_t>  quant; /**< Tensor quantization data. See @c tensor_quant_t
                                           for further description */
  storage_sptr_type              storage; /**< A shared pointer to tensor storage. See
                                             @c tensor_storage_t for further description */

  bool                           allocate; /**< Allocate strorage to tensor */
  std::string                    name; /**< Tensor name. This is relevant only for profiling
                                          and diagnostic purposes. */
};

} //memory

namespace interface {
using tensor_t = zendnnl::memory::tensor_t;
} //export

} //zendnnl


#endif
