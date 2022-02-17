/*******************************************************************************
* Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
* Notified per clause 4(b) of the license.
*******************************************************************************/

/*******************************************************************************
* Copyright 2016-2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/// @file
/// C API types definitions

#ifndef ZENDNN_TYPES_H
#define ZENDNN_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

/// @cond DO_NOT_DOCUMENT_THIS
#include <stddef.h>
#include <stdint.h>
/// @endcond

/// @addtogroup zendnn_api
/// @{

/// @addtogroup zendnn_api_utils
/// @{

/// Status values returned by the library functions.
typedef enum {
    /// The operation was successful
    zendnn_success = 0,
    /// The operation failed due to an out-of-memory condition
    zendnn_out_of_memory = 1,
    /// The operation failed because of incorrect function arguments
    zendnn_invalid_arguments = 2,
    /// The operation failed because requested functionality is not implemented
    zendnn_unimplemented = 3,
    /// Primitive iterator passed over last primitive descriptor
    zendnn_iterator_ends = 4,
    /// Primitive or engine failed on execution
    zendnn_runtime_error = 5,
    /// Queried element is not required for given primitive
    zendnn_not_required = 6,
} zendnn_status_t;

/// @} zendnn_api_utils

/// @addtogroup zendnn_api_memory
/// @{

/// Data type specification
typedef enum {
    /// Undefined data type, used for empty memory descriptors.
    zendnn_data_type_undef = 0,
    /// 16-bit/half-precision floating point.
    zendnn_f16 = 1,
    /// non-standard 16-bit (bfloat16 w/ 7 bit mantissa) floating point.
    zendnn_bf16 = 2,
    /// 32-bit/single-precision floating point.
    zendnn_f32 = 3,
    /// 32-bit signed integer.
    zendnn_s32 = 4,
    /// 8-bit signed integer.
    zendnn_s8 = 5,
    /// 8-bit unsigned integer.
    zendnn_u8 = 6,
} zendnn_data_type_t;

/// Memory format kind
typedef enum {
    /// Undefined memory format kind, used for empty memory descriptors.
    zendnn_format_kind_undef = 0,
    /// Unspecified format kind.
    /// The primitive selects a format automatically.
    zendnn_format_kind_any,
    /// A tensor in a generic format described by the stride and blocking
    /// values in each dimension. See @ref zendnn_blocking_desc_t for more
    /// information.
    zendnn_blocked,
    /// Weights format used in 8bit Winograd convolution
    zendnn_format_kind_wino,
    /// Packed weights format used in RNN
    zendnn_format_kind_rnn_packed,
} zendnn_format_kind_t;

/// Memory format tag specification.
///
/// ZENDNN formats describe physical data layout. The physical layout
/// is described as a sequence of the dimensions as they are laid out in the
/// memory (from the outer-most to the inner-most). Note that this order
/// doesn't affect the logical order of the dimensions that is kept in the
/// `dims` field of the zendnn_memory_desc_t structure. The logical order of the
/// dimensions is specified by the primitive that uses the tensor.
///
/// For example, CNN 5D tensor always has its logical dimensions in the order
/// `(batch, channels, depth, height, width)`, while the physical layout might be
/// `NCDHW` (corresponds to #zendnn_ncdhw format tag) or
/// `NDHWC` (corresponds to #zendnn_ndhwc format tag).
///
/// ~~~cpp
/// int batch = 2, channels = 16, depth = 13, height = 13, width = 13;
///
/// int ndims = 5; // 5D tensor
/// zendnn_dims_t dims = {batch, channels, depth, height, width};
/// zendnn_memory_desc_t data_in_ncdhw;
/// zendnn_memory_desc_init_by_tag(
///      &data_in_ncdhw, 5, dims, zendnn_f32, zendnn_ncdhw);
///
/// // note that in both cases dims passed are the same
/// zendnn_memory_desc_t data_in_ndhwc;
/// zendnn_memory_desc_init_by_tag(
///      &data_in_ndhwc, 5, dims, zendnn_f32, zendnn_ndhwc);
/// ~~~
///
/// Memory format tags can be further divided into two categories:
///  - Domain-agnostic names, i.e. names the do not depend on the tensor usage
///    in the specific primitive. These names use letters from `a` to `l` to
///    denote logical dimension from 1 to 12, and form the order in which the
///    dimensions are laid in memory. For instance, #zendnn_ab is used to denote
///    2D tensor where the second logical dimension (aka `b`) is the innermost,
///    i.e. has stride = 1, and the first logical dimension (`a`) laid out in
///    memory with stride equal to the size of second dimension. On the other
///    hand, #zendnn_ba is just transposed version of the same tensor: the
///    first dimension (`a`) becomes the innermost one.
///  - Domain-specific names, i.e. names that make sense only in the context of
///    a certain domain, such as CNN. This names are just aliases to the
///    corresponding domain-agnostic tags and used mostly for the convenience.
///    For example, #zendnn_nc is used to denote 2D CNN activations tensor
///    memory format, where channels are the innermost dimension and batch is an
///    outermost one. Moreover, #zendnn_nc is just an alias to #zendnn_ab,
///    since for ZENDNN CNN primitives the logical dimensions of
///    activations tensors come in order: batch, channels, spatial.
///    In other words, batch corresponds to the first logical dimension (`a`),
///    channels correspond to the second one (`b`).
///
/// The following domain-specific notation applies to memory format tags:
///  - @c 'n' denotes the mini-batch dimension
///  - @c 'c' denotes a channels dimension
///  - When there are multiple channel dimensions (for example, in convolution
///    weights tensor), @c 'i' and @c 'o' denote dimensions of input and output
///    channels
///  - @c 'd', @c 'h', and @c 'w' denote spatial depth, height, and width
///    respectively
///
/// Upper-case letters indicate that the data is laid out in blocks for a
/// particular dimension. In such cases, the format name contains both upper-
/// and lower-case letters for that dimension with a lower-case letter preceded
/// by the block size. For example: #zendnn_nChw8c describes a format where the
/// outermost dimension is mini-batch, followed by the channel block number,
/// followed by the spatial height and width, and finally followed by 8-element
/// channel blocks.
///
/// @sa @ref dev_guide_understanding_memory_formats
typedef enum {
    /// Undefined memory format tag
    zendnn_format_tag_undef = 0,
    /// Undefined memory format tag.
    /// The primitive selects a format automatically.
    zendnn_format_tag_any,

    // Semantic agnostic section
    // The physical order of dimensions is defined by the permutation of the
    // characters, assuming that ab..z defines the natural order.

    // Plain formats

    zendnn_a, ///< plain 1D tensor
    zendnn_ab, ///< plain 2D tensor
    zendnn_abc, ///< plain 3D tensor
    zendnn_abcd, ///< plain 4D tensor
    zendnn_acbd, ///< plain 4D tensor
    zendnn_abcde, ///< plain 5D tensor
    zendnn_abcdef, ///< plain 6D tensor
    zendnn_abcdefg, ///< plain 7D tensor
    zendnn_abcdefgh, ///< plain 8D tensor
    zendnn_abcdefghi, ///< plain 9D tensor
    zendnn_abcdefghij, ///< plain 10D tensor
    zendnn_abcdefghijk, ///< plain 11D tensor
    zendnn_abcdefghijkl, ///< plain 12D tensor

    // Permuted plain formats

    zendnn_abdc, ///< permuted 4D tensor
    zendnn_abdec, ///< permuted 5D tensor
    zendnn_acb, ///< permuted 3D tensor
    zendnn_acbde, ///< permuted 5D tensor
    zendnn_acbdef, ///< permuted 6D tensor
    zendnn_acdb, ///< permuted 4D tensor
    zendnn_acdeb, ///< permuted 5D tensor
    zendnn_ba, ///< permuted 2D tensor
    zendnn_bac, ///< permuted 3D tensor
    zendnn_bacd, ///< permuted 4D tensor
    zendnn_bacde, ///< permuted 5D tensor
    zendnn_bca, ///< permuted 3D tensor
    zendnn_bcda, ///< permuted 4D tensor
    zendnn_bcdea, ///< permuted 5D tensor
    zendnn_cba, ///< permuted 3D tensor
    zendnn_cdba, ///< permuted 4D tensor
    zendnn_dcab, ///< permuted 4D tensor
    zendnn_cdeba, ///< permuted 5D tensor
    zendnn_decab, ///< permuted 5D tensor
    zendnn_defcab, ///< permuted 6D tensor
    zendnn_abced, ///< permuted 5D tensor
    zendnn_abcdfe, ///< permuted 6D tensor
    zendnn_abcdegf, ///< permuted 7D tensor
    zendnn_abcdefhg, ///< permuted 8D tensor
    zendnn_abcdefgih, ///< permuted 9D tensor
    zendnn_abcdefghji, ///< permuted 10D tensor
    zendnn_abcdefghikj, ///< permuted 11D tensor
    zendnn_abcdefghijlk, ///< permuted 12D tensor

    // Opaque blocked formats

    zendnn_Abc16a,
    zendnn_ABc16a16b,
    zendnn_ABc32a32b,
    zendnn_ABc4a4b,
    /// 3D tensor blocked by 2nd dimension with block size 16
    zendnn_aBc16b,
    zendnn_ABc16b16a,
    zendnn_Abc4a,
    /// 3D tensor blocked by 2nd dimension with block size 32
    zendnn_aBc32b,
    /// 3D tensor blocked by 2nd dimension with block size 4
    zendnn_aBc4b,
    zendnn_ABc4b16a4b,
    zendnn_ABc2b8a4b,
    zendnn_ABc16b16a4b,
    zendnn_ABc16b16a2b,
    zendnn_ABc4b4a,
    zendnn_ABc8a16b2a,
    zendnn_ABc8a8b,
    zendnn_ABc8a4b,
    /// 3D tensor blocked by 2nd dimension with block size 8
    zendnn_aBc8b,
    zendnn_ABc8b16a2b,
    zendnn_BAc8a16b2a,
    zendnn_ABc8b8a,
    zendnn_Abcd16a,
    zendnn_Abcd8a,
    zendnn_ABcd16a16b,
    zendnn_Abcd32a,
    zendnn_ABcd32a32b,
    /// 4D tensor blocked by 2nd dimension with block size 16
    zendnn_aBcd16b,
    zendnn_ABcd16b16a,
    zendnn_aBCd16b16c,
    zendnn_aBCd16c16b,
    zendnn_Abcd4a,
    /// 4D tensor blocked by 2nd dimension with block size 32
    zendnn_aBcd32b,
    /// 4D tensor blocked by 2nd dimension with block size 4
    zendnn_aBcd4b,
    zendnn_ABcd4b16a4b,
    zendnn_ABcd16b16a4b,
    zendnn_ABcd16b16a2b,
    zendnn_ABcd4b4a,
    zendnn_ABcd4a4b,
    zendnn_aBCd2c4b2c,
    zendnn_aBCd4b8c2b,
    zendnn_aBCd4c16b4c,
    zendnn_aBCd2c8b4c,
    zendnn_aBCd16c16b4c,
    zendnn_aBCd16c16b2c,
    zendnn_aBCd4c4b,
    zendnn_aBCd4b4c,
    zendnn_ABcd8a16b2a,
    zendnn_ABcd2b8a4b,
    zendnn_ABcd8a8b,
    zendnn_ABcd8a4b,
    /// 4D tensor blocked by 2nd dimension with block size 8
    zendnn_aBcd8b,
    zendnn_aBCd4c8b2c,
    zendnn_ABcd8b16a2b,
    zendnn_aBCd8b16c2b,
    zendnn_BAcd8a16b2a,
    /// 4D tensor blocked by 1st and 2nd dimension with block size 8
    zendnn_ABcd8b8a,
    zendnn_aBCd8b8c,
    zendnn_aBCd8b4c,
    zendnn_aBCd8c16b2c,
    zendnn_ABcde8a16b2a,
    zendnn_aCBd8b16c2b,
    zendnn_aBCd8c8b,
    zendnn_Abcde16a,
    zendnn_Abcde32a,
    zendnn_ABcde16a16b,
    zendnn_BAcde8a16b2a,
    /// 4D tensor blocked by 3rd dimension with block size 4
    zendnn_aBCd2b4c2b,
    /// 5D tensor blocked by 1st dimension with block size 16
    zendnn_ABcde4b16a4b,
    /// 5D tensor blocked by 1st dimension with block size 8
    zendnn_ABcde2b8a4b,
    /// 5D tensor blocked by 2nd dimension with block size 16
    zendnn_aBcde16b,
    zendnn_ABcde16b16a,
    zendnn_aBCde16b16c,
    zendnn_aBCde16c16b,
    zendnn_aBCde2c8b4c,
    zendnn_Abcde4a,
    /// 5D tensor blocked by 2nd dimension with block size 32
    zendnn_aBcde32b,
    /// 5D tensor blocked by 2nd dimension with block size 4
    zendnn_aBcde4b,
    zendnn_ABcde4b4a,
    zendnn_ABcde4a4b,
    zendnn_aBCde4b4c,
    zendnn_aBCde2c4b2c,
    zendnn_aBCde4b8c2b,
    zendnn_aBCde4c16b4c,
    zendnn_aBCde16c16b4c,
    zendnn_aBCde16c16b2c,
    zendnn_aBCde4c4b,
    zendnn_Abcde8a,
    zendnn_ABcde8a8b,
    zendnn_ABcde8a4b,
    zendnn_BAcde16b16a,
    /// 5D tensor blocked by 2nd dimension with block size 8
    zendnn_aBcde8b,
    zendnn_ABcde8b16a2b,
    zendnn_aBCde8b16c2b,
    zendnn_aBCde4c8b2c,
    zendnn_aCBde8b16c2b,
    zendnn_ABcde8b8a,
    zendnn_ABcde32a32b,
    zendnn_aBCde8b8c,
    zendnn_aBCde8b4c,
    zendnn_ABc4a8b8a4b,
    zendnn_ABcd4a8b8a4b,
    zendnn_ABcde4a8b8a4b,
    zendnn_BAc4b8a8b4a,
    zendnn_BAcd4b8a8b4a,
    zendnn_BAcde4b8a8b4a,
    zendnn_ABcd2a8b8a2b,
    zendnn_aBCd4b8c8b4c,
    zendnn_aBCde4b8c8b4c,
    zendnn_aBCde2b8c8b2c,
    zendnn_aBCde8c16b2c,
    zendnn_aBCde8c8b,
    /// 5D tensor blocked by 3rd dimension with block size 4
    zendnn_aBCde2b4c2b,
    /// 6D tensor blocked by 2nd dimension with block size 16
    zendnn_aBcdef16b,
    zendnn_aBCdef16b16c,
    zendnn_aBCdef16c16b,
    zendnn_aBCdef4c16b4c,
    /// 6D tensor blocked by 2nd dimension with block size 8
    zendnn_aBCdef2c8b4c,
    zendnn_aBCdef4c8b2c,
    /// 6D tensor blocked by 3rd dimension with block size 4
    zendnn_aBCdef2b4c2b,
    /// 6D tensor blocked by 2nd dimension with block size 4
    zendnn_aBcdef4b,
    zendnn_aBCdef4c4b,
    zendnn_aBCdef4b4c,
    zendnn_aBCdef2c4b2c,
    zendnn_aBCdef4b8c2b,
    zendnn_aBCdef8b8c,
    zendnn_aBCdef8b4c,
    zendnn_aBCdef8c16b2c,
    zendnn_aBCdef4b8c8b4c,
    zendnn_aBCdef8b16c2b,
    zendnn_aCBdef8b16c2b,
    zendnn_aBCdef8c8b,
    zendnn_aBdc16b,
    zendnn_aBdC16b2c,
    zendnn_aBdC16b4c,
    zendnn_aBdc4b,
    zendnn_aBdc8b,
    zendnn_aBdec16b,
    zendnn_aBdeC16b2c,
    zendnn_aBdeC16b4c,
    zendnn_aBdec32b,
    zendnn_aBdec4b,
    zendnn_aBdec8b,
    zendnn_aBdefc16b,
    zendnn_aBdefC16b2c,
    zendnn_aCBdef16c16b,
    zendnn_aBdefc4b,
    zendnn_aBdefc8b,
    zendnn_Abcdef16a,
    zendnn_Abcdef32a,
    zendnn_aBedc16b,
    zendnn_Acb16a,
    zendnn_AcB16a2b,
    zendnn_AcB16a4b,
    zendnn_Acb4a,
    zendnn_Acb8a,
    zendnn_aCBd16b16c,
    zendnn_aCBd16c16b,
    zendnn_aCBde16b16c,
    zendnn_aCBde16c16b,
    zendnn_Acdb16a,
    zendnn_AcdB16a2b,
    zendnn_AcdB16a4b,
    zendnn_Acdb32a,
    zendnn_Acdb4a,
    zendnn_Acdb8a,
    zendnn_Acdeb16a,
    zendnn_AcdeB16a2b,
    zendnn_Acdeb4a,
    zendnn_Acdeb8a,
    zendnn_Adcb16a,
    zendnn_BAc16a16b,
    zendnn_BAc16b16a,
    zendnn_BAcd16a16b,
    zendnn_BAcd16b16a,
    zendnn_aCBd4c8b8c4b,
    zendnn_aCBde4c8b8c4b,
    zendnn_aCBdef4c8b8c4b,
    zendnn_BAcde16a16b,
    zendnn_aCBdef16b16c,
    zendnn_abdfce, ///< permuted 6D tensor
    zendnn_abdefc, ///< permuted 6D tensor
    zendnn_ABc16b32a,
    zendnn_ABc16b64a,
    zendnn_ABc4b32a4b,
    zendnn_ABc4b64a4b,
    zendnn_ABc8b32a2b,
    zendnn_ABc8b64a2b,
    zendnn_AB16b16a,
    zendnn_AB16b32a,
    zendnn_AB16b64a,
    zendnn_AB8b16a2b,
    zendnn_AB8b32a2b,
    zendnn_AB8b64a2b,
    zendnn_AB4b16a4b,
    zendnn_AB4b32a4b,
    zendnn_AB4b64a4b,
    zendnn_AB16b16a4b,
    zendnn_ABcd16b32a,
    zendnn_ABcd16b64a,
    zendnn_ABcd4b32a4b,
    zendnn_ABcd4b64a4b,
    zendnn_ABcd8b32a2b,
    zendnn_ABcd8b64a2b,
    zendnn_ABcde4b32a4b,
    zendnn_ABcde4b64a4b,
    zendnn_ABcde16b16a4b,
    zendnn_ABcde16b16a2b,
    zendnn_ABcde16b32a,
    zendnn_ABcde16b64a,
    zendnn_ABcde8b32a2b,
    zendnn_ABcde8b64a2b,
    zendnn_aBCdef16c16b4c,
    zendnn_aBCdef16c16b2c,
    zendnn_AB32a32b8a4b,
    zendnn_AB8a4b,
    zendnn_AB32a32b8a2b,
    zendnn_AB8a2b,
    zendnn_abDc32d,
    zendnn_abDC32d4c,
    zendnn_abdEc32e,
    zendnn_abdEC32e2c,
    zendnn_abdEC32e4c,
    zendnn_aBdefC16b4c,
    zendnn_AcdeB16a4b,
    zendnn_ABcd16a16b2a,
    zendnn_ABc16a16b2a,
    zendnn_aBCd16b16c2b,
    zendnn_aBCde16b16c2b,
    zendnn_Acb32a,
    zendnn_AcB32a2b,
    zendnn_AcB32a4b,
    zendnn_Acb48a,
    zendnn_AcB48a2b,
    zendnn_AcB48a4b,
    zendnn_Acb64a,
    zendnn_AcB64a2b,
    zendnn_AcB64a4b,
    zendnn_cBa2b,
    zendnn_cBa4b,
    zendnn_aBdc32b,
    zendnn_aBdC32b2c,
    zendnn_aBdC32b4c,
    zendnn_aBdc48b,
    zendnn_aBdC48b2c,
    zendnn_aBdC48b4c,
    zendnn_aBdc64b,
    zendnn_aBdC64b2c,
    zendnn_aBdC64b4c,
    zendnn_adcb,
    zendnn_adCb2c,
    zendnn_adCb4c,
    zendnn_AcdB32a2b,
    zendnn_AcdB32a4b,
    zendnn_Acdb48a,
    zendnn_AcdB48a2b,
    zendnn_AcdB48a4b,
    zendnn_Acdb64a,
    zendnn_AcdB64a2b,
    zendnn_AcdB64a4b,
    zendnn_cdBa2b,
    zendnn_cdBa4b,
    zendnn_aBdeC32b2c,
    zendnn_aBdeC32b4c,
    zendnn_aBdec48b,
    zendnn_aBdeC48b2c,
    zendnn_aBdeC48b4c,
    zendnn_aBdec64b,
    zendnn_aBdeC64b2c,
    zendnn_aBdeC64b4c,
    zendnn_adecb,
    zendnn_adeCb2c,
    zendnn_adeCb4c,
    zendnn_Acdeb32a,
    zendnn_AcdeB32a2b,
    zendnn_AcdeB32a4b,
    zendnn_Acdeb48a,
    zendnn_AcdeB48a2b,
    zendnn_AcdeB48a4b,
    zendnn_Acdeb64a,
    zendnn_AcdeB64a2b,
    zendnn_AcdeB64a4b,
    zendnn_cdeBa2b,
    zendnn_cdeBa4b,
    zendnn_aBdefc32b,
    zendnn_aBdefC32b2c,
    zendnn_aBdefC32b4c,
    zendnn_aBdefc48b,
    zendnn_aBdefC48b2c,
    zendnn_aBdefC48b4c,
    zendnn_aBdefc64b,
    zendnn_aBdefC64b2c,
    zendnn_aBdefC64b4c,
    zendnn_adefcb,
    zendnn_adefCb2c,
    zendnn_adefCb4c,
    zendnn_AB16b32a4b,
    zendnn_AB16b48a4b,
    zendnn_AB16b64a4b,
    zendnn_AB16b16a2b,
    zendnn_AB16b32a2b,
    zendnn_AB16b48a2b,
    zendnn_AB16b64a2b,
    zendnn_ABc16b32a4b,
    zendnn_ABc16b48a4b,
    zendnn_ABc16b64a4b,
    zendnn_ABc16b32a2b,
    zendnn_ABc16b48a2b,
    zendnn_ABc16b64a2b,
    zendnn_ABcd16b32a4b,
    zendnn_ABcd16b48a4b,
    zendnn_ABcd16b64a4b,
    zendnn_ABcd16b32a2b,
    zendnn_ABcd16b48a2b,
    zendnn_ABcd16b64a2b,
    zendnn_ABcde16b32a4b,
    zendnn_ABcde16b48a4b,
    zendnn_ABcde16b64a4b,
    zendnn_ABcde16b32a2b,
    zendnn_ABcde16b48a2b,
    zendnn_ABcde16b64a2b,

    /// Just a sentinel, not real memory format tag. Must be changed after new
    /// format tag is added.
    zendnn_format_tag_last,

    // Aliases

    /// 1D tensor, an alias to #zendnn_a
    zendnn_x = zendnn_a,
    /// 2D CNN activations tensor, an alias to #zendnn_ab
    zendnn_nc = zendnn_ab,
    /// 2D CNN activations tensor, an alias to #zendnn_ba
    zendnn_cn = zendnn_ba,
    /// 2D RNN statistics tensor, an alias to #zendnn_ab
    zendnn_tn = zendnn_ab,
    /// 2D RNN statistics tensor, an alias to #zendnn_ba
    zendnn_nt = zendnn_ba,
    /// 3D CNN activations tensor, an alias to #zendnn_abc
    zendnn_ncw = zendnn_abc,
    /// 3D CNN activations tensor, an alias to #zendnn_acb
    zendnn_nwc = zendnn_acb,
    /// 4D CNN activations tensor, an alias to #zendnn_abcd
    zendnn_nchw = zendnn_abcd,
    /// 4D CNN activations tensor, an alias to #zendnn_acdb
    zendnn_nhwc = zendnn_acdb,
    /// 4D CNN activations tensor, an alias to #zendnn_bcda
    zendnn_chwn = zendnn_bcda,
    /// 5D CNN activations tensor, an alias to #zendnn_abcde
    zendnn_ncdhw = zendnn_abcde,
    /// 5D CNN activations tensor, an alias to #zendnn_acdeb
    zendnn_ndhwc = zendnn_acdeb,

    /// 2D CNN weights tensor, an alias to #zendnn_ab
    zendnn_oi = zendnn_ab,
    /// 2D CNN weights tensor, an alias to #zendnn_ba
    zendnn_io = zendnn_ba,
    /// 3D CNN weights tensor, an alias to #zendnn_abc
    zendnn_oiw = zendnn_abc,
    /// 3D CNN weights tensor, an alias to #zendnn_acb
    zendnn_owi = zendnn_acb,
    /// 3D CNN weights tensor, an alias to #zendnn_cba
    zendnn_wio = zendnn_cba,
    /// 3D CNN weights tensor, an alias to #zendnn_bca
    zendnn_iwo = zendnn_bca,
    /// 4D CNN weights tensor, an alias to #zendnn_abcd
    zendnn_oihw = zendnn_abcd,
    /// 4D CNN weights tensor, an alias to #zendnn_cdba
    zendnn_hwio = zendnn_cdba,
    /// 4D CNN weights tensor, an alias to #zendnn_acdb
    zendnn_ohwi = zendnn_acdb,
    /// 4D CNN weights tensor, an alias to #zendnn_bcda
    zendnn_ihwo = zendnn_bcda,
    /// 4D CNN weights tensor, an alias to #zendnn_bacd
    zendnn_iohw = zendnn_bacd,
    /// 5D CNN weights tensor, an alias to #zendnn_abcde
    zendnn_oidhw = zendnn_abcde,
    /// 5D CNN weights tensor, an alias to #zendnn_bacde
    zendnn_iodhw = zendnn_bacde,
    /// 5D CNN weights tensor, an alias to #zendnn_cdeba
    zendnn_dhwio = zendnn_cdeba,
    /// 5D CNN weights tensor, an alias to #zendnn_acdeb
    zendnn_odhwi = zendnn_acdeb,
    /// 5D CNN weights tensor, an alias to #zendnn_bcdea
    zendnn_idhwo = zendnn_bcdea,

    /// 4D CNN weights tensor (incl. groups), an alias to #zendnn_abcd
    zendnn_goiw = zendnn_abcd,
    /// 4D CNN weights tensor (incl. groups), an alias to #zendnn_abdc
    zendnn_gowi = zendnn_abdc,
    /// 4D CNN weights tensor (incl. groups), an alias to #zendnn_dcab
    zendnn_wigo = zendnn_dcab,
    /// 5D CNN weights tensor (incl. groups), an alias to #zendnn_abcde
    zendnn_goihw = zendnn_abcde,
    /// 5D CNN weights tensor (incl. groups), an alias to #zendnn_abdec
    zendnn_gohwi = zendnn_abdec,
    /// 5D CNN weights tensor (incl. groups), an alias to #zendnn_decab
    zendnn_hwigo = zendnn_decab,
    /// 5D CNN weights tensor (incl. groups), an alias to #zendnn_acbde
    zendnn_giohw = zendnn_acbde,
    /// 6D CNN weights tensor (incl. groups), an alias to #zendnn_abcdef
    zendnn_goidhw = zendnn_abcdef,
    /// 6D CNN weights tensor (incl. groups), an alias to #zendnn_abdefc
    zendnn_godhwi = zendnn_abdefc,
    /// 6D CNN weights tensor (incl. groups), an alias to #zendnn_acbdef
    zendnn_giodhw = zendnn_acbdef,
    /// 6D CNN weights tensor (incl. groups), an alias to #zendnn_defcab
    zendnn_dhwigo = zendnn_defcab,

    /// 3D RNN data tensor in the format (seq_length, batch, input channels),
    /// an alias to #zendnn_abc.
    zendnn_tnc = zendnn_abc,
    /// 3D RNN data tensor in the format (batch, seq_length, input channels),
    /// an alias to #zendnn_bac.
    zendnn_ntc = zendnn_bac,
    /// 4D RNN states tensor in the format (num_layers, num_directions,
    /// batch, state channels), an alias to #zendnn_abcd.
    zendnn_ldnc = zendnn_abcd,
    /// 5D RNN weights tensor in the format (num_layers, num_directions,
    /// input_channels, num_gates, output_channels), an alias to #zendnn_abcde.
    ///
    ///  - For LSTM cells, the gates order is input, forget, candidate
    ///    and output gate.
    ///  - For GRU cells, the gates order is update, reset and output gate.
    zendnn_ldigo = zendnn_abcde,
    /// 5D RNN weights tensor in the format (num_layers, num_directions,
    /// num_gates, output_channels, input_channels), an alias to #zendnn_abdec.
    ///
    ///  - For LSTM cells, the gates order is input, forget, candidate
    ///    and output gate.
    ///  - For GRU cells, the gates order is update, reset and output gate.
    zendnn_ldgoi = zendnn_abdec,
    /// 4D LSTM projection tensor in the format (num_layers, num_directions,
    /// num_channels_in_hidden_state, num_channels_in_recurrent_projection),
    /// an alias to #zendnn_abcd.
    zendnn_ldio = zendnn_abcd,
    /// 4D LSTM projection tensor in the format (num_layers, num_directions,
    /// num_channels_in_recurrent_projection, num_channels_in_hidden_state),
    /// an alias to #zendnn_abdc.
    zendnn_ldoi = zendnn_abdc,
    /// 4D RNN bias tensor in the format (num_layers, num_directions,
    /// num_gates, output_channels), an alias to #zendnn_abcd.
    ///
    ///  - For LSTM cells, the gates order is input, forget, candidate
    ///    and output gate.
    ///  - For GRU cells, the gates order is update, reset and output gate.
    zendnn_ldgo = zendnn_abcd,
    /// 5D LSTM projection tensor
    zendnn_ldOi32o = zendnn_abDc32d,
    zendnn_ldOI32o4i = zendnn_abDC32d4c,
    /// 6D RNN weights tensor
    zendnn_ldgOi32o = zendnn_abdEc32e,
    zendnn_ldgOI32o2i = zendnn_abdEC32e2c,
    zendnn_ldgOI32o4i = zendnn_abdEC32e4c,

    // Opaque data types, are not to be used explicitly

    // data
    /// 5D CNN activations tensor blocked by channels with block size 32,
    /// an alias to #zendnn_aBcde32b
    zendnn_nCdhw32c = zendnn_aBcde32b,
    /// 5D CNN activations tensor blocked by channels with block size 16,
    /// an alias to #zendnn_aBcde16b
    zendnn_nCdhw16c = zendnn_aBcde16b,
    /// 5D CNN activations tensor blocked by channels with block size 4,
    /// an alias to #zendnn_aBcde4b
    zendnn_nCdhw4c = zendnn_aBcde4b,
    /// 5D CNN activations tensor blocked by channels with block size 8,
    /// an alias to #zendnn_aBcde8b
    zendnn_nCdhw8c = zendnn_aBcde8b,
    /// 4D CNN activations tensor blocked by channels with block size 32,
    /// an alias to #zendnn_aBcd32b
    zendnn_nChw32c = zendnn_aBcd32b,
    /// 4D CNN activations tensor blocked by channels with block size 16,
    /// an alias to #zendnn_aBcd16b
    zendnn_nChw16c = zendnn_aBcd16b,
    /// 4D CNN activations tensor blocked by channels with block size 4,
    /// an alias to #zendnn_aBcd4b
    zendnn_nChw4c = zendnn_aBcd4b,
    /// 4D CNN activations tensor blocked by channels with block size 8,
    /// an alias to #zendnn_aBcd8b
    zendnn_nChw8c = zendnn_aBcd8b,
    /// 3D CNN activations tensor blocked by channels with block size 32,
    /// an alias to #zendnn_aBc32b
    zendnn_nCw32c = zendnn_aBc32b,
    /// 3D CNN activations tensor blocked by channels with block size 16,
    /// an alias to #zendnn_aBc16b
    zendnn_nCw16c = zendnn_aBc16b,
    /// 3D CNN activations tensor blocked by channels with block size 4,
    /// an alias to #zendnn_aBc4b
    zendnn_nCw4c = zendnn_aBc4b,
    /// 3D CNN activations tensor blocked by channels with block size 8,
    /// an alias to #zendnn_aBc8b
    zendnn_nCw8c = zendnn_aBc8b,
    zendnn_NCw16n16c = zendnn_ABc16a16b,
    zendnn_NCdhw16n16c = zendnn_ABcde16a16b,
    zendnn_NChw16n16c = zendnn_ABcd16a16b,
    zendnn_NCw32n32c = zendnn_ABc32a32b,
    zendnn_NChw32n32c = zendnn_ABcd32a32b,
    zendnn_NCdhw32n32c = zendnn_ABcde32a32b,

    // weights, 2D
    zendnn_OI16i16o = zendnn_AB16b16a,
    zendnn_OI16i32o = zendnn_AB16b32a,
    zendnn_OI16i64o = zendnn_AB16b64a,
    zendnn_OI8i16o2i = zendnn_AB8b16a2b,
    zendnn_OI8i32o2i = zendnn_AB8b32a2b,
    zendnn_OI8i64o2i = zendnn_AB8b64a2b,
    zendnn_OI4i16o4i = zendnn_AB4b16a4b,
    zendnn_OI4i32o4i = zendnn_AB4b32a4b,
    zendnn_OI4i64o4i = zendnn_AB4b64a4b,
    zendnn_OI16i16o4i = zendnn_AB16b16a4b,
    // weights, 3D
    zendnn_IOw16o16i = zendnn_BAc16a16b,
    zendnn_IOw16i16o = zendnn_BAc16b16a,
    zendnn_OIw16i16o = zendnn_ABc16b16a,
    zendnn_OIw16i32o = zendnn_ABc16b32a,
    zendnn_OIw16i64o = zendnn_ABc16b64a,
    zendnn_OIw16o16i = zendnn_ABc16a16b,
    zendnn_Oiw16o = zendnn_Abc16a,
    zendnn_OIw4i16o4i = zendnn_ABc4b16a4b,
    zendnn_OIw4i32o4i = zendnn_ABc4b32a4b,
    zendnn_OIw4i64o4i = zendnn_ABc4b64a4b,
    zendnn_OIw2i8o4i = zendnn_ABc2b8a4b,
    zendnn_OIw16i16o4i = zendnn_ABc16b16a4b,
    zendnn_OIw16i16o2i = zendnn_ABc16b16a2b,
    zendnn_OIw16o16i2o = zendnn_ABc16a16b2a,
    zendnn_OIw4i4o = zendnn_ABc4b4a,
    zendnn_OIw4o4i = zendnn_ABc4a4b,
    zendnn_Oiw4o = zendnn_Abc4a,
    zendnn_OIw8i16o2i = zendnn_ABc8b16a2b,
    zendnn_OIw8i32o2i = zendnn_ABc8b32a2b,
    zendnn_OIw8i64o2i = zendnn_ABc8b64a2b,
    zendnn_OIw8i8o = zendnn_ABc8b8a,
    zendnn_OIw8o16i2o = zendnn_ABc8a16b2a,
    zendnn_IOw8o16i2o = zendnn_BAc8a16b2a,
    zendnn_OIw8o8i = zendnn_ABc8a8b,
    zendnn_OIw8o4i = zendnn_ABc8a4b,
    zendnn_Owi16o = zendnn_Acb16a,
    zendnn_OwI16o2i = zendnn_AcB16a2b,
    zendnn_OwI16o4i = zendnn_AcB16a4b,
    zendnn_Owi4o = zendnn_Acb4a,
    zendnn_Owi8o = zendnn_Acb8a,

    // weights, 4D
    zendnn_IOhw16i16o = zendnn_BAcd16b16a,
    zendnn_IOhw16o16i = zendnn_BAcd16a16b,
    zendnn_Ohwi16o = zendnn_Acdb16a,
    zendnn_OhwI16o2i = zendnn_AcdB16a2b,
    zendnn_OhwI16o4i = zendnn_AcdB16a4b,
    zendnn_Ohwi32o = zendnn_Acdb32a,
    zendnn_Ohwi4o = zendnn_Acdb4a,
    zendnn_Ohwi8o = zendnn_Acdb8a,
    zendnn_OIhw16i16o = zendnn_ABcd16b16a,
    zendnn_OIhw16i32o = zendnn_ABcd16b32a,
    zendnn_OIhw16i64o = zendnn_ABcd16b64a,
    zendnn_OIhw16o16i = zendnn_ABcd16a16b,
    zendnn_Oihw16o = zendnn_Abcd16a,
    zendnn_OIhw4i16o4i = zendnn_ABcd4b16a4b,
    zendnn_OIhw4i32o4i = zendnn_ABcd4b32a4b,
    zendnn_OIhw4i64o4i = zendnn_ABcd4b64a4b,
    zendnn_OIhw16i16o4i = zendnn_ABcd16b16a4b,
    zendnn_OIhw16i16o2i = zendnn_ABcd16b16a2b,
    zendnn_OIhw16o16i2o = zendnn_ABcd16a16b2a,
    zendnn_OIhw4i4o = zendnn_ABcd4b4a,
    zendnn_OIhw4o4i = zendnn_ABcd4a4b,
    zendnn_Oihw4o = zendnn_Abcd4a,
    zendnn_OIhw8i16o2i = zendnn_ABcd8b16a2b,
    zendnn_OIhw8i32o2i = zendnn_ABcd8b32a2b,
    zendnn_OIhw8i64o2i = zendnn_ABcd8b64a2b,
    zendnn_OIhw8i8o = zendnn_ABcd8b8a,
    zendnn_OIhw8o16i2o = zendnn_ABcd8a16b2a,
    zendnn_OIhw2i8o4i = zendnn_ABcd2b8a4b,
    zendnn_IOhw8o16i2o = zendnn_BAcd8a16b2a,
    zendnn_OIhw8o8i = zendnn_ABcd8a8b,
    zendnn_OIhw8o4i = zendnn_ABcd8a4b,
    zendnn_Owhi16o = zendnn_Adcb16a,

    // weights, 5D
    zendnn_Odhwi16o = zendnn_Acdeb16a,
    zendnn_OdhwI16o2i = zendnn_AcdeB16a2b,
    zendnn_OdhwI16o4i = zendnn_AcdeB16a4b,
    zendnn_Odhwi4o = zendnn_Acdeb4a,
    zendnn_Odhwi8o = zendnn_Acdeb8a,
    zendnn_OIdhw16i16o = zendnn_ABcde16b16a,
    zendnn_OIdhw16i32o = zendnn_ABcde16b32a,
    zendnn_OIdhw16i64o = zendnn_ABcde16b64a,
    zendnn_OIdhw16o16i = zendnn_ABcde16a16b,
    zendnn_Oidhw16o = zendnn_Abcde16a,
    zendnn_OIdhw4i4o = zendnn_ABcde4b4a,
    zendnn_OIdhw4o4i = zendnn_ABcde4a4b,
    zendnn_Oidhw4o = zendnn_Abcde4a,
    zendnn_OIdhw8i16o2i = zendnn_ABcde8b16a2b,
    zendnn_OIdhw8i32o2i = zendnn_ABcde8b32a2b,
    zendnn_OIdhw8i64o2i = zendnn_ABcde8b64a2b,
    zendnn_OIdhw8i8o = zendnn_ABcde8b8a,
    zendnn_OIdhw8o16i2o = zendnn_ABcde8a16b2a,
    zendnn_IOdhw8o16i2o = zendnn_BAcde8a16b2a,
    zendnn_OIdhw4i16o4i = zendnn_ABcde4b16a4b,
    zendnn_OIdhw4i32o4i = zendnn_ABcde4b32a4b,
    zendnn_OIdhw4i64o4i = zendnn_ABcde4b64a4b,
    zendnn_OIdhw16i16o4i = zendnn_ABcde16b16a4b,
    zendnn_OIdhw16i16o2i = zendnn_ABcde16b16a2b,
    zendnn_OIdhw2i8o4i = zendnn_ABcde2b8a4b,
    zendnn_OIdhw8o8i = zendnn_ABcde8a8b,
    zendnn_OIdhw8o4i = zendnn_ABcde8a4b,
    zendnn_IOdhw16i16o = zendnn_BAcde16b16a,
    zendnn_OIdhw4o8i8o4i = zendnn_ABcde4a8b8a4b,
    zendnn_IOdhw16o16i = zendnn_BAcde16a16b,

    // weights w/ groups, 3D
    zendnn_Goiw16g = zendnn_Abcd16a,
    zendnn_Goiw8g = zendnn_Abcd8a,
    zendnn_Goiw4g = zendnn_Abcd4a,
    zendnn_gIOw16o16i = zendnn_aCBd16b16c,
    zendnn_gIOw16i16o = zendnn_aCBd16c16b,
    zendnn_gOIw16i16o = zendnn_aBCd16c16b,
    zendnn_gOIw16o16i = zendnn_aBCd16b16c,
    zendnn_gOiw16o = zendnn_aBcd16b,
    zendnn_gOIw4i16o4i = zendnn_aBCd4c16b4c,
    zendnn_gOIw2i8o4i = zendnn_aBCd2c8b4c,
    zendnn_gOIw16i16o4i = zendnn_aBCd16c16b4c,
    zendnn_gOIw16i16o2i = zendnn_aBCd16c16b2c,
    zendnn_gOIw16o16i2o = zendnn_aBCd16b16c2b,
    zendnn_gOIw4i4o = zendnn_aBCd4c4b,
    zendnn_gOIw4o4i = zendnn_aBCd4b4c,
    zendnn_gOiw4o = zendnn_aBcd4b,
    zendnn_gOIw8i16o2i = zendnn_aBCd8c16b2c,
    zendnn_gOIw8i8o = zendnn_aBCd8c8b,
    zendnn_gOIw8o16i2o = zendnn_aBCd8b16c2b,
    zendnn_gIOw8o16i2o = zendnn_aCBd8b16c2b,
    zendnn_gOIw8o8i = zendnn_aBCd8b8c,
    zendnn_gOIw8o4i = zendnn_aBCd8b4c,
    zendnn_gOwi16o = zendnn_aBdc16b,
    zendnn_gOwI16o2i = zendnn_aBdC16b2c,
    zendnn_gOwI16o4i = zendnn_aBdC16b4c,
    zendnn_gOwi4o = zendnn_aBdc4b,
    zendnn_gOwi8o = zendnn_aBdc8b,
    zendnn_Goiw32g = zendnn_Abcd32a,
    zendnn_gOIw2i4o2i = zendnn_aBCd2c4b2c,
    zendnn_gOIw2o4i2o = zendnn_aBCd2b4c2b,
    zendnn_gOIw4i8o2i = zendnn_aBCd4c8b2c,
    zendnn_gOIw4o8i2o = zendnn_aBCd4b8c2b,

    // weights w/ groups, 4D
    zendnn_gIOhw16i16o = zendnn_aCBde16c16b,
    zendnn_gIOhw16o16i = zendnn_aCBde16b16c,
    zendnn_gOhwi16o = zendnn_aBdec16b,
    zendnn_gOhwI16o2i = zendnn_aBdeC16b2c,
    zendnn_gOhwI16o4i = zendnn_aBdeC16b4c,
    zendnn_gOhwi32o = zendnn_aBdec32b,
    zendnn_gOhwi4o = zendnn_aBdec4b,
    zendnn_gOhwi8o = zendnn_aBdec8b,
    zendnn_Goihw16g = zendnn_Abcde16a,
    zendnn_gOIhw16i16o = zendnn_aBCde16c16b,
    zendnn_gOIhw16o16i = zendnn_aBCde16b16c,
    zendnn_gOihw16o = zendnn_aBcde16b,
    zendnn_gOIhw2i8o4i = zendnn_aBCde2c8b4c,
    zendnn_gOIhw4i16o4i = zendnn_aBCde4c16b4c,
    zendnn_gOIhw16i16o4i = zendnn_aBCde16c16b4c,
    zendnn_gOIhw16i16o2i = zendnn_aBCde16c16b2c,
    zendnn_gOIhw16o16i2o = zendnn_aBCde16b16c2b,
    zendnn_gOIhw4i4o = zendnn_aBCde4c4b,
    zendnn_gOIhw4o4i = zendnn_aBCde4b4c,
    zendnn_gOihw4o = zendnn_aBcde4b,
    zendnn_Goihw8g = zendnn_Abcde8a,
    zendnn_Goihw4g = zendnn_Abcde4a,
    zendnn_gOIhw8i16o2i = zendnn_aBCde8c16b2c,
    zendnn_gOIhw8i8o = zendnn_aBCde8c8b,
    zendnn_gOIhw8o16i2o = zendnn_aBCde8b16c2b,
    zendnn_gIOhw8o16i2o = zendnn_aCBde8b16c2b,
    zendnn_gOIhw8o8i = zendnn_aBCde8b8c,
    zendnn_gOIhw8o4i = zendnn_aBCde8b4c,
    zendnn_Goihw32g = zendnn_Abcde32a,
    zendnn_gOwhi16o = zendnn_aBedc16b,

    zendnn_OIw4o8i8o4i = zendnn_ABc4a8b8a4b,
    zendnn_OIhw4o8i8o4i = zendnn_ABcd4a8b8a4b,
    zendnn_IOw4i8o8i4o = zendnn_BAc4b8a8b4a,
    zendnn_IOhw4i8o8i4o = zendnn_BAcd4b8a8b4a,
    zendnn_IOdhw4i8o8i4o = zendnn_BAcde4b8a8b4a,

    zendnn_OIhw2o8i8o2i = zendnn_ABcd2a8b8a2b,
    zendnn_gOIw4o8i8o4i = zendnn_aBCd4b8c8b4c,
    zendnn_gOIhw4o8i8o4i = zendnn_aBCde4b8c8b4c,
    zendnn_gOIdhw4o8i8o4i = zendnn_aBCdef4b8c8b4c,
    zendnn_gIOw4i8o8i4o = zendnn_aCBd4c8b8c4b,
    zendnn_gIOhw4i8o8i4o = zendnn_aCBde4c8b8c4b,
    zendnn_gIOdhw4i8o8i4o = zendnn_aCBdef4c8b8c4b,
    zendnn_gOIhw2o8i8o2i = zendnn_aBCde2b8c8b2c,
    zendnn_gOIhw2i4o2i = zendnn_aBCde2c4b2c,
    zendnn_gOIhw2o4i2o = zendnn_aBCde2b4c2b,
    zendnn_gOIhw4i8o2i = zendnn_aBCde4c8b2c,
    zendnn_gOIhw4o8i2o = zendnn_aBCde4b8c2b,

    // weights w/ groups, 6D
    zendnn_gIOdhw16i16o = zendnn_aCBdef16c16b,
    zendnn_gIOdhw16o16i = zendnn_aCBdef16b16c,
    zendnn_gOdhwi16o = zendnn_aBdefc16b,
    zendnn_gOdhwI16o2i = zendnn_aBdefC16b2c,
    zendnn_gOdhwI16o4i = zendnn_aBdefC16b4c,
    zendnn_gOdhwi4o = zendnn_aBdefc4b,
    zendnn_gOdhwi8o = zendnn_aBdefc8b,
    zendnn_gOIdhw16i16o = zendnn_aBCdef16c16b,
    zendnn_gOIdhw4i16o4i = zendnn_aBCdef4c16b4c,
    zendnn_gOIdhw16i16o4i = zendnn_aBCdef16c16b4c,
    zendnn_gOIdhw2i8o4i = zendnn_aBCdef2c8b4c,
    zendnn_gOIdhw16i16o2i = zendnn_aBCdef16c16b2c,
    zendnn_gOIdhw16o16i = zendnn_aBCdef16b16c,
    zendnn_gOidhw16o = zendnn_aBcdef16b,
    zendnn_gOIdhw4i4o = zendnn_aBCdef4c4b,
    zendnn_gOIdhw4o4i = zendnn_aBCdef4b4c,
    zendnn_gOidhw4o = zendnn_aBcdef4b,
    zendnn_gOIdhw8i16o2i = zendnn_aBCdef8c16b2c,
    zendnn_gOIdhw8i8o = zendnn_aBCdef8c8b,
    zendnn_gOIdhw8o16i2o = zendnn_aBCdef8b16c2b,
    zendnn_gIOdhw8o16i2o = zendnn_aCBdef8b16c2b,
    zendnn_gOIdhw8o8i = zendnn_aBCdef8b8c,
    zendnn_gOIdhw8o4i = zendnn_aBCdef8b4c,
    zendnn_Goidhw16g = zendnn_Abcdef16a,
    zendnn_Goidhw32g = zendnn_Abcdef32a,
    zendnn_gOIdhw2i4o2i = zendnn_aBCdef2c4b2c,
    zendnn_gOIdhw4i8o2i = zendnn_aBCdef4c8b2c,
    zendnn_gOIdhw2o4i2o = zendnn_aBCdef2b4c2b,
    zendnn_gOIdhw4o8i2o = zendnn_aBCdef4b8c2b,
    // weights, 3D
    zendnn_Owi32o = zendnn_Acb32a,
    zendnn_OwI32o2i = zendnn_AcB32a2b,
    zendnn_OwI32o4i = zendnn_AcB32a4b,
    zendnn_Owi48o = zendnn_Acb48a,
    zendnn_OwI48o2i = zendnn_AcB48a2b,
    zendnn_OwI48o4i = zendnn_AcB48a4b,
    zendnn_Owi64o = zendnn_Acb64a,
    zendnn_OwI64o2i = zendnn_AcB64a2b,
    zendnn_OwI64o4i = zendnn_AcB64a4b,
    zendnn_wIo2i = zendnn_cBa2b,
    zendnn_wIo4i = zendnn_cBa4b,
    zendnn_gOwi32o = zendnn_aBdc32b,
    zendnn_gOwI32o2i = zendnn_aBdC32b2c,
    zendnn_gOwI32o4i = zendnn_aBdC32b4c,
    zendnn_gOwi48o = zendnn_aBdc48b,
    zendnn_gOwI48o2i = zendnn_aBdC48b2c,
    zendnn_gOwI48o4i = zendnn_aBdC48b4c,
    zendnn_gOwi64o = zendnn_aBdc64b,
    zendnn_gOwI64o2i = zendnn_aBdC64b2c,
    zendnn_gOwI64o4i = zendnn_aBdC64b4c,
    zendnn_gwio = zendnn_adcb,
    zendnn_gwIo2i = zendnn_adCb2c,
    zendnn_gwIo4i = zendnn_adCb4c,
    // weights, 4D
    zendnn_OhwI32o = zendnn_Acdb32a,
    zendnn_OhwI32o2i = zendnn_AcdB32a2b,
    zendnn_OhwI32o4i = zendnn_AcdB32a4b,
    zendnn_Ohwi48o = zendnn_Acdb48a,
    zendnn_OhwI48o2i = zendnn_AcdB48a2b,
    zendnn_OhwI48o4i = zendnn_AcdB48a4b,
    zendnn_Ohwi64o = zendnn_Acdb64a,
    zendnn_OhwI64o2i = zendnn_AcdB64a2b,
    zendnn_OhwI64o4i = zendnn_AcdB64a4b,
    zendnn_hwIo2i = zendnn_cdBa2b,
    zendnn_hwIo4i = zendnn_cdBa4b,
    zendnn_gOhwI32o = zendnn_aBdec32b,
    zendnn_gOhwI32o2i = zendnn_aBdeC32b2c,
    zendnn_gOhwI32o4i = zendnn_aBdeC32b4c,
    zendnn_gOhwi48o = zendnn_aBdec48b,
    zendnn_gOhwI48o2i = zendnn_aBdeC48b2c,
    zendnn_gOhwI48o4i = zendnn_aBdeC48b4c,
    zendnn_gOhwi64o = zendnn_aBdec64b,
    zendnn_gOhwI64o2i = zendnn_aBdeC64b2c,
    zendnn_gOhwI64o4i = zendnn_aBdeC64b4c,
    zendnn_ghwio = zendnn_adecb,
    zendnn_ghwIo2i = zendnn_adeCb2c,
    zendnn_ghwIo4i = zendnn_adeCb4c,
    // weights, 5D
    zendnn_Odhwi32o = zendnn_Acdeb32a,
    zendnn_OdhwI32o2i = zendnn_AcdeB32a2b,
    zendnn_OdhwI32o4i = zendnn_AcdeB32a4b,
    zendnn_Odhwi48o = zendnn_Acdeb48a,
    zendnn_OdhwI48o2i = zendnn_AcdeB48a2b,
    zendnn_OdhwI48o4i = zendnn_AcdeB48a4b,
    zendnn_Odhwi64o = zendnn_Acdeb64a,
    zendnn_OdhwI64o2i = zendnn_AcdeB64a2b,
    zendnn_OdhwI64o4i = zendnn_AcdeB64a4b,
    zendnn_dhwIo2i = zendnn_cdeBa2b,
    zendnn_dhwIo4i = zendnn_cdeBa4b,
    zendnn_gOdhwi32o = zendnn_aBdefc32b,
    zendnn_gOdhwI32o2i = zendnn_aBdefC32b2c,
    zendnn_gOdhwI32o4i = zendnn_aBdefC32b4c,
    zendnn_gOdhwi48o = zendnn_aBdefc48b,
    zendnn_gOdhwI48o2i = zendnn_aBdefC48b2c,
    zendnn_gOdhwI48o4i = zendnn_aBdefC48b4c,
    zendnn_gOdhwi64o = zendnn_aBdefc64b,
    zendnn_gOdhwI64o2i = zendnn_aBdefC64b2c,
    zendnn_gOdhwI64o4i = zendnn_aBdefC64b4c,
    zendnn_gdhwio = zendnn_adefcb,
    zendnn_gdhwIo2i = zendnn_adefCb2c,
    zendnn_gdhwIo4i = zendnn_adefCb4c,
    zendnn_OI16i32o4i = zendnn_AB16b32a4b,
    zendnn_OI16i48o4i = zendnn_AB16b48a4b,
    zendnn_OI16i64o4i = zendnn_AB16b64a4b,
    zendnn_OI16i16o2i = zendnn_AB16b16a2b,
    zendnn_OI16i32o2i = zendnn_AB16b32a2b,
    zendnn_OI16i48o2i = zendnn_AB16b48a2b,
    zendnn_OI16i64o2i = zendnn_AB16b64a2b,
    zendnn_OIw16i32o4i = zendnn_ABc16b32a4b,
    zendnn_OIw16i48o4i = zendnn_ABc16b48a4b,
    zendnn_OIw16i64o4i = zendnn_ABc16b64a4b,
    zendnn_OIw16i32o2i = zendnn_ABc16b32a2b,
    zendnn_OIw16i48o2i = zendnn_ABc16b48a2b,
    zendnn_OIw16i64o2i = zendnn_ABc16b64a2b,
    zendnn_OIhw16i32o4i = zendnn_ABcd16b32a4b,
    zendnn_OIhw16i48o4i = zendnn_ABcd16b48a4b,
    zendnn_OIhw16i64o4i = zendnn_ABcd16b64a4b,
    zendnn_OIhw16i32o2i = zendnn_ABcd16b32a2b,
    zendnn_OIhw16i48o2i = zendnn_ABcd16b48a2b,
    zendnn_OIhw16i64o2i = zendnn_ABcd16b64a2b,
    zendnn_OIdhw16i32o4i = zendnn_ABcde16b32a4b,
    zendnn_OIdhw16i48o4i = zendnn_ABcde16b48a4b,
    zendnn_OIdhw16i64o4i = zendnn_ABcde16b64a4b,
    zendnn_OIdhw16i32o2i = zendnn_ABcde16b32a2b,
    zendnn_OIdhw16i48o2i = zendnn_ABcde16b48a2b,
    zendnn_OIdhw16i64o2i = zendnn_ABcde16b64a2b,
    /// 4D CNN activations tensor, an alias to #zendnn_cdba
    zendnn_hwcn = zendnn_cdba,
} zendnn_format_tag_t;

/// @} zendnn_api_memory

/// @addtogroup zendnn_api_primitives
/// @{
/// @addtogroup zendnn_api_primitives_common
/// @{

/// Kinds of propagation.
typedef enum {
    // TODO: suggest renames
    /// Undefined propagation type.
    zendnn_prop_kind_undef = 0,
    /// Forward data propagation (training mode). In this mode primitives
    /// perform computations necessary for subsequent backward propagation.
    zendnn_forward_training = 64,
    /// Forward data propagation (inference mode). In this mode primitives
    /// perform only computations that are necessary for inference and omit
    /// computations that are necessary only for backward propagation.
    zendnn_forward_inference = 96,
    /// Forward data propagation (alias for @c zendnn_forward_inference).
    zendnn_forward_scoring = zendnn_forward_inference,
    /// Forward data propagation (alias for @c zendnn_forward_training).
    zendnn_forward = zendnn_forward_training,
    /// Backward propagation (with respect to all parameters).
    zendnn_backward = 128,
    /// Backward data propagation.
    zendnn_backward_data = 160,
    /// Backward weights propagation.
    zendnn_backward_weights = 192,
    /// Backward bias propagation.
    zendnn_backward_bias = 193,
} zendnn_prop_kind_t;

/// Kinds of primitives. Used to implement a way to extend the library with new
/// primitives without changing the ABI.
typedef enum {
    /// Undefined primitive
    zendnn_undefined_primitive,
    /// A reorder primitive.
    zendnn_reorder,
    /// A shuffle primitive.
    zendnn_shuffle,
    /// A (out-of-place) concat primitive.
    zendnn_concat,
    /// A sum primitive.
    zendnn_sum,
    /// A convolution primitive.
    zendnn_convolution,
    /// A deconvolution primitive.
    zendnn_deconvolution,
    /// An element-wise primitive.
    zendnn_eltwise,
    /// A softmax primitive.
    zendnn_softmax,
    /// A pooling primitive.
    zendnn_pooling,
    /// An LRN primitive.
    zendnn_lrn,
    /// A batch normalization primitive.
    zendnn_batch_normalization,
    /// A layer normalization primitive.
    zendnn_layer_normalization,
    /// An inner product primitive.
    zendnn_inner_product,
    /// A rnn primitive.
    zendnn_rnn,
    /// A matrix multiplication primitive (internal).
    zendnn_gemm,
    /// A binary primitive.
    zendnn_binary,
    /// A logsoftmax primitive.
    zendnn_logsoftmax,
    /// A matrix multiplication primitive.
    zendnn_matmul,
    /// A resampling primitive.
    zendnn_resampling,
    /// A pooling version 2 primitive (pooling with dilation support).
    zendnn_pooling_v2,
    /// A reduction primitive.
    zendnn_reduction,
    /// A PReLU primitive.
    zendnn_prelu,

    /* add new primitive */
    /// An embedding bag primitive.
    zendnn_embedding_bag,

    /// Parameter to allow internal only primitives without undefined behavior.
    /// This parameter is chosen to be valid for so long as sizeof(int) >= 2.
    zendnn_primitive_kind_max = 0x7fff,
} zendnn_primitive_kind_t;

/// Kinds of algorithms.
typedef enum {
    zendnn_alg_kind_undef,
    /// Direct convolution
    zendnn_convolution_direct = 0x1,
    /// Winograd convolution
    zendnn_convolution_winograd = 0x2,
    /// Convolution algorithm(either direct or Winograd) is chosen just in time
    zendnn_convolution_auto = 0x3,
    /// GEMM convolution
    zendnn_convolution_gemm = 0x4,
    /// Ref convolution
    zendnn_convolution_ref = 0x5,
    /// Direct deconvolution
    zendnn_deconvolution_direct = 0xa,
    /// Winograd deconvolution
    zendnn_deconvolution_winograd = 0xb,
    /// Eltwise: ReLU
    zendnn_eltwise_relu = 0x1f,
    /// Eltwise: hyperbolic tangent non-linearity (tanh)
    zendnn_eltwise_tanh = 0x2f,
    /// Eltwise: exponential linear unit (elu)
    zendnn_eltwise_elu = 0x3f,
    /// Eltwise: square
    zendnn_eltwise_square = 0x4f,
    /// Eltwise: abs
    zendnn_eltwise_abs = 0x5f,
    /// Eltwise: square root
    zendnn_eltwise_sqrt = 0x6f,
    /// Eltwise: linear
    zendnn_eltwise_linear = 0x7f,
    /// Eltwise: bounded_relu
    zendnn_eltwise_bounded_relu = 0x8f,
    /// Eltwise: soft_relu
    zendnn_eltwise_soft_relu = 0x9f,
    /// Eltwise: logistic
    zendnn_eltwise_logistic = 0xaf,
    /// Eltwise: exponent
    zendnn_eltwise_exp = 0xbf,
    /// Eltwise: gelu
    ///
    /// @note Tanh approximation formula is used to approximate
    /// the cumulative distribution function of a Gaussian here
    zendnn_eltwise_gelu_tanh = 0xcf,
    /// Eltwise: tanh-based gelu (alias for zendnn_eltwise_gelu_tanh)
    zendnn_eltwise_gelu = zendnn_eltwise_gelu_tanh,
    /// Eltwise: swish
    zendnn_eltwise_swish = 0xdf,
    /// Eltwise: natural logarithm
    zendnn_eltwise_log = 0xef,
    /// Eltwise: clip
    zendnn_eltwise_clip = 0xff,
    /// Eltwise: clip version 2
    zendnn_eltwise_clip_v2 = 0x10,
    /// Eltwise: pow
    zendnn_eltwise_pow = 0x20,
    /// Eltwise: erf-based gelu
    zendnn_eltwise_gelu_erf = 0x30,
    /// Eltwise: round
    zendnn_eltwise_round = 0x40,
    /// Eltwise: logsigmoid
    zendnn_eltwise_logsigmoid = 0x50,
    /// Eltwise: mish
    zendnn_eltwise_mish = 0x60,
    /// Eltwise: hardswish
    zendnn_eltwise_hardswish = 0x70,
    /// Eltwise: ReLU (dst for backward)
    zendnn_eltwise_relu_use_dst_for_bwd = 0x100,
    /// Eltwise: hyperbolic tangent non-linearity (tanh) (dst for backward)
    zendnn_eltwise_tanh_use_dst_for_bwd = 0x101,
    /// Eltwise: exponential linear unit (elu) (dst for backward)
    zendnn_eltwise_elu_use_dst_for_bwd = 0x102,
    /// Eltwise: square root (dst for backward)
    zendnn_eltwise_sqrt_use_dst_for_bwd = 0x103,
    /// Eltwise: logistic (dst for backward)
    zendnn_eltwise_logistic_use_dst_for_bwd = 0x104,
    /// Eltwise: exp (dst for backward)
    zendnn_eltwise_exp_use_dst_for_bwd = 0x105,
    /// Eltwise: clip version 2 (dst for backward)
    zendnn_eltwise_clip_v2_use_dst_for_bwd = 0x106,
    /// Max pooling
    zendnn_pooling_max = 0x1ff,
    /// Average pooling include padding
    zendnn_pooling_avg_include_padding = 0x2ff,
    /// Average pooling exclude padding
    zendnn_pooling_avg_exclude_padding = 0x3ff,
    /// Average pooling (alias for #zendnn_pooling_avg_exclude_padding)
    zendnn_pooling_avg = zendnn_pooling_avg_exclude_padding,
    /// Local response normalization (LRN) across multiple channels
    zendnn_lrn_across_channels = 0xaff,
    /// LRN within a single channel
    zendnn_lrn_within_channel = 0xbff,
    /// RNN cell
    zendnn_vanilla_rnn = 0x1fff,
    /// LSTM cell
    zendnn_vanilla_lstm = 0x2fff,
    /// GRU cell
    zendnn_vanilla_gru = 0x3fff,
    /// GRU cell with linear before reset
    ///
    /// Modification of original GRU cell. Differs from #zendnn_vanilla_gru
    /// in how the new memory gate is calculated:
    /// \f[ c_t = tanh(W_c*x_t + b_{c_x} + r_t*(U_c*h_{t-1}+b_{c_h})) \f]
    /// Primitive expects 4 biases on input:
    /// \f$[b_{u}, b_{r}, b_{c_x}, b_{c_h}]\f$
    zendnn_lbr_gru = 0x4fff,
    /// Binary add
    zendnn_binary_add = 0x1fff0,
    /// Binary mul
    zendnn_binary_mul = 0x1fff1,
    /// Binary max
    zendnn_binary_max = 0x1fff2,
    /// Binary min
    zendnn_binary_min = 0x1fff3,
    /// Binary div
    zendnn_binary_div = 0x1fff4,
    /// Binary sub
    zendnn_binary_sub = 0x1fff5,
    /// Binary greater or equal
    zendnn_binary_ge = 0x1fff6,
    /// Binary greater than
    zendnn_binary_gt = 0x1fff7,
    /// Binary less or equal
    zendnn_binary_le = 0x1fff8,
    /// Binary less than
    zendnn_binary_lt = 0x1fff9,
    /// Binary equal
    zendnn_binary_eq = 0x1fffa,
    /// Binary not equal
    zendnn_binary_ne = 0x1fffb,
    /// Nearest Neighbor Resampling Method
    zendnn_resampling_nearest = 0x2fff0,
    /// Linear Resampling Method
    zendnn_resampling_linear = 0x2fff1,
    /// Reduction using max
    zendnn_reduction_max,
    /// Reduction using min
    zendnn_reduction_min,
    /// Reduction using sum
    zendnn_reduction_sum,
    /// Reduction using mul
    zendnn_reduction_mul,
    /// Reduction using mean
    zendnn_reduction_mean,
    /// Reduction using lp norm
    zendnn_reduction_norm_lp_max,
    /// Reduction using lp norm
    zendnn_reduction_norm_lp_sum,
    /// Reduction using lp norm without final pth-root
    zendnn_reduction_norm_lp_power_p_max,
    /// Reduction using lp norm without final pth-root
    zendnn_reduction_norm_lp_power_p_sum,

    /* add new primitive */
    zendnn_embedding_bag_sum  = 0x3000,
    zendnn_embedding_bag_mean,
    zendnn_embedding_bag_max,
} zendnn_alg_kind_t;

/// Flags for normalization primitives.
typedef enum {
    /// Use no normalization flags
    ///
    /// If specified
    ///  - on forward training propagation mean and variance are computed and
    ///    stored as output
    ///  - on backward propagation compute full derivative wrt data
    ///  - on backward propagation prop_kind == #zendnn_backward_data has the same
    ///    behavior as prop_kind == #zendnn_backward
    zendnn_normalization_flags_none = 0x0U,

    /// Use global statistics
    ///
    /// If specified
    ///  - on forward propagation use mean and variance provided by user (input)
    ///  - on backward propagation reduces the amount of computations, since
    ///    mean and variance are considered as constants
    ///
    ///  If not specified:
    ///   - on forward propagation mean and variance are computed and stored as
    ///     output
    ///   - on backward propagation compute full derivative wrt data
    zendnn_use_global_stats = 0x1U,

    /// Use scale and shift parameters
    ///
    /// If specified:
    ///  - on forward propagation use scale and shift (aka scale and bias) for
    ///    the batch normalization results
    ///  - on backward propagation (for prop_kind == #zendnn_backward) compute
    ///    diff wrt scale and shift (hence one extra output used)
    ///
    /// If no specified:
    ///  - on backward propagation prop_kind == #zendnn_backward_data has the
    ///    same behavior as prop_kind == #zendnn_backward
    zendnn_use_scaleshift = 0x2U,

    /// Fuse with ReLU
    ///
    /// The flag implies negative slope being 0. On training this is the only
    /// configuration supported. For inference, to use non-zero negative slope
    /// consider using @ref dev_guide_attributes_post_ops.
    ///
    /// If specified:
    ///  - on inference this option behaves the same as if the primitive were
    ///    fused with ReLU using post ops API with zero negative slope.
    ///  - on training primitive requires workspace (required to be able to
    ///    perform backward pass)
    zendnn_fuse_norm_relu = 0x4U,
} zendnn_normalization_flags_t;

/// @} zendnn_api_primitives_common
/// @} zendnn_api_primitives

/// @addtogroup zendnn_api_memory
/// @{

/// Maximum number of dimensions a tensor can have. Only restricts the amount
/// of space used for the tensor description. Individual computational
/// primitives may support only tensors of certain dimensions.
#define ZENDNN_MAX_NDIMS 12

/// A wildcard value for dimensions that are unknown at a primitive creation
/// time.
#define ZENDNN_RUNTIME_DIM_VAL INT64_MIN

/// A `size_t` counterpart of the ZENDNN_RUNTIME_DIM_VAL.
/// For instance, this value is returned by zendnn_memory_desc_get_size() if
/// either of the dimensions or strides equal to #ZENDNN_RUNTIME_DIM_VAL.
#define ZENDNN_RUNTIME_SIZE_VAL ((size_t)ZENDNN_RUNTIME_DIM_VAL)

/// @cond DO_NOT_DOCUMENT_THIS
/// Hex representation for a **special** quiet NAN (!= NAN from math.h)
static const union {
    unsigned u;
    float f;
} ZENDNN_RUNTIME_F32_VAL_REP = {0x7fc000d0};
/// @endcond

/// A wildcard value for floating point values that are unknown at a primitive
/// creation time.
#define ZENDNN_RUNTIME_F32_VAL (ZENDNN_RUNTIME_F32_VAL_REP.f)

/// @cond DO_NOT_DOCUMENT_THIS
static const int ZENDNN_RUNTIME_S32_VAL_REP = INT32_MIN;
/// @endcond

/// A wildcard value for int32_t values that are unknown at a primitive creation
/// time.
#define ZENDNN_RUNTIME_S32_VAL ZENDNN_RUNTIME_S32_VAL_REP

/// A type to describe tensor dimension.
typedef int64_t zendnn_dim_t;

/// A type to describe tensor dimensions.
typedef zendnn_dim_t zendnn_dims_t[ZENDNN_MAX_NDIMS];

/// Generic description of blocked data layout for most memory formats.
///
/// @sa @ref dev_guide_understanding_memory_formats
typedef struct {
    /// The strides between the outermost blocks.
    /// In case of plain (non-blocked) formats the strides between dimensions.
    zendnn_dims_t strides;
    // Innermost section
    // ASSUMPTION: the innermost blocks are always dense
    /// The number of innermost blocks, e.g. 3 in case of `OIhw_4i16o4i_`
    int inner_nblks;
    /// The size of the blocks, e.g. `{4, 16, 4}` in case of `OIhw_4i16o4i`
    zendnn_dims_t inner_blks;
    /// The logical indices of the blocks, e.g. `{1, 0, 1}` in case of
    /// `4i16o4i`, because `i` is the 1st dim and `o` is the 0st dim
    zendnn_dims_t inner_idxs;
} zendnn_blocking_desc_t;

/// Winograd-specific formats
typedef enum {
    /// Undefined memory format, used for empty memory descriptors.
    zendnn_wino_undef = 0,
    // Tensors of weights for 2x3 winograd convolutions.
    zendnn_wino_wei_aaOIoi, ///< Internal weights format for 2x3 Winograd
    zendnn_wino_wei_aaOio, ///< Internal weights format for 2x3 Winograd
    zendnn_wino_wei_aaOBiOo, ///< Internal weights format for 2x3 Winograd
    // Tensor of weights for 4x3 convolution.
    zendnn_wino_wei_OBaaIBOIio ///< Internal weights format for 4x3 Winograd
} zendnn_wino_memory_format_t;

/// Description of tensor of weights for winograd 2x3 convolution.
typedef struct {
    zendnn_wino_memory_format_t wino_format;
    int r;
    int alpha;
    int ic;
    int oc;
    int ic_block;
    int oc_block;
    int ic2_block;
    int oc2_block;
    float adj_scale;
    size_t size;
} zendnn_wino_desc_t;

typedef enum {
    zendnn_packed_format_undef = 0,
    zendnn_ldigo_p,
    zendnn_ldgoi_p,
    zendnn_ldio_p
} zendnn_rnn_packed_memory_format_t;

/// Maximum number of parts of RNN weights tensor that require separate
/// computation.
#define ZENDNN_RNN_MAX_N_PARTS 4

/// Description of tensor of packed weights for rnn.
typedef struct {
    zendnn_rnn_packed_memory_format_t format;
    int n_parts;
    int n;
    int ldb;
    int parts[ZENDNN_RNN_MAX_N_PARTS];
    size_t part_pack_size[ZENDNN_RNN_MAX_N_PARTS];
    unsigned pack_part[ZENDNN_RNN_MAX_N_PARTS];
    size_t offset_compensation;
    size_t size;
    char reserved[200];
} zendnn_rnn_packed_desc_t;

/// Flags for memory special features
typedef enum {
    zendnn_memory_extra_flag_none = 0x0U,
    /// Indicates the weights have an additional buffer, that depends on the
    /// @p compensation_mask.
    ///
    /// For instance, in 4D case with the compensation mask equals (1 << 0)
    /// the additional buffer would consist of OC values:
    /// O[oc : 0,OC] =
    ///  -128 * SUM(ic : 0,IC; kh : 0,KH; kw : 0,KW){ weights(oc, ic, kh, kw) }
    zendnn_memory_extra_flag_compensation_conv_s8s8 = 0x1U,
    zendnn_memory_extra_flag_scale_adjust = 0x2U,
    zendnn_memory_extra_flag_rnn_u8s8_compensation = 0x4U,
    zendnn_memory_extra_flag_gpu_rnn_u8s8_compensation
    = zendnn_memory_extra_flag_rnn_u8s8_compensation,
    zendnn_memory_extra_flag_compensation_conv_asymmetric_src = 0x8U,
} zendnn_memory_extra_flags_t;

/// Description of extra information stored in memory
typedef struct {
    /// The flags contain arbitrary extra information, such as compensation.
    /// @sa zendnn_memory_extra_flags_t
    uint64_t flags;
    /// Compensation mask
    int compensation_mask;
    /// Scale applied to the data
    float scale_adjust;
    /// Compensation mask for asymmetric quantization
    int asymm_compensation_mask;
    /// For future backwards compatibility
    char reserved[60];
} zendnn_memory_extra_desc_t;

/// Memory descriptor. The description is based on a number of dimensions,
/// dimensions themselves, plus information about elements type and memory
/// format. Additionally, contains format-specific descriptions of the data
/// layout.
typedef struct {
    /// Number of dimensions
    int ndims;
    /// Dimensions in the following order:
    /// - CNN data tensors: mini-batch, channel, spatial
    ///   (<code>{N, C, [[D,] H,] W}</code>)
    /// - CNN weight tensors: group (optional), output channel, input channel,
    ///   spatial (<code>{[G,] O, I, [[D,] H,] W}</code>)
    /// - RNN data tensors: time, mini-batch, channels (<code>{T, N, C}</code>)
    ///   or layers, directions, states, mini-batch, channels (<code>{L, D, S, N, C}</code>)
    /// - RNN weight tensor: layers, directions, input channel, gates, output channels
    ///   (<code>{L, D, I, G, O}</code>).
    ///
    /// @note
    ///    The order of dimensions does not depend on the memory format, so
    ///    whether the data is laid out in #zendnn_nchw or #zendnn_nhwc
    ///    the dims for 4D CN data tensor would be <code>{N, C, H, W}</code>.
    zendnn_dims_t dims;

    /// Data type of the tensor elements.
    zendnn_data_type_t data_type;

    /// Size of the data including padding in each dimension.
    zendnn_dims_t padded_dims;

    /// Per-dimension offset from the padding to actual data, the top-level
    /// tensor with offsets applied must lie within the padding area.
    zendnn_dims_t padded_offsets;

    /// Offset from memory origin to the current block, non-zero only in
    /// a description of a memory sub-block.
    zendnn_dim_t offset0;

    /// Memory format kind.
    zendnn_format_kind_t format_kind;
    union {
        /// Description of the data layout for memory formats that use
        /// blocking.
        zendnn_blocking_desc_t blocking;
        /// Tensor of weights for integer 8bit winograd convolution.
        zendnn_wino_desc_t wino_desc;
        /// Tensor of packed weights for RNN.
        zendnn_rnn_packed_desc_t rnn_packed_desc;
        // ... other descriptions possible
    } format_desc;

    zendnn_memory_extra_desc_t extra;
} zendnn_memory_desc_t;

/// @struct zendnn_memory
/// An opaque structure to describe a memory.
struct zendnn_memory;

/// A memory handle.
typedef struct zendnn_memory *zendnn_memory_t;

/// A constant memory handle.
typedef const struct zendnn_memory *const_zendnn_memory_t;

/// Special pointer value that indicates that a memory object should not have
/// an underlying buffer.
#define ZENDNN_MEMORY_NONE (NULL)

/// Special pointer value that indicates that the library needs to allocate an
/// underlying buffer for a memory object.
#define ZENDNN_MEMORY_ALLOCATE ((void *)(size_t)-1)

/// @} zendnn_api_memory

/// @addtogroup zendnn_api_primitives
/// @{
/// @addtogroup zendnn_api_primitives_common
/// @{

/// A pointer to any of the operation descriptors.
typedef void *zendnn_op_desc_t;
/// A pointer to any of the operation descriptors (constant variant).
typedef const void *const_zendnn_op_desc_t;

/// @} zendnn_api_primitives_common
/// @} zendnn_api_primitives

/// @addtogroup zendnn_api_primitives
/// @{

/// @addtogroup zendnn_api_convolution
/// @{

/// A descriptor of a convolution operation.
typedef struct {
    /// The kind of primitive. Used for self-identifying the primitive
    /// descriptor. Must be #zendnn_convolution.
    zendnn_primitive_kind_t primitive_kind;
    /// The kind of propagation. Possible values: #zendnn_forward_training,
    /// #zendnn_forward_inference, #zendnn_backward_data,
    /// #zendnn_backward_weights, and #zendnn_backward_bias.
    zendnn_prop_kind_t prop_kind;
    /// The kind of the convolution algorithm. Possible values:
    /// #zendnn_convolution_direct.
    zendnn_alg_kind_t alg_kind;
    /// Source memory descriptor.
    zendnn_memory_desc_t src_desc;
    /// Source gradient memory descriptor.
    zendnn_memory_desc_t diff_src_desc;
    /// Weights memory descriptor.
    zendnn_memory_desc_t weights_desc;
    /// Weights gradient memory descriptor.
    zendnn_memory_desc_t diff_weights_desc;
    /// Bias memory descriptor.
    zendnn_memory_desc_t bias_desc;
    /// Bias gradient memory descriptor.
    zendnn_memory_desc_t diff_bias_desc;
    /// Destination memory descriptor.
    zendnn_memory_desc_t dst_desc;
    /// Destination gradient memory descriptor.
    zendnn_memory_desc_t diff_dst_desc;
    /// Convolution strides in each spatial dimension.
    zendnn_dims_t strides;
    /// Convolution dilates in each spatial dimension.
    zendnn_dims_t dilates;
    /// Padding in each spatial dimension. padding[0] is a padding in the
    /// beginning (@p padding_l), padding[1] is a padding in the end (@p
    /// padding_r).
    zendnn_dims_t padding[2];
    /// The accumulator data type. Initialized automatically.
    zendnn_data_type_t accum_data_type;
    /// Flag for ReLU fusion
    bool reluFused;
    /// Flag for BatchNorm
    bool batchNormFused;
    /// BatchNorm scale memory descriptor
    zendnn_memory_desc_t batchNormScale_desc;
    /// BatchNorm mean memory descriptor
    zendnn_memory_desc_t batchNormMean_desc;
    /// BatchNorm offset memory descriptor
    zendnn_memory_desc_t batchNormOffset_desc;
} zendnn_convolution_desc_t;

/// @} zendnn_api_convolution

/// @addtogroup zendnn_api_deconvolution
/// @{

/// A descriptor of a deconvolution operation.
typedef zendnn_convolution_desc_t zendnn_deconvolution_desc_t;

/// @} zendnn_api_deconvolution

/// @addtogroup zendnn_api_shuffle
/// @{

/// A descriptor of a shuffle operation.
typedef struct {
    /// The kind of primitive. Used for self-identifying the primitive
    /// descriptor. Must be #zendnn_shuffle.
    zendnn_primitive_kind_t primitive_kind;
    /// The kind of propagation. Possible values: #zendnn_forward_training,
    /// #zendnn_forward_inference, and #zendnn_backward_data.
    zendnn_prop_kind_t prop_kind;
    /// Source and destination memory descriptor,
    /// and source and destination gradient memory descriptor.
    zendnn_memory_desc_t data_desc;
    /// Axis for shuffling.
    int axis;
    /// Number of groups.
    zendnn_dim_t group_size;
} zendnn_shuffle_desc_t;

/// @} zendnn_api_shuffle

/// @addtogroup zendnn_api_eltwise
/// @{

/// A descriptor of a element-wise operation.
typedef struct {
    /// The kind of primitive. Used for self-identifying the primitive
    /// descriptor. Must be #zendnn_eltwise.
    zendnn_primitive_kind_t primitive_kind;
    /// The kind of propagation. Possible values: #zendnn_forward_training,
    /// #zendnn_forward_inference, #zendnn_backward, and #zendnn_backward_data.
    zendnn_prop_kind_t prop_kind;
    /// The kind of eltwise algorithm. Possible values: #zendnn_eltwise_relu,
    /// #zendnn_eltwise_tanh, #zendnn_eltwise_elu, #zendnn_eltwise_square,
    /// #zendnn_eltwise_abs, #zendnn_eltwise_sqrt, #zendnn_eltwise_linear,
    /// #zendnn_eltwise_bounded_relu, #zendnn_eltwise_soft_relu,
    /// #zendnn_eltwise_logistic, #zendnn_eltwise_exp, #zendnn_eltwise_gelu_tanh,
    /// #zendnn_eltwise_swish, #zendnn_eltwise_log, #zendnn_eltwise_clip,
    /// #zendnn_eltwise_clip_v2, #zendnn_eltwise_pow, #zendnn_eltwise_gelu_erf,
    /// #zendnn_eltwise_round, #zendnn_eltwise_logsigmoid, #zendnn_eltwise_mish,
    /// #zendnn_eltwise_hardswish.
    /// Possible values for passing destination memory on backward:
    /// #zendnn_eltwise_relu_use_dst_for_bwd, #zendnn_eltwise_tanh_use_dst_for_bwd,
    /// #zendnn_eltwise_elu_use_dst_for_bwd, #zendnn_eltwise_sqrt_use_dst_for_bwd,
    /// #zendnn_eltwise_logistic_use_dst_for_bwd,
    /// #zendnn_eltwise_exp_use_dst_for_bwd,
    /// #zendnn_eltwise_clip_v2_use_dst_for_bwd.
    zendnn_alg_kind_t alg_kind;
    /// Source and destination memory descriptor.
    zendnn_memory_desc_t data_desc;
    /// Source and destination gradient memory descriptor.
    zendnn_memory_desc_t diff_data_desc;
    /// Algorithm specific parameter.
    /// Accordance table:
    ///  - #zendnn_eltwise_relu: @p alpha -- negative slope, @p beta ignored
    ///  - #zendnn_eltwise_tanh: @p alpha and @p beta ignored
    ///  - #zendnn_eltwise_elu: @p alpha -- negative slope, @p beta ignored
    ///  - #zendnn_eltwise_square: @p alpha and @p beta ignored
    ///  - #zendnn_eltwise_abs: @p alpha and @p beta ignored
    ///  - #zendnn_eltwise_sqrt: @p alpha and @p beta ignored
    ///  - #zendnn_eltwise_linear: @p alpha -- scale, @p beta -- shift
    ///  - #zendnn_eltwise_bounded_relu: @p alpha -- upper bound, @p beta ignored
    ///  - #zendnn_eltwise_soft_relu: @p alpha and @p beta ignored
    ///  - #zendnn_eltwise_logistic: @p alpha and @p beta ignored
    ///  - #zendnn_eltwise_exp: @p alpha and @p beta ignored
    ///  - #zendnn_eltwise_gelu_tanh: @p alpha and @p beta ignored
    ///  - #zendnn_eltwise_swish: @p alpha -- sigmoid arg scaling, @p beta ignored
    ///  - #zendnn_eltwise_log: @p alpha and @p beta ignored
    ///  - #zendnn_eltwise_clip: @p alpha -- lower bound, @p beta -- upper bound
    ///  - #zendnn_eltwise_clip_v2: @p alpha -- lower bound, @p beta -- upper bound
    ///  - #zendnn_eltwise_pow: @p alpha -- scale, @p beta -- exponent
    ///  - #zendnn_eltwise_gelu_erf: @p alpha and @p beta ignored
    ///  - #zendnn_eltwise_round: @p alpha and @p beta ignored
    ///  - #zendnn_eltwise_logsigmoid @p alpha and @p beta ignored
    ///  - #zendnn_eltwise_mish @p alpha and @p beta ignored
    ///  - #zendnn_eltwise_hardswish @p alpha and @p beta ignored
    float alpha, beta;
} zendnn_eltwise_desc_t;

/// @} zendnn_api_eltwise

/// @addtogroup zendnn_api_softmax
/// @{

/// A descriptor of a Softmax operation.
typedef struct {
    /// The kind of primitive. Used for self-identifying the primitive
    /// descriptor. Must be #zendnn_softmax.
    zendnn_primitive_kind_t primitive_kind;
    /// The kind of propagation. Possible values: #zendnn_forward_training and
    /// #zendnn_forward_inference.
    zendnn_prop_kind_t prop_kind;
    /// Source and destination memory descriptor.
    zendnn_memory_desc_t data_desc;
    /// Source and Destination of gradient memory descriptor.
    zendnn_memory_desc_t diff_desc;
    /// The axis along which to perform the softmax.
    int softmax_axis;
} zendnn_softmax_desc_t;

/// @} zendnn_api_softmax

/// @addtogroup zendnn_api_logsoftmax
/// @{

/// A descriptor of a LogSoftmax operation. An alias of Softmax structure, but
/// primitive_kind must be #zendnn_logsoftmax.
typedef zendnn_softmax_desc_t zendnn_logsoftmax_desc_t;

/// @} zendnn_api_logsoftmax

/// @addtogroup zendnn_api_pooling
/// @{

/// A descriptor of a pooling operation.
typedef struct {
    /// The kind of primitive. Used for self-identifying the primitive
    /// descriptor. Must be #zendnn_pooling.
    zendnn_primitive_kind_t primitive_kind;
    /// The kind of propagation. Possible values: #zendnn_forward_training,
    /// #zendnn_forward_inference, #zendnn_backward, and #zendnn_backward_data.
    zendnn_prop_kind_t prop_kind;
    /// The kind of pooling algorithm.
    /// Possible values: #zendnn_pooling_max,
    /// #zendnn_pooling_avg_include_padding, and
    /// #zendnn_pooling_avg_exclude_padding.
    zendnn_alg_kind_t alg_kind;
    /// Source memory descriptor.
    zendnn_memory_desc_t src_desc;
    /// Source gradient memory descriptor.
    zendnn_memory_desc_t diff_src_desc;
    /// Destination memory descriptor.
    zendnn_memory_desc_t dst_desc;
    /// Destination gradient memory descriptor.
    zendnn_memory_desc_t diff_dst_desc;
    /// Pooling kernel strides for spatial dimensions.
    zendnn_dims_t strides;
    /// Pooling kernel spatial dimensions.
    zendnn_dims_t kernel;
    /// Padding in each spatial dimension. padding[0] is a padding in the
    /// beginning (@p padding_l), padding[1] is a padding in the end (@p
    /// padding_r).
    zendnn_dims_t padding[2];
    /// The accumulator data type. Initialized automatically.
    zendnn_data_type_t accum_data_type;
} zendnn_pooling_desc_t;

/// @} zendnn_api_pooling

/// @addtogroup zendnn_api_pooling_v2
/// @{

/// A descriptor of a pooling operation.
typedef struct {
    /// The kind of primitive. Used for self-identifying the primitive
    /// descriptor. Must be #zendnn_pooling_v2.
    zendnn_primitive_kind_t primitive_kind;
    /// The kind of propagation. Possible values: #zendnn_forward_training,
    /// #zendnn_forward_inference, #zendnn_backward, and #zendnn_backward_data.
    zendnn_prop_kind_t prop_kind;
    /// The kind of pooling algorithm.
    /// Possible values: #zendnn_pooling_max,
    /// #zendnn_pooling_avg_include_padding, and
    /// #zendnn_pooling_avg_exclude_padding.
    zendnn_alg_kind_t alg_kind;
    /// Source memory descriptor.
    zendnn_memory_desc_t src_desc;
    /// Source gradient memory descriptor.
    zendnn_memory_desc_t diff_src_desc;
    /// Destination memory descriptor.
    zendnn_memory_desc_t dst_desc;
    /// Destination gradient memory descriptor.
    zendnn_memory_desc_t diff_dst_desc;
    /// Pooling kernel strides for spatial dimensions.
    zendnn_dims_t strides;
    /// Pooling kernel spatial dimensions.
    zendnn_dims_t kernel;
    /// Padding in each spatial dimension. padding[0] is a padding in the
    /// beginning (@p padding_l), padding[1] is a padding in the end (@p
    /// padding_r).
    zendnn_dims_t padding[2];
    /// The accumulator data type. Initialized automatically.
    zendnn_data_type_t accum_data_type;
    /// Pooling dilations for spatial dimensions.
    zendnn_dims_t dilation;
} zendnn_pooling_v2_desc_t;

/// @} zendnn_api_pooling_v2

/// @addtogroup zendnn_api_prelu
/// @{
typedef struct {
    /// The kind of primitive. Used for self-identifying the primitive
    /// descriptor. Must be #zendnn_prelu.
    zendnn_primitive_kind_t primitive_kind;
    /// The kind of propagation. Possible values: #zendnn_forward_training,
    /// #zendnn_forward_inference, #zendnn_backward
    zendnn_prop_kind_t prop_kind;
    /// Source and destination memory descriptor.
    zendnn_memory_desc_t data_desc;
    /// Learnable parameter alpha memory descriptor.
    /// Alpha describes negative slope.
    zendnn_memory_desc_t weights_desc;
    /// Source and destination gradient memory descriptor.
    zendnn_memory_desc_t diff_data_desc;
    /// Learnable parameter alpha gradient memory descriptor.
    zendnn_memory_desc_t diff_weights_desc;
} zendnn_prelu_desc_t;

/// @} zendnn_api_prelu

/// @addtogroup zendnn_api_lrn
/// @{

/// A descriptor of a Local Response Normalization (LRN) operation.
typedef struct {
    /// The kind of primitive. Used for self-identifying the primitive
    /// descriptor. Must be #zendnn_lrn.
    zendnn_primitive_kind_t primitive_kind;
    /// The kind of propagation. Possible values: #zendnn_forward_training,
    /// #zendnn_forward_inference, #zendnn_backward, and #zendnn_backward_data.
    zendnn_prop_kind_t prop_kind;
    /// LRN algorithm. Possible values: #zendnn_lrn_within_channel and
    /// #zendnn_lrn_across_channels.
    zendnn_alg_kind_t alg_kind;
    /// Source and destination memory descriptor.
    zendnn_memory_desc_t data_desc;
    /// Source and destination gradient memory descriptor.
    zendnn_memory_desc_t diff_data_desc;
    /// The number of channels to sum over (for cross-channel LRN) or the side
    /// length of the square region to sum over (for within-channel LRN).
    zendnn_dim_t local_size;
    /// LRN alpha parameter.
    float lrn_alpha;
    /// LRN beta parameter.
    float lrn_beta;
    /// LRN k parameter.
    float lrn_k;
} zendnn_lrn_desc_t;

/// @} zendnn_api_lrn

/// @addtogroup zendnn_api_batch_normalization
/// @{

/// A descriptor of a Batch Normalization operation.
typedef struct {
    /// The kind of primitive. Used for self-identifying the primitive
    /// descriptor. Must be #zendnn_batch_normalization.
    zendnn_primitive_kind_t primitive_kind;
    /// The kind of propagation. Possible values: #zendnn_forward_training,
    /// #zendnn_forward_inference, #zendnn_backward, and #zendnn_backward_data.
    zendnn_prop_kind_t prop_kind;
    /// Source and destination memory descriptor.
    zendnn_memory_desc_t data_desc;
    /// Source and destination gradient memory descriptor.
    zendnn_memory_desc_t diff_data_desc;
    /// Scale and shift data and gradient memory descriptors.
    ///
    /// Scaleshift memory descriptor uses 2D #zendnn_nc format[2,Channels]. 1-st
    /// dimension contains gamma parameter, 2-nd dimension contains beta
    /// parameter.
    zendnn_memory_desc_t data_scaleshift_desc;
    zendnn_memory_desc_t diff_data_scaleshift_desc;
    /// Statistics memory descriptor.
    ///
    /// Statistics (mean or variance) descriptor use 1D #zendnn_x format[Channels].
    zendnn_memory_desc_t stat_desc;
    /// Batch normalization epsilon parameter.
    float batch_norm_epsilon;
    unsigned flags;
} zendnn_batch_normalization_desc_t;

/// @} zendnn_api_batch_normalization

/// @addtogroup zendnn_api_layer_normalization
/// @{

/// A descriptor of a Layer Normalization operation.
typedef struct {
    /// The kind of primitive. Used for self-identifying the primitive
    /// descriptor. Must be #zendnn_layer_normalization.
    zendnn_primitive_kind_t primitive_kind;
    /// The kind of propagation. Possible values: #zendnn_forward_training,
    /// #zendnn_forward_inference, #zendnn_backward, and #zendnn_backward_data.
    zendnn_prop_kind_t prop_kind;
    /// Source and destination memory descriptor.
    zendnn_memory_desc_t data_desc;
    /// Source and destination gradient memory descriptor.
    zendnn_memory_desc_t diff_data_desc;
    /// Scale and shift data and gradient memory descriptors.
    ///
    /// Scaleshift memory descriptor uses 2D #zendnn_ab
    /// format[2, normalized_dim] where 1-st dimension contains gamma parameter,
    /// 2-nd dimension contains beta parameter. Normalized_dim is equal to the
    /// last logical dimension of the data tensor across which normalization is
    /// performed.
    zendnn_memory_desc_t data_scaleshift_desc;
    zendnn_memory_desc_t diff_data_scaleshift_desc;
    /// Mean and variance data memory descriptors.
    ///
    /// Statistics (mean and variance) memory descriptor is the k-dimensional tensor
    /// where k is equal to data_tensor_ndims - 1 and may have any plain
    /// (stride[last_dim] == 1) user-provided format.
    zendnn_memory_desc_t stat_desc;
    /// Layer normalization epsilon parameter.
    float layer_norm_epsilon;
    unsigned flags;
} zendnn_layer_normalization_desc_t;

/// @} zendnn_api_layer_normalization

/// @addtogroup zendnn_api_inner_product
/// @{

/// A descriptor of an inner product operation.
typedef struct {
    /// The kind of primitive. Used for self-identifying the primitive
    /// descriptor. Must be #zendnn_inner_product.
    zendnn_primitive_kind_t primitive_kind;
    /// The kind of propagation. Possible values: #zendnn_forward_training,
    /// #zendnn_forward_inference, #zendnn_backward_data,
    /// #zendnn_backward_weights, and #zendnn_backward_bias.
    zendnn_prop_kind_t prop_kind;
    /// Source memory descriptor.
    zendnn_memory_desc_t src_desc;
    /// Source gradient memory descriptor.
    zendnn_memory_desc_t diff_src_desc;
    /// Weights memory descriptor.
    zendnn_memory_desc_t weights_desc;
    /// Weights gradient memory descriptor.
    zendnn_memory_desc_t diff_weights_desc;
    /// Bias memory descriptor.
    zendnn_memory_desc_t bias_desc;
    /// Bias gradient memory descriptor.
    zendnn_memory_desc_t diff_bias_desc;
    /// Destination memory descriptor.
    zendnn_memory_desc_t dst_desc;
    /// Destination gradient memory descriptor.
    zendnn_memory_desc_t diff_dst_desc;
    /// The accumulator data type. Initialized automatically.
    zendnn_data_type_t accum_data_type;
} zendnn_inner_product_desc_t;

/// @} zendnn_api_inner_product

/// @addtogroup zendnn_api_rnn
/// @{

/// Flags for RNN cell.
typedef enum {
    /// Undefined RNN flags
    zendnn_rnn_flags_undef = 0x0
} zendnn_rnn_flags_t;

/// A direction of RNN primitive execution.
typedef enum {
    /// Unidirectional execution of RNN primitive from left to right.
    zendnn_unidirectional_left2right,
    /// Unidirectional execution of RNN primitive from right to left.
    zendnn_unidirectional_right2left,
    /// Bidirectional execution of RNN primitive with concatenation of the
    /// results.
    zendnn_bidirectional_concat,
    /// Bidirectional execution of RNN primitive with summation of the
    /// results.
    zendnn_bidirectional_sum,
    /// Alias for #zendnn_unidirectional_left2right.
    zendnn_unidirectional = zendnn_unidirectional_left2right,
} zendnn_rnn_direction_t;

/// A descriptor for an RNN operation.
typedef struct {
    /// The kind of primitive. Used for self-identifying the primitive
    /// descriptor. Must be #zendnn_rnn.
    zendnn_primitive_kind_t primitive_kind;
    /// The kind of propagation. Possible values: #zendnn_forward_training,
    /// #zendnn_forward_inference, and #zendnn_backward.
    zendnn_prop_kind_t prop_kind;
    /// RNN cell kind. Must be one of #zendnn_vanilla_rnn,
    /// #zendnn_vanilla_lstm, #zendnn_vanilla_gru, or #zendnn_lbr_gru.
    zendnn_alg_kind_t cell_kind;
    /// The direction of RNN primitive execution.
    zendnn_rnn_direction_t direction;
    /// Source layer memory descriptor.
    zendnn_memory_desc_t src_layer_desc;
    /// Source iteration memory descriptor for hidden state.
    zendnn_memory_desc_t src_iter_desc;
    /// Source iteration memory descriptor for cell state.
    zendnn_memory_desc_t src_iter_c_desc;
    /// Weights layer memory descriptor.
    zendnn_memory_desc_t weights_layer_desc;
    /// Weights iteration memory descriptor.
    zendnn_memory_desc_t weights_iter_desc;
    /// Bias memory descriptor.
    zendnn_memory_desc_t bias_desc;
    /// Destination layer memory descriptor.
    zendnn_memory_desc_t dst_layer_desc;
    /// Destination iter memory descriptor for hidden state.
    zendnn_memory_desc_t dst_iter_desc;
    /// Destination iter memory descriptor for cell state.
    zendnn_memory_desc_t dst_iter_c_desc;
    /// Weights peephole memory descriptor.
    /// This memory descriptor is equal to zero memory descriptor in case of
    /// non-peephole LSTMs and other non-LSTM RNNs.
    zendnn_memory_desc_t weights_peephole_desc;
    /// Weights projection memory descriptor.
    /// This memory descriptor is equal to zero memory descriptor in case of
    /// non-projection LSTMs and other non-LSTM RNNs.
    zendnn_memory_desc_t weights_projection_desc;

    /// Source gradient layer memory descriptor.
    zendnn_memory_desc_t diff_src_layer_desc;
    /// Source gradient iter memory descriptor for hidden state.
    zendnn_memory_desc_t diff_src_iter_desc;
    /// Source gradient iter memory descriptor for cell state.
    zendnn_memory_desc_t diff_src_iter_c_desc;
    /// Weights gradient layer memory descriptor.
    zendnn_memory_desc_t diff_weights_layer_desc;
    /// Weights gradient iter memory descriptor.
    zendnn_memory_desc_t diff_weights_iter_desc;
    /// Bias gradient memory descriptor.
    zendnn_memory_desc_t diff_bias_desc;
    /// Destination gradient layer memory descriptor.
    zendnn_memory_desc_t diff_dst_layer_desc;
    /// Destination gradient iteration memory descriptor for hidden state.
    zendnn_memory_desc_t diff_dst_iter_desc;
    /// Destination gradient iteration memory descriptor for cell state.
    zendnn_memory_desc_t diff_dst_iter_c_desc;
    /// Weights gradient peephole memory descriptor.
    /// This memory descriptor is equal to zero memory descriptor in case of
    /// non-peephole LSTMs and other non-LSTM RNNs.
    zendnn_memory_desc_t diff_weights_peephole_desc;
    /// Weights gradient projection memory descriptor.
    /// This memory descriptor is equal to zero memory descriptor in case of
    /// non-projection LSTMs and other non-LSTM RNNs.
    zendnn_memory_desc_t diff_weights_projection_desc;

    /// RNN cell flags
    unsigned int flags;
    /// Activation function used for vanilla_rnn cell kind.
    /// Must be either #zendnn_eltwise_relu or #zendnn_eltwise_tanh.
    zendnn_alg_kind_t activation_kind;
    float alpha;
    float beta;

} zendnn_rnn_desc_t;

/// @} zendnn_api_rnn

/// @addtogroup zendnn_api_binary
/// @{

/// A descriptor of a binary operation.
typedef struct {
    /// The kind of primitive. Used for self-identifying the primitive
    /// descriptor. Must be #zendnn_binary.
    zendnn_primitive_kind_t primitive_kind;
    /// The kind of the binary algorithm. Possible values:
    /// #zendnn_binary_add, #zendnn_binary_mul, #zendnn_binary_max, #zendnn_binary_min,
    /// #zendnn_binary_div and #zendnn_binary_sub.
    zendnn_alg_kind_t alg_kind;
    /// Source memory descriptors.
    zendnn_memory_desc_t src_desc[2];
    /// Destination memory descriptor.
    zendnn_memory_desc_t dst_desc;
} zendnn_binary_desc_t;

/// @} zendnn_api_binary

/// @addtogroup zendnn_api_matmul
/// @{

/// A descriptor of a matrix multiplication operation.
///
/// 2D case:
///     dst[m, n] = src[m, k] * weights[k, n] + bias[m, n]
///
/// 3D case:
///     dst[mb, m, n] = src[mb, m, k] * weights[mb, k, n] + bias[mb, m, n]
typedef struct {
    /// The kind of primitive. Used for self-identifying the primitive
    /// descriptor. Must be #zendnn_matmul.
    zendnn_primitive_kind_t primitive_kind;
    /// Source memory descriptor.
    zendnn_memory_desc_t src_desc;
    /// Weights memory descriptor.
    zendnn_memory_desc_t weights_desc;
    /// Bias memory descriptor.
    zendnn_memory_desc_t bias_desc;
    /// Destination memory descriptor.
    zendnn_memory_desc_t dst_desc;
    /// The accumulator data type. Initialized automatically.
    zendnn_data_type_t accum_data_type;
} zendnn_matmul_desc_t;

/// @} zendnn_api_matmul

/// @addtogroup zendnn_api_resampling
/// @{

/// A descriptor of resampling operation.
typedef struct {
    /// The kind of primitive. Used for self-identifying the primitive
    /// descriptor. Must be #zendnn_resampling.
    zendnn_primitive_kind_t primitive_kind;
    /// The kind of propagation. Possible values: #zendnn_forward_training,
    /// #zendnn_forward_inference, #zendnn_backward_data,
    zendnn_prop_kind_t prop_kind;
    /// The kind of the resampling algorithm. Possible values:
    /// #zendnn_resampling_nearest, #zendnn_resampling_linear.
    zendnn_alg_kind_t alg_kind;
    /// Source memory descriptor.
    zendnn_memory_desc_t src_desc;
    /// Source gradient memory descriptor.
    zendnn_memory_desc_t diff_src_desc;
    /// Destination memory descriptor.
    zendnn_memory_desc_t dst_desc;
    /// Destination gradient memory descriptor.
    zendnn_memory_desc_t diff_dst_desc;
    /// Resampling factor in each spatial dimension.
    float factors[ZENDNN_MAX_NDIMS];
} zendnn_resampling_desc_t;

/// @} zendnn_api_resampling

/// @addtogroup zendnn_api_reduction
/// @{

/// A descriptor of reduction operation.
typedef struct {
    /// The kind of primitive. Used for self-identifying the primitive
    /// descriptor. Must be #zendnn_reduction.
    zendnn_primitive_kind_t primitive_kind;
    /// The kind of reduction algorithm. Possible values:
    /// #zendnn_reduction_max, #zendnn_reduction_min, #zendnn_reduction_sum,
    /// #zendnn_reduction_mul, #zendnn_reduction_mean, #zendnn_reduction_norm_lp_max,
    /// #zendnn_reduction_norm_lp_sum, #zendnn_reduction_norm_lp_power_p_max,
    /// #zendnn_reduction_norm_lp_power_p_sum.
    zendnn_alg_kind_t alg_kind;
    /// Source memory descriptor.
    zendnn_memory_desc_t src_desc;
    /// Destination memory descriptor.
    zendnn_memory_desc_t dst_desc;
    /// Algorithm specific parameters.
    /// Accordance table:
    /// #zendnn_reduction_max: @p p and @p eps are ignored
    /// #zendnn_reduction_min: @p p and @p eps are ignored
    /// #zendnn_reduction_norm_lp_max: @p p -- power, @p eps -- epsilon
    /// #zendnn_reduction_norm_lp_sum: @p p -- power, @p eps -- epsilon
    /// #zendnn_reduction_norm_lp_power_p_max: @p p -- power, @p eps -- epsilon
    /// #zendnn_reduction_norm_lp_power_p_sum: @p p -- power, @p eps -- epsilon
    /// #zendnn_reduction_sum: @p p and @p eps are ignored
    /// #zendnn_reduction_mul: @p p and @p eps are ignored
    /// #zendnn_reduction_mean: @p p and @p eps are ignored
    float p, eps;
} zendnn_reduction_desc_t;

/// @} zendnn_api_reduction

/* add new primitive */

/// @addtogroup zendnn_api_embedding_bag
/// @{

/// A descriptor of embedding bag operation.
typedef struct {
    /// The kind of primitive. Used for self-identifying the primitive
    /// descriptor. Must be #zendnn_embedding_bag.
    zendnn_primitive_kind_t primitive_kind;

    /// The kind of propagation. Possible values: #zendnn_forward_inference
    zendnn_prop_kind_t prop_kind;

    /// The kind of embedding_bag algorithm. Possible values:
    /// #zendnn_embedding_bag_max, #zendnn_embedding_bag_min, or
    /// #zendnn_embedding_bag_sum,
    zendnn_alg_kind_t alg_kind;

    /// input (embedding table) memory descriptor.
    zendnn_memory_desc_t input_desc;

    /// indices memory descriptor.
    zendnn_memory_desc_t indices_desc;

    /// offsets memory descriptor.
    zendnn_memory_desc_t offsets_desc;

    /// weights memory descriptor.
    zendnn_memory_desc_t weights_desc;

    /// Destination memory descriptor.
    zendnn_memory_desc_t dst_desc;

    /// Algorithm specific parameters.
    int32_t padding_idx; //padding index, set to -1 if no padding index
    bool  is_weights; // true if weight memory descriptor provided

    int32_t num_threads; // no of parallel threads

} zendnn_embedding_bag_desc_t;

/// @} zendnn_api_embedding_bag
/// @} zendnn_api_primitives

/// @addtogroup zendnn_api_engine
/// @{

/// @brief Kinds of engines.
typedef enum {
    /// An unspecified engine.
    zendnn_any_engine,
    /// CPU engine.
    zendnn_cpu,
    /// GPU engine.
    zendnn_gpu,
} zendnn_engine_kind_t;

/// @struct zendnn_engine
/// @brief An opaque structure to describe an engine.
struct zendnn_engine;
/// @brief An engine handle.
typedef struct zendnn_engine *zendnn_engine_t;
#if 0
// FIXME: looks like this never happens
/// @brief A constant engine handle.
typedef const struct zendnn_engine *const_zendnn_engine_t;
#endif

/// @} zendnn_api_engine

/// @addtogroup zendnn_api_primitives
/// @{
/// @addtogroup zendnn_api_primitives_common
/// @{

/// @struct zendnn_primitive_desc_iterator
/// @brief An opaque structure to describe a primitive descriptor iterator.
struct zendnn_primitive_desc_iterator;

/// @brief A primitive descriptor iterator handle.
typedef struct zendnn_primitive_desc_iterator *zendnn_primitive_desc_iterator_t;

/// @brief A constant primitive descriptor iterator handle.
typedef const struct zendnn_primitive_desc_iterator
    *const_zendnn_primitive_desc_iterator_t;

/// @struct zendnn_primitive_desc
/// @brief An opaque structure to describe a primitive descriptor.
struct zendnn_primitive_desc;

/// @brief A primitive descriptor handle.
typedef struct zendnn_primitive_desc *zendnn_primitive_desc_t;

/// @brief A constant primitive descriptor handle.
typedef const struct zendnn_primitive_desc *const_zendnn_primitive_desc_t;

/// @} zendnn_api_primitives_common

/// @addtogroup zendnn_api_attributes
/// @{

/// Scratchpad mode
typedef enum {
    /// The library manages the scratchpad allocation according to the policy
    /// specified by the `ZENDNN_ENABLE_CONCURRENT_EXEC`
    /// [build option](@ref dev_guide_build_options) (default).
    ///
    /// When `ZENDNN_ENABLE_CONCURRENT_EXEC=OFF` (default), the library
    /// scratchpad is common to all primitives to reduce the memory footprint.
    /// This configuration comes with limited thread-safety properties, namely
    /// primitives can be created and executed in parallel but cannot migrate
    /// between threads (in other words, each primitive should be executed in
    /// the same thread it was created in).
    ///
    /// When `ZENDNN_ENABLE_CONCURRENT_EXEC=ON`, the library scratchpad is
    /// private to each primitive. The memory footprint is larger than when
    /// using `ZENDNN_ENABLE_CONCURRENT_EXEC=OFF` but different primitives can be
    /// created and run concurrently (the same primitive cannot be run
    /// concurrently from two different threads though).
    zendnn_scratchpad_mode_library,
    /// The user manages the scratchpad allocation by querying and providing
    /// the scratchpad memory to primitives. This mode is thread-safe as long
    /// as the scratchpad buffers are not used concurrently by two primitive
    /// executions.
    zendnn_scratchpad_mode_user,
} zendnn_scratchpad_mode_t;

/// @struct zendnn_primitive_attr
/// @brief An opaque structure for primitive descriptor attributes.
///
/// Attributes may contain:
///  - output scales (to scale the result prior to storing it to the memory)
struct zendnn_primitive_attr;

/// @brief A primitive descriptor attributes handle that controls primitive
/// behavior.
typedef struct zendnn_primitive_attr *zendnn_primitive_attr_t;

/// @brief A constant primitive descriptor attributes handle.
typedef const struct zendnn_primitive_attr *const_zendnn_primitive_attr_t;

/// @struct zendnn_post_ops
/// @brief An opaque structure for a chain of post operations.
///
/// zendnn_post_ops can be used to perform some (trivial) operations like
/// accumulation or eltwise after certain primitives like convolution.
///
/// Post operations might be combined together, making a chain of post
/// operations. For instance one can configure convolution followed by
/// accumulation followed by eltwise. This might be especially beneficial
/// for residual learning blocks.
///
/// @warning
///      Of course not all combinations are supported, so the user should handle
///      errors accordingly.
///
/// Supported post operations:
///  - accumulation (base primitive: convolution)
///  - eltwise (base primitive: convolution)
struct zendnn_post_ops;

/// @brief A post operation chain handle.
typedef struct zendnn_post_ops *zendnn_post_ops_t;

/// @brief A constant post operation chain handle.
typedef const struct zendnn_post_ops *const_zendnn_post_ops_t;

/// @} zendnn_api_attributes

/// @addtogroup zendnn_api_primitives_common
/// @{

/// @struct zendnn_primitive
/// An opaque structure to describe a primitive.
struct zendnn_primitive;
/// A primitive handle.
typedef struct zendnn_primitive *zendnn_primitive_t;
/// A constant primitive handle.
typedef const struct zendnn_primitive *const_zendnn_primitive_t;

/// Source argument #0.
#define ZENDNN_ARG_SRC_0 1
/// A special mnemonic for source argument for primitives that have a
/// single source. An alias for #ZENDNN_ARG_SRC_0.
#define ZENDNN_ARG_SRC ZENDNN_ARG_SRC_0
/// A special mnemonic for RNN input vector. An alias for
/// #ZENDNN_ARG_SRC_0.
#define ZENDNN_ARG_SRC_LAYER ZENDNN_ARG_SRC_0
/// A special mnemonic for reorder source argument. An alias for
/// #ZENDNN_ARG_SRC_0.
#define ZENDNN_ARG_FROM ZENDNN_ARG_SRC_0

/// Source argument #1.
#define ZENDNN_ARG_SRC_1 2
/// A special mnemonic for RNN input recurrent hidden state vector. An alias
/// for #ZENDNN_ARG_SRC_1.
#define ZENDNN_ARG_SRC_ITER ZENDNN_ARG_SRC_1

/// Source argument #2.
#define ZENDNN_ARG_SRC_2 3
/// A special mnemonic for RNN input recurrent cell state vector. An alias for
/// #ZENDNN_ARG_SRC_2.
#define ZENDNN_ARG_SRC_ITER_C ZENDNN_ARG_SRC_2

/// Source argument #3.
#define ZENDNN_ARG_SRC_3 4

/// Destination argument #0.
#define ZENDNN_ARG_DST_0 17
/// A special mnemonic for destination argument for primitives that have a
/// single destination. An alias for #ZENDNN_ARG_DST_0.
#define ZENDNN_ARG_DST ZENDNN_ARG_DST_0
/// A special mnemonic for reorder destination argument. An alias for
/// #ZENDNN_ARG_DST_0.
#define ZENDNN_ARG_TO ZENDNN_ARG_DST_0
/// A special mnemonic for RNN output vector. An alias for #ZENDNN_ARG_DST_0.
#define ZENDNN_ARG_DST_LAYER ZENDNN_ARG_DST_0

/// Destination argument #1.
#define ZENDNN_ARG_DST_1 18
/// A special mnemonic for RNN input recurrent hidden state vector. An
/// alias for #ZENDNN_ARG_DST_1.
#define ZENDNN_ARG_DST_ITER ZENDNN_ARG_DST_1

/// Destination argument #2.
#define ZENDNN_ARG_DST_2 19
/// A special mnemonic for LSTM output recurrent cell state vector. An
/// alias for #ZENDNN_ARG_DST_2.
#define ZENDNN_ARG_DST_ITER_C ZENDNN_ARG_DST_2

/// Weights argument #0.
#define ZENDNN_ARG_WEIGHTS_0 33
/// A special mnemonic for primitives that have a single weights
/// argument. Alias for #ZENDNN_ARG_WEIGHTS_0.
#define ZENDNN_ARG_WEIGHTS ZENDNN_ARG_WEIGHTS_0
/// A special mnemonic for scale and shift argument of normalization
/// primitives. Alias for #ZENDNN_ARG_WEIGHTS_0.
#define ZENDNN_ARG_SCALE_SHIFT ZENDNN_ARG_WEIGHTS_0
/// A special mnemonic for RNN weights applied to the layer input. An
/// alias for #ZENDNN_ARG_WEIGHTS_0.
#define ZENDNN_ARG_WEIGHTS_LAYER ZENDNN_ARG_WEIGHTS_0

/// Weights argument #1.
#define ZENDNN_ARG_WEIGHTS_1 34
/// A special mnemonic for RNN weights applied to the recurrent input.
/// An alias for #ZENDNN_ARG_WEIGHTS_1.
#define ZENDNN_ARG_WEIGHTS_ITER ZENDNN_ARG_WEIGHTS_1

/// Weights argument #2.
#define ZENDNN_ARG_WEIGHTS_2 35
/// A special mnemonic for RNN weights applied to the peephole weights.
/// An alias for #ZENDNN_ARG_WEIGHTS_2.
#define ZENDNN_ARG_WEIGHTS_PEEPHOLE ZENDNN_ARG_WEIGHTS_2

/// Weights argument #3.
#define ZENDNN_ARG_WEIGHTS_3 36
/// A special mnemonic for RNN weights applied to the projection weights.
/// An alias for #ZENDNN_ARG_WEIGHTS_3.
#define ZENDNN_ARG_WEIGHTS_PROJECTION ZENDNN_ARG_WEIGHTS_3

/// Bias tensor argument.
#define ZENDNN_ARG_BIAS 41

/// Mean values tensor argument.
#define ZENDNN_ARG_MEAN 49
/// Variance values tensor argument.
#define ZENDNN_ARG_VARIANCE 50

/// Workspace tensor argument. Workspace is used to pass information
/// from forward propagation to backward propagation computations.
#define ZENDNN_ARG_WORKSPACE 64
/// Scratchpad (temporary storage) tensor argument.
#define ZENDNN_ARG_SCRATCHPAD 80

/// Gradient (diff) of the source argument #0.
#define ZENDNN_ARG_DIFF_SRC_0 129
/// A special mnemonic for primitives that have a single diff source argument.
/// An alias for #ZENDNN_ARG_DIFF_SRC_0.
#define ZENDNN_ARG_DIFF_SRC ZENDNN_ARG_DIFF_SRC_0
/// A special mnemonic for gradient (diff) of RNN input vector. An alias for
/// #ZENDNN_ARG_DIFF_SRC_0.
#define ZENDNN_ARG_DIFF_SRC_LAYER ZENDNN_ARG_DIFF_SRC_0

/// Gradient (diff) of the source argument #1.
#define ZENDNN_ARG_DIFF_SRC_1 130
/// A special mnemonic for gradient (diff) of RNN input recurrent hidden state
/// vector. An alias for #ZENDNN_ARG_DIFF_SRC_1.
#define ZENDNN_ARG_DIFF_SRC_ITER ZENDNN_ARG_DIFF_SRC_1

/// Gradient (diff) of the source argument #2.
#define ZENDNN_ARG_DIFF_SRC_2 131
/// A special mnemonic for gradient (diff) of RNN input recurrent cell state
/// vector. An alias for #ZENDNN_ARG_DIFF_SRC_1.
#define ZENDNN_ARG_DIFF_SRC_ITER_C ZENDNN_ARG_DIFF_SRC_2

/// Gradient (diff) of the destination argument #0.
#define ZENDNN_ARG_DIFF_DST_0 145
/// A special mnemonic for primitives that have a single diff destination
/// argument. An alias for #ZENDNN_ARG_DIFF_DST_0.
#define ZENDNN_ARG_DIFF_DST ZENDNN_ARG_DIFF_DST_0
/// A special mnemonic for gradient (diff) of RNN output vector. An alias for
/// #ZENDNN_ARG_DIFF_DST_0.
#define ZENDNN_ARG_DIFF_DST_LAYER ZENDNN_ARG_DIFF_DST_0

/// Gradient (diff) of the destination argument #1.
#define ZENDNN_ARG_DIFF_DST_1 146
/// A special mnemonic for gradient (diff) of RNN input recurrent hidden state
/// vector. An alias for #ZENDNN_ARG_DIFF_DST_1.
#define ZENDNN_ARG_DIFF_DST_ITER ZENDNN_ARG_DIFF_DST_1

/// Gradient (diff) of the destination argument #2.
#define ZENDNN_ARG_DIFF_DST_2 147
/// A special mnemonic for gradient (diff) of RNN input recurrent cell state
/// vector. An alias for #ZENDNN_ARG_DIFF_DST_2.
#define ZENDNN_ARG_DIFF_DST_ITER_C ZENDNN_ARG_DIFF_DST_2

/// Gradient (diff) of the weights argument #0.
#define ZENDNN_ARG_DIFF_WEIGHTS_0 161
/// A special mnemonic for primitives that have a single diff weights
/// argument. Alias for #ZENDNN_ARG_DIFF_WEIGHTS_0.
#define ZENDNN_ARG_DIFF_WEIGHTS ZENDNN_ARG_DIFF_WEIGHTS_0
/// A special mnemonic for diff of scale and shift argument of normalization
/// primitives. Alias for #ZENDNN_ARG_DIFF_WEIGHTS_0.
#define ZENDNN_ARG_DIFF_SCALE_SHIFT ZENDNN_ARG_DIFF_WEIGHTS_0
/// A special mnemonic for diff of RNN weights applied to the layer input. An
/// alias for #ZENDNN_ARG_DIFF_WEIGHTS_0.
#define ZENDNN_ARG_DIFF_WEIGHTS_LAYER ZENDNN_ARG_DIFF_WEIGHTS_0

/// Gradient (diff) of the weights argument #1.
#define ZENDNN_ARG_DIFF_WEIGHTS_1 162
/// A special mnemonic for diff of RNN weights applied to the recurrent input.
/// An alias for #ZENDNN_ARG_DIFF_WEIGHTS_1.
#define ZENDNN_ARG_DIFF_WEIGHTS_ITER ZENDNN_ARG_DIFF_WEIGHTS_1

/// Gradient (diff) of the weights argument #2.
#define ZENDNN_ARG_DIFF_WEIGHTS_2 163
/// A special mnemonic for diff of RNN weights applied to the peephole weights.
/// An alias for #ZENDNN_ARG_DIFF_WEIGHTS_2.
#define ZENDNN_ARG_DIFF_WEIGHTS_PEEPHOLE ZENDNN_ARG_DIFF_WEIGHTS_2

/// Gradient (diff) of the weights argument #3.
#define ZENDNN_ARG_DIFF_WEIGHTS_3 164
/// A special mnemonic for diff of RNN weights applied to the projection
/// weights. An alias for #ZENDNN_ARG_DIFF_WEIGHTS_3.
#define ZENDNN_ARG_DIFF_WEIGHTS_PROJECTION ZENDNN_ARG_DIFF_WEIGHTS_3

/// Gradient (diff) of the bias tensor argument.
#define ZENDNN_ARG_DIFF_BIAS 169

/// Conv + BN fusion
#define ZENDNN_ARG_BN_SCALE     177
#define ZENDNN_ARG_BN_MEAN      178
#define ZENDNN_ARG_BN_OFFSET    179

/// Output scaling factors provided at execution time.
#define ZENDNN_ARG_ATTR_OUTPUT_SCALES 513

/// Starting index for source arguments for primitives that take a variable
/// number of source arguments.
#define ZENDNN_ARG_MULTIPLE_SRC 1024
/// Starting index for destination arguments for primitives that produce a
/// variable number of destination arguments.
#define ZENDNN_ARG_MULTIPLE_DST 2048

/// Zero points provided at execution time.
#define ZENDNN_ARG_ATTR_ZERO_POINTS 4096

/// Arguments for fused depthwise convolution.
/// See @ref dev_guide_attributes_post_ops_depthwise_fusion
#define ZENDNN_ARG_ATTR_POST_OP_DW 8192

/// Starting point for a binary post operation.
#define ZENDNN_ARG_ATTR_MULTIPLE_POST_OP_BASE 16384

/// Arguments for a binary post operation. Up to 32 arguments are supported.
/// See @ref dev_guide_attributes_post_ops_binary_fusion
#define ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(idx) \
    (ZENDNN_ARG_ATTR_MULTIPLE_POST_OP_BASE * ((idx) + 1))

// XXX: next define should have a (1 << 20) = 1048576 value to preserve 5 bits
// for ZENDNN_ARG_ATTR_MULTIPLE_POST_OP argument.

/// A structure that contains an index and a memory object, and is used to pass
/// arguments to zendnn_primitive_execute().
typedef struct {
    int arg; ///< An argument index, e.g. ZENDNN_ARG_SRC
    zendnn_memory_t memory; ///< Input/output memory
} zendnn_exec_arg_t;

/// @} zendnn_api_primitives_common

/// @addtogroup zendnn_api_primitives_common
/// @{

/// Primitive descriptor query specification
///
/// For generic function zendnn_primitive_desc_query(), the type of result must
/// agree with the queried argument. The correspondence table:
///
/// Query kind                      | Type of query result
/// --------------------------------|-----------------------------
/// #zendnn_query_engine              | #zendnn_engine_t *
/// #zendnn_query_scratchpad_engine   | #zendnn_engine_t *
/// #zendnn_query_primitive_kind      | #zendnn_primitive_kind_t *
/// zendnn_query_*_s32                | int *
/// zendnn_query_*_s64                | #zendnn_dim_t * (same as int64_t *)
/// zendnn_query_*_f64                | double *
/// zendnn_query_*_str                | const char **
/// #zendnn_query_op_d                | #const_zendnn_op_desc_t *
/// zendnn_query_*_md                 | const #zendnn_memory_desc_t **
/// zendnn_query_*_\<op\>_d           | const zendnn_\<op\>_desc_t **
/// zendnn_query_*_pd                 | #const_zendnn_primitive_desc_t *
///
/// @note
///     Rule of thumb: all opaque types and structures are returned by
///     reference. All numbers are returned by value.
///
/// @warning
///     All returned references point to constant objects and are valid only
///     during the lifetime of the queried primitive descriptor. Returned objects
///     must not be destroyed by the user. If you need to keep the object longer
///     than the lifetime of the queried primitive descriptor, use
///     zendnn_primitive_desc_clone() to make a copy.
typedef enum {
    zendnn_query_undef = 0, ///< no query

    zendnn_query_engine, ///< execution engine
    zendnn_query_primitive_kind, ///< primitive kind

    zendnn_query_num_of_inputs_s32, ///< number of inputs expected
    zendnn_query_num_of_outputs_s32, ///< number of outputs expected

    zendnn_query_time_estimate_f64, ///< runtime estimation (seconds)
    zendnn_query_memory_consumption_s64, ///< memory consumption -- extra
    ///  (scratch) memory, additional to
    ///  all inputs and outputs memory
    ///  (bytes)

    zendnn_query_scratchpad_engine, ///< scratchpad engine -- engine to be used
    ///  for creating scratchpad memory

    zendnn_query_impl_info_str, ///< implementation name

    zendnn_query_reorder_src_engine, ///< source engine
    zendnn_query_reorder_dst_engine, ///< destination engine

    zendnn_query_prop_kind, ///< propagation kind

    // memory and op descriptor section
    zendnn_query_some_d = 64, ///< stub
    zendnn_query_op_d, ///< op descriptor
    zendnn_query_convolution_d, ///< convolution descriptor
    zendnn_query_deconvolution_d, ///< deconvolution descriptor
    zendnn_query_shuffle_d, ///< shuffle descriptor
    zendnn_query_eltwise_d, ///< eltwise descriptor
    zendnn_query_softmax_d, ///< softmax descriptor
    zendnn_query_pooling_d, ///< pooling descriptor
    zendnn_query_lrn_d, ///< lrn descriptor
    zendnn_query_batch_normalization_d, ///< batch normalization descriptor
    zendnn_query_layer_normalization_d, ///< layer normalization descriptor
    zendnn_query_inner_product_d, ///< inner product descriptor
    zendnn_query_rnn_d, ///< rnn descriptor
    zendnn_query_gemm_d, ///< GEMM descriptor (internal)
    zendnn_query_binary_d, ///< binary descriptor
    zendnn_query_logsoftmax_d, ///< logsoftmax descriptor
    zendnn_query_matmul_d, ///< matrix multiplication (matmul) descriptor
    zendnn_query_resampling_d, ///< resampling descriptor
    zendnn_query_pooling_v2_d, ///< pooling version 2 descriptor
    zendnn_query_reduction_d, ///< reduction descriptor
    zendnn_query_prelu_d, ///< prelu descriptor

    /* add new primitive */
    zendnn_query_embedding_bag_d, ///< embedding_bag descriptor

    // memory descriptor section
    zendnn_query_some_md = 128, ///< stub
    zendnn_query_src_md, ///< source memory desc
    zendnn_query_diff_src_md, ///< source gradient memory desc
    zendnn_query_weights_md, ///< weights memory descriptor desc
    zendnn_query_diff_weights_md, ///< weights grad. memory desc
    zendnn_query_dst_md, ///< destination memory desc
    zendnn_query_diff_dst_md, ///< destination grad. memory desc
    zendnn_query_workspace_md, ///< workspace memory desc
    zendnn_query_scratchpad_md, ///< scratchpad memory desc
    zendnn_query_exec_arg_md = 255, ///< memory desc of an execute argument

    // Max value to prevent UB for internal use only zendnn_query_t
    zendnn_query_max = 0x7fff,
} zendnn_query_t;

/// @} zendnn_api_primitives_common

/// @} zendnn_api_primitives

/// @addtogroup zendnn_api_stream
/// @{

/// @brief Stream flags.
typedef enum {
    // In-order execution.
    zendnn_stream_in_order = 0x1U,
    /// Out-of-order execution.
    zendnn_stream_out_of_order = 0x2U,
    /// Default stream configuration.
    zendnn_stream_default_flags = zendnn_stream_in_order,
} zendnn_stream_flags_t;

/// @struct zendnn_stream
/// An opaque structure to describe an execution stream.
struct zendnn_stream;
/// An execution stream handle.
typedef struct zendnn_stream *zendnn_stream_t;
/// A constant execution stream handle.
typedef const struct zendnn_stream *const_zendnn_stream_t;

/// @} zendnn_api_stream

/// @addtogroup zendnn_api_service
/// @{

/// No runtime (disabled)
#define ZENDNN_RUNTIME_NONE 0u

/// Sequential runtime (CPU only)
#define ZENDNN_RUNTIME_SEQ 1u

/// OpenMP runtime (CPU only)
#define ZENDNN_RUNTIME_OMP 2u

/// TBB runtime (CPU only)
#define ZENDNN_RUNTIME_TBB 4u

/// Threadpool runtime (CPU only)
#define ZENDNN_RUNTIME_THREADPOOL 8u

/// OpenCL runtime
#define ZENDNN_RUNTIME_OCL 256u

/// SYCL runtime
#define ZENDNN_RUNTIME_SYCL 512u

/// DPC++ runtime
#define ZENDNN_RUNTIME_DPCPP ZENDNN_RUNTIME_SYCL

/// Structure containing version information as per [Semantic
/// Versioning](https://semver.org)
typedef struct {
    int major; ///< Major version
    int minor; ///< Minor version
    int patch; ///< Patch version
    const char *hash; ///< Git hash of the sources (may be absent)
    unsigned cpu_runtime; ///< CPU runtime
    unsigned gpu_runtime; ///< GPU runtime
} zendnn_version_t;

/// Disable profiling completely
#define ZENDNN_JIT_PROFILE_NONE 0u

/// Enable VTune Amplifier integration
#define ZENDNN_JIT_PROFILE_VTUNE 1u

/// Enable Linux perf integration via perfmap files
#define ZENDNN_JIT_PROFILE_LINUX_PERFMAP 2u

/// Enable Linux perf integration via jitdump files
#define ZENDNN_JIT_PROFILE_LINUX_JITDUMP 4u

/// Instruct Linux perf integration via jitdump files to use TSC. @ref
/// ZENDNN_JIT_PROFILE_LINUX_JITDUMP must be set too for this to take effect.
#define ZENDNN_JIT_PROFILE_LINUX_JITDUMP_USE_TSC 8u

/// Enable Linux perf integration (both jitdump and perfmap)
#define ZENDNN_JIT_PROFILE_LINUX_PERF \
    (ZENDNN_JIT_PROFILE_LINUX_JITDUMP | ZENDNN_JIT_PROFILE_LINUX_PERFMAP)

/// CPU instruction set flags
typedef enum {
    /// Any ISA (excepting those listed as initial support)
    zendnn_cpu_isa_all = 0x0,

    /// Intel Streaming SIMD Extensions 4.1 (Intel SSE4.1)
    zendnn_cpu_isa_sse41 = 0x1,

    /// Intel Advanced Vector Extensions (Intel AVX)
    zendnn_cpu_isa_avx = 0x3,

    /// Intel Advanced Vector Extensions 2 (Intel AVX2)
    zendnn_cpu_isa_avx2 = 0x7,

    /// Intel Advanced Vector Extensions 512 (Intel AVX-512) subset
    /// for Intel Xeon Phi processors x200 Series.
    zendnn_cpu_isa_avx512_mic = 0xf,

    /// Intel AVX-512 subset
    /// for Intel Xeon Phi processors 7235, 7285, 7295 Series.
    zendnn_cpu_isa_avx512_mic_4ops = 0x1f,

    /// Intel AVX-512 subset for Intel Xeon Scalable processor family
    /// and Intel Core processor family.
    zendnn_cpu_isa_avx512_core = 0x27,

    /// Intel AVX-512 and Intel Deep Learning Boost (Intel DL Boost) support
    /// for Intel Xeon Scalable processor family
    /// and Intel Core processor family.
    zendnn_cpu_isa_avx512_core_vnni = 0x67,

    /// Intel AVX-512, Intel DL Boost and bfloat16 support
    /// for Intel Xeon Scalable processor family
    /// and Intel Core processor family.
    zendnn_cpu_isa_avx512_core_bf16 = 0xe7,

    /// Intel AVX-512, Intel DL Boost and bfloat16 support and
    /// Intel AMX with 8-bit integer and bfloat16 support
    /// (initial support)
    zendnn_cpu_isa_avx512_core_amx = 0x3e7,

    /// Intel AVX2 and Intel Deep Learning Boost (Intel DL Boost) support
    zendnn_cpu_isa_avx2_vnni = 0x407,

} zendnn_cpu_isa_t;

/// CPU ISA hints flags
typedef enum {
    /// No hints (use default features)
    zendnn_cpu_isa_no_hints = 0x0,

    /// Prefer to exclusively use Ymm registers for computations
    zendnn_cpu_isa_prefer_ymm = 0x1,
} zendnn_cpu_isa_hints_t;

/// @} zendnn_api_service

/// @} zendnn_api

#ifdef __cplusplus
}
#endif

#endif /* ZENDNN_TYPES_H */
