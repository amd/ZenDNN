/*******************************************************************************
* Modifications Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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
/// C API

#ifndef ZENDNN_H
#define ZENDNN_H

#include "zendnn_config.h"
#include "zendnn_types.h"
#include "zendnn_version.h"

#ifdef __cplusplus
extern "C" {
#endif

/// @addtogroup zendnn_api
/// @{

/// @addtogroup zendnn_api_primitives
/// @{

/// @addtogroup zendnn_api_primitives_common
/// @{

/// Creates a primitive descriptor iterator.
///
/// @param iterator Output primitive descriptor iterator.
/// @param op_desc Operation descriptor.
/// @param attr Primitive attributes (can be NULL).
/// @param engine Engine to use.
/// @param hint_forward_primitive_desc For backward propagation: primitive
///     descriptor for a respective forward propagation primitive. Pass NULL
///     for forward propagation.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_primitive_desc_iterator_create(
        zendnn_primitive_desc_iterator_t *iterator, const_zendnn_op_desc_t op_desc,
        const_zendnn_primitive_attr_t attr, zendnn_engine_t engine,
        const_zendnn_primitive_desc_t hint_forward_primitive_desc);

/// Advances the primitive descriptor iterator to point to the next available
/// implementation.
///
/// @param iterator A primitive descriptor iterator to advance.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
/// @returns #zendnn_iterator_ends if no more implementations available.
zendnn_status_t ZENDNN_API zendnn_primitive_desc_iterator_next(
        zendnn_primitive_desc_iterator_t iterator);

/// Fetches the current primitive descriptor from a primitive descriptor
/// iterator.
///
/// @note
///     The user is responsible for deleting the resulting primitive
///     descriptor using zendnn_primitive_desc_destroy().
///
/// @param iterator A primitive descriptor iterator.
/// @returns A primitive descriptor.
zendnn_primitive_desc_t ZENDNN_API zendnn_primitive_desc_iterator_fetch(
        const_zendnn_primitive_desc_iterator_t iterator);

/// Destroys a primitive descriptor iterator.
///
/// @param iterator Primitive descriptor iterator to destroy.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_primitive_desc_iterator_destroy(
        zendnn_primitive_desc_iterator_t iterator);

/// Creates a primitive descriptor. This function is equivalent to a sequence
/// of #zendnn_primitive_desc_iterator_create() and
/// #zendnn_primitive_desc_iterator_fetch(). In other words, the library will
/// pick the first suitable implementation.
///
/// @param primitive_desc Output primitive descriptor.
/// @param op_desc Operation descriptor.
/// @param attr Primitive attributes (can be NULL).
/// @param engine Engine to use.
/// @param hint_forward_primitive_desc For backward propagation: primitive
///     descriptor for a respective forward propagation primitive. Pass NULL
///     for forward propagation.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_primitive_desc_create(
        zendnn_primitive_desc_t *primitive_desc, const_zendnn_op_desc_t op_desc,
        const_zendnn_primitive_attr_t attr, zendnn_engine_t engine,
        const_zendnn_primitive_desc_t hint_forward_primitive_desc);

/// Clones a primitive descriptor. The resulting primitive descriptor must be
/// destroyed separately.
///
/// @param primitive_desc Output primitive descriptor.
/// @param existing_primitive_desc Primitive descriptor to clone.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_primitive_desc_clone(
        zendnn_primitive_desc_t *primitive_desc,
        const_zendnn_primitive_desc_t existing_primitive_desc);

/// Returns a constant reference to the attributes of a primitive descriptor.
///
/// @warning
///     It is an error to destroy the resulting @p attr.
///
/// @warning
///     The lifetime of an @p attr is the same as that of a @p
///     primitive_desc, so it is an error to use the @p attr once the @p
///     primitive_desc has been destroyed.
///
/// @param primitive_desc Primitive descriptor.
/// @param attr Output primitive attributes.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_primitive_desc_get_attr(
        const_zendnn_primitive_desc_t primitive_desc,
        const_zendnn_primitive_attr_t *attr);

/// Destroys a primitive descriptor.
///
/// @param primitive_desc Primitive descriptor to destroy.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_primitive_desc_destroy(
        zendnn_primitive_desc_t primitive_desc);

/// Queries a primitive descriptor for various pieces of information.
///
/// The most common use case is to query a primitive descriptor, created with
/// source, weights, and destination memory descriptors with format tags set
/// to #zendnn_format_tag_any, for the corresponding memory descriptors (in this
/// case the @p what is set to #zendnn_query_src_md, #zendnn_query_weights_md, and
/// #zendnn_query_dst_md respectively) so that it is possible to create memory
/// objects and reorder primitives if necessary.
///
/// Another typical use case is to query a primitive descriptor for workspace
/// memory descriptor (with @p what set to #zendnn_query_workspace_md). If this
/// query returns #zendnn_not_required status, then workspace memory is not
/// required.
///
/// @note
///     When querying for a memory descriptor for a scratchpad, a workspace,
///     or an optional parameter, the query will return a pointer to a zero
///     memory descriptor if the parameter is not needed.
///
/// A few other use cases:
///  - query a primitive descriptor for the underlying operation descriptor
///    (#zendnn_query_convolution_d, #zendnn_query_eltwise_d, #zendnn_query_rnn_d,
///    etc.)
///  - query a primitive descriptor for the implementation information string
///    (#zendnn_query_impl_info_str)
///  - query a primitive descriptor for the number of inputs and outputs
///    (#zendnn_query_num_of_inputs_s32 and #zendnn_query_num_of_outputs_s32
///    respectively)
///
/// @sa zendnn_query_t for more options
///
/// @param primitive_desc Primitive descriptor.
/// @param what Parameter to query.
/// @param index Index of the parameter to query for.
/// @param result Output result. The type depends on the query. For example,
///     it must be a @c zendnn_memory_desc_t* if querying for a memory
///     descriptor.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_primitive_desc_query(
        const_zendnn_primitive_desc_t primitive_desc, zendnn_query_t what,
        int index, void *result);

/// Queries primitive descriptor for a memory descriptor.
///
/// @note
///     This function is a convenience version of
///     #zendnn_primitive_desc_query().
///
/// @param primitive_desc Primitive descriptor.
/// @param what Kind of memory descriptor parameter to query for.
/// @param index Index of the parameter to query.
/// @returns A pointer to the requested memory descriptor.
/// @returns A pointer to a zero memory descriptor if the parameter is not
///          needed.
/// @returns NULL in case of any error.
///
const zendnn_memory_desc_t ZENDNN_API *zendnn_primitive_desc_query_md(
        const_zendnn_primitive_desc_t primitive_desc, zendnn_query_t what,
        int index);

/// Queries primitive descriptor for a signed 32bit int.
///
/// @note
///     This function is a convenience version of
///     #zendnn_primitive_desc_query().
///
/// @param primitive_desc Primitive descriptor.
/// @param what Kind of the value to query for.
/// @param index Index of the parameter to query.
/// @returns The requested value.
/// @returns 0 in case of any error (in particular if the queried entity is
///     not of type int32_t). Note that 0 may also be the actual returned
///     value.
int ZENDNN_API zendnn_primitive_desc_query_s32(
        const_zendnn_primitive_desc_t primitive_desc, zendnn_query_t what,
        int index);

/// Creates a primitive.
///
/// @param primitive Output primitive.
/// @param primitive_desc Primitive descriptor used to create the primitive.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_primitive_create(zendnn_primitive_t *primitive,
        const_zendnn_primitive_desc_t primitive_desc);

/// Executes a primitive.
///
/// @param primitive Primitive to execute.
/// @param stream Stream to use.
/// @param nargs Number of arguments.
/// @param args Array of arguments. Each argument is an
///     <index, #zendnn_memory_t> pair. The index is one of the `ZENDNN_ARG_*`
///     values such as `ZENDNN_ARG_SRC`. Unless runtime shapes are used (see
///     #ZENDNN_RUNTIME_DIM_VAL), the memory object must have the same memory
///     descriptor as that returned by
///     #zendnn_primitive_desc_query_md(#zendnn_query_exec_arg_md, index).
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.

/// @note If any argument in @param args is padded (padded_dims >
/// dims), the primitive execution will assume properly zero-padded
/// input arguments, and produce zero-padded output arguments.
zendnn_status_t ZENDNN_API zendnn_primitive_execute(const_zendnn_primitive_t primitive,
        zendnn_stream_t stream, int nargs, const zendnn_exec_arg_t *args);

/// Retrieves a constant reference to the primitive descriptor of a given
/// primitive.
///
/// @warning
///     It is an error to destroy the returned object. It is owned by the
///     primitive. The @c const qualifier of the returned object prevents
///     such attempts.
///
/// @param primitive Primitive to query for the primitive descriptor.
/// @param primitive_desc Output primitive descriptor.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_primitive_get_primitive_desc(
        const_zendnn_primitive_t primitive,
        const_zendnn_primitive_desc_t *primitive_desc);

/// Destroys a primitive.
///
/// @param primitive The primitive to destroy.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_primitive_destroy(zendnn_primitive_t primitive);

/// @} zendnn_api_primitives_common

/// @addtogroup zendnn_api_attributes
/// @{

/// Creates an empty (default) primitive attributes with all the parameters
/// set to their default values.
///
/// Empty attributes are implied whenever the respective argument is NULL.
///
/// @param attr Output primitive attributes.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_primitive_attr_create(zendnn_primitive_attr_t *attr);

/// Clones primitive attributes.
///
/// @param attr Output primitive attributes.
/// @param existing_attr Primitive attributes to clone.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_primitive_attr_clone(
        zendnn_primitive_attr_t *attr, const_zendnn_primitive_attr_t existing_attr);

/// Destroys primitive attributes.
///
/// @param attr Primitive attributes to destroy.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_primitive_attr_destroy(zendnn_primitive_attr_t attr);

/// Returns the primitive attributes scratchpad mode.
///
/// @param attr Primitive attributes.
/// @param mode Output scratchpad mode.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_primitive_attr_get_scratchpad_mode(
        const_zendnn_primitive_attr_t attr, zendnn_scratchpad_mode_t *mode);

/// Sets primitive attributes scratchpad mode.
///
/// @param attr Primitive attributes.
/// @param mode Scratchpad mode. The possible values are:
///     #zendnn_scratchpad_mode_library (default) and
///     #zendnn_scratchpad_mode_user.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_primitive_attr_set_scratchpad_mode(
        zendnn_primitive_attr_t attr, zendnn_scratchpad_mode_t mode);

/// Returns primitive attributes output scaling factors correspondence mask
/// and values.
///
/// @warning
///     The @p scales array is an internal part of the primitive attributes
///     @p attr, so it is an error to modify or destroy the @p scales array.
///
/// @warning
///     The lifetime of @p scales array is the same as that of the primitive
///     attributes @p attr to which it belongs, so it is an error to use
///     @p scales after @p attr is destroyed.
///
/// @param attr Primitive attributes.
/// @param count Output length of the array of scaling factors @p scales.
/// @param mask Output scaling factors correspondence mask that defines the
///     correspondence between the output tensor dimensions and the @p scales
///     vector. The set i-th bit indicates that a dedicated output scaling
///     factor is used for each index along that dimension. The mask value of
///     0 implies a common output scaling factor for the whole output tensor.
/// @param scales Output pointer to a constant array of scaling factors.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_primitive_attr_get_output_scales(
        const_zendnn_primitive_attr_t attr, zendnn_dim_t *count, int *mask,
        const float **scales);

/// Sets output scaling factors correspondence mask and values.
///
/// @note
///     The order of dimensions does not depend on how elements are laid
///     out in memory. For example:
///     - for a 2D CNN activations tensor the order is always (n, c)
///     - for a 4D CNN activations tensor the order is always (n, c, h, w)
///     - for a 5D CNN weights tensor the order is always
///        (g, oc, ic, kh, kw)
///
/// Example usage:
/// @code
///     int mb = 32, oc = 32, oh = 14, ow = 14; // convolution output params
///     float scales[oc] = { ... }; // unique output scales per output channel
///     int oc_dim = 1; // mb_dim = 0, channel_dim = 1, height_dim = 2, ...
///
///     zendnn_convolution_desc_t conv_d; // create a convolution descriptor
///
///     zendnn_primitive_attr_t attr;
///     zendnn_primitive_attr_create(&attr); // create primitive attributes
///     zendnn_primitive_attr_set_output_scales(attr, oc, 1 << oc_dim, scales);
///
///     zendnn_primitive_desc_t conv_pd;
///     zendnn_primitive_desc_create(&conv_pd, &conv_d, attr, engine, NULL);
/// @endcode
///
/// @param attr Primitive attributes.
/// @param count Length of the array of scaling factors @p scales.
/// @param mask Scaling factors correspondence mask that defines the
///     correspondence between the output tensor dimensions and the @p scales
///     array. The set i-th bit indicates that a dedicated output scaling
///     factor is used for each index along that dimension. The mask value of
///     0 implies a common output scaling factor for the whole output tensor.
/// @param scales Array of output scaling factors. If the output scaling
///     factors are known at the time of this call, this array must contain @p
///     count values and the following equality must hold:
///     \f[count = \prod\limits_{d \in mask} output.dims[d].\f]
///     Violations can only be detected when the attributes are used to create
///     a primitive descriptor.
///     If the output scaling factors are not known at the time of the call,
///     this array must contain a single #ZENDNN_RUNTIME_F32_VAL value and the
///     output scaling factors must be passed at execution time as an argument
///     with index #ZENDNN_ARG_ATTR_OUTPUT_SCALES.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_primitive_attr_set_output_scales(
        zendnn_primitive_attr_t attr, zendnn_dim_t count, int mask,
        const float *scales);

/// Returns primitive attributes scaling factors correspondence mask and values
/// for a given memory argument.
///
/// @warning
///     The output @p scales array is an internal part of the primitive
///     attributes @p attr, so it is an error to modify or destroy the @p
///     scales array.
///
/// @warning
///     The lifetime of the @p scales array is the same as that of the primitive
///     attributes @p attr to which it belongs, so it is an error to use @p
///     scales after @p attr is destroyed.
///
///
/// @param attr Primitive attributes.
/// @param arg Parameter argument index as passed to the
///     zendnn_primitive_execute() call.
/// @param count Output length of the array of scaling factors @p scales.
/// @param mask Output scaling factors correspondence mask that defines the
///     correspondence between the output tensor dimensions and the @p
///     scales array. The set i-th bit indicates that a dedicated output scaling
///     factor is used for each index along that dimension. The mask value of 0
///     implies a common scaling factor for the whole output tensor.
/// @param scales Output pointer to a constant array of float scaling factors.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_primitive_attr_get_scales(
        zendnn_primitive_attr_t attr, int arg, zendnn_dim_t *count, int *mask,
        const float **scales);

/// Sets primitive attributes scaling factors for primitive operations for a
/// given memory argument.
///
/// @sa zendnn_primitive_attr_set_output_scales
///
///
/// @param attr Primitive attributes.
/// @param arg Parameter argument index as passed to the
///     zendnn_primitive_execute() call.
/// @param count Length of the array of scaling factors @p scales.
/// @param mask Scaling factors correspondence mask that defines the
///     correspondence between the tensor dimensions and the @p scales array.
///     The set i-th bit indicates that a dedicated scaling factor is used for
///     each index along that dimension. Set the mask to 0 to use a common
///     scaling factor for the whole output tensor.
/// @param scales Constant array of float scaling factors. This array must
///     contain @p count scales and the following equality must hold:
///     \f[count = \prod\limits_{d \in mask} output.dims[d].\f]
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_primitive_attr_set_scales(
        zendnn_primitive_attr_t attr, int arg, zendnn_dim_t count, int mask,
        const float *scales);

/// Returns @p count, correspondence zero point @p mask, and a pointer to a
/// constant int32_t array of @p zero_points for given @p attr and memory
/// argument (index), previously set by zendnn_primitive_attr_set_zero_points.
///
/// @warning
///     The output @p zero_points array is an internal part of the primitive
///     attributes @p attr, so it is an error to modify or destroy the @p
///     zero_points array.
///
/// @warning
///     The lifetime of @p zero_points array is the same as that of the
///     primitive attributes @p attr to which it belongs, so it is an error
///     to use @p zero_points after @p attr is destroyed.
///
///
/// @param attr Primitive attributes.
/// @param arg Parameter argument index as passed to the
///     zendnn_primitive_execute() call.
/// @param count Output length of the array of zero points @p zero_points.
/// @param mask Output zero points correspondence mask that defines the
///     correspondence between the output tensor dimensions and the @p
///     zero_points array. The set i-th bit indicates that a dedicated output
///     zero point is used for each index along that dimension. The mask
///     value of 0 implies a common zero point for the whole output tensor.
/// @param zero_points Output pointer to a constant array of int32_t zero
///     points.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_primitive_attr_get_zero_points(
        const_zendnn_primitive_attr_t attr, int arg, zendnn_dim_t *count, int *mask,
        const int32_t **zero_points);

/// Sets primitive attributes zero points for primitive operations for a given
/// memory argument.
///
/// @sa zendnn_primitive_attr_set_output_scales
///
///
/// @param attr Primitive attributes.
/// @param arg Parameter argument index as passed to the
///     zendnn_primitive_execute() call.
/// @param count Length of the array of zero points @p zero_points.
/// @param mask Zero point correspondence mask that defines the
///     correspondence between the tensor dimensions and the @p
///     zero_points array. The set i-th bit indicates that a dedicated
///     zero point is used for each index along that dimension. Set the
///     mask to 0 to use a common zero point for the whole output tensor.
/// @param zero_points Constant array of int32_t zero points. If the zero
///     points are known at the time of this call, this array must contain @p
///     count zero points and the following equality must hold:
///     \f[count = \prod\limits_{d \in mask} output.dims[d].\f]
///     If the zero points are not known at the time of the call, this array
///     must contain a single #ZENDNN_RUNTIME_S32_VAL and the zero points must
///     be passed at execution time as an argument with index
///     #ZENDNN_ARG_ATTR_ZERO_POINTS.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_primitive_attr_set_zero_points(
        zendnn_primitive_attr_t attr, int arg, zendnn_dim_t count, int mask,
        const int32_t *zero_points);

/// Returns primitive attributes post-ops.
///
/// @warning
///     The output @p post_ops points to the internal @p attr field, so it is
///     an error to modify or destroy them. The lifetime of @p post_ops is
///     the same as that of the @p attr it belongs to, so it is an error to
///     use @p post_ops after @p attr has been destroyed.
///
/// @param attr Primitive attributes.
/// @param post_ops Output post-ops.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_primitive_attr_get_post_ops(
        const_zendnn_primitive_attr_t attr, const_zendnn_post_ops_t *post_ops);

/// Sets primitive attributes post-ops.
///
/// @note
///     There is no way to check whether the post-ops would be supported by
///     the target primitive. Any error will be reported by the
///     zendnn_primitive_desc_create() function call.
///
/// @param attr Primitive attributes.
/// @param post_ops Post-ops to set.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_primitive_attr_set_post_ops(
        zendnn_primitive_attr_t attr, const_zendnn_post_ops_t post_ops);

/// Creates empty post-ops sequence.
///
/// @param post_ops Output post-ops.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_post_ops_create(zendnn_post_ops_t *post_ops);

/// Destroys post-ops.
///
/// @param post_ops Post-ops to destroy.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_post_ops_destroy(zendnn_post_ops_t post_ops);

/// Returns the length of post-ops.
///
/// @param post_ops Post-ops.
/// @returns The number of post-ops entries.
int ZENDNN_API zendnn_post_ops_len(const_zendnn_post_ops_t post_ops);

/// Returns the kind of a post-op entry.
///
/// @param post_ops Post-ops.
/// @param index Post-op entry index.
/// @returns The kind of the post-op with the specified index.
/// @returns #zendnn_undefined_primitive if there is no post-op at the specified
///     index.
zendnn_primitive_kind_t ZENDNN_API zendnn_post_ops_get_kind(
        const_zendnn_post_ops_t post_ops, int index);

/// Appends an accumulation (sum) to post-ops. Prior to accumulating the
/// result, the previous value is multiplied by a scale.
///
/// The kind of this post-op is #zendnn_sum.
///
/// This feature may improve performance for cases like residual learning
/// blocks, where the result of convolution is accumulated to the previously
/// computed activations. The parameter @p scale may be used for the
/// integer-based computations when the result and previous activations have
/// different logical scaling factors.
///
/// In the simplest case when the accumulation is the only post-op, the
/// computations would be:
///
///     dst[:] <- scale * dst[:] + op(...) // instead of dst[:] <- op(...)
///
/// @note
///     This post-op executes in-place and does not change the
///     destination layout.
///
/// @param post_ops Post-ops.
/// @param scale Accumulation scaling factor.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_post_ops_append_sum(
        zendnn_post_ops_t post_ops, float scale);

/// Appends an accumulation v2 (sum) to post-ops. Prior to accumulating the
/// result, the previous value is multiplied by a scale.
///
/// The kind of this post-op is #zendnn_sum.
///
/// This feature may improve performance for cases like residual learning
/// blocks, where the result of convolution is accumulated to the previously
/// computed activations. The parameter @p scale may be used for the
/// integer-based computations when the result and previous activations have
/// different logical scaling factors.
///
/// In the simplest case when the accumulation is the only post-op, the
/// computations would be:
///
///     dst[:] <- scale * dst[:] + op(...) // instead of dst[:] <- op(...)
///
/// If @p data_type is specified, original dst tensor will be reinterpreted
/// as a tensor with provided data type. Since it is reinterpretation,
/// data_type and dst data type should have same size.
/// As a result, computations would be:
///
///     dst[:] <- scale * as_data_type(dst[:]) + op(...)
///                                        // instead of dst[:] <- op(...)
/// @note
///     This post-op executes in-place and does not change the
///     destination layout.
///
/// @param post_ops Post-ops.
/// @param scale Accumulation scaling factor.
/// @param data_type Accumulation data_type.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_post_ops_append_sum_v2(
        zendnn_post_ops_t post_ops, float scale, zendnn_data_type_t data_type);

/// Returns the parameters of an accumulation (sum) post-op.
///
/// @param post_ops Post-ops.
/// @param index Index of the sum post-op.
/// @param scale Output accumulation scaling factor.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
/// @returns #zendnn_invalid_arguments if @p index does not refer to a sum
///     post-op.
zendnn_status_t ZENDNN_API zendnn_post_ops_get_params_sum(
        const_zendnn_post_ops_t post_ops, int index, float *scale);

/// Returns the parameters of an accumulation (sum) post-op with
/// a data type parameter.
///
/// @param post_ops Post-ops.
/// @param index Index of the sum post-op.
/// @param scale Output accumulation scaling factor.
/// @param data_type Data type for accumulation.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_post_ops_get_params_sum_v2(
        const_zendnn_post_ops_t post_ops, int index, float *scale,
        zendnn_data_type_t *data_type);

/// Appends an elementwise post-op.
///
/// The kind of this post operation is #zendnn_eltwise.
///
/// In the simplest case when the elementwise is the only post operation, the
/// computations would be:
///
///     dst[:] <- scale * eltwise_op (op(...)) // instead of dst[:] <- op(...)
///
/// where eltwise_op is configured with the given parameters.
///
/// @param post_ops Post-ops.
/// @param scale Scaling factor.
/// @param alg_kind Elementwise algorithm for the post-op.
/// @param alpha Alpha parameter for the elementwise algorithm.
/// @param beta Beta parameter for the elementwise algorithm.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_post_ops_append_eltwise(zendnn_post_ops_t post_ops,
        float scale, zendnn_alg_kind_t alg_kind, float alpha, float beta);

/// Returns the parameters of an elementwise post-op.
///
/// @param post_ops Post-ops.
/// @param index Index of the elementwise post-op.
/// @param scale Output scaling factor.
/// @param alg_kind Output elementwise algorithm kind.
/// @param alpha Output alpha parameter for the elementwise algorithm.
/// @param beta Output beta parameter for the elementwise algorithm.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
/// @returns #zendnn_invalid_arguments if @p index does not refer to an
///     elementwise post-op.
zendnn_status_t ZENDNN_API zendnn_post_ops_get_params_eltwise(
        const_zendnn_post_ops_t post_ops, int index, float *scale,
        zendnn_alg_kind_t *alg_kind, float *alpha, float *beta);

/// Appends a depthwise post-op convolution with stride 1.
///
/// This post-op can only be fused with a 2D 1x1 convolution (convolution with
/// weights spatial dimension equal to 1 i.e., kh=kw=1).
///
/// The kind of this post-op is #zendnn_convolution.
///
/// The number of outputs for primitive remain same as before. The output size
/// remain same as the original primitive due to stride=1.
///
/// The Post-op can be defined as:
///
///      dst[:] <- scales * (conv_dw(conv_1x1))
///
/// See @ref dev_guide_attributes_post_ops_depthwise and
/// @ref dev_guide_attributes_post_ops_depthwise_fusion for more info.
///
/// @param post_ops Post-ops.
/// @param weights_data_type Weights data type of depthwise post-op
/// @param bias_data_type Bias data type of depthwise post-op
/// @param dst_data_type Output data type of depthwise post-op
/// @param count Output length of the array of scaling factors @p scales.
/// @param mask Output scaling factors correspondence mask that defines the
///     correspondence between the output tensor dimensions and the @p
///     scales array. The set i-th bit indicates that a dedicated output scaling
///     factor is used for each index along that dimension. The mask value of 0
///     implies a common scaling factor for the whole output tensor.
/// @param scales Output pointer to a constant array of float scaling factors.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise
zendnn_status_t ZENDNN_API zendnn_post_ops_append_dw_k3s1p1(zendnn_post_ops_t post_ops,
        zendnn_data_type_t weights_data_type, zendnn_data_type_t bias_data_type,
        zendnn_data_type_t dst_data_type, zendnn_dim_t count, int mask,
        const float *scales);

/// Returns the parameters of an depthwise post-op with stride 1.
///
/// @param post_ops Post-ops.
/// @param index Index of the elementwise post-op.
/// @param weights_data_type Weights data type of depthwise post-op
/// @param bias_data_type Bias data type of depthwise post-op
/// @param dst_data_type Output data type of depthwise post-op
/// @param count Output length of the array of scaling factors @p scales.
/// @param mask Output scaling factors correspondence mask that defines the
///     correspondence between the output tensor dimensions and the @p
///     scales array. The set i-th bit indicates that a dedicated output scaling
///     factor is used for each index along that dimension. The mask value of 0
///     implies a common scaling factor for the whole output tensor.
/// @param scales Output pointer to a constant array of float scaling factors.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise
zendnn_status_t ZENDNN_API zendnn_post_ops_get_params_dw_k3s1p1(
        const_zendnn_post_ops_t post_ops, int index,
        zendnn_data_type_t *weights_data_type, zendnn_data_type_t *bias_data_type,
        zendnn_data_type_t *dst_data_type, zendnn_dim_t *count, int *mask,
        const float **scales);

/// Appends a depthwise post-op convolution with stride 2.
///
/// This post-op can only be fused with a 2D 1x1 convolution (convolution with
/// weights spatial dimension equal to 1 i.e., kh=kw=1).
///
/// The kind of this post-op is #zendnn_convolution.
///
/// The number of outputs for primitive remain same as before. The output
/// spatial size can be derived as below:
///
/// output_height = ceil(output_height_1x1_convolution, stride)
/// output_width = ceil(output_width_1x1_convolution, stride)
///
/// The Post-op can be defined as:
///
///      dst[:] <- scales * (conv_dw(conv_1x1))
///
/// See @ref dev_guide_attributes_post_ops_depthwise and
/// @ref dev_guide_attributes_post_ops_depthwise_fusion for more info.
///
/// @param post_ops Post-ops.
/// @param weights_data_type Weights data type of depthwise post-op
/// @param bias_data_type Bias data type of depthwise post-op
/// @param dst_data_type Output data type of depthwise post-op
/// @param count Output length of the array of scaling factors @p scales.
/// @param mask Output scaling factors correspondence mask that defines the
///     correspondence between the output tensor dimensions and the @p
///     scales array. The set i-th bit indicates that a dedicated output scaling
///     factor is used for each index along that dimension. The mask value of 0
///     implies a common scaling factor for the whole output tensor.
/// @param scales Output pointer to a constant array of float scaling factors.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise
zendnn_status_t ZENDNN_API zendnn_post_ops_append_dw_k3s2p1(zendnn_post_ops_t post_ops,
        zendnn_data_type_t weights_data_type, zendnn_data_type_t bias_data_type,
        zendnn_data_type_t dst_data_type, zendnn_dim_t count, int mask,
        const float *scales);

/// Returns the parameters of an depthwise post-op with stride 2.
///
/// @param post_ops Post-ops.
/// @param index Index of the elementwise post-op.
/// @param weights_data_type Weights data type of depthwise post-op
/// @param bias_data_type Bias data type of depthwise post-op
/// @param dst_data_type Output data type of depthwise post-op
/// @param count Output length of the array of scaling factors @p scales.
/// @param mask Output scaling factors correspondence mask that defines the
///     correspondence between the output tensor dimensions and the @p
///     scales array. The set i-th bit indicates that a dedicated output scaling
///     factor is used for each index along that dimension. The mask value of 0
///     implies a common scaling factor for the whole output tensor.
/// @param scales Output pointer to a constant array of float scaling factors.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise
zendnn_status_t ZENDNN_API zendnn_post_ops_get_params_dw_k3s2p1(
        const_zendnn_post_ops_t post_ops, int index,
        zendnn_data_type_t *weights_data_type, zendnn_data_type_t *bias_data_type,
        zendnn_data_type_t *dst_data_type, zendnn_dim_t *count, int *mask,
        const float **scales);

/// Appends a binary post-op.
///
/// The kind of this post operation is #zendnn_binary.
///
/// In the simplest case when the binary is the only post operation, the
/// computations would be:
///
///     dst[:] <- binary_op (dst[:], another_input[:])
///
/// where binary_op is configured with the given parameters. binary_op supports
/// broadcast semantics for a second operand.
///
/// @param post_ops Post-ops.
/// @param alg_kind Binary algorithm for the post-op.
/// @param src1_desc Memory descriptor of a second operand.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_post_ops_append_binary(zendnn_post_ops_t post_ops,
        zendnn_alg_kind_t alg_kind, const zendnn_memory_desc_t *src1_desc);

/// Returns the parameters of a binary post-op.
///
/// @param post_ops Post-ops.
/// @param index Index of the binary post-op.
/// @param alg_kind Output binary algorithm kind.
/// @param src1_desc Output memory descriptor of a second operand.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
/// @returns #zendnn_invalid_arguments if @p index does not refer to a binary
///     post-op.
zendnn_status_t ZENDNN_API zendnn_post_ops_get_params_binary(
        const_zendnn_post_ops_t post_ops, int index, zendnn_alg_kind_t *alg_kind,
        const zendnn_memory_desc_t **src1_desc);

/// @} zendnn_api_attributes

/// @} zendnn_api_primitives

/// @addtogroup zendnn_api_memory
/// @{

/// Initializes a memory descriptor using dimensions and strides.
///
/// @note
///     As always, the logical order of dimensions corresponds to the `abc...`
///     format tag, and the physical meaning of the dimensions depends on both
///     the primitive that consumes the memory and the context of that
///     consumption.
///
/// @param memory_desc Output memory descriptor.
/// @param ndims Number of dimensions
/// @param dims Array of dimensions.
/// @param data_type Elements data type.
/// @param strides Strides in each dimension.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_memory_desc_init_by_strides(
        zendnn_memory_desc_t *memory_desc, int ndims, const zendnn_dims_t dims,
        zendnn_data_type_t data_type, const zendnn_dims_t strides);

/// Initializes a memory descriptor using dimensions and memory format tag.
///
/// @note
///     As always, the logical order of dimensions corresponds to the `abc...`
///     format tag, and the physical meaning of the dimensions depends on both
///     the primitive that consumes the memory and the context of that
///     consumption.
///
/// @param memory_desc Output memory descriptor.
/// @param ndims Number of dimensions
/// @param dims Array of dimensions.
/// @param data_type Elements data type.
/// @param tag Memory format tag. Can be #zendnn_format_tag_any which would
///     allow a primitive to chose the final memory format. In this case the
///     format_kind field of the memory descriptor would be set to
///     #zendnn_format_kind_any.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_memory_desc_init_by_tag(
        zendnn_memory_desc_t *memory_desc, int ndims, const zendnn_dims_t dims,
        zendnn_data_type_t data_type, zendnn_format_tag_t tag);

/// Initializes a memory descriptor for a region inside an area
/// described by an existing memory descriptor.
///
/// @warning
///     Some combinations of physical memory layout and/or offsets or dims may
///     result in a failure to create a submemory.
//
/// @param memory_desc Output memory descriptor.
/// @param parent_memory_desc An existing memory descriptor.
/// @param dims Sizes of the region.
/// @param offsets Offsets to the region from the encompassing
///     memory object in each dimension
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_memory_desc_init_submemory(
        zendnn_memory_desc_t *memory_desc,
        const zendnn_memory_desc_t *parent_memory_desc, const zendnn_dims_t dims,
        const zendnn_dims_t offsets);

/// Initializes a memory descriptor by reshaping an existing one. The new
/// memory descriptor inherits the data type. This operation is valid only for
/// memory descriptors that have format_kind set to #zendnn_blocked or
/// #zendnn_format_kind_any.
///
/// The operation ensures the transformation of the physical memory format
/// corresponds to the transformation of the logical dimensions. If such
/// transformation is impossible, the function returns #zendnn_invalid_arguments.
///
/// The reshape operation can be described as a combination of the following
/// basic operations:
/// 1. Add a dimension of size `1`. This is always possible.
/// 2. Remove a dimension of size `1`. This is possible only if the dimension
///    has no padding (i.e. `padded_dims[dim] == dims[dim] && dims[dim] == 1`).
/// 3. Split a dimension into multiple ones. This is possible only if the size
///    of the dimension is exactly equal to the product of the split ones and
///    the dimension does not have padding (i.e.
///    `padded_dims[dim] = dims[dim]`).
/// 4. Joining multiple consecutive dimensions into a single one. As in the
///    cases above, this requires that the dimensions do not have padding and
///    that the memory format is such that in physical memory these dimensions
///    are dense and have the same order as their logical counterparts. This
///    also assumes that these dimensions are not blocked.
///    - Here, dense means:
///      `stride for dim[i] == (stride for dim[i + 1]) * dim[i + 1]`;
///    - And same order means:
///      `i < j` if and only if `stride for dim[j] <= stride for dim[i]`.
///
/// @warning
///     Some combinations of physical memory layout and/or offsets or
///     dimensions may result in a failure to make a reshape.
///
/// @param out_memory_desc Output memory descriptor.
/// @param in_memory_desc An existing memory descriptor. Must have format_kind
///     set to #zendnn_blocked or #zendnn_format_kind_any.
/// @param ndims Number of dimensions for the output memory descriptor.
/// @param dims Dimensions for the output memory descriptor.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_memory_desc_reshape(
        zendnn_memory_desc_t *out_memory_desc,
        const zendnn_memory_desc_t *in_memory_desc, int ndims,
        const zendnn_dims_t dims);

/// Initializes a memory descriptor by permuting axes in an existing one.
///
/// The physical memory layout representation is adjusted accordingly to
/// maintain the consistency between the logical and physical parts of the
/// memory descriptor.
///
/// The new memory descriptor inherits the data type. This operation is valid
/// only for memory descriptors that have format_kind set to #zendnn_blocked or
/// #zendnn_format_kind_any.
///
/// The logical axes will be permuted in the following manner:
/// ```
/// for (i: 0 .. in_memory_desc->ndims)
///     out_memory_desc->dims[permutation[i]] = in_memory_desc->dims[i];
/// ```
///
/// Example:
/// @code
///     zendnn_memory_desc_t in_md, out_md, expect_out_md;
///
///     const int permutation[] = {1, 0}; // swap the first and the second axes
///
///     zendnn_dims_t in_dims = {2, 3}, out_dims = {3, 2};
///     zendnn_format_tag_t in_tag = zendnn_ab, out_tag = zendnn_ba;
///
///     zendnn_memory_desc_init_by_tag(
///             &in_md, 2, in_dims, data_type, in_tag);
///     zendnn_memory_desc_init_by_tag(
///             &expect_out_md, 2, out_dims, data_type, out_tag);
///
///     zendnn_memory_desc_permute_axes(&out_md, in_md, permutation);
///     assert(zendnn_memory_desc_equal(&out_md, &expect_out_md));
/// @endcode
///
/// @param out_memory_desc Output memory descriptor.
/// @param in_memory_desc An existing memory descriptor. Must have format_kind
///     set to #zendnn_blocked or #zendnn_format_kind_any.
/// @param permutation Axes permutation (of size `in_memory_desc->ndims`).
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_memory_desc_permute_axes(
        zendnn_memory_desc_t *out_memory_desc,
        const zendnn_memory_desc_t *in_memory_desc, const int *permutation);

/// Compares two memory descriptors.
///
/// Use this function to identify whether a reorder is required between the
/// two memories
///
/// @param lhs Left-hand side of the comparison.
/// @param rhs Right-hand side of the comparison.
/// @returns 1 if the descriptors are the same.
/// @returns 0 if the descriptors are different.
int ZENDNN_API zendnn_memory_desc_equal(
        const zendnn_memory_desc_t *lhs, const zendnn_memory_desc_t *rhs);

/// Returns the size of a memory descriptor.
///
/// @param memory_desc Memory descriptor.
/// @returns The number of bytes required for memory described by a memory
///     descriptor.
size_t ZENDNN_API zendnn_memory_desc_get_size(
        const zendnn_memory_desc_t *memory_desc);

/// Returns the size of data type.
///
/// @param data_type Data type.
/// @returns The number of bytes occupied by data type.
size_t ZENDNN_API zendnn_data_type_size(zendnn_data_type_t data_type);

/// Creates a memory object.
///
/// Unless @p handle is equal to ZENDNN_MEMORY_NONE, the constructed memory
/// object will have the underlying buffer set. In this case, the buffer will
/// be initialized as if zendnn_memory_set_data_handle() had been called.
///
/// @sa zendnn_memory_set_data_handle()
///
/// @param memory Output memory object.
/// @param memory_desc Memory descriptor.
/// @param engine Engine to use.
/// @param handle Handle of the memory buffer to use as an underlying storage.
///     - A pointer to the user-allocated buffer. In this case the library
///       doesn't own the buffer.
///     - The ZENDNN_MEMORY_ALLOCATE special value. Instructs the library to
///       allocate the buffer for the memory object. In this case the library
///       owns the buffer.
///     - ZENDNN_MEMORY_NONE to create zendnn_memory without an underlying buffer.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_memory_create(zendnn_memory_t *memory,
        const zendnn_memory_desc_t *memory_desc, zendnn_engine_t engine,
        void *handle);

/// Returns the memory descriptor for a memory object.
///
/// @param memory Memory object.
/// @param memory_desc Output memory descriptor (a copy).
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_memory_get_memory_desc(
        const_zendnn_memory_t memory, const zendnn_memory_desc_t **memory_desc);

/// Returns the engine of a memory object.
///
/// @param memory Memory object.
/// @param engine Output engine on which the memory is located.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_memory_get_engine(
        const_zendnn_memory_t memory, zendnn_engine_t *engine);

/// Maps a memory object and returns a host-side pointer to a memory buffer
/// with a copy of its contents.
///
/// Mapping enables explicit direct access to memory contents for the engines
/// that do not support it implicitly.
///
/// Mapping is an exclusive operation - a memory object cannot be used in
/// other operations until this memory object is unmapped.
///
/// @note
///     Any primitives working with @p memory should be completed before
///     the memory is mapped. Use zendnn_stream_wait to synchronize the
///     corresponding execution stream.
///
/// @note
///     The zendnn_memory_map_data() and zendnn_memory_unmap_data() functions are
///     mainly provided for debug and testing purposes, and their performance
///     may be suboptimal.
///
/// @param memory Memory object.
/// @param mapped_ptr Output pointer to the mapped buffer.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_memory_map_data(
        const_zendnn_memory_t memory, void **mapped_ptr);

/// Unmaps a memory object and writes back any changes made to the previously
/// mapped memory buffer. The pointer to the mapped buffer must be obtained
/// via the zendnn_memory_map_data() call.
///
/// @note
///     The zendnn_memory_map_data() and zendnn_memory_unmap_data() functions are
///     mainly provided for debug and testing purposes, and their performance
///     may be suboptimal.
///
/// @param memory Memory object.
/// @param mapped_ptr Pointer to the mapped buffer that must have been
///     obtained using the zendnn_memory_map_data() function.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_memory_unmap_data(
        const_zendnn_memory_t memory, void *mapped_ptr);

/// Returns memory object's data handle.
///
/// @param memory Memory object.
/// @param handle Output data handle. For the CPU engine, the data handle is a
///     pointer to the actual data. For OpenCL it is a cl_mem.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_memory_get_data_handle(
        const_zendnn_memory_t memory, void **handle);

/// Sets the underlying memory buffer.
///
/// See the description of zendnn_memory_set_data_handle_v2() for more details.
///
/// @param memory Memory object.
/// @param handle Data handle. For the CPU engine, the data handle is a
///     pointer to the actual data. For OpenCL it is a `cl_mem`.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_memory_set_data_handle(
        zendnn_memory_t memory, void *handle);

/// Sets the underlying memory buffer.
///
/// @param memory Memory object.
/// @param handle Data handle. For the CPU engine, the data handle is a
///     pointer to the actual data. For OpenCL it is a `cl_mem`.
/// @param stream Stream to use to execute padding in.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_memory_set_data_handle_v2(
        zendnn_memory_t memory, void *handle, zendnn_stream_t stream);

/// Destroys a memory object.
///
/// @param memory Memory object to destroy.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_memory_destroy(zendnn_memory_t memory);

/// @} zendnn_api_memory

/// @addtogroup zendnn_api_primitives
/// @{

/// @addtogroup zendnn_api_reorder
/// @{

/// Creates a primitive descriptor for a reorder primitive.
///
/// @param reorder_primitive_desc Output primitive descriptor.
/// @param src_desc Source memory descriptor.
/// @param src_engine Engine on which the source memory object will be
///     located.
/// @param dst_desc Destination memory descriptor.
/// @param dst_engine Engine on which the destination memory object
///     will be located.
/// @param attr Primitive attributes to use (can be NULL).
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_reorder_primitive_desc_create(
        zendnn_primitive_desc_t *reorder_primitive_desc,
        const zendnn_memory_desc_t *src_desc, zendnn_engine_t src_engine,
        const zendnn_memory_desc_t *dst_desc, zendnn_engine_t dst_engine,
        const_zendnn_primitive_attr_t attr);

/// @} zendnn_api_reorder

/// @addtogroup zendnn_api_concat
/// @{

/// Creates a primitive descriptor for an out-of-place concatenation
/// primitive.
///
/// @param concat_primitive_desc Output primitive descriptor.
/// @param dst_desc Destination memory descriptor.
/// @param n Number of source parameters.
/// @param concat_dimension Source tensors will be concatenated over
///     dimension with this index. Note that order of dimensions does
///     not depend on memory format.
/// @param src_descs Array of source memory descriptors with @p n elements.
/// @param attr Primitive attributes to use (can be NULL).
/// @param engine Engine to use.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_concat_primitive_desc_create(
        zendnn_primitive_desc_t *concat_primitive_desc,
        const zendnn_memory_desc_t *dst_desc, int n, int concat_dimension,
        const zendnn_memory_desc_t *src_descs, const_zendnn_primitive_attr_t attr,
        zendnn_engine_t engine);

/// @} zendnn_api_concat

/// @addtogroup zendnn_api_sum
/// @{

/// Creates a primitive descriptor for an (out-of-place) sum primitive.
///
/// @param sum_primitive_desc Output primitive descriptor.
/// @param dst_desc Destination memory descriptor.
/// @param n Number of source parameters.
/// @param scales Vector of scales to multiply data in each source
///     memory by.
/// @param src_descs Array of source memory descriptors having @p n elements.
/// @param attr Primitive attributes to use (can be NULL).
/// @param engine Engine to use.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_sum_primitive_desc_create(
        zendnn_primitive_desc_t *sum_primitive_desc,
        const zendnn_memory_desc_t *dst_desc, int n, const float *scales,
        const zendnn_memory_desc_t *src_descs, const_zendnn_primitive_attr_t attr,
        zendnn_engine_t engine);

/// @} zendnn_api_sum

/// @addtogroup zendnn_api_binary
/// @{

/// Initializes a descriptor for a binary primitive.
///
/// @note
///     Memory descriptor @p dst_desc is allowed to be initialized with
///     #zendnn_format_tag_any or with format_kind set to #zendnn_format_kind_any.
///
/// @note
///     Both memory descriptors must have the same number of dimensions.
///     Element broadcasting is supported for memory descriptor @p src1_desc
///     and are applied to @ src1_desc dimensions that have size equal to 1.
///
/// @param binary_desc Output descriptor for a binary primitive.
/// @param alg_kind Algorithm kind. Valid values are #zendnn_binary_add,
///     #zendnn_binary_mul, #zendnn_binary_max, #zendnn_binary_min, #zendnn_binary_div,
///     #zendnn_binary_sub, #zendnn_binary_ge, #zendnn_binary_gt, #zendnn_binary_le,
///     #zendnn_binary_lt, #zendnn_binary_eq and #zendnn_binary_ne.
/// @param src0_desc Source 0 memory descriptor.
/// @param src1_desc Source 1 memory descriptor.
/// @param dst_desc Destination memory descriptor.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_binary_desc_init(zendnn_binary_desc_t *binary_desc,
        zendnn_alg_kind_t alg_kind, const zendnn_memory_desc_t *src0_desc,
        const zendnn_memory_desc_t *src1_desc,
        const zendnn_memory_desc_t *dst_desc);

/// @} zendnn_api_binary

/// @addtogroup zendnn_api_convolution
/// @{

/// Initializes a descriptor for a convolution forward propagation primitive.
///
/// @note
///     Memory descriptors can be initialized with
///     #zendnn_format_tag_any or with format_kind set to #zendnn_format_kind_any.
///
/// Arrays @p strides, @p padding_l, and @p padding_r contain values for
/// spatial dimensions only and hence must have the same number of elements as
/// there are spatial dimensions. The order of values is the same as in the
/// tensor: depth (for 3D tensors), height (for 3D and 2D tensors), and width.
///
/// @param conv_desc Output descriptor for a convolution primitive.
/// @param prop_kind Propagation kind. Possible values are
///     #zendnn_forward_training and #zendnn_forward_inference.
/// @param alg_kind Convolution algorithm. Possible values are
///     #zendnn_convolution_direct, #zendnn_convolution_winograd,
///     #zendnn_convolution_auto.
/// @param src_desc Source memory descriptor.
/// @param weights_desc Weights memory descriptor.
/// @param bias_desc Bias memory descriptor. Passing NULL, a zero memory
///     descriptor, or a memory descriptor with format_kind set to
///     #zendnn_format_kind_undef disables the bias term.
/// @param dst_desc Destination memory descriptor.
/// @param strides Array of strides for spatial dimension.
/// @param padding_l Array of padding values for low indices for each spatial
///     dimension `([[front,] top,] left)`.
/// @param padding_r Array of padding values for high indices for each spatial
///     dimension `([[back,] bottom,] right)`. Can be NULL in which case
///     padding is assumed to be symmetrical.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_convolution_forward_desc_init(
        zendnn_convolution_desc_t *conv_desc, zendnn_prop_kind_t prop_kind,
        zendnn_alg_kind_t alg_kind, const zendnn_memory_desc_t *src_desc,
        const zendnn_memory_desc_t *weights_desc,
        const zendnn_memory_desc_t *bias_desc, const zendnn_memory_desc_t *dst_desc,
        const zendnn_dims_t strides, const zendnn_dims_t padding_l,
        const zendnn_dims_t padding_r);

/// Initializes a descriptor for a dilated convolution forward propagation
/// primitive.
///
/// @note
///     Memory descriptors can be initialized with
///     #zendnn_format_tag_any or with format_kind set to #zendnn_format_kind_any.
///
/// Arrays @p strides, @p dilates, @p padding_l, and @p padding_r contain
/// values for spatial dimensions only and hence must have the same number of
/// elements as there are spatial dimensions. The order of values is the same
/// as in the tensor: depth (for 3D tensors), height (for 3D and 2D tensors),
/// and width.
///
/// @param conv_desc Output descriptor for a convolution primitive.
/// @param prop_kind Propagation kind. Possible values are
///     #zendnn_forward_training and #zendnn_forward_inference.
/// @param alg_kind Convolution algorithm. Possible values are
///     #zendnn_convolution_direct, #zendnn_convolution_winograd,
///     #zendnn_convolution_auto.
/// @param src_desc Source memory descriptor.
/// @param weights_desc Weights memory descriptor.
/// @param bias_desc Bias memory descriptor. Passing NULL, a zero memory
///     descriptor, or a memory descriptor with format_kind set to
///     #zendnn_format_kind_undef disables the bias term.
/// @param dst_desc Destination memory descriptor.
/// @param strides Array of strides for spatial dimension.
/// @param dilates Array of dilations for spatial dimension. A zero value
///     means no dilation in the corresponding dimension.
/// @param padding_l Array of padding values for low indices for each spatial
///     dimension `([[front,] top,] left)`.
/// @param padding_r Array of padding values for high indices for each spatial
///     dimension `([[back,] bottom,] right)`. Can be NULL in which case
///     padding is considered to be symmetrical.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_dilated_convolution_forward_desc_init(
        zendnn_convolution_desc_t *conv_desc, zendnn_prop_kind_t prop_kind,
        zendnn_alg_kind_t alg_kind, const zendnn_memory_desc_t *src_desc,
        const zendnn_memory_desc_t *weights_desc,
        const zendnn_memory_desc_t *bias_desc, const zendnn_memory_desc_t *dst_desc,
        const zendnn_dims_t strides, const zendnn_dims_t dilates,
        const zendnn_dims_t padding_l, const zendnn_dims_t padding_r);

zendnn_status_t ZENDNN_API zendnn_fused_convolution_forward_desc_init(
    zendnn_convolution_desc_t *conv_desc, zendnn_prop_kind_t prop_kind,
    zendnn_alg_kind_t alg_kind, const zendnn_memory_desc_t *src_desc,
    const zendnn_memory_desc_t *weights_desc,
    const zendnn_memory_desc_t *bias_desc, const zendnn_memory_desc_t *dst_desc,
    const zendnn_dims_t strides, const zendnn_dims_t padding_l,
    const zendnn_dims_t padding_r, bool reluFused, bool batchNormFused,
    const  zendnn_memory_desc_t *batchNormScale_desc,
    const  zendnn_memory_desc_t *batchNormMean_desc,
    const  zendnn_memory_desc_t *batchNormOffset_desc);

/// Initializes a descriptor for a convolution backward propagation primitive.
///
/// @note
///     Memory descriptors can be initialized with
///     #zendnn_format_tag_any or with format_kind set to #zendnn_format_kind_any.
///
/// Arrays @p strides, @p padding_l, and @p padding_r contain values for
/// spatial dimensions only and hence must have the same number of elements as
/// there are spatial dimensions. The order of values is the same as in the
/// tensor: depth (for 3D tensors), height (for 3D and 2D tensors), and width.
///
/// @param conv_desc Output descriptor for a convolution primitive.
/// @param alg_kind Convolution algorithm. Possible values are
///     #zendnn_convolution_direct, #zendnn_convolution_winograd,
///     #zendnn_convolution_auto.
/// @param diff_src_desc Diff source memory descriptor.
/// @param weights_desc Weights memory descriptor.
/// @param diff_dst_desc Diff destination memory descriptor.
/// @param strides Array of strides for spatial dimension.
/// @param padding_l Array of padding values for low indices for each spatial
///     dimension `([[front,] top,] left)`.
/// @param padding_r Array of padding values for high indices for each spatial
///     dimension `([[back,] bottom,] right)`. Can be NULL in which case
///     padding is assumed to be symmetrical.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_convolution_backward_data_desc_init(
        zendnn_convolution_desc_t *conv_desc, zendnn_alg_kind_t alg_kind,
        const zendnn_memory_desc_t *diff_src_desc,
        const zendnn_memory_desc_t *weights_desc,
        const zendnn_memory_desc_t *diff_dst_desc, const zendnn_dims_t strides,
        const zendnn_dims_t padding_l, const zendnn_dims_t padding_r);

/// Initializes a descriptor for a dilated convolution backward propagation
/// primitive.
///
/// @note
///     Memory descriptors can be initialized with
///     #zendnn_format_tag_any or with format_kind set to #zendnn_format_kind_any.
///
/// Arrays @p strides, @p dilates, @p padding_l, and @p padding_r contain
/// values for spatial dimensions only and hence must have the same number of
/// elements as there are spatial dimensions. The order of values is the same
/// as in the tensor: depth (for 3D tensors), height (for 3D and 2D tensors),
/// and width.
///
/// @param conv_desc Output descriptor for a convolution primitive.
/// @param alg_kind Convolution algorithm. Possible values are
///     #zendnn_convolution_direct, #zendnn_convolution_winograd,
///     #zendnn_convolution_auto.
/// @param diff_src_desc Diff source memory descriptor.
/// @param weights_desc Weights memory descriptor.
/// @param diff_dst_desc Diff destination memory descriptor.
/// @param strides Array of strides for spatial dimension.
/// @param dilates Array of dilations for spatial dimension. A zero value
///     means no dilation in the corresponding dimension.
/// @param padding_l Array of padding values for low indices for each spatial
///     dimension `([[front,] top,] left)`.
/// @param padding_r Array of padding values for high indices for each spatial
///     dimension `([[back,] bottom,] right)`. Can be NULL in which case
///     padding is considered to be symmetrical.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_dilated_convolution_backward_data_desc_init(
        zendnn_convolution_desc_t *conv_desc, zendnn_alg_kind_t alg_kind,
        const zendnn_memory_desc_t *diff_src_desc,
        const zendnn_memory_desc_t *weights_desc,
        const zendnn_memory_desc_t *diff_dst_desc, const zendnn_dims_t strides,
        const zendnn_dims_t dilates, const zendnn_dims_t padding_l,
        const zendnn_dims_t padding_r);

/// Initializes a descriptor for a convolution weights gradient primitive.
///
/// @note
///     Memory descriptors can be initialized with
///     #zendnn_format_tag_any or with format_kind set to #zendnn_format_kind_any.
///
/// Arrays @p strides, @p padding_l, and @p padding_r contain values for
/// spatial dimensions only and hence must have the same number of elements as
/// there are spatial dimensions. The order of values is the same as in the
/// tensor: depth (for 3D tensors), height (for 3D and 2D tensors), and width.
///
/// @param conv_desc Output descriptor for a convolution primitive.
/// @param alg_kind Convolution algorithm. Possible values are
///     #zendnn_convolution_direct, #zendnn_convolution_winograd,
///     #zendnn_convolution_auto.
/// @param src_desc Source memory descriptor.
/// @param diff_weights_desc Diff weights memory descriptor.
/// @param diff_bias_desc Diff bias memory descriptor. Passing NULL, a zero
///     memory descriptor, or a memory descriptor with format_kind set to
///     #zendnn_format_kind_undef disables the bias term.
/// @param diff_dst_desc Diff destination memory descriptor.
/// @param strides Array of strides for spatial dimension.
/// @param padding_l Array of padding values for low indices for each spatial
///     dimension `([[front,] top,] left)`.
/// @param padding_r Array of padding values for high indices for each spatial
///     dimension `([[back,] bottom,] right)`. Can be NULL in which case
///     padding is considered to be symmetrical.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_convolution_backward_weights_desc_init(
        zendnn_convolution_desc_t *conv_desc, zendnn_alg_kind_t alg_kind,
        const zendnn_memory_desc_t *src_desc,
        const zendnn_memory_desc_t *diff_weights_desc,
        const zendnn_memory_desc_t *diff_bias_desc,
        const zendnn_memory_desc_t *diff_dst_desc, const zendnn_dims_t strides,
        const zendnn_dims_t padding_l, const zendnn_dims_t padding_r);

/// Initializes a descriptor for a dilated convolution weights gradient
/// primitive.
///
/// @note
///     Memory descriptors can be initialized with
///     #zendnn_format_tag_any or with format_kind set to #zendnn_format_kind_any.
///
/// Arrays @p strides, @p dilates, @p padding_l, and @p padding_r contain
/// values for spatial dimensions only and hence must have the same number of
/// elements as there are spatial dimensions. The order of values is the same
/// as in the tensor: depth (for 3D tensors), height (for 3D and 2D tensors),
/// and width.
///
/// @param conv_desc Output descriptor for a convolution primitive.
/// @param alg_kind Convolution algorithm. Possible values are
///     #zendnn_convolution_direct, #zendnn_convolution_winograd,
///     #zendnn_convolution_auto.
/// @param src_desc Source memory descriptor.
/// @param diff_weights_desc Diff weights memory descriptor.
/// @param diff_bias_desc Diff bias memory descriptor. Passing NULL, a zero
///     memory descriptor, or a memory descriptor with format_kind set to
///     #zendnn_format_kind_undef disables the bias term.
/// @param diff_dst_desc Diff destination memory descriptor.
/// @param strides Array of strides for spatial dimension.
/// @param dilates Array of dilations for spatial dimension. A zero value
///     means no dilation in the corresponding dimension.
/// @param padding_l Array of padding values for low indices for each spatial
///     dimension `([[front,] top,] left)`.
/// @param padding_r Array of padding values for high indices for each spatial
///     dimension `([[back,] bottom,] right)`. Can be NULL in which case
///     padding is considered to be symmetrical.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_dilated_convolution_backward_weights_desc_init(
        zendnn_convolution_desc_t *conv_desc, zendnn_alg_kind_t alg_kind,
        const zendnn_memory_desc_t *src_desc,
        const zendnn_memory_desc_t *diff_weights_desc,
        const zendnn_memory_desc_t *diff_bias_desc,
        const zendnn_memory_desc_t *diff_dst_desc, const zendnn_dims_t strides,
        const zendnn_dims_t dilates, const zendnn_dims_t padding_l,
        const zendnn_dims_t padding_r);

/// @} zendnn_api_convolution

/// @addtogroup zendnn_api_deconvolution
/// @{

/// Initializes a descriptor for a deconvolution forward propagation primitive.
///
/// @note
///     Memory descriptors can be initialized with
///     #zendnn_format_tag_any or with format_kind set to #zendnn_format_kind_any.
///
/// Arrays @p strides, @p padding_l, and @p padding_r contain values for
/// spatial dimensions only and hence must have the same number of elements as
/// there are spatial dimensions. The order of values is the same as in the
/// tensor: depth (for 3D tensors), height (for 3D and 2D tensors), and width.
///
/// @param deconv_desc Output descriptor for a deconvolution primitive.
/// @param prop_kind Propagation kind. Possible values are
///     #zendnn_forward_training and #zendnn_forward_inference.
/// @param alg_kind Deconvolution algorithm. Possible values are
///     #zendnn_deconvolution_direct, #zendnn_deconvolution_winograd.
/// @param src_desc Source memory descriptor.
/// @param weights_desc Weights memory descriptor.
/// @param bias_desc Bias memory descriptor. Passing NULL, a zero memory
///     descriptor, or a memory descriptor with format_kind set to
///     #zendnn_format_kind_undef disables the bias term.
/// @param dst_desc Destination memory descriptor.
/// @param strides Array of strides for spatial dimension.
/// @param padding_l Array of padding values for low indices for each spatial
///     dimension `([[front,] top,] left)`.
/// @param padding_r Array of padding values for high indices for each spatial
///     dimension `([[back,] bottom,] right)`. Can be NULL in which case
///     padding is considered to be symmetrical.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_deconvolution_forward_desc_init(
        zendnn_deconvolution_desc_t *deconv_desc, zendnn_prop_kind_t prop_kind,
        zendnn_alg_kind_t alg_kind, const zendnn_memory_desc_t *src_desc,
        const zendnn_memory_desc_t *weights_desc,
        const zendnn_memory_desc_t *bias_desc, const zendnn_memory_desc_t *dst_desc,
        const zendnn_dims_t strides, const zendnn_dims_t padding_l,
        const zendnn_dims_t padding_r);

/// Initializes a descriptor for a dilated deconvolution forward propagation
/// primitive.
///
/// @note
///     Memory descriptors can be initialized with
///     #zendnn_format_tag_any or with format_kind set to #zendnn_format_kind_any.
///
/// Arrays @p strides, @p dilates, @p padding_l, and @p padding_r contain
/// values for spatial dimensions only and hence must have the same number of
/// elements as there are spatial dimensions. The order of values is the same
/// as in the tensor: depth (for 3D tensors), height (for 3D and 2D tensors),
/// and width.
///
/// @param deconv_desc Output descriptor for a deconvolution primitive.
/// @param prop_kind Propagation kind. Possible values are
///     #zendnn_forward_training and #zendnn_forward_inference.
/// @param alg_kind Deconvolution algorithm. Possible values are
///     #zendnn_deconvolution_direct, #zendnn_deconvolution_winograd.
/// @param src_desc Source memory descriptor.
/// @param weights_desc Weights memory descriptor.
/// @param bias_desc Bias memory descriptor. Passing NULL, a zero memory
///     descriptor, or a memory descriptor with format_kind set to
///     #zendnn_format_kind_undef disables the bias term.
/// @param dst_desc Destination memory descriptor.
/// @param strides Array of strides for spatial dimension.
/// @param dilates Array of dilations for spatial dimension. A zero value
///     means no dilation in the corresponding dimension.
/// @param padding_l Array of padding values for low indices for each spatial
///     dimension `([[front,] top,] left)`.
/// @param padding_r Array of padding values for high indices for each spatial
///     dimension `([[back,] bottom,] right)`. Can be NULL in which case
///     padding is considered to be symmetrical.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_dilated_deconvolution_forward_desc_init(
        zendnn_deconvolution_desc_t *deconv_desc, zendnn_prop_kind_t prop_kind,
        zendnn_alg_kind_t alg_kind, const zendnn_memory_desc_t *src_desc,
        const zendnn_memory_desc_t *weights_desc,
        const zendnn_memory_desc_t *bias_desc, const zendnn_memory_desc_t *dst_desc,
        const zendnn_dims_t strides, const zendnn_dims_t dilates,
        const zendnn_dims_t padding_l, const zendnn_dims_t padding_r);

/// Initializes a descriptor for a deconvolution backward propagation primitive.
///
/// @note
///     Memory descriptors can be initialized with
///     #zendnn_format_tag_any or with format_kind set to #zendnn_format_kind_any.
///
/// Arrays @p strides, @p padding_l, and @p padding_r contain values for
/// spatial dimensions only and hence must have the same number of elements as
/// there are spatial dimensions. The order of values is the same as in the
/// tensor: depth (for 3D tensors), height (for 3D and 2D tensors), and width.
///
/// @param deconv_desc Output descriptor for a deconvolution primitive.
/// @param alg_kind Deconvolution algorithm. Possible values are
///     #zendnn_deconvolution_direct, #zendnn_deconvolution_winograd.
/// @param diff_src_desc Diff source memory descriptor.
/// @param weights_desc Weights memory descriptor.
/// @param diff_dst_desc Diff destination memory descriptor.
/// @param strides Array of strides for spatial dimension.
/// @param padding_l Array of padding values for low indices for each spatial
///     dimension `([[front,] top,] left)`.
/// @param padding_r Array of padding values for high indices for each spatial
///     dimension `([[back,] bottom,] right)`. Can be NULL in which case
///     padding is considered to be symmetrical.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_deconvolution_backward_data_desc_init(
        zendnn_deconvolution_desc_t *deconv_desc, zendnn_alg_kind_t alg_kind,
        const zendnn_memory_desc_t *diff_src_desc,
        const zendnn_memory_desc_t *weights_desc,
        const zendnn_memory_desc_t *diff_dst_desc, const zendnn_dims_t strides,
        const zendnn_dims_t padding_l, const zendnn_dims_t padding_r);

/// Initializes a descriptor for a dilated deconvolution backward propagation
/// primitive.
///
/// @note
///     Memory descriptors can be initialized with
///     #zendnn_format_tag_any or with format_kind set to #zendnn_format_kind_any.
///
/// Arrays @p strides, @p dilates, @p padding_l, and @p padding_r contain
/// values for spatial dimensions only and hence must have the same number of
/// elements as there are spatial dimensions. The order of values is the same
/// as in the tensor: depth (for 3D tensors), height (for 3D and 2D tensors),
/// and width.
///
/// @param deconv_desc Output descriptor for a deconvolution primitive.
/// @param alg_kind Deconvolution algorithm. Possible values are
///     #zendnn_deconvolution_direct, #zendnn_deconvolution_winograd.
/// @param diff_src_desc Diff source memory descriptor.
/// @param weights_desc Weights memory descriptor.
/// @param diff_dst_desc Diff destination memory descriptor.
/// @param strides Array of strides for spatial dimension.
/// @param dilates Array of dilations for spatial dimension. A zero value
///     means no dilation in the corresponding dimension.
/// @param padding_l Array of padding values for low indices for each spatial
///     dimension `([[front,] top,] left)`.
/// @param padding_r Array of padding values for high indices for each spatial
///     dimension `([[back,] bottom,] right)`. Can be NULL in which case
///     padding is considered to be symmetrical.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_dilated_deconvolution_backward_data_desc_init(
        zendnn_deconvolution_desc_t *deconv_desc, zendnn_alg_kind_t alg_kind,
        const zendnn_memory_desc_t *diff_src_desc,
        const zendnn_memory_desc_t *weights_desc,
        const zendnn_memory_desc_t *diff_dst_desc, const zendnn_dims_t strides,
        const zendnn_dims_t dilates, const zendnn_dims_t padding_l,
        const zendnn_dims_t padding_r);

/// Initializes a descriptor for a deconvolution weights gradient primitive.
///
/// @note
///     Memory descriptors can be initialized with
///     #zendnn_format_tag_any or with format_kind set to #zendnn_format_kind_any.
///
/// Arrays @p strides, @p padding_l, and @p padding_r contain values for
/// spatial dimensions only and hence must have the same number of elements as
/// there are spatial dimensions. The order of values is the same as in the
/// tensor: depth (for 3D tensors), height (for 3D and 2D tensors), and width.
///
/// @param deconv_desc Output descriptor for a deconvolution primitive.
/// @param alg_kind Deconvolution algorithm. Possible values are
///     #zendnn_deconvolution_direct, #zendnn_deconvolution_winograd.
/// @param src_desc Source memory descriptor.
/// @param diff_weights_desc Diff weights memory descriptor.
/// @param diff_bias_desc Diff bias memory descriptor. Passing NULL, a zero
///     memory descriptor, or a memory descriptor with format_kind set to
///     #zendnn_format_kind_undef disables the bias term.
/// @param diff_dst_desc Diff destination memory descriptor.
/// @param strides Array of strides for spatial dimension.
/// @param padding_l Array of padding values for low indices for each spatial
///     dimension `([[front,] top,] left)`.
/// @param padding_r Array of padding values for high indices for each spatial
///     dimension `([[back,] bottom,] right)`. Can be NULL in which case
///     padding is considered to be symmetrical.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_deconvolution_backward_weights_desc_init(
        zendnn_deconvolution_desc_t *deconv_desc, zendnn_alg_kind_t alg_kind,
        const zendnn_memory_desc_t *src_desc,
        const zendnn_memory_desc_t *diff_weights_desc,
        const zendnn_memory_desc_t *diff_bias_desc,
        const zendnn_memory_desc_t *diff_dst_desc, const zendnn_dims_t strides,
        const zendnn_dims_t padding_l, const zendnn_dims_t padding_r);

/// Initializes a descriptor for a dilated deconvolution weights gradient
/// primitive.
///
/// @note
///     Memory descriptors can be initialized with
///     #zendnn_format_tag_any or with format_kind set to #zendnn_format_kind_any.
///
/// Arrays @p strides, @p dilates, @p padding_l, and @p padding_r contain
/// values for spatial dimensions only and hence must have the same number of
/// elements as there are spatial dimensions. The order of values is the same
/// as in the tensor: depth (for 3D tensors), height (for 3D and 2D tensors),
/// and width.
///
/// @param deconv_desc Output descriptor for a deconvolution primitive.
/// @param alg_kind Deconvolution algorithm. Possible values are
///     #zendnn_deconvolution_direct, #zendnn_deconvolution_winograd.
/// @param src_desc Source memory descriptor.
/// @param diff_weights_desc Diff weights memory descriptor.
/// @param diff_bias_desc Diff bias memory descriptor. Passing NULL, a zero
///     memory descriptor, or a memory descriptor with format_kind set to
///     #zendnn_format_kind_undef disables the bias term.
/// @param diff_dst_desc Diff destination memory descriptor.
/// @param strides Array of strides for spatial dimension.
/// @param dilates Array of dilations for spatial dimension. A zero value
///     means no dilation in the corresponding dimension.
/// @param padding_l Array of padding values for low indices for each spatial
///     dimension `([[front,] top,] left)`.
/// @param padding_r Array of padding values for high indices for each spatial
///     dimension `([[back,] bottom,] right)`. Can be NULL in which case
///     padding is considered to be symmetrical.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_dilated_deconvolution_backward_weights_desc_init(
        zendnn_deconvolution_desc_t *deconv_desc, zendnn_alg_kind_t alg_kind,
        const zendnn_memory_desc_t *src_desc,
        const zendnn_memory_desc_t *diff_weights_desc,
        const zendnn_memory_desc_t *diff_bias_desc,
        const zendnn_memory_desc_t *diff_dst_desc, const zendnn_dims_t strides,
        const zendnn_dims_t dilates, const zendnn_dims_t padding_l,
        const zendnn_dims_t padding_r);

/// @} zendnn_api_deconvolution

/// @addtogroup zendnn_api_shuffle
/// @{

/// Initializes a descriptor for shuffle forward propagation primitive.
///
/// @param shuffle_desc Output descriptor for a shuffle primitive.
/// @param prop_kind Propagation kind. Possible values are
///     #zendnn_forward_training and #zendnn_forward_inference.
/// @param data_desc Source and destination memory descriptor.
/// @param axis The axis along which the data is shuffled.
/// @param group_size Shuffle group size.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_shuffle_forward_desc_init(
        zendnn_shuffle_desc_t *shuffle_desc, zendnn_prop_kind_t prop_kind,
        const zendnn_memory_desc_t *data_desc, int axis, zendnn_dim_t group_size);

/// Initializes a descriptor for shuffle backward propagation primitive.
///
/// @param shuffle_desc Output descriptor for a shuffle primitive.
/// @param diff_data_desc Diff source and diff destination memory descriptor.
/// @param axis The axis along which the data is shuffled.
/// @param group_size Shuffle group size.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_shuffle_backward_desc_init(
        zendnn_shuffle_desc_t *shuffle_desc,
        const zendnn_memory_desc_t *diff_data_desc, int axis,
        zendnn_dim_t group_size);

/// @} zendnn_api_shuffle

/// @addtogroup zendnn_api_eltwise
/// @{

/// Initializes a descriptor for eltwise forward propagation primitive.
///
/// @param eltwise_desc Output descriptor for an eltwise primitive.
/// @param prop_kind Propagation kind. Possible values are
///     #zendnn_forward_training and #zendnn_forward_inference.
/// @param alg_kind Elementwise algorithm kind.
/// @param data_desc Source and destination memory descriptor.
/// @param alpha The alpha parameter for the elementwise operation. Specific
///     meaning depends on the algorithm.
/// @param beta The beta parameter for the elementwise operation. Specific
///     meaning depends on the algorithm.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_eltwise_forward_desc_init(
        zendnn_eltwise_desc_t *eltwise_desc, zendnn_prop_kind_t prop_kind,
        zendnn_alg_kind_t alg_kind, const zendnn_memory_desc_t *data_desc,
        float alpha, float beta);

/// Initializes a descriptor for eltwise backward propagation primitive.
///
/// @param eltwise_desc Output descriptor for an eltwise primitive.
/// @param alg_kind Elementwise algorithm kind.
/// @param diff_data_desc Diff source and diff destination memory descriptors.
/// @param data_desc Source and destination memory descriptor.
/// @param alpha The alpha parameter for the elementwise operation. Specific
///     meaning depends on the algorithm.
/// @param beta The beta parameter for the elementwise operation. Specific
///     meaning depends on the algorithm.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_eltwise_backward_desc_init(
        zendnn_eltwise_desc_t *eltwise_desc, zendnn_alg_kind_t alg_kind,
        const zendnn_memory_desc_t *diff_data_desc,
        const zendnn_memory_desc_t *data_desc, float alpha, float beta);

/// @} zendnn_api_eltwise

/// @addtogroup zendnn_api_softmax
/// @{

/// Initializes a descriptor for softmax forward propagation primitive.
///
/// @param softmax_desc Output descriptor for a softmax primitive.
/// @param prop_kind Propagation kind. Possible values are
///     #zendnn_forward_training and #zendnn_forward_inference.
/// @param data_desc Source and destination memory descriptor.
/// @param softmax_axis Axis over which softmax is computed.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_softmax_forward_desc_init(
        zendnn_softmax_desc_t *softmax_desc, zendnn_prop_kind_t prop_kind,
        const zendnn_memory_desc_t *data_desc, int softmax_axis);

/// Initializes a descriptor for softmax backward propagation primitive.
///
/// @param softmax_desc Output descriptor for a softmax primitive.
/// @param diff_data_desc Diff source and diff destination memory descriptors.
/// @param data_desc Destination memory descriptor.
/// @param softmax_axis Axis over which softmax is computed.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_softmax_backward_desc_init(
        zendnn_softmax_desc_t *softmax_desc,
        const zendnn_memory_desc_t *diff_data_desc,
        const zendnn_memory_desc_t *data_desc, int softmax_axis);

/// @} zendnn_api_softmax

/// @addtogroup zendnn_api_logsoftmax
/// @{

/// Initializes a descriptor for logsoftmax forward propagation primitive.
///
/// @param logsoftmax_desc Output descriptor for a logsoftmax primitive.
/// @param prop_kind Propagation kind. Possible values are
///     #zendnn_forward_training and #zendnn_forward_inference.
/// @param data_desc Source and destination memory descriptor.
/// @param logsoftmax_axis Axis over which logsoftmax is computed.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_logsoftmax_forward_desc_init(
        zendnn_logsoftmax_desc_t *logsoftmax_desc, zendnn_prop_kind_t prop_kind,
        const zendnn_memory_desc_t *data_desc, int logsoftmax_axis);

/// Initializes a descriptor for logsoftmax backward propagation primitive.
///
/// @param logsoftmax_desc Output descriptor for a logsoftmax primitive.
/// @param diff_data_desc Diff source and diff destination memory descriptors.
/// @param data_desc Destination memory descriptor.
/// @param logsoftmax_axis Axis over which softmax is computed.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_logsoftmax_backward_desc_init(
        zendnn_logsoftmax_desc_t *logsoftmax_desc,
        const zendnn_memory_desc_t *diff_data_desc,
        const zendnn_memory_desc_t *data_desc, int logsoftmax_axis);

/// @} zendnn_api_logsoftmax

/// @addtogroup zendnn_api_pooling
/// @{

/// Initializes a descriptor for pooling forward propagation primitive.
///
/// Arrays @p strides, @p kernel, @p padding_l, and @p padding_r contain values
/// for spatial dimensions only and hence must have the same number of elements
/// as there are spatial dimensions. The order of values is the same as in the
/// tensor: depth (for 3D tensors), height (for 3D and 2D tensors), and width.
///
/// @param pool_desc Output descriptor for a pooling primitive.
/// @param prop_kind Propagation kind. Possible values are
///     #zendnn_forward_training and #zendnn_forward_inference.
/// @param alg_kind Pooling algorithm kind: either #zendnn_pooling_max,
///     #zendnn_pooling_avg_include_padding, or #zendnn_pooling_avg (same as
///     #zendnn_pooling_avg_exclude_padding).
/// @param src_desc Source memory descriptor.
/// @param dst_desc Destination memory descriptor.
/// @param strides Array of strides for spatial dimension.
/// @param kernel Array of kernel spatial dimensions.
/// @param padding_l Array of padding values for low indices for each spatial
///     dimension `([[front,] top,] left)`.
/// @param padding_r Array of padding values for high indices for each spatial
///     dimension `([[back,] bottom,] right)`. Can be NULL in which case
///     padding is considered to be symmetrical.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_pooling_forward_desc_init(
        zendnn_pooling_desc_t *pool_desc, zendnn_prop_kind_t prop_kind,
        zendnn_alg_kind_t alg_kind, const zendnn_memory_desc_t *src_desc,
        const zendnn_memory_desc_t *dst_desc, const zendnn_dims_t strides,
        const zendnn_dims_t kernel, const zendnn_dims_t padding_l,
        const zendnn_dims_t padding_r);

/// Initializes a descriptor for pooling backward propagation primitive.
///
/// Arrays @p strides, @p kernel, @p padding_l, and @p padding_r contain values
/// for spatial dimensions only and hence must have the same number of elements
/// as there are spatial dimensions. The order of values is the same as in the
/// tensor: depth (for 3D tensors), height (for 3D and 2D tensors), and width.
///
/// @param pool_desc Output descriptor for a pooling primitive.
/// @param alg_kind Pooling algorithm kind: either #zendnn_pooling_max,
///     #zendnn_pooling_avg_include_padding, or #zendnn_pooling_avg (same as
///     #zendnn_pooling_avg_exclude_padding).
/// @param diff_src_desc Diff source memory descriptor.
/// @param diff_dst_desc Diff destination memory descriptor.
/// @param strides Array of strides for spatial dimension.
/// @param kernel Array of kernel spatial dimensions.
/// @param padding_l Array of padding values for low indices for each spatial
///     dimension `([[front,] top,] left)`.
/// @param padding_r Array of padding values for high indices for each spatial
///     dimension `([[back,] bottom,] right)`. Can be NULL in which case
///     padding is considered to be symmetrical.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_pooling_backward_desc_init(
        zendnn_pooling_desc_t *pool_desc, zendnn_alg_kind_t alg_kind,
        const zendnn_memory_desc_t *diff_src_desc,
        const zendnn_memory_desc_t *diff_dst_desc, const zendnn_dims_t strides,
        const zendnn_dims_t kernel, const zendnn_dims_t padding_l,
        const zendnn_dims_t padding_r);

/// @} zendnn_api_pooling

/// @addtogroup zendnn_api_pooling_v2
/// @{

/// Initializes a descriptor for pooling v2 (pooling with dilation support)
/// forward propagation primitive.
///
/// Arrays @p strides, @p kernel, @p dilation, @p padding_l and @p padding_r
/// contain values for spatial dimensions only and hence must have the same
/// number of elements as there are spatial dimensions. The order of values
/// is the same as in the tensor: depth (for 3D tensors),
/// height (for 3D and 2D tensors), and width.
///
/// @param pool_desc Output descriptor for a pooling primitive.
/// @param prop_kind Propagation kind. Possible values are
///     #zendnn_forward_training and #zendnn_forward_inference.
/// @param alg_kind Pooling algorithm kind: either #zendnn_pooling_max,
///     #zendnn_pooling_avg_include_padding, or #zendnn_pooling_avg (same as
///     #zendnn_pooling_avg_exclude_padding).
/// @param src_desc Source memory descriptor.
/// @param dst_desc Destination memory descriptor.
/// @param strides Array of strides for spatial dimension.
/// @param kernel Array of kernel spatial dimensions.
/// @param dilation Array of dilations for spatial dimension.
/// @param padding_l Array of padding values for low indices for each spatial
///     dimension `([[front,] top,] left)`.
/// @param padding_r Array of padding values for high indices for each spatial
///     dimension `([[back,] bottom,] right)`. Can be NULL in which case
///     padding is considered to be symmetrical.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_pooling_v2_forward_desc_init(
        zendnn_pooling_v2_desc_t *pool_desc, zendnn_prop_kind_t prop_kind,
        zendnn_alg_kind_t alg_kind, const zendnn_memory_desc_t *src_desc,
        const zendnn_memory_desc_t *dst_desc, const zendnn_dims_t strides,
        const zendnn_dims_t kernel, const zendnn_dims_t dilation,
        const zendnn_dims_t padding_l, const zendnn_dims_t padding_r);

/// Initializes a descriptor for pooling v2 (pooling with dilation support)
/// backward propagation primitive.
///
/// Arrays @p strides, @p kernel, @p dilation, @p padding_l and @p padding_r
/// contain values for spatial dimensions only and hence must have the same
/// number of elements as there are spatial dimensions. The order of values
/// is the same as in the tensor: depth (for 3D tensors),
/// height (for 3D and 2D tensors), and width.
///
/// @param pool_desc Output descriptor for a pooling primitive.
/// @param alg_kind Pooling algorithm kind: either #zendnn_pooling_max,
///     #zendnn_pooling_avg_include_padding, or #zendnn_pooling_avg (same as
///     #zendnn_pooling_avg_exclude_padding).
/// @param diff_src_desc Diff source memory descriptor.
/// @param diff_dst_desc Diff destination memory descriptor.
/// @param strides Array of strides for spatial dimension.
/// @param kernel Array of kernel spatial dimensions.
/// @param dilation Array of dilations for spatial dimension.
/// @param padding_l Array of padding values for low indices for each spatial
///     dimension `([[front,] top,] left)`.
/// @param padding_r Array of padding values for high indices for each spatial
///     dimension `([[back,] bottom,] right)`. Can be NULL in which case
///     padding is considered to be symmetrical.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_pooling_v2_backward_desc_init(
        zendnn_pooling_v2_desc_t *pool_desc, zendnn_alg_kind_t alg_kind,
        const zendnn_memory_desc_t *diff_src_desc,
        const zendnn_memory_desc_t *diff_dst_desc, const zendnn_dims_t strides,
        const zendnn_dims_t kernel, const zendnn_dims_t dilation,
        const zendnn_dims_t padding_l, const zendnn_dims_t padding_r);

/// @} zendnn_api_pooling_v2

/// @addtogroup zendnn_api_prelu
/// @{

/// Initializes a descriptor for PReLU
/// (leaky ReLU with trainable alpha parameter)
/// forward propagation primitive.
///
/// @note
///     weights descriptor is allowed to be initialized with
///     #zendnn_format_tag_any or with format_kind set to #zendnn_format_kind_any.
///
/// @param prelu_desc Output descriptor for a prelu primitive.
/// @param prop_kind Propagation kind. Possible values are
///     #zendnn_forward_training and #zendnn_forward_inference.
/// @param data_desc Source and destination memory descriptor.
/// @param weights_desc Alpha parameters memory descriptor.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_prelu_forward_desc_init(
        zendnn_prelu_desc_t *prelu_desc, zendnn_prop_kind_t prop_kind,
        const zendnn_memory_desc_t *data_desc,
        const zendnn_memory_desc_t *weights_desc);

/// Initializes a descriptor for PReLU
/// (leaky ReLU with trainable alpha parameter)
/// backward propagation primitive.
///
/// @note
///     weights descriptor and diff_weights descriptor are allowed
///     to be initialized with #zendnn_format_tag_any or with format_kind
///     set to #zendnn_format_kind_any.
///
/// @param prelu_desc Output descriptor for a prelu primitive.
/// @param data_desc Source and destination memory descriptor.
/// @param weights_desc Alpha parameters memory descriptor.
/// @param diff_data_desc Diff source and destination memory descriptor.
/// @param diff_weights_desc Diff alpha parameters memory descriptor.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_prelu_backward_desc_init(
        zendnn_prelu_desc_t *prelu_desc, const zendnn_memory_desc_t *data_desc,
        const zendnn_memory_desc_t *weights_desc,
        const zendnn_memory_desc_t *diff_data_desc,
        const zendnn_memory_desc_t *diff_weights_desc);

/// @} zendnn_api_prelu

/// @addtogroup zendnn_api_lrn
/// @{

/// Initializes a descriptor for LRN forward propagation primitive.
///
/// @param lrn_desc Output descriptor for a LRN primitive.
/// @param prop_kind Propagation kind. Possible values are
///     #zendnn_forward_training and #zendnn_forward_inference.
/// @param alg_kind LRN algorithm kind: either #zendnn_lrn_across_channels or
///     #zendnn_lrn_within_channel.
/// @param data_desc Source and destination memory descriptor.
/// @param local_size Regularization local size.
/// @param alpha The alpha regularization parameter.
/// @param beta The beta regularization parameter.
/// @param k The k regularization parameter.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_lrn_forward_desc_init(zendnn_lrn_desc_t *lrn_desc,
        zendnn_prop_kind_t prop_kind, zendnn_alg_kind_t alg_kind,
        const zendnn_memory_desc_t *data_desc, zendnn_dim_t local_size, float alpha,
        float beta, float k);

/// Initializes a descriptor for LRN backward propagation primitive.
///
/// @param lrn_desc Output descriptor for a LRN primitive.
/// @param alg_kind LRN algorithm kind: either #zendnn_lrn_across_channels or
///     #zendnn_lrn_within_channel.
/// @param diff_data_desc Diff source and diff destination memory descriptor.
/// @param data_desc Source memory descriptor.
/// @param local_size Regularization local size.
/// @param alpha The alpha regularization parameter.
/// @param beta The beta regularization parameter.
/// @param k The k regularization parameter.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_lrn_backward_desc_init(zendnn_lrn_desc_t *lrn_desc,
        zendnn_alg_kind_t alg_kind, const zendnn_memory_desc_t *diff_data_desc,
        const zendnn_memory_desc_t *data_desc, zendnn_dim_t local_size, float alpha,
        float beta, float k);

/// @} zendnn_api_lrn

/// @addtogroup zendnn_api_batch_normalization
/// @{

/// Initializes a descriptor for a batch normalization forward propagation
/// primitive.
///
/// @note
///     In-place operation is supported: the dst can refer to the same memory
///     as the src.
///
/// @param bnrm_desc Output descriptor for batch normalization primitive.
/// @param prop_kind Propagation kind. Possible values are
///     #zendnn_forward_training and #zendnn_forward_inference.
/// @param data_desc Source and destination memory descriptor.
/// @param epsilon Batch normalization epsilon parameter.
/// @param flags Batch normalization flags (@ref zendnn_normalization_flags_t).
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_batch_normalization_forward_desc_init(
        zendnn_batch_normalization_desc_t *bnrm_desc, zendnn_prop_kind_t prop_kind,
        const zendnn_memory_desc_t *data_desc, float epsilon, unsigned flags);

/// Initializes a descriptor for a batch normalization backward propagation
/// primitive.
///
/// @note
///     In-place operation is supported: the diff_dst can refer to the same
///     memory as the diff_src.
///
/// @param bnrm_desc Output descriptor for batch normalization primitive.
/// @param prop_kind Propagation kind. Possible values are
///     #zendnn_backward_data and #zendnn_backward (diffs for all parameters are
///     computed in this case).
/// @param diff_data_desc Diff source and diff destination memory descriptor.
/// @param data_desc Source memory descriptor.
/// @param epsilon Batch normalization epsilon parameter.
/// @param flags Batch normalization flags (@ref zendnn_normalization_flags_t).
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_batch_normalization_backward_desc_init(
        zendnn_batch_normalization_desc_t *bnrm_desc, zendnn_prop_kind_t prop_kind,
        const zendnn_memory_desc_t *diff_data_desc,
        const zendnn_memory_desc_t *data_desc, float epsilon, unsigned flags);

/// @} zendnn_api_batch_normalization

/// @addtogroup zendnn_api_layer_normalization
/// @{

/// Initializes a descriptor for layer normalization forward propagation
/// primitive.
///
/// @note
///     In-place operation is supported: the dst can refer to the same memory
///     as the src.
///
/// @param lnrm_desc Output descriptor for layer normalization primitive.
/// @param prop_kind Propagation kind. Possible values are
///     #zendnn_forward_training and #zendnn_forward_inference.
/// @param data_desc Source and destination memory descriptor.
/// @param stat_desc Memory descriptor for mean and variance. If this
///     parameter is NULL, a zero memory descriptor, or a memory descriptor
///     with format_kind set to #zendnn_format_kind_undef, then the memory
///     descriptor for stats is derived from @p data_desc by removing the last
///     dimension.
/// @param epsilon Layer normalization epsilon parameter.
/// @param flags Layer normalization flags (@ref zendnn_normalization_flags_t).
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_layer_normalization_forward_desc_init(
        zendnn_layer_normalization_desc_t *lnrm_desc, zendnn_prop_kind_t prop_kind,
        const zendnn_memory_desc_t *data_desc,
        const zendnn_memory_desc_t *stat_desc, float epsilon, unsigned flags);

/// Initializes a descriptor for a layer normalization backward propagation
/// primitive.
///
/// @note
///     In-place operation is supported: the diff_dst can refer to the same
///     memory as the diff_src.
///
/// @param lnrm_desc Output descriptor for layer normalization primitive.
/// @param prop_kind Propagation kind. Possible values are
///     #zendnn_backward_data and #zendnn_backward (diffs for all parameters are
///     computed in this case).
/// @param diff_data_desc Diff source and diff destination memory descriptor.
/// @param data_desc Source memory descriptor.
/// @param stat_desc Memory descriptor for mean and variance. If this
///     parameter is NULL, a zero memory descriptor, or a memory descriptor
///     with format_kind set to #zendnn_format_kind_undef, then the memory
///     descriptor for stats is derived from @p data_desc by removing the last
///     dimension.
/// @param epsilon Layer normalization epsilon parameter.
/// @param flags Layer normalization flags (@ref zendnn_normalization_flags_t).
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_layer_normalization_backward_desc_init(
        zendnn_layer_normalization_desc_t *lnrm_desc, zendnn_prop_kind_t prop_kind,
        const zendnn_memory_desc_t *diff_data_desc,
        const zendnn_memory_desc_t *data_desc,
        const zendnn_memory_desc_t *stat_desc, float epsilon, unsigned flags);

/// @} zendnn_api_layer_normalization

/// @addtogroup zendnn_api_inner_product
/// @{

/// Initializes descriptor for inner product forward propagation.
///
/// @note
///     Memory descriptors can be initialized with
///     #zendnn_format_tag_any or with format_kind set to #zendnn_format_kind_any.
///
/// @param ip_desc Output descriptor for inner product primitive.
/// @param prop_kind Propagation kind. Possible values are
///     #zendnn_forward_training and #zendnn_forward_inference.
/// @param src_desc Source memory descriptor.
/// @param weights_desc Weights memory descriptor.
/// @param bias_desc Bias memory descriptor. Passing NULL, a zero memory
///     descriptor, or a memory descriptor with format_kind set to
///     #zendnn_format_kind_undef disables the bias term.
/// @param dst_desc Destination memory descriptor.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_inner_product_forward_desc_init(
        zendnn_inner_product_desc_t *ip_desc, zendnn_prop_kind_t prop_kind,
        const zendnn_memory_desc_t *src_desc,
        const zendnn_memory_desc_t *weights_desc,
        const zendnn_memory_desc_t *bias_desc,
        const zendnn_memory_desc_t *dst_desc);

/// Initializes descriptor for inner product backward propagation.
///
/// @note
///     Memory descriptors can be initialized with
///     #zendnn_format_tag_any or with format_kind set to #zendnn_format_kind_any.
///
/// @param ip_desc Output descriptor for inner product primitive.
/// @param diff_src_desc Diff source memory descriptor.
/// @param weights_desc Weights memory descriptor.
/// @param diff_dst_desc Diff destination memory descriptor.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_inner_product_backward_data_desc_init(
        zendnn_inner_product_desc_t *ip_desc,
        const zendnn_memory_desc_t *diff_src_desc,
        const zendnn_memory_desc_t *weights_desc,
        const zendnn_memory_desc_t *diff_dst_desc);

/// Initializes descriptor for inner product weights gradient primitive.
///
/// @note
///     Memory descriptors can be initialized with
///     #zendnn_format_tag_any or with format_kind set to #zendnn_format_kind_any.
///
/// @param ip_desc Output descriptor for inner product primitive.
/// @param src_desc Source memory descriptor.
/// @param diff_weights_desc Diff weights memory descriptor.
/// @param diff_bias_desc Diff bias memory descriptor. Passing NULL, a zero
///     memory descriptor, or a memory descriptor with format_kind set to
///     #zendnn_format_kind_undef disables the bias term.
/// @param diff_dst_desc Diff destination memory descriptor.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_inner_product_backward_weights_desc_init(
        zendnn_inner_product_desc_t *ip_desc, const zendnn_memory_desc_t *src_desc,
        const zendnn_memory_desc_t *diff_weights_desc,
        const zendnn_memory_desc_t *diff_bias_desc,
        const zendnn_memory_desc_t *diff_dst_desc);

/// @} zendnn_api_inner_product

/// @addtogroup zendnn_api_attributes
/// @{

/// Set quantization scale and shift parameters for RNN data tensors.
///
/// For performance reasons, the low-precision configuration of the RNN
/// primitives expects input activations to have the unsigned 8-bit integer
/// data type. The scale and shift parameters are used to quantize
/// floating-point data to unsigned integer and must be passed to the RNN
/// primitive using attributes.
///
/// The quantization formula is `scale * data + shift`.
///
/// @note
///     Quantization scale and shift are common for src_layer, src_iter,
///     dst_iter, and dst_layer.
///
/// Example usage:
/// @code
///     // RNN parameters
///     int l = 2, t = 2, mb = 32, sic = 32, slc = 32, dic = 32, dlc = 32;
///     // Activations quantization parameters
///     float scale = 63.f, shift = 64.f;
///
///     zendnn_primitive_attr_t rnn_attr;
///     // Create default attributes
///     zendnn_primitive_attr_create(&rnn_attr);
///
///     // Set scale and shift for int8 quantization of activation
///     zendnn_primitive_attr_set_rnn_data_qparams(rnn_attr, scale, shift);
///
///     // Create and configure rnn op_desc
///     zendnn_rnn_desc_t rnn_d;
///     zendnn_primitive_desc_t rnn_pd;
///     zendnn_primitive_desc_create(&rnn_pd, &rnn_d, attr, engine, NULL);
/// @endcode
///
/// @param attr Primitive attributes.
/// @param scale The value to scale the data by.
/// @param shift The value to shift the data by.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_primitive_attr_set_rnn_data_qparams(
        zendnn_primitive_attr_t attr, const float scale, const float shift);

/// Returns the quantization scale and shift parameters for RNN data tensors.
///
/// @note
///     Quantization scale and shift are common for src_layer, src_iter,
///     dst_iter, and dst_layer.
///
/// @param attr Primitive attributes.
/// @param scale The value to scale the data by.
/// @param shift The value to shift the data by.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_primitive_attr_get_rnn_data_qparams(
        const_zendnn_primitive_attr_t attr, float *scale, float *shift);

/// Sets quantization scaling factors for RNN weights tensors. The
/// low-precision configuration of the RNN primitives expects input weights to
/// use the signed 8-bit integer data type. The scaling factors are used to
/// quantize floating-point data to signed integer and must be passed to RNN
/// primitives using attributes.
///
/// @note
///     The dimension order is always native and does not depend on the actual
///     layout used. For example, five-dimensional weights always have (l, d,
///     i, g, o) logical dimension ordering.
///
/// @note
///     Quantization scales are common for weights_layer and weights_iteration
///
/// @param attr Primitive attributes.
/// @param count Number of elements in the @p scales array.
/// @param mask Scaling factors correspondence mask that defines the
///     correspondence between the output tensor dimensions and the @p
///     scales vector. The set i-th bit indicates that a dedicated scaling
///     factor should be used for each index along that dimension. Set the
///     mask to 0 to use a common scaling factor for the whole output
///     tensor.
/// @param scales Array of output scaling factors that must contain @p count
///     values and the following equality must hold:
///     \f[count = \prod\limits_{d \in mask} weights.dims[d].\f]
///     Violations can only be detected when the attributes are used to create
///     a primitive descriptor.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_primitive_attr_set_rnn_weights_qparams(
        zendnn_primitive_attr_t attr, zendnn_dim_t count, int mask,
        const float *scales);

/// Returns the quantization scaling factors for RNN weights tensors.
///
/// @param attr Primitive attributes.
/// @param count Number of elements in the @p scales array.
/// @param mask Scaling factors correspondence mask that defines the
///     correspondence between the output tensor dimensions and the @p
///     scales vector. The set i-th bit indicates that a dedicated scaling
///     factor should be used for each index along that dimension. Set the
///     mask to 0 to use a common scaling factor for the whole output
///     tensor.
/// @param scales Array of output scaling factors that contain @p count
///     values and the following equality must hold:
///     \f[count = \prod\limits_{d \in mask} weights.dims[d].\f]
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_primitive_attr_get_rnn_weights_qparams(
        const_zendnn_primitive_attr_t attr, zendnn_dim_t *count, int *mask,
        const float **scales);

/// Sets quantization scaling factors for RNN projection weights tensors. The
/// low-precision configuration of the RNN primitives expects input weights to
/// use the signed 8-bit integer data type. The scaling factors are used to
/// quantize floating-point data to signed integer and must be passed to RNN
/// primitives using attributes.
///
/// @note
///     The dimension order is always native and does not depend on the actual
///     layout used. For example, five-dimensional weights always have (l, d,
///     i, g, o) logical dimension ordering.
///
/// @param attr Primitive attributes.
/// @param count Number of elements in the @p scales array.
/// @param mask Scaling factors correspondence mask that defines the
///     correspondence between the output tensor dimensions and the @p
///     scales vector. The set i-th bit indicates that a dedicated scaling
///     factor should be used for each index along that dimension. Set the
///     mask to 0 to use a common scaling factor for the whole output
///     tensor.
/// @param scales Array of output scaling factors that must contain @p count
///     values and the following equality must hold:
///     \f[count = \prod\limits_{d \in mask} weights.dims[d].\f]
///     Violations can only be detected when the attributes are used to create
///     a primitive descriptor.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_primitive_attr_set_rnn_weights_projection_qparams(
        zendnn_primitive_attr_t attr, zendnn_dim_t count, int mask,
        const float *scales);

/// Returns the quantization scaling factors for RNN projection weights tensors.
///
/// @param attr Primitive attributes.
/// @param count Number of elements in the @p scales array.
/// @param mask Scaling factors correspondence mask that defines the
///     correspondence between the output tensor dimensions and the @p
///     scales vector. The set i-th bit indicates that a dedicated scaling
///     factor should be used for each index along that dimension. Set the
///     mask to 0 to use a common scaling factor for the whole output
///     tensor.
/// @param scales Array of output scaling factors that contain @p count
///     values and the following equality must hold:
///     \f[count = \prod\limits_{d \in mask} weights.dims[d].\f]
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_primitive_attr_get_rnn_weights_projection_qparams(
        const_zendnn_primitive_attr_t attr, zendnn_dim_t *count, int *mask,
        const float **scales);

/// @} zendnn_api_attributes

/// @addtogroup zendnn_api_rnn
/// @{

/// Initializes a descriptor for vanilla RNN forward propagation primitive.
///
/// The following arguments may either be @c NULL or point to a zero memory
/// descriptor:
/// - @p src_iter_desc,
/// - @p bias_desc,
/// - @p dst_iter_desc.
///
/// This would then indicate that the RNN forward propagation primitive should
/// not use them and should default to zero values instead.
///
/// @note
///     All memory descriptors can be initialized with
///     #zendnn_format_tag_any or with format_kind set to #zendnn_format_kind_any.
///
/// @param rnn_desc Output descriptor for vanilla RNN primitive.
/// @param prop_kind Propagation kind. Possible values are
///     #zendnn_forward_training and #zendnn_forward_inference.
/// @param activation Activation kind. Possible values are #zendnn_eltwise_relu,
///     #zendnn_eltwise_tanh or #zendnn_eltwise_logistic.
/// @param direction RNN direction. See @ref zendnn_rnn_direction_t for more
///     info.
/// @param src_layer_desc Memory descriptor for the input vector.
/// @param src_iter_desc Memory descriptor for the input recurrent hidden
///     state vector.
/// @param weights_layer_desc Memory descriptor for the weights applied to the
///     layer input.
/// @param weights_iter_desc Memory descriptor for the weights applied to the
///     recurrent input.
/// @param bias_desc Bias memory descriptor.
/// @param dst_layer_desc Memory descriptor for the output vector.
/// @param dst_iter_desc Memory descriptor for the output recurrent hidden
///     state vector.
/// @param flags Unused.
/// @param alpha Negative slope if activation is #zendnn_eltwise_relu.
/// @param beta Unused.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_vanilla_rnn_forward_desc_init(
        zendnn_rnn_desc_t *rnn_desc, zendnn_prop_kind_t prop_kind,
        const zendnn_alg_kind_t activation, const zendnn_rnn_direction_t direction,
        const zendnn_memory_desc_t *src_layer_desc,
        const zendnn_memory_desc_t *src_iter_desc,
        const zendnn_memory_desc_t *weights_layer_desc,
        const zendnn_memory_desc_t *weights_iter_desc,
        const zendnn_memory_desc_t *bias_desc,
        const zendnn_memory_desc_t *dst_layer_desc,
        const zendnn_memory_desc_t *dst_iter_desc, unsigned flags, float alpha,
        float beta);

/// Initializes a descriptor for vanilla RNN backward propagation primitive.
///
/// The following arguments may either be @c NULL or point to a zero memory
/// descriptor:
/// - @p src_iter_desc together with @p diff_src_iter_desc,
/// - @p bias_desc together with @p diff_bias_desc,
/// - @p dst_iter_desc together with @p diff_dst_iter_desc.
///
/// This would then indicate that the RNN backward propagation primitive should
/// not use the respective data and should use zero values instead.
///
/// @note
///     All memory descriptors can be initialized with
///     #zendnn_format_tag_any or with format_kind set to #zendnn_format_kind_any.
///
/// @param rnn_desc Output descriptor for vanilla RNN primitive.
/// @param prop_kind Propagation kind. Must be #zendnn_backward.
/// @param activation Activation kind. Possible values are #zendnn_eltwise_relu,
///     #zendnn_eltwise_tanh or #zendnn_eltwise_logistic.
/// @param direction RNN direction. See @ref zendnn_rnn_direction_t for more
///     info.
/// @param src_layer_desc Memory descriptor for the input vector.
/// @param src_iter_desc Memory descriptor for the input recurrent hidden
///     state vector.
/// @param weights_layer_desc Memory descriptor for the weights applied to the
///     layer input.
/// @param weights_iter_desc Memory descriptor for the weights applied to the
///     recurrent input.
/// @param bias_desc Bias memory descriptor.
/// @param dst_layer_desc Memory descriptor for the output vector.
/// @param dst_iter_desc Memory descriptor for the output recurrent hidden
///     state vector.
/// @param diff_src_layer_desc Memory descriptor for the diff of input vector.
/// @param diff_src_iter_desc Memory descriptor for the diff of input recurrent
///     hidden state vector.
/// @param diff_weights_layer_desc Memory descriptor for the diff of weights
///     applied to the layer input.
/// @param diff_weights_iter_desc Memory descriptor for the diff of weights
///     applied to the recurrent input.
/// @param diff_bias_desc Diff bias memory descriptor.
/// @param diff_dst_layer_desc Memory descriptor for the diff of output
///     vector.
/// @param diff_dst_iter_desc Memory descriptor for the diff of output
///     recurrent hidden state vector.
/// @param flags Unused.
/// @param alpha Negative slope if activation is #zendnn_eltwise_relu.
/// @param beta Unused.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_vanilla_rnn_backward_desc_init(
        zendnn_rnn_desc_t *rnn_desc, zendnn_prop_kind_t prop_kind,
        const zendnn_alg_kind_t activation, const zendnn_rnn_direction_t direction,
        const zendnn_memory_desc_t *src_layer_desc,
        const zendnn_memory_desc_t *src_iter_desc,
        const zendnn_memory_desc_t *weights_layer_desc,
        const zendnn_memory_desc_t *weights_iter_desc,
        const zendnn_memory_desc_t *bias_desc,
        const zendnn_memory_desc_t *dst_layer_desc,
        const zendnn_memory_desc_t *dst_iter_desc,
        const zendnn_memory_desc_t *diff_src_layer_desc,
        const zendnn_memory_desc_t *diff_src_iter_desc,
        const zendnn_memory_desc_t *diff_weights_layer_desc,
        const zendnn_memory_desc_t *diff_weights_iter_desc,
        const zendnn_memory_desc_t *diff_bias_desc,
        const zendnn_memory_desc_t *diff_dst_layer_desc,
        const zendnn_memory_desc_t *diff_dst_iter_desc, unsigned flags,
        float alpha, float beta);

/// Initializes a descriptor for LSTM forward propagation primitive.
///
/// The following arguments may either be @c NULL or point to a zero memory
/// descriptor:
/// - @p src_iter_desc together with @p src_iter_c_desc,
/// - @p bias_desc,
/// - @p dst_iter_desc together with @p dst_iter_c_desc.
///
/// This would then indicate that the LSTM forward propagation primitive should
/// not use them and should default to zero values instead.
///
/// @note
///     All memory descriptors can be initialized with
///     #zendnn_format_tag_any or with format_kind set to #zendnn_format_kind_any.
///
/// @sa zendnn_lstm_forward_desc_init_v2 to initialize forward LSTM with and
///     without peephole
/// @sa zendnn_lstm_forward_desc_init_v3 to initialize forward LSTM with and
///     without peephole / recurrent projection layer
///
/// @param rnn_desc Output descriptor for LSTM primitive.
/// @param prop_kind Propagation kind. Possible values are
///     #zendnn_forward_training and #zendnn_forward_inference.
/// @param direction RNN direction. See @ref zendnn_rnn_direction_t for more
///     info.
/// @param src_layer_desc Memory descriptor for the input vector.
/// @param src_iter_desc Memory descriptor for the input recurrent hidden
///     state vector.
/// @param src_iter_c_desc Memory descriptor for the input recurrent cell
///     state vector.
/// @param weights_layer_desc Memory descriptor for the weights applied to the
///     layer input.
/// @param weights_iter_desc Memory descriptor for the weights applied to the
///     recurrent input.
/// @param bias_desc Bias memory descriptor.
/// @param dst_layer_desc Memory descriptor for the output vector.
/// @param dst_iter_desc Memory descriptor for the output recurrent hidden
///     state vector.
/// @param dst_iter_c_desc Memory descriptor for the output recurrent cell
///     state vector.
/// @param flags Unused.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_lstm_forward_desc_init(zendnn_rnn_desc_t *rnn_desc,
        zendnn_prop_kind_t prop_kind, zendnn_rnn_direction_t direction,
        const zendnn_memory_desc_t *src_layer_desc,
        const zendnn_memory_desc_t *src_iter_desc,
        const zendnn_memory_desc_t *src_iter_c_desc,
        const zendnn_memory_desc_t *weights_layer_desc,
        const zendnn_memory_desc_t *weights_iter_desc,
        const zendnn_memory_desc_t *bias_desc,
        const zendnn_memory_desc_t *dst_layer_desc,
        const zendnn_memory_desc_t *dst_iter_desc,
        const zendnn_memory_desc_t *dst_iter_c_desc, unsigned flags);

/// Initializes a descriptor for an LSTM (with or without peephole) forward
/// propagation primitive.
///
/// The following arguments may either be @c NULL or point to a zero memory
/// descriptor:
/// - @p src_iter_desc together with @p src_iter_c_desc,
/// - @p weights_peephole_desc,
/// - @p bias_desc,
/// - @p dst_iter_desc together with @p dst_iter_c_desc.
///
/// This would then indicate that the LSTM forward propagation primitive should
/// not use them and should default to zero values instead.
///
/// @note
///     All memory descriptors can be initialized with #zendnn_format_tag_any or
///     with format_kind set to #zendnn_format_kind_any.
///
/// @sa zendnn_lstm_forward_desc_init_v3 to initialize forward LSTM with and
///     without peephole / recurrent projection layer
///
/// @param rnn_desc Output descriptor for LSTM primitive.
/// @param prop_kind Propagation kind. Possible values are
///     #zendnn_forward_training and #zendnn_forward_inference.
/// @param direction RNN direction. See @ref zendnn_rnn_direction_t for more
///     info.
/// @param src_layer_desc Memory descriptor for the input vector.
/// @param src_iter_desc Memory descriptor for the input recurrent hidden
///     state vector.
/// @param src_iter_c_desc Memory descriptor for the input recurrent cell
///     state vector.
/// @param weights_layer_desc Memory descriptor for the weights applied to the
///     layer input.
/// @param weights_iter_desc Memory descriptor for the weights applied to the
///     recurrent input.
/// @param weights_peephole_desc Memory descriptor for the weights applied to
///     the cell states (according to the Peephole LSTM formula).
/// @param bias_desc Bias memory descriptor.
/// @param dst_layer_desc Memory descriptor for the output vector.
/// @param dst_iter_desc Memory descriptor for the output recurrent hidden
///     state vector.
/// @param dst_iter_c_desc Memory descriptor for the output recurrent cell
///     state vector.
/// @param flags Unused.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_lstm_forward_desc_init_v2(zendnn_rnn_desc_t *rnn_desc,
        zendnn_prop_kind_t prop_kind, zendnn_rnn_direction_t direction,
        const zendnn_memory_desc_t *src_layer_desc,
        const zendnn_memory_desc_t *src_iter_desc,
        const zendnn_memory_desc_t *src_iter_c_desc,
        const zendnn_memory_desc_t *weights_layer_desc,
        const zendnn_memory_desc_t *weights_iter_desc,
        const zendnn_memory_desc_t *weights_peephole_desc,
        const zendnn_memory_desc_t *bias_desc,
        const zendnn_memory_desc_t *dst_layer_desc,
        const zendnn_memory_desc_t *dst_iter_desc,
        const zendnn_memory_desc_t *dst_iter_c_desc, unsigned flags);

/// Initializes a descriptor for an LSTM (with or without peephole and with
/// or without recurrent projection layer) forward propagation primitive.
///
/// The following arguments may either be @c NULL or point to a zero memory
/// descriptor:
/// - @p src_iter_desc together with @p src_iter_c_desc,
/// - @p weights_peephole_desc,
/// - @p bias_desc,
/// - @p dst_iter_desc together with @p dst_iter_c_desc.
///
/// This would then indicate that the LSTM forward propagation primitive should
/// not use them and should default to zero values instead.
///
/// The @p weights_projection_desc could either be @c NULL or point to a zero
/// memory descriptor. This would then indicate that the LSTM doesn't have
/// recurrent projection layer.
///
/// @note
///     All memory descriptors can be initialized with #zendnn_format_tag_any or
///     with format_kind set to #zendnn_format_kind_any.
///
/// @param rnn_desc Output descriptor for LSTM primitive.
/// @param prop_kind Propagation kind. Possible values are
///     #zendnn_forward_training and #zendnn_forward_inference.
/// @param direction RNN direction. See @ref zendnn_rnn_direction_t for more
///     info.
/// @param src_layer_desc Memory descriptor for the input vector.
/// @param src_iter_desc Memory descriptor for the input recurrent hidden
///     state vector.
/// @param src_iter_c_desc Memory descriptor for the input recurrent cell
///     state vector.
/// @param weights_layer_desc Memory descriptor for the weights applied to the
///     layer input.
/// @param weights_iter_desc Memory descriptor for the weights applied to the
///     recurrent input.
/// @param weights_peephole_desc Memory descriptor for the weights applied to
///     the cell states (according to the Peephole LSTM formula).
/// @param weights_projection_desc Memory descriptor for the weights applied to
///     the hidden states to get the recurrent projection (according to the
///     Projection LSTM formula).
/// @param bias_desc Bias memory descriptor.
/// @param dst_layer_desc Memory descriptor for the output vector.
/// @param dst_iter_desc Memory descriptor for the output recurrent hidden
///     state vector.
/// @param dst_iter_c_desc Memory descriptor for the output recurrent cell
///     state vector.
/// @param flags Unused.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_lstm_forward_desc_init_v3(zendnn_rnn_desc_t *rnn_desc,
        zendnn_prop_kind_t prop_kind, zendnn_rnn_direction_t direction,
        const zendnn_memory_desc_t *src_layer_desc,
        const zendnn_memory_desc_t *src_iter_desc,
        const zendnn_memory_desc_t *src_iter_c_desc,
        const zendnn_memory_desc_t *weights_layer_desc,
        const zendnn_memory_desc_t *weights_iter_desc,
        const zendnn_memory_desc_t *weights_peephole_desc,
        const zendnn_memory_desc_t *weights_projection_desc,
        const zendnn_memory_desc_t *bias_desc,
        const zendnn_memory_desc_t *dst_layer_desc,
        const zendnn_memory_desc_t *dst_iter_desc,
        const zendnn_memory_desc_t *dst_iter_c_desc, unsigned flags);

/// Initializes a descriptor for an LSTM backward propagation primitive.
///
/// The following arguments may either be @c NULL or point to a zero memory
/// descriptor:
/// - @p src_iter_desc together with @p src_iter_c_desc, @p diff_src_iter_desc,
///   and @p diff_src_iter_c_desc,
/// - @p bias_desc together with @p diff_bias_desc,
/// - @p dst_iter_desc together with @p dst_iter_c_desc, @p diff_dst_iter_desc,
///   and @p diff_dst_iter_c_desc.
///
/// This would then indicate that the LSTM backward propagation primitive
/// should not use them and should default to zero values instead.
///
/// @note
///     All memory descriptors can be initialized with
///     #zendnn_format_tag_any or with format_kind set to #zendnn_format_kind_any.
///
/// @sa zendnn_lstm_backward_desc_init_v2 to initialize backward LSTM with and
///     without peephole
/// @sa zendnn_lstm_backward_desc_init_v3 to initialize backward LSTM with and
///     without peephole / recurrent projection layer
///
/// @param rnn_desc Output descriptor for LSTM primitive.
/// @param prop_kind Propagation kind. Must be #zendnn_backward.
/// @param direction RNN direction. See @ref zendnn_rnn_direction_t for more
///     info.
/// @param src_layer_desc Memory descriptor for the input vector.
/// @param src_iter_desc Memory descriptor for the input recurrent hidden
///     state vector.
/// @param src_iter_c_desc Memory descriptor for the input recurrent cell
///     state vector.
/// @param weights_layer_desc Memory descriptor for the weights applied to the
///     layer input.
/// @param weights_iter_desc Memory descriptor for the weights applied to the
///     recurrent input.
/// @param bias_desc Bias memory descriptor.
/// @param dst_layer_desc Memory descriptor for the output vector.
/// @param dst_iter_desc Memory descriptor for the output recurrent hidden
///     state vector.
/// @param dst_iter_c_desc Memory descriptor for the output recurrent cell
///     state vector.
/// @param diff_src_layer_desc Memory descriptor for the diff of input vector.
/// @param diff_src_iter_desc Memory descriptor for the diff of input recurrent
///     hidden state vector.
/// @param diff_src_iter_c_desc Memory descriptor for the diff of input
/// recurrent cell state vector.
/// @param diff_weights_layer_desc Memory descriptor for the diff of weights
///     applied to the layer input.
/// @param diff_weights_iter_desc Memory descriptor for the diff of weights
///     applied to the recurrent input.
/// @param diff_bias_desc Diff bias memory descriptor.
/// @param diff_dst_layer_desc Memory descriptor for the diff of output
///     vector.
/// @param diff_dst_iter_desc Memory descriptor for the diff of output
///     recurrent hidden state vector.
/// @param diff_dst_iter_c_desc Memory descriptor for the diff of output
///     recurrent cell state vector.
/// @param flags Unused.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_lstm_backward_desc_init(zendnn_rnn_desc_t *rnn_desc,
        zendnn_prop_kind_t prop_kind, zendnn_rnn_direction_t direction,
        const zendnn_memory_desc_t *src_layer_desc,
        const zendnn_memory_desc_t *src_iter_desc,
        const zendnn_memory_desc_t *src_iter_c_desc,
        const zendnn_memory_desc_t *weights_layer_desc,
        const zendnn_memory_desc_t *weights_iter_desc,
        const zendnn_memory_desc_t *bias_desc,
        const zendnn_memory_desc_t *dst_layer_desc,
        const zendnn_memory_desc_t *dst_iter_desc,
        const zendnn_memory_desc_t *dst_iter_c_desc,
        const zendnn_memory_desc_t *diff_src_layer_desc,
        const zendnn_memory_desc_t *diff_src_iter_desc,
        const zendnn_memory_desc_t *diff_src_iter_c_desc,
        const zendnn_memory_desc_t *diff_weights_layer_desc,
        const zendnn_memory_desc_t *diff_weights_iter_desc,
        const zendnn_memory_desc_t *diff_bias_desc,
        const zendnn_memory_desc_t *diff_dst_layer_desc,
        const zendnn_memory_desc_t *diff_dst_iter_desc,
        const zendnn_memory_desc_t *diff_dst_iter_c_desc, unsigned flags);

/// Initializes a descriptor for an LSTM (with or without peephole) backward
/// propagation primitive.
///
/// The following arguments may either be @c NULL or point to a zero memory
/// descriptor:
/// - @p src_iter_desc together with @p src_iter_c_desc, @p diff_src_iter_desc,
///   and @p diff_src_iter_c_desc,
/// - @p weights_peephole_desc together with @p diff_weights_peephole_desc,
/// - @p bias_desc together with @p diff_bias_desc,
/// - @p dst_iter_desc together with @p dst_iter_c_desc, @p diff_dst_iter_desc,
///   and @p diff_dst_iter_c_desc.
///
/// This would then indicate that the LSTM backward propagation primitive
/// should not use them and should default to zero values instead.
///
/// @note
///     All memory descriptors can be initialized with #zendnn_format_tag_any or
///     with format_kind set to #zendnn_format_kind_any.
///
/// @sa zendnn_lstm_backward_desc_init_v3 to initialize backward LSTM with and
///     without peephole / recurrent projection layer
///
/// @param rnn_desc Output descriptor for LSTM primitive.
/// @param prop_kind Propagation kind. Must be #zendnn_backward.
/// @param direction RNN direction. See @ref zendnn_rnn_direction_t for more
///     info.
/// @param src_layer_desc Memory descriptor for the input vector.
/// @param src_iter_desc Memory descriptor for the input recurrent hidden
///     state vector.
/// @param src_iter_c_desc Memory descriptor for the input recurrent cell
///     state vector.
/// @param weights_layer_desc Memory descriptor for the weights applied to the
///     layer input.
/// @param weights_iter_desc Memory descriptor for the weights applied to the
///     recurrent input.
/// @param weights_peephole_desc Memory descriptor for the weights applied to
///     the cell states (according to the Peephole LSTM formula).
/// @param bias_desc Bias memory descriptor.
/// @param dst_layer_desc Memory descriptor for the output vector.
/// @param dst_iter_desc Memory descriptor for the output recurrent hidden
///     state vector.
/// @param dst_iter_c_desc Memory descriptor for the output recurrent cell
///     state vector.
/// @param diff_src_layer_desc Memory descriptor for the diff of input vector.
/// @param diff_src_iter_desc Memory descriptor for the diff of input recurrent
///     hidden state vector.
/// @param diff_src_iter_c_desc Memory descriptor for the diff of input
/// recurrent cell state vector.
/// @param diff_weights_layer_desc Memory descriptor for the diff of weights
///     applied to the layer input.
/// @param diff_weights_iter_desc Memory descriptor for the diff of weights
///     applied to the recurrent input.
/// @param diff_weights_peephole_desc Memory descriptor for the diff of weights
///     applied to the cell states (according to the Peephole LSTM formula).
/// @param diff_bias_desc Diff bias memory descriptor.
/// @param diff_dst_layer_desc Memory descriptor for the diff of output
///     vector.
/// @param diff_dst_iter_desc Memory descriptor for the diff of output
///     recurrent hidden state vector.
/// @param diff_dst_iter_c_desc Memory descriptor for the diff of output
///     recurrent cell state vector.
/// @param flags Unused.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_lstm_backward_desc_init_v2(
        zendnn_rnn_desc_t *rnn_desc, zendnn_prop_kind_t prop_kind,
        zendnn_rnn_direction_t direction,
        const zendnn_memory_desc_t *src_layer_desc,
        const zendnn_memory_desc_t *src_iter_desc,
        const zendnn_memory_desc_t *src_iter_c_desc,
        const zendnn_memory_desc_t *weights_layer_desc,
        const zendnn_memory_desc_t *weights_iter_desc,
        const zendnn_memory_desc_t *weights_peephole_desc,
        const zendnn_memory_desc_t *bias_desc,
        const zendnn_memory_desc_t *dst_layer_desc,
        const zendnn_memory_desc_t *dst_iter_desc,
        const zendnn_memory_desc_t *dst_iter_c_desc,
        const zendnn_memory_desc_t *diff_src_layer_desc,
        const zendnn_memory_desc_t *diff_src_iter_desc,
        const zendnn_memory_desc_t *diff_src_iter_c_desc,
        const zendnn_memory_desc_t *diff_weights_layer_desc,
        const zendnn_memory_desc_t *diff_weights_iter_desc,
        const zendnn_memory_desc_t *diff_weights_peephole_desc,
        const zendnn_memory_desc_t *diff_bias_desc,
        const zendnn_memory_desc_t *diff_dst_layer_desc,
        const zendnn_memory_desc_t *diff_dst_iter_desc,
        const zendnn_memory_desc_t *diff_dst_iter_c_desc, unsigned flags);

/// Initializes a descriptor for an LSTM (with or without peephole and with or
/// with out recurrent projection layer) backward propagation primitive.
///
/// The following arguments may either be @c NULL or point to a zero memory
/// descriptor:
/// - @p src_iter_desc together with @p src_iter_c_desc, @p diff_src_iter_desc,
///   and @p diff_src_iter_c_desc,
/// - @p weights_peephole_desc together with @p diff_weights_peephole_desc,
/// - @p bias_desc together with @p diff_bias_desc,
/// - @p dst_iter_desc together with @p dst_iter_c_desc, @p diff_dst_iter_desc,
///   and @p diff_dst_iter_c_desc.
///
/// This would then indicate that the LSTM backward propagation primitive
/// should not use them and should default to zero values instead.
///
/// The @p weights_projection_desc together with @p
/// diff_weights_projection_desc could either be @c NULL or point to a zero
/// memory descriptor. This would then indicate that the LSTM doesn't have
/// recurrent projection layer.
///
/// @note
///     All memory descriptors can be initialized with #zendnn_format_tag_any or
///     with format_kind set to #zendnn_format_kind_any.
///
/// @param rnn_desc Output descriptor for LSTM primitive.
/// @param prop_kind Propagation kind. Must be #zendnn_backward.
/// @param direction RNN direction. See @ref zendnn_rnn_direction_t for more
///     info.
/// @param src_layer_desc Memory descriptor for the input vector.
/// @param src_iter_desc Memory descriptor for the input recurrent hidden
///     state vector.
/// @param src_iter_c_desc Memory descriptor for the input recurrent cell
///     state vector.
/// @param weights_layer_desc Memory descriptor for the weights applied to the
///     layer input.
/// @param weights_iter_desc Memory descriptor for the weights applied to the
///     recurrent input.
/// @param weights_peephole_desc Memory descriptor for the weights applied to
///     the cell states (according to the Peephole LSTM formula).
/// @param weights_projection_desc Memory descriptor for the weights applied to
///     the hidden states to get the recurrent projection (according to the
///     Projection LSTM formula).
/// @param bias_desc Bias memory descriptor.
/// @param dst_layer_desc Memory descriptor for the output vector.
/// @param dst_iter_desc Memory descriptor for the output recurrent hidden
///     state vector.
/// @param dst_iter_c_desc Memory descriptor for the output recurrent cell
///     state vector.
/// @param diff_src_layer_desc Memory descriptor for the diff of input vector.
/// @param diff_src_iter_desc Memory descriptor for the diff of input recurrent
///     hidden state vector.
/// @param diff_src_iter_c_desc Memory descriptor for the diff of input
/// recurrent cell state vector.
/// @param diff_weights_layer_desc Memory descriptor for the diff of weights
///     applied to the layer input.
/// @param diff_weights_iter_desc Memory descriptor for the diff of weights
///     applied to the recurrent input.
/// @param diff_weights_peephole_desc Memory descriptor for the diff of weights
///     applied to the cell states (according to the Peephole LSTM formula).
/// @param diff_weights_projection_desc Memory descriptor for the diff of
///     weights applied to the hidden states to get the recurrent projection
///     (according to the Projection LSTM formula).
/// @param diff_bias_desc Diff bias memory descriptor.
/// @param diff_dst_layer_desc Memory descriptor for the diff of output
///     vector.
/// @param diff_dst_iter_desc Memory descriptor for the diff of output
///     recurrent hidden state vector.
/// @param diff_dst_iter_c_desc Memory descriptor for the diff of output
///     recurrent cell state vector.
/// @param flags Unused.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_lstm_backward_desc_init_v3(
        zendnn_rnn_desc_t *rnn_desc, zendnn_prop_kind_t prop_kind,
        zendnn_rnn_direction_t direction,
        const zendnn_memory_desc_t *src_layer_desc,
        const zendnn_memory_desc_t *src_iter_desc,
        const zendnn_memory_desc_t *src_iter_c_desc,
        const zendnn_memory_desc_t *weights_layer_desc,
        const zendnn_memory_desc_t *weights_iter_desc,
        const zendnn_memory_desc_t *weights_peephole_desc,
        const zendnn_memory_desc_t *weights_projection_desc,
        const zendnn_memory_desc_t *bias_desc,
        const zendnn_memory_desc_t *dst_layer_desc,
        const zendnn_memory_desc_t *dst_iter_desc,
        const zendnn_memory_desc_t *dst_iter_c_desc,
        const zendnn_memory_desc_t *diff_src_layer_desc,
        const zendnn_memory_desc_t *diff_src_iter_desc,
        const zendnn_memory_desc_t *diff_src_iter_c_desc,
        const zendnn_memory_desc_t *diff_weights_layer_desc,
        const zendnn_memory_desc_t *diff_weights_iter_desc,
        const zendnn_memory_desc_t *diff_weights_peephole_desc,
        const zendnn_memory_desc_t *diff_weights_projection_desc,
        const zendnn_memory_desc_t *diff_bias_desc,
        const zendnn_memory_desc_t *diff_dst_layer_desc,
        const zendnn_memory_desc_t *diff_dst_iter_desc,
        const zendnn_memory_desc_t *diff_dst_iter_c_desc, unsigned flags);

/// Initializes a descriptor for GRU forward propagation primitive.
///
/// The following arguments may either be @c NULL or point to a zero memory
/// descriptor:
/// - @p src_iter_desc,
/// - @p bias_desc,
/// - @p dst_iter_desc.
///
/// This would then indicate that the GRU forward propagation primitive should
/// not use them and should default to zero values instead.
///
/// @note
///     All memory descriptors can be initialized with
///     #zendnn_format_tag_any or with format_kind set to #zendnn_format_kind_any.
///
/// @param rnn_desc Output descriptor for GRU primitive.
/// @param prop_kind Propagation kind. Possible values are
///     #zendnn_forward_training and #zendnn_forward_inference.
/// @param direction RNN direction. See @ref zendnn_rnn_direction_t for more
///     info.
/// @param src_layer_desc Memory descriptor for the input vector.
/// @param src_iter_desc Memory descriptor for the input recurrent hidden
///     state vector.
/// @param weights_layer_desc Memory descriptor for the weights applied to the
///     layer input.
/// @param weights_iter_desc Memory descriptor for the weights applied to the
///     recurrent input.
/// @param bias_desc Bias memory descriptor.
/// @param dst_layer_desc Memory descriptor for the output vector.
/// @param dst_iter_desc Memory descriptor for the output recurrent hidden
///     state vector.
/// @param flags Unused.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_gru_forward_desc_init(zendnn_rnn_desc_t *rnn_desc,
        zendnn_prop_kind_t prop_kind, zendnn_rnn_direction_t direction,
        const zendnn_memory_desc_t *src_layer_desc,
        const zendnn_memory_desc_t *src_iter_desc,
        const zendnn_memory_desc_t *weights_layer_desc,
        const zendnn_memory_desc_t *weights_iter_desc,
        const zendnn_memory_desc_t *bias_desc,
        const zendnn_memory_desc_t *dst_layer_desc,
        const zendnn_memory_desc_t *dst_iter_desc, unsigned flags);

/// Initializes a descriptor for GRU backward propagation primitive.
///
/// The following arguments may either be @c NULL or point to a zero memory
/// descriptor:
/// - @p src_iter_desc together with @p diff_src_iter_desc,
/// - @p bias_desc together with @p diff_bias_desc,
/// - @p dst_iter_desc together with @p diff_dst_iter_desc.
///
/// This would then indicate that the GRU backward propagation primitive
/// should not use them and should default to zero values instead.
///
/// @note
///     All memory descriptors can be initialized with
///     #zendnn_format_tag_any or with format_kind set to #zendnn_format_kind_any.
///
/// @param rnn_desc Output descriptor for GRU primitive.
/// @param prop_kind Propagation kind. Must be #zendnn_backward.
/// @param direction RNN direction. See @ref zendnn_rnn_direction_t for more
///     info.
/// @param src_layer_desc Memory descriptor for the input vector.
/// @param src_iter_desc Memory descriptor for the input recurrent hidden
///     state vector.
/// @param weights_layer_desc Memory descriptor for the weights applied to the
///     layer input.
/// @param weights_iter_desc Memory descriptor for the weights applied to the
///     recurrent input.
/// @param bias_desc Bias memory descriptor.
/// @param dst_layer_desc Memory descriptor for the output vector.
/// @param dst_iter_desc Memory descriptor for the output recurrent hidden
///     state vector.
/// @param diff_src_layer_desc Memory descriptor for the diff of input vector.
/// @param diff_src_iter_desc Memory descriptor for the diff of input recurrent
///     hidden state vector.
/// @param diff_weights_layer_desc Memory descriptor for the diff of weights
///     applied to the layer input.
/// @param diff_weights_iter_desc Memory descriptor for the diff of weights
///     applied to the recurrent input.
/// @param diff_bias_desc Diff bias memory descriptor.
/// @param diff_dst_layer_desc Memory descriptor for the diff of output
///     vector.
/// @param diff_dst_iter_desc Memory descriptor for the diff of output
///     recurrent hidden state vector.
/// @param flags Unused.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_gru_backward_desc_init(zendnn_rnn_desc_t *rnn_desc,
        zendnn_prop_kind_t prop_kind, zendnn_rnn_direction_t direction,
        const zendnn_memory_desc_t *src_layer_desc,
        const zendnn_memory_desc_t *src_iter_desc,
        const zendnn_memory_desc_t *weights_layer_desc,
        const zendnn_memory_desc_t *weights_iter_desc,
        const zendnn_memory_desc_t *bias_desc,
        const zendnn_memory_desc_t *dst_layer_desc,
        const zendnn_memory_desc_t *dst_iter_desc,
        const zendnn_memory_desc_t *diff_src_layer_desc,
        const zendnn_memory_desc_t *diff_src_iter_desc,
        const zendnn_memory_desc_t *diff_weights_layer_desc,
        const zendnn_memory_desc_t *diff_weights_iter_desc,
        const zendnn_memory_desc_t *diff_bias_desc,
        const zendnn_memory_desc_t *diff_dst_layer_desc,
        const zendnn_memory_desc_t *diff_dst_iter_desc, unsigned flags);

/// Initializes a descriptor for LBR GRU forward propagation primitive.
///
/// The following arguments may either be @c NULL or point to a zero memory
/// descriptor:
/// - @p src_iter_desc,
/// - @p bias_desc,
/// - @p dst_iter_desc.
///
/// This would then indicate that the LBR GRU forward propagation primitive
/// should not use them and should default to zero values instead.
///
/// @param rnn_desc Output descriptor for LBR GRU primitive.
/// @param prop_kind Propagation kind. Possible values are
///     #zendnn_forward_training and #zendnn_forward_inference.
/// @param direction RNN direction. See @ref zendnn_rnn_direction_t for more
///     info.
/// @param src_layer_desc Memory descriptor for the input vector.
/// @param src_iter_desc Memory descriptor for the input recurrent hidden
///     state vector.
/// @param weights_layer_desc Memory descriptor for the weights applied to the
///     layer input.
/// @param weights_iter_desc Memory descriptor for the weights applied to the
///     recurrent input.
/// @param bias_desc Bias memory descriptor.
/// @param dst_layer_desc Memory descriptor for the output vector.
/// @param dst_iter_desc Memory descriptor for the output recurrent hidden
///     state vector.
/// @param flags Unused.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_lbr_gru_forward_desc_init(zendnn_rnn_desc_t *rnn_desc,
        zendnn_prop_kind_t prop_kind, zendnn_rnn_direction_t direction,
        const zendnn_memory_desc_t *src_layer_desc,
        const zendnn_memory_desc_t *src_iter_desc,
        const zendnn_memory_desc_t *weights_layer_desc,
        const zendnn_memory_desc_t *weights_iter_desc,
        const zendnn_memory_desc_t *bias_desc,
        const zendnn_memory_desc_t *dst_layer_desc,
        const zendnn_memory_desc_t *dst_iter_desc, unsigned flags);

/// Initializes a descriptor for LBR GRU backward propagation primitive.
///
/// The following arguments may either be @c NULL or point to a zero memory
/// descriptor:
/// - @p src_iter_desc together with @p diff_src_iter_desc,
/// - @p bias_desc together with @p diff_bias_desc,
/// - @p dst_iter_desc together with @p diff_dst_iter_desc.
///
/// This would then indicate that the LBR GRU backward propagation primitive
/// should not use them and should default to zero values instead.
///
/// @note
///     All memory descriptors can be initialized with
///     #zendnn_format_tag_any or with format_kind set to #zendnn_format_kind_any.
///
/// @param rnn_desc Output descriptor for LBR GRU primitive.
/// @param prop_kind Propagation kind. Must be #zendnn_backward.
/// @param direction RNN direction. See @ref zendnn_rnn_direction_t for more
///     info.
/// @param src_layer_desc Memory descriptor for the input vector.
/// @param src_iter_desc Memory descriptor for the input recurrent hidden
///     state vector.
/// @param weights_layer_desc Memory descriptor for the weights applied to the
///     layer input.
/// @param weights_iter_desc Memory descriptor for the weights applied to the
///     recurrent input.
/// @param bias_desc Bias memory descriptor.
/// @param dst_layer_desc Memory descriptor for the output vector.
/// @param dst_iter_desc Memory descriptor for the output recurrent hidden
///     state vector.
/// @param diff_src_layer_desc Memory descriptor for the diff of input vector.
/// @param diff_src_iter_desc Memory descriptor for the diff of input recurrent
///     hidden state vector.
/// @param diff_weights_layer_desc Memory descriptor for the diff of weights
///     applied to the layer input.
/// @param diff_weights_iter_desc Memory descriptor for the diff of weights
///     applied to the recurrent input.
/// @param diff_bias_desc Diff bias memory descriptor.
/// @param diff_dst_layer_desc Memory descriptor for the diff of output
///     vector.
/// @param diff_dst_iter_desc Memory descriptor for the diff of output
///     recurrent hidden state vector.
/// @param flags Unused.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_lbr_gru_backward_desc_init(
        zendnn_rnn_desc_t *rnn_desc, zendnn_prop_kind_t prop_kind,
        zendnn_rnn_direction_t direction,
        const zendnn_memory_desc_t *src_layer_desc,
        const zendnn_memory_desc_t *src_iter_desc,
        const zendnn_memory_desc_t *weights_layer_desc,
        const zendnn_memory_desc_t *weights_iter_desc,
        const zendnn_memory_desc_t *bias_desc,
        const zendnn_memory_desc_t *dst_layer_desc,
        const zendnn_memory_desc_t *dst_iter_desc,
        const zendnn_memory_desc_t *diff_src_layer_desc,
        const zendnn_memory_desc_t *diff_src_iter_desc,
        const zendnn_memory_desc_t *diff_weights_layer_desc,
        const zendnn_memory_desc_t *diff_weights_iter_desc,
        const zendnn_memory_desc_t *diff_bias_desc,
        const zendnn_memory_desc_t *diff_dst_layer_desc,
        const zendnn_memory_desc_t *diff_dst_iter_desc, unsigned flags);

/// @} zendnn_api_rnn

/// @addtogroup zendnn_api_matmul
/// @{

/// Initializes a matrix multiplication descriptor.
///
/// @param matmul_desc Output descriptor for matmul primitive.
/// @param src_desc Source memory descriptor (matrix A)
/// @param weights_desc Weights memory descriptor (matrix B)
/// @param bias_desc Bias memory descriptor. Passing NULL, a zero memory
///     descriptor, or a memory descriptor with format_kind set to
///     #zendnn_format_kind_undef disables the bias term.
/// @param dst_desc Destination memory descriptor (matrix C).
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_matmul_desc_init(zendnn_matmul_desc_t *matmul_desc,
        const zendnn_memory_desc_t *src_desc,
        const zendnn_memory_desc_t *weights_desc,
        const zendnn_memory_desc_t *bias_desc,
        const zendnn_memory_desc_t *dst_desc);

/// @} zendnn_api_matmul

/// @addtogroup zendnn_api_resampling Resampling
/// @{

/// Initializes a descriptor for a resampling forward propagation primitive.
///
/// @note
///     Destination memory descriptor is allowed to be initialized with
///     #zendnn_format_tag_any or with format_kind set to #zendnn_format_kind_any.
///
///
/// @param resampling_desc Output descriptor for a resampling primitive.
/// @param prop_kind Propagation kind. Possible values are
///     #zendnn_forward_training and #zendnn_forward_inference.
/// @param alg_kind resampling algorithm kind: either #zendnn_resampling_nearest,
///     or #zendnn_resampling_linear.
/// @param factors Array of scaling factors for spatial dimension.
/// @param src_desc Source memory descriptor.
/// @param dst_desc Destination memory descriptor.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_resampling_forward_desc_init(
        zendnn_resampling_desc_t *resampling_desc, zendnn_prop_kind_t prop_kind,
        zendnn_alg_kind_t alg_kind, const float *factors,
        const zendnn_memory_desc_t *src_desc, const zendnn_memory_desc_t *dst_desc);

/// Initializes a descriptor for resampling backward propagation primitive.
///
/// @param resampling_desc Output descriptor for a resampling primitive.
/// @param alg_kind resamplinging algorithm kind: either
///     #zendnn_resampling_nearest, or #zendnn_resampling_linear.
/// @param diff_src_desc Diff source memory descriptor.
/// @param diff_dst_desc Diff destination memory descriptor.
/// @param factors Array of scaling factors for spatial dimension.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
///
zendnn_status_t ZENDNN_API zendnn_resampling_backward_desc_init(
        zendnn_resampling_desc_t *resampling_desc, zendnn_alg_kind_t alg_kind,
        const float *factors, const zendnn_memory_desc_t *diff_src_desc,
        const zendnn_memory_desc_t *diff_dst_desc);

/// @} zendnn_api_resampling

/// @addtogroup zendnn_api_reduction Reduction
/// @{

/// Initializes a descriptor for a reduction primitive.
///
/// @note
///     Destination memory descriptor is allowed to be initialized with
///     #zendnn_format_tag_any or with format_kind set to #zendnn_format_kind_any.
///
///
/// @param desc Output descriptor for a reduction primitive.
/// @param alg_kind reduction algorithm kind. Possible values:
///     #zendnn_reduction_max, #zendnn_reduction_min, #zendnn_reduction_sum,
///     #zendnn_reduction_mul, #zendnn_reduction_mean, #zendnn_reduction_norm_lp_max,
///     #zendnn_reduction_norm_lp_sum, #zendnn_reduction_norm_lp_power_p_max,
///     #zendnn_reduction_norm_lp_power_p_sum.
/// @param p Algorithm specific parameter.
/// @param eps Algorithm specific parameter.
/// @param src_desc Source memory descriptor.
/// @param dst_desc Destination memory descriptor.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
///
zendnn_status_t ZENDNN_API zendnn_reduction_desc_init(zendnn_reduction_desc_t *desc,
        zendnn_alg_kind_t alg_kind, const zendnn_memory_desc_t *src_desc,
        const zendnn_memory_desc_t *dst_desc, float p, float eps);


/// @} zendnn_api_reduction

/* add new primitive */

/// @addtogroup zendnn_api_embedding_bag EmbedddingBag
/// @{

/// Initializes a descriptor for an embedding_bag primitive.
///
/// @note
///     Destination memory descriptor is allowed to be initialized with
///     #zendnn_format_tag_any or with format_kind set to
///     #zendnn_format_kind_any. The embedding_bag primitive does not
///     allocate memory to destination and it should be pre-allocated.
///
/// @param desc Output descriptor for an embeding_bag primitive
/// @param prop_kind Propagation kind. currently only forward_inference is
///     supported.
/// @param alg_kind embedding_bag algorithm kind. Possible values:
///     #zendnn_embedding_bag_max, #zendnn_embedding_bag_sum,
///     #zendnn_embedding_bag_mean,
/// @param input_desc Input (embedding table) memory descriptor.
/// @param indices_desc Indices memory descriptor.
/// @param offsets_desc Offsets memory descriptor.
/// @param weights_desc Weights memory descriptor. This can be nullptr if
///     no weights vector is present.
/// @param dst_desc Destination memory descriptor.
/// @param padding_idx Padding Index. If no padding index is present then
///     this should be -1.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
///
zendnn_status_t ZENDNN_API
zendnn_embedding_bag_desc_init(zendnn_embedding_bag_desc_t *desc,
                               zendnn_prop_kind_t prop_kind,
                               zendnn_alg_kind_t alg_kind,
                               const zendnn_memory_desc_t *input_desc,
                               const zendnn_memory_desc_t *indices_desc,
                               const zendnn_memory_desc_t *offsets_desc,
                               const zendnn_memory_desc_t *weights_desc,
                               const zendnn_memory_desc_t *dst_desc,
                               int32_t padding_idx);
/// @} zendnn_api_embedding_bag

/// @} zendnn_api_primitives

/// @addtogroup zendnn_api_engine
/// @{

/// Returns the number of engines of a particular kind.
///
/// @param kind Kind of engines to count.
/// @returns Count of the engines.
size_t ZENDNN_API zendnn_engine_get_count(zendnn_engine_kind_t kind);

/// Creates an engine.
///
/// @param engine Output engine.
/// @param kind Engine kind.
/// @param index Engine index that should be between 0 and the count of
///     engines of the requested kind.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_engine_create(
        zendnn_engine_t *engine, zendnn_engine_kind_t kind, size_t index);

/// Returns the kind of an engine.
///
/// @param engine Engine to query.
/// @param kind Output engine kind.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_engine_get_kind(
        zendnn_engine_t engine, zendnn_engine_kind_t *kind);

/// Destroys an engine.
///
/// @param engine Engine to destroy.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_engine_destroy(zendnn_engine_t engine);

/// @} zendnn_api_engine

/// @addtogroup zendnn_api_stream
/// @{

/// Creates an execution stream.
///
/// @param stream Output execution stream.
/// @param engine Engine to create the execution stream on.
/// @param flags Stream behavior flags (@sa zendnn_stream_flags_t).
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_stream_create(
        zendnn_stream_t *stream, zendnn_engine_t engine, unsigned flags);

/// Returns the engine of a stream object.
///
/// @param stream Stream object.
/// @param engine Output engine on which the stream is created.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_stream_get_engine(
        const_zendnn_stream_t stream, zendnn_engine_t *engine);

/// Waits for all primitives in the execution stream to finish computations.
///
/// @param stream Execution stream.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_stream_wait(zendnn_stream_t stream);

/// Destroys an execution stream.
///
/// @param stream Execution stream to destroy.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_stream_destroy(zendnn_stream_t stream);

/// @} zendnn_api_stream

/// @addtogroup zendnn_api_primitive_cache
/// @{

/// Returns the number of primitives that can be held in the primitive cache
/// at the same time.
///
/// @param capacity Primitive cache capacity to query. Concurrently
/// accessing @p capacity is safe.
/// @returns #zendnn_invalid_arguments/#zendnn::status::invalid_arguments if the
///     @p capacity value is invalid, and #zendnn_success/#zendnn::status::success on
///     success.
zendnn_status_t ZENDNN_API zendnn_get_primitive_cache_capacity(int *capacity);

/// Sets a number of primitives that can be held in the primitive cache
/// at a time.
///
/// @param capacity Primitive cache capacity to set. If a new @p capacity is
/// less than a number of primitives that the primitive cache already has
/// then the excess entries will be evicted. Setting the @p capacity to 0
/// clears the primitive cache and disables it. Concurrently modifying
/// @p capacity is safe.
/// @returns #zendnn_invalid_arguments/#zendnn::status::invalid_arguments if the
///     @p capacity value is invalid, and #zendnn_success/#zendnn::status::success on
///     success.
zendnn_status_t ZENDNN_API zendnn_set_primitive_cache_capacity(int capacity);

/// @} zendnn_api_primitive_cache

/// @addtogroup zendnn_api_service
/// @{

/// Configures verbose output to stdout.
///
/// @note
///     Enabling verbose output affects performance.
///     This setting overrides the ZENDNN_VERBOSE environment variable.
///
/// @param level Verbosity level:
///  - 0: no verbose output (default),
///  - 1: primitive information at execution,
///  - 2: primitive information at creation and execution.
/// @returns #zendnn_invalid_arguments/#zendnn::status::invalid_arguments if the
///     @p level value is invalid, and #zendnn_success/#zendnn::status::success on
///     success.
zendnn_status_t ZENDNN_API zendnn_set_verbose(int level);

/// Configures dumping of JIT-generated code.
///
/// @note
///     This setting overrides the ZENDNN_JIT_DUMP environment variable.
///
/// @param enable Flag value. Set to 0 to disable and set to 1 to enable.
/// @returns #zendnn_invalid_arguments/#zendnn::status::invalid_arguments if the
///     @p flag value is invalid, and #zendnn_success/#zendnn::status::success on
///     success.
zendnn_status_t ZENDNN_API zendnn_set_jit_dump(int enable);

/// Returns library version information.
/// @returns Pointer to a constant structure containing
///  - major: major version number,
///  - minor: minor version number,
///  - patch: patch release number,
///  - hash: git commit hash.
const zendnn_version_t ZENDNN_API *zendnn_version(void);

/// Sets library profiling flags. The flags define which profilers are
/// supported.
///
/// @note
///     This setting overrides ZENDNN_JIT_PROFILE environment variable.
///
/// @sa @ref dev_guide_profilers
///
/// @param flags Profiling flags that can contain the following bits:
///     - @ref ZENDNN_JIT_PROFILE_VTUNE -- integration with VTune Amplifier
///         (on by default)
///     - @ref ZENDNN_JIT_PROFILE_LINUX_JITDUMP -- produce Linux-specific
///         jit-pid.dump output (off by default). The location of the output
///         is controlled via JITDUMPDIR environment variable or via
///         zendnn_set_jit_profiling_jitdumpdir() function.
///     - @ref ZENDNN_JIT_PROFILE_LINUX_PERFMAP -- produce Linux-specific
///         perf-pid.map output (off by default). The output is always placed
///         into /tmp.
///
///     Passing @ref ZENDNN_JIT_PROFILE_NONE disables profiling completely.
///
/// @returns #zendnn_invalid_arguments/#zendnn::status::invalid_arguments if the
///     @p flags value is invalid, and #zendnn_success/#zendnn::status::success on
///     success.
zendnn_status_t ZENDNN_API zendnn_set_jit_profiling_flags(unsigned flags);

/// Sets JIT dump output path. Only applicable to Linux and is only
/// used when profiling flags have ZENDNN_JIT_PROFILE_LINUX_PERF bit set.
///
/// After the first JIT kernel is generated, the jitdump output will be placed
/// into temporary directory created using the mkdtemp template
/// 'dir/.debug/jit/zendnn.XXXXXX'.
///
/// @sa @ref dev_guide_profilers
///
/// @note
///     This setting overrides JITDUMPDIR environment variable.  If
///     JITDUMPDIR is not set, and this function is never called, the path
///     defaults to HOME. Passing NULL reverts the value to default.
///
/// @note
///     The directory is accessed only when the first JIT kernel is being
///     created. JIT profiling will be disabled in case of any errors
///     accessing or creating this directory.
///
/// @param dir JIT dump output path.
/// @returns #zendnn_success/#zendnn::status::success if the
///     output directory was set correctly and an error status otherwise.
/// @returns #zendnn_unimplemented/#zendnn::status::unimplemented on Windows.
zendnn_status_t ZENDNN_API zendnn_set_jit_profiling_jitdumpdir(const char *dir);

/// Sets the maximal ISA the library can dispatch to on the CPU. See
/// #zendnn_cpu_isa_t and #zendnn::cpu_isa for the list of the values accepted by
/// the C and C++ API functions respectively.
///
/// This function has effect only before the first JIT kernel is generated and
/// will return an error afterwards.
///
/// This function overrides the ZENDNN_MAX_CPU_ISA environment variable. The
/// environment variable can be set to the desired maximal ISA name in upper
/// case and with zendnn_cpu_isa prefix removed. For example:
/// `ZENDNN_MAX_CPU_ISA=AVX2`.
///
/// @note
///     The ISAs are only partially ordered:
///         - SSE41 < AVX < AVX2,
///         - AVX2 < AVX512_MIC < AVX512_MIC_4OPS,
///         - AVX2 < AVX512_CORE < AVX512_CORE_VNNI < AVX512_CORE_BF16
///           < AVX512_CORE_AMX,
///         - AVX2 < AVX2_VNNI.
///
/// @sa @ref dev_guide_cpu_dispatcher_control for more details
///
/// @param isa Maximal ISA the library should dispatch to. Pass
///     #zendnn_cpu_isa_all/#zendnn::cpu_isa::all to remove ISA restrictions
///     (except for ISAs with initial support in the library).
/// @returns #zendnn_success/#zendnn::status::success on success and a
///     #zendnn_invalid_arguments/#zendnn::status::invalid_arguments if the @p isa
///     parameter is invalid or the ISA cannot be changed at this time.
/// @returns #zendnn_unimplemented/#zendnn::status::unimplemented if the feature
///     was disabled at build time (see @ref dev_guide_build_options for more
///     details).
zendnn_status_t ZENDNN_API zendnn_set_max_cpu_isa(zendnn_cpu_isa_t isa);

/// Gets the maximal ISA the library can dispatch to on the CPU. See
/// #zendnn_cpu_isa_t and #zendnn::cpu_isa for the list of the values returned by
/// the C and C++ API functions respectively.
///
/// @sa @ref dev_guide_cpu_dispatcher_control for more details
///
/// @returns #zendnn_cpu_isa_t value reflecting the maximal ISA the library may
///     dispatch to.
zendnn_cpu_isa_t ZENDNN_API zendnn_get_effective_cpu_isa(void);

/// Sets the hints flag for the CPU ISA. See #zendnn_cpu_isa_hints_t and
/// #zendnn::cpu_isa_hints for the list of the values accepted by the C and C++
/// API functions respectively.
///
/// This function has effect only before the first JIT kernel is generated and
/// will return an error afterwards.
///
/// This function overrides the ZENDNN_CPU_ISA_HINTS environment variable.
/// @sa @ref dev_guide_cpu_isa_hints for more details
///
/// @param isa_hints CPU ISA hints to be passed over to the implementation.
///     Pass #zendnn_cpu_isa_no_hints/#zendnn::cpu_isa_hints::no_hints to use
///     default features i.e. no hints.
/// @returns #zendnn_success/#zendnn::status::success on success and a
///     #zendnn_runtime_error/#zendnn::status::runtime_error if the ISA hints cannot
///     be specified at the current time.
/// @returns #zendnn_unimplemented/#zendnn::status::unimplemented if the feature
///     was disabled at build time (see @ref dev_guide_build_options for more
///     details).
zendnn_status_t ZENDNN_API zendnn_set_cpu_isa_hints(zendnn_cpu_isa_hints_t isa_hints);

/// Gets the ISA specific hints that library can follow. See
/// #zendnn_cpu_isa_hints_t and #zendnn::cpu_isa_hints for the list of the values
///  returned by the C and C++ API functions respectively.
///
/// @sa @ref dev_guide_cpu_isa_hints for more details
///
/// @returns #zendnn_cpu_isa_hints_t value reflecting the ISA specific hints the
/// library can follow.
zendnn_cpu_isa_hints_t ZENDNN_API zendnn_get_cpu_isa_hints(void);

/// @} zendnn_api_service

/// @addtogroup zendnn_api_blas
/// @{

/// Performs single-precision matrix-matrix multiply.
///
/// The operation is defined as:
///
/// `C := alpha * op( A ) * op( B ) + beta * C`
///
/// where
///  - `op( X ) = X` or `op( X ) = X**T`,
///  - `alpha` and `beta` are scalars, and
///  - `A`, `B`, and `C` are matrices:
///     - `op( A )` is an `MxK` matrix,
///     - `op( B )` is an `KxN` matrix,
///     - `C` is an `MxN` matrix.
///
/// The matrices are assumed to be stored in row-major order (the elements in
/// each of the matrix rows are contiguous in memory).
///
/// @note
///     This API does not support XERBLA. Instead, unlike the standard BLAS
///     functions, this one returns a zendnn_status_t value to allow error
///     handling.
///
/// @param transa Transposition flag for matrix A: 'N' or 'n' means A is not
///     transposed, and 'T' or 't' means that A is transposed.
/// @param transb Transposition flag for matrix B: 'N' or 'n' means B is not
///     transposed, and 'T' or 't' means that B is transposed.
/// @param M The M dimension.
/// @param N The N dimension.
/// @param K The K dimension.
/// @param alpha The alpha parameter that is used to scale the product of
///     matrices A and B.
/// @param A A pointer to the A matrix data.
/// @param lda The leading dimension for the matrix A.
/// @param B A pointer to the B matrix data.
/// @param ldb The leading dimension for the matrix B.
/// @param beta The beta parameter that is used to scale the matrix C.
/// @param C A pointer to the C matrix data.
/// @param ldc The leading dimension for the matrix C.
/// @returns #zendnn_success/#zendnn::status::success on success and a status
///     describing the error otherwise.
zendnn_status_t ZENDNN_API zendnn_sgemm(char transa, char transb, zendnn_dim_t M,
        zendnn_dim_t N, zendnn_dim_t K, float alpha, const float *A, zendnn_dim_t lda,
        const float *B, zendnn_dim_t ldb, float beta, float *C, zendnn_dim_t ldc);

/// Performs integer matrix-matrix multiply on 8-bit unsigned matrix A, 8-bit
/// signed matrix B, and 32-bit signed resulting matrix C.
///
/// The operation is defined as:
///
/// `C := alpha * (op(A) - A_offset) * (op(B) - B_offset) + beta * C + C_offset`
///
/// where
///  - `op( X ) = X` or `op( X ) = X**T`,
///  - `alpha` and `beta` are scalars, and
///  - `A`, `B`, and `C` are matrices:
///     - `op( A )` is an `MxK` matrix,
///     - `op( B )` is an `KxN` matrix,
///     - `C` is an `MxN` matrix.
///  - `A_offset` is an `MxK` matrix with every element equal the `ao` value,
///  - `B_offset` is an `KxN` matrix with every element equal the `bo` value,
///  - `C_offset` is an `MxN` matrix which is defined by the `co` array of size `len`:
///    - if `offsetc = F`: the `len` must be at least `1`,
///    - if `offsetc = C`: the `len` must be at least `max(1, m)`,
///    - if `offsetc = R`: the `len` must be at least `max(1, n)`,
///
/// The matrices are assumed to be stored in row-major order (the elements in
/// each of the matrix rows are contiguous in memory).
///
/// @note
///     This API does not support XERBLA. Instead, unlike the standard BLAS
///     functions, this one returns a zendnn_status_t value to allow error
///     handling.
///
/// @warning
///     On some architectures saturation may happen during intermediate
///     computations, which would lead to unexpected results. For more
///     details, refer to @ref dev_guide_int8_computations.
///
/// @param transa Transposition flag for matrix A: 'N' or 'n' means A is not
///     transposed, and 'T' or 't' means that A is transposed.
/// @param transb Transposition flag for matrix B: 'N' or 'n' means B is not
///     transposed, and 'T' or 't' means that B is transposed.
/// @param offsetc Flag specifying how offsets should be applied to matrix C:
///     - 'F' means that the same offset will be applied to each element of
///         the matrix C,
///     - 'C' means that individual offset will be applied to each element
///         within each column,
///     - 'R' means that individual offset will be applied to each element
///         within each row.
/// @param M The M dimension.
/// @param N The N dimension.
/// @param K The K dimension.
/// @param alpha The alpha parameter that is used to scale the product of
///     matrices A and B.
/// @param A A pointer to the A matrix data.
/// @param lda The leading dimension for the matrix A.
/// @param ao The offset value for the matrix A.
/// @param B A pointer to the B matrix data.
/// @param ldb The leading dimension for the matrix B.
/// @param bo The offset value for the matrix B.
/// @param beta The beta parameter that is used to scale the matrix C.
/// @param C A pointer to the C matrix data.
/// @param ldc The leading dimension for the matrix C.
/// @param co An array of offset values for the matrix C. The number of
///     elements in the array depends on the value of @p offsetc.
/// @returns #zendnn_success/#zendnn::status::success on success and a status
///     describing the error otherwise.
zendnn_status_t ZENDNN_API zendnn_gemm_u8s8s32(char transa, char transb, char offsetc,
        zendnn_dim_t M, zendnn_dim_t N, zendnn_dim_t K, float alpha, const uint8_t *A,
        zendnn_dim_t lda, uint8_t ao, const int8_t *B, zendnn_dim_t ldb, int8_t bo,
        float beta, int32_t *C, zendnn_dim_t ldc, const int32_t *co);

/// Performs integer matrix-matrix multiply on 8-bit signed matrix A, 8-bit
/// signed matrix B, and 32-bit signed resulting matrix C.
///
/// The operation is defined as:
///
/// `C := alpha * (op(A) - A_offset) * (op(B) - B_offset) + beta * C + C_offset`
///
/// where
///  - `op( X ) = X` or `op( X ) = X**T`,
///  - `alpha` and `beta` are scalars, and
///  - `A`, `B`, and `C` are matrices:
///     - `op( A )` is an `MxK` matrix,
///     - `op( B )` is an `KxN` matrix,
///     - `C` is an `MxN` matrix.
///  - `A_offset` is an `MxK` matrix with every element equal the `ao` value,
///  - `B_offset` is an `KxN` matrix with every element equal the `bo` value,
///  - `C_offset` is an `MxN` matrix which is defined by the `co` array of size `len`:
///    - if `offsetc = F`: the `len` must be at least `1`,
///    - if `offsetc = C`: the `len` must be at least `max(1, m)`,
///    - if `offsetc = R`: the `len` must be at least `max(1, n)`,
///
/// The matrices are assumed to be stored in row-major order (the elements in
/// each of the matrix rows are contiguous in memory).
///
/// @note
///     This API does not support XERBLA. Instead, unlike the standard BLAS
///     functions, this one returns a zendnn_status_t value to allow error
///     handling.
///
/// @warning
///     On some architectures saturation may happen during intermediate
///     computations, which would lead to unexpected results. For more
///     details, refer to @ref dev_guide_int8_computations.
///
/// @param transa Transposition flag for matrix A: 'N' or 'n' means A is not
///     transposed, and 'T' or 't' means that A is transposed.
/// @param transb Transposition flag for matrix B: 'N' or 'n' means B is not
///     transposed, and 'T' or 't' means that B is transposed.
/// @param offsetc Flag specifying how offsets should be applied to matrix C:
///     - 'F' means that the same offset will be applied to each element of
///         the matrix C,
///     - 'C' means that individual offset will be applied to each element
///         within each column,
///     - 'R' means that individual offset will be applied to each element
///         within each row.
/// @param M The M dimension.
/// @param N The N dimension.
/// @param K The K dimension.
/// @param alpha The alpha parameter that is used to scale the product of
///     matrices A and B.
/// @param A A pointer to the A matrix data.
/// @param lda The leading dimension for the matrix A.
/// @param ao The offset value for the matrix A.
/// @param B A pointer to the B matrix data.
/// @param ldb The leading dimension for the matrix B.
/// @param bo The offset value for the matrix B.
/// @param beta The beta parameter that is used to scale the matrix C.
/// @param C A pointer to the C matrix data.
/// @param ldc The leading dimension for the matrix C.
/// @param co An array of offset values for the matrix C. The number of
///     elements in the array depends on the value of @p offsetc.
/// @returns #zendnn_success/#zendnn::status::success on success and a status
///     describing the error otherwise.
zendnn_status_t ZENDNN_API zendnn_gemm_s8s8s32(char transa, char transb, char offsetc,
        zendnn_dim_t M, zendnn_dim_t N, zendnn_dim_t K, float alpha, const int8_t *A,
        zendnn_dim_t lda, int8_t ao, const int8_t *B, zendnn_dim_t ldb, int8_t bo,
        float beta, int32_t *C, zendnn_dim_t ldc, const int32_t *co);

/// @} zendnn_api_blas

/// @} zendnn_api

#ifdef __cplusplus
}
#endif

#endif /* ZENDNN_H */
