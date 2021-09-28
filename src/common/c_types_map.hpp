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

#ifndef COMMON_C_TYPES_MAP_HPP
#define COMMON_C_TYPES_MAP_HPP

#include "zendnn_types.h"

#include "gemm_types.hpp"
#include "internal_desc_types.hpp"

// These aliases should be in the global namespace as they are intended
// to give names that better reflects the meaning of the entities
using primitive_iface_t = zendnn_primitive;
using primitive_desc_iface_t = zendnn_primitive_desc;

namespace zendnn {
namespace impl {

// TODO: autogenerate this

using dim_t = zendnn_dim_t;
using dims_t = zendnn_dims_t;
using stride_t = zendnn_dim_t;
using strides_t = zendnn_dims_t;

using status_t = zendnn_status_t;
namespace status {
const status_t success = zendnn_success;
const status_t out_of_memory = zendnn_out_of_memory;
const status_t invalid_arguments = zendnn_invalid_arguments;
const status_t unimplemented = zendnn_unimplemented;
const status_t iterator_ends = zendnn_iterator_ends;
const status_t runtime_error = zendnn_runtime_error;
const status_t not_required = zendnn_not_required;
} // namespace status

using prop_kind_t = zendnn_prop_kind_t;
namespace prop_kind {
const prop_kind_t undef = zendnn_prop_kind_undef;
const prop_kind_t forward_training = zendnn_forward_training;
const prop_kind_t forward_inference = zendnn_forward_inference;
const prop_kind_t forward_scoring = zendnn_forward_scoring;
const prop_kind_t forward = zendnn_forward;
const prop_kind_t backward = zendnn_backward;
const prop_kind_t backward_data = zendnn_backward_data;
const prop_kind_t backward_weights = zendnn_backward_weights;
const prop_kind_t backward_bias = zendnn_backward_bias;
} // namespace prop_kind

using alg_kind_t = zendnn_alg_kind_t;
namespace alg_kind {
const alg_kind_t undef = zendnn_alg_kind_undef;
const alg_kind_t convolution_auto = zendnn_convolution_auto;
const alg_kind_t convolution_gemm = zendnn_convolution_gemm;
const alg_kind_t convolution_ref = zendnn_convolution_ref;
const alg_kind_t convolution_direct = zendnn_convolution_direct;
const alg_kind_t convolution_winograd = zendnn_convolution_winograd;
const alg_kind_t deconvolution_direct = zendnn_deconvolution_direct;
const alg_kind_t deconvolution_winograd = zendnn_deconvolution_winograd;
const alg_kind_t eltwise_relu = zendnn_eltwise_relu;
const alg_kind_t eltwise_tanh = zendnn_eltwise_tanh;
const alg_kind_t eltwise_elu = zendnn_eltwise_elu;
const alg_kind_t eltwise_square = zendnn_eltwise_square;
const alg_kind_t eltwise_abs = zendnn_eltwise_abs;
const alg_kind_t eltwise_sqrt = zendnn_eltwise_sqrt;
const alg_kind_t eltwise_swish = zendnn_eltwise_swish;
const alg_kind_t eltwise_linear = zendnn_eltwise_linear;
const alg_kind_t eltwise_bounded_relu = zendnn_eltwise_bounded_relu;
const alg_kind_t eltwise_soft_relu = zendnn_eltwise_soft_relu;
const alg_kind_t eltwise_logistic = zendnn_eltwise_logistic;
const alg_kind_t eltwise_logsigmoid = zendnn_eltwise_logsigmoid;
const alg_kind_t eltwise_mish = zendnn_eltwise_mish;
const alg_kind_t eltwise_exp = zendnn_eltwise_exp;
const alg_kind_t eltwise_gelu = zendnn_eltwise_gelu;
const alg_kind_t eltwise_log = zendnn_eltwise_log;
const alg_kind_t eltwise_clip = zendnn_eltwise_clip;
const alg_kind_t eltwise_clip_v2 = zendnn_eltwise_clip_v2;
const alg_kind_t eltwise_pow = zendnn_eltwise_pow;
const alg_kind_t eltwise_gelu_tanh = zendnn_eltwise_gelu_tanh;
const alg_kind_t eltwise_gelu_erf = zendnn_eltwise_gelu_erf;
const alg_kind_t eltwise_hardswish = zendnn_eltwise_hardswish;
const alg_kind_t eltwise_relu_use_dst_for_bwd
        = zendnn_eltwise_relu_use_dst_for_bwd;
const alg_kind_t eltwise_tanh_use_dst_for_bwd
        = zendnn_eltwise_tanh_use_dst_for_bwd;
const alg_kind_t eltwise_elu_use_dst_for_bwd = zendnn_eltwise_elu_use_dst_for_bwd;
const alg_kind_t eltwise_sqrt_use_dst_for_bwd
        = zendnn_eltwise_sqrt_use_dst_for_bwd;
const alg_kind_t eltwise_logistic_use_dst_for_bwd
        = zendnn_eltwise_logistic_use_dst_for_bwd;
const alg_kind_t eltwise_exp_use_dst_for_bwd = zendnn_eltwise_exp_use_dst_for_bwd;
const alg_kind_t eltwise_clip_v2_use_dst_for_bwd
        = zendnn_eltwise_clip_v2_use_dst_for_bwd;
const alg_kind_t eltwise_round = zendnn_eltwise_round;
const alg_kind_t pooling_max = zendnn_pooling_max;
const alg_kind_t pooling_avg = zendnn_pooling_avg;
const alg_kind_t pooling_avg_include_padding = zendnn_pooling_avg_include_padding;
const alg_kind_t pooling_avg_exclude_padding = zendnn_pooling_avg_exclude_padding;
const alg_kind_t lrn_across_channels = zendnn_lrn_across_channels;
const alg_kind_t lrn_within_channel = zendnn_lrn_within_channel;
const alg_kind_t vanilla_rnn = zendnn_vanilla_rnn;
const alg_kind_t vanilla_lstm = zendnn_vanilla_lstm;
const alg_kind_t vanilla_gru = zendnn_vanilla_gru;
const alg_kind_t lbr_gru = zendnn_lbr_gru;
const alg_kind_t binary_add = zendnn_binary_add;
const alg_kind_t binary_mul = zendnn_binary_mul;
const alg_kind_t binary_max = zendnn_binary_max;
const alg_kind_t binary_min = zendnn_binary_min;
const alg_kind_t binary_div = zendnn_binary_div;
const alg_kind_t binary_sub = zendnn_binary_sub;
const alg_kind_t binary_ge = zendnn_binary_ge;
const alg_kind_t binary_gt = zendnn_binary_gt;
const alg_kind_t binary_le = zendnn_binary_le;
const alg_kind_t binary_lt = zendnn_binary_lt;
const alg_kind_t binary_eq = zendnn_binary_eq;
const alg_kind_t binary_ne = zendnn_binary_ne;
const alg_kind_t resampling_nearest = zendnn_resampling_nearest;
const alg_kind_t resampling_linear = zendnn_resampling_linear;
const alg_kind_t reduction_max = zendnn_reduction_max;
const alg_kind_t reduction_min = zendnn_reduction_min;
const alg_kind_t reduction_sum = zendnn_reduction_sum;
const alg_kind_t reduction_mul = zendnn_reduction_mul;
const alg_kind_t reduction_mean = zendnn_reduction_mean;
const alg_kind_t reduction_norm_lp_max = zendnn_reduction_norm_lp_max;
const alg_kind_t reduction_norm_lp_sum = zendnn_reduction_norm_lp_sum;
const alg_kind_t reduction_norm_lp_power_p_max
        = zendnn_reduction_norm_lp_power_p_max;
const alg_kind_t reduction_norm_lp_power_p_sum
        = zendnn_reduction_norm_lp_power_p_sum;
/* add new primitive */
const alg_kind_t embedding_bag_sum  = zendnn_embedding_bag_sum;
const alg_kind_t embedding_bag_mean = zendnn_embedding_bag_mean;
const alg_kind_t embedding_bag_max  = zendnn_embedding_bag_max;
} // namespace alg_kind

using data_type_t = zendnn_data_type_t;
namespace data_type {
const data_type_t undef = zendnn_data_type_undef;
const data_type_t f16 = zendnn_f16;
const data_type_t bf16 = zendnn_bf16;
const data_type_t f32 = zendnn_f32;
const data_type_t s32 = zendnn_s32;
const data_type_t s8 = zendnn_s8;
const data_type_t u8 = zendnn_u8;
} // namespace data_type

using scratchpad_mode_t = zendnn_scratchpad_mode_t;
namespace scratchpad_mode {
const scratchpad_mode_t library = zendnn_scratchpad_mode_library;
const scratchpad_mode_t user = zendnn_scratchpad_mode_user;
} // namespace scratchpad_mode

using rnn_packed_format_t = zendnn_rnn_packed_memory_format_t;
namespace rnn_packed_format {
const rnn_packed_format_t undef = zendnn_packed_format_undef;
const rnn_packed_format_t ldigo_p = zendnn_ldigo_p;
const rnn_packed_format_t ldgoi_p = zendnn_ldgoi_p;
const rnn_packed_format_t ldio_p = zendnn_ldio_p;
} // namespace rnn_packed_format

using format_kind_t = zendnn_format_kind_t;
namespace format_kind {
const format_kind_t undef = zendnn_format_kind_undef;
const format_kind_t any = zendnn_format_kind_any;
const format_kind_t blocked = zendnn_blocked;
const format_kind_t wino = zendnn_format_kind_wino;
const format_kind_t rnn_packed = zendnn_format_kind_rnn_packed;
} // namespace format_kind

using format_tag_t = zendnn_format_tag_t;
namespace format_tag {
const format_tag_t undef = zendnn_format_tag_undef;
const format_tag_t any = zendnn_format_tag_any;
const format_tag_t a = zendnn_a;
const format_tag_t ab = zendnn_ab;
const format_tag_t abc = zendnn_abc;
const format_tag_t abcd = zendnn_abcd;
const format_tag_t abcde = zendnn_abcde;
const format_tag_t abcdef = zendnn_abcdef;
const format_tag_t abcdefg = zendnn_abcdefg;
const format_tag_t abcdefgh = zendnn_abcdefgh;
const format_tag_t abcdefghi = zendnn_abcdefghi;
const format_tag_t abcdefghij = zendnn_abcdefghij;
const format_tag_t abcdefghijk = zendnn_abcdefghijk;
const format_tag_t abcdefghijkl = zendnn_abcdefghijkl;
const format_tag_t abcdefghijlk = zendnn_abcdefghijlk;
const format_tag_t abcdefghikj = zendnn_abcdefghikj;
const format_tag_t abcdefghji = zendnn_abcdefghji;
const format_tag_t abcdefgih = zendnn_abcdefgih;
const format_tag_t abcdefhg = zendnn_abcdefhg;
const format_tag_t abcdegf = zendnn_abcdegf;
const format_tag_t abcdfe = zendnn_abcdfe;
const format_tag_t abced = zendnn_abced;
const format_tag_t abdc = zendnn_abdc;
const format_tag_t acbd = zendnn_acbd;
const format_tag_t abdec = zendnn_abdec;
const format_tag_t abdfce = zendnn_abdfce;
const format_tag_t acb = zendnn_acb;
const format_tag_t acbde = zendnn_acbde;
const format_tag_t acbdef = zendnn_acbdef;
const format_tag_t abdefc = zendnn_abdefc;
const format_tag_t acdb = zendnn_acdb;
const format_tag_t acdeb = zendnn_acdeb;
const format_tag_t ba = zendnn_ba;
const format_tag_t bac = zendnn_bac;
const format_tag_t bacd = zendnn_bacd;
const format_tag_t bca = zendnn_bca;
const format_tag_t bcda = zendnn_bcda;
const format_tag_t bcdea = zendnn_bcdea;
const format_tag_t bacde = zendnn_bacde;
const format_tag_t cba = zendnn_cba;
const format_tag_t cdba = zendnn_cdba;
const format_tag_t dcab = zendnn_dcab;
const format_tag_t cdeba = zendnn_cdeba;
const format_tag_t decab = zendnn_decab;
const format_tag_t defcab = zendnn_defcab;
const format_tag_t AB16b16a = zendnn_AB16b16a;
const format_tag_t AB16b32a = zendnn_AB16b32a;
const format_tag_t AB16b64a = zendnn_AB16b64a;
const format_tag_t AB8b16a2b = zendnn_AB8b16a2b;
const format_tag_t AB8b32a2b = zendnn_AB8b32a2b;
const format_tag_t AB8b64a2b = zendnn_AB8b64a2b;
const format_tag_t AB4b16a4b = zendnn_AB4b16a4b;
const format_tag_t AB4b32a4b = zendnn_AB4b32a4b;
const format_tag_t AB4b64a4b = zendnn_AB4b64a4b;
const format_tag_t AB16b16a4b = zendnn_AB16b16a4b;
const format_tag_t AB16b32a4b = zendnn_AB16b32a4b;
const format_tag_t AB16b48a4b = zendnn_AB16b48a4b;
const format_tag_t AB16b64a4b = zendnn_AB16b64a4b;
const format_tag_t AB16b16a2b = zendnn_AB16b16a2b;
const format_tag_t AB16b32a2b = zendnn_AB16b32a2b;
const format_tag_t AB16b48a2b = zendnn_AB16b48a2b;
const format_tag_t AB16b64a2b = zendnn_AB16b64a2b;
const format_tag_t Abc16a = zendnn_Abc16a;
const format_tag_t ABc16a16b = zendnn_ABc16a16b;
const format_tag_t ABc4a4b = zendnn_ABc4a4b;
const format_tag_t aBc16b = zendnn_aBc16b;
const format_tag_t aBc32b = zendnn_aBc32b;
const format_tag_t ABc16b16a = zendnn_ABc16b16a;
const format_tag_t ABc16b32a = zendnn_ABc16b32a;
const format_tag_t ABc16b64a = zendnn_ABc16b64a;
const format_tag_t Abc4a = zendnn_Abc4a;
const format_tag_t aBc4b = zendnn_aBc4b;
const format_tag_t ABc4b16a4b = zendnn_ABc4b16a4b;
const format_tag_t ABc4b32a4b = zendnn_ABc4b32a4b;
const format_tag_t ABc4b64a4b = zendnn_ABc4b64a4b;
const format_tag_t ABc2b8a4b = zendnn_ABc2b8a4b;
const format_tag_t ABc16b16a4b = zendnn_ABc16b16a4b;
const format_tag_t ABc16b32a4b = zendnn_ABc16b32a4b;
const format_tag_t ABc16b48a4b = zendnn_ABc16b48a4b;
const format_tag_t ABc16b64a4b = zendnn_ABc16b64a4b;
const format_tag_t ABc16b16a2b = zendnn_ABc16b16a2b;
const format_tag_t ABc16b32a2b = zendnn_ABc16b32a2b;
const format_tag_t ABc16b48a2b = zendnn_ABc16b48a2b;
const format_tag_t ABc16b64a2b = zendnn_ABc16b64a2b;
const format_tag_t ABc16a16b2a = zendnn_ABc16a16b2a;
const format_tag_t ABc4b4a = zendnn_ABc4b4a;
const format_tag_t ABc8a16b2a = zendnn_ABc8a16b2a;
const format_tag_t BAc8a16b2a = zendnn_BAc8a16b2a;
const format_tag_t ABc8a8b = zendnn_ABc8a8b;
const format_tag_t ABc8a4b = zendnn_ABc8a4b;
const format_tag_t aBc8b = zendnn_aBc8b;
const format_tag_t ABc8b16a2b = zendnn_ABc8b16a2b;
const format_tag_t ABc8b32a2b = zendnn_ABc8b32a2b;
const format_tag_t ABc8b64a2b = zendnn_ABc8b64a2b;
const format_tag_t ABc8b8a = zendnn_ABc8b8a;
const format_tag_t Abcd16a = zendnn_Abcd16a;
const format_tag_t Abcd8a = zendnn_Abcd8a;
const format_tag_t Abcd32a = zendnn_Abcd32a;
const format_tag_t ABcd16a16b = zendnn_ABcd16a16b;
const format_tag_t aBcd16b = zendnn_aBcd16b;
const format_tag_t aBcd32b = zendnn_aBcd32b;
const format_tag_t ABcd16b16a = zendnn_ABcd16b16a;
const format_tag_t ABcd16b32a = zendnn_ABcd16b32a;
const format_tag_t ABcd16b64a = zendnn_ABcd16b64a;
const format_tag_t aBCd16b16c = zendnn_aBCd16b16c;
const format_tag_t aBCd16c16b = zendnn_aBCd16c16b;
const format_tag_t Abcd4a = zendnn_Abcd4a;
const format_tag_t aBcd4b = zendnn_aBcd4b;
const format_tag_t ABcd4b16a4b = zendnn_ABcd4b16a4b;
const format_tag_t ABcd4b32a4b = zendnn_ABcd4b32a4b;
const format_tag_t ABcd4b64a4b = zendnn_ABcd4b64a4b;
const format_tag_t ABcd16b16a4b = zendnn_ABcd16b16a4b;
const format_tag_t ABcd16b32a4b = zendnn_ABcd16b32a4b;
const format_tag_t ABcd16b48a4b = zendnn_ABcd16b48a4b;
const format_tag_t ABcd16b64a4b = zendnn_ABcd16b64a4b;
const format_tag_t ABcd16b16a2b = zendnn_ABcd16b16a2b;
const format_tag_t ABcd16b32a2b = zendnn_ABcd16b32a2b;
const format_tag_t ABcd16b48a2b = zendnn_ABcd16b48a2b;
const format_tag_t ABcd16b64a2b = zendnn_ABcd16b64a2b;
const format_tag_t ABcd16a16b2a = zendnn_ABcd16a16b2a;
const format_tag_t ABcd4b4a = zendnn_ABcd4b4a;
const format_tag_t ABcd4a4b = zendnn_ABcd4a4b;
const format_tag_t aBCd4c16b4c = zendnn_aBCd4c16b4c;
const format_tag_t aBCd2c8b4c = zendnn_aBCd2c8b4c;
const format_tag_t aBCd16c16b4c = zendnn_aBCd16c16b4c;
const format_tag_t aBCd16c16b2c = zendnn_aBCd16c16b2c;
const format_tag_t aBCd16b16c2b = zendnn_aBCd16b16c2b;
const format_tag_t aBCd4c4b = zendnn_aBCd4c4b;
const format_tag_t aBCd4b4c = zendnn_aBCd4b4c;
const format_tag_t ABcd8a16b2a = zendnn_ABcd8a16b2a;
const format_tag_t BAcd8a16b2a = zendnn_BAcd8a16b2a;
const format_tag_t ABcd8a8b = zendnn_ABcd8a8b;
const format_tag_t ABcd8a4b = zendnn_ABcd8a4b;
const format_tag_t aBcd8b = zendnn_aBcd8b;
const format_tag_t ABcd8b16a2b = zendnn_ABcd8b16a2b;
const format_tag_t ABcd8b32a2b = zendnn_ABcd8b32a2b;
const format_tag_t ABcd8b64a2b = zendnn_ABcd8b64a2b;
const format_tag_t ABcd2b8a4b = zendnn_ABcd2b8a4b;
const format_tag_t aBCd8b16c2b = zendnn_aBCd8b16c2b;
const format_tag_t aCBd8b16c2b = zendnn_aCBd8b16c2b;
const format_tag_t ABcd8b8a = zendnn_ABcd8b8a;
const format_tag_t aBCd8b8c = zendnn_aBCd8b8c;
const format_tag_t aBCd8b4c = zendnn_aBCd8b4c;
const format_tag_t aBCd8c16b2c = zendnn_aBCd8c16b2c;
const format_tag_t aBCd8c8b = zendnn_aBCd8c8b;
const format_tag_t Abcde16a = zendnn_Abcde16a;
const format_tag_t Abcde32a = zendnn_Abcde32a;
const format_tag_t ABcde16a16b = zendnn_ABcde16a16b;
const format_tag_t aBcde16b = zendnn_aBcde16b;
const format_tag_t aBcde32b = zendnn_aBcde32b;
const format_tag_t ABcde16b16a = zendnn_ABcde16b16a;
const format_tag_t ABcde16b32a = zendnn_ABcde16b32a;
const format_tag_t ABcde16b64a = zendnn_ABcde16b64a;
const format_tag_t aBCde16b16c = zendnn_aBCde16b16c;
const format_tag_t aBCde16c16b = zendnn_aBCde16c16b;
const format_tag_t aBCde2c8b4c = zendnn_aBCde2c8b4c;
const format_tag_t Abcde4a = zendnn_Abcde4a;
const format_tag_t aBcde4b = zendnn_aBcde4b;
const format_tag_t ABcde4b4a = zendnn_ABcde4b4a;
const format_tag_t ABcde4a4b = zendnn_ABcde4a4b;
const format_tag_t aBCde4b4c = zendnn_aBCde4b4c;
const format_tag_t aBCde4c16b4c = zendnn_aBCde4c16b4c;
const format_tag_t aBCde16c16b4c = zendnn_aBCde16c16b4c;
const format_tag_t aBCde16c16b2c = zendnn_aBCde16c16b2c;
const format_tag_t aBCde16b16c2b = zendnn_aBCde16b16c2b;
const format_tag_t aBCde4c4b = zendnn_aBCde4c4b;
const format_tag_t Abcde8a = zendnn_Abcde8a;
const format_tag_t ABcde8a8b = zendnn_ABcde8a8b;
const format_tag_t ABcde8a4b = zendnn_ABcde8a4b;
const format_tag_t aBcde8b = zendnn_aBcde8b;
const format_tag_t ABcde8b16a2b = zendnn_ABcde8b16a2b;
const format_tag_t ABcde8b32a2b = zendnn_ABcde8b32a2b;
const format_tag_t ABcde8b64a2b = zendnn_ABcde8b64a2b;
const format_tag_t ABcde8a16b2a = zendnn_ABcde8a16b2a;
const format_tag_t BAcde8a16b2a = zendnn_BAcde8a16b2a;
const format_tag_t ABcde4b16a4b = zendnn_ABcde4b16a4b;
const format_tag_t ABcde4b32a4b = zendnn_ABcde4b32a4b;
const format_tag_t ABcde4b64a4b = zendnn_ABcde4b64a4b;
const format_tag_t ABcde16b16a4b = zendnn_ABcde16b16a4b;
const format_tag_t ABcde16b32a4b = zendnn_ABcde16b32a4b;
const format_tag_t ABcde16b48a4b = zendnn_ABcde16b48a4b;
const format_tag_t ABcde16b64a4b = zendnn_ABcde16b64a4b;
const format_tag_t ABcde2b8a4b = zendnn_ABcde2b8a4b;
const format_tag_t aBCde8b16c2b = zendnn_aBCde8b16c2b;
const format_tag_t aCBde8b16c2b = zendnn_aCBde8b16c2b;
const format_tag_t ABcde8b8a = zendnn_ABcde8b8a;
const format_tag_t aBCde8b8c = zendnn_aBCde8b8c;
const format_tag_t aBCde8b4c = zendnn_aBCde8b4c;
const format_tag_t ABc4a8b8a4b = zendnn_ABc4a8b8a4b;
const format_tag_t ABcd4a8b8a4b = zendnn_ABcd4a8b8a4b;
const format_tag_t ABcde4a8b8a4b = zendnn_ABcde4a8b8a4b;
const format_tag_t ABcd2a8b8a2b = zendnn_ABcd2a8b8a2b;
const format_tag_t aBCd4b8c8b4c = zendnn_aBCd4b8c8b4c;
const format_tag_t aBCde4b8c8b4c = zendnn_aBCde4b8c8b4c;
const format_tag_t aBCdef4b8c8b4c = zendnn_aBCdef4b8c8b4c;
const format_tag_t BAc4b8a8b4a = zendnn_BAc4b8a8b4a;
const format_tag_t BAcd4b8a8b4a = zendnn_BAcd4b8a8b4a;
const format_tag_t BAcde4b8a8b4a = zendnn_BAcde4b8a8b4a;
const format_tag_t aCBd4c8b8c4b = zendnn_aCBd4c8b8c4b;
const format_tag_t aCBde4c8b8c4b = zendnn_aCBde4c8b8c4b;
const format_tag_t aCBdef4c8b8c4b = zendnn_aCBdef4c8b8c4b;
const format_tag_t aBCde2b8c8b2c = zendnn_aBCde2b8c8b2c;
const format_tag_t aBCde8c16b2c = zendnn_aBCde8c16b2c;
const format_tag_t aBCde8c8b = zendnn_aBCde8c8b;
const format_tag_t aBcdef16b = zendnn_aBcdef16b;
const format_tag_t aBCdef16b16c = zendnn_aBCdef16b16c;
const format_tag_t aBCdef16c16b = zendnn_aBCdef16c16b;
const format_tag_t aBCdef4c16b4c = zendnn_aBCdef4c16b4c;
const format_tag_t aBCdef2c8b4c = zendnn_aBCdef2c8b4c;
const format_tag_t aBcdef4b = zendnn_aBcdef4b;
const format_tag_t aBCdef4c4b = zendnn_aBCdef4c4b;
const format_tag_t aBCdef4b4c = zendnn_aBCdef4b4c;
const format_tag_t aBCdef8b8c = zendnn_aBCdef8b8c;
const format_tag_t aBCdef8b4c = zendnn_aBCdef8b4c;
const format_tag_t aBCdef8c16b2c = zendnn_aBCdef8c16b2c;
const format_tag_t aBCdef8b16c2b = zendnn_aBCdef8b16c2b;
const format_tag_t aCBdef8b16c2b = zendnn_aCBdef8b16c2b;
const format_tag_t aBCdef8c8b = zendnn_aBCdef8c8b;
const format_tag_t aBdc16b = zendnn_aBdc16b;
const format_tag_t aBdC16b2c = zendnn_aBdC16b2c;
const format_tag_t aBdC16b4c = zendnn_aBdC16b4c;
const format_tag_t aBdc4b = zendnn_aBdc4b;
const format_tag_t aBdc8b = zendnn_aBdc8b;
const format_tag_t aBdec16b = zendnn_aBdec16b;
const format_tag_t aBdeC16b2c = zendnn_aBdeC16b2c;
const format_tag_t aBdeC16b4c = zendnn_aBdeC16b4c;
const format_tag_t aBdec4b = zendnn_aBdec4b;
const format_tag_t aBdec8b = zendnn_aBdec8b;
const format_tag_t aBdefc16b = zendnn_aBdefc16b;
const format_tag_t aBdefC16b2c = zendnn_aBdefC16b2c;
const format_tag_t aBdefC16b4c = zendnn_aBdefC16b4c;
const format_tag_t aCBdef16c16b = zendnn_aCBdef16c16b;
const format_tag_t aCBdef16b16c = zendnn_aCBdef16b16c;
const format_tag_t aBdefc4b = zendnn_aBdefc4b;
const format_tag_t aBdefc8b = zendnn_aBdefc8b;
const format_tag_t aBedc16b = zendnn_aBedc16b;
const format_tag_t Acb16a = zendnn_Acb16a;
const format_tag_t AcB16a2b = zendnn_AcB16a2b;
const format_tag_t AcB16a4b = zendnn_AcB16a4b;
const format_tag_t Acb4a = zendnn_Acb4a;
const format_tag_t Acb8a = zendnn_Acb8a;
const format_tag_t aCBd16b16c = zendnn_aCBd16b16c;
const format_tag_t aCBd16c16b = zendnn_aCBd16c16b;
const format_tag_t aCBde16b16c = zendnn_aCBde16b16c;
const format_tag_t aCBde16c16b = zendnn_aCBde16c16b;
const format_tag_t Acdb16a = zendnn_Acdb16a;
const format_tag_t AcdB16a2b = zendnn_AcdB16a2b;
const format_tag_t AcdB16a4b = zendnn_AcdB16a4b;
const format_tag_t Acdb4a = zendnn_Acdb4a;
const format_tag_t Acdb8a = zendnn_Acdb8a;
const format_tag_t Acdeb16a = zendnn_Acdeb16a;
const format_tag_t AcdeB16a2b = zendnn_AcdeB16a2b;
const format_tag_t AcdeB16a4b = zendnn_AcdeB16a4b;
const format_tag_t Acdeb4a = zendnn_Acdeb4a;
const format_tag_t Acdeb8a = zendnn_Acdeb8a;
const format_tag_t Adcb16a = zendnn_Adcb16a;
const format_tag_t BAc16a16b = zendnn_BAc16a16b;
const format_tag_t BAcd16a16b = zendnn_BAcd16a16b;
const format_tag_t ABc32a32b = zendnn_ABc32a32b;
const format_tag_t BAcde16a16b = zendnn_BAcde16a16b;
const format_tag_t ABcd32a32b = zendnn_ABcd32a32b;
const format_tag_t ABcde32a32b = zendnn_ABcde32a32b;
const format_tag_t BAcde16b16a = zendnn_BAcde16b16a;
const format_tag_t aBdec32b = zendnn_aBdec32b;
const format_tag_t Abcdef16a = zendnn_Abcdef16a;
const format_tag_t Abcdef32a = zendnn_Abcdef32a;
const format_tag_t Acdb32a = zendnn_Acdb32a;
const format_tag_t BAc16b16a = zendnn_BAc16b16a;
const format_tag_t BAcd16b16a = zendnn_BAcd16b16a;
const format_tag_t aBCd2b4c2b = zendnn_aBCd2b4c2b;
const format_tag_t aBCde2b4c2b = zendnn_aBCde2b4c2b;
const format_tag_t aBCdef2b4c2b = zendnn_aBCdef2b4c2b;
const format_tag_t aBCd2c4b2c = zendnn_aBCd2c4b2c;
const format_tag_t aBCde2c4b2c = zendnn_aBCde2c4b2c;
const format_tag_t aBCdef2c4b2c = zendnn_aBCdef2c4b2c;
const format_tag_t aBCd4b8c2b = zendnn_aBCd4b8c2b;
const format_tag_t aBCde4b8c2b = zendnn_aBCde4b8c2b;
const format_tag_t aBCdef4b8c2b = zendnn_aBCdef4b8c2b;
const format_tag_t aBCd4c8b2c = zendnn_aBCd4c8b2c;
const format_tag_t aBCde4c8b2c = zendnn_aBCde4c8b2c;
const format_tag_t aBCdef4c8b2c = zendnn_aBCdef4c8b2c;
const format_tag_t AB32a32b8a4b = zendnn_AB32a32b8a4b;
const format_tag_t AB8a4b = zendnn_AB8a4b;
const format_tag_t AB32a32b8a2b = zendnn_AB32a32b8a2b;
const format_tag_t AB8a2b = zendnn_AB8a2b;
const format_tag_t abDc32d = zendnn_abDc32d;
const format_tag_t abDC32d4c = zendnn_abDC32d4c;
const format_tag_t abdEc32e = zendnn_abdEc32e;
const format_tag_t abdEC32e2c = zendnn_abdEC32e2c;
const format_tag_t abdEC32e4c = zendnn_abdEC32e4c;
const format_tag_t aBCdef16c16b4c = zendnn_aBCdef16c16b4c;
const format_tag_t ABcde16b16a2b = zendnn_ABcde16b16a2b;
const format_tag_t ABcde16b32a2b = zendnn_ABcde16b32a2b;
const format_tag_t ABcde16b48a2b = zendnn_ABcde16b48a2b;
const format_tag_t ABcde16b64a2b = zendnn_ABcde16b64a2b;
const format_tag_t aBCdef16c16b2c = zendnn_aBCdef16c16b2c;
const format_tag_t cBa2b = zendnn_cBa2b;
const format_tag_t cBa4b = zendnn_cBa4b;
const format_tag_t adcb = zendnn_adcb;
const format_tag_t adCb2c = zendnn_adCb2c;
const format_tag_t adCb4c = zendnn_adCb4c;
const format_tag_t cdBa2b = zendnn_cdBa2b;
const format_tag_t cdBa4b = zendnn_cdBa4b;
const format_tag_t adecb = zendnn_adecb;
const format_tag_t adeCb2c = zendnn_adeCb2c;
const format_tag_t adeCb4c = zendnn_adeCb4c;
const format_tag_t cdeBa2b = zendnn_cdeBa2b;
const format_tag_t cdeBa4b = zendnn_cdeBa4b;
const format_tag_t adefcb = zendnn_adefcb;
const format_tag_t adefCb2c = zendnn_adefCb2c;
const format_tag_t adefCb4c = zendnn_adefCb4c;
const format_tag_t Acb32a = zendnn_Acb32a;
const format_tag_t AcB32a2b = zendnn_AcB32a2b;
const format_tag_t AcB32a4b = zendnn_AcB32a4b;
const format_tag_t Acb48a = zendnn_Acb48a;
const format_tag_t AcB48a2b = zendnn_AcB48a2b;
const format_tag_t AcB48a4b = zendnn_AcB48a4b;
const format_tag_t Acb64a = zendnn_Acb64a;
const format_tag_t AcB64a2b = zendnn_AcB64a2b;
const format_tag_t AcB64a4b = zendnn_AcB64a4b;
const format_tag_t aBdc32b = zendnn_aBdc32b;
const format_tag_t aBdC32b2c = zendnn_aBdC32b2c;
const format_tag_t aBdC32b4c = zendnn_aBdC32b4c;
const format_tag_t aBdc48b = zendnn_aBdc48b;
const format_tag_t aBdC48b2c = zendnn_aBdC48b2c;
const format_tag_t aBdC48b4c = zendnn_aBdC48b4c;
const format_tag_t aBdc64b = zendnn_aBdc64b;
const format_tag_t aBdC64b2c = zendnn_aBdC64b2c;
const format_tag_t aBdC64b4c = zendnn_aBdC64b4c;
const format_tag_t AcdB32a2b = zendnn_AcdB32a2b;
const format_tag_t AcdB32a4b = zendnn_AcdB32a4b;
const format_tag_t Acdb48a = zendnn_Acdb48a;
const format_tag_t AcdB48a2b = zendnn_AcdB48a2b;
const format_tag_t AcdB48a4b = zendnn_AcdB48a4b;
const format_tag_t Acdb64a = zendnn_Acdb64a;
const format_tag_t AcdB64a2b = zendnn_AcdB64a2b;
const format_tag_t AcdB64a4b = zendnn_AcdB64a4b;
const format_tag_t aBdeC32b2c = zendnn_aBdeC32b2c;
const format_tag_t aBdeC32b4c = zendnn_aBdeC32b4c;
const format_tag_t aBdec48b = zendnn_aBdec48b;
const format_tag_t aBdeC48b2c = zendnn_aBdeC48b2c;
const format_tag_t aBdeC48b4c = zendnn_aBdeC48b4c;
const format_tag_t aBdec64b = zendnn_aBdec64b;
const format_tag_t aBdeC64b2c = zendnn_aBdeC64b2c;
const format_tag_t aBdeC64b4c = zendnn_aBdeC64b4c;
const format_tag_t Acdeb32a = zendnn_Acdeb32a;
const format_tag_t AcdeB32a2b = zendnn_AcdeB32a2b;
const format_tag_t AcdeB32a4b = zendnn_AcdeB32a4b;
const format_tag_t Acdeb48a = zendnn_Acdeb48a;
const format_tag_t AcdeB48a2b = zendnn_AcdeB48a2b;
const format_tag_t AcdeB48a4b = zendnn_AcdeB48a4b;
const format_tag_t Acdeb64a = zendnn_Acdeb64a;
const format_tag_t AcdeB64a2b = zendnn_AcdeB64a2b;
const format_tag_t AcdeB64a4b = zendnn_AcdeB64a4b;
const format_tag_t aBdefc32b = zendnn_aBdefc32b;
const format_tag_t aBdefC32b2c = zendnn_aBdefC32b2c;
const format_tag_t aBdefC32b4c = zendnn_aBdefC32b4c;
const format_tag_t aBdefc48b = zendnn_aBdefc48b;
const format_tag_t aBdefC48b2c = zendnn_aBdefC48b2c;
const format_tag_t aBdefC48b4c = zendnn_aBdefC48b4c;
const format_tag_t aBdefc64b = zendnn_aBdefc64b;
const format_tag_t aBdefC64b2c = zendnn_aBdefC64b2c;
const format_tag_t aBdefC64b4c = zendnn_aBdefC64b4c;

const format_tag_t last = zendnn_format_tag_last;

const format_tag_t x = zendnn_x;
const format_tag_t nc = zendnn_nc;
const format_tag_t cn = zendnn_cn;
const format_tag_t ncw = zendnn_ncw;
const format_tag_t nwc = zendnn_nwc;
const format_tag_t nchw = zendnn_nchw;
const format_tag_t nhwc = zendnn_nhwc;
const format_tag_t chwn = zendnn_chwn;
const format_tag_t ncdhw = zendnn_ncdhw;
const format_tag_t ndhwc = zendnn_ndhwc;
const format_tag_t oi = zendnn_oi;
const format_tag_t io = zendnn_io;
const format_tag_t oiw = zendnn_oiw;
const format_tag_t wio = zendnn_wio;
const format_tag_t owi = zendnn_owi;
const format_tag_t iwo = zendnn_iwo;
const format_tag_t oihw = zendnn_oihw;
const format_tag_t hwio = zendnn_hwio;
const format_tag_t ohwi = zendnn_ohwi;
const format_tag_t ihwo = zendnn_ihwo;
const format_tag_t iohw = zendnn_iohw;
const format_tag_t oidhw = zendnn_oidhw;
const format_tag_t dhwio = zendnn_dhwio;
const format_tag_t odhwi = zendnn_odhwi;
const format_tag_t idhwo = zendnn_idhwo;
const format_tag_t iodhw = zendnn_iodhw;
const format_tag_t goiw = zendnn_goiw;
const format_tag_t goihw = zendnn_goihw;
const format_tag_t wigo = zendnn_wigo;
const format_tag_t hwigo = zendnn_hwigo;
const format_tag_t dhwigo = zendnn_dhwigo;
const format_tag_t giohw = zendnn_giohw;
const format_tag_t goidhw = zendnn_goidhw;
const format_tag_t giodhw = zendnn_giodhw;
const format_tag_t gowi = zendnn_gowi;
const format_tag_t gohwi = zendnn_gohwi;
const format_tag_t godhwi = zendnn_godhwi;
const format_tag_t tnc = zendnn_tnc;
const format_tag_t ntc = zendnn_ntc;
const format_tag_t ldnc = zendnn_ldnc;
const format_tag_t ldigo = zendnn_ldigo;
const format_tag_t ldgoi = zendnn_ldgoi;
const format_tag_t ldio = zendnn_ldio;
const format_tag_t ldoi = zendnn_ldoi;
const format_tag_t ldgo = zendnn_ldgo;
const format_tag_t nCdhw32c = zendnn_nCdhw32c;
const format_tag_t nCdhw16c = zendnn_nCdhw16c;
const format_tag_t nCdhw4c = zendnn_nCdhw4c;
const format_tag_t nCdhw8c = zendnn_nCdhw8c;
const format_tag_t nChw32c = zendnn_nChw32c;
const format_tag_t nChw16c = zendnn_nChw16c;
const format_tag_t nChw4c = zendnn_nChw4c;
const format_tag_t nChw8c = zendnn_nChw8c;
const format_tag_t nCw32c = zendnn_nCw32c;
const format_tag_t nCw16c = zendnn_nCw16c;
const format_tag_t nCw4c = zendnn_nCw4c;
const format_tag_t nCw8c = zendnn_nCw8c;
const format_tag_t NCw16n16c = zendnn_NCw16n16c;
const format_tag_t NChw16n16c = zendnn_NChw16n16c;
const format_tag_t NCdhw16n16c = zendnn_NCdhw16n16c;
const format_tag_t NCw32n32c = zendnn_NCw32n32c;
const format_tag_t NChw32n32c = zendnn_NChw32n32c;
const format_tag_t NCdhw32n32c = zendnn_NCdhw32n32c;
const format_tag_t OI16i16o = zendnn_OI16i16o;
const format_tag_t OI16i32o = zendnn_OI16i32o;
const format_tag_t OI16i64o = zendnn_OI16i64o;
const format_tag_t OI8i16o2i = zendnn_OI8i16o2i;
const format_tag_t OI8i32o2i = zendnn_OI8i32o2i;
const format_tag_t OI8i64o2i = zendnn_OI8i64o2i;
const format_tag_t OI4i16o4i = zendnn_OI4i16o4i;
const format_tag_t OI4i32o4i = zendnn_OI4i32o4i;
const format_tag_t OI4i64o4i = zendnn_OI4i64o4i;
const format_tag_t OI16i16o4i = zendnn_OI16i16o4i;
const format_tag_t OI16i32o4i = zendnn_OI16i32o4i;
const format_tag_t OI16i48o4i = zendnn_OI16i48o4i;
const format_tag_t OI16i64o4i = zendnn_OI16i64o4i;
const format_tag_t OI16i16o2i = zendnn_OI16i16o2i;
const format_tag_t OI16i32o2i = zendnn_OI16i32o2i;
const format_tag_t OI16i48o2i = zendnn_OI16i48o2i;
const format_tag_t OI16i64o2i = zendnn_OI16i64o2i;
const format_tag_t IOdhw16i16o = zendnn_IOdhw16i16o;
const format_tag_t IOhw16i16o = zendnn_IOhw16i16o;
const format_tag_t Ohwi32o = zendnn_Ohwi32o;
const format_tag_t gIOhw16i16o = zendnn_gIOhw16i16o;
const format_tag_t gOhwi32o = zendnn_gOhwi32o;
const format_tag_t Goidhw16g = zendnn_Goidhw16g;
const format_tag_t IOw16o16i = zendnn_IOw16o16i;
const format_tag_t IOw16i16o = zendnn_IOw16i16o;
const format_tag_t gIOw16i16o = zendnn_gIOw16i16o;
const format_tag_t OIw16i16o = zendnn_OIw16i16o;
const format_tag_t OIw16i32o = zendnn_OIw16i32o;
const format_tag_t OIw16i64o = zendnn_OIw16i64o;
const format_tag_t OIw16o16i = zendnn_OIw16o16i;
const format_tag_t Oiw16o = zendnn_Oiw16o;
const format_tag_t OIw4i16o4i = zendnn_OIw4i16o4i;
const format_tag_t OIw4i32o4i = zendnn_OIw4i32o4i;
const format_tag_t OIw4i64o4i = zendnn_OIw4i64o4i;
const format_tag_t OIw2i8o4i = zendnn_OIw2i8o4i;
const format_tag_t OIw16i16o4i = zendnn_OIw16i16o4i;
const format_tag_t OIw16i32o4i = zendnn_OIw16i32o4i;
const format_tag_t OIw16i48o4i = zendnn_OIw16i48o4i;
const format_tag_t OIw16i64o4i = zendnn_OIw16i64o4i;
const format_tag_t OIw16i16o2i = zendnn_OIw16i16o2i;
const format_tag_t OIw16i32o2i = zendnn_OIw16i32o2i;
const format_tag_t OIw16i48o2i = zendnn_OIw16i48o2i;
const format_tag_t OIw16i64o2i = zendnn_OIw16i64o2i;
const format_tag_t OIw16o16i2o = zendnn_OIw16o16i2o;
const format_tag_t OIw4i4o = zendnn_OIw4i4o;
const format_tag_t OIw4o4i = zendnn_OIw4o4i;
const format_tag_t Oiw4o = zendnn_Oiw4o;
const format_tag_t OIw8i16o2i = zendnn_OIw8i16o2i;
const format_tag_t OIw8i32o2i = zendnn_OIw8i32o2i;
const format_tag_t OIw8i64o2i = zendnn_OIw8i64o2i;
const format_tag_t OIw8i8o = zendnn_OIw8i8o;
const format_tag_t OIw8o16i2o = zendnn_OIw8o16i2o;
const format_tag_t IOw8o16i2o = zendnn_IOw8o16i2o;
const format_tag_t OIw8o8i = zendnn_OIw8o8i;
const format_tag_t OIw8o4i = zendnn_OIw8o4i;
const format_tag_t Owi16o = zendnn_Owi16o;
const format_tag_t OwI16o2i = zendnn_OwI16o2i;
const format_tag_t OwI16o4i = zendnn_OwI16o4i;
const format_tag_t Owi4o = zendnn_Owi4o;
const format_tag_t Owi8o = zendnn_Owi8o;
const format_tag_t IOdhw16o16i = zendnn_IOdhw16o16i;
const format_tag_t IOhw16o16i = zendnn_IOhw16o16i;
const format_tag_t Ohwi16o = zendnn_Ohwi16o;
const format_tag_t OhwI16o2i = zendnn_OhwI16o2i;
const format_tag_t OhwI16o4i = zendnn_OhwI16o4i;
const format_tag_t Ohwi4o = zendnn_Ohwi4o;
const format_tag_t Ohwi8o = zendnn_Ohwi8o;
const format_tag_t OIhw16i16o = zendnn_OIhw16i16o;
const format_tag_t OIhw16i32o = zendnn_OIhw16i32o;
const format_tag_t OIhw16i64o = zendnn_OIhw16i64o;
const format_tag_t OIhw16o16i = zendnn_OIhw16o16i;
const format_tag_t Oihw16o = zendnn_Oihw16o;
const format_tag_t OIhw4i16o4i = zendnn_OIhw4i16o4i;
const format_tag_t OIhw4i32o4i = zendnn_OIhw4i32o4i;
const format_tag_t OIhw4i64o4i = zendnn_OIhw4i64o4i;
const format_tag_t OIhw16i16o4i = zendnn_OIhw16i16o4i;
const format_tag_t OIhw16i32o4i = zendnn_OIhw16i32o4i;
const format_tag_t OIhw16i48o4i = zendnn_OIhw16i48o4i;
const format_tag_t OIhw16i64o4i = zendnn_OIhw16i64o4i;
const format_tag_t OIhw16i16o2i = zendnn_OIhw16i16o2i;
const format_tag_t OIhw16i32o2i = zendnn_OIhw16i32o2i;
const format_tag_t OIhw16i48o2i = zendnn_OIhw16i48o2i;
const format_tag_t OIhw16i64o2i = zendnn_OIhw16i64o2i;
const format_tag_t OIhw16o16i2o = zendnn_OIhw16o16i2o;
const format_tag_t OIhw4i4o = zendnn_OIhw4i4o;
const format_tag_t OIhw4o4i = zendnn_OIhw4o4i;
const format_tag_t Oihw4o = zendnn_Oihw4o;
const format_tag_t OIhw8i16o2i = zendnn_OIhw8i16o2i;
const format_tag_t OIhw8i32o2i = zendnn_OIhw8i32o2i;
const format_tag_t OIhw8i64o2i = zendnn_OIhw8i64o2i;
const format_tag_t OIhw2i8o4i = zendnn_OIhw2i8o4i;
const format_tag_t OIhw8i8o = zendnn_OIhw8i8o;
const format_tag_t OIhw8o16i2o = zendnn_OIhw8o16i2o;
const format_tag_t IOhw8o16i2o = zendnn_IOhw8o16i2o;
const format_tag_t OIhw8o8i = zendnn_OIhw8o8i;
const format_tag_t OIhw8o4i = zendnn_OIhw8o4i;
const format_tag_t Owhi16o = zendnn_Owhi16o;
const format_tag_t Odhwi16o = zendnn_Odhwi16o;
const format_tag_t OdhwI16o2i = zendnn_OdhwI16o2i;
const format_tag_t OdhwI16o4i = zendnn_OdhwI16o4i;
const format_tag_t Odhwi4o = zendnn_Odhwi4o;
const format_tag_t Odhwi8o = zendnn_Odhwi8o;
const format_tag_t OIdhw16i16o = zendnn_OIdhw16i16o;
const format_tag_t OIdhw16i32o = zendnn_OIdhw16i32o;
const format_tag_t OIdhw16i64o = zendnn_OIdhw16i64o;
const format_tag_t OIdhw16o16i = zendnn_OIdhw16o16i;
const format_tag_t Oidhw16o = zendnn_Oidhw16o;
const format_tag_t OIdhw4i4o = zendnn_OIdhw4i4o;
const format_tag_t OIdhw4o4i = zendnn_OIdhw4o4i;
const format_tag_t Oidhw4o = zendnn_Oidhw4o;
const format_tag_t OIdhw8i16o2i = zendnn_OIdhw8i16o2i;
const format_tag_t OIdhw8i32o2i = zendnn_OIdhw8i32o2i;
const format_tag_t OIdhw8i64o2i = zendnn_OIdhw8i64o2i;
const format_tag_t OIdhw4i16o4i = zendnn_OIdhw4i16o4i;
const format_tag_t OIdhw4i32o4i = zendnn_OIdhw4i32o4i;
const format_tag_t OIdhw4i64o4i = zendnn_OIdhw4i64o4i;
const format_tag_t OIdhw16i16o4i = zendnn_OIdhw16i16o4i;
const format_tag_t OIdhw16i32o4i = zendnn_OIdhw16i32o4i;
const format_tag_t OIdhw16i48o4i = zendnn_OIdhw16i48o4i;
const format_tag_t OIdhw16i64o4i = zendnn_OIdhw16i64o4i;
const format_tag_t OIdhw16i16o2i = zendnn_OIdhw16i16o2i;
const format_tag_t OIdhw16i32o2i = zendnn_OIdhw16i32o2i;
const format_tag_t OIdhw16i48o2i = zendnn_OIdhw16i48o2i;
const format_tag_t OIdhw16i64o2i = zendnn_OIdhw16i64o2i;
const format_tag_t OIdhw2i8o4i = zendnn_OIdhw2i8o4i;
const format_tag_t OIdhw8o16i2o = zendnn_OIdhw8o16i2o;
const format_tag_t IOdhw8o16i2o = zendnn_IOdhw8o16i2o;
const format_tag_t OIdhw8i8o = zendnn_OIdhw8i8o;
const format_tag_t OIdhw8o8i = zendnn_OIdhw8o8i;
const format_tag_t OIdhw8o4i = zendnn_OIdhw8o4i;
const format_tag_t gIOw16o16i = zendnn_gIOw16o16i;
const format_tag_t Goiw16g = zendnn_Goiw16g;
const format_tag_t Goiw8g = zendnn_Goiw8g;
const format_tag_t Goiw4g = zendnn_Goiw4g;
const format_tag_t gOIw16i16o = zendnn_gOIw16i16o;
const format_tag_t gOIw16o16i = zendnn_gOIw16o16i;
const format_tag_t gOiw16o = zendnn_gOiw16o;
const format_tag_t gOIw4i16o4i = zendnn_gOIw4i16o4i;
const format_tag_t gOIw2i8o4i = zendnn_gOIw2i8o4i;
const format_tag_t gOIw16i16o4i = zendnn_gOIw16i16o4i;
const format_tag_t gOIw16i16o2i = zendnn_gOIw16i16o2i;
const format_tag_t gOIw16o16i2o = zendnn_gOIw16o16i2o;
const format_tag_t gOIw4i4o = zendnn_gOIw4i4o;
const format_tag_t gOIw4o4i = zendnn_gOIw4o4i;
const format_tag_t gOiw4o = zendnn_gOiw4o;
const format_tag_t gOIw8i16o2i = zendnn_gOIw8i16o2i;
const format_tag_t gOIw8i8o = zendnn_gOIw8i8o;
const format_tag_t gOIw8o16i2o = zendnn_gOIw8o16i2o;
const format_tag_t gIOw8o16i2o = zendnn_gIOw8o16i2o;
const format_tag_t gOIw8o8i = zendnn_gOIw8o8i;
const format_tag_t gOIw8o4i = zendnn_gOIw8o4i;
const format_tag_t gOwi16o = zendnn_gOwi16o;
const format_tag_t gOwI16o2i = zendnn_gOwI16o2i;
const format_tag_t gOwI16o4i = zendnn_gOwI16o4i;
const format_tag_t gOwi4o = zendnn_gOwi4o;
const format_tag_t gOwi8o = zendnn_gOwi8o;
const format_tag_t gIOdhw16o16i = zendnn_gIOdhw16o16i;
const format_tag_t gIOhw16o16i = zendnn_gIOhw16o16i;
const format_tag_t gOhwi16o = zendnn_gOhwi16o;
const format_tag_t gOhwI16o2i = zendnn_gOhwI16o2i;
const format_tag_t gOhwI16o4i = zendnn_gOhwI16o4i;
const format_tag_t gOhwi4o = zendnn_gOhwi4o;
const format_tag_t gOhwi8o = zendnn_gOhwi8o;
const format_tag_t Goihw16g = zendnn_Goihw16g;
const format_tag_t gOIhw16i16o = zendnn_gOIhw16i16o;
const format_tag_t gOIhw16o16i = zendnn_gOIhw16o16i;
const format_tag_t gOihw16o = zendnn_gOihw16o;
const format_tag_t gOIhw2i8o4i = zendnn_gOIhw2i8o4i;
const format_tag_t gOIhw4i16o4i = zendnn_gOIhw4i16o4i;
const format_tag_t gOIhw16i16o4i = zendnn_gOIhw16i16o4i;
const format_tag_t gOIhw16i16o2i = zendnn_gOIhw16i16o2i;
const format_tag_t gOIhw16o16i2o = zendnn_gOIhw16o16i2o;
const format_tag_t gOIhw4i4o = zendnn_gOIhw4i4o;
const format_tag_t gOIhw4o4i = zendnn_gOIhw4o4i;
const format_tag_t gOihw4o = zendnn_gOihw4o;
const format_tag_t Goihw8g = zendnn_Goihw8g;
const format_tag_t Goihw4g = zendnn_Goihw4g;
const format_tag_t gOIhw8i16o2i = zendnn_gOIhw8i16o2i;
const format_tag_t gOIhw8i8o = zendnn_gOIhw8i8o;
const format_tag_t gOIhw8o16i2o = zendnn_gOIhw8o16i2o;
const format_tag_t OIw4o8i8o4i = zendnn_OIw4o8i8o4i;
const format_tag_t gIOhw8o16i2o = zendnn_gIOhw8o16i2o;
const format_tag_t OIhw4o8i8o4i = zendnn_OIhw4o8i8o4i;
const format_tag_t OIdhw4o8i8o4i = zendnn_OIdhw4o8i8o4i;
const format_tag_t IOw4i8o8i4o = zendnn_IOw4i8o8i4o;
const format_tag_t IOhw4i8o8i4o = zendnn_IOhw4i8o8i4o;
const format_tag_t IOdhw4i8o8i4o = zendnn_IOdhw4i8o8i4o;
const format_tag_t gIOw4i8o8i4o = zendnn_gIOw4i8o8i4o;
const format_tag_t gIOhw4i8o8i4o = zendnn_gIOhw4i8o8i4o;
const format_tag_t gIOdhw4i8o8i4o = zendnn_gIOdhw4i8o8i4o;
const format_tag_t OIhw2o8i8o2i = zendnn_OIhw2o8i8o2i;
const format_tag_t gOIw4o8i8o4i = zendnn_gOIw4o8i8o4i;
const format_tag_t gOIhw4o8i8o4i = zendnn_gOIhw4o8i8o4i;
const format_tag_t gOIdhw4o8i8o4i = zendnn_gOIdhw4o8i8o4i;
const format_tag_t gOIhw2o8i8o2i = zendnn_gOIhw2o8i8o2i;
const format_tag_t gOIhw8o8i = zendnn_gOIhw8o8i;
const format_tag_t gOIhw8o4i = zendnn_gOIhw8o4i;
const format_tag_t gOwhi16o = zendnn_gOwhi16o;
const format_tag_t gIOdhw16i16o = zendnn_gIOdhw16i16o;
const format_tag_t gOdhwi16o = zendnn_gOdhwi16o;
const format_tag_t gOdhwI16o2i = zendnn_gOdhwI16o2i;
const format_tag_t gOdhwI16o4i = zendnn_gOdhwI16o4i;
const format_tag_t gOdhwi4o = zendnn_gOdhwi4o;
const format_tag_t gOdhwi8o = zendnn_gOdhwi8o;
const format_tag_t gOIdhw16i16o = zendnn_gOIdhw16i16o;
const format_tag_t gOIdhw16o16i = zendnn_gOIdhw16o16i;
const format_tag_t gOidhw16o = zendnn_gOidhw16o;
const format_tag_t gOIdhw4i4o = zendnn_gOIdhw4i4o;
const format_tag_t gOIdhw4o4i = zendnn_gOIdhw4o4i;
const format_tag_t gOidhw4o = zendnn_gOidhw4o;
const format_tag_t gOIdhw8i16o2i = zendnn_gOIdhw8i16o2i;
const format_tag_t gOIdhw4i16o4i = zendnn_gOIdhw4i16o4i;
const format_tag_t gOIdhw16i16o4i = zendnn_gOIdhw16i16o4i;
const format_tag_t gOIdhw2i8o4i = zendnn_gOIdhw2i8o4i;
const format_tag_t gOIdhw16i16o2i = zendnn_gOIdhw16i16o2i;
const format_tag_t gOIdhw8o16i2o = zendnn_gOIdhw8o16i2o;
const format_tag_t gIOdhw8o16i2o = zendnn_gIOdhw8o16i2o;
const format_tag_t gOIdhw8i8o = zendnn_gOIdhw8i8o;
const format_tag_t gOIdhw8o8i = zendnn_gOIdhw8o8i;
const format_tag_t gOIdhw8o4i = zendnn_gOIdhw8o4i;
const format_tag_t Goiw32g = zendnn_Goiw32g;
const format_tag_t Goihw32g = zendnn_Goihw32g;
const format_tag_t Goidhw32g = zendnn_Goidhw32g;
const format_tag_t gOIw2i4o2i = zendnn_gOIw2i4o2i;
const format_tag_t gOIhw2i4o2i = zendnn_gOIhw2i4o2i;
const format_tag_t gOIdhw2i4o2i = zendnn_gOIdhw2i4o2i;
const format_tag_t gOIw2o4i2o = zendnn_gOIw2o4i2o;
const format_tag_t gOIhw2o4i2o = zendnn_gOIhw2o4i2o;
const format_tag_t gOIdhw2o4i2o = zendnn_gOIdhw2o4i2o;
const format_tag_t gOIw4i8o2i = zendnn_gOIw4i8o2i;
const format_tag_t gOIhw4i8o2i = zendnn_gOIhw4i8o2i;
const format_tag_t gOIdhw4i8o2i = zendnn_gOIdhw4i8o2i;
const format_tag_t gOIw4o8i2o = zendnn_gOIw4o8i2o;
const format_tag_t gOIhw4o8i2o = zendnn_gOIhw4o8i2o;
const format_tag_t gOIdhw4o8i2o = zendnn_gOIdhw4o8i2o;
const format_tag_t ldOi32o = zendnn_ldOi32o;
const format_tag_t ldOI32o4i = zendnn_ldOI32o4i;
const format_tag_t ldgOi32o = zendnn_ldgOi32o;
const format_tag_t ldgOI32o2i = zendnn_ldgOI32o2i;
const format_tag_t ldgOI32o4i = zendnn_ldgOI32o4i;

const format_tag_t wIo2i = zendnn_wIo2i;
const format_tag_t wIo4i = zendnn_wIo4i;
const format_tag_t gwio = zendnn_gwio;
const format_tag_t gwIo2i = zendnn_gwIo2i;
const format_tag_t gwIo4i = zendnn_gwIo4i;
const format_tag_t hwIo2i = zendnn_hwIo2i;
const format_tag_t hwIo4i = zendnn_hwIo4i;
const format_tag_t ghwio = zendnn_ghwio;
const format_tag_t ghwIo2i = zendnn_ghwIo2i;
const format_tag_t ghwIo4i = zendnn_ghwIo4i;
const format_tag_t dhwIo2i = zendnn_dhwIo2i;
const format_tag_t dhwIo4i = zendnn_dhwIo4i;
const format_tag_t gdhwio = zendnn_gdhwio;
const format_tag_t gdhwIo2i = zendnn_gdhwIo2i;
const format_tag_t gdhwIo4i = zendnn_gdhwIo4i;
const format_tag_t Owi32o = zendnn_Owi32o;
const format_tag_t OwI32o2i = zendnn_OwI32o2i;
const format_tag_t OwI32o4i = zendnn_OwI32o4i;
const format_tag_t Owi48o = zendnn_Owi48o;
const format_tag_t OwI48o2i = zendnn_OwI48o2i;
const format_tag_t OwI48o4i = zendnn_OwI48o4i;
const format_tag_t Owi64o = zendnn_Owi64o;
const format_tag_t OwI64o2i = zendnn_OwI64o2i;
const format_tag_t OwI64o4i = zendnn_OwI64o4i;
const format_tag_t OhwI32o2i = zendnn_OhwI32o2i;
const format_tag_t OhwI32o4i = zendnn_OhwI32o4i;
const format_tag_t Ohwi48o = zendnn_Ohwi48o;
const format_tag_t OhwI48o2i = zendnn_OhwI48o2i;
const format_tag_t OhwI48o4i = zendnn_OhwI48o4i;
const format_tag_t Ohwi64o = zendnn_Ohwi64o;
const format_tag_t OhwI64o2i = zendnn_OhwI64o2i;
const format_tag_t OhwI64o4i = zendnn_OhwI64o4i;
const format_tag_t Odhwi32o = zendnn_Odhwi32o;
const format_tag_t OdhwI32o2i = zendnn_OdhwI32o2i;
const format_tag_t OdhwI32o4i = zendnn_OdhwI32o4i;
const format_tag_t Odhwi48o = zendnn_Odhwi48o;
const format_tag_t OdhwI48o2i = zendnn_OdhwI48o2i;
const format_tag_t OdhwI48o4i = zendnn_OdhwI48o4i;
const format_tag_t Odhwi64o = zendnn_Odhwi64o;
const format_tag_t OdhwI64o2i = zendnn_OdhwI64o2i;
const format_tag_t OdhwI64o4i = zendnn_OdhwI64o4i;
const format_tag_t gOwi32o = zendnn_gOwi32o;
const format_tag_t gOwI32o2i = zendnn_gOwI32o2i;
const format_tag_t gOwI32o4i = zendnn_gOwI32o4i;
const format_tag_t gOwi48o = zendnn_gOwi48o;
const format_tag_t gOwI48o2i = zendnn_gOwI48o2i;
const format_tag_t gOwI48o4i = zendnn_gOwI48o4i;
const format_tag_t gOwi64o = zendnn_gOwi64o;
const format_tag_t gOwI64o2i = zendnn_gOwI64o2i;
const format_tag_t gOwI64o4i = zendnn_gOwI64o4i;
const format_tag_t gOhwI32o2i = zendnn_gOhwI32o2i;
const format_tag_t gOhwI32o4i = zendnn_gOhwI32o4i;
const format_tag_t gOhwi48o = zendnn_gOhwi48o;
const format_tag_t gOhwI48o2i = zendnn_gOhwI48o2i;
const format_tag_t gOhwI48o4i = zendnn_gOhwI48o4i;
const format_tag_t gOhwi64o = zendnn_gOhwi64o;
const format_tag_t gOhwI64o2i = zendnn_gOhwI64o2i;
const format_tag_t gOhwI64o4i = zendnn_gOhwI64o4i;
const format_tag_t gOdhwi32o = zendnn_gOdhwi32o;
const format_tag_t gOdhwI32o2i = zendnn_gOdhwI32o2i;
const format_tag_t gOdhwI32o4i = zendnn_gOdhwI32o4i;
const format_tag_t gOdhwi48o = zendnn_gOdhwi48o;
const format_tag_t gOdhwI48o2i = zendnn_gOdhwI48o2i;
const format_tag_t gOdhwI48o4i = zendnn_gOdhwI48o4i;
const format_tag_t gOdhwi64o = zendnn_gOdhwi64o;
const format_tag_t gOdhwI64o2i = zendnn_gOdhwI64o2i;
const format_tag_t gOdhwI64o4i = zendnn_gOdhwI64o4i;
const format_tag_t hwcn = zendnn_hwcn;
} // namespace format_tag

using memory_extra_flags_t = zendnn_memory_extra_flags_t;
namespace memory_extra_flags {
const memory_extra_flags_t none = zendnn_memory_extra_flag_none;
const memory_extra_flags_t compensation_conv_s8s8
        = zendnn_memory_extra_flag_compensation_conv_s8s8;
const memory_extra_flags_t scale_adjust = zendnn_memory_extra_flag_scale_adjust;
const memory_extra_flags_t rnn_u8s8_compensation
        = zendnn_memory_extra_flag_rnn_u8s8_compensation;
const memory_extra_flags_t compensation_conv_asymmetric_src
        = zendnn_memory_extra_flag_compensation_conv_asymmetric_src;
} // namespace memory_extra_flags

using engine_kind_t = zendnn_engine_kind_t;
namespace engine_kind {
const engine_kind_t any_engine = zendnn_any_engine;
const engine_kind_t cpu = zendnn_cpu;
const engine_kind_t gpu = zendnn_gpu;
} // namespace engine_kind

enum runtime_kind_t {
    zendnn_runtime_none,
    zendnn_runtime_seq,
    zendnn_runtime_omp,
    zendnn_runtime_tbb,
    zendnn_runtime_threadpool,
    zendnn_runtime_ocl,
    zendnn_runtime_sycl,
};

namespace runtime_kind {
const runtime_kind_t none = zendnn_runtime_none;
const runtime_kind_t seq = zendnn_runtime_seq;
const runtime_kind_t omp = zendnn_runtime_omp;
const runtime_kind_t tbb = zendnn_runtime_tbb;
const runtime_kind_t threadpool = zendnn_runtime_threadpool;
const runtime_kind_t ocl = zendnn_runtime_ocl;
const runtime_kind_t sycl = zendnn_runtime_sycl;
} // namespace runtime_kind

using primitive_kind_t = zendnn_primitive_kind_t;
namespace primitive_kind {
const primitive_kind_t undefined = zendnn_undefined_primitive;
const primitive_kind_t reorder = zendnn_reorder;
const primitive_kind_t concat = zendnn_concat;
const primitive_kind_t sum = zendnn_sum;
const primitive_kind_t convolution = zendnn_convolution;
const primitive_kind_t deconvolution = zendnn_deconvolution;
const primitive_kind_t shuffle = zendnn_shuffle;
const primitive_kind_t eltwise = zendnn_eltwise;
const primitive_kind_t softmax = zendnn_softmax;
const primitive_kind_t pooling = zendnn_pooling;
const primitive_kind_t pooling_v2 = zendnn_pooling_v2;
const primitive_kind_t prelu = zendnn_prelu;
const primitive_kind_t lrn = zendnn_lrn;
const primitive_kind_t batch_normalization = zendnn_batch_normalization;
const primitive_kind_t layer_normalization = zendnn_layer_normalization;
const primitive_kind_t inner_product = zendnn_inner_product;
const primitive_kind_t rnn = zendnn_rnn;
const primitive_kind_t gemm = zendnn_gemm;
const primitive_kind_t binary = zendnn_binary;
const primitive_kind_t logsoftmax = zendnn_logsoftmax;
const primitive_kind_t matmul = zendnn_matmul;
const primitive_kind_t resampling = zendnn_resampling;
const primitive_kind_t reduction = zendnn_reduction;
/* add new primitive */
const primitive_kind_t embedding_bag = zendnn_embedding_bag;

// Internal only primitive kinds.
const primitive_kind_t internal_only_start = (primitive_kind_t)(1 << 12);
const primitive_kind_t zero_pad = internal_only_start;
} // namespace primitive_kind

using query_t = zendnn_query_t;
namespace query {
const query_t undef = zendnn_query_undef;

const query_t engine = zendnn_query_engine;
const query_t primitive_kind = zendnn_query_primitive_kind;

const query_t num_of_inputs_s32 = zendnn_query_num_of_inputs_s32;
const query_t num_of_outputs_s32 = zendnn_query_num_of_outputs_s32;

const query_t time_estimate_f64 = zendnn_query_time_estimate_f64;
const query_t memory_consumption_s64 = zendnn_query_memory_consumption_s64;

const query_t scratchpad_engine = zendnn_query_scratchpad_engine;

const query_t impl_info_str = zendnn_query_impl_info_str;

const query_t reorder_src_engine = zendnn_query_reorder_src_engine;
const query_t reorder_dst_engine = zendnn_query_reorder_dst_engine;

const query_t prop_kind = zendnn_query_prop_kind;

const query_t some_d = zendnn_query_some_d;
const query_t op_d = zendnn_query_op_d;
const query_t convolution_d = zendnn_query_convolution_d;
const query_t deconvolution_d = zendnn_query_deconvolution_d;
const query_t shuffle_d = zendnn_query_shuffle_d;
const query_t eltwise_d = zendnn_query_eltwise_d;
const query_t softmax_d = zendnn_query_softmax_d;
const query_t pooling_d = zendnn_query_pooling_d;
const query_t pooling_v2_d = zendnn_query_pooling_v2_d;
const query_t prelu_d = zendnn_query_prelu_d;
const query_t lrn_d = zendnn_query_lrn_d;
const query_t batch_normalization_d = zendnn_query_batch_normalization_d;
const query_t layer_normalization_d = zendnn_query_layer_normalization_d;
const query_t inner_product_d = zendnn_query_inner_product_d;
const query_t rnn_d = zendnn_query_rnn_d;
const query_t gemm_d = zendnn_query_gemm_d;
const query_t binary_d = zendnn_query_binary_d;
const query_t logsoftmax_d = zendnn_query_logsoftmax_d;
const query_t matmul_d = zendnn_query_matmul_d;
const query_t resampling_d = zendnn_query_resampling_d;
const query_t reduction_d = zendnn_query_reduction_d;
/* add new primitive */
const query_t embedding_bag_d = zendnn_query_embedding_bag_d;
const query_t some_md = zendnn_query_some_md;
const query_t src_md = zendnn_query_src_md;
const query_t diff_src_md = zendnn_query_diff_src_md;
const query_t weights_md = zendnn_query_weights_md;
const query_t diff_weights_md = zendnn_query_diff_weights_md;
const query_t dst_md = zendnn_query_dst_md;
const query_t diff_dst_md = zendnn_query_diff_dst_md;
const query_t exec_arg_md = zendnn_query_exec_arg_md;

const query_t workspace_md = zendnn_query_workspace_md;
const query_t scratchpad_md = zendnn_query_scratchpad_md;

// Internal only query kinds.
const query_t internal_only_start = (query_t)(1 << 12);
const query_t zero_pad_d = internal_only_start;
} // namespace query

using blocking_desc_t = zendnn_blocking_desc_t;
using rnn_packed_desc_t = zendnn_rnn_packed_desc_t;
using wino_desc_t = zendnn_wino_desc_t;
using memory_extra_desc_t = zendnn_memory_extra_desc_t;
using memory_desc_t = zendnn_memory_desc_t;
using convolution_desc_t = zendnn_convolution_desc_t;
using deconvolution_desc_t = zendnn_deconvolution_desc_t;
using shuffle_desc_t = zendnn_shuffle_desc_t;
using pooling_desc_t = zendnn_pooling_desc_t;
using pooling_v2_desc_t = zendnn_pooling_v2_desc_t;
using prelu_desc_t = zendnn_prelu_desc_t;
using eltwise_desc_t = zendnn_eltwise_desc_t;
using softmax_desc_t = zendnn_softmax_desc_t;
using lrn_desc_t = zendnn_lrn_desc_t;
using batch_normalization_desc_t = zendnn_batch_normalization_desc_t;
using layer_normalization_desc_t = zendnn_layer_normalization_desc_t;
using inner_product_desc_t = zendnn_inner_product_desc_t;
using binary_desc_t = zendnn_binary_desc_t;
using logsoftmax_desc_t = zendnn_logsoftmax_desc_t;
using matmul_desc_t = zendnn_matmul_desc_t;
using resampling_desc_t = zendnn_resampling_desc_t;
using reduction_desc_t = zendnn_reduction_desc_t;

using rnn_direction_t = zendnn_rnn_direction_t;
using rnn_desc_t = zendnn_rnn_desc_t;

/* add new primitive */
using embedding_bag_desc_t = zendnn_embedding_bag_desc_t;

/* Internal type, declared in gemm_types.hpp */
using gemm_desc_t = zendnn_gemm_desc_t;

/* Internal types, for the primitives which don't have descs */
using concat_desc_t = zendnn_concat_desc_t;
using reorder_desc_t = zendnn_reorder_desc_t;
using sum_desc_t = zendnn_sum_desc_t;
using zero_pad_desc_t = zendnn_zero_pad_desc_t;

/* C op_desc_t, which eventually are just (void*) */
using c_op_desc_t = zendnn_op_desc_t;
using const_c_op_desc_t = const_zendnn_op_desc_t;

struct op_desc_t {
    union {
        primitive_kind_t kind;
        convolution_desc_t convolution;
        deconvolution_desc_t deconvolution;
        shuffle_desc_t shuffle;
        pooling_desc_t pooling;
        pooling_v2_desc_t pooling_v2;
        prelu_desc_t prelu;
        eltwise_desc_t eltwise;
        softmax_desc_t softmax;
        lrn_desc_t lrn;
        batch_normalization_desc_t batch_normalization;
        layer_normalization_desc_t layer_normalization;
        inner_product_desc_t inner_product;
        rnn_desc_t rnn;
        gemm_desc_t gemm;
        concat_desc_t concat;
        reorder_desc_t reorder;
        sum_desc_t sum;
        binary_desc_t binary;
        matmul_desc_t matmul;
        resampling_desc_t resampling;
        zero_pad_desc_t zero_pad;
        reduction_desc_t reduction;
        /* add new primitive */
        embedding_bag_desc_t embedding_bag;
    };

#define DECL_CTOR_AND_CONVERTERS(c_type) \
    op_desc_t(const c_type &) = delete; \
    static op_desc_t *convert_from_c(c_type *_) { \
        return reinterpret_cast<op_desc_t *>(_); \
    } \
    static const op_desc_t *convert_from_c(const c_type *_) { \
        return reinterpret_cast<const op_desc_t *>(_); \
    }

    DECL_CTOR_AND_CONVERTERS(convolution_desc_t);
    DECL_CTOR_AND_CONVERTERS(shuffle_desc_t);
    DECL_CTOR_AND_CONVERTERS(pooling_desc_t);
    DECL_CTOR_AND_CONVERTERS(pooling_v2_desc_t);
    DECL_CTOR_AND_CONVERTERS(prelu_desc_t);
    DECL_CTOR_AND_CONVERTERS(eltwise_desc_t);
    DECL_CTOR_AND_CONVERTERS(softmax_desc_t);
    DECL_CTOR_AND_CONVERTERS(lrn_desc_t);
    DECL_CTOR_AND_CONVERTERS(batch_normalization_desc_t);
    DECL_CTOR_AND_CONVERTERS(layer_normalization_desc_t);
    DECL_CTOR_AND_CONVERTERS(inner_product_desc_t);
    DECL_CTOR_AND_CONVERTERS(rnn_desc_t);
    DECL_CTOR_AND_CONVERTERS(gemm_desc_t);
    DECL_CTOR_AND_CONVERTERS(concat_desc_t);
    DECL_CTOR_AND_CONVERTERS(reorder_desc_t);
    DECL_CTOR_AND_CONVERTERS(sum_desc_t);
    DECL_CTOR_AND_CONVERTERS(binary_desc_t);
    DECL_CTOR_AND_CONVERTERS(matmul_desc_t);
    DECL_CTOR_AND_CONVERTERS(resampling_desc_t);
    DECL_CTOR_AND_CONVERTERS(zero_pad_desc_t);
    DECL_CTOR_AND_CONVERTERS(reduction_desc_t);

    // concat_desc_t and sum_desc_t have data members which have non-trivial
    // special member functions hence the default destructor is implicitly
    // deleted by the compiler which causes a warning on Windows so we should
    // delete the destructor explicitly.
    ~op_desc_t() = delete;

#undef DECL_CTOR_AND_CONVERTERS
};

using engine_t = zendnn_engine;
using primitive_desc_iterator_t = zendnn_primitive_desc_iterator;
using primitive_attr_t = zendnn_primitive_attr;
using post_ops_t = zendnn_post_ops;
using memory_t = zendnn_memory;

using stream_flags_t = zendnn_stream_flags_t;
namespace stream_flags {
const stream_flags_t in_order = zendnn_stream_in_order;
const stream_flags_t out_of_order = zendnn_stream_out_of_order;
const stream_flags_t default_flags = zendnn_stream_default_flags;
} // namespace stream_flags
using stream_t = zendnn_stream;

struct memory_storage_t;

/* forward declaration of the internal primitive_desc types */
struct batch_normalization_bwd_pd_t;
struct batch_normalization_fwd_pd_t;
struct batch_normalization_pd_t;
struct binary_pd_t;
struct concat_pd_t;
struct convolution_bwd_data_pd_t;
struct convolution_bwd_weights_pd_t;
struct convolution_fwd_pd_t;
struct convolution_pd_t;
struct deconvolution_bwd_data_pd_t;
struct deconvolution_bwd_weights_pd_t;
struct deconvolution_fwd_pd_t;
struct deconvolution_pd_t;
struct eltwise_bwd_pd_t;
struct eltwise_fwd_pd_t;
struct eltwise_pd_t;
struct gemm_pd_t;
struct inner_product_bwd_data_pd_t;
struct inner_product_bwd_weights_pd_t;
struct inner_product_fwd_pd_t;
struct inner_product_pd_t;
struct layer_normalization_bwd_pd_t;
struct layer_normalization_fwd_pd_t;
struct layer_normalization_pd_t;
struct lrn_bwd_pd_t;
struct lrn_fwd_pd_t;
struct lrn_pd_t;
struct matmul_pd_t;
struct pooling_bwd_pd_t;
struct pooling_fwd_pd_t;
struct pooling_pd_t;
struct prelu_pd_t;
struct reduction_pd_t;
struct reorder_pd_t;
struct resampling_pd_t;
struct rnn_bwd_pd_t;
struct rnn_fwd_pd_t;
struct rnn_pd_t;
struct shuffle_pd_t;
struct softmax_bwd_pd_t;
struct softmax_fwd_pd_t;
struct softmax_pd_t;
struct sum_pd_t;

} // namespace impl
} // namespace zendnn

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
