/********************************************************************************
# * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <type_traits>

#include "sdpa_encoder_ref_kernel.hpp"
#include "sdpa_encoder_kernel_helpers.hpp"

namespace zendnnl {
namespace ops {
using namespace zendnnl::error_handling;
using zendnnl::common::bfloat16_t;
using zendnnl::common::float16_t;

namespace {

// Per-(qkv_t) implementation of the reference SDPA encoder.
//
// All shared scaffolding (tensor / stride extraction, mask layout resolution,
// per-(b, h) parallel loop) lives here; the public execute() dispatches on
// the runtime QKV dtype to instantiate the correct typed entry.
//
// The lambda parameterises the inner template instantiation on the mask
// element type so the same body services both FP32 and BF16 masks. The
// dispatch cost is paid once per execute() call rather than per (b, h)
// iteration -- the parallel region is fully monomorphic.
template<typename qkv_t>
status_t execute_typed(
  const sdpa_encoder_context_t &context_,
  op_kernel_t<sdpa_encoder_context_t>::tensor_map_type &outputs_) {

  // Read input tensors from context
  auto query_opt = context_.get_param("query");
  auto key_opt = context_.get_param("key");
  auto value_opt = context_.get_param("value");
  auto mask_opt = context_.get_param("mask");

  if (!query_opt || !key_opt || !value_opt) {
    return status_t::failure;
  }

  auto query = query_opt.value();
  auto key = key_opt.value();
  auto value = value_opt.value();
  bool has_mask = context_.get_has_mask();
  // Guard against an inconsistent context: has_mask=true must be accompanied
  // by a "mask" tensor param.
  if (has_mask && !mask_opt) {
    apilog_error("SDPA ref kernel: has_mask=true but no \"mask\" tensor "
                 "parameter is present in the context");
    return status_t::failure;
  }
  // Mask buffer dtype is decided per-call (the operator's validate()
  // restricts the supported combinations):
  //   - FP32 QKV  -> mask must be FP32
  //   - BF16 QKV  -> mask may be FP32 or BF16
  //   - F16  QKV  -> mask may be FP32 or F16
  // Softmax always runs in FP32, so each mask element is converted to float
  // at add time inside apply_attention_mask<mask_t>.
  void *mask_base_void = nullptr;
  data_type_t mask_dtype = data_type_t::none;
  sdpa_encoder_ref::mask_layout mask_layout{0, 0};

  // Read output tensor
  auto output = outputs_["sdpa_output"];

  float scale = context_.get_scale();

  // Q is [B, H, S_q, D], K/V are [B, H, S_kv, D]. For self-attention
  // S_q == S_kv; the operator's validate() permits S_q != S_kv to support
  // cross-attention but enforces K.S == V.S.
  int64_t batch       = static_cast<int64_t>(query.get_size(0));
  int64_t num_heads   = static_cast<int64_t>(query.get_size(1));
  int64_t seq_len_q   = static_cast<int64_t>(query.get_size(2));
  int64_t seq_len_kv  = static_cast<int64_t>(key.get_size(2));
  int64_t head_dim    = static_cast<int64_t>(query.get_size(3));
  bool is_causal      = context_.get_is_causal();

  if (batch <= 0 || num_heads <= 0 || seq_len_q <= 0 || seq_len_kv <= 0 ||
      head_dim <= 0) {
    return status_t::failure;
  }

  // Resolve mask layout: 2D [S_q, S_kv] or 4D [B|1, H|1, S_q, S_kv] are all
  // supported; stride_b / stride_h are 0 for broadcast dimensions.
  if (has_mask) {
    auto mask = mask_opt.value();
    mask_dtype = mask.get_data_type();
    mask_base_void = mask.get_raw_handle_unsafe();
    mask_layout = sdpa_encoder_ref::compute_mask_strides(
                    mask.get_size(), seq_len_q, seq_len_kv);
  }

  const qkv_t *q_data   = static_cast<const qkv_t *>
                          (query.get_raw_handle_unsafe());
  const qkv_t *k_data   = static_cast<const qkv_t *>
                          (key.get_raw_handle_unsafe());
  const qkv_t *v_data   = static_cast<const qkv_t *>
                          (value.get_raw_handle_unsafe());
  qkv_t       *out_data = static_cast<qkv_t *>
                          (output.get_raw_handle_unsafe());

  // Per-tensor BHSD strides. Tensor is logically [B, H, S, D]; the actual
  // physical layout is one of the layouts currently accepted by validate():
  // BHSD canonical contiguous or BSHD physical via PyTorch .transpose(1,2).
  auto q_str   = query.get_stride();
  auto k_str   = key.get_stride();
  auto v_str   = value.get_stride();
  auto o_str   = output.get_stride();
  const int64_t q_sb = static_cast<int64_t>(q_str[0]);
  const int64_t q_sh = static_cast<int64_t>(q_str[1]);
  const int64_t q_ss = static_cast<int64_t>(q_str[2]);
  const int64_t k_sb = static_cast<int64_t>(k_str[0]);
  const int64_t k_sh = static_cast<int64_t>(k_str[1]);
  const int64_t k_ss = static_cast<int64_t>(k_str[2]);
  const int64_t v_sb = static_cast<int64_t>(v_str[0]);
  const int64_t v_sh = static_cast<int64_t>(v_str[1]);
  const int64_t v_ss = static_cast<int64_t>(v_str[2]);
  const int64_t o_sb = static_cast<int64_t>(o_str[0]);
  const int64_t o_sh = static_cast<int64_t>(o_str[1]);
  const int64_t o_ss = static_cast<int64_t>(o_str[2]);

  // Per-(b, h) work loop, parameterised on the mask element type so the same
  // body services both FP32 and BF16 masks. The inner kernel template is
  // selected once per execute() call rather than per (b, h) iteration, so the
  // dispatch cost is paid outside the parallel region.
  auto run_per_head_loop = [&](auto mask_tag) -> status_t {
    using mask_t = decltype(mask_tag);
    const mask_t *mask_base = static_cast<const mask_t *>(mask_base_void);
    // Per (b, h) base pointers come from each tensor's own (batch, head)
    // strides; the inner [seq, head_dim] slab is then walked using each
    // tensor's seq stride (head_dim for BHSD, num_heads*head_dim for BSHD).
    // Per (b, h): scores [S_q, S_kv] (FP32) -> output [S_q, D] (qkv_t).
    #pragma omp parallel for collapse(2)
    for (int64_t b = 0; b < batch; b++) {
      for (int64_t h = 0; h < num_heads; h++) {
        const qkv_t *q_bh = q_data   + b * q_sb + h * q_sh;
        const qkv_t *k_bh = k_data   + b * k_sb + h * k_sh;
        const qkv_t *v_bh = v_data   + b * v_sb + h * v_sh;
        qkv_t       *o_bh = out_data + b * o_sb + h * o_sh;
        const mask_t *mask_for_bh = has_mask
        ? mask_base + b * mask_layout.stride_b
        + h * mask_layout.stride_h
        : nullptr;
        sdpa_encoder_ref::compute_sdpa_per_head<qkv_t, mask_t>(
          q_bh, k_bh, v_bh, o_bh, mask_for_bh,
          seq_len_q, seq_len_kv, head_dim,
          q_ss, k_ss, v_ss, o_ss,
          scale, is_causal, has_mask);
      }
    }
    return status_t::success;
  };

  // No-mask and FP32-mask paths share the mask_t = float instantiation
  // (mask_for_bh is nullptr in the no-mask case, so mask_t is never read).
  if (!has_mask || mask_dtype == data_type_t::f32) {
    return run_per_head_loop(float{});
  }

  // Reduced-precision mask is only valid when the mask dtype matches the
  // QKV dtype (enforced by validate()): bf16 mask with bf16 QKV, f16 mask
  // with f16 QKV. The `if constexpr` blocks keep each reduced-precision
  // mask instantiation out of the QKV specialisations that don't accept it,
  // so the compiler can prove the FP32 specialisation never reaches the
  // unsupported-mask error path with a reduced-precision mask.
  if constexpr(std::is_same_v<qkv_t, bfloat16_t>) {
    if (mask_dtype == data_type_t::bf16) {
      return run_per_head_loop(bfloat16_t{});
    }
  }
  if constexpr(std::is_same_v<qkv_t, float16_t>) {
    if (mask_dtype == data_type_t::f16) {
      return run_per_head_loop(float16_t{});
    }
  }

  // validate() should have rejected any other mask dtype before reaching
  // execute(); fail closed if we somehow get here with an unsupported one.
  apilog_error("SDPA ref kernel: unsupported mask data_type = ",
               static_cast<int>(mask_dtype),
               " for QKV data_type = ",
               static_cast<int>(query.get_data_type()));
  return status_t::failure;
}

} // anonymous namespace

status_t sdpa_encoder_ref_kernel_t::execute(const context_type &context_,
    tensor_map_type &inputs_,
    tensor_map_type &outputs_) {
  (void)inputs_;

  // QKV dtype dispatch happens here (once per execute() call). The
  // supported-dtype list is owned by sdpa_encoder_impl_t::kernel_factory();
  // the cases below mirror it and fail closed on anything unexpected.
  auto query_opt = context_.get_param("query");
  if (!query_opt) {
    return status_t::failure;
  }
  const auto query_dtype = query_opt.value().get_data_type();

  if (query_dtype == data_type_t::f32) {
    return execute_typed<float>(context_, outputs_);
  }
  if (query_dtype == data_type_t::bf16) {
    return execute_typed<bfloat16_t>(context_, outputs_);
  }
  if (query_dtype == data_type_t::f16) {
    return execute_typed<float16_t>(context_, outputs_);
  }

  apilog_error("SDPA ref kernel: unsupported QKV data_type = ",
               static_cast<int>(query_dtype),
               " (expected f32, bf16, or f16)");
  return status_t::failure;
}

} //namespace ops
} //namespace zendnnl

extern "C" {
  zendnnl::ops::sdpa_encoder_ref_kernel_t *get_sdpa_encoder_ref_kernel() {
    return new zendnnl::ops::sdpa_encoder_ref_kernel_t();
  }
}
