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
#include "sdpa_encoder_operator_impl.hpp"
#include "sdpa_encoder_kernel_list.hpp"

namespace zendnnl {
namespace ops {

status_t sdpa_encoder_impl_t::validate() {
  LOG_DEBUG_INFO("Validating sdpa_encoder_impl_t");

  if (parent_type::validate() != status_t::success) {
    return status_t::failure;
  }

  // Validate mandatory output
  if (!get_output("sdpa_output")) {
    apilog_error("Missing mandatory sdpa_output");
    return status_t::failure;
  }

  // Validate context parameters (Q, K, V tensors)
  auto query = context.get_param("query");
  auto key = context.get_param("key");
  auto value = context.get_param("value");

  if (!query || !key || !value) {
    apilog_error("Missing mandatory Q, K, V parameters in context");
    return status_t::failure;
  }

  // Validate that the "mask" tensor param and context.has_mask() agree.
  //
  // The reference kernels (sdpa_encoder_ref_kernel_t) read the
  // additive-mask tensor only when context.get_has_mask() is true and
  // unconditionally dereference the "mask" param in that case. They have a
  // defensive runtime guard that returns status_t::failure on inconsistency,
  // but surfacing the same condition here lets callers fail fast during
  // operator execute() validation, before kernel dispatch.
  //
  // We require strict agreement in *both* directions:
  //   has_mask=true  + no  "mask" param  -> reject: execute() would fail.
  //   has_mask=false + has "mask" param  -> reject: the mask would be
  //                                         silently ignored, which is
  //                                         almost always a user error
  //                                         (forgot to call set_has_mask).
  auto mask = context.get_param("mask");
  const bool has_mask = context.get_has_mask();
  if (has_mask && !mask) {
    apilog_error("SDPA context.has_mask() is true but no \"mask\" tensor "
                 "param was set on the context");
    return status_t::failure;
  }
  if (!has_mask && mask) {
    apilog_error("SDPA \"mask\" tensor param was set but "
                 "context.has_mask() is false; the mask would be silently "
                 "ignored. Either call set_has_mask(true) or remove the "
                 "mask param from the context");
    return status_t::failure;
  }
  LOG_DEBUG_INFO(has_mask ? "Mask parameter found in context"
                 : "No mask parameter provided - "
                 "proceeding without mask");

  // Validate tensor dimensions compatibility
  auto q_size = query->get_size();
  auto k_size = key->get_size();
  auto v_size = value->get_size();

  if (q_size.size() != 4 || k_size.size() != 4 || v_size.size() != 4) {
    apilog_error("Q, K, V tensors must be 4D [B, H, S, D]");
    return status_t::failure;
  }

  // Check dimension compatibility:
  //   Q is [B, H, S_q,  head_dim]
  //   K is [B, H, S_kv, head_dim]
  //   V is [B, H, S_kv, head_dim]
  // Cross-attention is supported: S_q (q_size[2]) is allowed to differ from
  // S_kv, where the K and V sequence lengths are k_size[2] and v_size[2].
  // K and V must share the same S_kv. All other dims must match across Q/K/V.
  if (q_size[0] != k_size[0] || q_size[0] != v_size[0] ||  // Batch
      q_size[1] != k_size[1] || q_size[1] != v_size[1] ||  // Heads
      k_size[2] != v_size[2] ||                            // K & V share S_kv
      q_size[3] != k_size[3] || q_size[3] != v_size[3]) {  // head_dim
    apilog_error("Q, K, V tensor dimensions are incompatible "
                 "(expected B, H, head_dim equal across Q/K/V "
                 "and S_kv equal between K and V)");
    return status_t::failure;
  }

  // Validate output shape: the reference kernels (FP32 / BF16) use the
  // output tensor's strides and can support different physical layouts
  // (for example BHSD and BSHD), but they still index/iterate using Q's
  // logical dimensions [B, H, S_q, head_dim]. Therefore the output must be
  // 4D and match Q's per-dimension sizes so the kernel loops do not
  // silently misindex the destination or run past its allocation.
  auto output_size = get_output("sdpa_output")->get_size();
  if (output_size.size() != 4 ||
      output_size[0] != q_size[0] || output_size[1] != q_size[1] ||
      output_size[2] != q_size[2] || output_size[3] != q_size[3]) {
    apilog_error("SDPA output tensor must be 4D and match Q's "
                 "[B, H, S_q, head_dim] = [",
                 q_size[0], ", ", q_size[1], ", ", q_size[2], ", ",
                 q_size[3], "]; got rank=", output_size.size());
    return status_t::failure;
  }

  // Q, K, V, and output must all share the same data type. The kernel is
  // dispatched on Q's dtype (kernel_factory()) and unconditionally casts
  // K, V, and output buffers to that same type -- a mismatch would silently
  // misinterpret memory and could read past the allocation. We only enforce
  // the consistency property here; kernel_factory() remains the single
  // source of truth for the supported-dtype list (rejecting anything other
  // than f32 / bf16 / f16 with status_t::unimplemented).
  auto qkv_dtype = query->get_data_type();
  if (key->get_data_type() != qkv_dtype ||
      value->get_data_type() != qkv_dtype) {
    apilog_error("SDPA Q, K, V tensors must share the same dtype; got "
                 "Q=", static_cast<int>(qkv_dtype),
                 " K=", static_cast<int>(key->get_data_type()),
                 " V=", static_cast<int>(value->get_data_type()));
    return status_t::failure;
  }
  auto output_dtype = get_output("sdpa_output")->get_data_type();
  if (output_dtype != qkv_dtype) {
    apilog_error("SDPA output tensor dtype must match Q/K/V dtype (",
                 static_cast<int>(qkv_dtype), "); got output=",
                 static_cast<int>(output_dtype));
    return status_t::failure;
  }

  // Validate Q/K/V/output strides describe a supported layout for the
  // reference kernels.
  //
  // The tensors are *logically* 4D [B, H, S, D]; the *physical* layout is
  // encoded entirely in the strides, exactly as in the flash backend (and
  // as in PyTorch, where `.transpose(1, 2)` produces a logical-BHSD view of
  // BSHD memory).
  //
  // The current validator accepts exactly two full physical stride patterns
  // per tensor (independently), not arbitrary outer [B, H] strides:
  //   BHSD canonical contiguous : (s_B, s_H, s_S, s_D) =
  //                                (H*S*D, S*D,   D,   1)
  //   BSHD physical (logical BHSD via .transpose(1, 2)):
  //                                (s_B, s_H, s_S, s_D) =
  //                                (S*H*D,   D, H*D,   1)
  // Stride 0 is permitted on any size-1 dim (its index is always 0, so the
  // stride contributes nothing to the offset). Tensors may independently
  // pick BHSD or BSHD; the kernel uses each tensor's own strides.
  auto check_bhsd_or_bshd = [](const char *tensor_name,
                               const std::vector<uint64_t> &sizes,
                               const std::vector<uint64_t> &strides)
  -> status_t {
    if (sizes.size() != 4 || strides.size() != 4) {
      apilog_error("SDPA ", tensor_name,
                   " must have 4D shape and 4D strides; got sizes.size()=",
                   sizes.size(), " strides.size()=", strides.size());
      return status_t::failure;
    }
    const uint64_t B = sizes[0], H = sizes[1], S = sizes[2], D = sizes[3];
    auto stride_ok = [](uint64_t actual, uint64_t expected, bool size1) {
      // size-1 dim: index is always 0, so any stride works (matches the
      // size-1 / broadcast convention used by the mask validator).
      return size1 ? true : (actual == expected);
    };

    // BHSD: (H*S*D, S*D, D, 1)
    const bool is_bhsd =
      stride_ok(strides[0], H *S * D, B == 1) &&
      stride_ok(strides[1], S * D,     H == 1) &&
      stride_ok(strides[2], D,         S == 1) &&
      stride_ok(strides[3], 1u,        D == 1);

    // BSHD physical (logical BHSD): (S*H*D, D, H*D, 1)
    const bool is_bshd =
      stride_ok(strides[0], S *H * D, B == 1) &&
      stride_ok(strides[1], D,         H == 1) &&
      stride_ok(strides[2], H * D,     S == 1) &&
      stride_ok(strides[3], 1u,        D == 1);

    if (is_bhsd || is_bshd) {
      return status_t::success;
    }
    apilog_error("SDPA ", tensor_name, " has unsupported strides ",
                 "[", strides[0], ", ", strides[1], ", ", strides[2], ", ",
                 strides[3], "] for shape [", B, ", ", H, ", ", S, ", ", D,
                 "]; expected BHSD [", H *S * D, ", ", S * D, ", ", D,
                 ", 1] or BSHD [", S *H * D, ", ", D, ", ", H * D, ", 1]");
    return status_t::failure;
  };

  if (check_bhsd_or_bshd("query",  q_size, query->get_stride())
      != status_t::success ||
      check_bhsd_or_bshd("key",    k_size, key->get_stride())
      != status_t::success ||
      check_bhsd_or_bshd("value",  v_size, value->get_stride())
      != status_t::success ||
      check_bhsd_or_bshd("output", output_size,
                         get_output("sdpa_output")->get_stride())
      != status_t::success) {
    return status_t::failure;
  }

  // Validate mask dtype, shape, and physical layout if mask is present.
  // Supported layouts (broadcast to [B, H, S_q, S_kv] inside the kernel
  // via per-(batch, head) stride lookup):
  //   - 2D [S_q, S_kv]                broadcast across batch & heads
  //   - 4D [B|1, H|1, S_q, S_kv]      size-1 leading dims broadcast
  if (mask) {
    // Mask dtype rules (matching the LOWOHA flash backend):
    //   - FP32 QKV  -> mask must be FP32.
    //   - BF16 QKV  -> mask may be FP32 or BF16; the BF16 kernel dispatches
    //                  on the mask dtype at execute() time and converts
    //                  per-element to FP32 before adding to the score buffer.
    //   - F16  QKV  -> mask may be FP32 or F16; same dispatch pattern as the
    //                  BF16 path (per-element widening to FP32 inside
    //                  apply_attention_mask<mask_t>).
    // Both kernels reinterpret the mask buffer as the validated dtype, so a
    // mismatched dtype would silently misinterpret memory and could read
    // past the tensor's allocation.
    const auto mask_dt = mask->get_data_type();
    const bool mask_dt_ok =
      (qkv_dtype == data_type_t::f32  && mask_dt == data_type_t::f32) ||
      (qkv_dtype == data_type_t::bf16 && (mask_dt == data_type_t::f32 ||
                                          mask_dt == data_type_t::bf16)) ||
      (qkv_dtype == data_type_t::f16  && (mask_dt == data_type_t::f32 ||
                                          mask_dt == data_type_t::f16));
    if (!mask_dt_ok) {
      apilog_error("SDPA mask tensor dtype is not supported for the given "
                   "Q/K/V dtype. Allowed: f32 mask with f32 QKV, f32/bf16 "
                   "mask with bf16 QKV, or f32/f16 mask with f16 QKV. Got "
                   "QKV=", static_cast<int>(qkv_dtype),
                   " mask=", static_cast<int>(mask_dt));
      return status_t::failure;
    }

    auto mask_size = mask->get_size();
    const size_t rank = mask_size.size();

    if (rank != 2 && rank != 4) {
      apilog_error("Mask tensor must be 2D [S_q, S_kv] or 4D "
                   "[B|1, H|1, S_q, S_kv]");
      return status_t::failure;
    }

    // Inner [S_q, S_kv] dims must match Q/K's seq lens exactly in both
    // 2D and 4D layouts.
    if (mask_size[rank - 2] != q_size[2] || mask_size[rank - 1] != k_size[2]) {
      apilog_error("Mask inner dims must be [S_q, S_kv] = [",
                   q_size[2], ", ", k_size[2], "]; got [",
                   mask_size[rank - 2], ", ", mask_size[rank - 1], "]");
      return status_t::failure;
    }

    // 4D-only: each leading B/H dim must be 1 (broadcast) or match Q.
    if (rank == 4 &&
        ((mask_size[0] != 1 && mask_size[0] != q_size[0]) ||
         (mask_size[1] != 1 && mask_size[1] != q_size[1]))) {
      apilog_error("4D mask leading dims must be 1 (broadcast) or match "
                   "Q's [B, H] = [", q_size[0], ", ", q_size[1],
                   "]; got [", mask_size[0], ", ", mask_size[1], "]");
      return status_t::failure;
    }

    // Physical layout: the reference kernel indexes the inner [S_q, S_kv]
    // slab as mask_ptr[i*S_kv + j] (sdpa_encoder_kernel_helpers.hpp::
    // apply_attention_mask) and derives leading B/H strides purely from
    // the shape via compute_mask_strides -- the tensor's actual leading
    // strides are never consulted. So the mask's physical layout MUST
    // match the canonical row-major contiguous layout for its shape;
    // any transposed / padded / sliced view would be silently misindexed.
    //
    // Canonical row-major: stride[d] = product of mask_size[d+1..rank-1].
    // Stride 0 is permitted on size-1 dims by the broadcast convention.
    auto mask_strides = mask->get_stride();
    if (mask_strides.size() != rank) {
      apilog_error("SDPA mask stride/size dim mismatch: strides=",
                   mask_strides.size(), " sizes=", rank);
      return status_t::failure;
    }

    uint64_t expected = 1;
    for (size_t d = rank; d-- > 0;) {
      const bool is_broadcast = (mask_size[d] == 1);
      const bool ok = is_broadcast
                      ? (mask_strides[d] == 0u || mask_strides[d] == expected)
                      : (mask_strides[d] == expected);
      if (!ok) {
        apilog_error("SDPA mask stride at dim ", d, " must be ", expected,
                     is_broadcast ? " (or 0 for broadcast)" : "",
                     "; got ", mask_strides[d]);
        return status_t::failure;
      }
      expected *= mask_size[d];
    }
  }

  return status_t::success;
}

std::string sdpa_encoder_impl_t::op_create_info() {
  std::stringstream ss;

  ss << "SDPA Encoder operator create - ";
  if (!(get_name().empty())) {
    ss << get_name();
  }

  // Add context information
  ss << ", scale: " << context.get_scale();
  ss << ", is_dropout: " << (context.get_is_dropout() ? "true" : "false");
  ss << ", is_causal: " << (context.get_is_causal() ? "true" : "false");
  ss << ", has_mask: " << (context.get_has_mask() ? "true" : "false");

  return ss.str();
}

std::string sdpa_encoder_impl_t::op_execute_info() {
  std::stringstream ss;

  ss << "SDPA Encoder operator execute - ";
  if (!(get_name().empty())) {
    ss << get_name() << ",";
  }
  auto output = get_output("sdpa_output");

  ss << "Output: " << output.value().tensor_info();

  // Add Q, K, V tensor info
  auto query = context.get_param("query");
  auto key = context.get_param("key");
  auto value = context.get_param("value");
  auto mask = context.get_param("mask");

  if (query && key && value) {
    ss << ", Q: " << query.value().tensor_info()
       << ", K: " << key.value().tensor_info()
       << ", V: " << value.value().tensor_info();
  }

  // Add mask tensor info if present
  if (mask) {
    ss << ", Mask: " << mask.value().tensor_info();
  }

  return ss.str();
}

status_t sdpa_encoder_impl_t::kernel_factory() {
  LOG_DEBUG_INFO("Creating SDPA encoder kernel");

  auto input_dtype = context.get_param("query")->get_data_type();

  // The unified reference kernel handles every supported QKV dtype
  // internally; kernel_factory() remains the single source of truth for the
  // supported-dtype list so the caller (operator_impl_t::create()) gets the
  // canonical status_t::unimplemented for unsupported dtypes before the
  // kernel itself is exercised. The reference kernel performs all arithmetic
  // in FP32 internally (using each dtype's float-conversion operators at
  // load/store), so F16 is supported here without requiring an F16-capable
  // ISA -- unlike the LOWOHA flash backend, which gates F16 on AVX512-FP16 /
  // AVX-NE-CONVERT availability.
  if (input_dtype != data_type_t::f32 &&
      input_dtype != data_type_t::bf16 &&
      input_dtype != data_type_t::f16) {
    apilog_error("Unsupported data type for SDPA encoder: ",
                 static_cast<int>(input_dtype));
    return status_t::unimplemented;
  }

  kernel = std::shared_ptr<sdpa_encoder_ref_kernel_t>
           (get_sdpa_encoder_ref_kernel());  // SDPA reference kernel

  kernel->create();
  if (!kernel->check()) {
    apilog_error("SDPA encoder kernel creation failed");
    return status_t::failure;
  }

  return status_t::success;
}

} //namespace ops
} //namespace zendnnl
