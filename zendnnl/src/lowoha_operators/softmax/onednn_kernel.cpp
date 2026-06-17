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

#include "onednn_kernel.hpp"
#include "common/logging.hpp"
#include "common/profiler.hpp"
#include "common/bfloat16.hpp"
#include "common/float16.hpp"
#include <sstream>
#include <unordered_map>
#include <vector>

#if ZENDNNL_DEPENDS_ONEDNN
  #include "dnnl.hpp"
  using namespace dnnl;
#endif

namespace zendnnl {
namespace lowoha {
namespace softmax {

using zendnnl::common::float16_t;

#if ZENDNNL_DEPENDS_ONEDNN

status_t softmax_onednn_wrapper(
  const void *input,
  void *output,
  const softmax_params &params
) {
  try {
    // Create OneDNN engine and stream
    dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    dnnl::stream strm(eng);

    // Determine data type
    dnnl::memory::data_type dtype;
    if (params.src_dt == data_type_t::f32) {
      dtype = dnnl::memory::data_type::f32;
    }
    else if (params.src_dt == data_type_t::bf16) {
      dtype = dnnl::memory::data_type::bf16;
    }
    else if (params.src_dt == data_type_t::f16) {
      dtype = dnnl::memory::data_type::f16;
    }
    else {
      log_error("Softmax OneDNN: Unsupported data type");
      return status_t::failure;
    }

    // Validate that shape information is provided
    if (params.ndims <= 0 || params.ndims > SOFTMAX_MAX_NDIMS) {
      log_error("Softmax OneDNN: Invalid ndims: ", params.ndims,
                " (must be 1-", SOFTMAX_MAX_NDIMS, ").");
      return status_t::failure;
    }

    // Build OneDNN memory dimensions from original N-D shape
    dnnl::memory::dims src_dims;
    src_dims.reserve(params.ndims);
    uint64_t total_elems = 1;
    for (int i = 0; i < params.ndims; ++i) {
      src_dims.push_back(static_cast<dnnl::memory::dim>(params.shape[i]));
      total_elems *= params.shape[i];
    }

    // OneDNN has no softmin; run softmax on -input via a scratch buffer.
    // The scratch is allocated as a typed vector per dtype so OneDNN reads it
    // through a naturally-aligned pointer (f32 = 4-byte, bf16/f16 = 2-byte);
    // only the vector matching src_dt is allocated. 16-bit dtypes store the
    // raw bit pattern as uint16_t, which OneDNN interprets via its bf16/f16
    // memory descriptor.
    const void *effective_input = input;
    std::vector<float>    softmin_scratch_f32;
    std::vector<uint16_t> softmin_scratch_16;
    if (params.softmin) {
      if (params.src_dt == data_type_t::f32) {
        softmin_scratch_f32.resize(total_elems);
        const float *in = static_cast<const float *>(input);
        #pragma omp parallel for
        for (uint64_t i = 0; i < total_elems; ++i) {
          softmin_scratch_f32[i] = -in[i];
        }
        effective_input = softmin_scratch_f32.data();
      }
      else if (params.src_dt == data_type_t::bf16) {
        softmin_scratch_16.resize(total_elems);
        const bfloat16_t *in = static_cast<const bfloat16_t *>(input);
        #pragma omp parallel for
        for (uint64_t i = 0; i < total_elems; ++i) {
          softmin_scratch_16[i] = static_cast<uint16_t>(
                                    bfloat16_t::f32_to_bf16_val(-static_cast<float>(in[i])));
        }
        effective_input = softmin_scratch_16.data();
      }
      else {   // f16
        softmin_scratch_16.resize(total_elems);
        const float16_t *in = static_cast<const float16_t *>(input);
        #pragma omp parallel for
        for (uint64_t i = 0; i < total_elems; ++i) {
          softmin_scratch_16[i] = float16_t::f32_to_f16_val(-static_cast<float>(in[i]));
        }
        effective_input = softmin_scratch_16.data();
      }
    }

    // Normalize axis to positive value
    int softmax_axis = params.axis >= 0 ? params.axis : params.ndims + params.axis;

    // Reject an out-of-range axis with a clear message instead of letting an
    // invalid value reach OneDNN (which would throw an opaque dnnl::error).
    if (softmax_axis < 0 || softmax_axis >= params.ndims) {
      log_error("Softmax OneDNN: Invalid axis: ", params.axis, " for ",
                params.ndims, "D tensor (must be in [-", params.ndims, ", ",
                params.ndims, ")).");
      return status_t::failure;
    }

    // Get appropriate format tag for this dimensionality
    dnnl::memory::format_tag format;
    switch (params.ndims) {
    case 1:
      format = dnnl::memory::format_tag::a;
      break;
    case 2:
      format = dnnl::memory::format_tag::ab;
      break;
    case 3:
      format = dnnl::memory::format_tag::abc;
      break;
    case 4:
      format = dnnl::memory::format_tag::abcd;
      break;
    case 5:
      format = dnnl::memory::format_tag::abcde;
      break;
    default:
      log_error("Softmax OneDNN: Unsupported number of dimensions: ", params.ndims);
      return status_t::failure;
    }

    // Build shape string for logging
    std::ostringstream shape_str;
    shape_str << params.shape[0];
    for (int i = 1; i < params.ndims; ++i) {
      shape_str << "," << params.shape[i];
    }
    log_info("Softmax OneDNN: ", params.ndims, "D tensor (shape=[", shape_str.str(),
             "]), softmax on axis ", softmax_axis,
             ", log_softmax=", (params.log_softmax ? "true" : "false"),
             ", softmin=", (params.softmin ? "true" : "false"));

    // Create memory descriptors
    auto src_md = dnnl::memory::desc(src_dims, dtype, format);
    auto dst_md = dnnl::memory::desc(src_dims, dtype, format);

    // Create softmax primitive descriptor
    dnnl::softmax_forward::primitive_desc softmax_pd;

    if (params.log_softmax) {
      softmax_pd = dnnl::softmax_forward::primitive_desc(
                     eng,
                     dnnl::prop_kind::forward_inference,
                     dnnl::algorithm::softmax_log,
                     src_md,
                     dst_md,
                     softmax_axis
                   );
    }
    else {
      softmax_pd = dnnl::softmax_forward::primitive_desc(
                     eng,
                     dnnl::prop_kind::forward_inference,
                     dnnl::algorithm::softmax_accurate,
                     src_md,
                     dst_md,
                     softmax_axis
                   );
    }

    // Create memory objects (src points to -input scratch when softmin).
    auto src_mem = dnnl::memory(src_md, eng,
                                const_cast<void *>(effective_input));
    auto dst_mem = dnnl::memory(dst_md, eng, output);

    // Create softmax primitive
    auto softmax_prim = dnnl::softmax_forward(softmax_pd);

    // Setup arguments
    std::unordered_map<int, dnnl::memory> softmax_args;
    softmax_args.insert({DNNL_ARG_SRC, src_mem});
    softmax_args.insert({DNNL_ARG_DST, dst_mem});

    // Execute softmax
    softmax_prim.execute(strm, softmax_args);

    // Wait for completion
    strm.wait();

    log_info("Softmax OneDNN: Execution completed successfully");
    return status_t::success;

  }
  catch (const dnnl::error &e) {
    log_error("Softmax OneDNN error: ", e.what(), " (status: ", e.status, ")");
    return status_t::failure;
  }
  catch (const std::exception &e) {
    log_error("Softmax OneDNN exception: ", e.what());
    return status_t::failure;
  }
}

#else

status_t softmax_onednn_wrapper(
  const void *input,
  void *output,
  const softmax_params &params
) {
  log_error("Softmax OneDNN: OneDNN support not enabled");
  return status_t::failure;
}

#endif // ZENDNNL_DEPENDS_ONEDNN

} // namespace softmax
} // namespace lowoha
} // namespace zendnnl
