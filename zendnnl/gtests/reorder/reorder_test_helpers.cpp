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

/// @file reorder_test_helpers.cpp
/// @brief Implementations of the reorder-specific gtest helpers declared in
///        reorder_test_helpers.hpp (lifted out of gtest_utils.cpp, mirroring
///        the group_matmul/ helper TU).

#include "reorder_test_helpers.hpp"

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>

// @param test_index Index of current test (for partitioning)
// @param total_tests Total number of tests
ReorderType::ReorderType(const ReorderInput &reorder_input, uint32_t test_index,
                         uint32_t total_tests) {

  // LOWOHA-only mode: default to LOWOHA reorder tests. If the user passes
  // --lowoha with any value that does not parse as true ("true"/"1",
  // case-insensitive via `parse_bool_field()`) -- e.g. false/0 or any
  // invalid non-empty token -- the test fixture's SetUp() will skip with a
  // "Please use LOA API" message.
  if (cmd_lowoha.empty()) {
    is_lowoha_test = true;
  }
  else {
    is_lowoha_test = parse_cmd_lowoha().value_or(false);
  }

  if (!is_lowoha_test) {
    // Initialize Regular Reorder params
    inplace_reorder = reorder_input.inplace_reorder ?
                      *reorder_input.inplace_reorder :
                      rand() % 2;
    mat = MatmulType(reorder_input.matmul_input, test_index, total_tests);
    mat.use_LOWOHA = is_lowoha_test;
  }
  else {
    M = (reorder_input.matmul_input.m && *reorder_input.matmul_input.m > 0)
        ? *reorder_input.matmul_input.m
        : (MATMUL_SIZE_START + std::rand() % MATMUL_SIZE_END);
    N = (reorder_input.matmul_input.n && *reorder_input.matmul_input.n > 0)
        ? *reorder_input.matmul_input.n
        : (MATMUL_SIZE_START + std::rand() % MATMUL_SIZE_END);
    // Default batch to a 3D-capable value unless dimensionality overrides it.
    batch = (reorder_input.matmul_input.batch_size &&
             *reorder_input.matmul_input.batch_size > 0)
            ? *reorder_input.matmul_input.batch_size
            : 2 + std::rand() % (BATCH_END - 1);

    // dim_choice: LOWOHA tensor rank after M/N/batch defaults. CLI --dim_choice:
    // 1 -> 1D (M=1, batch=0), 2 -> 2D (batch=1), 3 -> leave sizes (3D). If
    // omitted, random dim_choice in [0,9] gives ~20% 1D, ~50% 2D, ~30% 3D.
    if (reorder_input.dim_choice) {
      int dim_choice = *reorder_input.dim_choice;
      if (dim_choice == 1) {
        M = 1;
        batch = 0;
      }
      else if (dim_choice == 2) {
        batch = 1;
      }
    }
    else {
      // Initialize LOWOHA params
      // Randomly decide dimensionality: 1D (20%), 2D (50%), 3D (30%)
      int dim_choice = std::rand() % 10;
      if (dim_choice < 2) {
        M = 1;
        batch = 0;
      }
      else if (dim_choice < 7) {
        batch = 1;
      }
      else {
        batch = (reorder_input.matmul_input.batch_size &&
                 *reorder_input.matmul_input.batch_size > 0) ?
                *reorder_input.matmul_input.batch_size : 2 + std::rand() %
                (BATCH_END - 1);
      }
    }
    // Default data types (will be overridden by individual TEST_P tests)
    src_dtype = reorder_input.matmul_input.src_dtype ?
                *reorder_input.matmul_input.src_dtype : data_type_t::f32;
    dst_dtype = reorder_input.matmul_input.dst_dtype ?
                *reorder_input.matmul_input.dst_dtype : data_type_t::s8;
    use_strided_src = false;
    lowoha_algo = reorder_algo_t::native;

    // Thread count
    if (cmd_num_threads) {
      num_threads = cmd_num_threads;
    }
    else {
      int max_threads = omp_get_max_threads();
      num_threads = 1 + (std::rand() % max_threads);
    }

    if (reorder_input.matmul_input.weight_granularity) {
      granularity = *reorder_input.matmul_input.weight_granularity;
      if (granularity == quant_granularity_t::group) {
        num_groups = reorder_input.num_groups.has_value() ? *reorder_input.num_groups :
                     1;
        const bool group_ok = (batch != 0) && (num_groups >= 1) &&
                              (M % num_groups == 0);
        if (!group_ok) {
          if (batch == 0) {
            log_info("LOWOHA: per-group quantization requires batch>0 (2D/3D); "
                     "falling back to per-channel");
          }
          else {
            log_info("LOWOHA: invalid num_groups=", num_groups, " for M=", M,
                     " (must be >= 1 and divide M); falling back to per-channel");
          }
          granularity = quant_granularity_t::channel;
          num_groups = 1;
        }
      }
      else {
        num_groups = 1;
      }
    }
    else {
      // Quantization granularity: per-tensor (60%), per-channel (30%), per-group (10%)
      int granularity_choice = std::rand() % 10;
      if (batch == 0) {
        granularity = (granularity_choice < 7) ? quant_granularity_t::tensor
                      : quant_granularity_t::channel;
        num_groups = 1;
      }
      else {
        // 2D/3D: per-tensor (60%), per-channel (30%), per-group (10%)
        if (granularity_choice < 6) {
          granularity = quant_granularity_t::tensor;
          num_groups = 1;
        }
        else if (granularity_choice < 9) {
          granularity = quant_granularity_t::channel;
          num_groups = 1;
        }
        else {
          granularity = quant_granularity_t::group;
          // For per-group, num_groups must divide M evenly
          std::vector<uint64_t> valid_groups;
          for (uint64_t g = 2; g <= M && g <= 16; ++g) {
            if (M % g == 0) {
              valid_groups.push_back(g);
            }
          }
          if (valid_groups.empty()) {
            // Fallback to per-channel if no valid group size
            granularity = quant_granularity_t::channel;
            num_groups = 1;
          }
          else {
            num_groups = valid_groups[std::rand() % valid_groups.size()];
          }
        }
      }
    }

    // Pin the per-instance sub-modes at construction: diverse across instances,
    // but reproducible and order-independent for a fixed --seed.
    is_symmetric = (std::rand() % 2 == 0);
    cvt_direction_swap = (std::rand() % 2 == 0);
    use_col_variant = (std::rand() % 2 == 0);
    const bool pad_rows = (std::rand() % 2 == 0);
    row_padding = pad_rows ? ((std::rand() % 16) + 1) : 0;
  }
}

void PrintTo(const ReorderType &value, ::std::ostream *os) {
  if (value.is_lowoha_test) {
    *os << "M=" << value.M << ", N=" << value.N << ", batch=" << value.batch
        << ", src_dtype=" << dtype_info(value.src_dtype)
        << ", dst_dtype=" << dtype_info(value.dst_dtype)
        << ", granularity=" << static_cast<int>(value.granularity)
        << ", num_groups=" << value.num_groups
        << ", use_strided_src=" << value.use_strided_src
        << ", lowoha_algo=" << static_cast<int>(value.lowoha_algo)
        << ", num_threads=" << value.num_threads << ", seed=" << seed;
  }
  else {
    *os << "inplace_reorder=" << value.inplace_reorder << ", ";
    PrintTo(value.mat, os);
  }
}

std::string lowoha_granularity_to_str(quant_granularity_t granularity) {
  switch (granularity) {
  case quant_granularity_t::tensor:
    return "per-tensor";
  case quant_granularity_t::channel:
    return "per-channel";
  case quant_granularity_t::group:
    return "per-group";
  default:
    return "unknown";
  }
}

std::string lowoha_reorder_algo_to_str(reorder_algo_t algo) {
  switch (algo) {
  case reorder_algo_t::DT:
    return "DT";
  case reorder_algo_t::native:
    return "native";
  case reorder_algo_t::reference:
    return "reference";
  case reorder_algo_t::none:
    return "none";
  default:
    return "unknown";
  }
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
      // Check if ISA is unsupported (e.g., F16 on non-F16 platform)
      if (reorder_operator.get_reorder_isa_status() == status_t::isa_unsupported) {
        return std::make_pair(tensor_t(), status_t::isa_unsupported);
      }
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
        // Check if ISA is unsupported (e.g., F16 on non-F16 platform)
        if (reorder_operator.get_reorder_isa_status() == status_t::isa_unsupported) {
          return std::make_pair(tensor_t(), status_t::isa_unsupported);
        }

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

status_t lowoha_reorder_kernel_test(tensor_t &src_tensor,
                                    tensor_t &dst_tensor,
                                    tensor_t &scale_tensor,
                                    tensor_t &zp_tensor,
                                    const ReorderType &params,
                                    bool dynamic_quant) {
  try {
    // Build reorder_params_t structure
    reorder_params_t reorder_params;
    reorder_params.src_dtype = params.src_dtype;
    reorder_params.dst_dtype = params.dst_dtype;
    reorder_params.algo = params.lowoha_algo;
    reorder_params.num_threads = params.num_threads;
    reorder_params.dynamic_quant = dynamic_quant;

    // Set shape based on dimensionality
    // batch = 0: 1D [N], batch = 1: 2D [M, N], batch > 1: 3D [batch, M, N]
    if (params.batch == 0) {
      // 1D
      reorder_params.src_shape = {static_cast<int64_t>(params.N)};
      reorder_params.dst_shape = reorder_params.src_shape;
    }
    else if (params.batch == 1) {
      // 2D
      reorder_params.src_shape = {static_cast<int64_t>(params.M),
                                  static_cast<int64_t>(params.N)
                                 };
      reorder_params.dst_shape = reorder_params.src_shape;
    }
    else {
      // 3D
      reorder_params.src_shape = {static_cast<int64_t>(params.batch),
                                  static_cast<int64_t>(params.M),
                                  static_cast<int64_t>(params.N)
                                 };
      reorder_params.dst_shape = reorder_params.src_shape;
    }

    // Set strides if using strided source
    if (params.use_strided_src) {
      auto src_strides = src_tensor.get_stride();
      reorder_params.src_strides.assign(src_strides.begin(), src_strides.end());
    }

    // Set quantization parameters
    if (scale_tensor.get_nelem() > 0) {
      reorder_params.quant_params.scale.buff = scale_tensor.get_raw_handle_unsafe();
      reorder_params.quant_params.scale.dt = scale_tensor.get_data_type();
      auto scale_size = scale_tensor.get_size();
      reorder_params.quant_params.scale.dims.assign(scale_size.begin(),
          scale_size.end());
    }

    if (zp_tensor.get_nelem() > 0) {
      reorder_params.quant_params.zero_point.buff = zp_tensor.get_raw_handle_unsafe();
      reorder_params.quant_params.zero_point.dt = zp_tensor.get_data_type();
      auto zp_size = zp_tensor.get_size();
      reorder_params.quant_params.zero_point.dims.assign(zp_size.begin(),
          zp_size.end());
    }

    // Get raw pointers
    void *src_ptr = src_tensor.get_raw_handle_unsafe();
    void *dst_ptr = dst_tensor.get_raw_handle_unsafe();

    // Execute LOWOHA reorder
    status_t status = reorder_direct(src_ptr, dst_ptr, reorder_params);

    if (status != status_t::success) {
      // Propagate the real status (e.g. isa_unsupported) so F16 tests can
      // GTEST_SKIP() on non-AVX512-FP16 hosts instead of hard-failing.
      log_error("LOWOHA reorder_direct execution failed with status: ",
                static_cast<int>(status));
      return status;
    }

    return status_t::success;
  }
  catch (const exception_t &ex) {
    log_error("LOWOHA reorder test exception: ", ex.what());
    return status_t::failure;
  }
  catch (const std::exception &e) {
    log_error("LOWOHA reorder test exception: ", e.what());
    return status_t::failure;
  }
}

void compare_lowoha_reorder_output(tensor_t &output_tensor,
                                   tensor_t &ref_tensor,
                                   const ReorderType &params,
                                   bool &is_comparison_successful) {
  const uint64_t batch = params.batch;
  const uint64_t M = params.M;
  const uint64_t N = params.N;

  // Determine tolerance based on output data type
  float tol;
  if (params.dst_dtype == data_type_t::s8 ||
      params.dst_dtype == data_type_t::u8) {
    tol = LOWOHA_REORDER_INT8_TOL;
  }
  else if (params.dst_dtype == data_type_t::bf16 ||
           params.dst_dtype == data_type_t::f16) {
    // Both bf16 (7 mantissa bits) and f16 (10 mantissa bits) are reduced
    // precision; reuse the same conservative tolerance for round-trip checks.
    tol = LOWOHA_REORDER_BF16_TOL;
  }
  else {
    tol = LOWOHA_REORDER_F32_TOL;
  }

  // Local atomic for the OpenMP early-exit flag (avoids racing on the caller's
  // bool&); written back to is_comparison_successful once after the loop.
  std::atomic<bool> success{true};

  // Compare based on dimensionality using tensor.at() which returns float
  // batch = 0: 1D [N], batch = 1: 2D [M, N], batch > 1: 3D [batch, M, N]
  if (batch == 0) {
    // 1D comparison
    #pragma omp parallel for
    for (uint64_t i = 0; i < N; ++i) {
      if (success.load(std::memory_order_relaxed)) {
        float actual_val = output_tensor.at({i});
        float ref_val = ref_tensor.at({i});

        float abs_err = std::fabs(ref_val - actual_val);
        float rel_tol = tol * std::fabs(ref_val);
        float allowed_err = tol + rel_tol;

        if (abs_err > allowed_err) {
          log_verbose("Mismatch at [", i, "]: actual=", actual_val,
                      ", ref=", ref_val, ", abs_err=", abs_err, ", allowed=", allowed_err);
          success.store(false, std::memory_order_relaxed);
        }
      }
    }
  }
  else if (batch == 1) {
    // 2D comparison
    #pragma omp parallel for collapse(2)
    for (uint64_t i = 0; i < M; ++i) {
      for (uint64_t j = 0; j < N; ++j) {
        if (success.load(std::memory_order_relaxed)) {
          float actual_val = output_tensor.at({i, j});
          float ref_val = ref_tensor.at({i, j});

          float abs_err = std::fabs(ref_val - actual_val);
          float rel_tol = tol * std::fabs(ref_val);
          float allowed_err = tol + rel_tol;

          if (abs_err > allowed_err) {
            log_verbose("Mismatch at [", i, ",", j, "]: actual=", actual_val,
                        ", ref=", ref_val, ", abs_err=", abs_err, ", allowed=", allowed_err);
            success.store(false, std::memory_order_relaxed);
          }
        }
      }
    }
  }
  else {
    // 3D comparison
    #pragma omp parallel for collapse(3)
    for (uint64_t b = 0; b < batch; ++b) {
      for (uint64_t i = 0; i < M; ++i) {
        for (uint64_t j = 0; j < N; ++j) {
          if (success.load(std::memory_order_relaxed)) {
            float actual_val = output_tensor.at({b, i, j});
            float ref_val = ref_tensor.at({b, i, j});

            float abs_err = std::fabs(ref_val - actual_val);
            float rel_tol = tol * std::fabs(ref_val);
            float allowed_err = tol + rel_tol;

            if (abs_err > allowed_err) {
              log_verbose("Mismatch at [", b, ",", i, ",", j, "]: actual=", actual_val,
                          ", ref=", ref_val, ", abs_err=", abs_err, ", allowed=", allowed_err);
              success.store(false, std::memory_order_relaxed);
            }
          }
        }
      }
    }
  }

  if (!success.load()) {
    is_comparison_successful = false;
  }
}

void log_lowoha_test_info(const ReorderType &params, data_type_t src_dt,
                          data_type_t dst_dt, bool strided, bool use_scale_zp) {
  // Helper: stringify a data_type_t for log output. Centralized here so a
  // future addition of a new dtype only needs one update site.
  auto dtype_to_label = [](data_type_t dt) -> std::string {
    switch (dt) {
    case data_type_t::bf16:
      return "BF16";
    case data_type_t::f32:
      return "F32";
    case data_type_t::f16:
      return "F16";
    case data_type_t::s8:
      return "S8";
    case data_type_t::u8:
      return "U8";
    default:
      return "unknown";
    }
  };
  std::string src_dt_str = dtype_to_label(src_dt);
  std::string dst_dt_str = dtype_to_label(dst_dt);
  log_info("LOWOHA Reorder: batch=", params.batch,
           " M=", params.M, " N=", params.N,
           " src=", src_dt_str, " dst=", dst_dt_str,
           " granularity=", lowoha_granularity_to_str(params.granularity),
           " groups=", params.num_groups, " strided=", strided,
           " scale_zp=", use_scale_zp, " threads=", params.num_threads);
}

std::vector<size_t> get_lowoha_shape(const ReorderType &params) {
  if (params.batch == 0) return {params.N};
  else if (params.batch == 1) return {params.M, params.N};
  else return {params.batch, params.M, params.N};
}

std::vector<size_t> get_lowoha_strided_shape(const ReorderType &params) {
  // Padding fixed at construction (params.row_padding) -> reproducible layout.
  uint64_t padding = params.row_padding;
  uint64_t M = params.M;
  uint64_t N = params.N;
  uint64_t batch = params.batch;

  if (batch == 0) {
    // 1D: [N]
    log_info("Strided 1D, padding=", padding, ", strided_shape={", N + padding,
             "}");
    return {N + padding};
  }
  else if (batch == 1) {
    // 2D: [M, N] -> strides {N+padding, 1}
    log_info("Strided 2D, padding=", padding, ", strided_shape={", M, ",",
             N + padding, "}",
             " strides={", N + padding, ",1}");
    return {M, N + padding};
  }
  else {
    // 3D: [batch, M, N] -> strides {M*(N+padding), N+padding, 1}
    log_info("Strided 3D, padding=", padding, ", strided_shape={", batch, ",", M,
             ",", N + padding, "}",
             " strides={", M * (N + padding), ",", N + padding, ",1}");
    return {batch, M, N + padding};
  }
}

std::vector<size_t> get_lowoha_quant_shape(const ReorderType &params) {
  uint64_t M = params.M;
  uint64_t N = params.N;
  uint64_t batch = params.batch;
  uint64_t num_groups = params.num_groups;
  quant_granularity_t granularity = params.granularity;

  // This helper supports all 5 granularities:
  //   1. per-tensor
  //   2. per-channel-row  (per-token)
  //   3. per-channel-col
  //   4. per-group-row
  //   5. per-group-col
  //
  // The base granularity is selected by the caller via params.granularity.
  // Within that choice:
  //   - For "channel" granularity: randomly pick per-row or per-col (50/50).
  //   - For "group"   granularity: randomly pick per-row or per-col (50/50),
  //     with fallback to per-row if num_groups does not divide N.

  if (batch == 0) {
    // 1D: per-tensor {1} or per-channel {N}  (only 2 granularities for 1D)
    return (granularity == quant_granularity_t::tensor) ?
           std::vector<size_t> {1} :
           std::vector<size_t> {N};
  }

  // 2D or 3D: support all 5 granularities. per-col vs per-row comes from the
  // param (fixed at construction) so the scale/zp shape is reproducible.
  bool use_col_variant = params.use_col_variant;

  if (granularity == quant_granularity_t::tensor) {
    // Per-tensor: single scale/zp for all elements
    if (batch == 1) {
      log_info("Granularity: per-tensor, dims={1,1}");
      return {1, 1};
    }
    else {
      log_info("Granularity: per-tensor, dims={1,1,1}");
      return {1, 1, 1};
    }
  }
  else if (granularity == quant_granularity_t::channel) {
    if (use_col_variant) {
      // Per-channel-col: one scale/zp per column
      if (batch == 1) {
        log_info("Granularity: per-channel-col, dims={1,", N, "}");
        return {1, N};
      }
      else {
        log_info("Granularity: per-channel-col, dims={1,1,", N, "}");
        return {1, 1, N};
      }
    }
    else {
      // Per-channel-row: one scale/zp per row (per-token)
      if (batch == 1) {
        log_info("Granularity: per-channel-row, dims={", M, ",1}");
        return {M, 1};
      }
      else {
        log_info("Granularity: per-channel-row, dims={1,", M, ",1}");
        return {1, M, 1};
      }
    }
  }
  else {
    // Group granularity: randomly pick per-group-row or per-group-col
    if (use_col_variant && N % num_groups == 0) {
      // Per-group-col: dims = {M, G} (2D) or {1, M, G} (3D)
      // Reuse params.num_groups only if it also divides N
      if (batch == 1) {
        log_info("Granularity: per-group-col, dims={", M, ",",
                 num_groups, "}, N=", N, ", N/G=", N / num_groups);
        return {M, num_groups};
      }
      else {
        log_info("Granularity: per-group-col, dims={1,", M, ",",
                 num_groups, "}, N=", N, ", N/G=", N / num_groups);
        return {1, M, num_groups};
      }
    }
    else {
      // Per-group-row: dims = {G, N} (2D) or {1, G, N} (3D)
      // num_groups already validated to divide M during ReorderType construction
      if (use_col_variant) {
        log_info("Granularity: per-group-col fallback to per-group-row "
                 "(num_groups=", num_groups, " does not divide N=", N, ")");
      }
      if (batch == 1) {
        log_info("Granularity: per-group-row, dims={", num_groups, ",",
                 N, "}, M=", M, ", M/G=", M / num_groups);
        return {num_groups, N};
      }
      else {
        log_info("Granularity: per-group-row, dims={1,", num_groups, ",",
                 N, "}, M=", M, ", M/G=", M / num_groups);
        return {1, num_groups, N};
      }
    }
  }
}

void compare_lowoha_quant_output(tensor_t &original_tensor,
                                 tensor_t &dequant_tensor,
                                 tensor_t &scale_tensor,
                                 const ReorderType &params,
                                 bool &is_comparison_successful) {

  const uint64_t batch = params.batch;
  const uint64_t M = params.M;
  const uint64_t N = params.N;

  // Compute tolerance based on the max scale value
  // Quantization round-trip error is bounded by scale/2 (rounding error)
  // plus additional epsilon for BF16 truncation and numerical noise.
  //
  // The scale tensor's storage dtype may be f32, bf16, or f16. Use the
  // dtype-aware load path (raw f32 for f32 storage, bf16/f16 widen helpers
  // for the others) instead of a blind float* cast — otherwise an f16 or
  // bf16 scale buffer would be read as garbage f32 bits.
  const data_type_t scale_dt = scale_tensor.get_data_type();
  const size_t scale_nelem = scale_tensor.get_nelem();
  float max_scale = 0.0f;
  if (scale_dt == data_type_t::f32) {
    const float *scale_ptr = static_cast<const float *>(
        scale_tensor.get_raw_handle_unsafe());
    for (size_t i = 0; i < scale_nelem; ++i) {
      max_scale = std::max(max_scale, std::fabs(scale_ptr[i]));
    }
  } else if (scale_dt == data_type_t::bf16) {
    const int16_t *scale_ptr = static_cast<const int16_t *>(
        scale_tensor.get_raw_handle_unsafe());
    for (size_t i = 0; i < scale_nelem; ++i) {
      const float v = bfloat16_t::bf16_to_f32_val(scale_ptr[i]);
      max_scale = std::max(max_scale, std::fabs(v));
    }
  } else if (scale_dt == data_type_t::f16) {
    const uint16_t *scale_ptr = static_cast<const uint16_t *>(
        scale_tensor.get_raw_handle_unsafe());
    for (size_t i = 0; i < scale_nelem; ++i) {
      const float v = float16_t::f16_to_f32_val(scale_ptr[i]);
      max_scale = std::max(max_scale, std::fabs(v));
    }
  } else {
    // Unknown scale dtype — fall back to f32 interpretation. This matches
    // the historical behavior and keeps existing tests passing for the
    // bf16/f32 source paths.
    const float *scale_ptr = static_cast<const float *>(
        scale_tensor.get_raw_handle_unsafe());
    for (size_t i = 0; i < scale_nelem; ++i) {
      max_scale = std::max(max_scale, std::fabs(scale_ptr[i]));
    }
  }

  // Base tolerance: half a quantization step (max rounding error from
  // round(src_val / scale)).
  float tol = max_scale / 2.0f;

  // Add epsilon for numerical noise and reduced-precision truncation:
  //   - BF16 has 7 mantissa bits  -> rel error ≈ 2^-7 ≈ 0.8% per truncation
  //   - F16  has 10 mantissa bits -> rel error ≈ 2^-10 ≈ 0.1% per truncation
  // For float<->float scaled round-trips, the intermediate (post-divide)
  // value can be up to ~|src|/scale, and f16 truncation of THAT intermediate
  // re-scales back to scale * |src|/scale * 2^-mantissa = |src| * 2^-mantissa.
  // We bake that into the epsilon when f16 is involved alongside bf16, where
  // the round-trip narrows twice with very different mantissa widths.
  const bool involves_bf16 = (params.src_dtype == data_type_t::bf16) ||
                             (params.dst_dtype == data_type_t::bf16);
  const bool involves_f16  = (params.src_dtype == data_type_t::f16)  ||
                             (params.dst_dtype == data_type_t::f16);
  if (involves_bf16 && involves_f16) {
    // BF16 <-> F16 scaled round-trip: error compounds across two narrows
    // (bf16 -> f16 narrows ~7 mantissa bits to ~10, and the reverse direction
    // narrows ~10 mantissa bits to ~7). For typical scale ~ 0.1 and src in
    // [-2,2] with zp in [-64,64], the intermediate value lives near
    // |src|/scale + zp ~ 80, which f16 represents with ~0.08 absolute step.
    tol += 0.1f;
  }
  else if (involves_bf16) {
    tol += 0.03f;   // BF16 truncation + numerical noise
  }
  else if (involves_f16) {
    tol += 0.03f;   // F16 truncation + numerical noise
  }
  else {
    tol += 0.001f;  // Small epsilon for F32 numerical noise
  }

  log_info("Round-trip tolerance: ", tol, " (max_scale=", max_scale,
           ", scale/2=", max_scale / 2.0f, ")");

  std::atomic<bool> success{true};

  // Compare based on dimensionality using tensor.at() which returns float
  // batch = 0: 1D [N], batch = 1: 2D [M, N], batch > 1: 3D [batch, M, N]
  if (batch == 0) {
    // 1D comparison
    #pragma omp parallel for
    for (uint64_t j = 0; j < N; ++j) {
      if (success.load(std::memory_order_relaxed)) {
        float orig_val = original_tensor.at({j});
        float deq_val = dequant_tensor.at({j});
        float abs_err = std::fabs(orig_val - deq_val);

        if (abs_err > tol) {
          log_verbose("Mismatch at [", j, "]: orig=", orig_val,
                      ", dequant=", deq_val, ", abs_err=", abs_err,
                      ", tol=", tol);
          success.store(false, std::memory_order_relaxed);
        }
      }
    }
  }
  else if (batch == 1) {
    // 2D comparison
    #pragma omp parallel for collapse(2)
    for (uint64_t i = 0; i < M; ++i) {
      for (uint64_t j = 0; j < N; ++j) {
        if (success.load(std::memory_order_relaxed)) {
          float orig_val = original_tensor.at({i, j});
          float deq_val = dequant_tensor.at({i, j});
          float abs_err = std::fabs(orig_val - deq_val);

          if (abs_err > tol) {
            log_verbose("Mismatch at [", i, ",", j, "]: orig=",
                        orig_val, ", dequant=", deq_val, ", abs_err=", abs_err,
                        ", tol=", tol);
            success.store(false, std::memory_order_relaxed);
          }
        }
      }
    }
  }
  else {
    // 3D comparison
    #pragma omp parallel for collapse(3)
    for (uint64_t b = 0; b < batch; ++b) {
      for (uint64_t i = 0; i < M; ++i) {
        for (uint64_t j = 0; j < N; ++j) {
          if (success.load(std::memory_order_relaxed)) {
            float orig_val = original_tensor.at({b, i, j});
            float deq_val = dequant_tensor.at({b, i, j});
            float abs_err = std::fabs(orig_val - deq_val);

            if (abs_err > tol) {
              log_verbose("Mismatch at [", b, ",", i, ",", j,
                          "]: orig=", orig_val, ", dequant=", deq_val,
                          ", abs_err=", abs_err, ", tol=", tol);
              success.store(false, std::memory_order_relaxed);
            }
          }
        }
      }
    }
  }

  if (!success.load()) {
    is_comparison_successful = false;
  }
}
