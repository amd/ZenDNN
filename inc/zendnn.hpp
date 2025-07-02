/*******************************************************************************
* Modifications Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
/// C++ API

#ifndef ZENDNN_HPP
#define ZENDNN_HPP

#include "zendnn_config.h"

/// @cond DO_NOT_DOCUMENT_THIS
#include <algorithm>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "zendnn.h"
#include "zendnn_logging.hpp"

/// @endcond

// __cpp_exceptions is referred from
// https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_exceptions.html
// gcc < 5 does not define __cpp_exceptions but __EXCEPTIONS,
// Microsoft C++ Compiler does not provide an option to disable exceptions
#ifndef ZENDNN_ENABLE_EXCEPTIONS
    #if __cpp_exceptions || __EXCEPTIONS \
        || (defined(_MSC_VER) && !defined(__clang__))
        #define ZENDNN_ENABLE_EXCEPTIONS 1
    #else
        #define ZENDNN_ENABLE_EXCEPTIONS 0
    #endif
#endif

#if defined(__GNUC__) || defined(__clang__)
    #define ZENDNN_TRAP() __builtin_trap()
#elif defined(__INTEL_COMPILER) || defined(_MSC_VER)
    #define ZENDNN_TRAP() __debugbreak()
#else
    #error "unknown compiler"
#endif

#if ZENDNN_ENABLE_EXCEPTIONS
#define ZENDNN_THROW_ERROR(status, msg) throw error(status, msg)
#else
#include <cstdio>
#define ZENDNN_THROW_ERROR(status, msg) \
    do { \
        fputs(msg, stderr); \
        ZENDNN_TRAP(); \
    } while (0)
#endif

/// @addtogroup zendnn_api ZENDNN API
/// @{

/// ZENDNN namespace
namespace zendnn {

typedef int16_t bfloat16;
/// @addtogroup zendnn_api_utils Utilities
/// Utility types and definitions.
/// @{

/// ZENDNN exception class.
///
/// This class captures the status returned by a failed C API function and
/// the error message from the call site.
struct error : public std::exception {
    zendnn_status_t status;
    const char *message;

    /// Constructs an instance of an exception class.
    ///
    /// @param status The error status returned by a C API function.
    /// @param message The error message.
    error(zendnn_status_t status, const char *message)
        : status(status), message(message) {}

    /// Returns the explanatory string.
    const char *what() const noexcept override {
        return message;
    }

    /// A convenience function for wrapping calls to C API functions. Checks
    /// the return status and throws an zendnn::error in case of failure.
    ///
    /// @param status The error status returned by a C API function.
    /// @param message The error message.
    static void wrap_c_api(zendnn_status_t status, const char *message) {
        if (status != zendnn_success) {
            ZENDNN_THROW_ERROR(status, message);
        }
    }
};

/// @cond DO_NOT_DOCUMENT_THIS
template <typename T>
void validate_container_size(const T &v, const char *error_message,
                             int min_size = 1, int max_size = -1) {
    const int size = (int)v.size();
    if (size < min_size || (max_size >= 0 && size > max_size)) {
        ZENDNN_THROW_ERROR(zendnn_invalid_arguments, error_message);
    }
}
/// @endcond

/// A class that provides the destructor for a ZENDNN C API handle.
template <typename T>
struct handle_traits {};

/// ZENDNN C API handle wrapper class.
///
/// This class is used as the base class for primitive (zendnn::primitive),
/// engine (zendnn::engine), and stream (zendnn::stream) classes, as well as
/// others. An object of the zendnn::handle class can be passed by value.
///
/// A handle can be weak, in which case it follows std::weak_ptr semantics.
/// Otherwise, it follows `std::shared_ptr` semantics.
///
/// @note
///     The implementation stores ZENDNN C API handles in a `std::shared_ptr`
///     with deleter set to a dummy function in the weak mode.
///
template <typename T, typename traits = handle_traits<T>>
struct handle {
  private:
    static zendnn_status_t dummy_destructor(T) {
        return zendnn_success;
    }
    std::shared_ptr<typename std::remove_pointer<T>::type> data_ {0};

  protected:
    bool operator==(const T other) const {
        return other == data_.get();
    }
    bool operator!=(const T other) const {
        return !(*this == other);
    }

  public:
    /// Constructs an empty handle object.
    ///
    /// @warning
    ///     Uninitialized object cannot be used in most library calls and is
    ///     equivalent to a null pointer. Any attempt to use its methods, or
    ///     passing it to the other library function, will cause an exception
    ///     to be thrown.
    handle() = default;

    /// Copy constructor.
    handle(const handle<T, traits> &) = default;
    /// Assignment operator.
    handle<T, traits> &operator=(const handle<T, traits> &) = default;
    /// Move constructor.
    handle(handle<T, traits> &&) = default;
    /// Move assignment operator.
    handle<T, traits> &operator=(handle<T, traits> &&) = default;

    /// Constructs a handle wrapper object from a C API handle.
    ///
    /// @param t The C API handle to wrap.
    /// @param weak A flag specifying whether to construct a weak wrapper;
    ///     defaults to @c false.
    explicit handle(T t, bool weak = false) {
        reset(t, weak);
    }

    /// Resets the handle wrapper objects to wrap a new C API handle.
    ///
    /// @param t The new value of the C API handle.
    /// @param weak A flag specifying whether the wrapper should be weak;
    ///     defaults to @c false.
    void reset(T t, bool weak = false) {
        data_.reset(t, weak ? &dummy_destructor : traits::destructor);
    }

    /// Returns the underlying C API handle.
    ///
    /// @param allow_empty A flag signifying whether the method is allowed to
    ///     return an empty (null) object without throwing an exception.
    /// @returns The underlying C API handle.
    T get(bool allow_empty = false) const {
        T result = data_.get();
        if (allow_empty == false && result == nullptr)
            ZENDNN_THROW_ERROR(
                zendnn_invalid_arguments, "object is not initialized");
        return result;
    }

    /// Converts a handle to the underlying C API handle type. Does not throw
    /// and returns `nullptr` if the object is empty.
    ///
    /// @returns The underlying C API handle.
    explicit operator T() const {
        return get(true);
    }

    /// Checks whether the object is not empty.
    ///
    /// @returns Whether the object is not empty.
    explicit operator bool() const {
        return get(true) != nullptr;
    }

    /// Equality operator.
    ///
    /// @param other Another handle wrapper.
    /// @returns @c true if this and the other handle wrapper manage the same
    ///     underlying C API handle, and @c false otherwise. Empty handle
    ///     objects are considered to be equal.
    bool operator==(const handle<T, traits> &other) const {
        return other.data_.get() == data_.get();
    }

    /// Inequality operator.
    ///
    /// @param other Another handle wrapper.
    /// @returns @c true if this and the other handle wrapper manage different
    ///     underlying C API handles, and @c false otherwise. Empty handle
    ///     objects are considered to be equal.
    bool operator!=(const handle &other) const {
        return !(*this == other);
    }
};

/// @cond DO_NOT_DOCUMENT_THIS
template <>
struct handle_traits<zendnn_memory_t> {
    static zendnn_status_t destructor(zendnn_memory_t p) {
        return zendnn_memory_destroy(p);
    }
};

template <>
struct handle_traits<zendnn_primitive_desc_t> {
    static zendnn_status_t destructor(zendnn_primitive_desc_t p) {
        return zendnn_primitive_desc_destroy(p);
    }
};

template <>
struct handle_traits<zendnn_primitive_t> {
    static zendnn_status_t destructor(zendnn_primitive_t p) {
        return zendnn_primitive_destroy(p);
    }
};

template <>
struct handle_traits<zendnn_primitive_desc_iterator_t> {
    static zendnn_status_t destructor(zendnn_primitive_desc_iterator_t p) {
        return zendnn_primitive_desc_iterator_destroy(p);
    }
};
/// @endcond

/// @} zendnn_api_utils

struct stream;
struct memory;
struct primitive_desc;

/// @addtogroup zendnn_api_primitives Primitives
/// Compute primitives
/// @sa @ref dev_guide_basic_concepts
/// @{

/// @addtogroup zendnn_api_primitives_common Common
/// Common operations to create, destroy and inspect primitives
/// @{

/// Base class for all computational primitives.
struct primitive : public handle<zendnn_primitive_t> {
    /// Kinds of primitives supported by the library.
    enum class kind {
        /// Undefined primitive
        undef = zendnn_undefined_primitive,
        /// A reorder primitive.
        reorder = zendnn_reorder,
        /// A shuffle primitive.
        shuffle = zendnn_shuffle,
        /// A (out-of-place) tensor concatenation primitive.
        concat = zendnn_concat,
        /// A summation primitive.
        sum = zendnn_sum,
        /// A convolution primitive.
        convolution = zendnn_convolution,
        /// A deconvolution primitive.
        deconvolution = zendnn_deconvolution,
        /// An element-wise primitive.
        eltwise = zendnn_eltwise,
        /// A softmax primitive.
        softmax = zendnn_softmax,
        /// A pooling primitive.
        pooling = zendnn_pooling,
        /// An LRN primitive.
        lrn = zendnn_lrn,
        /// A batch normalization primitive.
        batch_normalization = zendnn_batch_normalization,
        /// A layer normalization primitive.
        layer_normalization = zendnn_layer_normalization,
        /// An inner product primitive.
        inner_product = zendnn_inner_product,
        /// An RNN primitive.
        rnn = zendnn_rnn,
        /// A binary primitive.
        binary = zendnn_binary,
        /// A logsoftmax primitive.
        logsoftmax = zendnn_logsoftmax,
        /// A matmul (matrix multiplication) primitive.
        matmul = zendnn_matmul,
        /// A resampling primitive.
        resampling = zendnn_resampling,
        /// A pooling version 2 primitive.
        pooling_v2 = zendnn_pooling_v2,
        /// A reduction primitive.
        reduction = zendnn_reduction,
        /// A PReLU primitive.
        prelu = zendnn_prelu,
        /// A softmax version 2 primitive.
        softmax_v2 = zendnn_softmax_v2,
        /* add new primitive */
        /// An embedding bag primitive.
        embedding_bag = zendnn_embedding_bag,
        /// Attention primitive.
        attention = zendnn_attention,
    };

    using handle::handle;

    /// Default constructor. Constructs an empty object.
    primitive() = default;

    /// Constructs a primitive from a C API primitive descriptor.
    ///
    /// @param c_pd C API primitive descriptor.
    primitive(const_zendnn_primitive_desc_t c_pd);

    /// Constructs a primitive from a C API primitive descriptor and a cache blob.
    ///
    /// @param c_pd C API primitive descriptor.
    /// @param cache_blob Cache blob.
    primitive(const_zendnn_primitive_desc_t c_pd,
              const std::vector<uint8_t> &cache_blob);

    /// Constructs a primitive from a primitive descriptor.
    ///
    /// @param pd Primitive descriptor.
    primitive(const primitive_desc &pd);

    /// Constructs a primitive from a primitive descriptor and a cache blob.
    ///
    /// @param pd Primitive descriptor.
    /// @param cache_blob Cache blob.
    primitive(const primitive_desc &pd, const std::vector<uint8_t> &cache_blob);

    /// Returns the C API primitive descriptor of the underlying C API
    /// primitive.
    ///
    /// @returns The underlying C API primitive descriptor.
    inline const_zendnn_primitive_desc_t get_primitive_desc() const;

    /// Returns the kind of the primitive.
    ///
    /// @returns The primitive kind.
    inline kind get_kind() const;

    /// Returns a cache blob for the primitive.
    ///
    /// @returns Vector containing the cache blob.
    ///
    /// @note The cache blob can be empty. It's the user's responsibility to
    ///     check whether it's empty prior to passing it to the primitive
    ///     constructor.
    inline std::vector<uint8_t> get_cache_blob() const;

    /// Executes computations specified by the primitive in a specified stream.
    ///
    /// Arguments are passed via an arguments map containing <index,
    /// memory object> pairs. The index must be one of the `ZENDNN_ARG_*` values
    /// such as `ZENDNN_ARG_SRC`, and the memory must have a memory descriptor
    /// matching the one returned by
    /// primitive_desc::query_md(#query::exec_arg_md, index) unless using
    /// dynamic shapes (see #ZENDNN_RUNTIME_DIM_VAL).
    ///
    /// @param astream Stream object. The stream must belong to the same engine
    ///     as the primitive.
    /// @param args Arguments map.
    void execute(const stream &astream,
                 const std::unordered_map<int, memory> &args) const;
};

/// Converts primitive kind enum value from C++ API to C API type.
///
/// @param akind C++ API primitive kind enum value.
/// @returns Corresponding C API primitive kind enum value.
inline zendnn_primitive_kind_t convert_to_c(primitive::kind akind) {
    return static_cast<zendnn_primitive_kind_t>(akind);
}

const_zendnn_primitive_desc_t primitive::get_primitive_desc() const {
    const_zendnn_primitive_desc_t pd;
    error::wrap_c_api(zendnn_primitive_get_primitive_desc(get(), &pd),
                      "could not get a primitive descriptor from a primitive");
    return pd;
}

zendnn::primitive::kind primitive::get_kind() const {
    const_zendnn_primitive_desc_t pd = get_primitive_desc();
    // TODO (Roma): the code below is only needed because get_primitive_desc
    // returns a C type.
    zendnn_primitive_kind_t kind;
    error::wrap_c_api(zendnn_primitive_desc_query(
                          pd, zendnn_query_primitive_kind, 0, (void *)&kind),
                      "could not get a primitive kind from a primitive descriptor");
    return static_cast<zendnn::primitive::kind>(kind);
}

std::vector<uint8_t> primitive::get_cache_blob() const {
    size_t size;
    error::wrap_c_api(zendnn_primitive_get_cache_blob(get(), &size, nullptr),
                      "could not get cache blob size from a primitive");

    std::vector<uint8_t> cache_blob(size);
    error::wrap_c_api(
        zendnn_primitive_get_cache_blob(get(), &size, cache_blob.data()),
        "could not get a cache blob from a primitive");
    return cache_blob;
}

/// @} zendnn_api_primitives_common

/// @addtogroup zendnn_api_attributes
///
/// A container for parameters that extend primitives behavior.
///
/// Attributes can also contain Post-ops, which are computations executed
/// after the primitive.
///
/// @sa @ref dev_guide_attributes
/// @sa @ref dev_guide_attributes_post_ops
///
/// @{

/// Floating-point math mode
enum class fpmath_mode {
    /// Default behavior, no downconversions allowed
    strict = zendnn_fpmath_mode_strict,
    /// Implicit f32->bf16 conversions allowed
    bf16 = zendnn_fpmath_mode_bf16,
    /// Implicit f32->f16 conversions allowed
    f16 = zendnn_fpmath_mode_f16,
    /// Implicit f32->f16 or f32->bf16 conversions allowed
    any = zendnn_fpmath_mode_any
};

/// Converts an fpmath mode enum value from C++ API to C API type.
///
/// @param mode C++ API fpmath mode enum value.
/// @returns Corresponding C API fpmath mode enum value.
inline zendnn_fpmath_mode_t convert_to_c(fpmath_mode mode) {
    return static_cast<zendnn_fpmath_mode_t>(mode);
}

/// Scratchpad mode
enum class scratchpad_mode {
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
    library = zendnn_scratchpad_mode_library,
    /// The user manages the scratchpad allocation by querying and providing
    /// the scratchpad memory to primitives. This mode is thread-safe as long
    /// as the scratchpad buffers are not used concurrently by two primitive
    /// executions.
    user = zendnn_scratchpad_mode_user,
};

/// Converts a scratchpad mode enum value from C++ API to C API type.
///
/// @param mode C++ API scratchpad mode enum value.
/// @returns Corresponding C API scratchpad mode enum value.
inline zendnn_scratchpad_mode_t convert_to_c(scratchpad_mode mode) {
    return static_cast<zendnn_scratchpad_mode_t>(mode);
}

/// Propagation kind.
enum class prop_kind {
    /// Undefined propagation kind.
    undef = zendnn_prop_kind_undef,
    /// Forward data propagation (training mode). In this mode, primitives
    /// perform computations necessary for subsequent backward propagation.
    forward_training = zendnn_forward_training,
    /// Forward data propagation (inference mode). In this mode, primitives
    /// perform only computations that are necessary for inference and omit
    /// computations that are necessary only for backward propagation.
    forward_inference = zendnn_forward_inference,
    /// Forward data propagation,
    /// alias for #zendnn::prop_kind::forward_inference.
    forward_scoring = zendnn_forward_scoring,
    /// Forward data propagation,
    /// alias for #zendnn::prop_kind::forward_training.
    forward = zendnn_forward,
    /// Backward propagation (with respect to all parameters).
    backward = zendnn_backward,
    /// Backward data propagation.
    backward_data = zendnn_backward_data,
    /// Backward weights propagation.
    backward_weights = zendnn_backward_weights,
    /// Backward bias propagation.
    backward_bias = zendnn_backward_bias
};

/// Converts propagation kind enum value from C++ API to C API type.
///
/// @param akind C++ API propagation kind enum value.
/// @returns Corresponding C API propagation kind enum value.
inline zendnn_prop_kind_t convert_to_c(prop_kind akind) {
    return static_cast<zendnn_prop_kind_t>(akind);
}

/// Kinds of algorithms.
enum class algorithm {
    /// Undefined algorithm
    undef = zendnn_alg_kind_undef,
    /// Convolution algorithm that is chosen to be either direct or Winograd
    /// automatically
    convolution_auto = zendnn_convolution_auto,
    /// Reference convolution
    convolution_ref = zendnn_convolution_ref,
    /// Composable Kernel convolution
    convolution_ck = zendnn_convolution_ck,
    /// GEMM convolution
    convolution_gemm = zendnn_convolution_gemm,
    convolution_gemm_bf16bf16f32of32 = zendnn_convolution_gemm_bf16bf16f32of32,
    convolution_gemm_bf16bf16f32obf16 = zendnn_convolution_gemm_bf16bf16f32obf16,
    convolution_gemm_u8s8s16os16 = zendnn_convolution_gemm_u8s8s16os16,
    convolution_gemm_u8s8s16os8 = zendnn_convolution_gemm_u8s8s16os8,
    convolution_gemm_u8s8s16ou8 = zendnn_convolution_gemm_u8s8s16ou8,
    convolution_gemm_u8s8s32os32 = zendnn_convolution_gemm_u8s8s32os32,
    convolution_gemm_u8s8s32os8 = zendnn_convolution_gemm_u8s8s32os8,
    convolution_gemm_s8s8s32os32 = zendnn_convolution_gemm_s8s8s32os32,
    convolution_gemm_s8s8s32os8 = zendnn_convolution_gemm_s8s8s32os8,
    convolution_gemm_s8s8s16os16 = zendnn_convolution_gemm_s8s8s16os16,
    convolution_gemm_s8s8s16os8 = zendnn_convolution_gemm_s8s8s16os8,
    /// Direct convolution
    convolution_direct = zendnn_convolution_direct,
    /// Winograd convolution
    convolution_winograd = zendnn_convolution_winograd,
    /// Direct deconvolution
    deconvolution_direct = zendnn_deconvolution_direct,
    /// Winograd deconvolution
    deconvolution_winograd = zendnn_deconvolution_winograd,
    /// Elementwise: rectified linear unit (ReLU)
    eltwise_relu = zendnn_eltwise_relu,
    /// Elementwise: hyperbolic tangent non-linearity (tanh)
    eltwise_tanh = zendnn_eltwise_tanh,
    /// Elementwise: exponential linear unit (ELU)
    eltwise_elu = zendnn_eltwise_elu,
    /// Elementwise: square
    eltwise_square = zendnn_eltwise_square,
    /// Elementwise: abs
    eltwise_abs = zendnn_eltwise_abs,
    /// Elementwise: square root
    eltwise_sqrt = zendnn_eltwise_sqrt,
    /// Elementwise: swish (\f$x \cdot sigmoid(a \cdot x)\f$)
    eltwise_swish = zendnn_eltwise_swish,
    /// Elementwise: linear
    eltwise_linear = zendnn_eltwise_linear,
    /// Elementwise: bounded_relu
    eltwise_bounded_relu = zendnn_eltwise_bounded_relu,
    /// Elementwise: soft_relu
    eltwise_soft_relu = zendnn_eltwise_soft_relu,
    /// Elementwise: logsigmoid
    eltwise_logsigmoid = zendnn_eltwise_logsigmoid,
    /// Elementwise: mish
    eltwise_mish = zendnn_eltwise_mish,
    /// Elementwise: logistic
    eltwise_logistic = zendnn_eltwise_logistic,
    /// Elementwise: exponent
    eltwise_exp = zendnn_eltwise_exp,
    /// Elementwise: gelu
    /// alias for #zendnn::algorithm::eltwise_gelu_tanh
    eltwise_gelu = zendnn_eltwise_gelu,
    /// Elementwise: tanh-based gelu
    eltwise_gelu_tanh = zendnn_eltwise_gelu_tanh,
    /// Elementwise: erf-based gelu
    eltwise_gelu_erf = zendnn_eltwise_gelu_erf,
    /// Elementwise: natural logarithm
    eltwise_log = zendnn_eltwise_log,
    /// Elementwise: clip
    eltwise_clip = zendnn_eltwise_clip,
    /// Eltwise: clip version 2
    eltwise_clip_v2 = zendnn_eltwise_clip_v2,
    /// Elementwise: pow
    eltwise_pow = zendnn_eltwise_pow,
    /// Elementwise: round
    eltwise_round = zendnn_eltwise_round,
    /// Elementwise: hardswish
    eltwise_hardswish = zendnn_eltwise_hardswish,
    /// Elementwise: rectified linar unit (ReLU) (dst for backward)
    eltwise_relu_use_dst_for_bwd = zendnn_eltwise_relu_use_dst_for_bwd,
    /// Elementwise: hyperbolic tangent non-linearity (tanh) (dst for backward)
    eltwise_tanh_use_dst_for_bwd = zendnn_eltwise_tanh_use_dst_for_bwd,
    /// Elementwise: exponential linear unit (ELU) (dst for backward)
    eltwise_elu_use_dst_for_bwd = zendnn_eltwise_elu_use_dst_for_bwd,
    /// Elementwise: square root (dst for backward)
    eltwise_sqrt_use_dst_for_bwd = zendnn_eltwise_sqrt_use_dst_for_bwd,
    /// Elementwise: logistic (dst for backward)
    eltwise_logistic_use_dst_for_bwd = zendnn_eltwise_logistic_use_dst_for_bwd,
    /// Elementwise: exponent (dst for backward)
    eltwise_exp_use_dst_for_bwd = zendnn_eltwise_exp_use_dst_for_bwd,
    /// Elementwise: clip version 2 (dst for backward)
    eltwise_clip_v2_use_dst_for_bwd = zendnn_eltwise_clip_v2_use_dst_for_bwd,
    /// Local response normalization (LRN) across multiple channels
    lrn_across_channels = zendnn_lrn_across_channels,
    /// LRN within a single channel
    lrn_within_channel = zendnn_lrn_within_channel,
    /// Max pooling
    pooling_max = zendnn_pooling_max,
    /// Average pooling exclude padding,
    /// alias for #zendnn::algorithm::pooling_avg_exclude_padding
    pooling_avg = zendnn_pooling_avg,
    /// Average pooling include padding
    pooling_avg_include_padding = zendnn_pooling_avg_include_padding,
    /// Average pooling exclude padding
    pooling_avg_exclude_padding = zendnn_pooling_avg_exclude_padding,
    /// RNN cell
    vanilla_rnn = zendnn_vanilla_rnn,
    /// LSTM cell
    vanilla_lstm = zendnn_vanilla_lstm,
    /// GRU cell
    vanilla_gru = zendnn_vanilla_gru,
    /// GRU cell with linear before reset. Differs from the vanilla GRU
    /// in how the new memory gate is calculated:
    /// \f$c_t = tanh(W_c*x_t + b_{c_x} + r_t*(U_c*h_{t-1}+b_{c_h})) \f$
    /// LRB GRU expects 4 bias tensors on input:
    /// \f$[b_{u}, b_{r}, b_{c_x}, b_{c_h}]\f$
    lbr_gru = zendnn_lbr_gru,
    /// AUGRU cell
    vanilla_augru = zendnn_vanilla_augru,
    /// AUGRU cell with linear before reset
    lbr_augru = zendnn_lbr_augru,
    /// Binary add
    binary_add = zendnn_binary_add,
    /// Binary mul
    binary_mul = zendnn_binary_mul,
    /// Binary max
    binary_max = zendnn_binary_max,
    /// Binary min
    binary_min = zendnn_binary_min,
    /// Binary div
    binary_div = zendnn_binary_div,
    /// Binary sub
    binary_sub = zendnn_binary_sub,
    /// Binary greater than or equal
    binary_ge = zendnn_binary_ge,
    /// Binary greater than
    binary_gt = zendnn_binary_gt,
    /// Binary less than or equal
    binary_le = zendnn_binary_le,
    /// Binary less than
    binary_lt = zendnn_binary_lt,
    /// Binary equal
    binary_eq = zendnn_binary_eq,
    /// Binary not equal
    binary_ne = zendnn_binary_ne,
    /// Nearest Neighbor resampling method
    resampling_nearest = zendnn_resampling_nearest,
    /// Linear (Bilinear, Trilinear) resampling method
    resampling_linear = zendnn_resampling_linear,
    /// Reduction using max operation
    reduction_max = zendnn_reduction_max,
    /// Reduction using min operation
    reduction_min = zendnn_reduction_min,
    /// Reduction using sum operation
    reduction_sum = zendnn_reduction_sum,
    /// Reduction using mul operation
    reduction_mul = zendnn_reduction_mul,
    /// Reduction using mean operation
    reduction_mean = zendnn_reduction_mean,
    /// Reduction using norm_lp_max operation
    reduction_norm_lp_max = zendnn_reduction_norm_lp_max,
    /// Reduction using norm_lp_sum operation
    reduction_norm_lp_sum = zendnn_reduction_norm_lp_sum,
    /// Reduction using norm_lp_power_p_max operation
    reduction_norm_lp_power_p_max = zendnn_reduction_norm_lp_power_p_max,
    /// Reduction using norm_lp_power_p_sum operation
    reduction_norm_lp_power_p_sum = zendnn_reduction_norm_lp_power_p_sum,
    /// Softmax, numerically stable
    softmax_accurate = zendnn_softmax_accurate,
    /// LogSoftmax, numerically stable
    softmax_log = zendnn_softmax_log,
    /* add new primitive */
    embedding_bag_sum  = zendnn_embedding_bag_sum,
    embedding_bag_mean = zendnn_embedding_bag_mean,
    embedding_bag_max  = zendnn_embedding_bag_max,
    /* transformer attention variants */
    multihead_attention  = zendnn_multihead_attention,
    multihead_attention_flash_v1 = zendnn_multihead_attention_flash_v1,
    multihead_attention_flash_v2 = zendnn_multihead_attention_flash_v2,
    multiquery_attention  = zendnn_multiquery_attention,
    groupedquery_attention  = zendnn_groupedquery_attention,
};



enum class ActivationPostOp {
    NONE,
    RELU,
    SIGMOID,
    TANH,
    GELU_TANH,
    GELU_ERF,
    SILU
};

struct data_types{
    zendnn_data_type_t src_dt, wei_dt, bia_dt, dst_dt;
    //default values
    data_types(zendnn_data_type_t src = zendnn_f32,
               zendnn_data_type_t wei = zendnn_f32,
               zendnn_data_type_t bia = zendnn_f32,
               zendnn_data_type_t dst = zendnn_f32)
        : src_dt(src), wei_dt(wei), bia_dt(bia), dst_dt(dst) {}
};

class zendnn_custom_op {

  public:
//Embedding bag op API
    static void zendnn_embedding_bag(const memory &z_input,
                                     const memory &z_indices,
                                     const memory &z_offsets,
                                     const bool &z_scale_grad_by_freq,
                                     const algorithm &z_mode, const bool &z_sparse,
                                     const memory &z_per_sample_weights_opt,
                                     const bool &z_per_sample_weights_defined,
                                     const bool &z_include_last_offset, const int32_t &z_padding_idx,
                                     memory &z_destination, const char *plugin_op="", int thread_qty=1,
                                     const bool &scale_bias_last=true);

//Group embedding bag op API
    static void zendnn_grp_embedding_bag(std::vector <memory> &z_input,
                                         std::vector <memory> &z_indices,std::vector <memory> &z_offsets,
                                         std::vector <int32_t> &z_scale_grad_by_freq, std::vector <algorithm> &z_modes,
                                         std::vector <int32_t> &z_sparse, std::vector <memory> &z_per_sample_weights_opt,
                                         std::vector <int32_t> &z_per_sample_weights_defined,
                                         std::vector <int32_t> &z_include_last_offset,
                                         std::vector <int32_t> &z_padding_idx,
                                         std::vector <memory> &z_destination, const char *plugin_op="",
                                         const int &cat_dim=-1, const int &mlp_pos=-1, const int &output_stride=-1,
                                         int thread_qty=1, const bool &scale_bias_last=true);

//Embedding op API
    static void zendnn_embedding(const memory &z_input, const memory &z_indices,
                                 const int32_t &z_padding_idx, const bool &z_scale_grad_by_freq,
                                 const bool  &z_sparse,
                                 memory &z_destination, const char *plugin_op="", int thread_qty=1,
                                 const bool &scale_bias_last=true);

//Group Embedding op API
    static void zendnn_grp_embedding(std::vector <memory> &z_input,
                                     std::vector <memory> &z_indices,
                                     std::vector <int32_t> &z_padding_idx,
                                     std::vector <int32_t> &z_scale_grad_by_freq,
                                     std::vector <int32_t> &z_sparse,
                                     std::vector <memory> &z_destination, const char *plugin_op="", int thread_qty=1,
                                     const bool &scale_bias_last=true);

//Group MLP op API
    static void zendnn_grp_mlp(const std::vector<memory> &z_input,
                               const std::vector<memory> &z_weight,
                               const std::vector<memory> &z_bias,
                               const std::vector<float> &z_alpha,
                               const std::vector<float> &z_beta,
                               const std::vector<bool> &z_bias_defined,
                               const std::vector<std::vector<int64_t>> &z_post_op_ids,
                               const std::vector<std::vector<memory>> &z_post_op_buffers,
                               const std::vector<memory> &z_result, const char *plugin_op="");
//Group Embedding_Bag and MLP op API
    static void zendnn_grp_ebag_mlp(std::vector <memory> &z_eb_input,
                                    std::vector <memory> &z_eb_indices, std::vector <memory> &z_eb_offsets,
                                    std::vector <int32_t> &z_eb_scale_grad_by_freq,
                                    std::vector <algorithm> &z_eb_modes,
                                    std::vector <int32_t> &z_eb_sparse,
                                    std::vector <memory> &z_eb_per_sample_weights_opt,
                                    std::vector <int32_t> &z_eb_per_sample_weights_defined,
                                    std::vector <int32_t> &z_eb_include_last_offset,
                                    std::vector <int32_t> &z_eb_padding_idx,
                                    std::vector <memory> &z_eb_destination,
                                    const std::vector<memory> &z_mm_input,
                                    const std::vector<memory> &z_mm_weight,
                                    const std::vector<memory> &z_mm_bias,
                                    const std::vector<float> &z_mm_alpha,
                                    const std::vector<float> &z_mm_beta,
                                    const std::vector<bool> &z_mm_bias_defined,
                                    const std::vector<std::vector<int64_t>> &z_post_op_ids,
                                    const std::vector<std::vector<memory>> &z_post_op_buffers,
                                    const std::vector<memory> &z_mm_result, const char *plugin_op="");

//Group Embedding and MLP op API
    static void zendnn_grp_embedding_mlp(std::vector <memory> &z_embed_input,
                                         std::vector <memory> &z_embed_indices,
                                         std::vector <int32_t> &z_embed_scale_grad_by_freq,
                                         std::vector <int32_t> &z_embed_sparse,
                                         std::vector <int32_t> &z_embed_padding_idx,
                                         std::vector <memory> &z_embed_destination,
                                         const std::vector<memory> &z_mm_input,
                                         const std::vector<memory> &z_mm_weight,
                                         const std::vector<memory> &z_mm_bias,
                                         const std::vector<float> &z_mm_alpha,
                                         const std::vector<float> &z_mm_beta,
                                         const std::vector<bool> &z_mm_bias_defined,
                                         const std::vector<std::vector<int64_t>> &z_post_op_ids,
                                         const std::vector<std::vector<memory>> &z_post_op_buffers,
                                         const std::vector<memory> &z_mm_result);
    //SDPA OP API
    static void zendnn_sdpa_attention(
        const memory &input_Q_mem, const memory &input_K_mem,
        const memory &input_V_mem,
        memory &input_mask_mem,
        memory &output_mem);

    // ZenDNN Reorder API
    static bool zendnn_reorder(void *src, void *dst, uint k, uint n, bool trans,
                               zendnn_data_type_t weight_dt,
                               zendnn_data_type_t src_dt = zendnn_u8, int src_zp = 0,
                               bool is_resized = false);

    // ZenDNN Reorder Size API
    static size_t zendnn_reorder_size(uint k, uint n, bool trans,
                                      zendnn_data_type_t src_dt,
                                      int src_zp, zendnn_data_type_t wei_dt);

    static size_t matmul_direct_select_kernel(int M, int N, int K);
    static void zendnn_batched_matmul_fp32(const std::vector<float *> &src_batch,
                                           const std::vector<float *> &weight_batch,
                                           std::vector<float *> &dst_batch,
                                           const std::vector<float *> &bias_batch,
                                           const std::vector<float> &alpha_array,
                                           const std::vector<float> &beta_array,
                                           const std::vector<int> &m_array,
                                           const std::vector<int> &n_array,
                                           const std::vector<int> &k_array,
                                           const std::vector<bool> &transB_array,
                                           const std::vector<ActivationPostOp> &post_op_array,
                                           int group_count,
                                           const std::vector<int> &group_size_array);

    static void zendnn_matmul_direct_fp32(const void *src, const void *weight,
                                          void *dst, const void *bias,  float alpha, float beta,
                                          int M, int N, int K, bool transA, bool transB, int lda, int ldb, int ldc, data_types dt,
                                          ActivationPostOp post_op,
                                          int Batch_A=1, int Batch_B=1);


    static void quantize_bf16_to_int8(const void *input_bf16, void *output_int8,
                                      size_t count, float scale, int zero_point);
    static void dequantize_int8_to_bf16(const void *input_int8, void *output_bf16,
                                        size_t count, float scale, int zero_point);
};

/// Converts algorithm kind enum value from C++ API to C API type.
/// @param aalgorithm C++ API algorithm kind enum value.
/// @returns Corresponding C API algorithm kind enum value.
inline zendnn_alg_kind_t convert_to_c(algorithm aalgorithm) {
    return static_cast<zendnn_alg_kind_t>(aalgorithm);
}

/// @} zendnn_api_attributes

/// @addtogroup zendnn_api_primitives_common
/// @{

/// Flags for normalization primitives.
enum class normalization_flags : unsigned {
    /// Use no normalization flags. If specified, the library computes mean and
    /// variance on forward propagation for training and inference, outputs them
    /// on forward propagation for training, and computes the respective
    /// derivatives on backward propagation.
    none = zendnn_normalization_flags_none,

    /// Use global statistics. If specified, the library uses mean and
    /// variance provided by the user as an input on forward propagation and
    /// does not compute their derivatives on backward propagation. Otherwise,
    /// the library computes mean and variance on forward propagation for
    /// training and inference, outputs them on forward propagation for
    /// training, and computes the respective derivatives on backward
    /// propagation.
    use_global_stats = zendnn_use_global_stats,

    /// Use scale and shift parameters. If specified, the user is expected to
    /// pass scale and shift as inputs on forward propagation. On backward
    /// propagation of type #zendnn::prop_kind::backward, the library computes
    /// their derivatives. If not specified, the scale and shift parameters
    /// are not used by the library in any way.
    use_scale_shift = zendnn_use_scaleshift,

    /// Fuse normalization with ReLU. On training, normalization will require
    /// the workspace to implement backward propagation. On inference, the
    /// workspace is not required and behavior is the same as when normalization
    /// is fused with ReLU using the post-ops API.
    fuse_norm_relu = zendnn_fuse_norm_relu,

    /// Use scale parameter. If specified, the user is expected to pass scale as
    /// input on forward propagation. On backward propagation of type
    /// #zendnn::prop_kind::backward, the library computes its derivative.
    use_scale = zendnn_use_scale,

    /// Use shift parameter. If specified, the user is expected to pass shift as
    /// input on forward propagation. On backward propagation of type
    /// #zendnn::prop_kind::backward, the library computes its derivative.
    use_shift = zendnn_use_shift,
};

/// Converts normalization flags enum value from C++ API to C API type.
/// @param flags C++ API normalization flags enum value.
/// @returns Corresponding C API normalization flags enum value.
inline zendnn_normalization_flags_t convert_to_c(normalization_flags flags) {
    return static_cast<zendnn_normalization_flags_t>(flags);
}

/// @} zendnn_api_primitives_common

/// @addtogroup zendnn_api_rnn
/// @{

/// RNN cell flags.
enum class rnn_flags : unsigned {
    /// Undefined RNN flags
    undef = zendnn_rnn_flags_undef
};

/// Converts RNN cell flags enum value from C++ API to C API type.
/// @param flags C++ API RNN cell flags enum value.
/// @returns Corresponding C API RNN cell flags enum value.
inline zendnn_rnn_flags_t convert_to_c(rnn_flags flags) {
    return static_cast<zendnn_rnn_flags_t>(flags);
}

#define ZENDNN_DEFINE_BITMASK_OPS(enum_name) \
    inline enum_name operator|(enum_name lhs, enum_name rhs) { \
        return static_cast<enum_name>( \
                static_cast<unsigned>(lhs) | static_cast<unsigned>(rhs)); \
    } \
\
    inline enum_name operator&(enum_name lhs, enum_name rhs) { \
        return static_cast<enum_name>( \
                static_cast<unsigned>(lhs) & static_cast<unsigned>(rhs)); \
    } \
\
    inline enum_name operator^(enum_name lhs, enum_name rhs) { \
        return static_cast<enum_name>( \
                static_cast<unsigned>(lhs) ^ static_cast<unsigned>(rhs)); \
    } \
\
    inline enum_name &operator|=(enum_name &lhs, enum_name rhs) { \
        lhs = static_cast<enum_name>( \
                static_cast<unsigned>(lhs) | static_cast<unsigned>(rhs)); \
        return lhs; \
    } \
\
    inline enum_name &operator&=(enum_name &lhs, enum_name rhs) { \
        lhs = static_cast<enum_name>( \
                static_cast<unsigned>(lhs) & static_cast<unsigned>(rhs)); \
        return lhs; \
    } \
\
    inline enum_name &operator^=(enum_name &lhs, enum_name rhs) { \
        lhs = static_cast<enum_name>( \
                static_cast<unsigned>(lhs) ^ static_cast<unsigned>(rhs)); \
        return lhs; \
    } \
\
    inline enum_name operator~(enum_name rhs) { \
        return static_cast<enum_name>(~static_cast<unsigned>(rhs)); \
    }

ZENDNN_DEFINE_BITMASK_OPS(normalization_flags)
ZENDNN_DEFINE_BITMASK_OPS(rnn_flags)

/// A direction of RNN primitive execution
enum class rnn_direction {
    /// Unidirectional execution of RNN primitive from left to right.
    unidirectional_left2right = zendnn_unidirectional_left2right,
    /// Unidirectional execution of RNN primitive from right to left.
    unidirectional_right2left = zendnn_unidirectional_right2left,
    /// Bidirectional execution of RNN primitive with concatenation of the
    /// results.
    bidirectional_concat = zendnn_bidirectional_concat,
    /// Bidirectional execution of RNN primitive with summation of the
    /// results.
    bidirectional_sum = zendnn_bidirectional_sum,
    /// Alias for #zendnn::rnn_direction::unidirectional_left2right
    unidirectional = zendnn_unidirectional,
};

/// Converts RNN direction enum value from C++ API to C API type.
/// @param dir C++ API RNN direction enum value.
/// @returns Corresponding C API RNN direction enum value.
inline zendnn_rnn_direction_t convert_to_c(rnn_direction dir) {
    return static_cast<zendnn_rnn_direction_t>(dir);
}

/// @} zendnn_api_rnn

/// @addtogroup zendnn_api_primitives_common
/// @{

/// Primitive descriptor query specification.
///
/// In general, queries are not used with the C++ API because most queries are
/// implemented as class members.
///
/// See @ref zendnn_query_t for more information.
enum class query {
    /// no query
    undef = zendnn_query_undef,

    /// execution engine
    engine = zendnn_query_engine,
    /// primitive kind
    primitive_kind = zendnn_query_primitive_kind,

    /// number of inputs expected
    num_of_inputs_s32 = zendnn_query_num_of_inputs_s32,
    /// number of outputs expected
    num_of_outputs_s32 = zendnn_query_num_of_outputs_s32,

    /// runtime estimation (seconds), unimplemented
    time_estimate_f64 = zendnn_query_time_estimate_f64,
    /// memory required for scratchpad (bytes)
    ///
    /// @sa @ref dev_guide_attributes_scratchpad
    memory_consumption_s64 = zendnn_query_memory_consumption_s64,

    /// scratchpad engine
    ///
    /// engine to be used for creating scratchpad memory
    scratchpad_engine = zendnn_query_scratchpad_engine,

    /// reorder source engine
    reorder_src_engine = zendnn_query_reorder_src_engine,
    /// reorder destination engine
    reorder_dst_engine = zendnn_query_reorder_dst_engine,

    /// implementation name
    impl_info_str = zendnn_query_impl_info_str,

    /// propagation kind
    prop_kind = zendnn_query_prop_kind,

    /// size of cache blob ID in bytes
    cache_blob_id_size_s64 = zendnn_query_cache_blob_id_size_s64,

    /// cache blob ID (pointer to array)
    cache_blob_id = zendnn_query_cache_blob_id,

    /// operation descriptor
    op_d = zendnn_query_op_d,
    /// convolution descriptor
    convolution_d = zendnn_query_convolution_d,
    /// deconvolution descriptor
    deconvolution_d = zendnn_query_deconvolution_d,
    /// shuffle descriptor
    shuffle_d = zendnn_query_shuffle_d,
    /// eltwise descriptor
    eltwise_d = zendnn_query_eltwise_d,
    /// softmax descriptor
    softmax_d = zendnn_query_softmax_d,
    /// pooling descriptor
    pooling_d = zendnn_query_pooling_d,
    /// lrn descriptor
    lrn_d = zendnn_query_lrn_d,
    /// batch normalization descriptor
    batch_normalization_d = zendnn_query_batch_normalization_d,
    /// layer normalization descriptor
    layer_normalization_d = zendnn_query_layer_normalization_d,
    /// inner product descriptor
    inner_product_d = zendnn_query_inner_product_d,
    /// rnn descriptor
    rnn_d = zendnn_query_rnn_d,
    /// binary descriptor
    binary_d = zendnn_query_binary_d,
    /// logsoftmax descriptor
    logsoftmax_d = zendnn_query_logsoftmax_d,
    /// matmul descriptor
    matmul_d = zendnn_query_matmul_d,
    /// resampling descriptor
    resampling_d = zendnn_query_resampling_d,
    /// reduction descriptor
    reduction_d = zendnn_query_reduction_d,

    /// source memory desc
    src_md = zendnn_query_src_md,
    /// source gradient (diff) memory desc
    diff_src_md = zendnn_query_diff_src_md,
    /// weights memory descriptor desc
    weights_md = zendnn_query_weights_md,
    /// weights gradient (diff) memory desc
    diff_weights_md = zendnn_query_diff_weights_md,
    /// destination memory desc
    dst_md = zendnn_query_dst_md,
    /// destination gradient (diff) memory desc
    diff_dst_md = zendnn_query_diff_dst_md,
    /// workspace memory desc
    workspace_md = zendnn_query_workspace_md,
    /// scratchpad memory desc
    scratchpad_md = zendnn_query_scratchpad_md,
    /// memory desc of an execute argument
    exec_arg_md = zendnn_query_exec_arg_md,
};

/// Converts query enum value from C++ API to C API type.
/// @param aquery C++ API query enum value.
/// @returns Corresponding C API query enum value.
inline zendnn_query_t convert_to_c(query aquery) {
    return static_cast<zendnn_query_t>(aquery);
}

/// @} zendnn_api_primitives_common

/// @} zendnn_api_primitives

/// @addtogroup zendnn_api_engine Engine
///
/// An abstraction of a computational device: a CPU, a specific GPU
/// card in the system, etc. Most primitives are created to execute
/// computations on one specific engine. The only exceptions are reorder
/// primitives that transfer data between two different engines.
///
/// @sa @ref dev_guide_basic_concepts
///
/// @{

/// @cond DO_NOT_DOCUMENT_THIS
template <>
struct handle_traits<zendnn_engine_t> {
    static zendnn_status_t destructor(zendnn_engine_t p) {
        return zendnn_engine_destroy(p);
    }
};
/// @endcond

/// An execution engine.
struct engine : public handle<zendnn_engine_t> {
    friend struct primitive;
    friend struct reorder;

    /// Kinds of engines.
    enum class kind {
        /// An unspecified engine
        any = zendnn_any_engine,
        /// CPU engine
        cpu = zendnn_cpu,
        /// GPU engine
        gpu = zendnn_gpu,
    };

    using handle::handle;

    /// Constructs an empty engine. An empty engine cannot be used in any
    /// operations.
    engine() = default;

    /// Returns the number of engines of a certain kind.
    ///
    /// @param akind The kind of engines to count.
    /// @returns The number of engines of the specified kind.
    static size_t get_count(kind akind) {
        return zendnn_engine_get_count(convert_to_c(akind));
    }

    /// Constructs an engine.
    ///
    /// @param akind The kind of engine to construct.
    /// @param index The index of the engine. Must be less than the value
    ///     returned by #get_count() for this particular kind of engine.
    engine(kind akind, size_t index) {
        zendnn_engine_t engine;
        zendnnInfo(ZENDNN_APILOG, "CPU Engine create");
        error::wrap_c_api(
            zendnn_engine_create(&engine, convert_to_c(akind), index),
            "could not create an engine");
        reset(engine);
    }

    /// Constructs an engine based on a primitive from the primitive
    /// descriptor @p pd by querying its engine.
    ///
    /// @param pd The primitive descriptor to query.
    engine(const handle<zendnn_primitive_desc_t> &pd) {
        zendnn_engine_t c_engine;
        error::wrap_c_api(
            zendnn_primitive_desc_query(pd.get(),
                                        zendnn::convert_to_c(zendnn::query::engine), 0, &c_engine),
            "could not get an engine from a primitive_desc");
        reset(c_engine, true);
    }

    /// Returns the kind of the engine.
    /// @returns The kind of the engine.
    kind get_kind() const {
        zendnn_engine_kind_t kind;
        error::wrap_c_api(zendnn_engine_get_kind(get(), &kind),
                          "could not get kind of an engine");
        return static_cast<engine::kind>(kind);
    }

    /// Returns the engine of a primitive descriptor.
    ///
    /// @param pd The primitive descriptor to query.
    /// @returns A weak handle to the engine that the primitive descriptor was
    ///     created with.
    template <typename primitive_desc>
    static engine query(const primitive_desc &pd) {
        return query(pd, zendnn::query::engine);
    }

  private:
    static zendnn_engine_kind_t convert_to_c(kind akind) {
        return static_cast<zendnn_engine_kind_t>(akind);
    }

    template <typename primitive_desc>
    static engine query(const primitive_desc &pd, zendnn::query what) {
        zendnn_engine_t c_engine;
        error::wrap_c_api(zendnn_primitive_desc_query(pd.get(),
                          zendnn::convert_to_c(what), 0, &c_engine),
                          "could not get an engine from a primitive_desc");
        return engine(c_engine, true);
    }
};

/// Converts engine kind enum value from C++ API to C API type.
///
/// @param akind C++ API engine kind enum value.
/// @returns Corresponding C API engine kind enum value.
inline zendnn_engine_kind_t convert_to_c(engine::kind akind) {
    return static_cast<zendnn_engine_kind_t>(akind);
}

/// @} zendnn_api_engine

/// @addtogroup zendnn_api_stream Stream
///
/// An encapsulation of execution context tied to a particular engine.
///
/// @sa @ref dev_guide_basic_concepts
///
/// @{

/// @cond DO_NOT_DOCUMENT_THIS
template <>
struct handle_traits<zendnn_stream_t> {
    static zendnn_status_t destructor(zendnn_stream_t p) {
        return zendnn_stream_destroy(p);
    }
};
/// @endcond

/// An execution stream.
struct stream : public handle<zendnn_stream_t> {
    using handle::handle;

    /// Stream flags. Can be combined using the bitwise OR operator.
    enum class flags : unsigned {
        /// In-order execution.
        in_order = zendnn_stream_in_order,
        /// Out-of-order execution.
        out_of_order = zendnn_stream_out_of_order,
        /// Default stream configuration.
        default_flags = zendnn_stream_default_flags,
    };

    /// Constructs an empty stream. An empty stream cannot be used in any
    /// operations.
    stream() = default;

    /// Constructs a stream for the specified engine and with behavior
    /// controlled by the specified flags.
    ///
    /// @param aengine Engine to create the stream on.
    /// @param aflags Flags controlling stream behavior.
    stream(const engine &aengine, flags aflags = flags::default_flags) {
        zendnn_stream_t stream;
        zendnnInfo(ZENDNN_APILOG, "CPU Stream create");
        error::wrap_c_api(zendnn_stream_create(&stream, aengine.get(),
                                               static_cast<zendnn_stream_flags_t>(aflags)),
                          "could not create a stream");
        reset(stream);
    }

    /// Returns the associated engine.
    engine get_engine() const {
        zendnn_engine_t c_engine;
        error::wrap_c_api(zendnn_stream_get_engine(get(), &c_engine),
                          "could not get an engine from a stream object");
        return engine(c_engine, true);
    }

    /// Waits for all primitives executing in the stream to finish.
    /// @returns The stream itself.
    stream &wait() {
        error::wrap_c_api(
            zendnn_stream_wait(get()), "could not wait on a stream");
        return *this;
    }
};

ZENDNN_DEFINE_BITMASK_OPS(stream::flags)

/// @} zendnn_api_stream

/// @addtogroup zendnn_api_memory Memory
///
/// A container that describes and stores data. Memory objects can contain
/// data of various types and formats. There are two levels of abstraction:
///
/// 1. **Memory descriptor** -- engine-agnostic logical description of data
///     (number of dimensions, dimension sizes, and data type), and,
///     optionally, the information about the physical format of data in
///     memory. If this information is not known yet, a memory descriptor can
///     be created with #zendnn::memory::format_tag::any. This allows
///     compute-intensive primitives to choose the best format for
///     computation. The user is responsible for reordering the data into the
///     chosen format when formats do not match.
///
///     A memory descriptor can be initialized either by specifying dimensions
///     and a memory format tag or strides for each of them, or by
///     manipulating the zendnn_memory_desc_t structure directly.
///
///     @warning
///         The latter approach requires understanding how the physical data
///         representation is mapped to the structure and is discouraged. This
///         topic is discussed in @ref dev_guide_understanding_memory_formats.
///
///     The user can query the amount of memory required by a memory
///     descriptor using the #zendnn::memory::desc::get_size() function. The
///     size of data in general cannot be computed as the product of
///     dimensions multiplied by the size of the data type. So users are
///     required to use this function for better code portability.
///
///     Two memory descriptors can be compared using the equality and
///     inequality operators.  The comparison is especially useful when
///     checking whether it is necessary to reorder data from the user's data
///     format to a primitive's format.
///
/// 2. **Memory object** -- an engine-specific object that handles the memory
///     buffer and its description (a memory descriptor). For the CPU engine or
///     with USM, the memory buffer handle is simply a pointer to @c void. The
///     memory buffer can be queried using #zendnn::memory::get_data_handle() and
///     set using #zendnn::memory::set_data_handle(). The underlying SYCL buffer,
///     when used, can be queried using #zendnn::sycl_interop::get_buffer and set
///     using #zendnn::sycl_interop::set_buffer. A memory object can also be
///     queried for the underlying memory descriptor and for its engine using
///     #zendnn::memory::get_desc() and zendnn::memory::get_engine().
///
/// Along with ordinary memory descriptors with all dimensions being positive,
/// the library supports *zero-volume*  memory descriptors with one or more
/// dimensions set to zero. This is used to support the NumPy\* convention.
/// If a zero-volume memory is passed to a primitive, the primitive typically
/// does not perform any computations with this memory. For example:
///
/// - A concatenation primitive would ignore all memory object with zeroes in
///   the concat dimension / axis.
///
/// - A forward convolution with a source memory object with zero in the
///   minibatch dimension would always produce a destination memory object
///   with a zero in the minibatch dimension and perform no computations.
///
/// - However, a forward convolution with a zero in one of the weights
///   dimensions is ill-defined and is considered to be an error by the
///   library because there is no clear definition of what the output values
///   should be.
///
/// Memory buffer of a zero-volume memory is never accessed.
///
/// @{

/// Memory object.
///
/// A memory object encapsulates a handle to a memory buffer allocated on a
/// specific engine, tensor dimensions, data type, and memory format, which is
/// the way tensor indices map to offsets in linear memory space. Memory
/// objects are passed to primitives during execution.
struct memory : public handle<zendnn_memory_t> {
    using handle::handle;

    /// Integer type for representing dimension sizes and indices.
    typedef zendnn_dim_t dim;
    /// Vector of dimensions. Implementations are free to force a limit on the
    /// vector's length.
    typedef std::vector<dim> dims;

    /// Helper function that validates that an `std::vector` of dimensions can
    /// be safely converted to the C API array ::zendnn_dims_t. Throws if
    /// validation fails.
    ///
    /// @param v Vector of dimensions.
    /// @param min_size Minimum expected size of the vector.
    template <typename T>
    static void validate_dims(const std::vector<T> &v, int min_size = 0) {
        validate_container_size(
            v, "dimensions are invalid", min_size, ZENDNN_MAX_NDIMS);
    }

    /// Data type specification.
    enum class data_type {
        /// Undefined data type (used for empty memory descriptors).
        undef = zendnn_data_type_undef,
        /// [16-bit/half-precision floating point].
        f16 = zendnn_f16,
        /// non-standard
        /// [16-bit floating point with 7-bit mantissa].
        bf16 = zendnn_bf16,
        /// [32-bit/single-precision floating point].
        f32 = zendnn_f32,
        /// 32-bit signed integer.
        s32 = zendnn_s32,
        s16 = zendnn_s16,
        /// 8-bit signed integer.
        s8 = zendnn_s8,
        /// 8-bit unsigned integer.
        u8 = zendnn_u8,
        /// 4-bit signed integer.
        s4 = zendnn_s4,
        /// 4-bit unsigned integer.
        u4 = zendnn_u4,
    };

    /// Returns size of data type in bytes.
    /// @returns The number of bytes occupied by data type.
    static size_t data_type_size(data_type adata_type) {
        return zendnn_data_type_size(convert_to_c(adata_type));
    }

    /// Memory format kind
    enum class format_kind {
        /// Undefined memory format kind, used for empty memory descriptors.
        undef = zendnn_format_kind_undef,
        /// Unspecified format kind.
        /// The primitive selects a format automatically.
        any = zendnn_format_kind_any,
        /// A tensor in a generic format described by the stride and blocking
        /// values in each dimension. See @ref zendnn_blocking_desc_t for more
        /// information.
        blocked = zendnn_blocked,
        /// Weights format used in 8-bit Winograd convolution.
        wino = zendnn_format_kind_wino,
        /// Packed weights format used in RNN.
        packed = zendnn_format_kind_rnn_packed,
    };

    /// Memory format tag specification.
    ///
    /// Memory format tags can be further divided into two categories:
    ///
    ///  - Domain-agnostic names, i.e. names that do not depend on the tensor
    ///    usage in the specific primitive. These names use letters from `a`
    ///    to `f` to denote logical dimensions and form the order in which the
    ///    dimensions are laid in memory. For example,
    ///    #zendnn::memory::format_tag::ab is used to denote a 2D tensor where the
    ///    second logical dimension (denoted as `b`) is the innermost, i.e.
    ///    has stride = 1, and the first logical dimension (`a`) is laid out in
    ///    memory with stride equal to the size of the second dimension. On the
    ///    other hand, #zendnn::memory::format_tag::ba is the transposed version
    ///    of the same tensor: the outermost dimension (`a`) becomes the
    ///    innermost one.
    ///
    ///  - Domain-specific names, i.e. names that make sense only in the
    ///    context of a certain domain, such as CNN. These names are
    ///    aliases to the corresponding domain-agnostic tags and used mostly
    ///    for convenience. For example, #zendnn::memory::format_tag::nc
    ///    is used to denote 2D CNN activations tensor memory format, where
    ///    the channels dimension is the innermost one and the batch dimension
    ///    is the outermost one. Moreover, #zendnn::memory::format_tag::nc is
    ///    an alias for #zendnn::memory::format_tag::ab, because for
    ///    CNN primitives the logical dimensions of activations tensors come
    ///    in order: batch, channels, spatial.  In other words, batch
    ///    corresponds to the first logical dimension (`a`), and channels
    ///    correspond to the second one (`b`).
    ///
    /// The following domain-specific notation applies to memory format tags:
    ///  - @c 'n' denotes the mini-batch dimension
    ///  - @c 'c' denotes a channels dimension
    ///  - When there are multiple channel dimensions (for example,
    ///    in convolution weights tensor), @c 'i' and @c 'o' denote dimensions
    ///    of input and output channels
    ///  - @c 'g' denotes a groups dimension for convolution weights
    ///  - @c 'd', @c 'h', and @c 'w' denote spatial depth, height, and width
    ///    respectively
    ///
    /// See @ref zendnn_format_tag_t for a detailed description.
    enum class format_tag {
        /// Undefined memory format tag
        undef = zendnn_format_tag_undef,
        /// Placeholder memory format tag. Used to instruct the primitive to
        /// select a format automatically.
        any = zendnn_format_tag_any,

        /// plain 1D tensor
        a = zendnn_a,

        /// plain 2D tensor
        ab = zendnn_ab,
        /// permuted 2D tensor
        ba = zendnn_ba,

        /// plain 3D tensor
        abc = zendnn_abc,
        /// permuted 3D tensor
        acb = zendnn_acb,
        /// permuted 3D tensor
        bac = zendnn_bac,
        /// permuted 3D tensor
        bca = zendnn_bca,
        /// permuted 3D tensor
        cba = zendnn_cba,

        /// plain 4D tensor
        abcd = zendnn_abcd,
        /// permuted 4D tensor
        abdc = zendnn_abdc,
        /// permuted 4D tensor
        acbd = zendnn_acbd,
        /// permuted 4D tensor
        acdb = zendnn_acdb,
        /// permuted 4D tensor
        adbc = zendnn_adbc,
        /// permuted 4D tensor
        bacd = zendnn_bacd,
        /// permuted 4D tensor
        bcda = zendnn_bcda,
        /// permuted 4D tensor
        cdba = zendnn_cdba,
        /// permuted 4D tensor
        dcab = zendnn_dcab,

        /// plain 5D tensor
        abcde = zendnn_abcde,
        /// permuted 5D tensor
        abdec = zendnn_abdec,
        /// permuted 5D tensor
        acbde = zendnn_acbde,
        /// permuted 5D tensor
        acdeb = zendnn_acdeb,
        /// permuted 5D tensor
        bacde = zendnn_bacde,
        /// permuted 5D tensor
        bcdea = zendnn_bcdea,
        /// permuted 5D tensor
        cdeba = zendnn_cdeba,
        /// permuted 5D tensor
        decab = zendnn_decab,
        /// permuted 5D tensor
        abced = zendnn_abced,

        /// plain 6D tensor
        abcdef = zendnn_abcdef,
        /// permuted 6D tensor
        abdfce = zendnn_abdfce,
        /// permuted 6D tensor
        acbdef = zendnn_acbdef,
        /// permuted 6D tensor
        abdefc = zendnn_abdefc,
        /// permuted 6D tensor
        defcab = zendnn_defcab,
        /// permuted 6D tensor
        abcdfe = zendnn_abcdfe,

        /// plain 7D tensor
        abcdefg = zendnn_abcdefg,
        /// permuted 7D tensor
        abcdegf = zendnn_abcdegf,

        /// plain 8D tensor
        abcdefgh = zendnn_abcdefgh,
        /// permuted 8D tensor
        abcdefhg = zendnn_abcdefhg,

        /// plain 9D tensor
        abcdefghi = zendnn_abcdefghi,
        /// permuted 9D tensor
        abcdefgih = zendnn_abcdefgih,

        /// plain 10D tensor
        abcdefghij = zendnn_abcdefghij,
        /// permuted 10D tensor
        abcdefghji = zendnn_abcdefghji,

        /// plain 11D tensor
        abcdefghijk = zendnn_abcdefghijk,
        /// permuted 11D tensor
        abcdefghikj = zendnn_abcdefghikj,

        /// plain 12D tensor
        abcdefghijkl = zendnn_abcdefghijkl,
        /// permuted 12D tensor
        abcdefghijlk = zendnn_abcdefghijlk,

        /// 1D tensor; an alias for #zendnn::memory::format_tag::a
        x = a,
        /// 2D CNN activations tensor; an alias for #zendnn::memory::format_tag::ab
        nc = ab,
        /// 2D CNN activations tensor; an alias for #zendnn::memory::format_tag::ba
        cn = ba,
        /// 2D RNN statistics tensor; an alias for #zendnn::memory::format_tag::ab
        tn = ab,
        /// 2D RNN statistics tensor; an alias for #zendnn::memory::format_tag::ba
        nt = ba,
        /// 3D CNN activations tensor; an alias for #zendnn::memory::format_tag::abc
        ncw = abc,
        /// 3D CNN activations tensor; an alias for #zendnn::memory::format_tag::acb
        nwc = acb,
        /// 4D CNN activations tensor; an alias for #zendnn::memory::format_tag::abcd
        nchw = abcd,
        /// 4D CNN activations tensor; an alias for #zendnn::memory::format_tag::acdb
        nhwc = acdb,
        /// 4D CNN activations tensor; an alias for #zendnn::memory::format_tag::bcda
        chwn = bcda,
        /// 5D CNN activations tensor; an alias for #zendnn::memory::format_tag::abcde
        ncdhw = abcde,
        /// 5D CNN activations tensor; an alias for #zendnn::memory::format_tag::acdeb
        ndhwc = acdeb,

        //hwcn format_tag
        hwcn = zendnn_hwcn,

        /// 2D CNN weights tensor; an alias for #zendnn::memory::format_tag::ab
        oi = ab,
        /// 2D CNN weights tensor; an alias for #zendnn::memory::format_tag::ba
        io = ba,
        /// 3D CNN weights tensor; an alias for #zendnn::memory::format_tag::abc
        oiw = abc,
        /// 3D CNN weights tensor; an alias for #zendnn::memory::format_tag::acb
        owi = acb,
        /// 3D CNN weights tensor; an alias for #zendnn::memory::format_tag::cba
        wio = cba,
        /// 3D CNN weights tensor; an alias for #zendnn::memory::format_tag::bca
        iwo = bca,
        /// 4D CNN weights tensor; an alias for #zendnn::memory::format_tag::abcd
        oihw = abcd,
        /// 4D CNN weights tensor; an alias for #zendnn::memory::format_tag::cdba
        hwio = cdba,
        /// 4D CNN weights tensor; an alias for #zendnn::memory::format_tag::acdb
        ohwi = acdb,
        /// 4D CNN weights tensor; an alias for #zendnn::memory::format_tag::bcda
        ihwo = bcda,
        /// 4D CNN weights tensor; an alias for #zendnn::memory::format_tag::bacd
        iohw = bacd,
        /// 5D CNN weights tensor; an alias for #zendnn::memory::format_tag::abcde
        oidhw = abcde,
        /// 5D CNN weights tensor; an alias for #zendnn::memory::format_tag::cdeba
        dhwio = cdeba,
        /// 5D CNN weights tensor; an alias for #zendnn::memory::format_tag::acdeb
        odhwi = acdeb,
        /// 5D CNN weights tensor; an alias for #zendnn::memory::format_tag::bacde
        iodhw = bacde,
        /// 5D CNN weights tensor; an alias for #zendnn::memory::format_tag::bcdea
        idhwo = bcdea,

        /// 4D CNN weights tensor with groups; an alias for #zendnn::memory::format_tag::abcd
        goiw = abcd,
        /// 4D CNN weights tensor with groups; an alias for #zendnn::memory::format_tag::abdc
        gowi = abdc,
        /// 4D CNN weights tensor with groups; an alias for #zendnn::memory::format_tag::dcab
        wigo = dcab,
        /// 5D CNN weights tensor with groups; an alias for #zendnn::memory::format_tag::abdec
        gohwi = abdec,
        /// 5D CNN weights tensor with groups; an alias for #zendnn::memory::format_tag::abcde
        goihw = abcde,
        /// 5D CNN weights tensor with groups; an alias for #zendnn::memory::format_tag::decab
        hwigo = decab,
        /// 5D CNN weights tensor with groups; an alias for #zendnn::memory::format_tag::acbde
        giohw = acbde,
        /// 6D CNN weights tensor with groups; an alias for #zendnn::memory::format_tag::abcdef
        goidhw = abcdef,
        /// 6D CNN weights tensor with groups; an alias for #zendnn::memory::format_tag::abcdef
        giodhw = acbdef,
        /// 6D CNN weights tensor with groups; an alias for #zendnn::memory::format_tag::abdefc
        godhwi = abdefc,
        /// 6D CNN weights tensor with groups; an alias for #zendnn::memory::format_tag::defcab
        dhwigo = defcab,

        /// 3D RNN data tensor in the format (seq_length, batch, input
        /// channels); an alias for #zendnn::memory::format_tag::abc.
        tnc = abc,
        /// 3D RNN data tensor in the format (batch, seq_length, input
        /// channels); an alias for #zendnn::memory::format_tag::bac.
        ntc = bac,
        /// 4D RNN states tensor in the format (num_layers, num_directions,
        /// batch, state channels); an alias for #zendnn::memory::format_tag::abcd.
        ldnc = abcd,
        /// 5D RNN weights tensor in the format (num_layers, num_directions,
        /// input_channels, num_gates, output_channels);
        /// an alias for #zendnn::memory::format_tag::abcde.
        ///
        ///  - For LSTM cells, the gates order is input, forget, candidate
        ///    and output gate.
        ///  - For GRU cells, the gates order is update, reset and output gate.
        ldigo = abcde,
        /// 5D RNN weights tensor in the format (num_layers, num_directions,
        /// num_gates, output_channels, input_channels);
        /// an alias for #zendnn::memory::format_tag::abdec.
        ///
        ///  - For LSTM cells, the gates order is input, forget, candidate
        ///    and output gate.
        ///  - For GRU cells, the gates order is update, reset and output gate.
        ldgoi = abdec,
        /// 4D LSTM projection tensor in the format (num_layers, num_directions,
        /// num_channels_in_hidden_state, num_channels_in_recurrent_projection);
        /// an alias for #zendnn::memory::format_tag::abcd.
        ldio = abcd,
        /// 4D LSTM projection tensor in the format (num_layers, num_directions,
        /// num_channels_in_recurrent_projection, num_channels_in_hidden_state);
        /// an alias for #zendnn::memory::format_tag::abdc.
        ldoi = abdc,
        /// 4D RNN bias tensor in the format (num_layers, num_directions,
        /// num_gates, output_channels);
        /// an alias for #zendnn::memory::format_tag::abcd.
        ///
        ///  - For LSTM cells, the gates order is input, forget, candidate
        ///    and output gate.
        ///  - For GRU cells, the gates order is update, reset and output gate.
        ldgo = abcd,

        // Opaque blocked formats

        AB16b16a = zendnn_AB16b16a,
        AB16b32a = zendnn_AB16b32a,
        AB16b64a = zendnn_AB16b64a,
        AB8b16a2b = zendnn_AB8b16a2b,
        AB8b32a2b = zendnn_AB8b32a2b,
        AB8b64a2b = zendnn_AB8b64a2b,
        AB4b16a4b = zendnn_AB4b16a4b,
        AB4b32a4b = zendnn_AB4b32a4b,
        AB4b64a4b = zendnn_AB4b64a4b,
        AB16b16a4b = zendnn_AB16b16a4b,
        AB16b32a4b = zendnn_AB16b32a4b,
        AB16b48a4b = zendnn_AB16b48a4b,
        AB16b64a4b = zendnn_AB16b64a4b,
        AB16b16a2b = zendnn_AB16b16a2b,
        AB16b32a2b = zendnn_AB16b32a2b,
        AB16b48a2b = zendnn_AB16b48a2b,
        AB16b64a2b = zendnn_AB16b64a2b,
        Abc16a = zendnn_Abc16a,
        ABc16a16b = zendnn_ABc16a16b,
        ABc4a4b = zendnn_ABc4a4b,
        aBc16b = zendnn_aBc16b,
        aBc32b = zendnn_aBc32b,
        ABc16b16a = zendnn_ABc16b16a,
        ABc16b32a = zendnn_ABc16b32a,
        ABc16b64a = zendnn_ABc16b64a,
        Abc4a = zendnn_Abc4a,
        aBc4b = zendnn_aBc4b,
        ABc4b16a4b = zendnn_ABc4b16a4b,
        ABc4b32a4b = zendnn_ABc4b32a4b,
        ABc4b64a4b = zendnn_ABc4b64a4b,
        ABc2b8a4b = zendnn_ABc2b8a4b,
        ABc16a16b2a = zendnn_ABc16a16b2a,
        ABc16b16a4b = zendnn_ABc16b16a4b,
        ABc16b32a4b = zendnn_ABc16b32a4b,
        ABc16b48a4b = zendnn_ABc16b48a4b,
        ABc16b64a4b = zendnn_ABc16b64a4b,
        ABc16b16a2b = zendnn_ABc16b16a2b,
        ABc16b32a2b = zendnn_ABc16b32a2b,
        ABc16b48a2b = zendnn_ABc16b48a2b,
        ABc16b64a2b = zendnn_ABc16b64a2b,
        ABc4b4a = zendnn_ABc4b4a,
        ABc8a16b2a = zendnn_ABc8a16b2a,
        ABc8a8b = zendnn_ABc8a8b,
        ABc8a4b = zendnn_ABc8a4b,
        aBc8b = zendnn_aBc8b,
        ABc8b16a2b = zendnn_ABc8b16a2b,
        ABc8b32a2b = zendnn_ABc8b32a2b,
        ABc8b64a2b = zendnn_ABc8b64a2b,
        ABc8b8a = zendnn_ABc8b8a,
        Abcd8a = zendnn_Abcd8a,
        Abcd16a = zendnn_Abcd16a,
        Abcd32a = zendnn_Abcd32a,
        ABcd16a16b = zendnn_ABcd16a16b,
        aBcd16b = zendnn_aBcd16b,
        aBcd32b = zendnn_aBcd32b,
        ABcd16b16a = zendnn_ABcd16b16a,
        ABcd16b32a = zendnn_ABcd16b32a,
        ABcd16b64a = zendnn_ABcd16b64a,
        aBCd16b16c = zendnn_aBCd16b16c,
        aBCd16c16b = zendnn_aBCd16c16b,
        Abcd4a = zendnn_Abcd4a,
        aBcd4b = zendnn_aBcd4b,
        ABcd4b16a4b = zendnn_ABcd4b16a4b,
        ABcd4b32a4b = zendnn_ABcd4b32a4b,
        ABcd4b64a4b = zendnn_ABcd4b64a4b,
        ABcd2b8a4b = zendnn_ABcd2b8a4b,
        ABcd4b4a = zendnn_ABcd4b4a,
        ABcd4a4b = zendnn_ABcd4a4b,
        aBCd4c16b4c = zendnn_aBCd4c16b4c,
        aBCd2c8b4c = zendnn_aBCd2c8b4c,
        ABcd16a16b2a = zendnn_ABcd16a16b2a,
        ABcd16b16a4b = zendnn_ABcd16b16a4b,
        ABcd16b32a4b = zendnn_ABcd16b32a4b,
        ABcd16b48a4b = zendnn_ABcd16b48a4b,
        ABcd16b64a4b = zendnn_ABcd16b64a4b,
        ABcd16b16a2b = zendnn_ABcd16b16a2b,
        ABcd16b32a2b = zendnn_ABcd16b32a2b,
        ABcd16b48a2b = zendnn_ABcd16b48a2b,
        ABcd16b64a2b = zendnn_ABcd16b64a2b,
        aBCd16b16c2b = zendnn_aBCd16b16c2b,
        aBCd16c16b4c = zendnn_aBCd16c16b4c,
        aBCd16c16b2c = zendnn_aBCd16c16b2c,
        aBCd4c4b = zendnn_aBCd4c4b,
        aBCd4b4c = zendnn_aBCd4b4c,
        ABcd8a16b2a = zendnn_ABcd8a16b2a,
        ABcd8a8b = zendnn_ABcd8a8b,
        ABcd8a4b = zendnn_ABcd8a4b,
        ABcd8a2b = zendnn_ABcd8a2b,
        /// 4D tensor blocked by 2nd dimension with block size 8
        aBcd8b = zendnn_aBcd8b,
        ABcd8b16a2b = zendnn_ABcd8b16a2b,
        ABcd8b32a2b = zendnn_ABcd8b32a2b,
        ABcd8b64a2b = zendnn_ABcd8b64a2b,
        aBCd8b16c2b = zendnn_aBCd8b16c2b,
        /// 4D tensor blocked by 1st and 2nd dimension with block size 8
        ABcd8b8a = zendnn_ABcd8b8a,
        aBCd8b8c = zendnn_aBCd8b8c,
        aBCd8b4c = zendnn_aBCd8b4c,
        aBCd8c16b2c = zendnn_aBCd8c16b2c,
        aBCd8c8b = zendnn_aBCd8c8b,
        Abcde16a = zendnn_Abcde16a,
        Abcde32a = zendnn_Abcde32a,
        ABcde16a16b = zendnn_ABcde16a16b,
        aBcde16b = zendnn_aBcde16b,
        aBcde32b = zendnn_aBcde32b,
        ABcde16b16a = zendnn_ABcde16b16a,
        ABcde16b32a = zendnn_ABcde16b32a,
        ABcde16b64a = zendnn_ABcde16b64a,
        aBCde16b16c = zendnn_aBCde16b16c,
        aBCde16c16b = zendnn_aBCde16c16b,
        aBCde2c8b4c = zendnn_aBCde2c8b4c,
        Abcde4a = zendnn_Abcde4a,
        aBcde4b = zendnn_aBcde4b,
        ABcde4b4a = zendnn_ABcde4b4a,
        ABcde4a4b = zendnn_ABcde4a4b,
        aBCde4b4c = zendnn_aBCde4b4c,
        aBCde4c16b4c = zendnn_aBCde4c16b4c,
        aBCde16b16c2b = zendnn_aBCde16b16c2b,
        aBCde16c16b4c = zendnn_aBCde16c16b4c,
        aBCde16c16b2c = zendnn_aBCde16c16b2c,
        aBCdef16c16b2c = zendnn_aBCdef16c16b2c,
        aBCde4c4b = zendnn_aBCde4c4b,
        Abcde8a = zendnn_Abcde8a,
        ABcde8a8b = zendnn_ABcde8a8b,
        ABcde8a4b = zendnn_ABcde8a4b,
        aBcde8b = zendnn_aBcde8b,
        ABcde8b16a2b = zendnn_ABcde8b16a2b,
        ABcde8b32a2b = zendnn_ABcde8b32a2b,
        ABcde8b64a2b = zendnn_ABcde8b64a2b,
        ABcde4b16a4b = zendnn_ABcde4b16a4b,
        ABcde4b32a4b = zendnn_ABcde4b32a4b,
        ABcde4b64a4b = zendnn_ABcde4b64a4b,
        ABcde16b16a4b = zendnn_ABcde16b16a4b,
        ABcde16b32a4b = zendnn_ABcde16b32a4b,
        ABcde16b48a4b = zendnn_ABcde16b48a4b,
        ABcde16b64a4b = zendnn_ABcde16b64a4b,
        ABcde16b16a2b = zendnn_ABcde16b16a2b,
        ABcde16b32a2b = zendnn_ABcde16b32a2b,
        ABcde16b48a2b = zendnn_ABcde16b48a2b,
        ABcde16b64a2b = zendnn_ABcde16b64a2b,
        ABcde2b8a4b = zendnn_ABcde2b8a4b,
        aBCde8b16c2b = zendnn_aBCde8b16c2b,
        ABcde8b8a = zendnn_ABcde8b8a,
        aBCde8b8c = zendnn_aBCde8b8c,
        aBCde8b4c = zendnn_aBCde8b4c,
        ABcd4a8b8a4b = zendnn_ABcd4a8b8a4b,
        ABcd2a8b8a2b = zendnn_ABcd2a8b8a2b,
        aBCde4b8c8b4c = zendnn_aBCde4b8c8b4c,
        aBCde2b8c8b2c = zendnn_aBCde2b8c8b2c,
        aBCde8c16b2c = zendnn_aBCde8c16b2c,
        aBCde8c8b = zendnn_aBCde8c8b,
        aBcdef16b = zendnn_aBcdef16b,
        aBCdef16b16c = zendnn_aBCdef16b16c,
        aBCdef16c16b = zendnn_aBCdef16c16b,
        aBcdef4b = zendnn_aBcdef4b,
        aBCdef2c8b4c = zendnn_aBCdef2c8b4c,
        aBCdef4c4b = zendnn_aBCdef4c4b,
        aBCdef4b4c = zendnn_aBCdef4b4c,
        aBCdef8b8c = zendnn_aBCdef8b8c,
        aBCdef8b4c = zendnn_aBCdef8b4c,
        aBCdef8c16b2c = zendnn_aBCdef8c16b2c,
        aBCdef4c16b4c = zendnn_aBCdef4c16b4c,
        aBCdef8c8b = zendnn_aBCdef8c8b,
        aBdc16b = zendnn_aBdc16b,
        aBdc4b = zendnn_aBdc4b,
        aBdc8b = zendnn_aBdc8b,
        aBdec16b = zendnn_aBdec16b,
        aBdec4b = zendnn_aBdec4b,
        aBdec8b = zendnn_aBdec8b,
        aBdefc16b = zendnn_aBdefc16b,
        aCBdef16c16b = zendnn_aCBdef16c16b,
        aCBdef16b16c = zendnn_aCBdef16b16c,
        aBdefc4b = zendnn_aBdefc4b,
        aBdefc8b = zendnn_aBdefc8b,
        Acb16a = zendnn_Acb16a,
        Acb4a = zendnn_Acb4a,
        Acb8a = zendnn_Acb8a,
        aCBd16b16c = zendnn_aCBd16b16c,
        aCBd16c16b = zendnn_aCBd16c16b,
        aCBde16b16c = zendnn_aCBde16b16c,
        aCBde16c16b = zendnn_aCBde16c16b,
        Acdb16a = zendnn_Acdb16a,
        Acdb4a = zendnn_Acdb4a,
        Acdb8a = zendnn_Acdb8a,
        Acdeb16a = zendnn_Acdeb16a,
        Acdeb4a = zendnn_Acdeb4a,
        Acdeb8a = zendnn_Acdeb8a,
        BAc16a16b = zendnn_BAc16a16b,
        BAc16b16a = zendnn_BAc16b16a,
        BAcd16a16b = zendnn_BAcd16a16b,
        BAcd16b16a = zendnn_BAcd16b16a,
        ABcd32a32b = zendnn_ABcd32a32b,
        BAcde16b16a = zendnn_BAcde16b16a,
        BAcde16a16b = zendnn_BAcde16a16b,
        aBdec32b = zendnn_aBdec32b,
        Abcdef16a = zendnn_Abcdef16a,
        Abcdef32a = zendnn_Abcdef32a,
        Acdb32a = zendnn_Acdb32a,
        aBCd2b4c2b = zendnn_aBCd2b4c2b,
        aBCde2b4c2b = zendnn_aBCde2b4c2b,
        aBCdef2b4c2b = zendnn_aBCdef2b4c2b,
        aBCd2c4b2c = zendnn_aBCd2c4b2c,
        aBCde2c4b2c = zendnn_aBCde2c4b2c,
        aBCdef2c4b2c = zendnn_aBCdef2c4b2c,
        aBCd4b8c2b = zendnn_aBCd4b8c2b,
        aBCde4b8c2b = zendnn_aBCde4b8c2b,
        aBCdef4b8c2b = zendnn_aBCdef4b8c2b,
        aBCd4c8b2c = zendnn_aBCd4c8b2c,
        aBCde4c8b2c = zendnn_aBCde4c8b2c,
        aBCdef4c8b2c = zendnn_aBCdef4c8b2c,
        AB32a32b8a4b = zendnn_AB32a32b8a4b,
        AB32a32b8a2b = zendnn_AB32a32b8a2b,
        AB8a4b = zendnn_AB8a4b,
        AB8a2b = zendnn_AB8a2b,
        abDc32d = zendnn_abDc32d,
        abDC32d4c = zendnn_abDC32d4c,
        abCd32c = zendnn_abCd32c,
        abdEc32e = zendnn_abdEc32e,
        abdEC32e2c = zendnn_abdEC32e2c,
        abdEC32e4c = zendnn_abdEC32e4c,
        abdCe32c = zendnn_abdCe32c,
        abdCE32c2e = zendnn_abdCE32c2e,
        aBCdef16c16b4c = zendnn_aBCdef16c16b4c,
        aBdC16b4c = zendnn_aBdC16b4c,
        aBdeC16b4c = zendnn_aBdeC16b4c,
        AcB16a4b = zendnn_AcB16a4b,
        AcdB16a2b = zendnn_AcdB16a2b,
        aBdefC16b4c = zendnn_aBdefC16b4c,
        AcdeB16a4b = zendnn_AcdeB16a4b,

        Acb32a = zendnn_Acb32a,
        AcB32a2b = zendnn_AcB32a2b,
        AcB32a4b = zendnn_AcB32a4b,
        Acb48a = zendnn_Acb48a,
        AcB48a2b = zendnn_AcB48a2b,
        AcB48a4b = zendnn_AcB48a4b,
        Acb64a = zendnn_Acb64a,
        AcB64a2b = zendnn_AcB64a2b,
        AcB64a4b = zendnn_AcB64a4b,
        cBa2b = zendnn_cBa2b,
        cBa4b = zendnn_cBa4b,
        aBdc32b = zendnn_aBdc32b,
        aBdC32b2c = zendnn_aBdC32b2c,
        aBdC32b4c = zendnn_aBdC32b4c,
        aBdc48b = zendnn_aBdc48b,
        aBdC48b2c = zendnn_aBdC48b2c,
        aBdC48b4c = zendnn_aBdC48b4c,
        aBdc64b = zendnn_aBdc64b,
        aBdC64b2c = zendnn_aBdC64b2c,
        aBdC64b4c = zendnn_aBdC64b4c,
        adcb = zendnn_adcb,
        adCb2c = zendnn_adCb2c,
        adCb4c = zendnn_adCb4c,
        AcdB32a2b = zendnn_AcdB32a2b,
        AcdB32a4b = zendnn_AcdB32a4b,
        Acdb48a = zendnn_Acdb48a,
        AcdB48a2b = zendnn_AcdB48a2b,
        AcdB48a4b = zendnn_AcdB48a4b,
        Acdb64a = zendnn_Acdb64a,
        AcdB64a2b = zendnn_AcdB64a2b,
        AcdB64a4b = zendnn_AcdB64a4b,
        cdBa2b = zendnn_cdBa2b,
        cdBa4b = zendnn_cdBa4b,
        aBdeC32b2c = zendnn_aBdeC32b2c,
        aBdeC32b4c = zendnn_aBdeC32b4c,
        aBdec48b = zendnn_aBdec48b,
        aBdeC48b2c = zendnn_aBdeC48b2c,
        aBdeC48b4c = zendnn_aBdeC48b4c,
        aBdec64b = zendnn_aBdec64b,
        aBdeC64b2c = zendnn_aBdeC64b2c,
        aBdeC64b4c = zendnn_aBdeC64b4c,
        adecb = zendnn_adecb,
        adeCb2c = zendnn_adeCb2c,
        adeCb4c = zendnn_adeCb4c,
        Acdeb32a = zendnn_Acdeb32a,
        AcdeB32a2b = zendnn_AcdeB32a2b,
        AcdeB32a4b = zendnn_AcdeB32a4b,
        Acdeb48a = zendnn_Acdeb48a,
        AcdeB48a2b = zendnn_AcdeB48a2b,
        AcdeB48a4b = zendnn_AcdeB48a4b,
        Acdeb64a = zendnn_Acdeb64a,
        AcdeB64a2b = zendnn_AcdeB64a2b,
        AcdeB64a4b = zendnn_AcdeB64a4b,
        cdeBa2b = zendnn_cdeBa2b,
        cdeBa4b = zendnn_cdeBa4b,
        aBdefc32b = zendnn_aBdefc32b,
        aBdefC32b2c = zendnn_aBdefC32b2c,
        aBdefC32b4c = zendnn_aBdefC32b4c,
        aBdefc48b = zendnn_aBdefc48b,
        aBdefC48b2c = zendnn_aBdefC48b2c,
        aBdefC48b4c = zendnn_aBdefC48b4c,
        aBdefc64b = zendnn_aBdefc64b,
        aBdefC64b2c = zendnn_aBdefC64b2c,
        aBdefC64b4c = zendnn_aBdefC64b4c,
        adefcb = zendnn_adefcb,
        adefCb2c = zendnn_adefCb2c,
        adefCb4c = zendnn_adefCb4c,
        ABc32a32b = zendnn_ABc32a32b,
        BAc8a16b2a = zendnn_BAc8a16b2a,
        BAcd8a16b2a = zendnn_BAcd8a16b2a,
        ABcde8a16b2a = zendnn_ABcde8a16b2a,
        aCBd8b16c2b = zendnn_aCBd8b16c2b,
        BAcde8a16b2a = zendnn_BAcde8a16b2a,
        aCBde8b16c2b = zendnn_aCBde8b16c2b,
        ABcde32a32b = zendnn_ABcde32a32b,
        ABc4a8b8a4b = zendnn_ABc4a8b8a4b,
        ABcde4a8b8a4b = zendnn_ABcde4a8b8a4b,
        BAc4b8a8b4a = zendnn_BAc4b8a8b4a,
        BAcd4b8a8b4a = zendnn_BAcd4b8a8b4a,
        BAcde4b8a8b4a = zendnn_BAcde4b8a8b4a,
        aBCd4b8c8b4c = zendnn_aBCd4b8c8b4c,
        aBCdef4b8c8b4c = zendnn_aBCdef4b8c8b4c,
        aBCdef8b16c2b = zendnn_aBCdef8b16c2b,
        aCBdef8b16c2b = zendnn_aCBdef8b16c2b,
        aBdC16b2c = zendnn_aBdC16b2c,
        aBdeC16b2c = zendnn_aBdeC16b2c,
        aBdefC16b2c = zendnn_aBdefC16b2c,
        aBedc16b = zendnn_aBedc16b,
        AcB16a2b = zendnn_AcB16a2b,
        AcdB16a4b = zendnn_AcdB16a4b,
        AcdeB16a2b = zendnn_AcdeB16a2b,
        Adcb16a = zendnn_Adcb16a,
        aCBd4c8b8c4b = zendnn_aCBd4c8b8c4b,
        aCBde4c8b8c4b = zendnn_aCBde4c8b8c4b,
        aCBdef4c8b8c4b = zendnn_aCBdef4c8b8c4b,
        ABc32a16b = zendnn_ABc32a16b,
        ABcd32a16b = zendnn_ABcd32a16b,
        ABcde32a16b = zendnn_ABcde32a16b,
        AB48a16b = zendnn_AB48a16b,
        AB48a32b = zendnn_AB48a32b,
        ABc40a16b = zendnn_ABc40a16b,
        ABc40a32b = zendnn_ABc40a32b,
        aBC48b16c = zendnn_aBC48b16c,
        aBC48b32c = zendnn_aBC48b32c,
        ABcd40a16b = zendnn_ABcd40a16b,
        ABcd40a32b = zendnn_ABcd40a32b,
        BA16a16b = zendnn_BA16a16b,
        BA16a32b = zendnn_BA16a32b,
        BA16a48b = zendnn_BA16a48b,
        BA16a64b = zendnn_BA16a64b,
        BA16a16b2a = zendnn_BA16a16b2a,
        BA16a32b2a = zendnn_BA16a32b2a,
        BA16a48b2a = zendnn_BA16a48b2a,
        BA16a64b2a = zendnn_BA16a64b2a,
        BA16a16b4a = zendnn_BA16a16b4a,
        BA16a32b4a = zendnn_BA16a32b4a,
        BA16a48b4a = zendnn_BA16a48b4a,
        BA16a64b4a = zendnn_BA16a64b4a,
        decbA16a = zendnn_decbA16a,

        format_tag_last = zendnn_format_tag_last,

        nCdhw16c = zendnn_nCdhw16c,
        nCdhw4c = zendnn_nCdhw4c,
        nCdhw8c = zendnn_nCdhw8c,
        nChw16c = zendnn_nChw16c,
        nChw4c = zendnn_nChw4c,
        nChw8c = zendnn_nChw8c,
        nCw16c = zendnn_nCw16c,
        nCw4c = zendnn_nCw4c,
        nCw8c = zendnn_nCw8c,
        NCw16n16c = zendnn_NCw16n16c,
        NChw16n16c = zendnn_NChw16n16c,
        NCdhw16n16c = zendnn_NCdhw16n16c,
        NCdhw32n32c = zendnn_NCdhw32n32c,
        NChw32n32c = zendnn_NChw32n32c,
        IOhw16i16o = zendnn_IOhw16i16o,
        OI16i16o = zendnn_OI16i16o,
        OI16i32o = zendnn_OI16i32o,
        OI16i64o = zendnn_OI16i64o,
        OI8i16o2i = zendnn_OI8i16o2i,
        OI8i32o2i = zendnn_OI8i32o2i,
        OI8i64o2i = zendnn_OI8i64o2i,
        OI4i16o4i = zendnn_OI4i16o4i,
        OI4i32o4i = zendnn_OI4i32o4i,
        OI4i64o4i = zendnn_OI4i64o4i,
        Ohwi32o = zendnn_Ohwi32o,
        IOdhw16i16o = zendnn_IOdhw16i16o,
        gIOhw16i16o = zendnn_gIOhw16i16o,
        gOhwi32o = zendnn_gOhwi32o,
        Goidhw16g = zendnn_Goidhw16g,
        IOw16o16i = zendnn_IOw16o16i,
        OIw16i16o = zendnn_OIw16i16o,
        OIw16i32o = zendnn_OIw16i32o,
        OIw16i64o = zendnn_OIw16i64o,
        IOw16i16o = zendnn_IOw16i16o,
        gIOw16i16o = zendnn_gIOw16i16o,
        OIw16o16i = zendnn_OIw16o16i,
        Oiw16o = zendnn_Oiw16o,
        OIw4i16o4i = zendnn_OIw4i16o4i,
        OIw4i32o4i = zendnn_OIw4i32o4i,
        OIw4i64o4i = zendnn_OIw4i64o4i,
        OIw2i8o4i = zendnn_OIw2i8o4i,
        OIw4i4o = zendnn_OIw4i4o,
        OIw4o4i = zendnn_OIw4o4i,
        Oiw4o = zendnn_Oiw4o,
        OIw8i16o2i = zendnn_OIw8i16o2i,
        OIw8i32o2i = zendnn_OIw8i32o2i,
        OIw8i64o2i = zendnn_OIw8i64o2i,
        OIw8i8o = zendnn_OIw8i8o,
        OIw8o16i2o = zendnn_OIw8o16i2o,
        OIw8o8i = zendnn_OIw8o8i,
        OIw8o4i = zendnn_OIw8o4i,
        OIw16i16o4i = zendnn_OIw16i16o4i,
        OIw16i32o4i = zendnn_OIw16i32o4i,
        OIw16i48o4i = zendnn_OIw16i48o4i,
        OIw16i64o4i = zendnn_OIw16i64o4i,
        OIw16i16o2i = zendnn_OIw16i16o2i,
        OIw16i32o2i = zendnn_OIw16i32o2i,
        OIw16i48o2i = zendnn_OIw16i48o2i,
        OIw16i64o2i = zendnn_OIw16i64o2i,
        OIw16o16i2o = zendnn_OIw16o16i2o,
        Owi16o = zendnn_Owi16o,
        OwI16o2i = zendnn_OwI16o2i,
        Owi4o = zendnn_Owi4o,
        Owi8o = zendnn_Owi8o,
        IOhw16o16i = zendnn_IOhw16o16i,
        Ohwi16o = zendnn_Ohwi16o,
        OhwI16o2i = zendnn_OhwI16o2i,
        Ohwi4o = zendnn_Ohwi4o,
        Ohwi8o = zendnn_Ohwi8o,
        OIhw16i16o = zendnn_OIhw16i16o,
        OIhw16i32o = zendnn_OIhw16i32o,
        OIhw16i64o = zendnn_OIhw16i64o,
        OIhw16o16i = zendnn_OIhw16o16i,
        Oihw16o = zendnn_Oihw16o,
        OIhw4i16o4i = zendnn_OIhw4i16o4i,
        OIhw4i32o4i = zendnn_OIhw4i32o4i,
        OIhw4i64o4i = zendnn_OIhw4i64o4i,
        OIhw4i4o = zendnn_OIhw4i4o,
        OIhw4o4i = zendnn_OIhw4o4i,
        Oihw4o = zendnn_Oihw4o,
        OIhw8i16o2i = zendnn_OIhw8i16o2i,
        OIhw8i32o2i = zendnn_OIhw8i32o2i,
        OIhw8i64o2i = zendnn_OIhw8i64o2i,
        OIhw8i8o = zendnn_OIhw8i8o,
        OIhw8o16i2o = zendnn_OIhw8o16i2o,
        OIhw8o8i = zendnn_OIhw8o8i,
        OIhw8o4i = zendnn_OIhw8o4i,
        OIhw2i8o4i = zendnn_OIhw2i8o4i,
        IOdhw16o16i = zendnn_IOdhw16o16i,
        Odhwi16o = zendnn_Odhwi16o,
        OdhwI16o2i = zendnn_OdhwI16o2i,
        Odhwi4o = zendnn_Odhwi4o,
        Odhwi8o = zendnn_Odhwi8o,
        OIdhw16i16o = zendnn_OIdhw16i16o,
        OIdhw16i32o = zendnn_OIdhw16i32o,
        OIdhw16i64o = zendnn_OIdhw16i64o,
        OIdhw16o16i = zendnn_OIdhw16o16i,
        OIdhw16o16i2o = zendnn_OIdhw16o16i2o,
        Oidhw16o = zendnn_Oidhw16o,
        OIdhw4i4o = zendnn_OIdhw4i4o,
        OIdhw4o4i = zendnn_OIdhw4o4i,
        Oidhw4o = zendnn_Oidhw4o,
        OIdhw8i16o2i = zendnn_OIdhw8i16o2i,
        OIdhw8i32o2i = zendnn_OIdhw8i32o2i,
        OIdhw8i64o2i = zendnn_OIdhw8i64o2i,
        OIdhw4i16o4i = zendnn_OIdhw4i16o4i,
        OIdhw16i16o4i = zendnn_OIdhw16i16o4i,
        OIdhw16i32o4i = zendnn_OIdhw16i32o4i,
        OIdhw16i48o4i = zendnn_OIdhw16i48o4i,
        OIdhw16i64o4i = zendnn_OIdhw16i64o4i,
        OIdhw16i16o2i = zendnn_OIdhw16i16o2i,
        OIdhw16i32o2i = zendnn_OIdhw16i32o2i,
        OIdhw16i48o2i = zendnn_OIdhw16i48o2i,
        OIdhw16i64o2i = zendnn_OIdhw16i64o2i,
        OIdhw4i32o4i = zendnn_OIdhw4i32o4i,
        OIdhw4i64o4i = zendnn_OIdhw4i64o4i,
        OIdhw2i8o4i = zendnn_OIdhw2i8o4i,
        OIdhw8i8o = zendnn_OIdhw8i8o,
        OIdhw8o8i = zendnn_OIdhw8o8i,
        OIdhw8o4i = zendnn_OIdhw8o4i,
        gIOw16o16i = zendnn_gIOw16o16i,
        gOIw16i16o = zendnn_gOIw16i16o,
        gOIw16o16i = zendnn_gOIw16o16i,
        gOiw16o = zendnn_gOiw16o,
        gOIw4i16o4i = zendnn_gOIw4i16o4i,
        gOIw2i8o4i = zendnn_gOIw2i8o4i,
        gOIw4i4o = zendnn_gOIw4i4o,
        gOIw4o4i = zendnn_gOIw4o4i,
        gOiw4o = zendnn_gOiw4o,
        gOIw8i16o2i = zendnn_gOIw8i16o2i,
        gOIw8i8o = zendnn_gOIw8i8o,
        gOIw8o16i2o = zendnn_gOIw8o16i2o,
        gOIw8o8i = zendnn_gOIw8o8i,
        gOIw8o4i = zendnn_gOIw8o4i,
        gOIw16i16o4i = zendnn_gOIw16i16o4i,
        gOIw16i16o2i = zendnn_gOIw16i16o2i,
        gOIw16o16i2o = zendnn_gOIw16o16i2o,
        gOwi16o = zendnn_gOwi16o,
        gOwI16o2i = zendnn_gOwI16o2i,
        gOwi4o = zendnn_gOwi4o,
        gOwi8o = zendnn_gOwi8o,
        Goiw8g = zendnn_Goiw8g,
        Goiw16g = zendnn_Goiw16g,
        gIOhw16o16i = zendnn_gIOhw16o16i,
        gOhwi16o = zendnn_gOhwi16o,
        gOhwI16o2i = zendnn_gOhwI16o2i,
        gOhwi4o = zendnn_gOhwi4o,
        gOhwi8o = zendnn_gOhwi8o,
        Goihw16g = zendnn_Goihw16g,
        gOIhw16i16o = zendnn_gOIhw16i16o,
        gOIhw16o16i = zendnn_gOIhw16o16i,
        gOihw16o = zendnn_gOihw16o,
        gOIhw4i16o4i = zendnn_gOIhw4i16o4i,
        gOIhw2i8o4i = zendnn_gOIhw2i8o4i,
        gOIhw4i4o = zendnn_gOIhw4i4o,
        gOIhw4o4i = zendnn_gOIhw4o4i,
        gOihw4o = zendnn_gOihw4o,
        Goihw8g = zendnn_Goihw8g,
        gOIhw8i16o2i = zendnn_gOIhw8i16o2i,
        gOIhw8i8o = zendnn_gOIhw8i8o,
        gOIhw8o16i2o = zendnn_gOIhw8o16i2o,
        OIw4o8i8o4i = zendnn_OIw4o8i8o4i,
        OIdhw4o8i8o4i = zendnn_OIdhw4o8i8o4i,
        OIhw4o8i8o4i = zendnn_OIhw4o8i8o4i,
        OIhw2o8i8o2i = zendnn_OIhw2o8i8o2i,
        gOIw4o8i8o4i = zendnn_gOIw4o8i8o4i,
        gOIdhw4o8i8o4i = zendnn_gOIdhw4o8i8o4i,
        gOIhw4o8i8o4i = zendnn_gOIhw4o8i8o4i,
        gOIhw2o8i8o2i = zendnn_gOIhw2o8i8o2i,
        OIhw16i16o4i = zendnn_OIhw16i16o4i,
        OIhw16i32o4i = zendnn_OIhw16i32o4i,
        OIhw16i48o4i = zendnn_OIhw16i48o4i,
        OIhw16i64o4i = zendnn_OIhw16i64o4i,
        OIhw16i16o2i = zendnn_OIhw16i16o2i,
        OIhw16i32o2i = zendnn_OIhw16i32o2i,
        OIhw16i48o2i = zendnn_OIhw16i48o2i,
        OIhw16i64o2i = zendnn_OIhw16i64o2i,
        OIhw16o16i2o = zendnn_OIhw16o16i2o,
        gOIhw16i16o4i = zendnn_gOIhw16i16o4i,
        gOIhw16i16o2i = zendnn_gOIhw16i16o2i,
        gOIhw16o16i2o = zendnn_gOIhw16o16i2o,
        gOIhw8o8i = zendnn_gOIhw8o8i,
        gOIhw8o4i = zendnn_gOIhw8o4i,
        gIOdhw16i16o = zendnn_gIOdhw16i16o,
        gIOdhw16o16i = zendnn_gIOdhw16o16i,
        gOdhwi16o = zendnn_gOdhwi16o,
        gOdhwI16o2i = zendnn_gOdhwI16o2i,
        gOdhwi4o = zendnn_gOdhwi4o,
        gOdhwi8o = zendnn_gOdhwi8o,
        gOIdhw16i16o = zendnn_gOIdhw16i16o,
        gOIdhw16o16i = zendnn_gOIdhw16o16i,
        gOIdhw16o16i2o = zendnn_gOIdhw16o16i2o,
        gOidhw16o = zendnn_gOidhw16o,
        gOIdhw4i4o = zendnn_gOIdhw4i4o,
        gOIdhw4o4i = zendnn_gOIdhw4o4i,
        gOidhw4o = zendnn_gOidhw4o,
        gOIdhw8i16o2i = zendnn_gOIdhw8i16o2i,
        gOIdhw4i16o4i = zendnn_gOIdhw4i16o4i,
        gOIdhw16i16o4i = zendnn_gOIdhw16i16o4i,
        gOIdhw16i16o2i = zendnn_gOIdhw16i16o2i,
        gOIdhw2i8o4i = zendnn_gOIdhw2i8o4i,
        gOIdhw8i8o = zendnn_gOIdhw8i8o,
        gOIdhw8o8i = zendnn_gOIdhw8o8i,
        gOIdhw8o4i = zendnn_gOIdhw8o4i,
        gOIw2i4o2i = zendnn_gOIw2i4o2i,
        gOIhw2i4o2i = zendnn_gOIhw2i4o2i,
        gOIdhw2i4o2i = zendnn_gOIdhw2i4o2i,
        gOIw2o4i2o = zendnn_gOIw2o4i2o,
        gOIhw2o4i2o = zendnn_gOIhw2o4i2o,
        gOIdhw2o4i2o = zendnn_gOIdhw2o4i2o,
        gOIw4i8o2i = zendnn_gOIw4i8o2i,
        gOIhw4i8o2i = zendnn_gOIhw4i8o2i,
        gOIdhw4i8o2i = zendnn_gOIdhw4i8o2i,
        gOIw4o8i2o = zendnn_gOIw4o8i2o,
        gOIhw4o8i2o = zendnn_gOIhw4o8i2o,
        gOIdhw4o8i2o = zendnn_gOIdhw4o8i2o,
        ldOi32o = abDc32d,
        ldOI32o4i = abDC32d4c,
        ldgOi32o = abdEc32e,
        ldgOI32o2i = abdEC32e2c,
        ldgOI32o4i = abdEC32e4c,
        OwI16o4i = zendnn_OwI16o4i,
        OhwI16o4i = zendnn_OhwI16o4i,
        gOwI16o4i = zendnn_gOwI16o4i,
        gOhwI16o4i = zendnn_gOhwI16o4i,
        OdhwI16o4i = zendnn_OdhwI16o4i,
        gOdhwI16o4i = zendnn_gOdhwI16o4i,

        Owi32o = zendnn_Owi32o,
        OwI32o2i = zendnn_OwI32o2i,
        OwI32o4i = zendnn_OwI32o4i,
        Owi48o = zendnn_Owi48o,
        OwI48o2i = zendnn_OwI48o2i,
        OwI48o4i = zendnn_OwI48o4i,
        Owi64o = zendnn_Owi64o,
        OwI64o2i = zendnn_OwI64o2i,
        OwI64o4i = zendnn_OwI64o4i,
        wIo2i = zendnn_wIo2i,
        wIo4i = zendnn_wIo4i,
        gOwi32o = zendnn_gOwi32o,
        gOwI32o2i = zendnn_gOwI32o2i,
        gOwI32o4i = zendnn_gOwI32o4i,
        gOwi48o = zendnn_gOwi48o,
        gOwI48o2i = zendnn_gOwI48o2i,
        gOwI48o4i = zendnn_gOwI48o4i,
        gOwi64o = zendnn_gOwi64o,
        gOwI64o2i = zendnn_gOwI64o2i,
        gOwI64o4i = zendnn_gOwI64o4i,
        gwio = zendnn_gwio,
        gwIo2i = zendnn_gwIo2i,
        gwIo4i = zendnn_gwIo4i,
        OhwI32o = zendnn_OhwI32o,
        OhwI32o2i = zendnn_OhwI32o2i,
        OhwI32o4i = zendnn_OhwI32o4i,
        Ohwi48o = zendnn_Ohwi48o,
        OhwI48o2i = zendnn_OhwI48o2i,
        OhwI48o4i = zendnn_OhwI48o4i,
        Ohwi64o = zendnn_Ohwi64o,
        OhwI64o2i = zendnn_OhwI64o2i,
        OhwI64o4i = zendnn_OhwI64o4i,
        hwIo2i = zendnn_hwIo2i,
        hwIo4i = zendnn_hwIo4i,
        gOhwI32o = zendnn_gOhwI32o,
        gOhwI32o2i = zendnn_gOhwI32o2i,
        gOhwI32o4i = zendnn_gOhwI32o4i,
        gOhwi48o = zendnn_gOhwi48o,
        gOhwI48o2i = zendnn_gOhwI48o2i,
        gOhwI48o4i = zendnn_gOhwI48o4i,
        gOhwi64o = zendnn_gOhwi64o,
        gOhwI64o2i = zendnn_gOhwI64o2i,
        gOhwI64o4i = zendnn_gOhwI64o4i,
        ghwio = zendnn_ghwio,
        ghwIo2i = zendnn_ghwIo2i,
        ghwIo4i = zendnn_ghwIo4i,
        Odhwi32o = zendnn_Odhwi32o,
        OdhwI32o2i = zendnn_OdhwI32o2i,
        OdhwI32o4i = zendnn_OdhwI32o4i,
        Odhwi48o = zendnn_Odhwi48o,
        OdhwI48o2i = zendnn_OdhwI48o2i,
        OdhwI48o4i = zendnn_OdhwI48o4i,
        Odhwi64o = zendnn_Odhwi64o,
        OdhwI64o2i = zendnn_OdhwI64o2i,
        OdhwI64o4i = zendnn_OdhwI64o4i,
        dhwIo2i = zendnn_dhwIo2i,
        dhwIo4i = zendnn_dhwIo4i,
        gOdhwi32o = zendnn_gOdhwi32o,
        gOdhwI32o2i = zendnn_gOdhwI32o2i,
        gOdhwI32o4i = zendnn_gOdhwI32o4i,
        gOdhwi48o = zendnn_gOdhwi48o,
        gOdhwI48o2i = zendnn_gOdhwI48o2i,
        gOdhwI48o4i = zendnn_gOdhwI48o4i,
        gOdhwi64o = zendnn_gOdhwi64o,
        gOdhwI64o2i = zendnn_gOdhwI64o2i,
        gOdhwI64o4i = zendnn_gOdhwI64o4i,
        gdhwio = zendnn_gdhwio,
        gdhwIo2i = zendnn_gdhwIo2i,
        gdhwIo4i = zendnn_gdhwIo4i,
        ldIo32i = zendnn_ldIo32i,
        ldgIo32i = zendnn_ldgIo32i,
        ldgIO32i2o = zendnn_ldgIO32i2o,
        nCdhw32c = zendnn_nCdhw32c,
        nChw32c = zendnn_nChw32c,
        nCw32c = zendnn_nCw32c,
        NCw32n16c = zendnn_NCw32n16c,
        NChw32n16c = zendnn_NChw32n16c,
        NCdhw32n16c = zendnn_NCdhw32n16c,
        NCw32n32c = zendnn_NCw32n32c,
        OI16i16o4i = zendnn_OI16i16o4i,
        IOw8o16i2o = zendnn_IOw8o16i2o,
        IOhw8o16i2o = zendnn_IOhw8o16i2o,
        Owhi16o = zendnn_Owhi16o,
        OIdhw8o16i2o = zendnn_OIdhw8o16i2o,
        IOdhw8o16i2o = zendnn_IOdhw8o16i2o,
        Goiw4g = zendnn_Goiw4g,
        gIOw8o16i2o = zendnn_gIOw8o16i2o,
        Goiw32g = zendnn_Goiw32g,
        Goihw4g = zendnn_Goihw4g,
        gIOhw8o16i2o = zendnn_gIOhw8o16i2o,
        Goihw32g = zendnn_Goihw32g,
        gOwhi16o = zendnn_gOwhi16o,
        IOw4i8o8i4o = zendnn_IOw4i8o8i4o,
        IOhw4i8o8i4o = zendnn_IOhw4i8o8i4o,
        IOdhw4i8o8i4o = zendnn_IOdhw4i8o8i4o,
        gIOw4i8o8i4o = zendnn_gIOw4i8o8i4o,
        gIOhw4i8o8i4o = zendnn_gIOhw4i8o8i4o,
        gIOdhw4i8o8i4o = zendnn_gIOdhw4i8o8i4o,
        gOIdhw8o16i2o = zendnn_gOIdhw8o16i2o,
        gIOdhw8o16i2o = zendnn_gIOdhw8o16i2o,
        Goidhw32g = zendnn_Goidhw32g,
        OI16i32o4i = zendnn_OI16i32o4i,
        OI16i48o4i = zendnn_OI16i48o4i,
        OI16i64o4i = zendnn_OI16i64o4i,
        OI16i16o2i = zendnn_OI16i16o2i,
        OI16i32o2i = zendnn_OI16i32o2i,
        OI16i48o2i = zendnn_OI16i48o2i,
        OI16i64o2i = zendnn_OI16i64o2i,
        aBdeC16c16b4c = zendnn_aBdeC16c16b4c,
        AcB16b16a2b = zendnn_AcB16b16a2b,
        aBdC16c16b2c = zendnn_aBdC16c16b2c,
        AcB16b16a4b = zendnn_AcB16b16a4b,
        aBdC16c16b4c = zendnn_aBdC16c16b4c,
        AcdB16b16a2b = zendnn_AcdB16b16a2b,
        aBdefC16c16b4c = zendnn_aBdefC16c16b4c,
        AcdeB16b16a4b = zendnn_AcdeB16b16a4b,
        AcB16b32a2b = zendnn_AcB16b32a2b,
        AcB16b32a4b = zendnn_AcB16b32a4b,
        AcB16b48a2b = zendnn_AcB16b48a2b,
        AcB16b48a4b = zendnn_AcB16b48a4b,
        AcB16b64a2b = zendnn_AcB16b64a2b,
        AcB16b64a4b = zendnn_AcB16b64a4b,
        aBdC16c32b2c = zendnn_aBdC16c32b2c,
        aBdC16c32b4c = zendnn_aBdC16c32b4c,
        aBdC16c48b2c = zendnn_aBdC16c48b2c,
        aBdC16c48b4c = zendnn_aBdC16c48b4c,
        aBdC16c64b2c = zendnn_aBdC16c64b2c,
        aBdC16c64b4c = zendnn_aBdC16c64b4c,
        AcdB16b32a2b = zendnn_AcdB16b32a2b,
        AcdB16b32a4b = zendnn_AcdB16b32a4b,
        AcdB16b48a2b = zendnn_AcdB16b48a2b,
        AcdB16b48a4b = zendnn_AcdB16b48a4b,
        AcdB16b64a2b = zendnn_AcdB16b64a2b,
        AcdB16b64a4b = zendnn_AcdB16b64a4b,
        aBdeC16c32b2c = zendnn_aBdeC16c32b2c,
        aBdeC16c32b4c = zendnn_aBdeC16c32b4c,
        aBdeC16c48b2c = zendnn_aBdeC16c48b2c,
        aBdeC16c48b4c = zendnn_aBdeC16c48b4c,
        aBdeC16c64b2c = zendnn_aBdeC16c64b2c,
        aBdeC16c64b4c = zendnn_aBdeC16c64b4c,
        AcdeB16b32a2b = zendnn_AcdeB16b32a2b,
        AcdeB16b32a4b = zendnn_AcdeB16b32a4b,
        AcdeB16b48a2b = zendnn_AcdeB16b48a2b,
        AcdeB16b48a4b = zendnn_AcdeB16b48a4b,
        AcdeB16b64a2b = zendnn_AcdeB16b64a2b,
        AcdeB16b64a4b = zendnn_AcdeB16b64a4b,
        aBdefC16c32b2c = zendnn_aBdefC16c32b2c,
        aBdefC16c32b4c = zendnn_aBdefC16c32b4c,
        aBdefC16c48b2c = zendnn_aBdefC16c48b2c,
        aBdefC16c48b4c = zendnn_aBdefC16c48b4c,
        aBdefC16c64b2c = zendnn_aBdefC16c64b2c,
        aBdefC16c64b4c = zendnn_aBdefC16c64b4c,
        OwI16i16o2i = zendnn_OwI16i16o2i,
        gOwI16i16o2i = zendnn_gOwI16i16o2i,
        OhwI16i16o2i = zendnn_OhwI16i16o2i,
        gOhwI16i16o2i = zendnn_gOhwI16i16o2i,
        OdhwI16i16o2i = zendnn_OdhwI16i16o2i,
        gOdhwI16i16o2i = zendnn_gOdhwI16i16o2i,
        OwI16i16o4i = zendnn_OwI16i16o4i,
        gOwI16i16o4i = zendnn_gOwI16i16o4i,
        OhwI16i16o4i = zendnn_OhwI16i16o4i,
        gOhwI16i16o4i = zendnn_gOhwI16i16o4i,
        OdhwI16i16o4i = zendnn_OdhwI16i16o4i,
        gOdhwI16i16o4i = zendnn_gOdhwI16i16o4i,
        OwI16i32o2i = zendnn_OwI16i32o2i,
        OwI16i32o4i = zendnn_OwI16i32o4i,
        OwI16i48o2i = zendnn_OwI16i48o2i,
        OwI16i48o4i = zendnn_OwI16i48o4i,
        OwI16i64o2i = zendnn_OwI16i64o2i,
        OwI16i64o4i = zendnn_OwI16i64o4i,
        gOwI16i32o2i = zendnn_gOwI16i32o2i,
        gOwI16i32o4i = zendnn_gOwI16i32o4i,
        gOwI16i48o2i = zendnn_gOwI16i48o2i,
        gOwI16i48o4i = zendnn_gOwI16i48o4i,
        gOwI16i64o2i = zendnn_gOwI16i64o2i,
        gOwI16i64o4i = zendnn_gOwI16i64o4i,
        OhwI16i32o2i = zendnn_OhwI16i32o2i,
        OhwI16i32o4i = zendnn_OhwI16i32o4i,
        OhwI16i48o2i = zendnn_OhwI16i48o2i,
        OhwI16i48o4i = zendnn_OhwI16i48o4i,
        OhwI16i64o2i = zendnn_OhwI16i64o2i,
        OhwI16i64o4i = zendnn_OhwI16i64o4i,
        gOhwI16i32o2i = zendnn_gOhwI16i32o2i,
        gOhwI16i32o4i = zendnn_gOhwI16i32o4i,
        gOhwI16i48o2i = zendnn_gOhwI16i48o2i,
        gOhwI16i48o4i = zendnn_gOhwI16i48o4i,
        gOhwI16i64o2i = zendnn_gOhwI16i64o2i,
        gOhwI16i64o4i = zendnn_gOhwI16i64o4i,
        OdhwI16i32o2i = zendnn_OdhwI16i32o2i,
        OdhwI16i32o4i = zendnn_OdhwI16i32o4i,
        OdhwI16i48o2i = zendnn_OdhwI16i48o2i,
        OdhwI16i48o4i = zendnn_OdhwI16i48o4i,
        OdhwI16i64o2i = zendnn_OdhwI16i64o2i,
        OdhwI16i64o4i = zendnn_OdhwI16i64o4i,
        gOdhwI16i32o2i = zendnn_gOdhwI16i32o2i,
        gOdhwI16i32o4i = zendnn_gOdhwI16i32o4i,
        gOdhwI16i48o2i = zendnn_gOdhwI16i48o2i,
        gOdhwI16i48o4i = zendnn_gOdhwI16i48o4i,
        gOdhwI16i64o2i = zendnn_gOdhwI16i64o2i,
        gOdhwI16i64o4i = zendnn_gOdhwI16i64o4i,
        aBdeC16c16b2c = zendnn_aBdeC16c16b2c,
        aBdefC16c16b2c = zendnn_aBdefC16c16b2c,
        AcdB16b16a4b = zendnn_AcdB16b16a4b,
        AcdeB16b16a2b = zendnn_AcdeB16b16a2b,
        hwioG16g = zendnn_hwioG16g,
        ABc4a2b = zendnn_ABc4a2b,
        ABc8a2b = zendnn_ABc8a2b,
        ABcd4a2b = zendnn_ABcd4a2b,
        ABcde4a2b = zendnn_ABcde4a2b,
        ABcde8a2b = zendnn_ABcde8a2b,
        ABcd4a8b8a2b = zendnn_ABcd4a8b8a2b,
        NCdhw40n32c = zendnn_NCdhw40n32c,
        NChw40n32c = zendnn_NChw40n32c,
        NCw40n32c = zendnn_NCw40n32c,
        OIdhw4o8i8o2i = zendnn_OIdhw4o8i8o2i,
        OIhw4o8i8o2i = zendnn_OIhw4o8i8o2i,
        OIw4o8i8o2i = zendnn_OIw4o8i8o2i,
        gOIdhw4o8i8o2i = zendnn_gOIdhw4o8i8o2i,
        gOIhw4o8i8o2i = zendnn_gOIhw4o8i8o2i,
        gOIw4o8i8o2i = zendnn_gOIw4o8i8o2i,
        IOdhw4i8o8i2o = zendnn_IOdhw4i8o8i2o,
        IOhw4i8o8i2o = zendnn_IOhw4i8o8i2o,
        IOw4i8o8i2o = zendnn_IOw4i8o8i2o,
        gIOdhw4i8o8i2o = zendnn_gIOdhw4i8o8i2o,
        gIOhw4i8o8i2o = zendnn_gIOhw4i8o8i2o,
        gIOw4i8o8i2o = zendnn_gIOw4i8o8i2o,
        aBCd8b2c = zendnn_aBCd8b2c,
        ABcde40a16b = zendnn_ABcde40a16b,
        ABcde40a32b = zendnn_ABcde40a32b,
        aBCde8b2c = zendnn_aBCde8b2c,
        ABcde4a8b8a2b = zendnn_ABcde4a8b8a2b,
        ABc4a8b8a2b = zendnn_ABc4a8b8a2b,
        aBCdef4b8c8b2c = zendnn_aBCdef4b8c8b2c,
        aBCde4b8c8b2c = zendnn_aBCde4b8c8b2c,
        aBCd4b8c8b2c = zendnn_aBCd4b8c8b2c,
        BAcde4b8a8b2a = zendnn_BAcde4b8a8b2a,
        BAcd4b8a8b2a = zendnn_BAcd4b8a8b2a,
        BAc4b8a8b2a = zendnn_BAc4b8a8b2a,
        aCBdef4c8b8c2b = zendnn_aCBdef4c8b8c2b,
        aCBde4c8b8c2b = zendnn_aCBde4c8b8c2b,
        aCBd4c8b8c2b = zendnn_aCBd4c8b8c2b,
        aBCdef8b2c = zendnn_aBCdef8b2c,
        AB32a16b = zendnn_AB32a16b,
        AB32a32b = zendnn_AB32a32b,
        BA4b8a8b2a = zendnn_BA4b8a8b2a,
        BA4b8a8b4a = zendnn_BA4b8a8b4a,
        aBC32b16c = zendnn_aBC32b16c,
        aBC32b32c = zendnn_aBC32b32c,
        aCB4c8b8c2b = zendnn_aCB4c8b8c2b,
        aCB4c8b8c4b = zendnn_aCB4c8b8c4b,
        ABc2b8a16b4a = zendnn_ABc2b8a16b4a,
        ABcd2b8a16b4a = zendnn_ABcd2b8a16b4a,
        ABcde2b8a16b4a = zendnn_ABcde2b8a16b4a,
        ABc2a8b16a4b = zendnn_ABc2a8b16a4b,
        ABc2a8b16a2b = zendnn_ABc2a8b16a2b,
        ABc2b32a8b = zendnn_ABc2b32a8b,
        ABcd2a8b16a4b = zendnn_ABcd2a8b16a4b,
        ABcd2a8b16a2b = zendnn_ABcd2a8b16a2b,
        aCBd2c8b16c2b = zendnn_aCBd2c8b16c2b,
        ABcd2b32a8b = zendnn_ABcd2b32a8b,
        aBCd2c8b16c2b = zendnn_aBCd2c8b16c2b,
        ABcde2a8b16a4b = zendnn_ABcde2a8b16a4b,
        ABcde2a8b16a2b = zendnn_ABcde2a8b16a2b,
        aCBde2c8b16c2b = zendnn_aCBde2c8b16c2b,
        ABcde2b32a8b = zendnn_ABcde2b32a8b,
        aBC2b8c16b2c = zendnn_aBC2b8c16b2c,
        aBCd2b8c16b2c = zendnn_aBCd2b8c16b2c,
        aBCde2b8c16b2c = zendnn_aBCde2b8c16b2c,
        aBCdef2b8c16b2c = zendnn_aBCdef2b8c16b2c,
        BAcde2b8a16b4a = zendnn_BAcde2b8a16b4a,
        BAcd2b8a16b4a = zendnn_BAcd2b8a16b4a,
        BAc2b8a16b4a = zendnn_BAc2b8a16b4a,
        BAcde2b8a16b2a = zendnn_BAcde2b8a16b2a,
        BAcd2b8a16b2a = zendnn_BAcd2b8a16b2a,
        BAc2b8a16b2a = zendnn_BAc2b8a16b2a,
        aBCde2c8b16c2b = zendnn_aBCde2c8b16c2b,
        aBCdef2c8b16c2b = zendnn_aBCdef2c8b16c2b,
        aCBdef2c8b16c2b = zendnn_aCBdef2c8b16c2b,
        aBCd2b8c16b4c = zendnn_aBCd2b8c16b4c,
        aBCde2b8c16b4c = zendnn_aBCde2b8c16b4c,
        NCdhw40n16c = zendnn_NCdhw40n16c,
        NCw40n16c = zendnn_NCw40n16c,
        NChw40n16c = zendnn_NChw40n16c,
        NCw2c32n8c = zendnn_NCw2c32n8c,
        NChw2c32n8c = zendnn_NChw2c32n8c,
        NCdhw2c32n8c = zendnn_NCdhw2c32n8c,
        OIw2i8o16i4o = zendnn_OIw2i8o16i4o,
        OIhw2i8o16i4o = zendnn_OIhw2i8o16i4o,
        OIdhw2i8o16i4o = zendnn_OIdhw2i8o16i4o,
        OIw2o8i16o4i = zendnn_OIw2o8i16o4i,
        OIw2o8i16o2i = zendnn_OIw2o8i16o2i,
        IOw2i8o16i4o = zendnn_IOw2i8o16i4o,
        IOw2i8o16i2o = zendnn_IOw2i8o16i2o,
        OIhw2o8i16o4i = zendnn_OIhw2o8i16o4i,
        OIhw2o8i16o2i = zendnn_OIhw2o8i16o2i,
        IOhw2i8o16i4o = zendnn_IOhw2i8o16i4o,
        IOhw2i8o16i2o = zendnn_IOhw2i8o16i2o,
        OIdhw2o8i16o4i = zendnn_OIdhw2o8i16o4i,
        OIdhw2o8i16o2i = zendnn_OIdhw2o8i16o2i,
        IOdhw2i8o16i4o = zendnn_IOdhw2i8o16i4o,
        IOdhw2i8o16i2o = zendnn_IOdhw2i8o16i2o,
        gOIw2o8i16o2i = zendnn_gOIw2o8i16o2i,
        gIOw2i8o16i2o = zendnn_gIOw2i8o16i2o,
        gIOhw2i8o16i2o = zendnn_gIOhw2i8o16i2o,
        gIOdhw2i8o16i2o = zendnn_gIOdhw2i8o16i2o,
        gOIhw2o8i16o2i = zendnn_gOIhw2o8i16o2i,
        gOIdhw2o8i16o2i = zendnn_gOIdhw2o8i16o2i,
        gOIw2o8i16o4i = zendnn_gOIw2o8i16o4i,
        gOIhw2o8i16o4i = zendnn_gOIhw2o8i16o4i,
        BA4b8a16b2a = zendnn_BA4b8a16b2a,
        BA4b8a16b4a = zendnn_BA4b8a16b4a,
        aCB4c8b16c2b = zendnn_aCB4c8b16c2b,
        aCB4c8b16c4b = zendnn_aCB4c8b16c4b,
        aCB16c2b = zendnn_aCB16c2b,
        aCB16c4b = zendnn_aCB16c4b,
        BA16b2a = zendnn_BA16b2a,
        BA16b4a = zendnn_BA16b4a,
        aBC16b16c = zendnn_aBC16b16c,
        aBC16b32c = zendnn_aBC16b32c,
        AB16a16b = zendnn_AB16a16b,
        AB16a32b = zendnn_AB16a32b,
        ABcde16a16b2a = zendnn_ABcde16a16b2a,
        aBCdef16b16c2b = zendnn_aBCdef16b16c2b,
    };

    /// A memory descriptor.
    struct desc {
        friend struct memory;
        /// The underlying C API data structure.
        zendnn_memory_desc_t data;

        /// Constructs a zero (empty) memory descriptor. Such a memory
        /// descriptor can be used to indicate absence of an argument.
        desc() : data() {}

        /// Constructs a memory descriptor.
        ///
        /// @note
        ///     The logical order of dimensions corresponds to the `abc...`
        ///     format tag, and the physical meaning of the dimensions depends
        ///     both on the primitive that would operate on this memory and
        ///     the operation context.
        ///
        /// @param adims Tensor dimensions.
        /// @param adata_type Data precision/type.
        /// @param aformat_tag Memory format tag.
        /// @param is_memory_const A Flag to indicate if memory will remain
        ///     constant(Required for weight caching).
        /// @param is_inplace A flag to indicate if the memory can be
        ///     overwritten by the reordered memory.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case a
        ///     zero memory descriptor will be constructed. This flag is
        ///     optional and defaults to false.
        desc(const dims &adims, data_type adata_type, format_tag aformat_tag,
             bool is_memory_const = true, bool is_inplace = true,
             bool allow_empty = false)
            : data() {
            validate_dims(adims);
            zendnnInfo(ZENDNN_APILOG, "Memory create");
            zendnn_status_t status = zendnn_memory_desc_init_by_tag(&data,
                                     (int)adims.size(), adims.data(), convert_to_c(adata_type),
                                     convert_to_c(aformat_tag), is_memory_const, is_inplace);
            if (!allow_empty)
                error::wrap_c_api(status,
                                  "could not construct a memory descriptor using a "
                                  "format tag");
        }

        /// Constructs a memory descriptor by strides.
        ///
        /// @note
        ///     The logical order of dimensions corresponds to the `abc...`
        ///     format tag, and the physical meaning of the dimensions depends
        ///     both on the primitive that would operate on this memory and
        ///     the operation context.
        ///
        /// @param adims Tensor dimensions.
        /// @param adata_type Data precision/type.
        /// @param strides Strides for each dimension.
        /// @param is_memory_const A Flag to indicate if memory will remain
        ///     constant(Required for weight caching).
        /// @param is_inplace A flag to indicate if the memory can be
        ///     overwritten by the reordered memory.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case a
        ///     zero memory descriptor will be constructed. This flag is
        ///     optional and defaults to false.
        desc(const dims &adims, data_type adata_type, const dims &strides,
             bool is_memory_const = true, bool is_inplace = true,
             bool allow_empty = false)
            : data() {
            validate_dims(adims);
            if (!strides.empty()) {
                validate_dims(strides, (int)adims.size());
            }
            zendnnInfo(ZENDNN_APILOG, "Memory create - strides");
            zendnn_status_t status = zendnn_memory_desc_init_by_strides(&data,
                                     (int)adims.size(), adims.data(), convert_to_c(adata_type),
                                     strides.empty() ? nullptr : &strides[0], is_memory_const,
                                     is_inplace);
            if (!allow_empty)
                error::wrap_c_api(status,
                                  "could not construct a memory descriptor using "
                                  "strides");
        }

        /// Constructs a memory descriptor from a C API data structure.
        ///
        /// @param data A C API ::zendnn_memory_desc_t structure.
        desc(const zendnn_memory_desc_t &data) : data(data) {}

        /// Constructs a memory descriptor for a region inside an area
        /// described by this memory descriptor.
        //
        /// @param adims Sizes of the region.
        /// @param offsets Offsets to the region from the encompassing
        ///     memory object in each dimension.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case a
        ///     zero memory descriptor will be returned. This flag is optional
        ///     and defaults to false.
        /// @returns A memory descriptor for the region.
        desc submemory_desc(const dims &adims, const dims &offsets,
                            bool allow_empty = false) const {
            validate_dims(adims, data.ndims);
            validate_dims(offsets, data.ndims);
            zendnn_memory_desc_t sub_md = zendnn_memory_desc_t();
            zendnn_status_t status = zendnn_memory_desc_init_submemory(
                                         &sub_md, &data, adims.data(), offsets.data());
            if (!allow_empty) {
                error::wrap_c_api(status, "could not construct a sub-memory");
            }
            return desc(sub_md);
        }

        /// Constructs a memory descriptor by reshaping an existing one. The
        /// new memory descriptor inherits the data type. This operation is
        /// valid only for memory descriptors that have format_kind set to
        /// #zendnn::memory::format_kind::blocked or
        /// #zendnn::memory::format_kind::any.
        ///
        /// The operation ensures that the transformation of the physical memory
        /// format corresponds to the transformation of the logical dimensions.
        /// If such transformation is impossible, the function either throws an
        /// exception (default) or returns a zero memory descriptor depending on
        /// the `allow_empty` flag.
        ///
        /// The reshape operation can be described as a combination of the
        /// following basic operations:
        /// 1. Add a dimension of size `1`. This is always possible.
        /// 2. Remove a dimension of size `1`. This is possible only if the
        ///    dimension has no padding (i.e.
        ///    `padded_dims[dim] == dims[dim] && dims[dim] == 1`).
        /// 3. Split a dimension into multiple ones. This is possible only if
        ///    the product of all tensor dimensions stays constant and the
        ///    dimension being split does not have padding (i.e.
        ///    `padded_dims[dim] = dims[dim]`).
        /// 4. Join multiple consecutive dimensions into a single one. As in
        ///    the cases above, this requires that the dimensions do not have
        ///    padding and that the memory format is such that in physical
        ///    memory these dimensions are dense and have the same order as
        ///    their logical counterparts. This also assumes that these
        ///    dimensions are not blocked.
        ///    - Here, 'dense' means:
        ///      `stride for dim[i] == (stride for dim[i + 1]) * dim[i + 1]`;
        ///    - And 'same order' means:
        ///      `i < j` if and only if `stride for dim[j] <= stride for dim[i]`.
        ///
        /// @warning
        ///     Some combinations of physical memory layout and/or offsets or
        ///     dimensions may result in a failure to make a reshape.
        ///
        /// @param adims New dimensions. The product of dimensions must
        ///     remain constant.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case a
        ///     zero memory descriptor will be returned. This flag is optional
        ///     and defaults to false.
        /// @returns A new memory descriptor with new dimensions.
        desc reshape(const dims &adims, bool allow_empty = false) const {
            if (data.ndims) {
                validate_dims(adims, 1);
            }
            zendnn_memory_desc_t out_md = zendnn_memory_desc_t();
            zendnn_status_t status = zendnn_memory_desc_reshape(
                                         &out_md, &data, (int)adims.size(), adims.data());
            if (!allow_empty)
                error::wrap_c_api(
                    status, "could not reshape a memory descriptor");
            return desc(out_md);
        }

        /// Constructs a memory descriptor by permuting axes in an existing
        /// one.
        ///
        /// The physical memory layout representation is adjusted accordingly
        /// to maintain the consistency between the logical and physical parts
        /// of the memory descriptor. The new memory descriptor inherits the
        /// data type.
        ///
        /// The new memory descriptor inherits the data type. This operation is
        /// valid only for memory descriptors that have format_kind set to
        /// #zendnn::memory::format_kind::blocked or
        /// #zendnn::memory::format_kind::any.
        ///
        /// The logical axes will be permuted in the following manner:
        /// @code
        /// for (i = 0; i < ndims(); i++)
        ///     new_desc.dims()[permutation[i]] = dims()[i];
        /// @endcode
        ///
        /// Example:
        /// @code
        ///     std::vector<int> permutation = {1, 0}; // swap the first and
        ///                                            // the second axes
        ///     zendnn::memory::desc in_md(
        ///             {2, 3}, data_type, memory::format_tag::ab);
        ///     zendnn::memory::desc expect_out_md(
        ///             {3, 2}, data_type, memory::format_tag::ba);
        ///
        ///     assert(in_md.permute_axes(permutation) == expect_out_md);
        /// @endcode
        ///
        /// @param permutation Axes permutation.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case a
        ///     zero memory descriptor will be returned. This flag is optional
        ///     and defaults to false.
        /// @returns A new memory descriptor with new dimensions.
        desc permute_axes(const std::vector<int> &permutation,
                          bool allow_empty = false) const {
            validate_dims(permutation, data.ndims);
            zendnn_memory_desc_t out_md = zendnn_memory_desc_t();
            zendnn_status_t status = zendnn_memory_desc_permute_axes(
                                         &out_md, &data, permutation.data());
            if (!allow_empty)
                error::wrap_c_api(status,
                                  "could not permute axes of a memory descriptor");
            return desc(out_md);
        }

        /// Returns the data type of the memory descriptor.
        /// @returns The data type.
        memory::data_type data_type() const {
            return static_cast<memory::data_type>(data.data_type);
        }

        /// Returns dimensions of the memory descriptor.
        ///
        /// Potentially expensive due to the data copy involved.
        /// @returns A copy of the dimensions vector.
        memory::dims dims() const {
            return memory::dims(data.dims, data.dims + data.ndims);
        }

        /// Returns size of the memory descriptor in bytes.
        /// @returns The number of bytes required to allocate a memory buffer
        ///     for the memory object described by this memory descriptor
        ///     including the padding area.
        size_t get_size() const {
            return zendnn_memory_desc_get_size(&data);
        }

        /// Checks whether the memory descriptor is zero (empty).
        /// @returns @c true if the memory descriptor describes an empty
        ///     memory and @c false otherwise.
        bool is_zero() const {
            return data.ndims == 0;
        }

        /// An equality operator.
        /// @param other Another memory descriptor.
        /// @returns Whether this and the other memory descriptors have
        ///     the same format tag, dimensions, strides, blocking, etc.
        bool operator==(const desc &other) const {
            return zendnn_memory_desc_equal(&data, &other.data) != 0;
        }

        /// An inequality operator.
        /// @param other Another memory descriptor.
        /// @returns Whether this and the other memory descriptors describe
        ///     different memory.
        bool operator!=(const desc &other) const {
            return !operator==(other);
        }

        /// Checks whether the object is not empty.
        ///
        /// @returns Whether the object is not empty.
        explicit operator bool() const {
            return data.ndims != 0;
        }
    };

    /// Default constructor.
    ///
    /// Constructs an empty memory object, which can be used to indicate
    /// absence of a parameter.
    memory() = default;

    /// Constructs a memory object.
    ///
    /// Unless @p handle is equal to #ZENDNN_MEMORY_NONE, the constructed memory
    /// object will have the underlying buffer set. In this case, the buffer
    /// will be initialized as if #zendnn::memory::set_data_handle() had been
    /// called.
    ///
    /// @sa memory::set_data_handle()
    ///
    /// @param md Memory descriptor.
    /// @param aengine Engine to store the data on.
    /// @param handle Handle of the memory buffer to use.
    ///     - A pointer to the user-allocated buffer. In this case the library
    ///       doesn't own the buffer.
    ///     - The #ZENDNN_MEMORY_ALLOCATE special value. Instructs the library to
    ///       allocate the buffer for the memory object. In this case the
    ///       library owns the buffer.
    ///     - #ZENDNN_MEMORY_NONE to create zendnn::memory without an underlying
    ///       buffer.
    memory(const desc &md, const engine &aengine, void *handle) {
        zendnn_memory_t result;
        error::wrap_c_api(
            zendnn_memory_create(&result, &md.data, aengine.get(), handle),
            "could not create a memory object");
        reset(result);
    }

    /// Constructs a memory object.
    ///
    /// The underlying buffer for the memory will be allocated by the library.
    ///
    /// @param md Memory descriptor.
    /// @param aengine Engine to store the data on.
    memory(const desc &md, const engine &aengine)
        : memory(md, aengine, ZENDNN_MEMORY_ALLOCATE) {}

    /// Returns the associated memory descriptor.
    desc get_desc() const {
        const zendnn_memory_desc_t *cdesc;
        error::wrap_c_api(zendnn_memory_get_memory_desc(get(), &cdesc),
                          "could not get a memory descriptor from a memory object");
        return desc(*cdesc);
    }

    /// Returns the associated engine.
    engine get_engine() const {
        zendnn_engine_t c_engine;
        error::wrap_c_api(zendnn_memory_get_engine(get(), &c_engine),
                          "could not get an engine from a memory object");
        return engine(c_engine, true);
    }

    /// Returns the underlying memory buffer.
    ///
    /// On the CPU engine, or when using USM, this is a pointer to the
    /// allocated memory.
    void *get_data_handle() const {
        void *handle;
        error::wrap_c_api(zendnn_memory_get_data_handle(get(), &handle),
                          "could not get a native handle from a memory object");
        return handle;
    }

    /// Sets the underlying memory buffer.
    ///
    /// This function may write zero values to the memory specified by the @p
    /// handle if the memory object has a zero padding area. This may be time
    /// consuming and happens each time this function is called. The
    /// operation is always blocking and the stream parameter is a hint.
    ///
    /// @note
    ///     The zero padding is required by memory objects created with
    ///     blocked memory format tags like #zendnn_aBcd8b when any of the
    ///     dimensions is not a multiple of the corresponding block size. For
    ///     "plain" formats like #zendnn::memory::format_tag::nchw or
    ///     #zendnn::memory::format_tag::nhwc zero padding area needs to be set
    ///     up explicitly when creating the corresponding memory descriptors.
    ///     See @ref dev_guide_understanding_memory_formats for more details.
    ///
    /// @note
    ///     Even when the memory object is used to hold values that stay
    ///     constant during the execution of the program (pre-packed weights
    ///     during inference, for example), the function will still write
    ///     zeroes to the padding area if it exists. Hence, the @p handle
    ///     parameter cannot and does not have a const qualifier.
    ///
    /// @param handle Memory buffer to use. On the CPU engine or when USM is
    ///     used, the memory buffer is a pointer to the actual data. For OpenCL
    ///     it is a cl_mem. It must have at least
    ///     #zendnn::memory::desc::get_size() bytes allocated.
    /// @param astream Stream to use to execute padding in.
    void set_data_handle(void *handle, const stream &astream) const {
        error::wrap_c_api(zendnn_memory_set_data_handle_v2(
                              get(), handle, astream.get(true)),
                          "could not set native handle of a memory object");
    }

    /// Sets the underlying memory buffer.
    ///
    /// See documentation for
    /// #zendnn::memory::set_data_handle(void *, const stream &) const
    /// for more information.
    ///
    /// @param handle Memory buffer to use. For the CPU engine, the memory
    ///     buffer is a pointer to the actual data. For OpenCL it is a cl_mem.
    ///     It must have at least #zendnn::memory::desc::get_size() bytes
    ///     allocated.
    void set_data_handle(void *handle) const {
        error::wrap_c_api(
            zendnn_memory_set_data_handle_v2(get(), handle, nullptr),
            "could not set native handle of a memory object");
    }

    /// Maps a memory object and returns a host-side pointer to a memory
    /// buffer with a copy of its contents.
    ///
    /// Mapping enables read/write directly from/to the memory contents for
    /// engines that do not support direct memory access.
    ///
    /// Mapping is an exclusive operation - a memory object cannot be used in
    /// other operations until it is unmapped via #zendnn::memory::unmap_data()
    /// call.
    ///
    /// @note
    ///     Any primitives working with the memory should be completed before
    ///     the memory is mapped. Use #zendnn::stream::wait() to synchronize the
    ///     corresponding execution stream.
    ///
    /// @note
    ///     The map_data and unmap_data functions are provided mainly for
    ///     debug and testing purposes and their performance may be suboptimal.
    ///
    /// @tparam T Data type to return a pointer to.
    /// @returns Pointer to the mapped memory.
    template <typename T = void>
    T *map_data() const {
        void *mapped_ptr;
        error::wrap_c_api(zendnn_memory_map_data(get(), &mapped_ptr),
                          "could not map memory object data");
        return static_cast<T *>(mapped_ptr);
    }

    /// Unmaps a memory object and writes back any changes made to the
    /// previously mapped memory buffer.
    ///
    /// @note
    ///     The map_data and unmap_data functions are provided mainly for
    ///     debug and testing purposes and their performance may be
    ///     suboptimal.
    ///
    /// @param mapped_ptr A pointer previously returned by
    ///     #zendnn::memory::map_data().
    void unmap_data(void *mapped_ptr) const {
        error::wrap_c_api(zendnn_memory_unmap_data(get(), mapped_ptr),
                          "could not unmap memory object data");
    }

    static zendnn_data_type_t convert_to_c(data_type adata_type) {
        return static_cast<zendnn_data_type_t>(adata_type);
    }
    static zendnn_format_tag_t convert_to_c(format_tag format) {
        return static_cast<zendnn_format_tag_t>(format);
    }
};

inline bool operator==(zendnn_data_type_t a, memory::data_type b) {
    return a == memory::convert_to_c(b);
}
inline bool operator!=(zendnn_data_type_t a, memory::data_type b) {
    return !(a == b);
}
inline bool operator==(memory::data_type a, zendnn_data_type_t b) {
    return b == a;
}
inline bool operator!=(memory::data_type a, zendnn_data_type_t b) {
    return !(a == b);
}

inline bool operator==(zendnn_format_tag_t a, memory::format_tag b) {
    return a == memory::convert_to_c(b);
}
inline bool operator!=(zendnn_format_tag_t a, memory::format_tag b) {
    return !(a == b);
}
inline bool operator==(memory::format_tag a, zendnn_format_tag_t b) {
    return b == a;
}
inline bool operator!=(memory::format_tag a, zendnn_format_tag_t b) {
    return !(a == b);
}

/// @} zendnn_api_memory

/// @addtogroup zendnn_api_primitives
/// @{
/// @addtogroup zendnn_api_attributes Attributes
///
/// A container for parameters that extend primitives behavior.
///
/// @{

/// @cond DO_NOT_DOCUMENT_THIS
template <>
struct handle_traits<zendnn_post_ops_t> {
    static zendnn_status_t destructor(zendnn_post_ops_t p) {
        return zendnn_post_ops_destroy(p);
    }
};
/// @endcond

/// Post-ops.
///
/// Post-ops are computations executed after the main primitive computations
/// and are attached to the primitive via primitive attributes.
///
/// @sa @ref dev_guide_attributes_post_ops
///
struct post_ops : public handle<zendnn_post_ops_t> {
    using handle<zendnn_post_ops_t>::handle;

    /// Constructs an empty sequence of post-ops.
    post_ops() {
        zendnn_post_ops_t result;
        error::wrap_c_api(
            zendnn_post_ops_create(&result), "could not create post-ops");
        reset(result);
    }

    /// Returns the number of post-ops entries.
    int len() const {
        return zendnn_post_ops_len(get());
    }

    /// Returns the primitive kind of post-op at entry with a certain index.
    /// @param index Index of the post-op to return the kind for.
    /// @returns Primitive kind of the post-op at the specified index.
    primitive::kind kind(int index) const {
        error::wrap_c_api(index < len() ? zendnn_success : zendnn_invalid_arguments,
                          "post-ops index is out of range");
        return static_cast<primitive::kind>(
                   zendnn_post_ops_get_kind(get(), index));
    }

    /// Appends an accumulation (sum) post-op. Prior to accumulating the
    /// result, the previous value would be multiplied by a scaling factor
    /// @p scale.
    ///
    /// The kind of this post-op is #zendnn::primitive::kind::sum.
    ///
    /// This feature may improve performance for cases like residual learning
    /// blocks, where the result of convolution is accumulated to the
    /// previously computed activations. The parameter @p scale may be used
    /// for the integer-based computations when the result and previous
    /// activations have different logical scaling factors.
    ///
    /// In the simplest case when the accumulation is the only post-op,
    /// the computations will be `dst[:] := scale * dst[:] + op(...)`
    /// instead of `dst[:] := op(...)`.
    ///
    /// If @p data_type is specified, the original dst tensor will be
    /// reinterpreted as a tensor with the provided data type. Because it is a
    /// reinterpretation, data_type and dst data type should have the same size.
    /// As a result, computations will be `dst[:] <- scale *
    /// as_data_type(dst[:]) + op(...)` instead of `dst[:] <- op(...)`.
    ///
    /// @note
    ///     This post-op executes in-place and does not change the
    ///     destination layout.
    ///
    /// @param scale Scaling factor.
    /// @param data_type Data type.
    void append_sum(float scale = 1.f,
                    memory::data_type data_type = memory::data_type::undef) {
        if (data_type == memory::data_type::undef)
            error::wrap_c_api(zendnn_post_ops_append_sum(get(), scale),
                              "could not append a sum post-op");
        else
            error::wrap_c_api(zendnn_post_ops_append_sum_v2(get(), scale,
                              memory::convert_to_c(data_type)),
                              "could not append a sum post-op");
    }

    /// Appends an accumulation (sum) post-op. Prior to accumulating the
    /// result, the previous value will be will be reduced by zero point
    /// @p zero_point and multiplied by a scaling factor @p scale.
    ///
    /// The kind of this post-op is #zendnn::primitive::kind::sum.
    ///
    /// This feature may improve performance for cases like dequantize the
    /// asymmetrically quantized sum's src1 tensor to f32 domain before
    /// performing the sum operation by subtracting @p zero_point before the
    /// scaling.
    ///
    /// In the simplest case when the accumulation is the only post-op,
    /// the computations will be `dst[:] := scale * (dst[:] - zero_point) +
    /// op(...)` instead of `dst[:] := op(...)`.
    ///
    /// If @p data_type is specified, the original dst tensor will be
    /// reinterpreted as a tensor with the provided data type. Because it is a
    /// reinterpretation, data_type and dst data type should have the same size.
    /// As a result, computations will be `dst[:] <- scale *
    /// (as_data_type(dst[:]) - zero_point) + op(...)` instead of
    /// `dst[:] <- op(...)`.
    ///
    /// @note
    ///     This post-op executes in-place and does not change the
    ///     destination layout.
    ///
    /// @param scale Scaling factor.
    /// @param zero_point Zero point.
    /// @param data_type Data type.
    void append_sum(float scale, int32_t zero_point,
                    memory::data_type data_type = memory::data_type::undef) {
        error::wrap_c_api(zendnn_post_ops_append_sum_v3(get(), scale, zero_point,
                          memory::convert_to_c(data_type)),
                          "could not append a sum post-op");
    }

    /// Returns the parameters of an accumulation (sum) post-op.
    ///
    /// @param index Index of the sum post-op.
    /// @param scale Scaling factor of the sum post-op.
    void get_params_sum(int index, float &scale) const {
        error::wrap_c_api(zendnn_post_ops_get_params_sum(get(), index, &scale),
                          "could not get parameters of a sum post-op");
    }

    /// Returns the parameters of an accumulation (sum) post-op.
    ///
    /// @param index Index of the sum post-op.
    /// @param scale Scaling factor of the sum post-op.
    /// @param data_type Data type of the sum post-op.
    void get_params_sum(
        int index, float &scale, memory::data_type &data_type) const {
        zendnn_data_type_t c_data_type;
        error::wrap_c_api(zendnn_post_ops_get_params_sum_v2(
                              get(), index, &scale, &c_data_type),
                          "could not get parameters of a sum post-op");
        data_type = static_cast<memory::data_type>(c_data_type);
    }

    /// Returns the parameters of an accumulation (sum) post-op.
    ///
    /// @param index Index of the sum post-op.
    /// @param scale Scaling factor of the sum post-op.
    /// @param zero_point Single scalar int32_t value of zeropoint.
    /// @param data_type Data type of the sum post-op.
    void get_params_sum(int index, float &scale, int32_t &zero_point,
                        memory::data_type &data_type) const {
        zendnn_data_type_t c_data_type;
        error::wrap_c_api(zendnn_post_ops_get_params_sum_v3(get(), index, &scale,
                          &zero_point, &c_data_type),
                          "could not get parameters of a sum post-op");
        data_type = static_cast<memory::data_type>(c_data_type);
    }

    /// Appends an elementwise post-op.
    ///
    /// The kind of this post-op is #zendnn::primitive::kind::eltwise.
    ///
    /// In the simplest case when the elementwise is the only post-op, the
    /// computations would be `dst[:] := scale * eltwise_op (op(...))` instead
    /// of `dst[:] <- op(...)`, where eltwise_op is configured with the given
    /// parameters.
    ///
    /// @param scale Scaling factor.
    /// @param aalgorithm Elementwise algorithm.
    /// @param alpha Alpha parameter for the elementwise algorithm.
    /// @param beta Beta parameter for the elementwise algorithm.
    void append_eltwise(
        float scale, algorithm aalgorithm, float alpha, float beta) {
        error::wrap_c_api(zendnn_post_ops_append_eltwise(get(), scale,
                          convert_to_c(aalgorithm), alpha, beta),
                          "could not append an elementwise post-op");
    }

    /// Returns parameters of an elementwise post-op.
    ///
    /// @param index Index of the post-op.
    /// @param scale Output scaling factor.
    /// @param aalgorithm Output elementwise algorithm kind.
    /// @param alpha Output alpha parameter for the elementwise algorithm.
    /// @param beta Output beta parameter for the elementwise algorithm.
    void get_params_eltwise(int index, float &scale, algorithm &aalgorithm,
                            float &alpha, float &beta) const {
        zendnn_alg_kind_t c_alg;
        error::wrap_c_api(zendnn_post_ops_get_params_eltwise(
                              get(), index, &scale, &c_alg, &alpha, &beta),
                          "could not get parameters of an elementwise post-op");
        aalgorithm = static_cast<zendnn::algorithm>(c_alg);
    }

    /// Appends a depthwise post-op convolution.
    ///
    /// This post-op can only be fused with a 2D 1x1 convolution (convolution
    /// with weights spatial dimension equal to 1 i.e., kh=kw=1).
    ///
    /// The kind of this post-op is #zendnn_convolution.
    ///
    /// The number of outputs for primitive remain same as before. The output
    /// spatial size can be derived as below:
    ///
    /// output_height = ceil(output_height_1x1_convolution, stride)
    /// output_width = ceil(output_width_1x1_convolution, stride)
    ///
    /// See @ref dev_guide_attributes_post_ops_depthwise and
    /// @ref dev_guide_attributes_post_ops_depthwise_fusion for more info.
    ///
    /// @param weights_data_type Weights data type of depthwise post-op
    /// @param bias_data_type Bias data type of depthwise post-op
    /// @param dst_data_type Output data type of depthwise post-op
    /// @param kernel_size Size of kernel of depthwise post-op
    /// @param stride_size Size of stride of depthwise post-op
    /// @param padding_l_size Size of left and top paddings of depthwise post-op
    /// @param mask Output scaling factors correspondence mask that defines the
    ///     correspondence between the output tensor dimensions and the
    ///     @p scales array. The set i-th bit indicates that a dedicated output
    ///     scaling factor is used for each index along that dimension. The mask
    ///     value of 0 implies a common scaling factor for the whole output
    ///     tensor.
    /// @param scales Output pointer to a constant array of float scaling
    ///     factors.
    void append_dw(memory::data_type weights_data_type,
                   memory::data_type bias_data_type, memory::data_type dst_data_type,
                   memory::dim kernel_size, memory::dim stride_size,
                   memory::dim padding_l_size, int mask,
                   const std::vector<float> &scales) {

        error::wrap_c_api(zendnn_post_ops_append_dw(get(),
                          memory::convert_to_c(weights_data_type),
                          memory::convert_to_c(bias_data_type),
                          memory::convert_to_c(dst_data_type),
                          kernel_size, stride_size, padding_l_size,
                          scales.size(), mask, scales.data()),
                          "could not append depthwise post-op");
    }

    /// Returns the parameters of an depthwise post-op with stride 1.
    ///
    /// @param index Index of the elementwise post-op.
    /// @param weights_data_type Weights data type of depthwise post-op
    /// @param bias_data_type Bias data type of depthwise post-op
    /// @param dst_data_type Output data type of depthwise post-op
    /// @param mask Output scaling factors correspondence mask that defines the
    ///     correspondence between the output tensor dimensions and the
    ///     @p scales array. The set i-th bit indicates that a dedicated output
    ///     scaling factor is used for each index along that dimension. The mask
    ///     value of 0 implies a common scaling factor for the whole output
    ///     tensor.
    /// @param scales Output pointer to a constant array of float scaling
    ///     factors.
    void get_params_dw(int index, memory::data_type &weights_data_type,
                       memory::data_type &bias_data_type, memory::data_type &dst_data_type,
                       memory::dim &kernel_size, memory::dim &stride_size,
                       memory::dim &padding_l_size, int &mask,
                       std::vector<float> &scales) const {

        zendnn_data_type_t c_weights_data_type;
        zendnn_data_type_t c_bias_data_type;
        zendnn_data_type_t c_dst_data_type;
        zendnn_dim_t c_kernel_size;
        zendnn_dim_t c_stride_size;
        zendnn_dim_t c_padding_l_size;
        zendnn_dim_t count;
        int c_mask;
        const float *c_scales;
        error::wrap_c_api(
            zendnn_post_ops_get_params_dw(get(), index, &c_weights_data_type,
                                          &c_bias_data_type, &c_dst_data_type, &c_kernel_size,
                                          &c_stride_size, &c_padding_l_size, &count, &c_mask,
                                          &c_scales),
            "could not get parameters of depthwise post-op");

        weights_data_type = static_cast<memory::data_type>(c_weights_data_type);
        bias_data_type = static_cast<memory::data_type>(c_bias_data_type);
        dst_data_type = static_cast<memory::data_type>(c_dst_data_type);
        kernel_size = c_kernel_size;
        stride_size = c_stride_size;
        padding_l_size = c_padding_l_size;
        scales.resize(count);

        mask = c_mask;
        for (zendnn_dim_t c = 0; c < count; ++c) {
            scales[c] = c_scales[c];
        }
        return;
    }

    /// Appends a depthwise post-op convolution with stride 1.
    ///
    /// This post-op can only be fused with a 2D 1x1 convolution (convolution
    /// with weights spatial dimension equal to 1 i.e., kh=kw=1).
    ///
    /// The kind of this post-op is #zendnn_convolution.
    ///
    /// The number of outputs for primitive remain same as before. The output
    /// size remain same as the original primitive due to stride=1.
    ///
    /// The Post-op can be defined as:
    ///
    ///      dst[:] <- scales * (conv_dw(conv_1x1))
    ///
    /// See @ref dev_guide_attributes_post_ops_depthwise and
    /// @ref dev_guide_attributes_post_ops_depthwise_fusion for more info.
    ///
    /// @param weights_data_type Weights data type of depthwise post-op
    /// @param bias_data_type Bias data type of depthwise post-op
    /// @param dst_data_type Output data type of depthwise post-op
    /// @param mask Output scaling factors correspondence mask that defines the
    ///     correspondence between the output tensor dimensions and the
    ///     @p scales array. The set i-th bit indicates that a dedicated output
    ///     scaling factor is used for each index along that dimension. The mask
    ///     value of 0 implies a common scaling factor for the whole output
    ///     tensor.
    /// @param scales Output pointer to a constant array of float scaling
    ///     factors.
    void append_dw_k3s1p1(memory::data_type weights_data_type,
                          memory::data_type bias_data_type, memory::data_type dst_data_type,
                          int mask, const std::vector<float> &scales) {

        append_dw(weights_data_type, bias_data_type, dst_data_type, 3, 1, 1,
                  mask, scales);
    }

    /// Returns the parameters of an depthwise post-op with stride 1.
    ///
    /// @param index Index of the elementwise post-op.
    /// @param weights_data_type Weights data type of depthwise post-op
    /// @param bias_data_type Bias data type of depthwise post-op
    /// @param dst_data_type Output data type of depthwise post-op
    /// @param mask Output scaling factors correspondence mask that defines the
    ///     correspondence between the output tensor dimensions and the
    ///     @p scales array. The set i-th bit indicates that a dedicated output
    ///     scaling factor is used for each index along that dimension. The mask
    ///     value of 0 implies a common scaling factor for the whole output
    ///     tensor.
    /// @param scales Output pointer to a constant array of float scaling
    ///     factors.
    void get_params_dw_k3s1p1(int index, memory::data_type &weights_data_type,
                              memory::data_type &bias_data_type, memory::data_type &dst_data_type,
                              int &mask, std::vector<float> &scales) const {

        memory::dim kernel_size;
        memory::dim stride_size;
        memory::dim padding_l_size;
        get_params_dw(index, weights_data_type, bias_data_type, dst_data_type,
                      kernel_size, stride_size, padding_l_size, mask, scales);
    }

    /// Appends a depthwise post-op convolution with stride 2.
    ///
    /// This post-op can only be fused with a 2D 1x1 convolution (convolution
    /// with weights spatial dimension equal to 1 i.e., kh=kw=1).
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
    /// @param weights_data_type Weights data type of depthwise post-op
    /// @param bias_data_type Bias data type of depthwise post-op
    /// @param dst_data_type Output data type of depthwise post-op
    /// @param mask Output scaling factors correspondence mask that defines the
    ///     correspondence between the output tensor dimensions and the
    ///     @p scales array. The set i-th bit indicates that a dedicated output
    ///     scaling factor is used for each index along that dimension. The mask
    ///     value of 0 implies a common scaling factor for the whole output
    ///     tensor.
    /// @param scales Output pointer to a constant array of float scaling
    ///     factors.
    /// @returns #zendnn_success on success and a status describing the error
    ///     otherwise
    void append_dw_k3s2p1(memory::data_type weights_data_type,
                          memory::data_type bias_data_type, memory::data_type dst_data_type,
                          int mask, const std::vector<float> &scales) {
        append_dw(weights_data_type, bias_data_type, dst_data_type, 3, 2, 1,
                  mask, scales);
    }

    /// Returns the parameters of an depthwise post-op with stride 2.
    ///
    /// @param index Index of the elementwise post-op.
    /// @param weights_data_type Weights data type of depthwise post-op
    /// @param bias_data_type Bias data type of depthwise post-op
    /// @param dst_data_type Output data type of depthwise post-op
    /// @param mask Output scaling factors correspondence mask that defines the
    ///     correspondence between the output tensor dimensions and the
    ///     @p scales array. The set i-th bit indicates that a dedicated output
    ///     scaling factor is used for each index along that dimension. The mask
    ///     value of 0 implies a common scaling factor for the whole output
    ///     tensor.
    /// @param scales Output pointer to a constant array of float scaling
    ///     factors.
    void get_params_dw_k3s2p1(int index, memory::data_type &weights_data_type,
                              memory::data_type &bias_data_type, memory::data_type &dst_data_type,
                              int &mask, std::vector<float> &scales) const {

        memory::dim kernel_size;
        memory::dim stride_size;
        memory::dim padding_l_size;
        get_params_dw(index, weights_data_type, bias_data_type, dst_data_type,
                      kernel_size, stride_size, padding_l_size, mask, scales);
    }

    /// Appends a binary post-op.
    ///
    /// The kind of this post operation is #zendnn_binary.
    ///
    /// In the simplest case when the binary is the only post operation, the
    /// computations would be:
    ///
    ///     dst[:] <- binary_op (dst[:], another_input[:])
    ///
    /// where binary_op is configured with the given parameters. binary_op
    /// supports broadcast semantics for a second operand.
    ///
    /// @param aalgorithm Binary algorithm for the post-op.
    /// @param src1_desc Memory descriptor of a second operand.
    void append_binary(algorithm aalgorithm, const memory::desc &src1_desc) {
        error::wrap_c_api(zendnn_post_ops_append_binary(get(),
                          convert_to_c(aalgorithm), &src1_desc.data),
                          "could not append a binary post-op");
    }

    /// Returns the parameters of a binary post-op.
    ///
    /// @param index Index of the binary post-op.
    /// @param aalgorithm Output binary algorithm kind.
    /// @param src1_desc Output memory descriptor of a second operand.
    void get_params_binary(
        int index, algorithm &aalgorithm, memory::desc &src1_desc) const {
        zendnn_alg_kind_t c_alg;
        const zendnn_memory_desc_t *data;
        error::wrap_c_api(
            zendnn_post_ops_get_params_binary(get(), index, &c_alg, &data),
            "could not get parameters of a binary post-op");
        aalgorithm = static_cast<zendnn::algorithm>(c_alg);
        src1_desc.data = *data;
    }

    /// Appends a prelu forward post-op.
    ///
    /// The kind of this post-op is #zendnn::primitive::kind::prelu.
    ///
    /// The post-op can be defined as:
    ///
    ///      dst[:] <- prelu(dst[:], weights[:])
    ///      prelu:
    ///      dst[:] <- dst[:] if dst[:] > 0
    ///      dst[:] <- dst[:] * weights[:] if dst[:] <= 0
    ///
    ///
    /// Example usage:
    /// @code
    ///     int mb = 32, oc = 32,
    ///         oh = 14, ow = 14; // convolution output params
    ///     // unique weights per output channel
    ///     vector<float> weights = { ... };
    ///     int oc_dim = 1; // mb_dim = 0, channel_dim = 1, height_dim = 2, ...
    ///
    ///     // construct a convolution descriptor
    ///     zendnn::convolution::desc conv_d;
    ///
    ///     zendnn::primitive_attr attr;
    ///     attr.append_prelu(1 << oc_dim);
    ///
    ///     zendnn::primitive_desc conv_pd(conv_d, attr, engine);
    ///     memory prelu_weights({{1}, dt::f32, {1}}, eng, weights.data());
    ///
    ///     std::unordered_map<int, memory> conv_args;
    ///
    ///     conv_args.insert(
    ///      {ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(0) | ZENDNN_ARG_WEIGHTS, prelu_weights})

    /// @note
    ///     The order of dimensions does not depend on how elements are laid
    ///     out in memory. For example:
    ///     - for a 2D CNN activations tensor the order is always (n, c)
    ///     - for a 4D CNN activations tensor the order is always (n, c, h, w)
    ///     - for a 5D CNN weights tensor the order is always
    ///        (g, oc, ic, kh, kw)
    ///
    ///    Prelu weights tensor is passed in runtime execution phase. Prelu
    ///    weights tensor data type is implicitly assumed as f32 using plain
    ///    layout (a, ab, acb, acdb, acdeb)

    /// @param mask Defines the correspondence between the output tensor
    ///     dimensions and the prelu weights tensor. The set i-th bit indicates
    ///     that a dedicated weights value is used for each index along that
    ///     dimension. Set the mask to 0 to use a common weights value
    ///     for the whole output tensor.
    void append_prelu(int mask) {
        error::wrap_c_api(zendnn_post_ops_append_prelu(get(), mask),
                          "could not append a prelu post-op");
    }

    /// Returns the parameters of a prelu post-op.
    ///
    /// @param index Index of the prelu post-op.
    /// @param maks Weights mask of prelu post-op.
    void get_params_prelu(int index, int &mask) const {
        error::wrap_c_api(zendnn_post_ops_get_params_prelu(get(), index, &mask),
                          "could not get parameters of a binary post-op");
    }
};

/// @cond DO_NOT_DOCUMENT_THIS
template <>
struct handle_traits<zendnn_primitive_attr_t> {
    static zendnn_status_t destructor(zendnn_primitive_attr_t p) {
        return zendnn_primitive_attr_destroy(p);
    }
};
/// @endcond

/// Primitive attributes.
///
/// @sa @ref dev_guide_attributes
struct primitive_attr : public handle<zendnn_primitive_attr_t> {
    using handle<zendnn_primitive_attr_t>::handle;

    /// Constructs default (empty) primitive attributes.
    primitive_attr() {
        zendnn_primitive_attr_t result;
        error::wrap_c_api(zendnn_primitive_attr_create(&result),
                          "could not create primitive attribute");
        reset(result);
    }

    /// Creates primitive attributes from a C API ::zendnn_primitive_attr_t
    /// handle. The resulting handle is not weak and the C handle will be
    /// destroyed during the destruction of the C++ object.
    ///
    /// @param attr The C API primitive attributes.
    primitive_attr(zendnn_primitive_attr_t attr)
        : handle<zendnn_primitive_attr_t>(attr) {}

    /// Returns the fpmath mode
    fpmath_mode get_fpmath_mode() const {
        zendnn_fpmath_mode_t result;
        error::wrap_c_api(zendnn_primitive_attr_get_fpmath_mode(get(), &result),
                          "could not get fpmath mode primitive attribute");
        return fpmath_mode(result);
    }

    /// Sets autoTunerEnable Flag.
    void set_autoTunerEnable(bool autoTunerFlag) {
        error::wrap_c_api(zendnn_primitive_attr_set_autoTunerEnable(
                              get(), autoTunerFlag),
                          "could not set autoTuner Enable primitive attribute");
    }

    /// Sets woq weight sacle.
    /// @param mask Scales correspondence mask that defines the
    ///     correspondence between the tensor dimensions and the @p
    ///     scales vector. The set i-th bit indicates that a dedicated 0100 {K*N}
    ///     scale is used for each index along that dimension. Set the
    ///     mask to 0 to use a common scale for the whole output tensor.
    /// @param groups Scaling factors correspondence groups that define the
    ///     correspondence between the tensor dimensions and the scales array.
    ///     The set i-th dimension indicates a number of groups of scaling
    ///     factors used for that logical dimension in a memory indicated by @p arg.
    /// @param data_type Scaling factors data_type.
    void set_woq_scale(int mask, const memory::dims &groups,
                       memory::data_type data_type = memory::data_type::f32) {
        error::wrap_c_api(zendnn_primitive_attr_set_woq_weight_scale(
                              get(), mask, (int)groups.size(), groups.data(),
                              memory::convert_to_c(data_type)),
                          "could not set WOQ weight scale primitive attribute");
    }

    /// Sets PLUGIN Op name.
    void set_plugin_op_name(const std::string &plugin_op_name) {
        error::wrap_c_api(zendnn_primitive_attr_set_plugin_op_name(
                              get(), plugin_op_name),
                          "could not set FWK op primitive attribute");
    }

    /// Sets fpmath mode.
    ///
    /// @param mode Specified fpmath mode.
    void set_fpmath_mode(fpmath_mode mode) {
        error::wrap_c_api(zendnn_primitive_attr_set_fpmath_mode(
                              get(), zendnn::convert_to_c(mode)),
                          "could not set fpmath mode primitive attribute");
    }

    /// Returns the scratchpad mode.
    scratchpad_mode get_scratchpad_mode() const {
        zendnn_scratchpad_mode_t result;
        error::wrap_c_api(
            zendnn_primitive_attr_get_scratchpad_mode(get(), &result),
            "could not get scratchpad mode primitive attribute");
        return scratchpad_mode(result);
    }

    /// Sets scratchpad mode.
    ///
    /// @param mode Specified scratchpad mode.
    void set_scratchpad_mode(scratchpad_mode mode) {
        error::wrap_c_api(zendnn_primitive_attr_set_scratchpad_mode(
                              get(), zendnn::convert_to_c(mode)),
                          "could not set scratchpad mode primitive attribute");
    }

    /// Returns output scaling factors correspondence mask and values.
    ///
    /// @param mask Scaling factors correspondence mask that defines the
    ///     correspondence between the output tensor dimensions and the @p
    ///     scales vector. The set i-th bit indicates that a dedicated output
    ///     scaling factor is used for each index along that dimension. The
    ///     mask value of 0 implies a common output scaling factor for the
    ///     whole output tensor.
    /// @param scales Vector of output scaling factors.
    void get_output_scales(int &mask, std::vector<float> &scales) const {
        zendnn_dim_t count;
        int c_mask;
        const float *c_scales;
        error::wrap_c_api(zendnn_primitive_attr_get_output_scales(
                              get(), &count, &c_mask, &c_scales),
                          "could not get output scales primitive attribute");
        scales.resize(count);

        mask = c_mask;
        for (zendnn_dim_t c = 0; c < count; ++c) {
            scales[c] = c_scales[c];
        }
    }

    /// Sets output scaling factors correspondence mask and values.
    ///
    /// Example usage:
    /// @code
    ///     int mb = 32, oc = 32,
    ///         oh = 14, ow = 14; // convolution output params
    ///     // unique output scales per output channel
    ///     vector<float> scales = { ... };
    ///     int oc_dim = 1; // mb_dim = 0, channel_dim = 1, height_dim = 2, ...
    ///
    ///     // construct a convolution descriptor
    ///     zendnn::convolution::desc conv_d;
    ///
    ///     zendnn::primitive_attr attr;
    ///     attr.set_output_scales(attr, oc, 1 << oc_dim, scales);
    ///
    ///     zendnn::primitive_desc conv_pd(conv_d, attr, engine);
    /// @endcode
    ///
    /// @note
    ///     The order of dimensions does not depend on how elements are laid
    ///     out in memory. For example:
    ///     - for a 2D CNN activations tensor the order is always (n, c)
    ///     - for a 4D CNN activations tensor the order is always (n, c, h, w)
    ///     - for a 5D CNN weights tensor the order is always
    ///        (g, oc, ic, kh, kw)
    ///
    /// @param mask Defines the correspondence between the output tensor
    ///     dimensions and the @p scales vector. The set i-th bit indicates
    ///     that a dedicated scaling factor is used for each index along that
    ///     dimension. Set the mask to 0 to use a common output scaling factor
    ///     for the whole output tensor.
    /// @param scales Constant vector of output scaling factors. If the
    ///     scaling factors are known at the time of this call, the following
    ///     equality must hold:
    ///     \f$scales.size() = \prod\limits_{d \in mask} output.dims[d].\f$
    ///     Violations can only be detected when the attributes
    ///     are used to create a primitive descriptor.
    ///     If the scaling factors are not known at the time of the call,
    ///     this vector must contain a single #ZENDNN_RUNTIME_F32_VAL value and
    ///     the output scaling factors must be passed at execution time as an
    ///     argument with index #ZENDNN_ARG_ATTR_OUTPUT_SCALES.
    void set_output_scales(int mask, const std::vector<float> &scales) {
        error::wrap_c_api(
            zendnn_primitive_attr_set_output_scales(
                get(), (zendnn_dim_t)scales.size(), mask, scales.data()),
            "could not set output scales primitive attribute");
    }

    /// Returns scaling factors correspondence mask and values for a given
    /// memory argument.
    ///
    /// @param arg Parameter argument index as passed to the
    ///     primitive::execute() call.
    /// @param mask Scaling factors correspondence mask that defines the
    ///     correspondence between the output tensor dimensions and the @p
    ///     scales vector. The set i-th bit indicates that a dedicated scaling
    ///     factor is used for each index along that dimension. Set the mask to
    ///     0 to use a common scaling factor for the whole output tensor.
    /// @param scales Output vector of scaling factors.
    void get_scales(int arg, int &mask, std::vector<float> &scales) const {
        zendnn_dim_t count;
        int c_mask;
        const float *c_scales;
        error::wrap_c_api(zendnn_primitive_attr_get_scales(
                              get(), arg, &count, &c_mask, &c_scales),
                          "could not get scales primitive attributes");
        scales.resize(count);

        mask = c_mask;
        for (zendnn_dim_t c = 0; c < count; ++c) {
            scales[c] = c_scales[c];
        }
    }

    /// Sets scaling factors for primitive operations for a given memory
    /// argument.
    ///
    /// @sa zendnn_primitive_attr_set_scales
    /// @sa zendnn::primitive_attr::set_output_scales
    ///
    /// @param arg Parameter argument index as passed to the
    ///     primitive::execute() call.
    /// @param mask Scaling factors correspondence mask that defines the
    ///     correspondence between the tensor dimensions and the @p scales
    ///     vector. The set i-th bit indicates that a dedicated scaling factor
    ///     is used for each index along that dimension. Set the mask to 0 to
    ///     use a common scaling factor for the whole output tensor.
    /// @param scales Constant vector of scaling factors. The following equality
    ///     must hold:
    ///     \f$scales.size() = \prod\limits_{d \in mask} argument.dims[d].\f$
    void set_scales(int arg, int mask, const std::vector<float> &scales) {
        error::wrap_c_api(
            zendnn_primitive_attr_set_scales(get(), arg,
                                             (zendnn_dim_t)scales.size(), mask, scales.data()),
            "could not set scales primitive attribute");
    }

    /// Sets scaling factors for primitive operations for a given memory
    /// argument. The scaling factors must be passed at execution time
    /// as an argument with index #ZENDNN_ARG_ATTR_SCALES | arg.
    ///
    /// @sa zendnn_primitive_attr_set_scales_mask
    ///
    /// @param arg Parameter argument index as passed to the
    ///     primitive::execute() call.
    /// @param mask Scaling factors correspondence mask that defines the
    ///     correspondence between the tensor dimensions and the @p scales
    ///     vector. The set i-th bit indicates that a dedicated scaling factor
    ///     is used for each index along that dimension. Set the mask to 0 to
    ///     use a common scaling factor for the whole output tensor.
    /// @param groups Scaling factors correspondence groups that define the
    ///     correspondence between the tensor dimensions and the scales array.
    ///     The set i-th dimension indicates a number of groups of scaling
    ///     factors used for that logical dimension in a memory indicated by @p arg.
    /// @param data_type Scaling factors data_type.
    void set_scales_mask(int arg, int mask, const memory::dims &groups,
                         memory::data_type data_type = memory::data_type::f32) {
        error::wrap_c_api(zendnn_primitive_attr_set_scales_mask(get(),
                          arg, mask, (int)groups.size(), groups.data(),
                          memory::convert_to_c(data_type)),
                          "could not set scales primitive attribute");
    }

    /// Returns zero points correspondence mask and values.
    ///
    /// @param arg Parameter argument index as passed to the
    ///     primitive::execute() call.
    /// @param mask Zero points correspondence mask that defines the
    ///     correspondence between the output tensor dimensions and the @p
    ///     zero_points vector. The set i-th bit indicates that a dedicated
    ///     zero point is used for each index along that dimension. Set the
    ///     mask to 0 to use a common zero point for the whole output tensor.
    /// @param zero_points Output vector of zero points.
    void get_zero_points(
        int arg, int &mask, std::vector<int32_t> &zero_points) const {
        zendnn_dim_t count;
        int c_mask;
        const int32_t *c_zero_points;
        error::wrap_c_api(zendnn_primitive_attr_get_zero_points(
                              get(), arg, &count, &c_mask, &c_zero_points),
                          "could not get zero points primitive attribute");
        zero_points.resize(count);

        mask = c_mask;
        for (zendnn_dim_t c = 0; c < count; ++c) {
            zero_points[c] = c_zero_points[c];
        }
    }

    /// Sets zero points for primitive operations for a given memory argument.
    ///
    /// @sa zendnn_primitive_attr_set_zero_points
    /// @sa zendnn::primitive_attr::set_output_scales
    ///
    /// @param arg Parameter argument index as passed to the
    ///     primitive::execute() call.
    /// @param mask Zero point correspondence mask that defines the
    ///     correspondence between the tensor dimensions and the @p
    ///     zero_points vector. The set i-th bit indicates that a dedicated
    ///     zero point is used for each index along that dimension. Set the
    ///     mask to 0 to use a common zero point for the whole output tensor.
    /// @param zero_points Constant vector of zero points. If the zero points
    ///     are known at the time of this call, the following equality must
    ///     hold: \f$zero\_points.size() = \prod\limits_{d \in mask}
    ///     argument.dims[d].\f$ If the zero points are not known at the time
    ///     of the call, this vector must contain a single
    ///     #ZENDNN_RUNTIME_S32_VAL value and the zero points must be passed at
    ///     execution time as an argument with index
    ///     #ZENDNN_ARG_ATTR_ZERO_POINTS.
    void set_zero_points(
        int arg, int mask, const std::vector<int32_t> &zero_points) {
        error::wrap_c_api(zendnn_primitive_attr_set_zero_points(get(), arg,
                          (zendnn_dim_t)zero_points.size(), mask,
                          zero_points.data()),
                          "could not set zero points primitive attribute");
    }

    /// Returns post-ops previously set via set_post_ops().
    ///
    /// @returns Post-ops.
    const post_ops get_post_ops() const {
        post_ops result;
        const_zendnn_post_ops_t c_result;
        error::wrap_c_api(zendnn_primitive_attr_get_post_ops(get(), &c_result),
                          "could not get post-ops primitive attribute");
        result.reset(const_cast<zendnn_post_ops_t>(c_result), true);
        return result;
    }

    /// Sets post-ops.
    ///
    /// @note
    ///     There is no way to check whether the post-ops would be supported
    ///     by the target primitive. Any error will be reported
    ///     by the respective primitive descriptor constructor.
    ///
    /// @param ops Post-ops object to copy post-ops from.
    void set_post_ops(const post_ops ops) {
        error::wrap_c_api(zendnn_primitive_attr_set_post_ops(get(), ops.get()),
                          "could not set post-ops primitive attribute");
    }

    /// Sets quantization scale and shift parameters for RNN data tensors.
    ///
    /// For performance reasons, the low-precision configuration of the RNN
    /// primitives expect input activations to have the unsigned 8-bit integer
    /// data type. The scale and shift parameters are used to quantize
    /// floating-point data to unsigned integer and must be passed to the RNN
    /// primitive using attributes.
    ///
    /// The quantization formula is `scale * data + shift`.
    ///
    /// Example usage:
    /// @code
    ///     // RNN parameters
    ///     int l = 2, t = 2, mb = 32, sic = 32, slc = 32, dic = 32, dlc = 32;
    ///     // Activations quantization parameters
    ///     float scale = 63.f, shift = 64.f;
    ///
    ///     primitive_attr attr;
    ///
    ///     // Set scale and shift for int8 quantization of activation
    ///     attr.set_rnn_data_qparams(scale, shift);
    ///
    ///     // Create and configure rnn op_desc
    ///     vanilla_rnn_forward::desc rnn_d(/* arguments */);
    ///     vanilla_rnn_forward::primitive_desc rnn_d(rnn_d, attr, engine);
    /// @endcode
    ///
    /// @note
    ///     Quantization scale and shift are common for src_layer, src_iter,
    ///     dst_iter, and dst_layer.
    ///
    /// @param scale The value to scale the data by.
    /// @param shift The value to shift the data by.
    void set_rnn_data_qparams(float scale, float shift) {
        error::wrap_c_api(
            zendnn_primitive_attr_set_rnn_data_qparams(get(), scale, shift),
            "could not set RNN data quantization parameters primitive "
            "attribute");
    }

    /// Returns the quantization scale and shift parameters for RNN data
    /// tensors.
    ///
    /// @note
    ///     Quantization scale and shift are common for src_layer, src_iter,
    ///     dst_iter, and dst_layer.
    ///
    /// @param scale The value to scale the data by.
    /// @param shift The value to shift the data by.
    void get_rnn_data_qparams(float &scale, float &shift) {
        float c_scale, c_shift;
        error::wrap_c_api(zendnn_primitive_attr_get_rnn_data_qparams(
                              get(), &c_scale, &c_shift),
                          "could not set RNN data quantization parameters primitive "
                          "attribute");
        scale = c_scale;
        shift = c_shift;
    }

    /// Sets quantization scaling factors for RNN weights tensors. The
    /// low-precision configuration of the RNN primitives expect input weights
    /// to use the signed 8-bit integer data type. The scaling factors are
    /// used to quantize floating-point data to signed integer and must be
    /// passed to RNN primitives using attributes.
    ///
    /// @note
    ///     The dimension order is always native and does not depend on the
    ///     actual layout used. For example, five-dimensional weights always
    ///     have (l, d, i, g, o) logical dimension ordering.
    ///
    /// @note
    ///     Quantization scales are common for weights_layer and
    ///     weights_iteration
    ///
    /// @param mask Scaling factors correspondence mask that defines the
    ///     correspondence between the output tensor dimensions and the @p
    ///     scales vector. The set i-th bit indicates that a dedicated scaling
    ///     factor should be used each index along that dimension. Set the
    ///     mask to 0 to use a common scaling factor for the whole output
    ///     tensor.
    /// @param scales Constant vector of output scaling factors. The following
    ///     equality must hold:
    ///     \f$scales.size() = \prod\limits_{d \in mask} weights.dims[d].\f$
    ///     Violations can only be detected when the attributes are used to
    ///     create a primitive descriptor.
    void set_rnn_weights_qparams(int mask, const std::vector<float> &scales) {
        error::wrap_c_api(zendnn_primitive_attr_set_rnn_weights_qparams(get(),
                          (int)scales.size(), mask, scales.data()),
                          "could not set RNN weights quantization parameters primitive "
                          "attribute");
    }

    /// Returns the quantization scaling factors for RNN projection weights
    /// tensors.
    ///
    /// @note
    ///     The dimension order is always native and does not depend on the
    ///     actual layout used. For example, five-dimensional weights always
    ///     have (l, d, i, g, o) logical dimension ordering.
    ///
    /// @param mask Scaling factors correspondence mask that defines the
    ///     correspondence between the output tensor dimensions and the @p
    ///     scales vector. The set i-th bit indicates that a dedicated scaling
    ///     factor should be used each index along that dimension. Set the
    ///     mask to 0 to use a common scaling factor for the whole output
    ///     tensor.
    /// @param scales Constant vector of output scaling factors. The following
    ///     equality must hold:
    ///     \f$scales.size() = \prod\limits_{d \in mask} weights.dims[d].\f$
    ///     Violations can only be detected when the attributes are used to
    ///     create a primitive descriptor.
    void get_rnn_weights_qparams(int &mask, std::vector<float> &scales) {
        zendnn_dim_t count;
        int c_mask;
        const float *c_scales;
        error::wrap_c_api(zendnn_primitive_attr_get_rnn_weights_qparams(
                              get(), &count, &c_mask, &c_scales),
                          "could not get primitive RNN weights quantization "
                          "parameters attributes");
        scales.resize(count);

        mask = c_mask;
        for (zendnn_dim_t c = 0; c < count; c++) {
            scales[c] = c_scales[c];
        }
    }

    /// Sets quantization scaling factors for RNN projection weights tensors.
    //  The low-precision configuration of the RNN primitives expect input
    //  weights to use the signed 8-bit integer data type. The scaling factors
    //  are used to quantize floating-point data to signed integer and must be
    /// passed to RNN primitives using attributes.
    ///
    /// @note
    ///     The dimension order is always native and does not depend on the
    ///     actual layout used. For example, five-dimensional weights always
    ///     have (l, d, i, g, o) logical dimension ordering.
    ///
    /// @note
    ///     Quantization scales are common for weights_layer and
    ///     weights_iteration
    ///
    /// @param mask Scaling factors correspondence mask that defines the
    ///     correspondence between the output tensor dimensions and the @p
    ///     scales vector. The set i-th bit indicates that a dedicated scaling
    ///     factor should be used each index along that dimension. Set the
    ///     mask to 0 to use a common scaling factor for the whole output
    ///     tensor.
    /// @param scales Constant vector of output scaling factors. The following
    ///     equality must hold:
    ///     \f$scales.size() = \prod\limits_{d \in mask} weights.dims[d].\f$
    ///     Violations can only be detected when the attributes are used to
    ///     create a primitive descriptor.
    void set_rnn_weights_projection_qparams(
        int mask, const std::vector<float> &scales) {
        error::wrap_c_api(
            zendnn_primitive_attr_set_rnn_weights_projection_qparams(
                get(), (int)scales.size(), mask, scales.data()),
            "could not set primitive RNN weights projection quantization "
            "parameters attributes");
    }

    /// Returns the quantization scaling factors for RNN projection weights
    /// tensors.
    ///
    /// @note
    ///     The dimension order is always native and does not depend on the
    ///     actual layout used. For example, five-dimensional weights always
    ///     have (l, d, i, g, o) logical dimension ordering.
    ///
    /// @param mask Scaling factors correspondence mask that defines the
    ///     correspondence between the output tensor dimensions and the @p
    ///     scales vector. The set i-th bit indicates that a dedicated scaling
    ///     factor should be used each index along that dimension. Set the
    ///     mask to 0 to use a common scaling factor for the whole output
    ///     tensor.
    /// @param scales Constant vector of output scaling factors. The following
    ///     equality must hold:
    ///     \f$scales.size() = \prod\limits_{d \in mask} weights.dims[d].\f$
    ///     Violations can only be detected when the attributes are used to
    ///     create a primitive descriptor.
    void get_rnn_weights_projection_qparams(
        int &mask, std::vector<float> &scales) {
        zendnn_dim_t count;
        int c_mask;
        const float *c_scales;
        error::wrap_c_api(
            zendnn_primitive_attr_get_rnn_weights_projection_qparams(
                get(), &count, &c_mask, &c_scales),
            "could not get primitive RNN weights projection quantization "
            "parameters attributes");
        scales.resize(count);

        mask = c_mask;
        for (zendnn_dim_t c = 0; c < count; c++) {
            scales[c] = c_scales[c];
        }
    }
};

/// @} zendnn_api_attributes

/// @addtogroup zendnn_api_primitives_common
/// @{

/// Base class for all primitive descriptors.
struct primitive_desc_base : public handle<zendnn_primitive_desc_t> {
    using handle<zendnn_primitive_desc_t>::handle;

    /// Default constructor. Produces an empty object.
    primitive_desc_base() = default;

    /// Returns the engine of the primitive descriptor.
    /// @returns The engine of the primitive descriptor.
    engine get_engine() const {
        return engine::query(*this);
    }

    /// Returns implementation name.
    /// @returns The implementation name.
    const char *impl_info_str() const {
        const char *res;
        error::wrap_c_api(zendnn_primitive_desc_query(
                              get(), zendnn_query_impl_info_str, 0, &res),
                          "could not retrieve implementation info string from a "
                          "primitive descriptor");
        return res;
    }

    /// Returns a memory::dim value (same as int64_t).
    /// @param what The value to query.
    /// @returns The result of the query.
    memory::dim query_s64(query what) const {
        memory::dim res;
        zendnn_status_t status = zendnn_primitive_desc_query(
                                     get(), zendnn::convert_to_c(what), 0, &res);
        return status == zendnn_success ? res : 0;
    }

    /// Returns a memory descriptor.
    ///
    /// @note
    ///     There are also convenience methods
    ///     #zendnn::primitive_desc_base::src_desc(),
    ///     #zendnn::primitive_desc_base::dst_desc(), and others.
    ///
    /// @param what The kind of parameter to query; can be
    ///     #zendnn::query::src_md, #zendnn::query::dst_md, etc.
    /// @param idx Index of the parameter. For example, convolution bias can
    ///     be queried with what = #zendnn::query::weights_md and idx = 1.
    /// @returns The requested memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///     parameter of the specified kind or index.
    memory::desc query_md(query what, int idx = 0) const {
        std::vector<query> valid_q {query::src_md, query::diff_src_md,
                                    query::weights_md, query::diff_weights_md, query::dst_md,
                                    query::diff_dst_md, query::workspace_md, query::scratchpad_md,
                                    query::exec_arg_md};
        if (!std::any_of(valid_q.cbegin(), valid_q.cend(),
        [=](query q) {
        return what == q;
    }))
        ZENDNN_THROW_ERROR(zendnn_invalid_arguments,
                           "memory descriptor query is invalid");

        const zendnn_memory_desc_t *cdesc = zendnn_primitive_desc_query_md(
                                                get(), zendnn::convert_to_c(what), idx);
        return cdesc ? memory::desc(*cdesc) : memory::desc();
    }

    /// Returns a source memory descriptor.
    /// @param idx Source index.
    /// @returns Source memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///     source parameter with index @p idx.
    memory::desc src_desc(int idx) const {
        return query_md(query::src_md, idx);
    }

    /// Returns a destination memory descriptor.
    /// @param idx Destination index.
    /// @returns Destination memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///     destination parameter with index @p idx.
    memory::desc dst_desc(int idx) const {
        return query_md(query::dst_md, idx);
    }

    /// Returns a weights memory descriptor.
    /// @param idx Weights index.
    /// @returns Weights memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///     weights parameter with index @p idx.
    memory::desc weights_desc(int idx) const {
        return query_md(query::weights_md, idx);
    }

    /// Returns a diff source memory descriptor.
    /// @param idx Diff source index.
    /// @returns Diff source memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///     diff source parameter with index @p idx.
    memory::desc diff_src_desc(int idx) const {
        return query_md(query::diff_src_md, idx);
    }

    /// Returns a diff destination memory descriptor.
    /// @param idx Diff destination index.
    /// @returns Diff destination memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///     diff destination parameter with index @p idx.
    memory::desc diff_dst_desc(int idx) const {
        return query_md(query::diff_dst_md, idx);
    }

    /// Returns a diff weights memory descriptor.
    /// @param idx Diff weights index.
    /// @returns Diff weights memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///     diff weights parameter with index @p idx.
    memory::desc diff_weights_desc(int idx) const {
        return query_md(query::diff_weights_md, idx);
    }

    // Separate versions without the index argument for documentation
    // purposes.

    /// Returns a source memory descriptor.
    /// @returns Source memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///     source parameter.
    memory::desc src_desc() const {
        return src_desc(0);
    }

    /// Returns a destination memory descriptor.
    /// @returns Destination memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///     destination parameter.
    memory::desc dst_desc() const {
        return dst_desc(0);
    }

    /// Returns a weights memory descriptor.
    /// @returns Weights memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///     weights parameter.
    memory::desc weights_desc() const {
        return weights_desc(0);
    }

    /// Returns a diff source memory descriptor.
    /// @returns Diff source memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///     diff source memory with.
    memory::desc diff_src_desc() const {
        return diff_src_desc(0);
    }

    /// Returns a diff destination memory descriptor.
    /// @returns Diff destination memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///     diff destination parameter.
    memory::desc diff_dst_desc() const {
        return diff_dst_desc(0);
    }

    /// Returns a diff weights memory descriptor.
    /// @returns Diff weights memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///     diff weights parameter.
    memory::desc diff_weights_desc() const {
        return diff_weights_desc(0);
    }

    /// Returns the workspace memory descriptor.
    /// @returns Workspace memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not require
    ///     workspace parameter.
    memory::desc workspace_desc() const {
        return query_md(query::workspace_md, 0);
    }

    /// Returns the scratchpad memory descriptor.
    /// @returns scratchpad memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not require
    ///     scratchpad parameter.
    /// @sa @ref dev_guide_attributes_scratchpad
    memory::desc scratchpad_desc() const {
        return query_md(query::scratchpad_md, 0);
    }

    /// Returns the engine on which the scratchpad memory is located.
    /// @returns The engine on which the scratchpad memory is located.
    engine scratchpad_engine() const {
        zendnn_engine_t c_engine;
        error::wrap_c_api(zendnn_primitive_desc_query(get(),
                          zendnn::convert_to_c(query::scratchpad_engine),
                          0, &c_engine),
                          "could not retrieve scratchpad engine from a primitive "
                          "descriptor");
        return engine(c_engine, true);
    }

    /// Returns the primitive attributes.
    /// @returns The primitive attributes.
    primitive_attr get_primitive_attr() const {
        const_zendnn_primitive_attr_t const_c_attr;
        error::wrap_c_api(zendnn_primitive_desc_get_attr(get(), &const_c_attr),
                          "could not get attributes from a primitive descriptor");
        zendnn_primitive_attr_t c_attr;
        error::wrap_c_api(zendnn_primitive_attr_clone(&c_attr, const_c_attr),
                          "could not clone primitive attributes");
        return primitive_attr(c_attr);
    }

    /// Returns the kind of the primitive descriptor.
    /// @returns The kind of the primitive descriptor.
    zendnn::primitive::kind get_kind() const {
        zendnn_primitive_kind_t kind;
        error::wrap_c_api(zendnn_primitive_desc_query(get(),
                          zendnn_query_primitive_kind, 0, (void *)&kind),
                          "could not get primitive kind from a primitive descriptor");
        return static_cast<zendnn::primitive::kind>(kind);
    }

    /// Returns the cache blob ID of the primitive descriptor.
    /// @returns The cache blob ID of the primitive descriptor.
    std::vector<uint8_t> get_cache_blob_id() const {
        zendnn_dim_t count;
        const uint8_t *c_id;
        error::wrap_c_api(
            zendnn_primitive_desc_query(get(),
                                        zendnn::convert_to_c(query::cache_blob_id_size_s64), 0,
                                        (void *)&count),
            "could not get size of cache blob ID from a primitive "
            "descriptor");
        error::wrap_c_api(zendnn_primitive_desc_query(get(),
                          zendnn::convert_to_c(query::cache_blob_id), 0,
                          (void **)&c_id),
                          "could not get cache blob ID from a primitive descriptor");
        std::vector<uint8_t> id(c_id, c_id + count);
        return id;
    }

  protected:
    /// Resets the value of the handle to a clone of a C API primitive
    /// descriptor.
    /// @param pd A C API primitive descriptor to clone.
    void reset_with_clone(const_zendnn_primitive_desc_t pd) {
        zendnn_primitive_desc_t new_pd;
        error::wrap_c_api(zendnn_primitive_desc_clone(&new_pd, pd),
                          "could not clone a primitive descriptor");
        reset(new_pd);
    }

    /// Constructs a primitive descriptor base object from a clone of a C API
    /// primitive descriptor after verifying that it is what the caller
    /// expects.
    ///
    /// @note
    ///     The @p prim_kind should map to a primitive that does not have
    ///     different values of propagation kind (e.g. #zendnn::binary).
    /// @note
    ///     Primitive descriptor base constructed this way does not support
    ///     next_impl() (will throw).
    ///
    /// @param pd C API primitive descriptor to clone.
    /// @param prim_kind Expected primitive kind.
    primitive_desc_base(
        zendnn_primitive_desc_t pd, zendnn::primitive::kind prim_kind)
        : primitive_desc_base(pd, prim_kind, zendnn::prop_kind::undef) {}

    /// Constructs a primitive descriptor base object from a clone of a C API
    /// primitive descriptor after verifying that it is what the caller
    /// expects.
    ///
    /// @note
    ///     Primitive descriptor base constructed this way does not support
    ///     next_impl() (will throw).
    ///
    /// @param pd C API primitive descriptor to clone.
    /// @param prim_kind Expected primitive kind.
    /// @param aprop_kind Expected propagation kind.
    primitive_desc_base(zendnn_primitive_desc_t pd,
                        zendnn::primitive::kind prim_kind, zendnn::prop_kind aprop_kind)
        : primitive_desc_base(pd, prim_kind, aprop_kind, aprop_kind) {}

    /// Constructs a primitive descriptor base object from a clone of a C API
    /// primitive descriptor after verifying that it is what the caller
    /// expects.
    ///
    /// @note
    ///     Primitive descriptor base constructed this way does not support
    ///     next_impl() (will throw).
    ///
    /// @param pd C API primitive descriptor to clone.
    /// @param prim_kind Expected primitive kind.
    /// @param prop_kind1 Expected propagation kind (option 1).
    /// @param prop_kind2 Expected propagation kind (option 2). This value is
    ///     checked if the check with @p prop_kind1 fails.
    primitive_desc_base(zendnn_primitive_desc_t pd,
                        zendnn::primitive::kind prim_kind, zendnn::prop_kind prop_kind1,
                        zendnn::prop_kind prop_kind2) {
        // It is OK to pass an empty primitive descriptor
        if (pd == nullptr) {
            return;
        }

        zendnn_status_t rc;

        zendnn_primitive_kind_t c_prim_kind = convert_to_c(prim_kind);
        zendnn_prop_kind_t c_prop_kind1 = convert_to_c(prop_kind1);
        zendnn_prop_kind_t c_prop_kind2 = convert_to_c(prop_kind2);

        // Check that primitive kind matches
        zendnn_primitive_kind_t pd_kind;
        rc = zendnn_primitive_desc_query(
                 pd, zendnn_query_primitive_kind, 0, (void *)&pd_kind);
        error::wrap_c_api(
            rc, "could not get primitive kind from a primitive descriptor");
        if (pd_kind != c_prim_kind)
            ZENDNN_THROW_ERROR(zendnn_invalid_arguments,
                               "primitive descriptor operation kind mismatch");

        // Check that propagation kind matches
        zendnn_prop_kind_t pd_prop_kind;
        rc = zendnn_primitive_desc_query(
                 pd, zendnn_query_prop_kind, 0, (void *)&pd_prop_kind);

        // Something went wrong
        if (rc != zendnn_success && rc != zendnn_unimplemented)
            ZENDNN_THROW_ERROR(zendnn_invalid_arguments,
                               "could not get propagation kind from the primitive "
                               "descriptor");

        // Everything is fine
        if ((rc == zendnn_unimplemented && c_prop_kind1 == zendnn_prop_kind_undef)
                || (rc == zendnn_success
                    && (pd_prop_kind == c_prop_kind1
                        || pd_prop_kind == c_prop_kind2))) {
            reset_with_clone(pd);
            return;
        }

        // We could get the propagation kind but there is a mismatch
        ZENDNN_THROW_ERROR(zendnn_invalid_arguments,
                           "primitive descriptor propagation kind mismatch");
    }

    using base = primitive_desc_base;
};

/// @} zendnn_api_primitives_common

/// @addtogroup zendnn_api_reorder Reorder
///
/// A primitive to copy data between two memory objects. This primitive is
/// typically used to change the way the data is laid out in memory.
///
/// @sa @ref dev_guide_reorder in developer guide
///
/// @{

/// Reorder primitive.
struct reorder : public primitive {
    /// Primitive descriptor for a reorder primitive.
    struct primitive_desc : public primitive_desc_base {
        using primitive_desc_base::primitive_desc_base;

        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for reorder primitive.
        ///
        /// @note
        ///     If @p allow_empty is true, the constructor does not throw if a
        ///     primitive descriptor cannot be created.
        ///
        /// @param src_engine Engine on which the source memory object will be
        ///     located.
        /// @param src_md Source memory descriptor.
        /// @param dst_engine Engine on which the destination memory object
        ///     will be located.
        /// @param dst_md Destination memory descriptor.
        /// @param attr Primitive attributes to use (optional).
        /// @param allow_empty A flag signifying whether construction is allowed
        ///     to fail without throwing an exception. In this case an empty
        ///     object will be produced. This flag is optional and defaults to
        ///     false.
        primitive_desc(const engine &src_engine, const memory::desc &src_md,
                       const engine &dst_engine, const memory::desc &dst_md,
                       const primitive_attr &attr = primitive_attr(),
                       bool allow_empty = false) {
            zendnn_primitive_desc_t result;
            zendnn_status_t status = zendnn_reorder_primitive_desc_create(&result,
                                     &src_md.data, src_engine.get(), &dst_md.data,
                                     dst_engine.get(), attr.get());
            if (!allow_empty)
                error::wrap_c_api(status,
                                  "could not create a primitive descriptor for a reorder "
                                  "primitive");
            reset(status == zendnn_success ? result : zendnn_primitive_desc_t());
        }

        /// Constructs a primitive descriptor for reorder primitive.
        ///
        /// @param src Source memory object. It is used to obtain the source
        ///     memory descriptor and engine.
        /// @param dst Destination memory object. It is used to obtain the
        ///     destination memory descriptor and engine.
        /// @param attr Primitive attributes to use (optional).
        /// @param allow_empty A flag signifying whether construction is allowed
        ///     to fail without throwing an exception. In this case an empty
        ///     object will be produced. This flag is optional and defaults to
        ///     false.
        primitive_desc(const memory &src, const memory &dst,
                       const primitive_attr &attr = primitive_attr(),
                       bool allow_empty = false) {
            zendnn_primitive_desc_t result;
            auto src_md = src.get_desc();
            auto dst_md = dst.get_desc();
            zendnn_status_t status = zendnn_reorder_primitive_desc_create(&result,
                                     &src_md.data, src.get_engine().get(), &dst_md.data,
                                     dst.get_engine().get(), attr.get());
            if (!allow_empty)
                error::wrap_c_api(status,
                                  "could not create a primitive descriptor for a reorder "
                                  "primitive");
            reset(status == zendnn_success ? result : zendnn_primitive_desc_t());
        }

        /// Constructs a primitive descriptor for reorder primitive from a C
        /// API primitive descriptor which must have a matching kind.
        ///
        /// @param pd C API primitive descriptor for reorder primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : primitive_desc_base(pd, zendnn::primitive::kind::reorder) {}

        /// Returns the engine on which the source memory is allocated.
        /// @returns The engine on which the source memory is allocated.
        engine get_src_engine() const {
            return engine::query(*this, zendnn::query::reorder_src_engine);
        }

        /// Returns the engine on which the destination memory is allocated.
        /// @returns The engine on which the destination memory is allocated.
        engine get_dst_engine() const {
            return engine::query(*this, zendnn::query::reorder_dst_engine);
        }

        /// @copydoc zendnn::primitive_desc_base::src_desc()const
        memory::desc src_desc() const {
            return base::src_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const {
            return base::dst_desc(0);
        }
    };

    /// Default constructor. Produces an empty object.
    reorder() = default;

    /// Constructs a reorder primitive.
    /// @param pd Primitive descriptor for reorder primitive.
    reorder(const primitive_desc &pd) : primitive(pd.get()) {}

    /// Constructs a reorder primitive from a cache blob.
    /// @param pd Primitive descriptor for reorder primitive.
    /// @param cache_blob Cache blob.
    reorder(const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd.get(), cache_blob) {}

    /// Constructs a reorder primitive that would reorder data between memory
    /// objects having the same memory descriptors as memory objects @p src and
    /// @p dst.
    ///
    /// @param src Source memory object.
    /// @param dst Destination memory object.
    /// @param attr Primitive attributes to use (optional).
    reorder(const memory &src, const memory &dst,
            const primitive_attr &attr = primitive_attr())
        : primitive(primitive_desc(src, dst, attr).get()) {}

    using primitive::execute;

    /// Executes the reorder primitive.
    ///
    /// @param astream Stream object. The stream must belong to the same engine
    ///     as the primitive.
    /// @param src Source memory object.
    /// @param dst Destination memory object.
    void execute(const stream &astream, memory &src, memory &dst) const {
        primitive::execute(astream, {{ZENDNN_ARG_FROM, src}, {ZENDNN_ARG_TO, dst}});
    }
};

/// @} zendnn_api_reorder

/// @addtogroup zendnn_api_concat Concat
///
/// A primitive to concatenate data by arbitrary dimension.
///
/// @sa @ref dev_guide_concat in developer guide
///
/// @{

/// @cond DO_NOT_DOCUMENT_THIS
inline std::vector<zendnn_memory_desc_t> convert_to_c(
    const std::vector<memory::desc> &mems) {
    std::vector<zendnn_memory_desc_t> c_mems;
    c_mems.reserve(mems.size());
    for (const auto &s : mems) {
        c_mems.push_back(s.data);
    }
    return c_mems;
}
/// @endcond

/// Tensor concatenation (concat) primitive.
struct concat : public primitive {
    /// Primitive descriptor for a concat primitive.
    struct primitive_desc : public primitive_desc_base {
        using primitive_desc_base::primitive_desc_base;

        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an out-of-place concatenation
        /// primitive.
        ///
        /// @param dst Destination memory descriptor.
        /// @param concat_dimension Source tensors will be concatenated over
        ///     dimension with this index. Note that order of dimensions does
        ///     not depend on memory format.
        /// @param srcs Vector of source memory descriptors.
        /// @param aengine Engine to perform the operation on.
        /// @param attr Primitive attributes to use (optional).
        primitive_desc(const memory::desc &dst, int concat_dimension,
                       const std::vector<memory::desc> &srcs, const engine &aengine,
                       const primitive_attr &attr = primitive_attr()) {
            auto c_srcs = convert_to_c(srcs);

            zendnn_primitive_desc_t result;
            error::wrap_c_api(
                zendnn_concat_primitive_desc_create(&result, &dst.data,
                                                    (int)c_srcs.size(), concat_dimension, c_srcs.data(),
                                                    attr.get(), aengine.get()),
                "could not create a primitive descriptor for a concat "
                "primitive");
            reset(result);
        }

        /// Constructs a primitive descriptor for an out-of-place concatenation
        /// primitive.
        ///
        /// This version derives the destination memory descriptor
        /// automatically.
        ///
        /// @param concat_dimension Source tensors will be concatenated over
        ///     dimension with this index. Note that order of dimensions does
        ///     not depend on memory format.
        /// @param srcs Vector of source memory descriptors.
        /// @param aengine Engine to perform the operation on.
        /// @param attr Primitive attributes to use (optional).
        primitive_desc(int concat_dimension,
                       const std::vector<memory::desc> &srcs, const engine &aengine,
                       const primitive_attr &attr = primitive_attr()) {
            auto c_api_srcs = convert_to_c(srcs);

            zendnn_primitive_desc_t result;
            error::wrap_c_api(
                zendnn_concat_primitive_desc_create(&result, nullptr,
                                                    (int)c_api_srcs.size(), concat_dimension,
                                                    c_api_srcs.data(), attr.get(), aengine.get()),
                "could not create a primitive descriptor for a concat "
                "primitive");
            reset(result);
        }

        /// Constructs a primitive descriptor for concat primitive from a C
        /// API primitive descriptor which must have a matching kind.
        ///
        /// @param pd C API primitive descriptor for concat primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : primitive_desc_base(pd, zendnn::primitive::kind::concat) {}

        /// @copydoc zendnn::primitive_desc_base::src_desc(int)const
        memory::desc src_desc(int idx = 0) const {
            return base::src_desc(idx);
        }

        /// @copydoc zendnn::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const {
            return base::dst_desc(0);
        }
    };

    /// Default constructor. Produces an empty object.
    concat() = default;

    /// Constructs a concatenation primitive.
    /// @param pd Primitive descriptor for concatenation primitive.
    concat(const primitive_desc &pd) : primitive(pd.get()) {}

    /// Constructs a concatenation primitive from a cache blob.
    /// @param pd Primitive descriptor for concatenation primitive.
    /// @param cache_blob Cache blob.
    concat(const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd.get(), cache_blob) {}
};

/// @} zendnn_api_concat

/// @addtogroup zendnn_api_sum Sum
///
/// A primitive to sum multiple tensors.
///
/// @sa @ref dev_guide_sum in developer guide
///
/// @{

/// Out-of-place summation (sum) primitive.
struct sum : public primitive {
    /// Primitive descriptor for a sum primitive.
    struct primitive_desc : public primitive_desc_base {
        using primitive_desc_base::primitive_desc_base;

        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a sum primitive.
        ///
        /// @param dst Destination memory descriptor.
        /// @param scales Vector of scales to multiply data in each source
        ///     memory by.
        /// @param srcs Vector of source memory descriptors.
        /// @param aengine Engine to perform the operation on.
        /// @param attr Primitive attributes to use (optional).
        primitive_desc(const memory::desc &dst,
                       const std::vector<float> &scales,
                       const std::vector<memory::desc> &srcs, const engine &aengine,
                       const primitive_attr &attr = primitive_attr()) {
            validate_container_size(scales,
                                    "counts of scales and sources are not equal",
                                    (int)srcs.size(), (int)srcs.size());

            auto c_api_srcs = convert_to_c(srcs);

            zendnn_primitive_desc_t result;
            error::wrap_c_api(
                zendnn_sum_primitive_desc_create(&result, &dst.data,
                                                 (int)c_api_srcs.size(), scales.data(),
                                                 c_api_srcs.data(), attr.get(), aengine.get()),
                "could not create a primitive descriptor for a sum "
                "primitive");
            reset(result);
        }

        /// Constructs a primitive descriptor for a sum primitive.
        ///
        /// This version derives the destination memory descriptor
        /// automatically.
        ///
        /// @param scales Vector of scales by which to multiply data in each
        ///     source memory object.
        /// @param srcs Vector of source memory descriptors.
        /// @param aengine Engine on which to perform the operation.
        /// @param attr Primitive attributes to use (optional).
        primitive_desc(const std::vector<float> &scales,
                       const std::vector<memory::desc> &srcs, const engine &aengine,
                       const primitive_attr &attr = primitive_attr()) {
            validate_container_size(scales,
                                    "counts of scales and sources are not equal",
                                    (int)srcs.size(), (int)srcs.size());

            auto c_api_srcs = convert_to_c(srcs);
            zendnn_primitive_desc_t result;
            error::wrap_c_api(
                zendnn_sum_primitive_desc_create(&result, nullptr,
                                                 (int)c_api_srcs.size(), scales.data(),
                                                 c_api_srcs.data(), attr.get(), aengine.get()),
                "could not create a primitive descriptor for a sum "
                "primitive");
            reset(result);
        }

        /// Constructs a primitive descriptor for sum primitive from a C API
        /// primitive descriptor which must have a matching kind.
        ///
        /// @param pd C API primitive descriptor for reorder primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : primitive_desc_base(pd, zendnn::primitive::kind::sum) {}

        /// @copydoc zendnn::primitive_desc_base::src_desc(int)const
        memory::desc src_desc(int idx = 0) const {
            return base::src_desc(idx);
        }

        /// @copydoc zendnn::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const {
            return base::dst_desc(0);
        }
    };

    /// Default constructor. Produces an empty object.
    sum() = default;

    /// Constructs a sum primitive.
    /// @param pd Primitive descriptor for sum primitive.
    sum(const primitive_desc &pd) : primitive(pd.get()) {}

    /// Constructs a sum primitive from a cache blob.
    /// @param pd Primitive descriptor for sum primitive.
    /// @param cache_blob Cache blob.
    sum(const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd.get(), cache_blob) {}
};

/// @} zendnn_api_sum

/// @addtogroup zendnn_api_primitives_common
/// @{

/// A base class for descriptors of all primitives that have an operation
/// descriptor and that support iteration over multiple implementations.
struct primitive_desc : public primitive_desc_base {
    using primitive_desc_base::primitive_desc_base;

    primitive_desc() = default;

    /// Constructs a primitive descriptor.
    ///
    /// @note
    ///     If @p allow_empty is true, the constructor does not throw if a
    ///     primitive descriptor cannot be created. But calling next_impl() in
    ///     this case will throw.
    ///
    /// @note
    ///     This is a low-level implementation detail that is typically not
    ///     needed in application code.
    ///
    /// @param desc Constant C API operation descriptor.
    /// @param attr Pointer to primitive attributes. It is safe to pass
    ///     nullptr to indicate absence of attributes.
    /// @param aengine Engine to use.
    /// @param hint_fwd_pd C API primitive descriptor for a forward
    ///     propagation primitive. It is used as a hint for deciding which
    ///     memory format to use for backward propagation or weights gradient.
    /// @param allow_empty A flag signifying whether construction is allowed
    ///     to fail without throwing an exception. In this case an empty
    ///     object will be produced. This flag is optional and defaults to
    ///     false.
    primitive_desc(const_zendnn_op_desc_t desc, const primitive_attr *attr,
                   const engine &aengine, const_zendnn_primitive_desc_t hint_fwd_pd,
                   bool allow_empty = false)
        : allow_empty_(allow_empty) {
        zendnn_primitive_desc_iterator_t iterator = nullptr;
        zendnn_status_t status = zendnn_primitive_desc_iterator_create(&iterator,
                                 desc, attr ? attr->get() : nullptr, aengine.get(), hint_fwd_pd);
        if (!allow_empty)
            error::wrap_c_api(
                status, "could not create a primitive descriptor iterator");
        pd_iterator.reset(iterator);
        fetch_impl();
    }

    /// Advances the primitive iterator to the next implementation.
    ///
    /// @returns @c true on success, and @c false if the last implementation
    ///     reached, and the primitive descriptor itself is kept unchanged
    bool next_impl() {
        zendnn_status_t status
            = zendnn_primitive_desc_iterator_next(pd_iterator.get());
        if (status == zendnn_iterator_ends) {
            return false;
        }
        error::wrap_c_api(
            status, "could not advance a primitive descriptor iterator");
        fetch_impl();
        return true;
    }

  private:
    bool allow_empty_ = false;
    handle<zendnn_primitive_desc_iterator_t> pd_iterator;
    void fetch_impl() {
        zendnn_primitive_desc_t pd = zendnn_primitive_desc_iterator_fetch(
                                         pd_iterator.get(allow_empty_));
        error::wrap_c_api(pd != nullptr || allow_empty_ ? zendnn_success
                          : zendnn_out_of_memory,
                          "could not fetch a primitive descriptor from a primitive "
                          "descriptor iterator");
        reset(pd);
    }
};

/// @} zendnn_api_primitives_common

/// @addtogroup zendnn_api_convolution Convolution
///
/// A primitive to perform 1D, 2D or 3D convolution. Supported variants are
/// forward propagation, backward propagation, and weights gradient with or
/// without bias.
///
/// @sa @ref dev_guide_convolution in developer guide
///
/// @{

/// Convolution forward propagation primitive.
struct convolution_forward : public primitive {
    /// Descriptor for a convolution forward propagation primitive.
    struct desc {
        zendnn_convolution_desc_t data;

        /// Constructs a descriptor for a convolution forward propagation
        /// primitive with bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #zendnn::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p padding_l, and @p padding_r contain values
        /// for spatial dimensions only and hence must have the same number of
        /// elements as there are spatial dimensions. The order of values is
        /// the same as in the tensor: depth (for 3D tensors), height (for 3D
        /// and 2D tensors), and width.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #zendnn::prop_kind::forward_training, and
        ///     #zendnn::prop_kind::forward_inference.
        /// @param aalgorithm Convolution algorithm. Possible values are
        ///     #zendnn::algorithm::convolution_direct,
        ///     #zendnn::algorithm::convolution_winograd, and
        ///     #zendnn::algorithm::convolution_auto.
        /// @param src_desc Source memory descriptor.
        /// @param weights_desc Weights memory descriptor.
        /// @param bias_desc Bias memory descriptor. Passing zero memory
        ///     descriptor disables the bias term.
        /// @param dst_desc Destination memory descriptor.
        /// @param strides Strides for each spatial dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        desc(prop_kind aprop_kind, algorithm aalgorithm,
             const memory::desc &src_desc, const memory::desc &weights_desc,
             const memory::desc &bias_desc, const memory::desc &dst_desc,
             const memory::dims &strides, const memory::dims &padding_l,
             const memory::dims &padding_r) {
            memory::validate_dims(strides, src_desc.data.ndims - 2);
            memory::validate_dims(padding_l, src_desc.data.ndims - 2);
            memory::validate_dims(padding_r, src_desc.data.ndims - 2);
            zendnnInfo(ZENDNN_APILOG, "Covolution forward desc create - bias");
            error::wrap_c_api(
                zendnn_convolution_forward_desc_init(&data,
                        zendnn::convert_to_c(aprop_kind),
                        convert_to_c(aalgorithm), &src_desc.data,
                        &weights_desc.data, &bias_desc.data, &dst_desc.data,
                        &strides[0], &padding_l[0], &padding_r[0]),
                "could not create a descriptor for a convolution forward "
                "propagation primitive");
        }

        /// Constructs a descriptor for a convolution forward propagation
        /// primitive without bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #zendnn::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p padding_l, and @p padding_r contain values
        /// for spatial dimensions only and hence must have the same number of
        /// elements as there are spatial dimensions. The order of values is
        /// the same as in the tensor: depth (for 3D tensors), height (for 3D
        /// and 2D tensors), and width.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #zendnn::prop_kind::forward_training, and
        ///     #zendnn::prop_kind::forward_inference.
        /// @param aalgorithm Convolution algorithm. Possible values are
        ///     #zendnn::algorithm::convolution_direct,
        ///     #zendnn::algorithm::convolution_winograd, and
        ///     #zendnn::algorithm::convolution_auto.
        /// @param src_desc Source memory descriptor.
        /// @param weights_desc Weights memory descriptor.
        /// @param dst_desc Destination memory descriptor.
        /// @param strides Strides for each spatial dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        desc(prop_kind aprop_kind, algorithm aalgorithm,
             const memory::desc &src_desc, const memory::desc &weights_desc,
             const memory::desc &dst_desc, const memory::dims &strides,
             const memory::dims &padding_l, const memory::dims &padding_r) {
            memory::validate_dims(strides, src_desc.data.ndims - 2);
            memory::validate_dims(padding_l, src_desc.data.ndims - 2);
            memory::validate_dims(padding_r, src_desc.data.ndims - 2);
            zendnnInfo(ZENDNN_APILOG, "Covolution forward desc create - no bias");
            error::wrap_c_api(
                zendnn_convolution_forward_desc_init(&data,
                        zendnn::convert_to_c(aprop_kind),
                        convert_to_c(aalgorithm), &src_desc.data,
                        &weights_desc.data, nullptr, &dst_desc.data,
                        &strides[0], &padding_l[0], &padding_r[0]),
                "could not create a descriptor for a convolution forward "
                "propagation primitive");
        }

        /// Constructs a descriptor for a dilated convolution forward
        /// propagation primitive with bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #zendnn::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p dilates, @p padding_l, and @p padding_r
        /// contain values for spatial dimensions only and hence must have the
        /// same number of elements as there are spatial dimensions. The order
        /// of values is the same as in the tensor: depth (for 3D tensors),
        /// height (for 3D and 2D tensors), and width.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #zendnn::prop_kind::forward_training, and
        ///     #zendnn::prop_kind::forward_inference.
        /// @param aalgorithm Convolution algorithm. Possible values are
        ///     #zendnn::algorithm::convolution_direct,
        ///     #zendnn::algorithm::convolution_winograd, and
        ///     #zendnn::algorithm::convolution_auto.
        /// @param src_desc Source memory descriptor.
        /// @param weights_desc Weights memory descriptor.
        /// @param bias_desc Bias memory descriptor. Passing zero memory
        ///     descriptor disables the bias term.
        /// @param dst_desc Destination memory descriptor.
        /// @param strides Strides for each spatial dimension.
        /// @param dilates Dilations for each spatial dimension. A zero value
        ///     means no dilation in the corresponding dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        desc(prop_kind aprop_kind, algorithm aalgorithm,
             const memory::desc &src_desc, const memory::desc &weights_desc,
             const memory::desc &bias_desc, const memory::desc &dst_desc,
             const memory::dims &strides, const memory::dims &dilates,
             const memory::dims &padding_l, const memory::dims &padding_r) {
            memory::validate_dims(strides, src_desc.data.ndims - 2);
            memory::validate_dims(dilates, src_desc.data.ndims - 2);
            memory::validate_dims(padding_l, src_desc.data.ndims - 2);
            memory::validate_dims(padding_r, src_desc.data.ndims - 2);
            error::wrap_c_api(zendnn_dilated_convolution_forward_desc_init(&data,
                              zendnn::convert_to_c(aprop_kind),
                              convert_to_c(aalgorithm), &src_desc.data,
                              &weights_desc.data, &bias_desc.data,
                              &dst_desc.data, &strides[0], &dilates[0],
                              &padding_l[0], &padding_r[0]),
                              "could not create a descriptor for a dilated convolution "
                              "forward propagation primitive");
        }

        /// Constructs a descriptor for a dilated convolution forward
        /// propagation primitive without bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #zendnn::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p dilates, @p padding_l, and @p padding_r
        /// contain values for spatial dimensions only and hence must have the
        /// same number of elements as there are spatial dimensions. The order
        /// of values is the same as in the tensor: depth (for 3D tensors),
        /// height (for 3D and 2D tensors), and width.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #zendnn::prop_kind::forward_training, and
        ///     #zendnn::prop_kind::forward_inference.
        /// @param aalgorithm Convolution algorithm. Possible values are
        ///     #zendnn::algorithm::convolution_direct,
        ///     #zendnn::algorithm::convolution_winograd, and
        ///     #zendnn::algorithm::convolution_auto.
        /// @param src_desc Source memory descriptor.
        /// @param weights_desc Weights memory descriptor.
        /// @param dst_desc Destination memory descriptor.
        /// @param strides Strides for each spatial dimension.
        /// @param dilates Dilations for each spatial dimension. A zero value
        ///     means no dilation in the corresponding dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        desc(prop_kind aprop_kind, algorithm aalgorithm,
             const memory::desc &src_desc, const memory::desc &weights_desc,
             const memory::desc &dst_desc, const memory::dims &strides,
             const memory::dims &dilates, const memory::dims &padding_l,
             const memory::dims &padding_r) {
            memory::validate_dims(strides, src_desc.data.ndims - 2);
            memory::validate_dims(dilates, src_desc.data.ndims - 2);
            memory::validate_dims(padding_l, src_desc.data.ndims - 2);
            memory::validate_dims(padding_r, src_desc.data.ndims - 2);
            error::wrap_c_api(zendnn_dilated_convolution_forward_desc_init(&data,
                              zendnn::convert_to_c(aprop_kind),
                              convert_to_c(aalgorithm), &src_desc.data,
                              &weights_desc.data, nullptr,
                              &dst_desc.data, &strides[0], &dilates[0],
                              &padding_l[0], &padding_r[0]),
                              "could not create a descriptor for a dilated convolution "
                              "forward propagation primitive");
        }

        /// Initializes a descriptor for convolution forward propagation with
        /// bias using @p aprop_kind (possible values are
        /// #zendnn::forward_training and #zendnn::forward_inference),
        /// @p aalgorithm, memory descriptors, @p strides, @p padding_l, and
        /// @p padding_r. relu fusion
        ///
        /// @note Memory descriptors are allowed to be initialized with
        ///       #zendnn::memory::format_tag::any value of @p format_kind.
        desc(prop_kind aprop_kind, algorithm aalgorithm,
             const memory::desc &src_desc, const memory::desc &weights_desc,
             const memory::desc &bias_desc, const memory::desc &dst_desc,
             const memory::dims &strides, const memory::dims &padding_l,
             const memory::dims &padding_r, bool reluFused) {
            memory::validate_dims(strides, src_desc.data.ndims - 2);
            memory::validate_dims(padding_l, src_desc.data.ndims - 2);
            memory::validate_dims(padding_r, src_desc.data.ndims - 2);
            zendnnInfo(ZENDNN_APILOG, "Covolution forward desc create - relu");
            error::wrap_c_api(
                zendnn_fused_convolution_forward_desc_init(&data,
                        zendnn::convert_to_c(aprop_kind),
                        convert_to_c(aalgorithm), &src_desc.data,
                        &weights_desc.data, &bias_desc.data, &dst_desc.data,
                        &strides[0], &padding_l[0], &padding_r[0], reluFused,
                        false, nullptr, nullptr, nullptr),
                "could not create a descriptor for a relu fused convolution "
                "forward propagation primitive");
        }

        /// Initializes a descriptor for convolution forward propagation with
        /// bias using @p aprop_kind (possible values are
        /// #zendnn::forward_training and #zendnn::forward_inference),
        /// @p aalgorithm, memory descriptors, @p strides, @p padding_l, and
        /// @p padding_r. relu fusion, BatchNorm fusion
        ///
        /// @note Memory descriptors are allowed to be initialized with
        ///       #zendnn::memory::format_tag::any value of @p format_kind.
        desc(prop_kind aprop_kind, algorithm aalgorithm,
             const memory::desc &src_desc, const memory::desc &weights_desc,
             const memory::desc &bias_desc, const memory::desc &dst_desc,
             const memory::dims &strides, const memory::dims &padding_l,
             const memory::dims &padding_r, bool reluFused,
             bool batchNormFused, const memory::desc &batchNormScale_desc,
             const memory::desc &batchNormMean_desc,
             const memory::desc &batchNormOffset_desc) {
            memory::validate_dims(strides, src_desc.data.ndims - 2);
            memory::validate_dims(padding_l, src_desc.data.ndims - 2);
            memory::validate_dims(padding_r, src_desc.data.ndims - 2);
            zendnnInfo(ZENDNN_APILOG, "Covolution forward desc create -  relu, batchNorm");
            error::wrap_c_api(
                zendnn_fused_convolution_forward_desc_init(&data,
                        zendnn::convert_to_c(aprop_kind),
                        convert_to_c(aalgorithm), &src_desc.data,
                        &weights_desc.data, &bias_desc.data, &dst_desc.data,
                        &strides[0], &padding_l[0], &padding_r[0], reluFused,
                        batchNormFused, &batchNormScale_desc.data,
                        &batchNormMean_desc.data, &batchNormOffset_desc.data),
                "could not create a descriptor for a relu or batchnorm fused "
                "convolution forward propagation primitive");
        }
    };

    /// Primitive descriptor for a convolution forward propagation primitive.
    struct primitive_desc : public zendnn::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a convolution forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a convolution forward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case
        ///     an empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                       bool allow_empty = false)
            : zendnn::primitive_desc(
                  &adesc.data, nullptr, aengine, nullptr, allow_empty) {
            zendnnInfo(ZENDNN_APILOG, "Convolution primitive descriptor create - no attr");
        }

        /// Constructs a primitive descriptor for a convolution forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a convolution forward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param attr Primitive attributes to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case
        ///     an empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                       const engine &aengine, bool allow_empty = false)
            : zendnn::primitive_desc(
                  &adesc.data, &attr, aengine, nullptr, allow_empty) {
            zendnnInfo(ZENDNN_APILOG, "Convolution primitive descriptor create - attr");
        }

        /// Constructs a primitive descriptor for a convolution forward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a convolution forward
        ///     propagation primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : zendnn::primitive_desc(pd, zendnn::primitive::kind::convolution,
                                     zendnn::prop_kind::forward_training,
                                     zendnn::prop_kind::forward_inference) {
            zendnnInfo(ZENDNN_APILOG, "Convolution primitive descriptor create - C API");
        }

        /// @copydoc zendnn::primitive_desc_base::src_desc()const
        memory::desc src_desc() const {
            return base::src_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::weights_desc()const
        memory::desc weights_desc() const {
            return base::weights_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const {
            return base::dst_desc(0);
        }

        /// Returns the bias memory descriptor.
        /// @returns The bias memory descriptor.
        /// @returns A zero memory descriptor of the primitive does not have a
        ///     bias parameter.
        memory::desc bias_desc() const {
            return base::weights_desc(1);
        }
    };

    /// Default constructor. Produces an empty object.
    convolution_forward() = default;

    /// Constructs a convolution forward propagation primitive.
    /// @param pd Primitive descriptor for a convolution forward propagation
    ///     primitive.
    convolution_forward(const primitive_desc &pd) : primitive(pd) {
        zendnnInfo(ZENDNN_APILOG, "Convolution primitive create");
    }
    /// Constructs a convolution forward propagation primitive from a cache
    ///     blob.
    /// @param pd Primitive descriptor for a convolution forward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    convolution_forward(
        const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// Convolution backward propagation primitive.
struct convolution_backward_data : public primitive {

    /// Descriptor for a convolution backward propagation primitive.
    struct desc {
        zendnn_convolution_desc_t data;

        /// Constructs a descriptor for a convolution backward propagation
        /// primitive.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #zendnn::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p padding_l, and @p padding_r contain values
        /// for spatial dimensions only and hence must have the same number of
        /// elements as there are spatial dimensions. The order of values is
        /// the same as in the tensor: depth (for 3D tensors), height (for 3D
        /// and 2D tensors), and width.
        ///
        /// @param aalgorithm Convolution algorithm. Possible values are
        ///     #zendnn::algorithm::convolution_direct,
        ///     #zendnn::algorithm::convolution_winograd, and
        ///     #zendnn::algorithm::convolution_auto.
        /// @param diff_src_desc Diff source memory descriptor.
        /// @param weights_desc Weights memory descriptor.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param strides Strides for each spatial dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        desc(algorithm aalgorithm, const memory::desc &diff_src_desc,
             const memory::desc &weights_desc,
             const memory::desc &diff_dst_desc, const memory::dims &strides,
             const memory::dims &padding_l, const memory::dims &padding_r) {
            memory::validate_dims(strides, diff_src_desc.data.ndims - 2);
            memory::validate_dims(padding_l, diff_src_desc.data.ndims - 2);
            memory::validate_dims(padding_r, diff_src_desc.data.ndims - 2);
            error::wrap_c_api(
                zendnn_convolution_backward_data_desc_init(&data,
                        convert_to_c(aalgorithm), &diff_src_desc.data,
                        &weights_desc.data, &diff_dst_desc.data,
                        &strides[0], &padding_l[0], &padding_r[0]),
                "could not create a descriptor for a convolution backward "
                "propagation primitive");
        }

        /// Constructs a descriptor for dilated convolution backward
        /// propagation primitive.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #zendnn::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p dilates, @p padding_l, and @p padding_r
        /// contain values for spatial dimensions only and hence must have the
        /// same number of elements as there are spatial dimensions. The order
        /// of values is the same as in the tensor: depth (for 3D tensors),
        /// height (for 3D and 2D tensors), and width.
        ///
        /// @param aalgorithm Convolution algorithm. Possible values are
        ///     #zendnn::algorithm::convolution_direct,
        ///     #zendnn::algorithm::convolution_winograd, and
        ///     #zendnn::algorithm::convolution_auto.
        /// @param diff_src_desc Diff source memory descriptor.
        /// @param weights_desc Weights memory descriptor.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param strides Strides for each spatial dimension.
        /// @param dilates Dilations for each spatial dimension. A zero value
        ///     means no dilation in the corresponding dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        desc(algorithm aalgorithm, const memory::desc &diff_src_desc,
             const memory::desc &weights_desc,
             const memory::desc &diff_dst_desc, const memory::dims &strides,
             const memory::dims &dilates, const memory::dims &padding_l,
             const memory::dims &padding_r) {
            memory::validate_dims(strides, diff_src_desc.data.ndims - 2);
            memory::validate_dims(dilates, diff_src_desc.data.ndims - 2);
            memory::validate_dims(padding_l, diff_src_desc.data.ndims - 2);
            memory::validate_dims(padding_r, diff_src_desc.data.ndims - 2);
            error::wrap_c_api(
                zendnn_dilated_convolution_backward_data_desc_init(&data,
                        convert_to_c(aalgorithm), &diff_src_desc.data,
                        &weights_desc.data, &diff_dst_desc.data,
                        &strides[0], &dilates[0], &padding_l[0],
                        &padding_r[0]),
                "could not create a descriptor for a dilated convolution "
                "backward propagation primitive");
        }
    };

    /// Primitive descriptor for a convolution backward propagation primitive.
    struct primitive_desc : public zendnn::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a convolution backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a convolution backward propagation
        ///     primitive.
        /// @param aengine Engine to perform the operation on.
        /// @param hint_fwd_pd Primitive descriptor for a convolution forward
        ///     propagation primitive. It is used as a hint for deciding which
        ///     memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case
        ///     an empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                       const convolution_forward::primitive_desc &hint_fwd_pd,
                       bool allow_empty = false)
            : zendnn::primitive_desc(&adesc.data, nullptr, aengine,
                                     hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a convolution backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a convolution backward propagation
        ///     primitive.
        /// @param aengine Engine to perform the operation on.
        /// @param attr Primitive attributes to use.
        /// @param hint_fwd_pd Primitive descriptor for a convolution forward
        ///     propagation primitive. It is used as a hint for deciding which
        ///     memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case
        ///     an empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                       const engine &aengine,
                       const convolution_forward::primitive_desc &hint_fwd_pd,
                       bool allow_empty = false)
            : zendnn::primitive_desc(&adesc.data, &attr, aengine,
                                     hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a convolution backward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a convolution backward
        ///     propagation primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : zendnn::primitive_desc(pd, zendnn::primitive::kind::convolution,
                                     zendnn::prop_kind::backward_data) {}

        /// @copydoc zendnn::primitive_desc_base::diff_src_desc()const
        memory::desc diff_src_desc() const {
            return base::diff_src_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::weights_desc()const
        memory::desc weights_desc() const {
            return base::weights_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::diff_dst_desc()const
        memory::desc diff_dst_desc() const {
            return base::diff_dst_desc(0);
        }
    };

    /// Default constructor. Produces an empty object.
    convolution_backward_data() = default;

    /// Constructs a convolution backward propagation primitive.
    /// @param pd Primitive descriptor for a convolution backward propagation
    ///     primitive.
    convolution_backward_data(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a convolution backward propagation primitive from a cache
    ///     blob.
    /// @param pd Primitive descriptor for a convolution backward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    convolution_backward_data(
        const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// Convolution weights gradient primitive.
struct convolution_backward_weights : public primitive {
    /// Descriptor for a convolution weights gradient primitive.
    struct desc {
        zendnn_convolution_desc_t data;

        /// Constructs a descriptor for a convolution weights gradient primitive
        /// with bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #zendnn::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p padding_l, and @p padding_r contain values
        /// for spatial dimensions only and hence must have the same number of
        /// elements as there are spatial dimensions. The order of values is
        /// the same as in the tensor: depth (for 3D tensors), height (for 3D
        /// and 2D tensors), and width.
        ///
        /// @param aalgorithm Convolution algorithm. Possible values are
        ///     #zendnn::algorithm::convolution_direct,
        ///     #zendnn::algorithm::convolution_winograd, and
        ///     #zendnn::algorithm::convolution_auto.
        /// @param src_desc Source memory descriptor.
        /// @param diff_weights_desc Diff weights memory descriptor.
        /// @param diff_bias_desc Diff bias memory descriptor. Passing zero
        ///     memory descriptor disables the bias term.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param strides Strides for each spatial dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        desc(algorithm aalgorithm, const memory::desc &src_desc,
             const memory::desc &diff_weights_desc,
             const memory::desc &diff_bias_desc,
             const memory::desc &diff_dst_desc, const memory::dims &strides,
             const memory::dims &padding_l, const memory::dims &padding_r) {
            memory::validate_dims(strides, src_desc.data.ndims - 2);
            memory::validate_dims(padding_l, src_desc.data.ndims - 2);
            memory::validate_dims(padding_r, src_desc.data.ndims - 2);
            error::wrap_c_api(
                zendnn_convolution_backward_weights_desc_init(&data,
                        convert_to_c(aalgorithm), &src_desc.data,
                        &diff_weights_desc.data, &diff_bias_desc.data,
                        &diff_dst_desc.data, &strides[0], &padding_l[0],
                        &padding_r[0]),
                "could not create a descriptor for a convolution weights "
                "update primitive");
        }

        /// Constructs a descriptor for a convolution weights gradient primitive
        /// without bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #zendnn::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p padding_l, and @p padding_r contain values
        /// for spatial dimensions only and hence must have the same number of
        /// elements as there are spatial dimensions. The order of values is
        /// the same as in the tensor: depth (for 3D tensors), height (for 3D
        /// and 2D tensors), and width.
        ///
        /// @param aalgorithm Convolution algorithm. Possible values are
        ///     #zendnn::algorithm::convolution_direct,
        ///     #zendnn::algorithm::convolution_winograd, and
        ///     #zendnn::algorithm::convolution_auto.
        /// @param src_desc Source memory descriptor.
        /// @param diff_weights_desc Diff weights memory descriptor.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param strides Strides for each spatial dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        desc(algorithm aalgorithm, const memory::desc &src_desc,
             const memory::desc &diff_weights_desc,
             const memory::desc &diff_dst_desc, const memory::dims &strides,
             const memory::dims &padding_l, const memory::dims &padding_r) {
            memory::validate_dims(strides, src_desc.data.ndims - 2);
            memory::validate_dims(padding_l, src_desc.data.ndims - 2);
            memory::validate_dims(padding_r, src_desc.data.ndims - 2);
            error::wrap_c_api(zendnn_convolution_backward_weights_desc_init(&data,
                              convert_to_c(aalgorithm), &src_desc.data,
                              &diff_weights_desc.data, nullptr,
                              &diff_dst_desc.data, &strides[0],
                              &padding_l[0], &padding_r[0]),
                              "could not create a descriptor for a convolution weights "
                              "update primitive");
        }

        /// Constructs a descriptor for a dilated convolution weights gradient
        /// primitive with bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #zendnn::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p dilates, @p padding_l, and @p padding_r
        /// contain values for spatial dimensions only and hence must have the
        /// same number of elements as there are spatial dimensions. The order
        /// of values is the same as in the tensor: depth (for 3D tensors),
        /// height (for 3D and 2D tensors), and width.
        ///
        /// @param aalgorithm Convolution algorithm. Possible values are
        ///     #zendnn::algorithm::convolution_direct,
        ///     #zendnn::algorithm::convolution_winograd, and
        ///     #zendnn::algorithm::convolution_auto.
        /// @param src_desc Source memory descriptor.
        /// @param diff_weights_desc Diff weights memory descriptor.
        /// @param diff_bias_desc Diff bias memory descriptor. Passing zero
        ///     memory descriptor disables the bias term.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param strides Strides for each spatial dimension.
        /// @param dilates Dilations for each spatial dimension. A zero value
        ///     means no dilation in the corresponding dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        desc(algorithm aalgorithm, const memory::desc &src_desc,
             const memory::desc &diff_weights_desc,
             const memory::desc &diff_bias_desc,
             const memory::desc &diff_dst_desc, const memory::dims &strides,
             const memory::dims &dilates, const memory::dims &padding_l,
             const memory::dims &padding_r) {
            memory::validate_dims(strides, src_desc.data.ndims - 2);
            memory::validate_dims(dilates, src_desc.data.ndims - 2);
            memory::validate_dims(padding_l, src_desc.data.ndims - 2);
            memory::validate_dims(padding_r, src_desc.data.ndims - 2);
            error::wrap_c_api(
                zendnn_dilated_convolution_backward_weights_desc_init(&data,
                        convert_to_c(aalgorithm), &src_desc.data,
                        &diff_weights_desc.data, &diff_bias_desc.data,
                        &diff_dst_desc.data, &strides[0], &dilates[0],
                        &padding_l[0], &padding_r[0]),
                "could not create a descriptor for a dilated convolution "
                "weights gradient primitive");
        }

        /// Constructs a descriptor for a dilated convolution weights gradient
        /// primitive without bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #zendnn::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p dilates, @p padding_l, and @p padding_r
        /// contain values for spatial dimensions only and hence must have the
        /// same number of elements as there are spatial dimensions. The order
        /// of values is the same as in the tensor: depth (for 3D tensors),
        /// height (for 3D and 2D tensors), and width.
        ///
        /// @param aalgorithm Convolution algorithm. Possible values are
        ///     #zendnn::algorithm::convolution_direct,
        ///     #zendnn::algorithm::convolution_winograd, and
        ///     #zendnn::algorithm::convolution_auto.
        /// @param src_desc Source memory descriptor.
        /// @param diff_weights_desc Diff weights memory descriptor.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param strides Strides for each spatial dimension.
        /// @param dilates Dilations for each spatial dimension. A zero value
        ///     means no dilation in the corresponding dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        desc(algorithm aalgorithm, const memory::desc &src_desc,
             const memory::desc &diff_weights_desc,
             const memory::desc &diff_dst_desc, const memory::dims &strides,
             const memory::dims &dilates, const memory::dims &padding_l,
             const memory::dims &padding_r) {
            memory::validate_dims(strides, src_desc.data.ndims - 2);
            memory::validate_dims(dilates, src_desc.data.ndims - 2);
            memory::validate_dims(padding_l, src_desc.data.ndims - 2);
            memory::validate_dims(padding_r, src_desc.data.ndims - 2);
            error::wrap_c_api(
                zendnn_dilated_convolution_backward_weights_desc_init(&data,
                        convert_to_c(aalgorithm), &src_desc.data,
                        &diff_weights_desc.data, nullptr,
                        &diff_dst_desc.data, &strides[0], &dilates[0],
                        &padding_l[0], &padding_r[0]),
                "could not create a descriptor for a dilated convolution "
                "weights gradient primitive");
        }
    };

    /// Primitive descriptor for a convolution weights gradient primitive.
    struct primitive_desc : public zendnn::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a convolution weights gradient
        /// primitive.
        ///
        /// @param adesc Descriptor for a convolution weights gradient primitive.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a convolution forward
        ///     propagation primitive. It is used as a hint for deciding which
        ///     memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case
        ///     an empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                       const convolution_forward::primitive_desc &hint_fwd_pd,
                       bool allow_empty = false)
            : zendnn::primitive_desc(&adesc.data, nullptr, aengine,
                                     hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a convolution weights gradient
        /// primitive.
        ///
        /// @param adesc Descriptor for a convolution weights gradient primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a convolution forward
        ///     propagation primitive. It is used as a hint for deciding which
        ///     memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case
        ///     an empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                       const engine &aengine,
                       const convolution_forward::primitive_desc &hint_fwd_pd,
                       bool allow_empty = false)
            : zendnn::primitive_desc(&adesc.data, &attr, aengine,
                                     hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a convolution weights gradient
        /// primitive from a C API primitive descriptor that must have a
        /// matching kind.
        ///
        /// @param pd C API primitive descriptor for a convolution weights
        ///     gradient primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : zendnn::primitive_desc(pd, zendnn::primitive::kind::convolution,
                                     zendnn::prop_kind::backward_weights) {}

        /// @copydoc zendnn::primitive_desc_base::src_desc()const
        memory::desc src_desc() const {
            return base::src_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::diff_weights_desc()const
        memory::desc diff_weights_desc() const {
            return base::diff_weights_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::diff_dst_desc()const
        memory::desc diff_dst_desc() const {
            return base::diff_dst_desc(0);
        }

        /// Returns the diff bias memory descriptor.
        /// @returns The diff bias memory descriptor.
        /// @returns A zero memory descriptor of the primitive does not have a
        ///          diff bias parameter.
        memory::desc diff_bias_desc() const {
            return base::diff_weights_desc(1);
        }
    };

    /// Default constructor. Produces an empty object.
    convolution_backward_weights() = default;

    /// Constructs a convolution weights gradient primitive.
    /// @param pd Primitive descriptor for a convolution weights gradient
    ///     primitive.
    convolution_backward_weights(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a convolution weights gradient primitive from a cache blob.
    /// @param pd Primitive descriptor for a convolution weights gradient
    ///     primitive.
    /// @param cache_blob Cache blob.
    convolution_backward_weights(
        const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// @} zendnn_api_convolution
//
/// @addtogroup zendnn_api_deconvolution Deconvolution
///
/// A primitive to perform 1D, 2D or 3D deconvolution. Supported variants are
/// forward propagation, backward propagation, and weights gradient with or
/// without bias.
///
/// @{

/// Deconvolution forward propagation primitive.
struct deconvolution_forward : public primitive {
    /// Descriptor for a deconvolution forward propagation primitive.
    struct desc {
        zendnn_deconvolution_desc_t data;

        /// Constructs a descriptor for a deconvolution forward propagation
        /// primitive with bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #zendnn::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p padding_l, and @p padding_r contain values
        /// for spatial dimensions only and hence must have the same number of
        /// elements as there are spatial dimensions. The order of values is
        /// the same as in the tensor: depth (for 3D tensors), height (for 3D
        /// and 2D tensors), and width.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #zendnn::prop_kind::forward_training, and
        ///     #zendnn::prop_kind::forward_inference.
        /// @param aalgorithm Deconvolution algorithm:
        ///     #zendnn::algorithm::deconvolution_direct, and
        ///     #zendnn::algorithm::deconvolution_winograd.
        /// @param src_desc Source memory descriptor.
        /// @param weights_desc Weights memory descriptor.
        /// @param bias_desc Bias memory descriptor. Passing zero memory
        ///     descriptor disables the bias term.
        /// @param dst_desc Destination memory descriptor.
        /// @param strides Vector of strides for spatial dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        desc(prop_kind aprop_kind, algorithm aalgorithm,
             const memory::desc &src_desc, const memory::desc &weights_desc,
             const memory::desc &bias_desc, const memory::desc &dst_desc,
             const memory::dims &strides, const memory::dims &padding_l,
             const memory::dims &padding_r) {
            memory::validate_dims(strides, src_desc.data.ndims - 2);
            memory::validate_dims(padding_l, src_desc.data.ndims - 2);
            memory::validate_dims(padding_r, src_desc.data.ndims - 2);
            error::wrap_c_api(
                zendnn_deconvolution_forward_desc_init(&data,
                        zendnn::convert_to_c(aprop_kind),
                        convert_to_c(aalgorithm), &src_desc.data,
                        &weights_desc.data, &bias_desc.data, &dst_desc.data,
                        &strides[0], &padding_l[0], &padding_r[0]),
                "could not create a descriptor for a deconvolution forward "
                "propagation primitive");
        }

        /// Constructs a descriptor for a deconvolution forward propagation
        /// primitive without bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #zendnn::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p padding_l, and @p padding_r contain values
        /// for spatial dimensions only and hence must have the same number of
        /// elements as there are spatial dimensions. The order of values is
        /// the same as in the tensor: depth (for 3D tensors), height (for 3D
        /// and 2D tensors), and width.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #zendnn::prop_kind::forward_training, and
        ///     #zendnn::prop_kind::forward_inference.
        /// @param aalgorithm Deconvolution algorithm:
        ///     #zendnn::algorithm::deconvolution_direct, and
        ///     #zendnn::algorithm::deconvolution_winograd.
        /// @param src_desc Source memory descriptor.
        /// @param weights_desc Weights memory descriptor.
        /// @param dst_desc Destination memory descriptor.
        /// @param strides Vector of strides for spatial dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        desc(prop_kind aprop_kind, algorithm aalgorithm,
             const memory::desc &src_desc, const memory::desc &weights_desc,
             const memory::desc &dst_desc, const memory::dims &strides,
             const memory::dims &padding_l, const memory::dims &padding_r) {
            memory::validate_dims(strides, src_desc.data.ndims - 2);
            memory::validate_dims(padding_l, src_desc.data.ndims - 2);
            memory::validate_dims(padding_r, src_desc.data.ndims - 2);
            error::wrap_c_api(
                zendnn_deconvolution_forward_desc_init(&data,
                        zendnn::convert_to_c(aprop_kind),
                        convert_to_c(aalgorithm), &src_desc.data,
                        &weights_desc.data, nullptr, &dst_desc.data,
                        &strides[0], &padding_l[0], &padding_r[0]),
                "could not create a descriptor for a deconvolution forward "
                "propagation primitive");
        }

        /// Constructs a descriptor for a dilated deconvolution forward
        /// propagation primitive with bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #zendnn::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p dilates, @p padding_l, and @p padding_r
        /// contain values for spatial dimensions only and hence must have the
        /// same number of elements as there are spatial dimensions. The order
        /// of values is the same as in the tensor: depth (for 3D tensors),
        /// height (for 3D and 2D tensors), and width.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #zendnn::prop_kind::forward_training, and
        ///     #zendnn::prop_kind::forward_inference.
        /// @param aalgorithm Deconvolution algorithm:
        ///     #zendnn::algorithm::deconvolution_direct, and
        ///     #zendnn::algorithm::deconvolution_winograd.
        /// @param src_desc Source memory descriptor.
        /// @param weights_desc Weights memory descriptor.
        /// @param bias_desc Bias memory descriptor. Passing zero memory
        ///     descriptor disables the bias term.
        /// @param dst_desc Destination memory descriptor.
        /// @param strides Vector of strides for spatial dimension.
        /// @param dilates Dilations for each spatial dimension. A zero value
        ///     means no dilation in the corresponding dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        desc(prop_kind aprop_kind, algorithm aalgorithm,
             const memory::desc &src_desc, const memory::desc &weights_desc,
             const memory::desc &bias_desc, const memory::desc &dst_desc,
             const memory::dims &strides, const memory::dims &dilates,
             const memory::dims &padding_l, const memory::dims &padding_r) {
            memory::validate_dims(strides, src_desc.data.ndims - 2);
            memory::validate_dims(dilates, src_desc.data.ndims - 2);
            memory::validate_dims(padding_l, src_desc.data.ndims - 2);
            memory::validate_dims(padding_r, src_desc.data.ndims - 2);
            error::wrap_c_api(zendnn_dilated_deconvolution_forward_desc_init(
                                  &data, zendnn::convert_to_c(aprop_kind),
                                  convert_to_c(aalgorithm), &src_desc.data,
                                  &weights_desc.data, &bias_desc.data,
                                  &dst_desc.data, &strides[0], &dilates[0],
                                  &padding_l[0], &padding_r[0]),
                              "could not create a descriptor for a dilated deconvolution "
                              "forward propagation primitive");
        }

        /// Constructs a descriptor for a dilated deconvolution forward
        /// propagation primitive without bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #zendnn::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p dilates, @p padding_l, and @p padding_r
        /// contain values for spatial dimensions only and hence must have the
        /// same number of elements as there are spatial dimensions. The order
        /// of values is the same as in the tensor: depth (for 3D tensors),
        /// height (for 3D and 2D tensors), and width.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #zendnn::prop_kind::forward_training, and
        ///     #zendnn::prop_kind::forward_inference.
        /// @param aalgorithm Deconvolution algorithm:
        ///     #zendnn::algorithm::deconvolution_direct, and
        ///     #zendnn::algorithm::deconvolution_winograd.
        /// @param src_desc Source memory descriptor.
        /// @param weights_desc Weights memory descriptor.
        /// @param dst_desc Destination memory descriptor.
        /// @param strides Vector of strides for spatial dimension.
        /// @param dilates Dilations for each spatial dimension. A zero value
        ///     means no dilation in the corresponding dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        desc(prop_kind aprop_kind, algorithm aalgorithm,
             const memory::desc &src_desc, const memory::desc &weights_desc,
             const memory::desc &dst_desc, const memory::dims &strides,
             const memory::dims &dilates, const memory::dims &padding_l,
             const memory::dims &padding_r) {
            memory::validate_dims(strides, src_desc.data.ndims - 2);
            memory::validate_dims(dilates, src_desc.data.ndims - 2);
            memory::validate_dims(padding_l, src_desc.data.ndims - 2);
            memory::validate_dims(padding_r, src_desc.data.ndims - 2);
            error::wrap_c_api(zendnn_dilated_deconvolution_forward_desc_init(
                                  &data, zendnn::convert_to_c(aprop_kind),
                                  convert_to_c(aalgorithm), &src_desc.data,
                                  &weights_desc.data, nullptr,
                                  &dst_desc.data, &strides[0], &dilates[0],
                                  &padding_l[0], &padding_r[0]),
                              "could not create a descriptor for a dilated deconvolution "
                              "forward propagation primitive");
        }
    };

    /// Primitive descriptor for a deconvolution forward propagation primitive.
    struct primitive_desc : public zendnn::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a deconvolution forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a deconvolution forward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                       bool allow_empty = false)
            : zendnn::primitive_desc(
                  &adesc.data, nullptr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a deconvolution forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a deconvolution forward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param attr Primitive attributes to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                       const engine &aengine, bool allow_empty = false)
            : zendnn::primitive_desc(
                  &adesc.data, &attr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a deconvolution forward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a deconvolution forward
        ///     propagation primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : zendnn::primitive_desc(pd, zendnn::primitive::kind::deconvolution,
                                     zendnn::prop_kind::forward_training,
                                     zendnn::prop_kind::forward_inference) {}

        /// @copydoc zendnn::primitive_desc_base::src_desc()const
        memory::desc src_desc() const {
            return base::src_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::weights_desc()const
        memory::desc weights_desc() const {
            return base::weights_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const {
            return base::dst_desc(0);
        }

        /// @copydoc zendnn::convolution_forward::primitive_desc::bias_desc()const
        memory::desc bias_desc() const {
            return base::weights_desc(1);
        }
    };

    /// Default constructor. Produces an empty object.
    deconvolution_forward() = default;

    /// Constructs a deconvolution forward propagation primitive.
    /// @param pd Primitive descriptor for a deconvolution forward propagation
    ///     primitive.
    deconvolution_forward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a deconvolution forward propagation primitive from a cache
    ///     blob.
    /// @param pd Primitive descriptor for a deconvolution forward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    deconvolution_forward(
        const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// Deconvolution backward propagation primitive.
struct deconvolution_backward_data : public primitive {
    /// Descriptor for a deconvolution backward propagation primitive.
    struct desc {
        zendnn_deconvolution_desc_t data;

        /// Constructs a descriptor for a deconvolution backward propagation
        /// primitive.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #zendnn::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p padding_l, and @p padding_r contain values
        /// for spatial dimensions only and hence must have the same number of
        /// elements as there are spatial dimensions. The order of values is
        /// the same as in the tensor: depth (for 3D tensors), height (for 3D
        /// and 2D tensors), and width.
        ///
        /// @param aalgorithm Deconvolution algorithm
        ///     (#zendnn::algorithm::convolution_direct,
        ///     #zendnn::algorithm::convolution_winograd).
        /// @param diff_src_desc Diff source memory descriptor.
        /// @param weights_desc Weights memory descriptor.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param strides Strides for each spatial dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        desc(algorithm aalgorithm, const memory::desc &diff_src_desc,
             const memory::desc &weights_desc,
             const memory::desc &diff_dst_desc, const memory::dims &strides,
             const memory::dims &padding_l, const memory::dims &padding_r) {
            memory::validate_dims(strides, diff_src_desc.data.ndims - 2);
            memory::validate_dims(padding_l, diff_src_desc.data.ndims - 2);
            memory::validate_dims(padding_r, diff_src_desc.data.ndims - 2);
            error::wrap_c_api(
                zendnn_deconvolution_backward_data_desc_init(&data,
                        convert_to_c(aalgorithm), &diff_src_desc.data,
                        &weights_desc.data, &diff_dst_desc.data,
                        &strides[0], &padding_l[0], &padding_r[0]),
                "could not create a descriptor for a deconvolution "
                "backward propagation primitive");
        }

        /// Constructs a descriptor for a dilated deconvolution backward
        /// propagation primitive.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #zendnn::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p dilates, @p padding_l, and @p padding_r
        /// contain values for spatial dimensions only and hence must have the
        /// same number of elements as there are spatial dimensions. The order
        /// of values is the same as in the tensor: depth (for 3D tensors),
        /// height (for 3D and 2D tensors), and width.
        ///
        /// @param aalgorithm Deconvolution algorithm
        ///     (#zendnn::algorithm::convolution_direct,
        ///     #zendnn::algorithm::convolution_winograd).
        /// @param diff_src_desc Diff source memory descriptor.
        /// @param weights_desc Weights memory descriptor.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param strides Strides for each spatial dimension.
        /// @param dilates Dilations for each spatial dimension. A zero value
        ///     means no dilation in the corresponding dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        desc(algorithm aalgorithm, const memory::desc &diff_src_desc,
             const memory::desc &weights_desc,
             const memory::desc &diff_dst_desc, const memory::dims &strides,
             const memory::dims &dilates, const memory::dims &padding_l,
             const memory::dims &padding_r) {
            memory::validate_dims(strides, diff_src_desc.data.ndims - 2);
            memory::validate_dims(dilates, diff_src_desc.data.ndims - 2);
            memory::validate_dims(padding_l, diff_src_desc.data.ndims - 2);
            memory::validate_dims(padding_r, diff_src_desc.data.ndims - 2);
            error::wrap_c_api(
                zendnn_dilated_deconvolution_backward_data_desc_init(&data,
                        convert_to_c(aalgorithm), &diff_src_desc.data,
                        &weights_desc.data, &diff_dst_desc.data,
                        &strides[0], &dilates[0], &padding_l[0],
                        &padding_r[0]),
                "could not create a descriptor for a dilated deconvolution "
                "backward propagation primitive");
        }
    };

    /// Primitive descriptor for a deconvolution backward propagation primitive.
    struct primitive_desc : public zendnn::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a deconvolution backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a deconvolution backward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a deconvolution forward
        ///     propagation primitive. It is used as a hint for deciding which
        ///     memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                       const deconvolution_forward::primitive_desc &hint_fwd_pd,
                       bool allow_empty = false)
            : zendnn::primitive_desc(&adesc.data, nullptr, aengine,
                                     hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a deconvolution backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a deconvolution backward propagation
        ///     primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a deconvolution forward
        ///     propagation primitive. It is used as a hint for deciding which
        ///     memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                       const engine &aengine,
                       const deconvolution_forward::primitive_desc &hint_fwd_pd,
                       bool allow_empty = false)
            : zendnn::primitive_desc(&adesc.data, &attr, aengine,
                                     hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a deconvolution backward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a deconvolution backward
        ///     propagation primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : zendnn::primitive_desc(pd, zendnn::primitive::kind::deconvolution,
                                     zendnn::prop_kind::backward_data) {}

        /// @copydoc zendnn::primitive_desc_base::diff_src_desc()const
        memory::desc diff_src_desc() const {
            return base::diff_src_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::weights_desc()const
        memory::desc weights_desc() const {
            return base::weights_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::diff_dst_desc()const
        memory::desc diff_dst_desc() const {
            return base::diff_dst_desc(0);
        }
    };

    /// Default constructor. Produces an empty object.
    deconvolution_backward_data() = default;

    /// Constructs a deconvolution backward propagation primitive.
    /// @param pd Primitive descriptor for a deconvolution backward propagation
    ///     primitive.
    deconvolution_backward_data(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a deconvolution backward propagation primitive from a cache
    ///     blob.
    /// @param pd Primitive descriptor for a deconvolution backward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    deconvolution_backward_data(
        const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// Deconvolution weights gradient primitive.
struct deconvolution_backward_weights : public primitive {
    /// Descriptor for a deconvolution weights gradient primitive.
    struct desc {
        zendnn_deconvolution_desc_t data;

        /// Constructs a descriptor for a deconvolution weights gradient
        /// primitive with bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #zendnn::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p padding_l, and @p padding_r contain values
        /// for spatial dimensions only and hence must have the same number of
        /// elements as there are spatial dimensions. The order of values is
        /// the same as in the tensor: depth (for 3D tensors), height (for 3D
        /// and 2D tensors), and width.
        ///
        /// @param aalgorithm Deconvolution algorithm. Possible values are
        ///     #zendnn::algorithm::deconvolution_direct, and
        ///     #zendnn::algorithm::deconvolution_winograd.
        /// @param src_desc Source memory descriptor.
        /// @param diff_weights_desc Diff weights memory descriptor.
        /// @param diff_bias_desc Diff bias memory descriptor. Passing zero
        ///     memory descriptor disables the bias term.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param strides Strides for each spatial dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        desc(algorithm aalgorithm, const memory::desc &src_desc,
             const memory::desc &diff_weights_desc,
             const memory::desc &diff_bias_desc,
             const memory::desc &diff_dst_desc, const memory::dims &strides,
             const memory::dims &padding_l, const memory::dims &padding_r) {
            memory::validate_dims(strides, src_desc.data.ndims - 2);
            memory::validate_dims(padding_l, src_desc.data.ndims - 2);
            memory::validate_dims(padding_r, src_desc.data.ndims - 2);
            error::wrap_c_api(
                zendnn_deconvolution_backward_weights_desc_init(&data,
                        convert_to_c(aalgorithm), &src_desc.data,
                        &diff_weights_desc.data, &diff_bias_desc.data,
                        &diff_dst_desc.data, &strides[0], &padding_l[0],
                        &padding_r[0]),
                "could not create a descriptor for a deconvolution weights "
                "update primitive");
        }

        /// Constructs a descriptor for a deconvolution weights gradient
        /// primitive without bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #zendnn::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p padding_l, and @p padding_r contain values
        /// for spatial dimensions only and hence must have the same number of
        /// elements as there are spatial dimensions. The order of values is
        /// the same as in the tensor: depth (for 3D tensors), height (for 3D
        /// and 2D tensors), and width.
        ///
        /// @param aalgorithm Deconvolution algorithm. Possible values are
        ///     #zendnn::algorithm::deconvolution_direct, and
        ///     #zendnn::algorithm::deconvolution_winograd.
        /// @param src_desc Source memory descriptor.
        /// @param diff_weights_desc Diff weights memory descriptor.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param strides Strides for each spatial dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        desc(algorithm aalgorithm, const memory::desc &src_desc,
             const memory::desc &diff_weights_desc,
             const memory::desc &diff_dst_desc, const memory::dims &strides,
             const memory::dims &padding_l, const memory::dims &padding_r) {
            memory::validate_dims(strides, src_desc.data.ndims - 2);
            memory::validate_dims(padding_l, src_desc.data.ndims - 2);
            memory::validate_dims(padding_r, src_desc.data.ndims - 2);
            error::wrap_c_api(zendnn_deconvolution_backward_weights_desc_init(
                                  &data, convert_to_c(aalgorithm),
                                  &src_desc.data, &diff_weights_desc.data,
                                  nullptr, &diff_dst_desc.data, &strides[0],
                                  &padding_l[0], &padding_r[0]),
                              "could not create a descriptor for a deconvolution weights "
                              "update primitive");
        }

        /// Constructs a descriptor for a dilated deconvolution weights gradient
        /// primitive with bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #zendnn::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p dilates, @p padding_l, and @p padding_r
        /// contain values for spatial dimensions only and hence must have the
        /// same number of elements as there are spatial dimensions. The order
        /// of values is the same as in the tensor: depth (for 3D tensors),
        /// height (for 3D and 2D tensors), and width.
        ///
        /// @param aalgorithm Deconvolution algorithm. Possible values are
        ///     #zendnn::algorithm::deconvolution_direct, and
        ///     #zendnn::algorithm::deconvolution_winograd.
        /// @param src_desc Source memory descriptor.
        /// @param diff_weights_desc Diff weights memory descriptor.
        /// @param diff_bias_desc Diff bias memory descriptor. Passing zero
        ///     memory descriptor disables the bias term.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param strides Strides for each spatial dimension.
        /// @param dilates Dilations for each spatial dimension. A zero value
        ///     means no dilation in the corresponding dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        desc(algorithm aalgorithm, const memory::desc &src_desc,
             const memory::desc &diff_weights_desc,
             const memory::desc &diff_bias_desc,
             const memory::desc &diff_dst_desc, const memory::dims &strides,
             const memory::dims &dilates, const memory::dims &padding_l,
             const memory::dims &padding_r) {
            memory::validate_dims(strides, src_desc.data.ndims - 2);
            memory::validate_dims(dilates, src_desc.data.ndims - 2);
            memory::validate_dims(padding_l, src_desc.data.ndims - 2);
            memory::validate_dims(padding_r, src_desc.data.ndims - 2);
            error::wrap_c_api(
                zendnn_dilated_deconvolution_backward_weights_desc_init(&data,
                        convert_to_c(aalgorithm), &src_desc.data,
                        &diff_weights_desc.data, &diff_bias_desc.data,
                        &diff_dst_desc.data, &strides[0], &dilates[0],
                        &padding_l[0], &padding_r[0]),
                "could not create a descriptor for a dilated deconvolution "
                "weights gradient primitive");
        }

        /// Constructs a descriptor for a dilated deconvolution weights gradient
        /// primitive without bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #zendnn::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p dilates, @p padding_l, and @p padding_r
        /// contain values for spatial dimensions only and hence must have the
        /// same number of elements as there are spatial dimensions. The order
        /// of values is the same as in the tensor: depth (for 3D tensors),
        /// height (for 3D and 2D tensors), and width.
        ///
        /// @param aalgorithm Deconvolution algorithm. Possible values are
        ///     #zendnn::algorithm::deconvolution_direct, and
        ///     #zendnn::algorithm::deconvolution_winograd.
        /// @param src_desc Source memory descriptor.
        /// @param diff_weights_desc Diff weights memory descriptor.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param strides Strides for each spatial dimension.
        /// @param dilates Dilations for each spatial dimension. A zero value
        ///     means no dilation in the corresponding dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        desc(algorithm aalgorithm, const memory::desc &src_desc,
             const memory::desc &diff_weights_desc,
             const memory::desc &diff_dst_desc, const memory::dims &strides,
             const memory::dims &dilates, const memory::dims &padding_l,
             const memory::dims &padding_r) {
            memory::validate_dims(strides, src_desc.data.ndims - 2);
            memory::validate_dims(dilates, src_desc.data.ndims - 2);
            memory::validate_dims(padding_l, src_desc.data.ndims - 2);
            memory::validate_dims(padding_r, src_desc.data.ndims - 2);
            error::wrap_c_api(
                zendnn_dilated_deconvolution_backward_weights_desc_init(&data,
                        convert_to_c(aalgorithm), &src_desc.data,
                        &diff_weights_desc.data, nullptr,
                        &diff_dst_desc.data, &strides[0], &dilates[0],
                        &padding_l[0], &padding_r[0]),
                "could not create a descriptor for a dilated deconvolution "
                "weights gradient primitive");
        }
    };

    /// Primitive descriptor for a deconvolution weights gradient primitive.
    struct primitive_desc : public zendnn::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a deconvolution weights
        /// update primitive.
        ///
        /// @param adesc Descriptor for a deconvolution weights gradient
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a deconvolution forward
        ///     propagation primitive. It is used as a hint for deciding which
        ///     memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception.  In this case
        ///     an empty object will be produced.  This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                       const deconvolution_forward::primitive_desc &hint_fwd_pd,
                       bool allow_empty = false)
            : zendnn::primitive_desc(&adesc.data, nullptr, aengine,
                                     hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a deconvolution weights
        /// update primitive.
        ///
        /// @param adesc Descriptor for a deconvolution weights gradient
        ///     primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a deconvolution forward
        ///     propagation primitive. It is used as a hint for deciding which
        ///     memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                       const engine &aengine,
                       const deconvolution_forward::primitive_desc &hint_fwd_pd,
                       bool allow_empty = false)
            : zendnn::primitive_desc(&adesc.data, &attr, aengine,
                                     hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a deconvolution weights
        /// gradient primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a deconvolution weights
        ///     gradient primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : zendnn::primitive_desc(pd, zendnn::primitive::kind::deconvolution,
                                     zendnn::prop_kind::backward_weights) {}

        /// @copydoc zendnn::primitive_desc_base::src_desc()const
        memory::desc src_desc() const {
            return base::src_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::diff_weights_desc()const
        memory::desc diff_weights_desc() const {
            return base::diff_weights_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::diff_dst_desc()const
        memory::desc diff_dst_desc() const {
            return base::diff_dst_desc(0);
        }

        /// @copydoc zendnn::convolution_backward_weights::primitive_desc::diff_bias_desc()const
        memory::desc diff_bias_desc() const {
            return base::diff_weights_desc(1);
        }
    };

    /// Default constructor. Produces an empty object.
    deconvolution_backward_weights() = default;

    /// Constructs a deconvolution weights gradient primitive.
    /// @param pd Primitive descriptor for a deconvolution weights gradient
    ///     primitive.
    deconvolution_backward_weights(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a deconvolution weights gradient primitive from a cache
    ///     blob.
    /// @param pd Primitive descriptor for a deconvolution weights gradient
    ///     primitive.
    /// @param cache_blob Cache blob.
    deconvolution_backward_weights(
        const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// @} zendnn_api_deconvolution

/// @addtogroup zendnn_api_lrn LRN
///
/// A primitive to perform local response normalization (LRN) across or within
/// channels.
///
/// @sa @ref dev_guide_lrn in developer guide
///
/// @{

/// Local response normalization (LRN) forward propagation primitive.
struct lrn_forward : public primitive {
    /// Descriptor for an LRN forward propagation primitive.
    struct desc {
        zendnn_lrn_desc_t data;

        /// Constructs a descriptor for a LRN forward propagation primitive.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #zendnn::prop_kind::forward_training, and
        ///     #zendnn::prop_kind::forward_inference.
        /// @param aalgorithm LRN algorithm kind: either
        ///     #zendnn::algorithm::lrn_across_channels, or
        ///     #zendnn::algorithm::lrn_within_channel.
        /// @param data_desc Source and destination memory descriptors.
        /// @param local_size Regularization local size.
        /// @param alpha The alpha regularization parameter.
        /// @param beta The beta regularization parameter.
        /// @param k The k regularization parameter.
        desc(prop_kind aprop_kind, algorithm aalgorithm,
             const memory::desc &data_desc, memory::dim local_size,
             float alpha, float beta, float k = 1.f) {
            error::wrap_c_api(zendnn_lrn_forward_desc_init(&data,
                              zendnn::convert_to_c(aprop_kind),
                              convert_to_c(aalgorithm), &data_desc.data,
                              local_size, alpha, beta, k),
                              "could not create a descriptor for a lrn forward "
                              "propagation primitive");
        }
    };

    /// Primitive descriptor for an LRN forward propagation primitive.
    struct primitive_desc : public zendnn::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an LRN forward propagation
        /// primitive.
        ///
        /// @param adesc Descriptor for an LRN forward propagation primitive.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                       bool allow_empty = false)
            : zendnn::primitive_desc(
                  &adesc.data, nullptr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for an LRN forward propagation
        /// primitive.
        ///
        /// @param adesc Descriptor for an LRN forward propagation primitive.
        /// @param aengine Engine to use.
        /// @param attr Primitive attributes to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                       const engine &aengine, bool allow_empty = false)
            : zendnn::primitive_desc(
                  &adesc.data, &attr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for an LRN forward propagation
        /// primitive from a C API primitive descriptor that must have a
        /// matching kind.
        ///
        /// @param pd C API primitive descriptor for an LRN forward
        ///     propagation primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : zendnn::primitive_desc(pd, zendnn::primitive::kind::lrn,
                                     zendnn::prop_kind::forward_training,
                                     zendnn::prop_kind::forward_inference) {}

        /// @copydoc zendnn::primitive_desc_base::src_desc()const
        memory::desc src_desc() const {
            return base::src_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const {
            return base::dst_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const {
            return base::workspace_desc();
        }
    };

    /// Default constructor. Produces an empty object.
    lrn_forward() = default;

    /// Constructs an LRN forward propagation primitive.
    /// @param pd Primitive descriptor for an LRN forward propagation
    ///     primitive.
    lrn_forward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs an LRN forward propagation primitive from a cache blob.
    /// @param pd Primitive descriptor for an LRN forward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    lrn_forward(
        const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// Local response normalization (LRN) backward propagation primitive.
struct lrn_backward : public primitive {
    /// Descriptor for an LRN backward propagation primitive.
    struct desc {
        zendnn_lrn_desc_t data;

        /// Constructs a descriptor for an LRN backward propagation primitive.
        ///
        /// @param aalgorithm LRN algorithm kind: either
        ///     #zendnn::algorithm::lrn_across_channels, or
        ///     #zendnn::algorithm::lrn_within_channel.
        /// @param diff_data_desc Diff source and diff destination memory
        ///     descriptor.
        /// @param data_desc Source memory descriptor.
        /// @param local_size Regularization local size.
        /// @param alpha The alpha regularization parameter.
        /// @param beta The beta regularization parameter.
        /// @param k The k regularization parameter.
        desc(algorithm aalgorithm, const memory::desc &data_desc,
             const memory::desc &diff_data_desc, memory::dim local_size,
             float alpha, float beta, float k = 1.f) {
            error::wrap_c_api(
                zendnn_lrn_backward_desc_init(&data, convert_to_c(aalgorithm),
                                              &diff_data_desc.data, &data_desc.data, local_size,
                                              alpha, beta, k),
                "could not create a descriptor for a lrn backward "
                "propagation primitive");
        }
    };

    /// Primitive descriptor for an LRN backward propagation primitive.
    struct primitive_desc : public zendnn::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an LRN backward propagation
        /// primitive.
        ///
        /// @param adesc Descriptor for an LRN backward propagation primitive.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for an LRN forward
        ///     propagation primitive. It is used as a hint for deciding which
        ///     memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                       const lrn_forward::primitive_desc &hint_fwd_pd,
                       bool allow_empty = false)
            : zendnn::primitive_desc(&adesc.data, nullptr, aengine,
                                     hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for an LRN backward propagation
        /// primitive.
        ///
        /// @param adesc Descriptor for an LRN backward propagation primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for an LRN forward
        ///     propagation primitive. It is used as a hint for deciding which
        ///     memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                       const engine &aengine,
                       const lrn_forward::primitive_desc &hint_fwd_pd,
                       bool allow_empty = false)
            : zendnn::primitive_desc(&adesc.data, &attr, aengine,
                                     hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for an LRN backward propagation
        /// primitive from a C API primitive descriptor that must have a
        /// matching kind.
        ///
        /// @param pd C API primitive descriptor for an LRN backward
        ///     propagation primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : zendnn::primitive_desc(pd, zendnn::primitive::kind::lrn,
                                     zendnn::prop_kind::backward_data) {}

        /// @copydoc zendnn::primitive_desc_base::src_desc()const
        memory::desc diff_src_desc() const {
            return base::diff_src_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::diff_dst_desc()const
        memory::desc diff_dst_desc() const {
            return base::diff_dst_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const {
            return base::workspace_desc();
        }
    };

    /// Default constructor. Produces an empty object.
    lrn_backward() = default;

    /// Constructs an LRN backward propagation primitive.
    /// @param pd Primitive descriptor for an LRN backward propagation
    ///     primitive.
    lrn_backward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs an LRN backward propagation primitive from a cache blob.
    /// @param pd Primitive descriptor for an LRN backward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    lrn_backward(
        const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// @} zendnn_api_lrn

/// @addtogroup zendnn_api_pooling Pooling
///
/// A primitive to perform max or average pooling.
///
/// @sa @ref dev_guide_pooling in developer guide
///
/// @{

/// Pooling forward propagation primitive.
struct pooling_forward : public primitive {
    /// Descriptor for a pooling forward propagation primitive.
    struct desc {
        zendnn_pooling_desc_t data;

        /// Constructs a descriptor for pooling forward propagation primitive.
        ///
        /// Arrays @p strides, @p kernel, @p padding_l, and @p padding_r
        /// contain values for spatial dimensions only and hence must have the
        /// same number of elements as there are spatial dimensions. The order
        /// of values is the same as in the tensor: depth (for 3D tensors),
        /// height (for 3D and 2D tensors), and width.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #zendnn::prop_kind::forward_training, and
        ///     #zendnn::prop_kind::forward_inference.
        /// @param aalgorithm Pooling algorithm kind: either
        ///     #zendnn::algorithm::pooling_max,
        ///     #zendnn::algorithm::pooling_avg_include_padding,
        ///     or #zendnn::algorithm::pooling_avg (same as
        ///     #zendnn::algorithm::pooling_avg_exclude_padding).
        /// @param src_desc Source memory descriptor.
        /// @param dst_desc Destination memory descriptor.
        /// @param strides Vector of strides for spatial dimension.
        /// @param kernel Vector of kernel spatial dimensions.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        desc(prop_kind aprop_kind, algorithm aalgorithm,
             const memory::desc &src_desc, const memory::desc &dst_desc,
             const memory::dims &strides, const memory::dims &kernel,
             const memory::dims &padding_l, const memory::dims &padding_r) {
            memory::validate_dims(strides, src_desc.data.ndims - 2);
            memory::validate_dims(kernel, src_desc.data.ndims - 2);
            memory::validate_dims(padding_l, src_desc.data.ndims - 2);
            memory::validate_dims(padding_r, src_desc.data.ndims - 2);
            error::wrap_c_api(zendnn_pooling_forward_desc_init(&data,
                              zendnn::convert_to_c(aprop_kind),
                              convert_to_c(aalgorithm), &src_desc.data,
                              &dst_desc.data, &strides[0], &kernel[0],
                              &padding_l[0], &padding_r[0]),
                              "could not create a descriptor for a pooling forward "
                              "propagation primitive");
        }
    };

    /// Primitive descriptor for a pooling forward propagation primitive.
    struct primitive_desc : public zendnn::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a pooling forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a pooling forward propagation primitive.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                       bool allow_empty = false)
            : zendnn::primitive_desc(
                  &adesc.data, nullptr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a pooling forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a pooling forward propagation primitive.
        /// @param aengine Engine to use.
        /// @param attr Primitive attributes to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                       const engine &aengine, bool allow_empty = false)
            : zendnn::primitive_desc(
                  &adesc.data, &attr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a pooling forward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a pooling forward
        ///     propagation primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : zendnn::primitive_desc(pd, zendnn::primitive::kind::pooling,
                                     zendnn::prop_kind::forward_training,
                                     zendnn::prop_kind::forward_inference) {}

        /// @copydoc zendnn::primitive_desc_base::src_desc()const
        memory::desc src_desc() const {
            return base::src_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const {
            return base::dst_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const {
            return base::workspace_desc();
        }
    };

    /// Default constructor. Produces an empty object.
    pooling_forward() = default;

    /// Constructs a pooling forward propagation primitive.
    /// @param pd Primitive descriptor for a pooling forward propagation
    ///     primitive.
    pooling_forward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a pooling forward propagation primitive from a cache blob.
    /// @param pd Primitive descriptor for a pooling forward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    pooling_forward(
        const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// Pooling backward propagation primitive.
struct pooling_backward : public primitive {
    /// Descriptor for a pooling backward propagation primitive.
    struct desc {
        zendnn_pooling_desc_t data;

        /// Constructs a descriptor for pooling backward propagation primitive.
        ///
        /// Arrays @p strides, @p kernel, @p padding_l, and @p padding_r
        /// contain values for spatial dimensions only and hence must have the
        /// same number of elements as there are spatial dimensions. The order
        /// of values is the same as in the tensor: depth (for 3D tensors),
        /// height (for 3D and 2D tensors), and width.
        ///
        /// @param aalgorithm Pooling algorithm kind: either
        ///     #zendnn::algorithm::pooling_max,
        ///     #zendnn::algorithm::pooling_avg_include_padding,
        ///     or #zendnn::algorithm::pooling_avg (same as
        ///     #zendnn::algorithm::pooling_avg_exclude_padding).
        /// @param diff_src_desc Diff source memory descriptor.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param strides Vector of strides for spatial dimension.
        /// @param kernel Vector of kernel spatial dimensions.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        desc(algorithm aalgorithm, const memory::desc &diff_src_desc,
             const memory::desc &diff_dst_desc, const memory::dims &strides,
             const memory::dims &kernel, const memory::dims &padding_l,
             const memory::dims &padding_r) {
            memory::validate_dims(strides, diff_src_desc.data.ndims - 2);
            memory::validate_dims(kernel, diff_src_desc.data.ndims - 2);
            memory::validate_dims(padding_l, diff_src_desc.data.ndims - 2);
            memory::validate_dims(padding_r, diff_src_desc.data.ndims - 2);
            error::wrap_c_api(
                zendnn_pooling_backward_desc_init(&data,
                                                  convert_to_c(aalgorithm), &diff_src_desc.data,
                                                  &diff_dst_desc.data, &strides[0], &kernel[0],
                                                  &padding_l[0], &padding_r[0]),
                "could not create a descriptor for a pooling backward "
                "propagation primitive");
        }
    };

    /// Primitive descriptor for a pooling backward propagation primitive.
    struct primitive_desc : public zendnn::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a pooling backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a pooling backward propagation primitive.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a pooling forward
        ///     propagation primitive. It is used as a hint for deciding which
        ///     memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                       const pooling_forward::primitive_desc &hint_fwd_pd,
                       bool allow_empty = false)
            : zendnn::primitive_desc(&adesc.data, nullptr, aengine,
                                     hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a pooling backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a pooling backward propagation primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a pooling forward
        ///     propagation primitive. It is used as a hint for deciding which
        ///     memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                       const engine &aengine,
                       const pooling_forward::primitive_desc &hint_fwd_pd,
                       bool allow_empty = false)
            : zendnn::primitive_desc(&adesc.data, &attr, aengine,
                                     hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a pooling backward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a pooling backward
        ///     propagation primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : zendnn::primitive_desc(pd, zendnn::primitive::kind::pooling,
                                     zendnn::prop_kind::backward_data) {}

        /// @copydoc zendnn::primitive_desc_base::src_desc()const
        memory::desc diff_src_desc() const {
            return base::diff_src_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::diff_dst_desc()const
        memory::desc diff_dst_desc() const {
            return base::diff_dst_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const {
            return base::workspace_desc();
        }
    };

    /// Default constructor. Produces an empty object.
    pooling_backward() = default;

    /// Constructs a pooling backward propagation primitive.
    /// @param pd Primitive descriptor for a pooling backward propagation
    ///     primitive.
    pooling_backward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a pooling backward propagation primitive froma cache blob.
    /// @param pd Primitive descriptor for a pooling backward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    pooling_backward(
        const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// @} zendnn_api_pooling

/// @addtogroup zendnn_api_eltwise Eltwise
///
/// A primitive to perform elementwise operations such as the
/// rectifier linear unit (ReLU).
///
/// Both forward and backward propagation primitives support in-place
/// operation; that is, src and dst can refer to the same memory for forward
/// propagation, and diff_dst and diff_src can refer to the same memory for
/// backward propagation.
///
/// @warning
///     Because the original source data is required for backward propagation,
///     in-place forward propagation is not generally supported in the
///     training mode. However, for algorithms supporting destination as input
///     memory, dst can be used for the backward propagation, which makes it
///     possible to get performance benefit even in the training mode.
///
/// @sa @ref dev_guide_eltwise in developer guide
///
/// @{

/// Elementwise unary operation forward propagation primitive.
struct eltwise_forward : public primitive {
    /// Descriptor for an elementwise forward propagation primitive.
    struct desc {
        zendnn_eltwise_desc_t data;

        /// Constructs a descriptor for an elementwise forward propagation
        /// primitive.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #zendnn::prop_kind::forward_training, and
        ///     #zendnn::prop_kind::forward_inference.
        /// @param aalgorithm Elementwise algorithm kind.
        /// @param data_desc Source and destination memory descriptors.
        /// @param alpha The alpha parameter for the elementwise operation.
        ///     Specific meaning depends on the algorithm.
        /// @param beta The beta parameter for the elementwise operation.
        ///     Specific meaning depends on the algorithm.
        desc(prop_kind aprop_kind, algorithm aalgorithm,
             const memory::desc &data_desc, float alpha = 0,
             float beta = 0) {
            error::wrap_c_api(zendnn_eltwise_forward_desc_init(&data,
                              zendnn::convert_to_c(aprop_kind),
                              zendnn::convert_to_c(aalgorithm),
                              &data_desc.data, alpha, beta),
                              "could not create a descriptor for an eltwise forward "
                              "propagation primitive");
        }
    };

    /// Primitive descriptor for an elementwise forward propagation primitive.
    struct primitive_desc : public zendnn::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an elementwise forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for an elementwise forward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                       bool allow_empty = false)
            : zendnn::primitive_desc(
                  &adesc.data, nullptr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for an elementwise forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for an elementwise forward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param attr Primitive attributes to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                       const engine &aengine, bool allow_empty = false)
            : zendnn::primitive_desc(
                  &adesc.data, &attr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for an eltwise forward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for an eltwise forward
        ///     propagation primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : zendnn::primitive_desc(pd, zendnn::primitive::kind::eltwise,
                                     zendnn::prop_kind::forward_training,
                                     zendnn::prop_kind::forward_inference) {}

        /// @copydoc zendnn::primitive_desc_base::src_desc()const
        memory::desc src_desc() const {
            return base::src_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const {
            return base::dst_desc(0);
        }
    };

    /// Default constructor. Produces an empty object.
    eltwise_forward() = default;

    /// Constructs an eltwise forward propagation primitive.
    /// @param pd Primitive descriptor for an eltwise forward propagation
    ///     primitive.
    eltwise_forward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs an eltwise forward propagation primitive from a cache blob.
    /// @param pd Primitive descriptor for an eltwise forward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    eltwise_forward(
        const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// Elementwise unary operation backward propagation primitive.
struct eltwise_backward : public primitive {
    /// Descriptor for an elementwise backward propagation primitive.
    struct desc {
        zendnn_eltwise_desc_t data;

        /// Constructs a descriptor for an elementwise backward propagation
        /// primitive.
        ///
        /// @param aalgorithm Elementwise algorithm kind.
        /// @param diff_data_desc Diff source and destination memory
        ///     descriptors.
        /// @param data_desc Source memory descriptor.
        /// @param alpha The alpha parameter for the elementwise operation.
        ///     Specific meaning depends on the algorithm.
        /// @param beta The beta parameter for the elementwise operation.
        ///     Specific meaning depends on the algorithm.
        desc(algorithm aalgorithm, const memory::desc &diff_data_desc,
             const memory::desc &data_desc, float alpha = 0,
             float beta = 0) {
            error::wrap_c_api(
                zendnn_eltwise_backward_desc_init(&data,
                                                  zendnn::convert_to_c(aalgorithm),
                                                  &diff_data_desc.data, &data_desc.data, alpha, beta),
                "could not create a descriptor for an eltwise backward "
                "propagation primitive");
        }
    };

    /// Primitive descriptor for eltwise backward propagation.
    struct primitive_desc : public zendnn::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an elementwise backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for an elementwise backward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for an elementwise forward
        ///     propagation primitive. It is used as a hint for deciding which
        ///     memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                       const eltwise_forward::primitive_desc &hint_fwd_pd,
                       bool allow_empty = false)
            : zendnn::primitive_desc(&adesc.data, nullptr, aengine,
                                     hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for an elementwise backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for an elementwise backward propagation
        ///     primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for an elementwise forward
        ///     propagation primitive. It is used as a hint for deciding which
        ///     memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                       const engine &aengine,
                       const eltwise_forward::primitive_desc &hint_fwd_pd,
                       bool allow_empty = false)
            : zendnn::primitive_desc(&adesc.data, &attr, aengine,
                                     hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for an eltwise backward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for an eltwise backward
        ///     propagation primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : zendnn::primitive_desc(pd, zendnn::primitive::kind::eltwise,
                                     zendnn::prop_kind::backward_data) {}

        /// @copydoc zendnn::primitive_desc_base::src_desc()const
        memory::desc src_desc() const {
            return base::src_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::diff_src_desc()const
        memory::desc diff_src_desc() const {
            return base::diff_src_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::diff_dst_desc()const
        memory::desc diff_dst_desc() const {
            return base::diff_dst_desc(0);
        }
    };

    /// Default constructor. Produces an empty object.
    eltwise_backward() = default;

    /// Constructs an eltwise backward propagation primitive.
    /// @param pd Primitive descriptor for an eltwise backward propagation
    ///     primitive.
    eltwise_backward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs an eltwise backward propagation primitive from a cache blob.
    /// @param pd Primitive descriptor for an eltwise backward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    eltwise_backward(
        const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// @} zendnn_api_eltwise

/// @addtogroup zendnn_api_softmax Softmax
///
/// A primitive to perform softmax.
///
/// @sa @ref dev_guide_softmax in developer guide
///
/// @{

/// Softmax forward propagation primitive.
struct softmax_forward : public primitive {
    /// Descriptor for a softmax forward propagation primitive.
    struct desc {
        zendnn_softmax_desc_t data;

        /// Default constructor. Produces an empty object.
        desc() = default;

        /// Constructs a descriptor for a softmax forward propagation
        /// primitive.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #zendnn::prop_kind::forward_training, and
        ///     #zendnn::prop_kind::forward_inference.
        /// @param data_desc Source and destination memory descriptor.
        /// @param softmax_axis Axis over which softmax is computed.
        desc(prop_kind aprop_kind, const memory::desc &data_desc,
             int softmax_axis) {
            error::wrap_c_api(zendnn_softmax_forward_desc_init(&data,
                              zendnn::convert_to_c(aprop_kind),
                              &data_desc.data, softmax_axis),
                              "could not create a descriptor for a softmax forward "
                              "propagation primitive");
        }
    };

    /// Primitive descriptor for a softmax forward propagation primitive.
    struct primitive_desc : public zendnn::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a softmax forward
        /// propagation primitive.
        ///
        /// @param adesc descriptor for a softmax forward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                       bool allow_empty = false)
            : zendnn::primitive_desc(
                  &adesc.data, nullptr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a softmax forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a softmax forward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param attr Primitive attributes to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                       const engine &aengine, bool allow_empty = false)
            : zendnn::primitive_desc(
                  &adesc.data, &attr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a softmax forward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a softmax forward
        ///     propagation primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : zendnn::primitive_desc(pd, zendnn::primitive::kind::softmax,
                                     zendnn::prop_kind::forward_training,
                                     zendnn::prop_kind::forward_inference) {}

        /// @copydoc zendnn::primitive_desc_base::src_desc()const
        memory::desc src_desc() const {
            return base::src_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const {
            return base::dst_desc(0);
        }
    };

    /// Default constructor. Produces an empty object.
    softmax_forward() = default;

    /// Constructs a softmax forward propagation primitive.
    /// @param pd Primitive descriptor for a softmax forward propagation
    ///     primitive.
    softmax_forward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a softmax forward propagation primitive from a cache blob.
    /// @param pd Primitive descriptor for a softmax forward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    softmax_forward(
        const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// Softmax backward propagation primitive.
struct softmax_backward : public primitive {
    /// Descriptor for a softmax backward propagation primitive.
    struct desc {
        zendnn_softmax_desc_t data;

        /// Default constructor. Produces an empty object.
        desc() = default;

        /// Constructs a descriptor for a softmax backward propagation
        /// primitive.
        ///
        /// @param diff_data_desc Diff source and diff destination memory
        ///     descriptor.
        /// @param data_desc Destination memory descriptor.
        /// @param softmax_axis Axis over which softmax is computed.
        desc(const memory::desc &diff_data_desc, const memory::desc &data_desc,
             int softmax_axis) {
            error::wrap_c_api(
                zendnn_softmax_backward_desc_init(&data, &diff_data_desc.data,
                                                  &data_desc.data, softmax_axis),
                "could not create a descriptor for a softmax backward "
                "propagation primitive");
        }
    };

    /// Primitive descriptor for a softmax backward propagation primitive.
    struct primitive_desc : public zendnn::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a softmax backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a softmax backward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a softmax forward
        ///     propagation primitive. It is used as a hint for deciding which
        ///     memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                       const softmax_forward::primitive_desc &hint_fwd_pd,
                       bool allow_empty = false)
            : zendnn::primitive_desc(&adesc.data, nullptr, aengine,
                                     hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a softmax backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a softmax backward propagation
        ///     primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a softmax forward
        ///     propagation primitive. It is used as a hint for deciding which
        ///     memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                       const engine &aengine,
                       const softmax_forward::primitive_desc &hint_fwd_pd,
                       bool allow_empty = false)
            : zendnn::primitive_desc(&adesc.data, &attr, aengine,
                                     hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a softmax backward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a softmax backward
        ///     propagation primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : zendnn::primitive_desc(pd, zendnn::primitive::kind::softmax,
                                     zendnn::prop_kind::backward_data) {}

        /// @copydoc zendnn::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const {
            return base::dst_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::diff_src_desc()const
        memory::desc diff_src_desc() const {
            return base::diff_src_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::dst_desc()const
        memory::desc diff_dst_desc() const {
            return base::diff_dst_desc(0);
        }
    };

    /// Default constructor. Produces an empty object.
    softmax_backward() = default;

    /// Constructs a softmax backward propagation primitive.
    /// @param pd Primitive descriptor for a softmax backward propagation
    ///     primitive.
    softmax_backward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a softmax backward propagation primitive from a cache blob.
    /// @param pd Primitive descriptor for a softmax backward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    softmax_backward(
        const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// @} zendnn_api_softmax

/// @addtogroup zendnn_api_softmax_v2 Softmax_v2
///
/// A primitive to perform softmax.
///
/// @sa @ref dev_guide_softmax in developer guide
///
/// @{

/// Softmax forward propagation primitive.
struct softmax_v2_forward : public primitive {
    /// Descriptor for a softmax forward propagation primitive.
    struct desc {
        zendnn_softmax_v2_desc_t data;

        /// Default constructor. Produces an empty object.
        desc() = default;

        /// Constructs a descriptor for a softmax forward propagation
        /// primitive.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #zendnn::prop_kind::forward_training, and
        ///     #zendnn::prop_kind::forward_inference.
        /// @param aalgorithm Softmax algorithm kind: either
        ///     #zendnn::algorithm::softmax_accurate,
        ///     or #zendnn::algorithm::softmax_log.
        /// @param src_desc Source memory descriptor.
        /// @param dst_desc Destination memory descriptor.
        /// @param softmax_axis Axis over which softmax is computed.
        desc(prop_kind aprop_kind, algorithm aalgorithm,
             const memory::desc &src_desc, const memory::desc &dst_desc,
             int softmax_axis) {
            error::wrap_c_api(
                zendnn_softmax_v2_forward_desc_init(&data,
                                                    zendnn::convert_to_c(aprop_kind),
                                                    zendnn::convert_to_c(aalgorithm), &src_desc.data,
                                                    &dst_desc.data, softmax_axis),
                "could not create a descriptor for a softmax forward "
                "propagation primitive");
        }
    };

    /// Primitive descriptor for a softmax forward propagation primitive.
    struct primitive_desc : public zendnn::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a softmax forward
        /// propagation primitive.
        ///
        /// @param adesc descriptor for a softmax forward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                       bool allow_empty = false)
            : zendnn::primitive_desc(
                  &adesc.data, nullptr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a softmax forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a softmax forward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param attr Primitive attributes to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                       const engine &aengine, bool allow_empty = false)
            : zendnn::primitive_desc(
                  &adesc.data, &attr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a softmax forward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a softmax forward
        ///     propagation primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : zendnn::primitive_desc(pd, zendnn::primitive::kind::softmax_v2,
                                     zendnn::prop_kind::forward_training,
                                     zendnn::prop_kind::forward_inference) {}

        /// @copydoc zendnn::primitive_desc_base::src_desc()const
        memory::desc src_desc() const {
            return base::src_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const {
            return base::dst_desc(0);
        }
    };

    /// Default constructor. Produces an empty object.
    softmax_v2_forward() = default;

    /// Constructs a softmax forward propagation primitive.
    /// @param pd Primitive descriptor for a softmax forward propagation
    ///     primitive.
    softmax_v2_forward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a softmax forward propagation primitive from a cache blob.
    /// @param pd Primitive descriptor for a softmax forward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    softmax_v2_forward(
        const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// Softmax backward propagation primitive.
struct softmax_v2_backward : public primitive {
    /// Descriptor for a softmax backward propagation primitive.
    struct desc {
        zendnn_softmax_v2_desc_t data;

        /// Default constructor. Produces an empty object.
        desc() = default;

        /// Constructs a descriptor for a softmax backward propagation
        /// primitive.
        ///
        /// @param aalgorithm Softmax algorithm kind: either
        ///     #zendnn::algorithm::softmax_accurate,
        ///     or #zendnn::algorithm::softmax_log.
        /// @param diff_src_desc Diff source memory descriptor.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param dst_desc Destination memory descriptor.
        /// @param softmax_axis Axis over which softmax is computed.
        desc(algorithm aalgorithm, const memory::desc &diff_src_desc,
             const memory::desc &diff_dst_desc, const memory::desc &dst_desc,
             int softmax_axis) {
            error::wrap_c_api(
                zendnn_softmax_v2_backward_desc_init(&data,
                        zendnn::convert_to_c(aalgorithm), &diff_src_desc.data,
                        &diff_dst_desc.data, &dst_desc.data, softmax_axis),
                "could not create a descriptor for a softmax backward "
                "propagation primitive");
        }
    };

    /// Primitive descriptor for a softmax backward propagation primitive.
    struct primitive_desc : public zendnn::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a softmax backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a softmax backward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a softmax forward
        ///     propagation primitive. It is used as a hint for deciding which
        ///     memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                       const softmax_v2_forward::primitive_desc &hint_fwd_pd,
                       bool allow_empty = false)
            : zendnn::primitive_desc(&adesc.data, nullptr, aengine,
                                     hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a softmax backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a softmax backward propagation
        ///     primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a softmax forward
        ///     propagation primitive. It is used as a hint for deciding which
        ///     memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                       const engine &aengine,
                       const softmax_v2_forward::primitive_desc &hint_fwd_pd,
                       bool allow_empty = false)
            : zendnn::primitive_desc(&adesc.data, &attr, aengine,
                                     hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a softmax backward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a softmax backward
        ///     propagation primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : zendnn::primitive_desc(pd, zendnn::primitive::kind::softmax_v2,
                                     zendnn::prop_kind::backward_data) {}

        /// @copydoc zendnn::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const {
            return base::dst_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::diff_src_desc()const
        memory::desc diff_src_desc() const {
            return base::diff_src_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::dst_desc()const
        memory::desc diff_dst_desc() const {
            return base::diff_dst_desc(0);
        }
    };

    /// Default constructor. Produces an empty object.
    softmax_v2_backward() = default;

    /// Constructs a softmax backward propagation primitive.
    /// @param pd Primitive descriptor for a softmax backward propagation
    ///     primitive.
    softmax_v2_backward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a softmax backward propagation primitive from a cache blob.
    /// @param pd Primitive descriptor for a softmax backward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    softmax_v2_backward(
        const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// @} zendnn_api_softmax_v2

/// @addtogroup zendnn_api_logsoftmax LogSoftmax
///
/// A primitive to perform logsoftmax.
///
/// @sa @ref dev_guide_logsoftmax in developer guide
///
/// @{

/// Logsoftmax forward propagation primitive.
struct logsoftmax_forward : public primitive {
    /// Descriptor for a logsoftmax forward propagation primitive.
    struct desc {
        zendnn_logsoftmax_desc_t data;

        /// Default constructor. Produces an empty object.
        desc() = default;

        /// Constructs a descriptor for a logsoftmax forward propagation
        /// primitive.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #zendnn::prop_kind::forward_training, and
        ///     #zendnn::prop_kind::forward_inference.
        /// @param data_desc Source and destination memory descriptor.
        /// @param logsoftmax_axis Axis over which softmax is computed.
        desc(prop_kind aprop_kind, const memory::desc &data_desc,
             int logsoftmax_axis) {
            error::wrap_c_api(zendnn_logsoftmax_forward_desc_init(&data,
                              zendnn::convert_to_c(aprop_kind),
                              &data_desc.data, logsoftmax_axis),
                              "could not create a descriptor for a logsoftmax forward "
                              "propagation primitive");
        }
    };

    /// Primitive descriptor for a logsoftmax forward propagation primitive.
    struct primitive_desc : public zendnn::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a logsoftmax forward
        /// propagation primitive.
        ///
        /// @param adesc descriptor for a logsoftmax forward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                       bool allow_empty = false)
            : zendnn::primitive_desc(
                  &adesc.data, nullptr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a logsoftmax forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a logsoftmax forward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param attr Primitive attributes to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                       const engine &aengine, bool allow_empty = false)
            : zendnn::primitive_desc(
                  &adesc.data, &attr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a logsoftmax forward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a logsoftmax forward
        ///     propagation primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : zendnn::primitive_desc(pd,
                                     // Logsoftmax and softmax share the implementation and
                                     // currently report the same primitive kind. Hence this
                                     // must be softmax and not logsoftmax.
                                     zendnn::primitive::kind::softmax,
                                     zendnn::prop_kind::forward_training,
                                     zendnn::prop_kind::forward_inference) {}

        /// @copydoc zendnn::primitive_desc_base::src_desc()const
        memory::desc src_desc() const {
            return base::src_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const {
            return base::dst_desc(0);
        }
    };

    /// Default constructor. Produces an empty object.
    logsoftmax_forward() = default;

    /// Constructs a logsoftmax forward propagation primitive.
    /// @param pd Primitive descriptor for a logsoftmax forward propagation
    ///     primitive.
    logsoftmax_forward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a logsoftmax forward propagation primitive from a cache
    ///     blob.
    /// @param pd Primitive descriptor for a logsoftmax forward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    logsoftmax_forward(
        const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// Logsoftmax backward propagation primitive.
struct logsoftmax_backward : public primitive {
    /// Descriptor for a logsoftmax backward propagation primitive.
    struct desc {
        zendnn_logsoftmax_desc_t data;

        /// Default constructor. Produces an empty object.
        desc() = default;

        /// Constructs a descriptor for a logsoftmax backward propagation
        /// primitive.
        ///
        /// @param diff_data_desc Diff source and diff destination memory
        ///     descriptors.
        /// @param data_desc Destination memory descriptor.
        /// @param logsoftmax_axis Axis over which softmax is computed.
        desc(const memory::desc &diff_data_desc, const memory::desc &data_desc,
             int logsoftmax_axis) {
            error::wrap_c_api(zendnn_logsoftmax_backward_desc_init(&data,
                              &diff_data_desc.data, &data_desc.data,
                              logsoftmax_axis),
                              "could not create a descriptor for a logsoftmax backward "
                              "propagation primitive");
        }
    };

    /// Primitive descriptor for a logsoftmax backward propagation primitive.
    struct primitive_desc : public zendnn::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a logsoftmax backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a logsoftmax backward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a logsoftmax forward
        ///     propagation primitive. It is used as a hint for deciding which
        ///     memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                       const logsoftmax_forward::primitive_desc &hint_fwd_pd,
                       bool allow_empty = false)
            : zendnn::primitive_desc(&adesc.data, nullptr, aengine,
                                     hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a logsoftmax backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a logsoftmax backward propagation
        ///     primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a logsoftmax forward
        ///     propagation primitive. It is used as a hint for deciding which
        ///     memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                       const engine &aengine,
                       const logsoftmax_forward::primitive_desc &hint_fwd_pd,
                       bool allow_empty = false)
            : zendnn::primitive_desc(&adesc.data, &attr, aengine,
                                     hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a logsoftmax backward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a logsoftmax backward
        ///     propagation primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : zendnn::primitive_desc(pd,
                                     // Logsoftmax and softmax share the implementation and
                                     // currently report the same primitive kind. Hence this
                                     // must be softmax and not logsoftmax.
                                     zendnn::primitive::kind::softmax,
                                     zendnn::prop_kind::backward_data) {}

        /// @copydoc zendnn::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const {
            return base::dst_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::diff_src_desc()const
        memory::desc diff_src_desc() const {
            return base::diff_src_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::dst_desc()const
        memory::desc diff_dst_desc() const {
            return base::diff_dst_desc(0);
        }
    };

    /// Default constructor. Produces an empty object.
    logsoftmax_backward() = default;

    /// Constructs a logsoftmax backward propagation primitive.
    /// @param pd Primitive descriptor for a logsoftmax backward propagation
    ///     primitive.
    logsoftmax_backward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a logsoftmax backward propagation primitive from a cache
    ///     blob.
    /// @param pd Primitive descriptor for a logsoftmax backward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    logsoftmax_backward(
        const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// @} zendnn_api_logsoftmax

/// @addtogroup zendnn_api_batch_normalization Batch Normalization
///
/// A primitive to perform batch normalization.
///
/// Both forward and backward propagation primitives support in-place
/// operation; that is, src and dst can refer to the same memory for forward
/// propagation, and diff_dst and diff_src can refer to the same memory for
/// backward propagation.
///
/// The batch normalization primitives computations can be controlled by
/// specifying different @ref zendnn::normalization_flags values. For example,
/// batch normalization forward propagation can be configured to either
/// compute the mean and variance or take them as arguments. It can either
/// perform scaling and shifting using gamma and beta parameters or not.
/// Optionally, it can also perform a fused ReLU, which in case of training
/// would also require a workspace.
///
/// @sa @ref dev_guide_batch_normalization in developer guide
///
/// @{

/// Batch normalization forward propagation primitive.
struct batch_normalization_forward : public primitive {
    /// Descriptor for a batch normalization forward propagation primitive.
    struct desc {
        zendnn_batch_normalization_desc_t data;

        /// Constructs a batch normalization descriptor for forward
        /// propagation.
        ///
        /// @note
        ///     In-place operation is supported: the dst can refer to the same
        ///     memory as the src.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #zendnn::prop_kind::forward_training and
        ///     #zendnn::prop_kind::forward_inference.
        /// @param data_desc Source and destination memory descriptors.
        /// @param epsilon Batch normalization epsilon parameter.
        /// @param flags Batch normalization flags (@ref
        ///     zendnn::normalization_flags).
        desc(prop_kind aprop_kind, const memory::desc &data_desc, float epsilon,
             normalization_flags flags) {
            error::wrap_c_api(
                zendnn_batch_normalization_forward_desc_init(&data,
                        zendnn::convert_to_c(aprop_kind), &data_desc.data,
                        epsilon, convert_to_c(flags)),
                "could not create a descriptor for a batch normalization "
                "forward propagation primitive");
        }
    };

    /// Primitive descriptor for a batch normalization forward propagation
    /// primitive.
    struct primitive_desc : public zendnn::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a batch normalization forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a batch normalization forward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                       bool allow_empty = false)
            : zendnn::primitive_desc(
                  &adesc.data, nullptr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a batch normalization forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a batch normalization forward propagation
        ///     primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                       const engine &aengine, bool allow_empty = false)
            : zendnn::primitive_desc(
                  &adesc.data, &attr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a batch normalization
        /// forward propagation primitive from a C API primitive descriptor
        /// that must have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a batch normalization
        ///     forward propagation primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : zendnn::primitive_desc(pd,
                                     zendnn::primitive::kind::batch_normalization,
                                     zendnn::prop_kind::forward_training,
                                     zendnn::prop_kind::forward_inference) {}

        /// @copydoc zendnn::primitive_desc_base::src_desc()const
        memory::desc src_desc() const {
            return base::src_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const {
            return base::dst_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::weights_desc()const
        memory::desc weights_desc() const {
            return base::weights_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const {
            return base::workspace_desc();
        }

        /// Returns memory descriptor for mean.
        /// @returns Memory descriptor for mean.
        memory::desc mean_desc() const {
            return stat_desc(mean);
        }

        /// Returns memory descriptor for variance.
        /// @returns Memory descriptor for variance.
        memory::desc variance_desc() const {
            return stat_desc(var);
        }

      private:
        enum {
            mean = 1,
            var = 2,
        };
        memory::desc stat_desc(int kind) const {
            zendnn_batch_normalization_desc_t *p;
            error::wrap_c_api(
                zendnn_primitive_desc_query(get(),
                                            zendnn::convert_to_c(query::batch_normalization_d), 0,
                                            &p),
                "could not retrieve a descriptor from a primitive "
                "descriptor for batch normalization forward propagation "
                "primitive");
            return query_md(p->flags & zendnn_use_global_stats ? query::src_md
                            : query::dst_md,
                            kind);
        }
    };

    /// Default constructor. Produces an empty object.
    batch_normalization_forward() = default;

    /// Constructs a batch normalization forward propagation primitive.
    /// @param pd Primitive descriptor for a batch normalization forward
    ///     propagation primitive.
    batch_normalization_forward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a batch normalization forward propagation primitive from
    ///     a cache blob.
    /// @param pd Primitive descriptor for a batch normalization forward
    ///     propagation primitive.
    /// @param cache_blob Cache blob.
    batch_normalization_forward(
        const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// Batch normalization backward propagation primitive.
struct batch_normalization_backward : public primitive {
    /// Descriptor for a batch normalization backward propagation primitive.
    struct desc {
        zendnn_batch_normalization_desc_t data;

        /// Constructs a batch normalization descriptor for backward
        /// propagation.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #zendnn::prop_kind::backward_data and #zendnn::prop_kind::backward
        ///     (diffs for all parameters are computed in this case).
        /// @param diff_data_desc Diff source and diff destination memory
        ///     descriptor.
        /// @param data_desc Source memory descriptor.
        /// @param epsilon Batch normalization epsilon parameter.
        /// @param flags Batch normalization flags (@ref
        ///     zendnn::normalization_flags).
        desc(prop_kind aprop_kind, const memory::desc &diff_data_desc,
             const memory::desc &data_desc, float epsilon,
             normalization_flags flags) {
            error::wrap_c_api(zendnn_batch_normalization_backward_desc_init(&data,
                              zendnn::convert_to_c(aprop_kind),
                              &diff_data_desc.data, &data_desc.data,
                              epsilon, convert_to_c(flags)),
                              "could not create a descriptor for a batch normalization "
                              "backward propagation primitive");
        }
    };

    /// Primitive descriptor for a batch normalization backward propagation
    /// primitive.
    struct primitive_desc : public zendnn::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a batch normalization backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a batch normalization backward
        ///     propagation primitive.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a batch normalization
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                       const batch_normalization_forward::primitive_desc &hint_fwd_pd,
                       bool allow_empty = false)
            : zendnn::primitive_desc(&adesc.data, nullptr, aengine,
                                     hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a batch normalization backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a batch normalization backward
        ///     propagation primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a batch normalization
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                       const engine &aengine,
                       const batch_normalization_forward::primitive_desc &hint_fwd_pd,
                       bool allow_empty = false)
            : zendnn::primitive_desc(&adesc.data, &attr, aengine,
                                     hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a batch normalization
        /// backward propagation primitive from a C API primitive descriptor
        /// that must have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a batch normalization
        ///     backward propagation primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : zendnn::primitive_desc(pd,
                                     zendnn::primitive::kind::batch_normalization,
                                     zendnn::prop_kind::backward, zendnn::prop_kind::backward_data) {
        }

        /// @copydoc zendnn::primitive_desc_base::src_desc()const
        memory::desc src_desc() const {
            return base::src_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::weights_desc()const
        memory::desc weights_desc() const {
            return base::weights_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const {
            return base::dst_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::diff_src_desc()const
        memory::desc diff_src_desc() const {
            return base::diff_src_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::diff_dst_desc()const
        memory::desc diff_dst_desc() const {
            return base::diff_dst_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::diff_weights_desc()const
        memory::desc diff_weights_desc() const {
            return base::diff_weights_desc(0);
        }

        /// @copydoc zendnn::batch_normalization_forward::primitive_desc::mean_desc()const
        memory::desc mean_desc() const {
            return query_md(query::src_md, 1);
        }

        /// @copydoc zendnn::batch_normalization_forward::primitive_desc::variance_desc()const
        memory::desc variance_desc() const {
            return query_md(query::src_md, 2);
        }

        /// @copydoc zendnn::primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const {
            return base::workspace_desc();
        }
    };

    /// Default constructor. Produces an empty object.
    batch_normalization_backward() = default;

    /// Constructs a batch normalization backward propagation primitive.
    /// @param pd Primitive descriptor for a batch normalization backward
    ///     propagation primitive.
    batch_normalization_backward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a batch normalization backward propagation primitive from
    ///     a cache blob.
    /// @param pd Primitive descriptor for a batch normalization backward
    ///     propagation primitive.
    /// @param cache_blob Cache blob.
    batch_normalization_backward(
        const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// @} zendnn_api_batch_normalization

/// @addtogroup zendnn_api_layer_normalization Layer Normalization
///
/// A primitive to perform layer normalization. Normalization is performed
/// within the last logical dimension of data tensor.
///
/// Both forward and backward propagation primitives support in-place
/// operation; that is, src and dst can refer to the same memory for forward
/// propagation, and diff_dst and diff_src can refer to the same memory for
/// backward propagation.
///
/// The layer normalization primitives computations can be controlled by
/// specifying different @ref zendnn::normalization_flags values. For example,
/// layer normalization forward propagation can be configured to either
/// compute the mean and variance or take them as arguments. It can either
/// perform scaling and shifting using gamma and beta parameters or not.
///
/// @sa @ref dev_guide_layer_normalization in developer guide
///
/// @{

/// Layer normalization forward propagation primitive.
struct layer_normalization_forward : public primitive {
    /// Descriptor for a layer normalization forward propagation primitive.
    struct desc {
        zendnn_layer_normalization_desc_t data;

        /// Constructs a descriptor for layer normalization forward
        /// propagation primitive.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #zendnn::prop_kind::forward_training, and
        ///     #zendnn::prop_kind::forward_inference.
        /// @param data_desc Source and destination memory descriptor.
        /// @param stat_desc Statistics memory descriptors.
        /// @param epsilon Layer normalization epsilon parameter.
        /// @param flags Layer normalization flags (@ref
        ///     zendnn::normalization_flags).
        desc(prop_kind aprop_kind, const memory::desc &data_desc,
             const memory::desc &stat_desc, float epsilon,
             normalization_flags flags) {
            error::wrap_c_api(
                zendnn_layer_normalization_forward_desc_init(&data,
                        zendnn::convert_to_c(aprop_kind), &data_desc.data,
                        &stat_desc.data, epsilon, convert_to_c(flags)),
                "could not create a descriptor for a layer normalization "
                "forward propagation primitive");
        }

        /// Constructs a descriptor for layer normalization forward
        /// propagation primitive.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #zendnn::prop_kind::forward_training, and
        ///     #zendnn::prop_kind::forward_inference.
        /// @param data_desc Source and destination memory descriptor.
        /// @param epsilon Layer normalization epsilon parameter.
        /// @param flags Layer normalization flags (@ref
        ///     zendnn::normalization_flags).
        desc(prop_kind aprop_kind, const memory::desc &data_desc, float epsilon,
             normalization_flags flags) {
            error::wrap_c_api(
                zendnn_layer_normalization_forward_desc_init(&data,
                        zendnn::convert_to_c(aprop_kind), &data_desc.data,
                        nullptr, epsilon, convert_to_c(flags)),
                "could not create a descriptor for a layer normalization "
                "forward propagation primitive");
        }
    };

    /// Primitive descriptor for a layer normalization forward propagation
    /// primitive.
    struct primitive_desc : public zendnn::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a layer normalization forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a layer normalization forward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                       bool allow_empty = false)
            : zendnn::primitive_desc(
                  &adesc.data, nullptr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a layer normalization forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a layer normalization forward propagation
        ///     primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                       const engine &aengine, bool allow_empty = false)
            : zendnn::primitive_desc(
                  &adesc.data, &attr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a layer normalization
        /// forward propagation primitive from a C API primitive descriptor
        /// that must have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a layer normalization
        ///     forward propagation primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : zendnn::primitive_desc(pd,
                                     zendnn::primitive::kind::layer_normalization,
                                     zendnn::prop_kind::forward_training,
                                     zendnn::prop_kind::forward_inference) {}

        /// @copydoc zendnn::primitive_desc_base::src_desc()const
        memory::desc src_desc() const {
            return base::src_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const {
            return base::dst_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::weights_desc()const
        memory::desc weights_desc() const {
            return base::weights_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const {
            return base::workspace_desc();
        }

        /// @copydoc zendnn::batch_normalization_forward::primitive_desc::mean_desc()const
        memory::desc mean_desc() const {
            return stat_desc(mean);
        }

        /// @copydoc zendnn::batch_normalization_forward::primitive_desc::variance_desc()const
        memory::desc variance_desc() const {
            return stat_desc(var);
        }

      private:
        enum {
            mean = 1,
            var = 2,
        };
        memory::desc stat_desc(int kind) const {
            zendnn_layer_normalization_desc_t *p;
            error::wrap_c_api(
                zendnn_primitive_desc_query(get(),
                                            zendnn::convert_to_c(query::layer_normalization_d), 0,
                                            &p),
                "could not retrieve a descriptor from a primitive "
                "descriptor for layer normalization forward propagation "
                "primitive");
            return query_md(p->flags & zendnn_use_global_stats ? query::src_md
                            : query::dst_md,
                            kind);
        }
    };

    /// Default constructor. Produces an empty object.
    layer_normalization_forward() = default;

    /// Constructs a layer normalization forward propagation primitive.
    /// @param pd Primitive descriptor for a layer normalization forward
    ///     propagation primitive.
    layer_normalization_forward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a layer normalization forward propagation primitive from
    ///     a cache blob.
    /// @param pd Primitive descriptor for a layer normalization forward
    ///     propagation primitive.
    /// @param cache_blob Cache blob.
    layer_normalization_forward(
        const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// Layer normalization backward propagation primitive.
struct layer_normalization_backward : public primitive {
    /// Descriptor for a layer normalization backward propagation primitive.
    struct desc {
        zendnn_layer_normalization_desc_t data;

        /// Constructs a descriptor for layer normalization backward
        /// propagation primitive.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #zendnn::prop_kind::backward_data and #zendnn::prop_kind::backward
        ///     (diffs for all parameters are computed in this case).
        /// @param diff_data_desc Diff source and diff destination memory
        ///     descriptor.
        /// @param data_desc Source memory descriptor.
        /// @param stat_desc Statistics memory descriptors.
        /// @param epsilon Layer normalization epsilon parameter.
        /// @param flags Layer normalization flags (@ref
        ///     zendnn::normalization_flags).
        desc(prop_kind aprop_kind, const memory::desc &diff_data_desc,
             const memory::desc &data_desc, const memory::desc &stat_desc,
             float epsilon, normalization_flags flags) {
            error::wrap_c_api(
                zendnn_layer_normalization_backward_desc_init(&data,
                        zendnn::convert_to_c(aprop_kind),
                        &diff_data_desc.data, &data_desc.data,
                        &stat_desc.data, epsilon, convert_to_c(flags)),
                "could not create a descriptor for a batch normalization "
                "backward propagation primitive");
        }

        /// Constructs a descriptor for layer normalization backward
        /// propagation primitive.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #zendnn::prop_kind::backward_data and #zendnn::prop_kind::backward
        ///     (diffs for all parameters are computed in this case).
        /// @param diff_data_desc Diff source and diff destination memory
        ///     descriptor.
        /// @param data_desc Source memory descriptor.
        /// @param epsilon Layer normalization epsilon parameter.
        /// @param flags Layer normalization flags (@ref
        ///     zendnn::normalization_flags).
        desc(prop_kind aprop_kind, const memory::desc &diff_data_desc,
             const memory::desc &data_desc, float epsilon,
             normalization_flags flags) {
            error::wrap_c_api(zendnn_layer_normalization_backward_desc_init(&data,
                              zendnn::convert_to_c(aprop_kind),
                              &diff_data_desc.data, &data_desc.data,
                              nullptr, epsilon, convert_to_c(flags)),
                              "could not create a descriptor for a batch normalization "
                              "backward propagation primitive");
        }
    };

    /// Primitive descriptor for a layer normalization backward propagation
    /// primitive.
    struct primitive_desc : public zendnn::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a layer normalization backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a layer normalization backward
        ///     propagation primitive.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a layer normalization
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                       const layer_normalization_forward::primitive_desc &hint_fwd_pd,
                       bool allow_empty = false)
            : zendnn::primitive_desc(&adesc.data, nullptr, aengine,
                                     hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a layer normalization backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a layer normalization backward
        ///     propagation primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a layer normalization
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                       const engine &aengine,
                       const layer_normalization_forward::primitive_desc &hint_fwd_pd,
                       bool allow_empty = false)
            : zendnn::primitive_desc(&adesc.data, &attr, aengine,
                                     hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a layer normalization
        /// backward propagation primitive from a C API primitive descriptor
        /// that must have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a layer normalization
        ///     backward propagation primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : zendnn::primitive_desc(pd,
                                     zendnn::primitive::kind::layer_normalization,
                                     zendnn::prop_kind::backward, zendnn::prop_kind::backward_data) {
        }

        /// @copydoc zendnn::primitive_desc_base::src_desc()const
        memory::desc src_desc() const {
            return base::src_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::weights_desc()const
        memory::desc weights_desc() const {
            return base::weights_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const {
            return base::dst_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::diff_src_desc()const
        memory::desc diff_src_desc() const {
            return base::diff_src_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::diff_dst_desc()const
        memory::desc diff_dst_desc() const {
            return base::diff_dst_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::diff_weights_desc()const
        memory::desc diff_weights_desc() const {
            return base::diff_weights_desc(0);
        }

        /// @copydoc zendnn::batch_normalization_forward::primitive_desc::mean_desc()const
        memory::desc mean_desc() const {
            return query_md(query::src_md, 1);
        }

        /// @copydoc zendnn::batch_normalization_forward::primitive_desc::variance_desc()const
        memory::desc variance_desc() const {
            return query_md(query::src_md, 2);
        }

        /// @copydoc zendnn::primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const {
            return base::workspace_desc();
        }
    };

    /// Default constructor. Produces an empty object.
    layer_normalization_backward() = default;

    /// Constructs a layer normalization backward propagation primitive.
    /// @param pd Primitive descriptor for a layer normalization backward
    ///     propagation primitive.
    layer_normalization_backward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a layer normalization backward propagation primitive from
    ///     a cache blob.
    /// @param pd Primitive descriptor for a layer normalization backward
    ///     propagation primitive.
    /// @param cache_blob Cache blob.
    layer_normalization_backward(
        const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// @} zendnn_api_layer_normalization

/// @addtogroup zendnn_api_inner_product Inner Product
///
/// A primitive to compute an inner product.
///
/// @sa @ref dev_guide_inner_product in developer guide
///
/// @{

/// Inner product forward propagation primitive.
struct inner_product_forward : public primitive {
    /// Descriptor for an inner product forward propagation primitive.
    struct desc {
        zendnn_inner_product_desc_t data;

        /// Constructs a descriptor for an inner product forward propagation
        /// primitive with bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #zendnn::memory::format_tag::any value of @p format_tag.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #zendnn::prop_kind::forward_training, and
        ///     #zendnn::prop_kind::forward_inference.
        /// @param src_desc Memory descriptor for src.
        /// @param weights_desc Memory descriptor for weights.
        /// @param bias_desc Memory descriptor for bias.
        /// @param dst_desc Memory descriptor for dst.
        desc(prop_kind aprop_kind, const memory::desc &src_desc,
             const memory::desc &weights_desc, const memory::desc &bias_desc,
             const memory::desc &dst_desc) {
            error::wrap_c_api(zendnn_inner_product_forward_desc_init(&data,
                              zendnn::convert_to_c(aprop_kind),
                              &src_desc.data, &weights_desc.data,
                              &bias_desc.data, &dst_desc.data),
                              "could not create a descriptor for an inner product "
                              "forward propagation primitive");
        }

        /// Constructs a descriptor for an inner product forward propagation
        /// primitive without bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #zendnn::memory::format_tag::any value of @p format_tag.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #zendnn::prop_kind::forward_training, and
        ///     #zendnn::prop_kind::forward_inference.
        /// @param src_desc Memory descriptor for src.
        /// @param weights_desc Memory descriptor for weights.
        /// @param dst_desc Memory descriptor for dst.
        desc(prop_kind aprop_kind, const memory::desc &src_desc,
             const memory::desc &weights_desc,
             const memory::desc &dst_desc) {
            error::wrap_c_api(
                zendnn_inner_product_forward_desc_init(&data,
                        zendnn::convert_to_c(aprop_kind), &src_desc.data,
                        &weights_desc.data, nullptr, &dst_desc.data),
                "could not create a descriptor for an inner product "
                "forward propagation primitive");
        }
    };

    /// Primitive descriptor for an inner product forward propagation primitive.
    struct primitive_desc : public zendnn::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an inner product forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for an inner product forward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                       bool allow_empty = false)
            : zendnn::primitive_desc(
                  &adesc.data, nullptr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for an inner product forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for an inner product forward propagation
        ///     primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                       const engine &aengine, bool allow_empty = false)
            : zendnn::primitive_desc(
                  &adesc.data, &attr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for an inner product forward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for an inner product forward
        ///     propagation primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : zendnn::primitive_desc(pd, zendnn::primitive::kind::inner_product,
                                     zendnn::prop_kind::forward_training,
                                     zendnn::prop_kind::forward_inference) {}

        /// @copydoc zendnn::primitive_desc_base::src_desc()const
        memory::desc src_desc() const {
            return base::src_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::weights_desc()const
        memory::desc weights_desc() const {
            return base::weights_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const {
            return base::dst_desc(0);
        }

        /// @copydoc zendnn::convolution_forward::primitive_desc::bias_desc()const
        memory::desc bias_desc() const {
            return base::weights_desc(1);
        }
    };

    /// Default constructor. Produces an empty object.
    inner_product_forward() = default;

    /// Constructs an inner product forward propagation primitive.
    /// @param pd Primitive descriptor for an inner product forward
    ///     propagation primitive.
    inner_product_forward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs an inner product forward propagation primitive from
    ///     a cache blob.
    /// @param pd Primitive descriptor for an inner product forward
    ///     propagation primitive.
    /// @param cache_blob Cache blob.
    inner_product_forward(
        const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// Inner product backward propagation primitive.
struct inner_product_backward_data : public primitive {
    /// Descriptor for an inner product backward propagation primitive.
    struct desc {
        zendnn_inner_product_desc_t data;

        /// Constructs a descriptor for an inner product backward propagation
        /// primitive.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #zendnn::memory::format_tag::any value of @p format_tag.
        ///
        /// @param diff_src_desc Memory descriptor for diff src.
        /// @param weights_desc Memory descriptor for weights.
        /// @param diff_dst_desc Memory descriptor for diff dst.
        desc(const memory::desc &diff_src_desc,
             const memory::desc &weights_desc,
             const memory::desc &diff_dst_desc) {
            error::wrap_c_api(zendnn_inner_product_backward_data_desc_init(&data,
                              &diff_src_desc.data, &weights_desc.data,
                              &diff_dst_desc.data),
                              "could not create a descriptor for an inner product "
                              "backward propagation primitive");
        }
    };

    /// Primitive descriptor for an inner product backward propagation
    /// primitive.
    struct primitive_desc : public zendnn::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an inner product backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for an inner product backward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for an inner product
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                       const inner_product_forward::primitive_desc &hint_fwd_pd,
                       bool allow_empty = false)
            : zendnn::primitive_desc(&adesc.data, nullptr, aengine,
                                     hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for an inner product backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for an inner product backward propagation
        ///     primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for an inner product
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                       const engine &aengine,
                       const inner_product_forward::primitive_desc &hint_fwd_pd,
                       bool allow_empty = false)
            : zendnn::primitive_desc(&adesc.data, &attr, aengine,
                                     hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for an inner product backward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for an inner product backward
        ///     propagation primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : zendnn::primitive_desc(pd, zendnn::primitive::kind::inner_product,
                                     zendnn::prop_kind::backward_data) {}

        /// @copydoc zendnn::primitive_desc_base::diff_src_desc()const
        memory::desc diff_src_desc() const {
            return base::diff_src_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::weights_desc()const
        memory::desc weights_desc() const {
            return base::weights_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::diff_dst_desc()const
        memory::desc diff_dst_desc() const {
            return base::diff_dst_desc(0);
        }
    };

    /// Default constructor. Produces an empty object.
    inner_product_backward_data() = default;

    /// Constructs an inner product backward propagation primitive.
    /// @param pd Primitive descriptor for an inner product backward
    ///     propagation primitive.
    inner_product_backward_data(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs an inner product backward propagation primitive from
    /// a cache blob.
    /// @param pd Primitive descriptor for an inner product backward
    ///     propagation primitive.
    /// @param cache_blob Cache blob.
    inner_product_backward_data(
        const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// Inner product weights gradient primitive.
struct inner_product_backward_weights : public primitive {
    /// Descriptor for an inner product weights gradient primitive.
    struct desc {
        zendnn_inner_product_desc_t data;

        /// Constructs a descriptor for an inner product descriptor weights
        /// update primitive with bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #zendnn::memory::format_tag::any value of @p format_tag.
        ///
        /// @param src_desc Memory descriptor for src.
        /// @param diff_weights_desc Memory descriptor for diff weights.
        /// @param diff_bias_desc Memory descriptor for diff bias.
        /// @param diff_dst_desc Memory descriptor for diff dst.
        desc(const memory::desc &src_desc,
             const memory::desc &diff_weights_desc,
             const memory::desc &diff_bias_desc,
             const memory::desc &diff_dst_desc) {
            error::wrap_c_api(
                zendnn_inner_product_backward_weights_desc_init(&data,
                        &src_desc.data, &diff_weights_desc.data,
                        &diff_bias_desc.data, &diff_dst_desc.data),
                "could not create a descriptor for an inner product "
                "weights gradient primitive");
        }

        /// Constructs a descriptor for an inner product descriptor weights
        /// update primitive without bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #zendnn::memory::format_tag::any value of @p format_tag.
        ///
        /// @param src_desc Memory descriptor for src.
        /// @param diff_weights_desc Memory descriptor for diff weights.
        /// @param diff_dst_desc Memory descriptor for diff dst.
        desc(const memory::desc &src_desc,
             const memory::desc &diff_weights_desc,
             const memory::desc &diff_dst_desc) {
            error::wrap_c_api(
                zendnn_inner_product_backward_weights_desc_init(&data,
                        &src_desc.data, &diff_weights_desc.data, nullptr,
                        &diff_dst_desc.data),
                "could not create a descriptor for an inner product "
                "weights gradient primitive");
        }
    };

    /// Primitive descriptor for an inner product weights gradient primitive.
    struct primitive_desc : public zendnn::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an inner product weights
        /// update primitive.
        ///
        /// @param adesc Descriptor for an inner product weights gradient
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for an inner product
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                       const inner_product_forward::primitive_desc &hint_fwd_pd,
                       bool allow_empty = false)
            : zendnn::primitive_desc(&adesc.data, nullptr, aengine,
                                     hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for an inner product weights
        /// update primitive.
        ///
        /// @param adesc Descriptor for an inner product weights gradient
        ///     primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for an inner product
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                       const engine &aengine,
                       const inner_product_forward::primitive_desc &hint_fwd_pd,
                       bool allow_empty = false)
            : zendnn::primitive_desc(&adesc.data, &attr, aengine,
                                     hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for an inner product weights
        /// update primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for an inner product weights
        ///     gradient primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : zendnn::primitive_desc(pd, zendnn::primitive::kind::inner_product,
                                     zendnn::prop_kind::backward_weights) {}

        /// @copydoc zendnn::primitive_desc_base::src_desc()const
        memory::desc src_desc() const {
            return base::src_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::diff_weights_desc()const
        memory::desc diff_weights_desc() const {
            return base::diff_weights_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::diff_dst_desc()const
        memory::desc diff_dst_desc() const {
            return base::diff_dst_desc(0);
        }

        /// @copydoc zendnn::convolution_backward_weights::primitive_desc::diff_bias_desc()const
        memory::desc diff_bias_desc() const {
            return base::diff_weights_desc(1);
        }
    };

    /// Default constructor. Produces an empty object.
    inner_product_backward_weights() = default;

    /// Constructs an inner product weights gradient primitive.
    /// @param pd Primitive descriptor for an inner product weights gradient
    ///     primitive.
    inner_product_backward_weights(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs an inner product weights gradient primitive from a cache
    ///     blob.
    /// @param pd Primitive descriptor for an inner product weights gradient
    ///     primitive.
    /// @param cache_blob Cache blob.
    inner_product_backward_weights(
        const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// @} zendnn_api_inner_product

/// @addtogroup zendnn_api_rnn RNN
///
/// A primitive to compute recurrent neural network layers.
///
/// @sa @ref dev_guide_rnn in developer guide
///
/// @{

/// Base class for primitive descriptors for RNN primitives.
struct rnn_primitive_desc_base : public primitive_desc {
    using primitive_desc::primitive_desc;

    /// Default constructor. Produces an empty object.
    rnn_primitive_desc_base() = default;

    /// Constructs an RNN primitive descriptor base from a C API primitive
    /// descriptor while checking that it actually describes the expected
    /// primitive by comparing propagation and primitive kinds.
    ///
    /// @param pd C API primitive descriptor.
    /// @param aprop_kind Expected propagation kind.
    /// @param cell_kind Expected cell kind.
    rnn_primitive_desc_base(zendnn_primitive_desc_t pd,
                            zendnn::prop_kind aprop_kind, zendnn::algorithm cell_kind)
        : rnn_primitive_desc_base(pd, aprop_kind, aprop_kind, cell_kind) {}

    /// Returns source layer memory descriptor.
    /// @returns Source layer memory descriptor.
    memory::desc src_layer_desc() const {
        return base::query_md(query::exec_arg_md, ZENDNN_ARG_SRC_LAYER);
    }

    /// Returns AUGRU attention memory descriptor.
    /// @returns AUGRU attention memory descriptor.
    memory::desc augru_attention_desc() const {
        return base::query_md(query::exec_arg_md, ZENDNN_ARG_AUGRU_ATTENTION);
    }

    /// Returns source iteration memory descriptor.
    /// @returns Source iteration memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///          source iteration parameter.
    memory::desc src_iter_desc() const {
        return base::query_md(query::exec_arg_md, ZENDNN_ARG_SRC_ITER);
    }

    /// Returns source recurrent cell state memory descriptor.
    /// @returns Source recurrent cell state memory descriptor.
    memory::desc src_iter_c_desc() const {
        return base::query_md(query::exec_arg_md, ZENDNN_ARG_SRC_ITER_C);
    }

    /// Returns weights layer memory descriptor.
    /// @returns Weights layer memory descriptor.
    memory::desc weights_layer_desc() const {
        return base::query_md(query::exec_arg_md, ZENDNN_ARG_WEIGHTS_LAYER);
    }

    /// Returns weights iteration memory descriptor.
    /// @returns Weights iteration memory descriptor.
    memory::desc weights_iter_desc() const {
        return base::query_md(query::exec_arg_md, ZENDNN_ARG_WEIGHTS_ITER);
    }

    /// Returns weights peephole memory descriptor.
    /// @returns Weights peephole memory descriptor.
    memory::desc weights_peephole_desc() const {
        return base::query_md(query::exec_arg_md, ZENDNN_ARG_WEIGHTS_PEEPHOLE);
    }

    /// Returns weights projection memory descriptor.
    /// @returns Weights projection memory descriptor.
    memory::desc weights_projection_desc() const {
        return base::query_md(query::exec_arg_md, ZENDNN_ARG_WEIGHTS_PROJECTION);
    }

    /// Returns bias memory descriptor.
    /// @returns Bias memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///          bias parameter.
    memory::desc bias_desc() const {
        return base::query_md(query::exec_arg_md, ZENDNN_ARG_BIAS);
    }

    /// Returns destination layer memory descriptor.
    /// @returns Destination layer memory descriptor.
    memory::desc dst_layer_desc() const {
        return base::query_md(query::exec_arg_md, ZENDNN_ARG_DST_LAYER);
    }

    /// Returns destination iteration memory descriptor.
    /// @returns Destination iteration memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///          destination iteration parameter.
    memory::desc dst_iter_desc() const {
        return base::query_md(query::exec_arg_md, ZENDNN_ARG_DST_ITER);
    }

    /// Returns destination recurrent cell state memory descriptor.
    /// @returns Destination recurrent cell state memory descriptor.
    memory::desc dst_iter_c_desc() const {
        return base::query_md(query::exec_arg_md, ZENDNN_ARG_DST_ITER_C);
    }

    /// Returns diff source layer memory descriptor.
    /// @returns Diff source layer memory descriptor.
    memory::desc diff_src_layer_desc() const {
        return base::query_md(query::exec_arg_md, ZENDNN_ARG_DIFF_SRC_LAYER);
    }

    /// Returns diff AUGRU attention memory descriptor.
    /// @returns Diff AUGRU attention memory descriptor.
    memory::desc diff_augru_attention_desc() const {
        return base::query_md(
                   query::exec_arg_md, ZENDNN_ARG_DIFF_AUGRU_ATTENTION);
    }

    /// Returns diff source iteration memory descriptor.
    /// @returns Diff source iteration memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///          diff source iteration parameter.
    memory::desc diff_src_iter_desc() const {
        return base::query_md(query::exec_arg_md, ZENDNN_ARG_DIFF_SRC_ITER);
    }

    /// Returns diff source recurrent cell state memory descriptor.
    /// @returns Diff source recurrent cell state memory descriptor.
    memory::desc diff_src_iter_c_desc() const {
        return base::query_md(query::exec_arg_md, ZENDNN_ARG_DIFF_SRC_ITER_C);
    }

    /// Returns diff weights layer memory descriptor.
    /// @returns Diff weights layer memory descriptor.
    memory::desc diff_weights_layer_desc() const {
        return base::query_md(query::exec_arg_md, ZENDNN_ARG_DIFF_WEIGHTS_LAYER);
    }

    /// Returns diff weights iteration memory descriptor.
    /// @returns Diff weights iteration memory descriptor.
    memory::desc diff_weights_iter_desc() const {
        return base::query_md(query::exec_arg_md, ZENDNN_ARG_DIFF_WEIGHTS_ITER);
    }

    /// Returns diff weights peephole memory descriptor.
    /// @returns Diff weights peephole memory descriptor.
    memory::desc diff_weights_peephole_desc() const {
        return base::query_md(
                   query::exec_arg_md, ZENDNN_ARG_DIFF_WEIGHTS_PEEPHOLE);
    }

    /// Returns diff weights projection memory descriptor.
    /// @returns Diff weights projection memory descriptor.
    memory::desc diff_weights_projection_desc() const {
        return base::query_md(
                   query::exec_arg_md, ZENDNN_ARG_DIFF_WEIGHTS_PROJECTION);
    }

    /// Returns diff bias memory descriptor.
    /// @returns Diff bias memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///          diff bias parameter.
    memory::desc diff_bias_desc() const {
        return base::query_md(query::exec_arg_md, ZENDNN_ARG_DIFF_BIAS);
    }

    /// Returns diff destination layer memory descriptor.
    /// @returns Diff destination layer memory descriptor.
    memory::desc diff_dst_layer_desc() const {
        return base::query_md(query::exec_arg_md, ZENDNN_ARG_DIFF_DST_LAYER);
    }

    /// Returns diff destination iteration memory descriptor.
    /// @returns Diff destination iteration memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///          diff destination iteration parameter.
    memory::desc diff_dst_iter_desc() const {
        return base::query_md(query::exec_arg_md, ZENDNN_ARG_DIFF_DST_ITER);
    }

    /// Returns diff destination recurrent cell state memory descriptor.
    /// @returns Diff destination recurrent cell state memory descriptor.
    memory::desc diff_dst_iter_c_desc() const {
        return base::query_md(query::exec_arg_md, ZENDNN_ARG_DIFF_DST_ITER_C);
    }

  protected:
    using rnn_base = rnn_primitive_desc_base;

    // (Deliberately not using doxygen comments)
    //
    // Constructs an RNN primitive descriptor base from a C API primitive
    // descriptor while checking that it actually describes the expected
    // primitive by comparing propagation and primitive kinds. Caller can
    // pass two options propagation kinds. This is typically used to check
    // that propagation kind is inference or training forward propagation.
    //
    // @param pd C API primitive descriptor.
    // @param prop_kind1 Expected propagation kind.
    // @param prop_kind2 Expected propagation kind.
    // @param cell_kind Expected cell kind.
    rnn_primitive_desc_base(zendnn_primitive_desc_t pd,
                            zendnn::prop_kind prop_kind1, zendnn::prop_kind prop_kind2,
                            zendnn::algorithm cell_kind) {
        zendnn_rnn_desc_t *rnn_d;
        zendnn_status_t rc;
        rc = zendnn_primitive_desc_query(pd, zendnn_query_rnn_d, 0, &rnn_d);
        error::wrap_c_api(rc,
                          "could not retrieve a descriptor from a primitive descriptor "
                          "for an RNN primitive");

        zendnn_prop_kind_t c_prop_kind1 = convert_to_c(prop_kind1);
        zendnn_prop_kind_t c_prop_kind2 = convert_to_c(prop_kind2);
        zendnn_alg_kind_t c_cell_kind = convert_to_c(cell_kind);

        bool ok = rnn_d->primitive_kind == zendnn_rnn
                  && (rnn_d->prop_kind == c_prop_kind1
                      || rnn_d->prop_kind == c_prop_kind2)
                  && rnn_d->cell_kind == c_cell_kind;

        if (!ok)
            ZENDNN_THROW_ERROR(zendnn_invalid_arguments,
                               "mismatch between expected and provided descriptors for an "
                               "RNN primitive");

        reset_with_clone(pd);
    }
};

/// Vanilla RNN forward propagation primitive.
struct vanilla_rnn_forward : public primitive {
    /// Descriptor for a vanilla RNN forward propagation primitive.
    struct desc {
        zendnn_rnn_desc_t data;

        /// Constructs a descriptor for a vanilla RNN forward propagation
        /// primitive.
        ///
        /// The following arguments may point to a zero memory descriptor:
        /// - @p src_iter_desc,
        /// - @p bias_desc,
        /// - @p dst_iter_desc.
        ///
        /// This would then indicate that the RNN forward propagation primitive
        /// should not use them and should default to zero values instead.
        ///
        /// @note
        ///     All memory descriptors except @p src_iter_desc can be
        ///     initialized with an #zendnn::memory::format_tag::any value of @p
        ///     format_tag.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #zendnn::prop_kind::forward_training, and
        ///     #zendnn::prop_kind::forward_inference.
        /// @param activation Activation kind. Possible values are
        ///     #zendnn::algorithm::eltwise_relu,
        ///     #zendnn::algorithm::eltwise_tanh, or
        ///     #zendnn::algorithm::eltwise_logistic.
        /// @param direction RNN direction. See @ref zendnn::rnn_direction for
        ///     more info.
        /// @param src_layer_desc Memory descriptor for the input vector.
        /// @param src_iter_desc Memory descriptor for the input recurrent
        ///     hidden state vector.
        /// @param weights_layer_desc Memory descriptor for the weights
        ///     applied to the layer input.
        /// @param weights_iter_desc Memory descriptor for the weights applied
        ///     to the recurrent input.
        /// @param bias_desc Bias memory descriptor.
        /// @param dst_layer_desc Memory descriptor for the output vector.
        /// @param dst_iter_desc Memory descriptor for the output recurrent
        ///     hidden state vector.
        /// @param flags Unused.
        /// @param alpha Negative slope if activation is
        ///     #zendnn::algorithm::eltwise_relu.
        /// @param beta Unused.
        desc(prop_kind aprop_kind, algorithm activation,
             rnn_direction direction, const memory::desc &src_layer_desc,
             const memory::desc &src_iter_desc,
             const memory::desc &weights_layer_desc,
             const memory::desc &weights_iter_desc,
             const memory::desc &bias_desc,
             const memory::desc &dst_layer_desc,
             const memory::desc &dst_iter_desc,
             rnn_flags flags = rnn_flags::undef, float alpha = 0.0f,
             float beta = 0.0f) {
            error::wrap_c_api(
                zendnn_vanilla_rnn_forward_desc_init(&data,
                        zendnn::convert_to_c(aprop_kind),
                        zendnn::convert_to_c(activation),
                        zendnn::convert_to_c(direction), &src_layer_desc.data,
                        &src_iter_desc.data, &weights_layer_desc.data,
                        &weights_iter_desc.data, &bias_desc.data,
                        &dst_layer_desc.data, &dst_iter_desc.data,
                        zendnn::convert_to_c(flags), alpha, beta),
                "could not create a descriptor for a vanilla RNN forward "
                "propagation primitive");
        }
    };

    /// Primitive descriptor for a vanilla RNN forward propagation primitive.
    struct primitive_desc : public rnn_primitive_desc_base {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a vanilla RNN forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a vanilla RNN forward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                       bool allow_empty = false)
            : rnn_primitive_desc_base(
                  &adesc.data, nullptr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a vanilla RNN forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a vanilla RNN forward propagation
        ///     primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                       const engine &aengine, bool allow_empty = false)
            : rnn_primitive_desc_base(
                  &adesc.data, &attr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a vanilla RNN forward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a vanilla RNN forward
        ///     propagation primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : rnn_primitive_desc_base(pd, zendnn::prop_kind::forward_training,
                                      zendnn::prop_kind::forward_inference,
                                      zendnn::algorithm::vanilla_rnn) {}

        /// @copydoc zendnn::rnn_primitive_desc_base::src_layer_desc()const
        memory::desc src_layer_desc() const {
            return rnn_base::src_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::src_iter_desc()const
        memory::desc src_iter_desc() const {
            return rnn_base::src_iter_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::weights_layer_desc()const
        memory::desc weights_layer_desc() const {
            return rnn_base::weights_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::weights_iter_desc()const
        memory::desc weights_iter_desc() const {
            return rnn_base::weights_iter_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::bias_desc()const
        memory::desc bias_desc() const {
            return rnn_base::bias_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::dst_layer_desc()const
        memory::desc dst_layer_desc() const {
            return rnn_base::dst_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::dst_iter_desc()const
        memory::desc dst_iter_desc() const {
            return rnn_base::dst_iter_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const {
            return rnn_base::workspace_desc();
        }
    };

    /// Default constructor. Produces an empty object.
    vanilla_rnn_forward() = default;

    /// Constructs a vanilla RNN forward propagation primitive.
    /// @param pd Primitive descriptor for a vanilla RNN forward
    ///     propagation primitive.
    vanilla_rnn_forward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a vanilla RNN forward propagation primitive from
    ///     a cache blob.
    /// @param pd Primitive descriptor for a vanilla RNN forward
    ///     propagation primitive.
    /// @param cache_blob Cache blob.
    vanilla_rnn_forward(
        const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// Vanilla RNN backward propagation primitive.
struct vanilla_rnn_backward : public primitive {
    /// Descriptor for a vanilla RNN backward propagation primitive.
    struct desc {
        zendnn_rnn_desc_t data;

        /// Constructs a descriptor for a vanilla RNN backward propagation
        /// primitive.
        ///
        /// The following arguments may point to a zero memory descriptor:
        /// - @p src_iter_desc together with @p diff_src_iter_desc,
        /// - @p bias_desc together with @p diff_bias_desc,
        /// - @p dst_iter_desc together with @p diff_dst_iter_desc.
        ///
        /// This would then indicate that the RNN backward propagation
        /// primitive should not use the respective data and should use zero
        /// values instead.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #zendnn::memory::format_tag::any value of @p format_tag.
        ///
        /// @param aprop_kind Propagation kind. Must be
        ///     #zendnn::prop_kind::backward.
        /// @param activation Activation kind. Possible values are
        ///     #zendnn::algorithm::eltwise_relu,
        ///     #zendnn::algorithm::eltwise_tanh, or
        ///     #zendnn::algorithm::eltwise_logistic.
        /// @param direction RNN direction. See @ref zendnn::rnn_direction for
        ///     more info.
        /// @param src_layer_desc Memory descriptor for the input vector.
        /// @param src_iter_desc Memory descriptor for the input recurrent
        ///     hidden state vector.
        /// @param weights_layer_desc Memory descriptor for the weights
        ///     applied to the layer input.
        /// @param weights_iter_desc Memory descriptor for the weights applied
        ///     to the recurrent input.
        /// @param bias_desc Bias memory descriptor.
        /// @param dst_layer_desc Memory descriptor for the output vector.
        /// @param dst_iter_desc Memory descriptor for the output recurrent
        ///     hidden state vector.
        /// @param diff_src_layer_desc Memory descriptor for the diff of input
        ///     vector.
        /// @param diff_src_iter_desc Memory descriptor for the diff of input
        ///     recurrent hidden state vector.
        /// @param diff_weights_layer_desc Memory descriptor for the diff of
        ///     weights applied to the layer input.
        /// @param diff_weights_iter_desc Memory descriptor for the diff of
        ///     weights applied to the recurrent input.
        /// @param diff_bias_desc Diff bias memory descriptor.
        /// @param diff_dst_layer_desc Memory descriptor for the diff of
        ///     output vector.
        /// @param diff_dst_iter_desc Memory descriptor for the diff of output
        ///     recurrent hidden state vector.
        /// @param flags Unused.
        /// @param alpha Negative slope if activation is
        ///     #zendnn::algorithm::eltwise_relu.
        /// @param beta Unused.
        desc(prop_kind aprop_kind, algorithm activation,
             rnn_direction direction, const memory::desc &src_layer_desc,
             const memory::desc &src_iter_desc,
             const memory::desc &weights_layer_desc,
             const memory::desc &weights_iter_desc,
             const memory::desc &bias_desc,
             const memory::desc &dst_layer_desc,
             const memory::desc &dst_iter_desc,
             const memory::desc &diff_src_layer_desc,
             const memory::desc &diff_src_iter_desc,
             const memory::desc &diff_weights_layer_desc,
             const memory::desc &diff_weights_iter_desc,
             const memory::desc &diff_bias_desc,
             const memory::desc &diff_dst_layer_desc,
             const memory::desc &diff_dst_iter_desc,
             rnn_flags flags = rnn_flags::undef, float alpha = 0.0f,
             float beta = 0.0f) {
            error::wrap_c_api(
                zendnn_vanilla_rnn_backward_desc_init(&data,
                        zendnn::convert_to_c(aprop_kind),
                        zendnn::convert_to_c(activation),
                        zendnn::convert_to_c(direction), &src_layer_desc.data,
                        &src_iter_desc.data, &weights_layer_desc.data,
                        &weights_iter_desc.data, &bias_desc.data,
                        &dst_layer_desc.data, &dst_iter_desc.data,
                        &diff_src_layer_desc.data, &diff_src_iter_desc.data,
                        &diff_weights_layer_desc.data,
                        &diff_weights_iter_desc.data, &diff_bias_desc.data,
                        &diff_dst_layer_desc.data, &diff_dst_iter_desc.data,
                        zendnn::convert_to_c(flags), alpha, beta),
                "could not create a descriptor for a vanilla RNN backward "
                "propagation primitive");
        }
    };

    /// Primitive descriptor for an RNN backward propagation primitive.
    struct primitive_desc : public rnn_primitive_desc_base {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a vanilla RNN backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a vanilla RNN backward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a vanilla RNN
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                       const vanilla_rnn_forward::primitive_desc &hint_fwd_pd,
                       bool allow_empty = false)
            : rnn_primitive_desc_base(&adesc.data, nullptr, aengine,
                                      hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a vanilla RNN backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a vanilla RNN backward propagation
        ///     primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a vanilla RNN
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                       const engine &aengine,
                       const vanilla_rnn_forward::primitive_desc &hint_fwd_pd,
                       bool allow_empty = false)
            : rnn_primitive_desc_base(&adesc.data, &attr, aengine,
                                      hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a vanilla RNN backward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a vanilla RNN backward
        ///     propagation primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : rnn_primitive_desc_base(pd, zendnn::prop_kind::backward,
                                      zendnn::algorithm::vanilla_rnn) {}

        /// @copydoc zendnn::rnn_primitive_desc_base::src_layer_desc()const
        memory::desc src_layer_desc() const {
            return rnn_base::src_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::src_iter_desc()const
        memory::desc src_iter_desc() const {
            return rnn_base::src_iter_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::weights_layer_desc()const
        memory::desc weights_layer_desc() const {
            return rnn_base::weights_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::weights_iter_desc()const
        memory::desc weights_iter_desc() const {
            return rnn_base::weights_iter_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::bias_desc()const
        memory::desc bias_desc() const {
            return rnn_base::bias_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::dst_layer_desc()const
        memory::desc dst_layer_desc() const {
            return rnn_base::dst_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::dst_iter_desc()const
        memory::desc dst_iter_desc() const {
            return rnn_base::dst_iter_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const {
            return rnn_base::workspace_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::diff_src_layer_desc()const
        memory::desc diff_src_layer_desc() const {
            return rnn_base::diff_src_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::diff_src_iter_desc()const
        memory::desc diff_src_iter_desc() const {
            return rnn_base::diff_src_iter_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::diff_weights_layer_desc()const
        memory::desc diff_weights_layer_desc() const {
            return rnn_base::diff_weights_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::diff_weights_iter_desc()const
        memory::desc diff_weights_iter_desc() const {
            return rnn_base::diff_weights_iter_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::diff_bias_desc()const
        memory::desc diff_bias_desc() const {
            return rnn_base::diff_bias_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::diff_dst_layer_desc()const
        memory::desc diff_dst_layer_desc() const {
            return rnn_base::diff_dst_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::diff_dst_iter_desc()const
        memory::desc diff_dst_iter_desc() const {
            return rnn_base::diff_dst_iter_desc();
        }
    };

    /// Default constructor. Produces an empty object.
    vanilla_rnn_backward() = default;

    /// Constructs a vanilla RNN backward propagation primitive.
    /// @param pd Primitive descriptor for a vanilla RNN backward
    ///     propagation primitive.
    vanilla_rnn_backward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a vanilla RNN backward propagation primitive from
    ///     a cache blob.
    /// @param pd Primitive descriptor for a vanilla RNN backward
    ///     propagation primitive.
    /// @param cache_blob Cache blob.
    vanilla_rnn_backward(
        const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// LSTM forward propagation primitive.
struct lstm_forward : public primitive {
    /// Descriptor for an LSTM forward propagation primitive.
    struct desc {
        zendnn_rnn_desc_t data;

        /// Constructs a descriptor for an LSTM (with or without peephole and
        /// with or without projection) forward propagation primitive.
        ///
        /// The following arguments may point to a zero memory descriptor:
        /// - @p src_iter_desc together with @p src_iter_c_desc,
        /// - @p weights_peephole_desc,
        /// - @p bias_desc,
        /// - @p dst_iter_desc together with @p dst_iter_c_desc.
        ///
        /// This would then indicate that the LSTM forward propagation
        /// primitive should not use them and should default to zero values
        /// instead.
        ///
        /// The @p weights_projection_desc may point to a zero memory
        /// descriptor. This would then indicate that the LSTM doesn't have
        /// recurrent projection layer.
        ///
        /// @note
        ///     All memory descriptors can be initialized with an
        ///     #zendnn::memory::format_tag::any value of @p format_tag.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #zendnn::prop_kind::forward_training, and
        ///     #zendnn::prop_kind::forward_inference.
        /// @param direction RNN direction. See @ref zendnn::rnn_direction for
        ///     more info.
        /// @param src_layer_desc Memory descriptor for the input vector.
        /// @param src_iter_desc Memory descriptor for the input recurrent
        ///     hidden state vector.
        /// @param src_iter_c_desc Memory descriptor for the input recurrent
        ///     cell state vector.
        /// @param weights_layer_desc Memory descriptor for the weights
        ///     applied to the layer input.
        /// @param weights_iter_desc Memory descriptor for the weights applied
        ///     to the recurrent input.
        /// @param weights_peephole_desc Memory descriptor for the weights
        ///     applied to the cell states (according to the Peephole LSTM
        ///     formula).
        /// @param weights_projection_desc Memory descriptor for the weights
        ///     applied to the hidden states to get the recurrent projection
        ///     (according to the Projection LSTM formula).
        /// @param bias_desc Bias memory descriptor.
        /// @param dst_layer_desc Memory descriptor for the output vector.
        /// @param dst_iter_desc Memory descriptor for the output recurrent
        ///     hidden state vector.
        /// @param dst_iter_c_desc Memory descriptor for the output recurrent
        ///     cell state vector.
        /// @param flags Unused.
        desc(prop_kind aprop_kind, rnn_direction direction,
             const memory::desc &src_layer_desc,
             const memory::desc &src_iter_desc,
             const memory::desc &src_iter_c_desc,
             const memory::desc &weights_layer_desc,
             const memory::desc &weights_iter_desc,
             const memory::desc &weights_peephole_desc,
             const memory::desc &weights_projection_desc,
             const memory::desc &bias_desc,
             const memory::desc &dst_layer_desc,
             const memory::desc &dst_iter_desc,
             const memory::desc &dst_iter_c_desc,
             rnn_flags flags = rnn_flags::undef) {
            error::wrap_c_api(
                zendnn_lstm_forward_desc_init_v3(&data,
                                                 zendnn::convert_to_c(aprop_kind),
                                                 zendnn::convert_to_c(direction), &src_layer_desc.data,
                                                 &src_iter_desc.data, &src_iter_c_desc.data,
                                                 &weights_layer_desc.data, &weights_iter_desc.data,
                                                 &weights_peephole_desc.data,
                                                 &weights_projection_desc.data, &bias_desc.data,
                                                 &dst_layer_desc.data, &dst_iter_desc.data,
                                                 &dst_iter_c_desc.data, zendnn::convert_to_c(flags)),
                "could not create a descriptor for an LSTM forward "
                "propagation primitive");
        }

        /// Constructs a descriptor for an LSTM (with or without peephole)
        /// forward propagation primitive.
        ///
        /// The following arguments may point to a zero memory descriptor:
        /// - @p src_iter_desc together with @p src_iter_c_desc,
        /// - @p weights_peephole_desc,
        /// - @p bias_desc,
        /// - @p dst_iter_desc together with @p dst_iter_c_desc.
        ///
        /// This would then indicate that the LSTM forward propagation
        /// primitive should not use them and should default to zero values
        /// instead.
        ///
        /// @note
        ///     All memory descriptors can be initialized with an
        ///     #zendnn::memory::format_tag::any value of @p format_tag.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #zendnn::prop_kind::forward_training, and
        ///     #zendnn::prop_kind::forward_inference.
        /// @param direction RNN direction. See @ref zendnn::rnn_direction for
        ///     more info.
        /// @param src_layer_desc Memory descriptor for the input vector.
        /// @param src_iter_desc Memory descriptor for the input recurrent
        ///     hidden state vector.
        /// @param src_iter_c_desc Memory descriptor for the input recurrent
        ///     cell state vector.
        /// @param weights_layer_desc Memory descriptor for the weights
        ///     applied to the layer input.
        /// @param weights_iter_desc Memory descriptor for the weights applied
        ///     to the recurrent input.
        /// @param weights_peephole_desc Memory descriptor for the weights
        ///     applied to the cell states (according to the Peephole LSTM
        ///     formula).
        /// @param bias_desc Bias memory descriptor.
        /// @param dst_layer_desc Memory descriptor for the output vector.
        /// @param dst_iter_desc Memory descriptor for the output recurrent
        ///     hidden state vector.
        /// @param dst_iter_c_desc Memory descriptor for the output recurrent
        ///     cell state vector.
        /// @param flags Unused.
        desc(prop_kind aprop_kind, rnn_direction direction,
             const memory::desc &src_layer_desc,
             const memory::desc &src_iter_desc,
             const memory::desc &src_iter_c_desc,
             const memory::desc &weights_layer_desc,
             const memory::desc &weights_iter_desc,
             const memory::desc &weights_peephole_desc,
             const memory::desc &bias_desc,
             const memory::desc &dst_layer_desc,
             const memory::desc &dst_iter_desc,
             const memory::desc &dst_iter_c_desc,
             rnn_flags flags = rnn_flags::undef) {
            error::wrap_c_api(
                zendnn_lstm_forward_desc_init_v2(&data,
                                                 zendnn::convert_to_c(aprop_kind),
                                                 zendnn::convert_to_c(direction), &src_layer_desc.data,
                                                 &src_iter_desc.data, &src_iter_c_desc.data,
                                                 &weights_layer_desc.data, &weights_iter_desc.data,
                                                 &weights_peephole_desc.data, &bias_desc.data,
                                                 &dst_layer_desc.data, &dst_iter_desc.data,
                                                 &dst_iter_c_desc.data, zendnn::convert_to_c(flags)),
                "could not create a descriptor for an LSTM forward "
                "propagation primitive");
        }

        /// Constructs a descriptor for an LSTM forward propagation primitive.
        ///
        /// The following arguments may point to a zero memory descriptor:
        /// - @p src_iter_desc together with @p src_iter_c_desc,
        /// - @p bias_desc,
        /// - @p dst_iter_desc together with @p dst_iter_c_desc.
        ///
        /// This would then indicate that the LSTM forward propagation
        /// primitive should not use them and should default to zero values
        /// instead.
        ///
        /// @note
        ///     All memory descriptors can be initialized with an
        ///     #zendnn::memory::format_tag::any value of @p format_tag.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #zendnn::prop_kind::forward_training, and
        ///     #zendnn::prop_kind::forward_inference.
        /// @param direction RNN direction. See @ref zendnn::rnn_direction for
        ///     more info.
        /// @param src_layer_desc Memory descriptor for the input vector.
        /// @param src_iter_desc Memory descriptor for the input recurrent
        ///     hidden state vector.
        /// @param src_iter_c_desc Memory descriptor for the input recurrent
        ///     cell state vector.
        /// @param weights_layer_desc Memory descriptor for the weights
        ///     applied to the layer input.
        /// @param weights_iter_desc Memory descriptor for the weights applied
        ///     to the recurrent input.
        /// @param bias_desc Bias memory descriptor.
        /// @param dst_layer_desc Memory descriptor for the output vector.
        /// @param dst_iter_desc Memory descriptor for the output recurrent
        ///     hidden state vector.
        /// @param dst_iter_c_desc Memory descriptor for the output recurrent
        ///     cell state vector.
        /// @param flags Unused.
        desc(prop_kind aprop_kind, rnn_direction direction,
             const memory::desc &src_layer_desc,
             const memory::desc &src_iter_desc,
             const memory::desc &src_iter_c_desc,
             const memory::desc &weights_layer_desc,
             const memory::desc &weights_iter_desc,
             const memory::desc &bias_desc,
             const memory::desc &dst_layer_desc,
             const memory::desc &dst_iter_desc,
             const memory::desc &dst_iter_c_desc,
             rnn_flags flags = rnn_flags::undef) {
            error::wrap_c_api(
                zendnn_lstm_forward_desc_init(&data,
                                              zendnn::convert_to_c(aprop_kind),
                                              zendnn::convert_to_c(direction), &src_layer_desc.data,
                                              &src_iter_desc.data, &src_iter_c_desc.data,
                                              &weights_layer_desc.data, &weights_iter_desc.data,
                                              &bias_desc.data, &dst_layer_desc.data,
                                              &dst_iter_desc.data, &dst_iter_c_desc.data,
                                              zendnn::convert_to_c(flags)),
                "could not create a descriptor for an LSTM forward "
                "propagation primitive");
        }
    };

    /// Primitive descriptor for an LSTM forward propagation primitive.
    struct primitive_desc : public rnn_primitive_desc_base {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an LSTM forward propagation
        /// primitive.
        ///
        /// @param adesc Descriptor for an LSTM forward propagation primitive.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                       bool allow_empty = false)
            : rnn_primitive_desc_base(
                  &adesc.data, nullptr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for an LSTM forward propagation
        /// primitive.
        ///
        /// @param adesc Descriptor for an LSTM forward propagation primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                       const engine &aengine, bool allow_empty = false)
            : rnn_primitive_desc_base(
                  &adesc.data, &attr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for an LSTM forward propagation
        /// primitive from a C API primitive descriptor that must have a
        /// matching kind.
        ///
        /// @param pd C API primitive descriptor for an LSTM forward
        ///     propagation primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : rnn_primitive_desc_base(pd, zendnn::prop_kind::forward_training,
                                      zendnn::prop_kind::forward_inference,
                                      zendnn::algorithm::vanilla_lstm) {}

        /// @copydoc zendnn::rnn_primitive_desc_base::src_layer_desc()const
        memory::desc src_layer_desc() const {
            return rnn_base::src_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::src_iter_desc()const
        memory::desc src_iter_desc() const {
            return rnn_base::src_iter_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::src_iter_desc()const
        memory::desc src_iter_c_desc() const {
            return rnn_base::src_iter_c_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::weights_layer_desc()const
        memory::desc weights_layer_desc() const {
            return rnn_base::weights_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::weights_iter_desc()const
        memory::desc weights_iter_desc() const {
            return rnn_base::weights_iter_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::weights_peephole_desc()const
        memory::desc weights_peephole_desc() const {
            return rnn_base::weights_peephole_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::weights_projection_desc()const
        memory::desc weights_projection_desc() const {
            return rnn_base::weights_projection_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::bias_desc()const
        memory::desc bias_desc() const {
            return rnn_base::bias_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::dst_layer_desc()const
        memory::desc dst_layer_desc() const {
            return rnn_base::dst_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::dst_iter_desc()const
        memory::desc dst_iter_desc() const {
            return rnn_base::dst_iter_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::src_iter_desc()const
        memory::desc dst_iter_c_desc() const {
            return rnn_base::dst_iter_c_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const {
            return rnn_base::workspace_desc();
        }
    };

    /// Default constructor. Produces an empty object.
    lstm_forward() = default;

    /// Constructs an LSTM forward propagation primitive.
    /// @param pd Primitive descriptor for an LSTM forward propagation
    ///     primitive.
    lstm_forward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs an LSTM forward propagation primitive from a cache blob.
    /// @param pd Primitive descriptor for an LSTM forward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    lstm_forward(
        const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// LSTM backward propagation primitive.
struct lstm_backward : public primitive {
    /// Descriptor for an LSTM backward propagation primitive.
    struct desc {
        zendnn_rnn_desc_t data;

        /// Constructs an LSTM (with or without peephole and with or without
        /// projection) descriptor for backward propagation using @p prop_kind,
        /// @p direction, and memory descriptors.
        ///
        /// The following arguments may point to a zero memory descriptor:
        /// - @p src_iter_desc together with @p src_iter_c_desc,
        ///   @p diff_src_iter_desc, and @p diff_src_iter_c_desc,
        /// - @p weights_peephole_desc together with
        ///   @p diff_weights_peephole_desc
        /// - @p bias_desc together with @p diff_bias_desc,
        /// - @p dst_iter_desc together with @p dst_iter_c_desc,
        ///   @p diff_dst_iter_desc, and @p diff_dst_iter_c_desc.
        ///
        /// This would then indicate that the LSTM backward propagation
        /// primitive should not use them and should default to zero values
        /// instead.
        ///
        /// The @p weights_projection_desc together with @p
        /// diff_weights_projection_desc may point to a zero memory descriptor.
        /// This would then indicate that the LSTM doesn't have recurrent
        /// projection layer.
        ///
        /// @note
        ///     All memory descriptors can be initialized with
        ///     #zendnn::memory::format_tag::any value of @p format_tag.
        ///
        /// @param aprop_kind Propagation kind. Must be
        ///     #zendnn::prop_kind::backward.
        /// @param direction RNN direction. See @ref zendnn::rnn_direction for
        ///     more info.
        /// @param src_layer_desc Memory descriptor for the input vector.
        /// @param src_iter_desc Memory descriptor for the input recurrent
        ///     hidden state vector.
        /// @param src_iter_c_desc Memory descriptor for the input recurrent
        ///     cell state vector.
        /// @param weights_layer_desc Memory descriptor for the weights
        ///     applied to the layer input.
        /// @param weights_iter_desc Memory descriptor for the weights applied
        ///     to the recurrent input.
        /// @param weights_peephole_desc Memory descriptor for the weights
        ///     applied to the cell states (according to the Peephole LSTM
        ///     formula).
        /// @param weights_projection_desc Memory descriptor for the weights
        ///     applied to the hidden states to get the recurrent projection
        ///     (according to the Projection LSTM formula).
        /// @param bias_desc Bias memory descriptor.
        /// @param dst_layer_desc Memory descriptor for the output vector.
        /// @param dst_iter_desc Memory descriptor for the output recurrent
        ///     hidden state vector.
        /// @param dst_iter_c_desc Memory descriptor for the output recurrent
        ///     cell state vector.
        /// @param diff_src_layer_desc Memory descriptor for the diff of input
        ///     vector.
        /// @param diff_src_iter_desc Memory descriptor for the diff of input
        ///     recurrent hidden state vector.
        /// @param diff_src_iter_c_desc Memory descriptor for the diff of
        ///     input recurrent cell state vector.
        /// @param diff_weights_layer_desc Memory descriptor for the diff of
        ///     weights applied to the layer input.
        /// @param diff_weights_iter_desc Memory descriptor for the diff of
        ///     weights applied to the recurrent input.
        /// @param diff_weights_peephole_desc Memory descriptor for the diff of
        ///     weights applied to the cell states (according to the Peephole
        ///     LSTM formula).
        /// @param diff_weights_projection_desc Memory descriptor for the diff
        ///     of weights applied to the hidden states to get the recurrent
        ///     projection (according to the Projection LSTM formula).
        /// @param diff_bias_desc Diff bias memory descriptor.
        /// @param diff_dst_layer_desc Memory descriptor for the diff of
        ///     output vector.
        /// @param diff_dst_iter_desc Memory descriptor for the diff of output
        ///     recurrent hidden state vector.
        /// @param diff_dst_iter_c_desc Memory descriptor for the diff of
        ///     output recurrent cell state vector.
        /// @param flags Unused.
        desc(prop_kind aprop_kind, rnn_direction direction,
             const memory::desc &src_layer_desc,
             const memory::desc &src_iter_desc,
             const memory::desc &src_iter_c_desc,
             const memory::desc &weights_layer_desc,
             const memory::desc &weights_iter_desc,
             const memory::desc &weights_peephole_desc,
             const memory::desc &weights_projection_desc,
             const memory::desc &bias_desc,
             const memory::desc &dst_layer_desc,
             const memory::desc &dst_iter_desc,
             const memory::desc &dst_iter_c_desc,
             const memory::desc &diff_src_layer_desc,
             const memory::desc &diff_src_iter_desc,
             const memory::desc &diff_src_iter_c_desc,
             const memory::desc &diff_weights_layer_desc,
             const memory::desc &diff_weights_iter_desc,
             const memory::desc &diff_weights_peephole_desc,
             const memory::desc &diff_weights_projection_desc,
             const memory::desc &diff_bias_desc,
             const memory::desc &diff_dst_layer_desc,
             const memory::desc &diff_dst_iter_desc,
             const memory::desc &diff_dst_iter_c_desc,
             rnn_flags flags = rnn_flags::undef) {
            error::wrap_c_api(
                zendnn_lstm_backward_desc_init_v3(&data,
                                                  zendnn::convert_to_c(aprop_kind),
                                                  zendnn::convert_to_c(direction), &src_layer_desc.data,
                                                  &src_iter_desc.data, &src_iter_c_desc.data,
                                                  &weights_layer_desc.data, &weights_iter_desc.data,
                                                  &weights_peephole_desc.data,
                                                  &weights_projection_desc.data, &bias_desc.data,
                                                  &dst_layer_desc.data, &dst_iter_desc.data,
                                                  &dst_iter_c_desc.data, &diff_src_layer_desc.data,
                                                  &diff_src_iter_desc.data,
                                                  &diff_src_iter_c_desc.data,
                                                  &diff_weights_layer_desc.data,
                                                  &diff_weights_iter_desc.data,
                                                  &diff_weights_peephole_desc.data,
                                                  &diff_weights_projection_desc.data,
                                                  &diff_bias_desc.data, &diff_dst_layer_desc.data,
                                                  &diff_dst_iter_desc.data,
                                                  &diff_dst_iter_c_desc.data,
                                                  zendnn::convert_to_c(flags)),
                "could not create a descriptor for an LSTM backward "
                "propagation primitive");
        }

        /// Constructs an LSTM (with or without peephole) descriptor for
        /// backward propagation using @p prop_kind, @p direction, and memory
        /// descriptors.
        ///
        /// The following arguments may point to a zero memory descriptor:
        /// - @p src_iter_desc together with @p src_iter_c_desc,
        ///   @p diff_src_iter_desc, and @p diff_src_iter_c_desc,
        /// - @p weights_peephole_desc together with
        ///   @p diff_weights_peephole_desc
        /// - @p bias_desc together with @p diff_bias_desc,
        /// - @p dst_iter_desc together with @p dst_iter_c_desc,
        ///   @p diff_dst_iter_desc, and @p diff_dst_iter_c_desc.
        ///
        /// This would then indicate that the LSTM backward propagation
        /// primitive should not use them and should default to zero values
        /// instead.
        ///
        /// @note
        ///     All memory descriptors may be initialized with
        ///     #zendnn::memory::format_tag::any value of @p format_tag.
        ///
        /// @param aprop_kind Propagation kind. Must be
        ///     #zendnn::prop_kind::backward.
        /// @param direction RNN direction. See @ref zendnn::rnn_direction for
        ///     more info.
        /// @param src_layer_desc Memory descriptor for the input vector.
        /// @param src_iter_desc Memory descriptor for the input recurrent
        ///     hidden state vector.
        /// @param src_iter_c_desc Memory descriptor for the input recurrent
        ///     cell state vector.
        /// @param weights_layer_desc Memory descriptor for the weights
        ///     applied to the layer input.
        /// @param weights_iter_desc Memory descriptor for the weights applied
        ///     to the recurrent input.
        /// @param weights_peephole_desc Memory descriptor for the weights
        ///     applied to the cell states (according to the Peephole LSTM
        ///     formula).
        /// @param bias_desc Bias memory descriptor.
        /// @param dst_layer_desc Memory descriptor for the output vector.
        /// @param dst_iter_desc Memory descriptor for the output recurrent
        ///     hidden state vector.
        /// @param dst_iter_c_desc Memory descriptor for the output recurrent
        ///     cell state vector.
        /// @param diff_src_layer_desc Memory descriptor for the diff of input
        ///     vector.
        /// @param diff_src_iter_desc Memory descriptor for the diff of input
        ///     recurrent hidden state vector.
        /// @param diff_src_iter_c_desc Memory descriptor for the diff of
        ///     input recurrent cell state vector.
        /// @param diff_weights_layer_desc Memory descriptor for the diff of
        ///     weights applied to the layer input.
        /// @param diff_weights_iter_desc Memory descriptor for the diff of
        ///     weights applied to the recurrent input.
        /// @param diff_weights_peephole_desc Memory descriptor for the diff of
        ///     weights applied to the cell states (according to the Peephole
        ///     LSTM formula).
        /// @param diff_bias_desc Diff bias memory descriptor.
        /// @param diff_dst_layer_desc Memory descriptor for the diff of
        ///     output vector.
        /// @param diff_dst_iter_desc Memory descriptor for the diff of output
        ///     recurrent hidden state vector.
        /// @param diff_dst_iter_c_desc Memory descriptor for the diff of
        ///     output recurrent cell state vector.
        /// @param flags Unused.
        desc(prop_kind aprop_kind, rnn_direction direction,
             const memory::desc &src_layer_desc,
             const memory::desc &src_iter_desc,
             const memory::desc &src_iter_c_desc,
             const memory::desc &weights_layer_desc,
             const memory::desc &weights_iter_desc,
             const memory::desc &weights_peephole_desc,
             const memory::desc &bias_desc,
             const memory::desc &dst_layer_desc,
             const memory::desc &dst_iter_desc,
             const memory::desc &dst_iter_c_desc,
             const memory::desc &diff_src_layer_desc,
             const memory::desc &diff_src_iter_desc,
             const memory::desc &diff_src_iter_c_desc,
             const memory::desc &diff_weights_layer_desc,
             const memory::desc &diff_weights_iter_desc,
             const memory::desc &diff_weights_peephole_desc,
             const memory::desc &diff_bias_desc,
             const memory::desc &diff_dst_layer_desc,
             const memory::desc &diff_dst_iter_desc,
             const memory::desc &diff_dst_iter_c_desc,
             rnn_flags flags = rnn_flags::undef) {
            error::wrap_c_api(
                zendnn_lstm_backward_desc_init_v2(&data,
                                                  zendnn::convert_to_c(aprop_kind),
                                                  zendnn::convert_to_c(direction), &src_layer_desc.data,
                                                  &src_iter_desc.data, &src_iter_c_desc.data,
                                                  &weights_layer_desc.data, &weights_iter_desc.data,
                                                  &weights_peephole_desc.data, &bias_desc.data,
                                                  &dst_layer_desc.data, &dst_iter_desc.data,
                                                  &dst_iter_c_desc.data, &diff_src_layer_desc.data,
                                                  &diff_src_iter_desc.data,
                                                  &diff_src_iter_c_desc.data,
                                                  &diff_weights_layer_desc.data,
                                                  &diff_weights_iter_desc.data,
                                                  &diff_weights_peephole_desc.data,
                                                  &diff_bias_desc.data, &diff_dst_layer_desc.data,
                                                  &diff_dst_iter_desc.data,
                                                  &diff_dst_iter_c_desc.data,
                                                  zendnn::convert_to_c(flags)),
                "could not create a descriptor for an LSTM backward "
                "propagation primitive");
        }

        /// Constructs an LSTM descriptor for backward propagation using @p
        /// prop_kind, @p direction, and memory descriptors.
        ///
        /// The following arguments may point to a zero memory descriptor:
        /// - @p src_iter_desc together with @p src_iter_c_desc,
        ///   @p diff_src_iter_desc, and @p diff_src_iter_c_desc,
        /// - @p bias_desc together with @p diff_bias_desc,
        /// - @p dst_iter_desc together with @p dst_iter_c_desc,
        ///   @p diff_dst_iter_desc, and @p diff_dst_iter_c_desc.
        ///
        /// This would then indicate that the LSTM backward propagation
        /// primitive should not use them and should default to zero values
        /// instead.
        ///
        /// @note
        ///     All memory descriptors may be initialized with
        ///     #zendnn::memory::format_tag::any value of @p format_tag.
        ///
        /// @param aprop_kind Propagation kind. Must be
        ///     #zendnn::prop_kind::backward.
        /// @param direction RNN direction. See @ref zendnn::rnn_direction for
        ///     more info.
        /// @param src_layer_desc Memory descriptor for the input vector.
        /// @param src_iter_desc Memory descriptor for the input recurrent
        ///     hidden state vector.
        /// @param src_iter_c_desc Memory descriptor for the input recurrent
        ///     cell state vector.
        /// @param weights_layer_desc Memory descriptor for the weights
        ///     applied to the layer input.
        /// @param weights_iter_desc Memory descriptor for the weights applied
        ///     to the recurrent input.
        /// @param bias_desc Bias memory descriptor.
        /// @param dst_layer_desc Memory descriptor for the output vector.
        /// @param dst_iter_desc Memory descriptor for the output recurrent
        ///     hidden state vector.
        /// @param dst_iter_c_desc Memory descriptor for the output recurrent
        ///     cell state vector.
        /// @param diff_src_layer_desc Memory descriptor for the diff of input
        ///     vector.
        /// @param diff_src_iter_desc Memory descriptor for the diff of input
        ///     recurrent hidden state vector.
        /// @param diff_src_iter_c_desc Memory descriptor for the diff of
        ///     input recurrent cell state vector.
        /// @param diff_weights_layer_desc Memory descriptor for the diff of
        ///     weights applied to the layer input.
        /// @param diff_weights_iter_desc Memory descriptor for the diff of
        ///     weights applied to the recurrent input.
        /// @param diff_bias_desc Diff bias memory descriptor.
        /// @param diff_dst_layer_desc Memory descriptor for the diff of
        ///     output vector.
        /// @param diff_dst_iter_desc Memory descriptor for the diff of output
        ///     recurrent hidden state vector.
        /// @param diff_dst_iter_c_desc Memory descriptor for the diff of
        ///     output recurrent cell state vector.
        /// @param flags Unused.
        desc(prop_kind aprop_kind, rnn_direction direction,
             const memory::desc &src_layer_desc,
             const memory::desc &src_iter_desc,
             const memory::desc &src_iter_c_desc,
             const memory::desc &weights_layer_desc,
             const memory::desc &weights_iter_desc,
             const memory::desc &bias_desc,
             const memory::desc &dst_layer_desc,
             const memory::desc &dst_iter_desc,
             const memory::desc &dst_iter_c_desc,
             const memory::desc &diff_src_layer_desc,
             const memory::desc &diff_src_iter_desc,
             const memory::desc &diff_src_iter_c_desc,
             const memory::desc &diff_weights_layer_desc,
             const memory::desc &diff_weights_iter_desc,
             const memory::desc &diff_bias_desc,
             const memory::desc &diff_dst_layer_desc,
             const memory::desc &diff_dst_iter_desc,
             const memory::desc &diff_dst_iter_c_desc,
             rnn_flags flags = rnn_flags::undef) {
            error::wrap_c_api(
                zendnn_lstm_backward_desc_init(&data,
                                               zendnn::convert_to_c(aprop_kind),
                                               zendnn::convert_to_c(direction), &src_layer_desc.data,
                                               &src_iter_desc.data, &src_iter_c_desc.data,
                                               &weights_layer_desc.data, &weights_iter_desc.data,
                                               &bias_desc.data, &dst_layer_desc.data,
                                               &dst_iter_desc.data, &dst_iter_c_desc.data,
                                               &diff_src_layer_desc.data, &diff_src_iter_desc.data,
                                               &diff_src_iter_c_desc.data,
                                               &diff_weights_layer_desc.data,
                                               &diff_weights_iter_desc.data, &diff_bias_desc.data,
                                               &diff_dst_layer_desc.data, &diff_dst_iter_desc.data,
                                               &diff_dst_iter_c_desc.data,
                                               zendnn::convert_to_c(flags)),
                "could not create a descriptor for an LSTM backward "
                "propagation primitive");
        }
    };

    /// Primitive descriptor for an LSTM backward propagation primitive.
    struct primitive_desc : public rnn_primitive_desc_base {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an LSTM backward propagation
        /// primitive.
        ///
        /// @param adesc Descriptor for LSTM backward propagation primitive.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for an LSTM
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                       const lstm_forward::primitive_desc &hint_fwd_pd,
                       bool allow_empty = false)
            : rnn_primitive_desc_base(&adesc.data, nullptr, aengine,
                                      hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for an LSTM backward propagation
        /// primitive.
        ///
        /// @param adesc Descriptor for an LSTM backward propagation primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for an LSTM
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                       const engine &aengine,
                       const lstm_forward::primitive_desc &hint_fwd_pd,
                       bool allow_empty = false)
            : rnn_primitive_desc_base(&adesc.data, &attr, aengine,
                                      hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for an LSTM backward propagation
        /// primitive from a C API primitive descriptor that must have a
        /// matching kind.
        ///
        /// @param pd C API primitive descriptor for an LSTM backward
        ///     propagation primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : rnn_primitive_desc_base(pd, zendnn::prop_kind::backward,
                                      zendnn::algorithm::vanilla_lstm) {}

        /// @copydoc zendnn::rnn_primitive_desc_base::src_layer_desc()const
        memory::desc src_layer_desc() const {
            return rnn_base::src_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::src_iter_desc()const
        memory::desc src_iter_desc() const {
            return rnn_base::src_iter_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::src_iter_desc()const
        memory::desc src_iter_c_desc() const {
            return rnn_base::src_iter_c_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::weights_layer_desc()const
        memory::desc weights_layer_desc() const {
            return rnn_base::weights_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::weights_iter_desc()const
        memory::desc weights_iter_desc() const {
            return rnn_base::weights_iter_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::weights_peephole_desc()const
        memory::desc weights_peephole_desc() const {
            return rnn_base::weights_peephole_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::weights_projection_desc()const
        memory::desc weights_projection_desc() const {
            return rnn_base::weights_projection_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::bias_desc()const
        memory::desc bias_desc() const {
            return rnn_base::bias_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::dst_layer_desc()const
        memory::desc dst_layer_desc() const {
            return rnn_base::dst_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::dst_iter_desc()const
        memory::desc dst_iter_desc() const {
            return rnn_base::dst_iter_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::src_iter_desc()const
        memory::desc dst_iter_c_desc() const {
            return rnn_base::dst_iter_c_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const {
            return rnn_base::workspace_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::diff_src_layer_desc()const
        memory::desc diff_src_layer_desc() const {
            return rnn_base::diff_src_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::diff_src_iter_desc()const
        memory::desc diff_src_iter_desc() const {
            return rnn_base::diff_src_iter_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::diff_src_iter_c_desc()const
        memory::desc diff_src_iter_c_desc() const {
            return rnn_base::diff_src_iter_c_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::diff_weights_layer_desc()const
        memory::desc diff_weights_layer_desc() const {
            return rnn_base::diff_weights_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::diff_weights_iter_desc()const
        memory::desc diff_weights_iter_desc() const {
            return rnn_base::diff_weights_iter_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::diff_weights_peephole_desc()const
        memory::desc diff_weights_peephole_desc() const {
            return rnn_base::diff_weights_peephole_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::diff_weights_projection_desc()const
        memory::desc diff_weights_projection_desc() const {
            return rnn_base::diff_weights_projection_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::diff_bias_desc()const
        memory::desc diff_bias_desc() const {
            return rnn_base::diff_bias_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::diff_dst_layer_desc()const
        memory::desc diff_dst_layer_desc() const {
            return rnn_base::diff_dst_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::diff_dst_iter_desc()const
        memory::desc diff_dst_iter_desc() const {
            return rnn_base::diff_dst_iter_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::diff_dst_iter_c_desc()const
        memory::desc diff_dst_iter_c_desc() const {
            return rnn_base::diff_dst_iter_c_desc();
        }
    };

    /// Default constructor. Produces an empty object.
    lstm_backward() = default;

    /// Constructs an LSTM backward propagation primitive.
    /// @param pd Primitive descriptor for an LSTM backward propagation
    ///     primitive.
    lstm_backward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs an LSTM backward propagation primitive from a cache blob.
    /// @param pd Primitive descriptor for an LSTM backward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    lstm_backward(
        const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// GRU forward propagation primitive.
struct gru_forward : public primitive {
    /// Descriptor for a GRU forward propagation primitive.
    struct desc {
        zendnn_rnn_desc_t data;

        /// Constructs a descriptor for a GRU forward propagation primitive.
        ///
        /// The following arguments may point to a zero memory descriptor:
        /// - @p src_iter_desc,
        /// - @p bias_desc,
        /// - @p dst_iter_desc.
        ///
        /// This would then indicate that the GRU forward propagation primitive
        /// should not use them and should default to zero values instead.
        ///
        /// @note
        ///     All memory descriptors except @p src_iter_desc may be
        ///     initialized with an #zendnn::memory::format_tag::any value of @p
        ///     format_tag.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #zendnn::prop_kind::forward_training, and
        ///     #zendnn::prop_kind::forward_inference.
        /// @param direction RNN direction. See @ref zendnn::rnn_direction for
        ///     more info.
        /// @param src_layer_desc Memory descriptor for the input vector.
        /// @param src_iter_desc Memory descriptor for the input recurrent
        ///     hidden state vector.
        /// @param weights_layer_desc Memory descriptor for the weights
        ///     applied to the layer input.
        /// @param weights_iter_desc Memory descriptor for the weights applied
        ///     to the recurrent input.
        /// @param bias_desc Bias memory descriptor.
        /// @param dst_layer_desc Memory descriptor for the output vector.
        /// @param dst_iter_desc Memory descriptor for the output recurrent
        ///     hidden state vector.
        /// @param flags Unused.
        desc(prop_kind aprop_kind, rnn_direction direction,
             const memory::desc &src_layer_desc,
             const memory::desc &src_iter_desc,
             const memory::desc &weights_layer_desc,
             const memory::desc &weights_iter_desc,
             const memory::desc &bias_desc,
             const memory::desc &dst_layer_desc,
             const memory::desc &dst_iter_desc,
             rnn_flags flags = rnn_flags::undef) {
            error::wrap_c_api(
                zendnn_gru_forward_desc_init(&data,
                                             zendnn::convert_to_c(aprop_kind),
                                             zendnn::convert_to_c(direction), &src_layer_desc.data,
                                             &src_iter_desc.data, &weights_layer_desc.data,
                                             &weights_iter_desc.data, &bias_desc.data,
                                             &dst_layer_desc.data, &dst_iter_desc.data,
                                             zendnn::convert_to_c(flags)),
                "could not create a descriptor for a GRU forward "
                "propagation primitive");
        }
    };

    /// Primitive descriptor for a GRU forward propagation primitive.
    struct primitive_desc : public rnn_primitive_desc_base {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a GRU forward propagation
        /// primitive.
        ///
        /// @param adesc Descriptor for a GRU forward propagation primitive.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                       bool allow_empty = false)
            : rnn_primitive_desc_base(
                  &adesc.data, nullptr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a GRU forward propagation
        /// primitive.
        ///
        /// @param adesc Descriptor for a GRU forward propagation primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                       const engine &aengine, bool allow_empty = false)
            : rnn_primitive_desc_base(
                  &adesc.data, &attr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a GRU forward propagation
        /// primitive from a C API primitive descriptor that must have a
        /// matching kind.
        ///
        /// @param pd C API primitive descriptor for a GRU forward
        ///     propagation primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : rnn_primitive_desc_base(pd, zendnn::prop_kind::forward_training,
                                      zendnn::prop_kind::forward_inference,
                                      zendnn::algorithm::vanilla_gru) {}

        /// @copydoc zendnn::rnn_primitive_desc_base::src_layer_desc()const
        memory::desc src_layer_desc() const {
            return rnn_base::src_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::src_iter_desc()const
        memory::desc src_iter_desc() const {
            return rnn_base::src_iter_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::weights_layer_desc()const
        memory::desc weights_layer_desc() const {
            return rnn_base::weights_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::weights_iter_desc()const
        memory::desc weights_iter_desc() const {
            return rnn_base::weights_iter_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::bias_desc()const
        memory::desc bias_desc() const {
            return rnn_base::bias_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::dst_layer_desc()const
        memory::desc dst_layer_desc() const {
            return rnn_base::dst_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::dst_iter_desc()const
        memory::desc dst_iter_desc() const {
            return rnn_base::dst_iter_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const {
            return rnn_base::workspace_desc();
        }
    };

    /// Default constructor. Produces an empty object.
    gru_forward() = default;

    /// Constructs a GRU forward propagation primitive.
    /// @param pd Primitive descriptor for a GRU forward propagation
    ///     primitive.
    gru_forward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a GRU forward propagation primitive from a cache blob.
    /// @param pd Primitive descriptor for a GRU forward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    gru_forward(
        const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// GRU backward propagation primitive.
struct gru_backward : public primitive {
    /// Descriptor for a GRU backward propagation primitive.
    struct desc {
        zendnn_rnn_desc_t data;

        /// Constructs a descriptor for a GRU backward propagation primitive.
        ///
        /// The following arguments may point to a zero memory descriptor:
        /// - @p src_iter_desc together with @p diff_src_iter_desc,
        /// - @p bias_desc together with @p diff_bias_desc,
        /// - @p dst_iter_desc together with @p diff_dst_iter_desc.
        ///
        /// This would then indicate that the GRU backward propagation
        /// primitive should not use them and should default to zero values
        /// instead.
        ///
        /// @note
        ///     All memory descriptors may be initialized with
        ///     #zendnn::memory::format_tag::any value of @p format_tag.
        ///
        /// @param aprop_kind Propagation kind. Must be
        ///     #zendnn::prop_kind::backward.
        /// @param direction RNN direction. See @ref zendnn::rnn_direction for
        ///     more info.
        /// @param src_layer_desc Memory descriptor for the input vector.
        /// @param src_iter_desc Memory descriptor for the input recurrent
        ///     hidden state vector.
        /// @param weights_layer_desc Memory descriptor for the weights
        ///     applied to the layer input.
        /// @param weights_iter_desc Memory descriptor for the weights applied
        ///     to the recurrent input.
        /// @param bias_desc Bias memory descriptor.
        /// @param dst_layer_desc Memory descriptor for the output vector.
        /// @param dst_iter_desc Memory descriptor for the output recurrent
        ///     hidden state vector.
        /// @param diff_src_layer_desc Memory descriptor for the diff of input
        ///     vector.
        /// @param diff_src_iter_desc Memory descriptor for the diff of input
        ///     recurrent hidden state vector.
        /// @param diff_weights_layer_desc Memory descriptor for the diff of
        ///     weights applied to the layer input.
        /// @param diff_weights_iter_desc Memory descriptor for the diff of
        ///     weights applied to the recurrent input.
        /// @param diff_bias_desc Diff bias memory descriptor.
        /// @param diff_dst_layer_desc Memory descriptor for the diff of
        ///     output vector.
        /// @param diff_dst_iter_desc Memory descriptor for the diff of output
        ///     recurrent hidden state vector.
        /// @param flags Unused.
        desc(prop_kind aprop_kind, rnn_direction direction,
             const memory::desc &src_layer_desc,
             const memory::desc &src_iter_desc,
             const memory::desc &weights_layer_desc,
             const memory::desc &weights_iter_desc,
             const memory::desc &bias_desc,
             const memory::desc &dst_layer_desc,
             const memory::desc &dst_iter_desc,
             const memory::desc &diff_src_layer_desc,
             const memory::desc &diff_src_iter_desc,
             const memory::desc &diff_weights_layer_desc,
             const memory::desc &diff_weights_iter_desc,
             const memory::desc &diff_bias_desc,
             const memory::desc &diff_dst_layer_desc,
             const memory::desc &diff_dst_iter_desc,
             rnn_flags flags = rnn_flags::undef) {
            error::wrap_c_api(
                zendnn_gru_backward_desc_init(&data,
                                              zendnn::convert_to_c(aprop_kind),
                                              zendnn::convert_to_c(direction), &src_layer_desc.data,
                                              &src_iter_desc.data, &weights_layer_desc.data,
                                              &weights_iter_desc.data, &bias_desc.data,
                                              &dst_layer_desc.data, &dst_iter_desc.data,
                                              &diff_src_layer_desc.data, &diff_src_iter_desc.data,
                                              &diff_weights_layer_desc.data,
                                              &diff_weights_iter_desc.data, &diff_bias_desc.data,
                                              &diff_dst_layer_desc.data, &diff_dst_iter_desc.data,
                                              zendnn::convert_to_c(flags)),
                "could not create a descriptor for a GRU backward "
                "propagation primitive");
        }
    };

    /// Primitive descriptor for a GRU backward propagation primitive.
    struct primitive_desc : public rnn_primitive_desc_base {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a GRU backward propagation
        /// primitive.
        ///
        /// @param adesc Descriptor for a GRU backward propagation primitive.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a GRU
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                       const gru_forward::primitive_desc &hint_fwd_pd,
                       bool allow_empty = false)
            : rnn_primitive_desc_base(&adesc.data, nullptr, aengine,
                                      hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a GRU backward propagation
        /// primitive.
        ///
        /// @param adesc Descriptor for a GRU backward propagation primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a GRU
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                       const engine &aengine,
                       const gru_forward::primitive_desc &hint_fwd_pd,
                       bool allow_empty = false)
            : rnn_primitive_desc_base(&adesc.data, &attr, aengine,
                                      hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a GRU backward propagation
        /// primitive from a C API primitive descriptor that must have a
        /// matching kind.
        ///
        /// @param pd C API primitive descriptor for a GRU backward
        ///     propagation primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : rnn_primitive_desc_base(pd, zendnn::prop_kind::backward,
                                      zendnn::algorithm::vanilla_gru) {}

        /// @copydoc zendnn::rnn_primitive_desc_base::src_layer_desc()const
        memory::desc src_layer_desc() const {
            return rnn_base::src_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::src_iter_desc()const
        memory::desc src_iter_desc() const {
            return rnn_base::src_iter_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::weights_layer_desc()const
        memory::desc weights_layer_desc() const {
            return rnn_base::weights_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::weights_iter_desc()const
        memory::desc weights_iter_desc() const {
            return rnn_base::weights_iter_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::bias_desc()const
        memory::desc bias_desc() const {
            return rnn_base::bias_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::dst_layer_desc()const
        memory::desc dst_layer_desc() const {
            return rnn_base::dst_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::dst_iter_desc()const
        memory::desc dst_iter_desc() const {
            return rnn_base::dst_iter_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const {
            return rnn_base::workspace_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::diff_src_layer_desc()const
        memory::desc diff_src_layer_desc() const {
            return rnn_base::diff_src_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::diff_src_iter_desc()const
        memory::desc diff_src_iter_desc() const {
            return rnn_base::diff_src_iter_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::diff_weights_layer_desc()const
        memory::desc diff_weights_layer_desc() const {
            return rnn_base::diff_weights_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::diff_weights_iter_desc()const
        memory::desc diff_weights_iter_desc() const {
            return rnn_base::diff_weights_iter_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::diff_bias_desc()const
        memory::desc diff_bias_desc() const {
            return rnn_base::diff_bias_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::diff_dst_layer_desc()const
        memory::desc diff_dst_layer_desc() const {
            return rnn_base::diff_dst_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::diff_dst_iter_desc()const
        memory::desc diff_dst_iter_desc() const {
            return rnn_base::diff_dst_iter_desc();
        }
    };

    /// Default constructor. Produces an empty object.
    gru_backward() = default;

    /// Constructs a GRU backward propagation primitive.
    /// @param pd Primitive descriptor for a GRU backward propagation
    ///     primitive.
    gru_backward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a GRU backward propagation primitive from a cache blob.
    /// @param pd Primitive descriptor for a GRU backward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    gru_backward(
        const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// LBR GRU forward propagation primitive.
struct lbr_gru_forward : public primitive {
    /// Descriptor for an LBR GRU forward propagation primitive.
    struct desc {
        zendnn_rnn_desc_t data;

        /// Constructs a descriptor for LBR GRU forward propagation primitive.
        ///
        /// The following arguments may point to a zero memory descriptor:
        /// - @p src_iter_desc,
        /// - @p bias_desc,
        /// - @p dst_iter_desc.
        ///
        /// This would then indicate that the LBR GRU forward propagation
        /// primitive should not use them and should default to zero values
        /// instead.
        ///
        /// @note
        ///     All memory descriptors except @p src_iter_desc may be
        ///     initialized with an #zendnn::memory::format_tag::any value of @p
        ///     format_tag.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #zendnn::prop_kind::forward_training, and
        ///     #zendnn::prop_kind::forward_inference.
        /// @param direction RNN direction. See @ref zendnn::rnn_direction for
        ///     more info.
        /// @param src_layer_desc Memory descriptor for the input vector.
        /// @param src_iter_desc Memory descriptor for the input recurrent
        ///     hidden state vector.
        /// @param weights_layer_desc Memory descriptor for the weights
        ///     applied to the layer input.
        /// @param weights_iter_desc Memory descriptor for the weights applied
        ///     to the recurrent input.
        /// @param bias_desc Bias memory descriptor.
        /// @param dst_layer_desc Memory descriptor for the output vector.
        /// @param dst_iter_desc Memory descriptor for the output recurrent
        ///     hidden state vector.
        /// @param flags Unused.
        desc(prop_kind aprop_kind, rnn_direction direction,
             const memory::desc &src_layer_desc,
             const memory::desc &src_iter_desc,
             const memory::desc &weights_layer_desc,
             const memory::desc &weights_iter_desc,
             const memory::desc &bias_desc,
             const memory::desc &dst_layer_desc,
             const memory::desc &dst_iter_desc,
             rnn_flags flags = rnn_flags::undef) {
            error::wrap_c_api(
                zendnn_lbr_gru_forward_desc_init(&data,
                                                 zendnn::convert_to_c(aprop_kind),
                                                 zendnn::convert_to_c(direction), &src_layer_desc.data,
                                                 &src_iter_desc.data, &weights_layer_desc.data,
                                                 &weights_iter_desc.data, &bias_desc.data,
                                                 &dst_layer_desc.data, &dst_iter_desc.data,
                                                 zendnn::convert_to_c(flags)),
                "could not create a descriptor for an LBR GRU forward "
                "propagation primitive");
        }
    };

    /// Primitive descriptor for an LBR GRU forward propagation primitive.
    struct primitive_desc : public rnn_primitive_desc_base {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a LBR GRU forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a LBR GRU forward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                       bool allow_empty = false)
            : rnn_primitive_desc_base(
                  &adesc.data, nullptr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a LBR GRU forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a LBR GRU forward propagation
        ///     primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                       const engine &aengine, bool allow_empty = false)
            : rnn_primitive_desc_base(
                  &adesc.data, &attr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a LBR GRU forward propagation
        /// primitive from a C API primitive descriptor that must have a
        /// matching kind.
        ///
        /// @param pd C API primitive descriptor for a LBR GRU forward
        ///     propagation primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : rnn_primitive_desc_base(pd, zendnn::prop_kind::forward_training,
                                      zendnn::prop_kind::forward_inference,
                                      zendnn::algorithm::lbr_gru) {}

        /// @copydoc zendnn::rnn_primitive_desc_base::src_layer_desc()const
        memory::desc src_layer_desc() const {
            return rnn_base::src_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::src_iter_desc()const
        memory::desc src_iter_desc() const {
            return rnn_base::src_iter_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::weights_layer_desc()const
        memory::desc weights_layer_desc() const {
            return rnn_base::weights_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::weights_iter_desc()const
        memory::desc weights_iter_desc() const {
            return rnn_base::weights_iter_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::bias_desc()const
        memory::desc bias_desc() const {
            return rnn_base::bias_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::dst_layer_desc()const
        memory::desc dst_layer_desc() const {
            return rnn_base::dst_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::dst_iter_desc()const
        memory::desc dst_iter_desc() const {
            return rnn_base::dst_iter_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const {
            return rnn_base::workspace_desc();
        }
    };

    /// Default constructor. Produces an empty object.
    lbr_gru_forward() = default;

    /// Constructs an LBR GRU forward propagation primitive.
    /// @param pd Primitive descriptor for an LBR GRU forward propagation
    ///     primitive.
    lbr_gru_forward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs an LBR GRU forward propagation primitive from a cache blob.
    /// @param pd Primitive descriptor for an LBR GRU forward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    lbr_gru_forward(
        const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// LBR GRU backward propagation primitive.
struct lbr_gru_backward : public primitive {
    /// Descriptor for a LBR GRU backward propagation primitive.
    struct desc {
        zendnn_rnn_desc_t data;

        /// Constructs a descriptor for LBR GRU backward propagation
        /// primitive.
        ///
        /// The following arguments may point to a zero memory descriptor:
        /// - @p src_iter_desc together with @p diff_src_iter_desc,
        /// - @p bias_desc together with @p diff_bias_desc,
        /// - @p dst_iter_desc together with @p diff_dst_iter_desc.
        ///
        /// This would then indicate that the LBR GRU backward propagation
        /// primitive should not use them and should default to zero values
        /// instead.
        ///
        /// @note
        ///     All memory descriptors may be initialized with
        ///     #zendnn::memory::format_tag::any value of @p format_tag.
        ///
        /// @param aprop_kind Propagation kind. Must be
        ///     #zendnn::prop_kind::backward.
        /// @param direction RNN direction. See @ref zendnn::rnn_direction for
        ///     more info.
        /// @param src_layer_desc Memory descriptor for the input vector.
        /// @param src_iter_desc Memory descriptor for the input recurrent
        ///     hidden state vector.
        /// @param weights_layer_desc Memory descriptor for the weights
        ///     applied to the layer input.
        /// @param weights_iter_desc Memory descriptor for the weights applied
        ///     to the recurrent input.
        /// @param bias_desc Bias memory descriptor.
        /// @param dst_layer_desc Memory descriptor for the output vector.
        /// @param dst_iter_desc Memory descriptor for the output recurrent
        ///     hidden state vector.
        /// @param diff_src_layer_desc Memory descriptor for the diff of input
        ///     vector.
        /// @param diff_src_iter_desc Memory descriptor for the diff of input
        ///     recurrent hidden state vector.
        /// @param diff_weights_layer_desc Memory descriptor for the diff of
        ///     weights applied to the layer input.
        /// @param diff_weights_iter_desc Memory descriptor for the diff of
        ///     weights applied to the recurrent input.
        /// @param diff_bias_desc Diff bias memory descriptor.
        /// @param diff_dst_layer_desc Memory descriptor for the diff of
        ///     output vector.
        /// @param diff_dst_iter_desc Memory descriptor for the diff of output
        ///     recurrent hidden state vector.
        /// @param flags Unused.
        desc(prop_kind aprop_kind, rnn_direction direction,
             const memory::desc &src_layer_desc,
             const memory::desc &src_iter_desc,
             const memory::desc &weights_layer_desc,
             const memory::desc &weights_iter_desc,
             const memory::desc &bias_desc,
             const memory::desc &dst_layer_desc,
             const memory::desc &dst_iter_desc,
             const memory::desc &diff_src_layer_desc,
             const memory::desc &diff_src_iter_desc,
             const memory::desc &diff_weights_layer_desc,
             const memory::desc &diff_weights_iter_desc,
             const memory::desc &diff_bias_desc,
             const memory::desc &diff_dst_layer_desc,
             const memory::desc &diff_dst_iter_desc,
             rnn_flags flags = rnn_flags::undef) {
            error::wrap_c_api(
                zendnn_lbr_gru_backward_desc_init(&data,
                                                  zendnn::convert_to_c(aprop_kind),
                                                  zendnn::convert_to_c(direction), &src_layer_desc.data,
                                                  &src_iter_desc.data, &weights_layer_desc.data,
                                                  &weights_iter_desc.data, &bias_desc.data,
                                                  &dst_layer_desc.data, &dst_iter_desc.data,
                                                  &diff_src_layer_desc.data, &diff_src_iter_desc.data,
                                                  &diff_weights_layer_desc.data,
                                                  &diff_weights_iter_desc.data, &diff_bias_desc.data,
                                                  &diff_dst_layer_desc.data, &diff_dst_iter_desc.data,
                                                  zendnn::convert_to_c(flags)),
                "could not create a descriptor for an LBR GRU backward "
                "propagation primitive");
        }
    };

    /// Primitive descriptor for an LBR GRU backward propagation primitive.
    struct primitive_desc : public rnn_primitive_desc_base {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an LBR GRU backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for an LBR GRU backward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for an LBR GRU
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                       const lbr_gru_forward::primitive_desc &hint_fwd_pd,
                       bool allow_empty = false)
            : rnn_primitive_desc_base(&adesc.data, nullptr, aengine,
                                      hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for an LBR GRU backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for an LBR GRU backward propagation
        ///     primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for an LBR GRU
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                       const engine &aengine,
                       const lbr_gru_forward::primitive_desc &hint_fwd_pd,
                       bool allow_empty = false)
            : rnn_primitive_desc_base(&adesc.data, &attr, aengine,
                                      hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a LBR GRU backward propagation
        /// primitive from a C API primitive descriptor that must have a
        /// matching kind.
        ///
        /// @param pd C API primitive descriptor for a LBR GRU backward
        ///     propagation primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : rnn_primitive_desc_base(
                  pd, zendnn::prop_kind::backward, zendnn::algorithm::lbr_gru) {}

        /// @copydoc zendnn::rnn_primitive_desc_base::src_layer_desc()const
        memory::desc src_layer_desc() const {
            return rnn_base::src_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::src_iter_desc()const
        memory::desc src_iter_desc() const {
            return rnn_base::src_iter_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::weights_layer_desc()const
        memory::desc weights_layer_desc() const {
            return rnn_base::weights_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::weights_iter_desc()const
        memory::desc weights_iter_desc() const {
            return rnn_base::weights_iter_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::bias_desc()const
        memory::desc bias_desc() const {
            return rnn_base::bias_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::dst_layer_desc()const
        memory::desc dst_layer_desc() const {
            return rnn_base::dst_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::dst_iter_desc()const
        memory::desc dst_iter_desc() const {
            return rnn_base::dst_iter_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const {
            return rnn_base::workspace_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::diff_src_layer_desc()const
        memory::desc diff_src_layer_desc() const {
            return rnn_base::diff_src_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::diff_src_iter_desc()const
        memory::desc diff_src_iter_desc() const {
            return rnn_base::diff_src_iter_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::diff_weights_layer_desc()const
        memory::desc diff_weights_layer_desc() const {
            return rnn_base::diff_weights_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::diff_weights_iter_desc()const
        memory::desc diff_weights_iter_desc() const {
            return rnn_base::diff_weights_iter_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::diff_bias_desc()const
        memory::desc diff_bias_desc() const {
            return rnn_base::diff_bias_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::diff_dst_layer_desc()const
        memory::desc diff_dst_layer_desc() const {
            return rnn_base::diff_dst_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::diff_dst_iter_desc()const
        memory::desc diff_dst_iter_desc() const {
            return rnn_base::diff_dst_iter_desc();
        }
    };

    /// Default constructor. Produces an empty object.
    lbr_gru_backward() = default;

    /// Constructs an LBR GRU backward propagation primitive.
    /// @param pd Primitive descriptor for an LBR GRU backward propagation
    ///     primitive.
    lbr_gru_backward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs an LBR GRU backward propagation primitive from a cache blob.
    /// @param pd Primitive descriptor for an LBR GRU backward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    lbr_gru_backward(
        const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// AUGRU forward propagation primitive.
struct augru_forward : public primitive {
    /// Descriptor for an AUGRU forward propagation primitive.
    struct desc {
        zendnn_rnn_desc_t data;

        /// Constructs a descriptor for an AUGRU forward propagation primitive.
        ///
        /// The following arguments may point to a zero memory descriptor:
        /// - @p src_iter_desc,
        /// - @p bias_desc,
        /// - @p dst_iter_desc.
        ///
        /// This would then indicate that the AUGRU forward propagation
        /// primitive should not use them and should default to zero values
        /// instead.
        ///
        /// @note
        ///     All memory descriptors except @p src_iter_desc may be
        ///     initialized with an #zendnn::memory::format_tag::any value of @p
        ///     format_tag.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #zendnn::prop_kind::forward_training, and
        ///     #zendnn::prop_kind::forward_inference.
        /// @param direction RNN direction. See @ref zendnn::rnn_direction for
        ///     more info.
        /// @param src_layer_desc Memory descriptor for the input vector.
        /// @param src_iter_desc Memory descriptor for the input recurrent
        ///     hidden state vector.
        /// @param attention_desc Memory descriptor for the attention vector.
        /// @param weights_layer_desc Memory descriptor for the weights
        ///     applied to the layer input.
        /// @param weights_iter_desc Memory descriptor for the weights applied
        ///     to the recurrent input.
        /// @param bias_desc Bias memory descriptor.
        /// @param dst_layer_desc Memory descriptor for the output vector.
        /// @param dst_iter_desc Memory descriptor for the output recurrent
        ///     hidden state vector.
        /// @param flags Unused.
        desc(prop_kind aprop_kind, rnn_direction direction,
             const memory::desc &src_layer_desc,
             const memory::desc &src_iter_desc,
             const memory::desc &attention_desc,
             const memory::desc &weights_layer_desc,
             const memory::desc &weights_iter_desc,
             const memory::desc &bias_desc,
             const memory::desc &dst_layer_desc,
             const memory::desc &dst_iter_desc,
             rnn_flags flags = rnn_flags::undef) {
            error::wrap_c_api(
                zendnn_augru_forward_desc_init(&data,
                                               zendnn::convert_to_c(aprop_kind),
                                               zendnn::convert_to_c(direction), &src_layer_desc.data,
                                               &src_iter_desc.data, &attention_desc.data,
                                               &weights_layer_desc.data, &weights_iter_desc.data,
                                               &bias_desc.data, &dst_layer_desc.data,
                                               &dst_iter_desc.data, zendnn::convert_to_c(flags)),
                "could not create a descriptor for an AUGRU forward "
                "propagation primitive");
        }
    };

    /// Primitive descriptor for an AUGRU forward propagation primitive.
    struct primitive_desc : public rnn_primitive_desc_base {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an AUGRU forward propagation
        /// primitive.
        ///
        /// @param adesc Descriptor for an AUGRU forward propagation primitive.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                       bool allow_empty = false)
            : rnn_primitive_desc_base(
                  &adesc.data, nullptr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for an AUGRU forward propagation
        /// primitive.
        ///
        /// @param adesc Descriptor for an AUGRU forward propagation primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                       const engine &aengine, bool allow_empty = false)
            : rnn_primitive_desc_base(
                  &adesc.data, &attr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for an AUGRU forward propagation
        /// primitive from a C API primitive descriptor that must have a
        /// matching kind.
        ///
        /// @param pd C API primitive descriptor for an AUGRU forward
        ///     propagation primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : rnn_primitive_desc_base(pd, zendnn::prop_kind::forward_training,
                                      zendnn::prop_kind::forward_inference,
                                      zendnn::algorithm::vanilla_augru) {}

        /// @copydoc zendnn::rnn_primitive_desc_base::src_layer_desc()const
        memory::desc src_layer_desc() const {
            return rnn_base::src_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::src_iter_desc()const
        memory::desc src_iter_desc() const {
            return rnn_base::src_iter_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::attention_desc()const
        memory::desc attention_desc() const {
            return rnn_base::augru_attention_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::weights_layer_desc()const
        memory::desc weights_layer_desc() const {
            return rnn_base::weights_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::weights_iter_desc()const
        memory::desc weights_iter_desc() const {
            return rnn_base::weights_iter_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::bias_desc()const
        memory::desc bias_desc() const {
            return rnn_base::bias_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::dst_layer_desc()const
        memory::desc dst_layer_desc() const {
            return rnn_base::dst_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::dst_iter_desc()const
        memory::desc dst_iter_desc() const {
            return rnn_base::dst_iter_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const {
            return rnn_base::workspace_desc();
        }
    };

    /// Default constructor. Produces an empty object.
    augru_forward() = default;

    /// Constructs an AUGRU forward propagation primitive.
    /// @param pd Primitive descriptor for an AUGRU forward propagation
    ///     primitive.
    augru_forward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs an AUGRU forward propagation primitive from a cache blob.
    /// @param pd Primitive descriptor for an AUGRU forward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    augru_forward(
        const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// AUGRU backward propagation primitive.
struct augru_backward : public primitive {
    /// Descriptor for an AUGRU backward propagation primitive.
    struct desc {
        zendnn_rnn_desc_t data;

        /// Constructs a descriptor for an AUGRU backward propagation primitive.
        ///
        /// The following arguments may point to a zero memory descriptor:
        /// - @p src_iter_desc together with @p diff_src_iter_desc,
        /// - @p bias_desc together with @p diff_bias_desc,
        /// - @p dst_iter_desc together with @p diff_dst_iter_desc.
        ///
        /// This would then indicate that the AUGRU backward propagation
        /// primitive should not use them and should default to zero values
        /// instead.
        ///
        /// @note
        ///     All memory descriptors may be initialized with
        ///     #zendnn::memory::format_tag::any value of @p format_tag.
        ///
        /// @param aprop_kind Propagation kind. Must be
        ///     #zendnn::prop_kind::backward.
        /// @param direction RNN direction. See @ref zendnn::rnn_direction for
        ///     more info.
        /// @param src_layer_desc Memory descriptor for the input vector.
        /// @param src_iter_desc Memory descriptor for the input recurrent
        ///     hidden state vector.
        /// @param attention_desc Memory descriptor for the attention vector.
        /// @param weights_layer_desc Memory descriptor for the weights
        ///     applied to the layer input.
        /// @param weights_iter_desc Memory descriptor for the weights applied
        ///     to the recurrent input.
        /// @param bias_desc Bias memory descriptor.
        /// @param dst_layer_desc Memory descriptor for the output vector.
        /// @param dst_iter_desc Memory descriptor for the output recurrent
        ///     hidden state vector.
        /// @param diff_src_layer_desc Memory descriptor for the diff of input
        ///     vector.
        /// @param diff_src_iter_desc Memory descriptor for the diff of input
        ///     recurrent hidden state vector.
        /// @param diff_attention_desc Memory descriptor for the diff of
        ///     attention vector.
        /// @param diff_weights_layer_desc Memory descriptor for the diff of
        ///     weights applied to the layer input.
        /// @param diff_weights_iter_desc Memory descriptor for the diff of
        ///     weights applied to the recurrent input.
        /// @param diff_bias_desc Diff bias memory descriptor.
        /// @param diff_dst_layer_desc Memory descriptor for the diff of
        ///     output vector.
        /// @param diff_dst_iter_desc Memory descriptor for the diff of output
        ///     recurrent hidden state vector.
        /// @param flags Unused.
        desc(prop_kind aprop_kind, rnn_direction direction,
             const memory::desc &src_layer_desc,
             const memory::desc &src_iter_desc,
             const memory::desc &attention_desc,
             const memory::desc &weights_layer_desc,
             const memory::desc &weights_iter_desc,
             const memory::desc &bias_desc,
             const memory::desc &dst_layer_desc,
             const memory::desc &dst_iter_desc,
             const memory::desc &diff_src_layer_desc,
             const memory::desc &diff_src_iter_desc,
             const memory::desc &diff_attention_desc,
             const memory::desc &diff_weights_layer_desc,
             const memory::desc &diff_weights_iter_desc,
             const memory::desc &diff_bias_desc,
             const memory::desc &diff_dst_layer_desc,
             const memory::desc &diff_dst_iter_desc,
             rnn_flags flags = rnn_flags::undef) {
            error::wrap_c_api(
                zendnn_augru_backward_desc_init(&data,
                                                zendnn::convert_to_c(aprop_kind),
                                                zendnn::convert_to_c(direction), &src_layer_desc.data,
                                                &src_iter_desc.data, &attention_desc.data,
                                                &weights_layer_desc.data, &weights_iter_desc.data,
                                                &bias_desc.data, &dst_layer_desc.data,
                                                &dst_iter_desc.data, &diff_src_layer_desc.data,
                                                &diff_src_iter_desc.data, &diff_attention_desc.data,
                                                &diff_weights_layer_desc.data,
                                                &diff_weights_iter_desc.data, &diff_bias_desc.data,
                                                &diff_dst_layer_desc.data, &diff_dst_iter_desc.data,
                                                zendnn::convert_to_c(flags)),
                "could not create a descriptor for an AUGRU backward "
                "propagation primitive");
        }
    };

    /// Primitive descriptor for an AUGRU backward propagation primitive.
    struct primitive_desc : public rnn_primitive_desc_base {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an AUGRU backward propagation
        /// primitive.
        ///
        /// @param adesc Descriptor for an AUGRU backward propagation primitive.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for an AUGRU
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                       const augru_forward::primitive_desc &hint_fwd_pd,
                       bool allow_empty = false)
            : rnn_primitive_desc_base(&adesc.data, nullptr, aengine,
                                      hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for an AUGRU backward propagation
        /// primitive.
        ///
        /// @param adesc Descriptor for an AUGRU backward propagation primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for an AUGRU
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                       const engine &aengine,
                       const augru_forward::primitive_desc &hint_fwd_pd,
                       bool allow_empty = false)
            : rnn_primitive_desc_base(&adesc.data, &attr, aengine,
                                      hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for an AUGRU backward propagation
        /// primitive from a C API primitive descriptor that must have a
        /// matching kind.
        ///
        /// @param pd C API primitive descriptor for an AUGRU backward
        ///     propagation primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : rnn_primitive_desc_base(pd, zendnn::prop_kind::backward,
                                      zendnn::algorithm::vanilla_augru) {}

        /// @copydoc zendnn::rnn_primitive_desc_base::src_layer_desc()const
        memory::desc src_layer_desc() const {
            return rnn_base::src_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::src_iter_desc()const
        memory::desc src_iter_desc() const {
            return rnn_base::src_iter_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::attention_desc()const
        memory::desc attention_desc() const {
            return rnn_base::augru_attention_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::weights_layer_desc()const
        memory::desc weights_layer_desc() const {
            return rnn_base::weights_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::weights_iter_desc()const
        memory::desc weights_iter_desc() const {
            return rnn_base::weights_iter_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::bias_desc()const
        memory::desc bias_desc() const {
            return rnn_base::bias_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::dst_layer_desc()const
        memory::desc dst_layer_desc() const {
            return rnn_base::dst_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::dst_iter_desc()const
        memory::desc dst_iter_desc() const {
            return rnn_base::dst_iter_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const {
            return rnn_base::workspace_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::diff_src_layer_desc()const
        memory::desc diff_src_layer_desc() const {
            return rnn_base::diff_src_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::diff_src_iter_desc()const
        memory::desc diff_src_iter_desc() const {
            return rnn_base::diff_src_iter_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::diff_attention_desc()const
        memory::desc diff_attention_desc() const {
            return rnn_base::diff_augru_attention_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::diff_weights_layer_desc()const
        memory::desc diff_weights_layer_desc() const {
            return rnn_base::diff_weights_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::diff_weights_iter_desc()const
        memory::desc diff_weights_iter_desc() const {
            return rnn_base::diff_weights_iter_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::diff_bias_desc()const
        memory::desc diff_bias_desc() const {
            return rnn_base::diff_bias_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::diff_dst_layer_desc()const
        memory::desc diff_dst_layer_desc() const {
            return rnn_base::diff_dst_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::diff_dst_iter_desc()const
        memory::desc diff_dst_iter_desc() const {
            return rnn_base::diff_dst_iter_desc();
        }
    };

    /// Default constructor. Produces an empty object.
    augru_backward() = default;

    /// Constructs an AUGRU backward propagation primitive.
    /// @param pd Primitive descriptor for an AUGRU backward propagation
    ///     primitive.
    augru_backward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs an AUGRU backward propagation primitive from a cache blob.
    /// @param pd Primitive descriptor for an AUGRU backward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    augru_backward(
        const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// LBR AUGRU forward propagation primitive.
struct lbr_augru_forward : public primitive {
    /// Descriptor for an LBR AUGRU forward propagation primitive.
    struct desc {
        zendnn_rnn_desc_t data;

        /// Constructs a descriptor for LBR AUGRU forward propagation primitive.
        ///
        /// The following arguments may point to a zero memory descriptor:
        /// - @p src_iter_desc,
        /// - @p bias_desc,
        /// - @p dst_iter_desc.
        ///
        /// This would then indicate that the LBR AUGRU forward propagation
        /// primitive should not use them and should default to zero values
        /// instead.
        ///
        /// @note
        ///     All memory descriptors except @p src_iter_desc may be
        ///     initialized with an #zendnn::memory::format_tag::any value of @p
        ///     format_tag.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #zendnn::prop_kind::forward_training, and
        ///     #zendnn::prop_kind::forward_inference.
        /// @param direction RNN direction. See @ref zendnn::rnn_direction for
        ///     more info.
        /// @param src_layer_desc Memory descriptor for the input vector.
        /// @param src_iter_desc Memory descriptor for the input recurrent
        ///     hidden state vector.
        /// @param attention_desc Memory descriptor for the attention vector.
        /// @param weights_layer_desc Memory descriptor for the weights
        ///     applied to the layer input.
        /// @param weights_iter_desc Memory descriptor for the weights applied
        ///     to the recurrent input.
        /// @param bias_desc Bias memory descriptor.
        /// @param dst_layer_desc Memory descriptor for the output vector.
        /// @param dst_iter_desc Memory descriptor for the output recurrent
        ///     hidden state vector.
        /// @param flags Unused.
        desc(prop_kind aprop_kind, rnn_direction direction,
             const memory::desc &src_layer_desc,
             const memory::desc &src_iter_desc,
             const memory::desc &attention_desc,
             const memory::desc &weights_layer_desc,
             const memory::desc &weights_iter_desc,
             const memory::desc &bias_desc,
             const memory::desc &dst_layer_desc,
             const memory::desc &dst_iter_desc,
             rnn_flags flags = rnn_flags::undef) {
            error::wrap_c_api(
                zendnn_lbr_augru_forward_desc_init(&data,
                                                   zendnn::convert_to_c(aprop_kind),
                                                   zendnn::convert_to_c(direction), &src_layer_desc.data,
                                                   &src_iter_desc.data, &attention_desc.data,
                                                   &weights_layer_desc.data, &weights_iter_desc.data,
                                                   &bias_desc.data, &dst_layer_desc.data,
                                                   &dst_iter_desc.data, zendnn::convert_to_c(flags)),
                "could not create a descriptor for an LBR AUGRU forward "
                "propagation primitive");
        }
    };

    /// Primitive descriptor for an LBR AUGRU forward propagation primitive.
    struct primitive_desc : public rnn_primitive_desc_base {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an LBR AUGRU forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for an LBR AUGRU forward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                       bool allow_empty = false)
            : rnn_primitive_desc_base(
                  &adesc.data, nullptr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for an LBR AUGRU forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for an LBR AUGRU forward propagation
        ///     primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                       const engine &aengine, bool allow_empty = false)
            : rnn_primitive_desc_base(
                  &adesc.data, &attr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for an LBR AUGRU forward propagation
        /// primitive from a C API primitive descriptor that must have a
        /// matching kind.
        ///
        /// @param pd C API primitive descriptor for an LBR AUGRU forward
        ///     propagation primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : rnn_primitive_desc_base(pd, zendnn::prop_kind::forward_training,
                                      zendnn::prop_kind::forward_inference,
                                      zendnn::algorithm::lbr_augru) {}

        /// @copydoc zendnn::rnn_primitive_desc_base::src_layer_desc()const
        memory::desc src_layer_desc() const {
            return rnn_base::src_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::src_iter_desc()const
        memory::desc src_iter_desc() const {
            return rnn_base::src_iter_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::attention_desc()const
        memory::desc attention_desc() const {
            return rnn_base::augru_attention_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::weights_layer_desc()const
        memory::desc weights_layer_desc() const {
            return rnn_base::weights_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::weights_iter_desc()const
        memory::desc weights_iter_desc() const {
            return rnn_base::weights_iter_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::bias_desc()const
        memory::desc bias_desc() const {
            return rnn_base::bias_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::dst_layer_desc()const
        memory::desc dst_layer_desc() const {
            return rnn_base::dst_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::dst_iter_desc()const
        memory::desc dst_iter_desc() const {
            return rnn_base::dst_iter_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const {
            return rnn_base::workspace_desc();
        }
    };

    /// Default constructor. Produces an empty object.
    lbr_augru_forward() = default;

    /// Constructs an LBR AUGRU forward propagation primitive.
    /// @param pd Primitive descriptor for an LBR AUGRU forward propagation
    ///     primitive.
    lbr_augru_forward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs an LBR AUGRU forward propagation primitive from a cache blob.
    /// @param pd Primitive descriptor for an LBR AUGRU forward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    lbr_augru_forward(
        const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// LBR AUGRU backward propagation primitive.
struct lbr_augru_backward : public primitive {
    /// Descriptor for an LBR AUGRU backward propagation primitive.
    struct desc {
        zendnn_rnn_desc_t data;

        /// Constructs a descriptor for LBR AUGRU backward propagation
        /// primitive.
        ///
        /// The following arguments may point to a zero memory descriptor:
        /// - @p src_iter_desc together with @p diff_src_iter_desc,
        /// - @p bias_desc together with @p diff_bias_desc,
        /// - @p dst_iter_desc together with @p diff_dst_iter_desc.
        ///
        /// This would then indicate that the LBR AUGRU backward propagation
        /// primitive should not use them and should default to zero values
        /// instead.
        ///
        /// @note
        ///     All memory descriptors may be initialized with
        ///     #zendnn::memory::format_tag::any value of @p format_tag.
        ///
        /// @param aprop_kind Propagation kind. Must be
        ///     #zendnn::prop_kind::backward.
        /// @param direction RNN direction. See @ref zendnn::rnn_direction for
        ///     more info.
        /// @param src_layer_desc Memory descriptor for the input vector.
        /// @param src_iter_desc Memory descriptor for the input recurrent
        ///     hidden state vector.
        /// @param attention_desc Memory descriptor for the attention vector.
        /// @param weights_layer_desc Memory descriptor for the weights
        ///     applied to the layer input.
        /// @param weights_iter_desc Memory descriptor for the weights applied
        ///     to the recurrent input.
        /// @param bias_desc Bias memory descriptor.
        /// @param dst_layer_desc Memory descriptor for the output vector.
        /// @param dst_iter_desc Memory descriptor for the output recurrent
        ///     hidden state vector.
        /// @param diff_src_layer_desc Memory descriptor for the diff of input
        ///     vector.
        /// @param diff_src_iter_desc Memory descriptor for the diff of input
        ///     recurrent hidden state vector.
        /// @param diff_attention_desc Memory descriptor for the diff of
        ///     attention vector.
        /// @param diff_weights_layer_desc Memory descriptor for the diff of
        ///     weights applied to the layer input.
        /// @param diff_weights_iter_desc Memory descriptor for the diff of
        ///     weights applied to the recurrent input.
        /// @param diff_bias_desc Diff bias memory descriptor.
        /// @param diff_dst_layer_desc Memory descriptor for the diff of
        ///     output vector.
        /// @param diff_dst_iter_desc Memory descriptor for the diff of output
        ///     recurrent hidden state vector.
        /// @param flags Unused.
        desc(prop_kind aprop_kind, rnn_direction direction,
             const memory::desc &src_layer_desc,
             const memory::desc &src_iter_desc,
             const memory::desc &attention_desc,
             const memory::desc &weights_layer_desc,
             const memory::desc &weights_iter_desc,
             const memory::desc &bias_desc,
             const memory::desc &dst_layer_desc,
             const memory::desc &dst_iter_desc,
             const memory::desc &diff_src_layer_desc,
             const memory::desc &diff_src_iter_desc,
             const memory::desc &diff_attention_desc,
             const memory::desc &diff_weights_layer_desc,
             const memory::desc &diff_weights_iter_desc,
             const memory::desc &diff_bias_desc,
             const memory::desc &diff_dst_layer_desc,
             const memory::desc &diff_dst_iter_desc,
             rnn_flags flags = rnn_flags::undef) {
            error::wrap_c_api(
                zendnn_lbr_augru_backward_desc_init(&data,
                                                    zendnn::convert_to_c(aprop_kind),
                                                    zendnn::convert_to_c(direction), &src_layer_desc.data,
                                                    &src_iter_desc.data, &attention_desc.data,
                                                    &weights_layer_desc.data, &weights_iter_desc.data,
                                                    &bias_desc.data, &dst_layer_desc.data,
                                                    &dst_iter_desc.data, &diff_src_layer_desc.data,
                                                    &diff_src_iter_desc.data, &diff_attention_desc.data,
                                                    &diff_weights_layer_desc.data,
                                                    &diff_weights_iter_desc.data, &diff_bias_desc.data,
                                                    &diff_dst_layer_desc.data, &diff_dst_iter_desc.data,
                                                    zendnn::convert_to_c(flags)),
                "could not create a descriptor for an LBR AUGRU backward "
                "propagation primitive");
        }
    };

    /// Primitive descriptor for an LBR AUGRU backward propagation primitive.
    struct primitive_desc : public rnn_primitive_desc_base {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an LBR AUGRU backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for an LBR AUGRU backward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for an LBR AUGRU
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                       const lbr_augru_forward::primitive_desc &hint_fwd_pd,
                       bool allow_empty = false)
            : rnn_primitive_desc_base(&adesc.data, nullptr, aengine,
                                      hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for an LBR AUGRU backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for an LBR AUGRU backward propagation
        ///     primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for an LBR AUGRU
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                       const engine &aengine,
                       const lbr_augru_forward::primitive_desc &hint_fwd_pd,
                       bool allow_empty = false)
            : rnn_primitive_desc_base(&adesc.data, &attr, aengine,
                                      hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for an LBR AUGRU backward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for an LBR AUGRU backward
        ///     propagation primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : rnn_primitive_desc_base(pd, zendnn::prop_kind::backward,
                                      zendnn::algorithm::lbr_augru) {}

        /// @copydoc zendnn::rnn_primitive_desc_base::src_layer_desc()const
        memory::desc src_layer_desc() const {
            return rnn_base::src_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::src_iter_desc()const
        memory::desc src_iter_desc() const {
            return rnn_base::src_iter_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::attention_desc()const
        memory::desc attention_desc() const {
            return rnn_base::augru_attention_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::weights_layer_desc()const
        memory::desc weights_layer_desc() const {
            return rnn_base::weights_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::weights_iter_desc()const
        memory::desc weights_iter_desc() const {
            return rnn_base::weights_iter_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::bias_desc()const
        memory::desc bias_desc() const {
            return rnn_base::bias_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::dst_layer_desc()const
        memory::desc dst_layer_desc() const {
            return rnn_base::dst_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::dst_iter_desc()const
        memory::desc dst_iter_desc() const {
            return rnn_base::dst_iter_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const {
            return rnn_base::workspace_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::diff_src_layer_desc()const
        memory::desc diff_src_layer_desc() const {
            return rnn_base::diff_src_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::diff_src_iter_desc()const
        memory::desc diff_src_iter_desc() const {
            return rnn_base::diff_src_iter_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::diff_attention_desc()const
        memory::desc diff_attention_desc() const {
            return rnn_base::diff_augru_attention_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::diff_weights_layer_desc()const
        memory::desc diff_weights_layer_desc() const {
            return rnn_base::diff_weights_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::diff_weights_iter_desc()const
        memory::desc diff_weights_iter_desc() const {
            return rnn_base::diff_weights_iter_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::diff_bias_desc()const
        memory::desc diff_bias_desc() const {
            return rnn_base::diff_bias_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::diff_dst_layer_desc()const
        memory::desc diff_dst_layer_desc() const {
            return rnn_base::diff_dst_layer_desc();
        }

        /// @copydoc zendnn::rnn_primitive_desc_base::diff_dst_iter_desc()const
        memory::desc diff_dst_iter_desc() const {
            return rnn_base::diff_dst_iter_desc();
        }
    };

    /// Default constructor. Produces an empty object.
    lbr_augru_backward() = default;

    /// Constructs an LBR AUGRU backward propagation primitive.
    /// @param pd Primitive descriptor for an LBR AUGRU backward propagation
    ///     primitive.
    lbr_augru_backward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs an LBR AUGRU backward propagation primitive from a cache blob.
    /// @param pd Primitive descriptor for an LBR AUGRU backward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    lbr_augru_backward(
        const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// @} zendnn_api_rnn

/// @addtogroup zendnn_api_shuffle Shuffle
///
/// A primitive to shuffle tensor data along an axis.
///
/// @sa @ref dev_guide_shuffle in developer guide
///
/// @{

/// Shuffle forward propagation primitive.
struct shuffle_forward : public primitive {
    /// Descriptor for a shuffle forward propagation primitive.
    struct desc {
        zendnn_shuffle_desc_t data;

        /// Constructs a descriptor for a shuffle forward propagation
        /// primitive.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #zendnn::prop_kind::forward_training, and
        ///     #zendnn::prop_kind::forward_inference.
        /// @param data_desc Source and destination memory descriptor.
        /// @param axis The axis along which the data is shuffled.
        /// @param group_size Shuffle group size.
        desc(prop_kind aprop_kind, const memory::desc &data_desc, int axis,
             int group_size) {
            error::wrap_c_api(zendnn_shuffle_forward_desc_init(&data,
                              zendnn::convert_to_c(aprop_kind),
                              &data_desc.data, axis, group_size),
                              "could not create a descriptor for a shuffle forward "
                              "propagation primitive");
        }
    };

    /// Primitive descriptor for a shuffle forward propagation primitive.
    struct primitive_desc : public zendnn::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a shuffle forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a shuffle forward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param attr Primitive attributes to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                       const primitive_attr &attr = primitive_attr(),
                       bool allow_empty = false)
            : zendnn::primitive_desc(
                  &adesc.data, &attr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a shuffle forward propagation
        /// primitive from a C API primitive descriptor that must have a
        /// matching kind.
        ///
        /// @param pd C API primitive descriptor for a shuffle forward
        ///     propagation primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : zendnn::primitive_desc(pd, zendnn::primitive::kind::shuffle,
                                     zendnn::prop_kind::forward_training,
                                     zendnn::prop_kind::forward_inference) {}

        /// @copydoc zendnn::primitive_desc_base::src_desc()const
        memory::desc src_desc() const {
            return base::src_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const {
            return base::dst_desc(0);
        }
    };

    /// Default constructor. Produces an empty object.
    shuffle_forward() = default;

    /// Constructs a shuffle forward propagation primitive.
    /// @param pd Primitive descriptor for a shuffle forward propagation
    ///     primitive.
    shuffle_forward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a shuffle forward propagation primitive from a cache blob.
    /// @param pd Primitive descriptor for a shuffle forward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    shuffle_forward(
        const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// Shuffle backward propagation primitive.
struct shuffle_backward : public primitive {
    /// Descriptor for a shuffle primitive backward propagation
    /// primitive.
    struct desc {
        zendnn_shuffle_desc_t data;

        /// Constructs a descriptor for a shuffle backward propagation
        /// primitive.
        ///
        /// @param diff_data_desc Diff source and diff destination memory
        ///     descriptor.
        /// @param axis The axis along which the data is shuffled.
        /// @param group_size Shuffle group size.
        desc(const memory::desc &diff_data_desc, int axis, int group_size) {
            error::wrap_c_api(zendnn_shuffle_backward_desc_init(&data,
                              &diff_data_desc.data, axis, group_size),
                              "could not create a descriptor for a shuffle backward "
                              "propagation primitive");
        }
    };

    /// Primitive descriptor for a shuffle backward propagation primitive.
    struct primitive_desc : public zendnn::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a shuffle backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a shuffle backward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param attr Primitive attributes to use.
        /// @param hint_fwd_pd Primitive descriptor for a shuffle
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                       const shuffle_forward::primitive_desc &hint_fwd_pd,
                       const primitive_attr &attr = primitive_attr(),
                       bool allow_empty = false)
            : zendnn::primitive_desc(&adesc.data, &attr, aengine,
                                     hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a shuffle backward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a shuffle backward
        ///     propagation primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : zendnn::primitive_desc(pd, zendnn::primitive::kind::shuffle,
                                     zendnn::prop_kind::backward_data) {}

        /// @copydoc zendnn::primitive_desc_base::diff_src_desc()const
        memory::desc diff_src_desc() const {
            return base::diff_src_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::diff_dst_desc()const
        memory::desc diff_dst_desc() const {
            return base::diff_dst_desc(0);
        }
    };

    /// Default constructor. Produces an empty object.
    shuffle_backward() = default;

    /// Constructs a shuffle backward propagation primitive.
    /// @param pd Primitive descriptor for a shuffle backward propagation
    ///     primitive.
    shuffle_backward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a shuffle backward propagation primitive from a cache blob.
    /// @param pd Primitive descriptor for a shuffle backward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    shuffle_backward(
        const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// @} zendnn_api_shuffle

/// @addtogroup zendnn_api_binary Binary
///
/// A primitive to perform tensor operations over two tensors.
///
/// @sa @ref dev_guide_binary in developer guide
///
/// @{

/// Elementwise binary operator primitive.
struct binary : public primitive {
    /// Descriptor for an elementwise binary operator primitive.
    struct desc {
        /// Underlying C operation descriptor.
        zendnn_binary_desc_t data;

        /// Default constructor. Produces an empty object.
        desc() = default;

        /// Constructs a descriptor for an elementwise binary operator
        /// primitive.
        ///
        /// @param aalgorithm Elementwise binary algorithm.
        /// @param src0 Memory descriptor for source tensor #0.
        /// @param src1 Memory descriptor for source tensor #1.
        /// @param dst Memory descriptor for destination tensor.
        desc(algorithm aalgorithm, const memory::desc &src0,
             const memory::desc &src1, const memory::desc &dst) {
            error::wrap_c_api(
                zendnn_binary_desc_init(&data, zendnn::convert_to_c(aalgorithm),
                                        &src0.data, &src1.data, &dst.data),
                "could not create a descriptor for a binary operation "
                "primitive");
        }
    };

    /// Primitive descriptor for an elementwise binary operator primitive.
    struct primitive_desc : public zendnn::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an elementwise binary operator
        /// primitive.
        ///
        /// @param adesc Descriptor for an elementwise binary operator primitive.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                       bool allow_empty = false)
            : zendnn::primitive_desc(
                  &adesc.data, nullptr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for an elementwise binary operator
        /// primitive.
        ///
        /// @param adesc Descriptor for an elementwise binary operator primitive.
        /// @param aengine Engine to use.
        /// @param attr Primitive attributes to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                       const engine &aengine, bool allow_empty = false)
            : zendnn::primitive_desc(
                  &adesc.data, &attr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a binary primitive from a C
        /// API primitive descriptor that must have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a binary primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : zendnn::primitive_desc(pd, zendnn::primitive::kind::binary) {}

        /// @copydoc zendnn::primitive_desc_base::src_desc(int)const
        memory::desc src_desc(int idx = 0) const {
            return base::src_desc(idx);
        }

        /// Returns the memory descriptor for source #0.
        memory::desc src0_desc() const {
            return base::src_desc(0);
        }

        /// Returns the memory descriptor for source #1.
        memory::desc src1_desc() const {
            return base::src_desc(1);
        }

        /// @copydoc zendnn::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const {
            return base::dst_desc(0);
        }
    };

    /// Default constructor. Produces an empty object.
    binary() = default;

    /// Constructs an elementwise binary operation primitive.
    /// @param pd Primitive descriptor for an elementwise binary operation
    ///     primitive.
    binary(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs an elementwise binary operation primitive from a cache blob.
    /// @param pd Primitive descriptor for an elementwise binary operation
    ///     primitive.
    /// @param cache_blob Cache blob.
    binary(const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// @} zendnn_api_binary

/// @addtogroup zendnn_api_matmul Matrix Multiplication
///
/// A primitive to perform matrix-matrix multiplication. The batched mode
/// is supported with 3D tensors.
///
/// @sa @ref dev_guide_matmul in developer guide
///
///
/// @{

/// Matrix multiplication (matmul) primitive.
struct matmul : public primitive {
    /// Descriptor for a matmul primitive.
    struct desc {
        zendnn_matmul_desc_t data;

        /// Constructs a descriptor for a matmul primitive.
        ///
        /// @param src_desc Memory descriptor for source (matrix A).
        /// @param weights_desc Memory descriptor for weights (matrix B).
        /// @param dst_desc Memory descriptor for destination (matrix C).
        desc(const memory::desc &src_desc, const memory::desc &weights_desc,
             const memory::desc &dst_desc) {
            zendnnInfo(ZENDNN_APILOG, "matmul desc create - no bias");
            error::wrap_c_api(
                zendnn_matmul_desc_init(&data, &src_desc.data,
                                        &weights_desc.data, nullptr, &dst_desc.data),
                "could not create a descriptor for a matmul primitive");
        }

        /// Constructs a descriptor for a matmul primitive.
        ///
        /// @param src_desc Memory descriptor for source (matrix A).
        /// @param weights_desc Memory descriptor for weights (matrix B).
        /// @param dst_desc Memory descriptor for destination (matrix C).
        /// @param bias_desc Memory descriptor for bias.
        desc(const memory::desc &src_desc, const memory::desc &weights_desc,
             const memory::desc &bias_desc, const memory::desc &dst_desc) {
            zendnnInfo(ZENDNN_APILOG, "matmul desc create - bias");
            error::wrap_c_api(zendnn_matmul_desc_init(&data, &src_desc.data,
                              &weights_desc.data, &bias_desc.data,
                              &dst_desc.data),
                              "could not create a descriptor for a matmul primitive");
        }
    };

    /// Primitive descriptor for a matmul primitive.
    struct primitive_desc : public zendnn::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a matmul primitive.
        ///
        /// @param adesc Descriptor for a matmul primitive.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                       bool allow_empty = false)
            : zendnn::primitive_desc(
                  &adesc.data, nullptr, aengine, nullptr, allow_empty) {
            zendnnInfo(ZENDNN_APILOG, "matmul primitive_desc create - no attr");
        }

        /// Constructs a primitive descriptor for a matmul primitive.
        ///
        /// @param adesc Descriptor for a matmul primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                       const engine &aengine, bool allow_empty = false)
            : zendnn::primitive_desc(
                  &adesc.data, &attr, aengine, nullptr, allow_empty) {
            zendnnInfo(ZENDNN_APILOG, "matmul primitive_desc create - attr");
        }

        /// Constructs a primitive descriptor for a matmul primitive from a C
        /// API primitive descriptor that must have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a matmul primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : zendnn::primitive_desc(pd, zendnn::primitive::kind::matmul) {
            zendnnInfo(ZENDNN_APILOG, "matmul primitive_desc create - C API");
        }

        /// @copydoc zendnn::primitive_desc_base::src_desc()const
        memory::desc src_desc() const {
            return query_md(query::src_md, 0);
        }

        /// @copydoc zendnn::primitive_desc_base::weights_desc()const
        memory::desc weights_desc() const {
            return query_md(query::weights_md, 0);
        }

        /// @copydoc zendnn::convolution_forward::primitive_desc::bias_desc()const
        memory::desc bias_desc() const {
            return query_md(query::weights_md, 1);
        }

        /// @copydoc zendnn::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const {
            return query_md(query::dst_md, 0);
        }
    };

    /// Default constructor. Produces an empty object.
    matmul() = default;

    /// Constructs a matmul primitive.
    /// @param pd Primitive descriptor for a matmul primitive.
    matmul(const primitive_desc &pd) : primitive(pd) {
        zendnnInfo(ZENDNN_APILOG, "matmul primitive create");
    }

    /// Constructs a matmul primitive from a cache blob.
    /// @param pd Primitive descriptor for a matmul primitive.
    /// @param cache_blob Cache blob.
    matmul(const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// @} zendnn_api_matmul

/// @addtogroup zendnn_api_resampling Resampling
///
/// A primitive to compute resampling operation on 1D, 2D or 3D data tensor
/// using Nearest Neighbor, or Linear (Bilinear, Trilinear) interpolation
/// method.
///
/// @sa @ref dev_guide_resampling in developer guide
///
/// @{

/// Resampling forward propagation.
struct resampling_forward : public primitive {
    /// Descriptor for resampling forward propagation.
    struct desc {
        zendnn_resampling_desc_t data;

        /// Constructs a descriptor for a resampling forward propagation
        /// primitive using source and destination memory descriptors.
        ///
        /// @note
        ///     Destination memory descriptor may be initialized with
        ///     #zendnn::memory::format_tag::any value of @p format_tag.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #zendnn::prop_kind::forward_training, and
        ///     #zendnn::prop_kind::forward_inference.
        /// @param aalgorithm resampling algorithm kind: either
        ///     #zendnn::algorithm::resampling_nearest, or
        ///     #zendnn::algorithm::resampling_linear
        /// @param src_desc Source memory descriptor.
        /// @param dst_desc Destination memory descriptor.
        desc(prop_kind aprop_kind, algorithm aalgorithm,
             const memory::desc &src_desc, const memory::desc &dst_desc) {
            error::wrap_c_api(zendnn_resampling_forward_desc_init(&data,
                              zendnn::convert_to_c(aprop_kind),
                              convert_to_c(aalgorithm), nullptr,
                              &src_desc.data, &dst_desc.data),
                              "could not create a resampling forward descriptor");
        }

        /// Constructs a descriptor for a resampling forward propagation
        /// primitive using source memory descriptor and factors.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #zendnn::prop_kind::forward_training, and
        ///     #zendnn::prop_kind::forward_inference.
        /// @param aalgorithm resampling algorithm kind: either
        ///     #zendnn::algorithm::resampling_nearest, or
        ///     #zendnn::algorithm::resampling_linear
        /// @param factors Vector of scaling factors for spatial dimension.
        /// @param src_desc Source memory descriptor.
        desc(prop_kind aprop_kind, algorithm aalgorithm,
             const std::vector<float> &factors,
             const memory::desc &src_desc) {
            memory::validate_dims(factors, src_desc.data.ndims - 2);
            error::wrap_c_api(zendnn_resampling_forward_desc_init(&data,
                              zendnn::convert_to_c(aprop_kind),
                              convert_to_c(aalgorithm), &factors[0],
                              &src_desc.data, nullptr),
                              "could not create a resampling forward descriptor");
        }

        /// Constructs a descriptor for a resampling forward propagation
        /// primitive.
        ///
        /// @note
        ///     The destination memory descriptor may be initialized with
        ///     #zendnn::memory::format_tag::any value of @p format_tag.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #zendnn::prop_kind::forward_training, and
        ///     #zendnn::prop_kind::forward_inference.
        /// @param aalgorithm resampling algorithm kind: either
        ///     #zendnn::algorithm::resampling_nearest, or
        ///     #zendnn::algorithm::resampling_linear
        /// @param factors Vector of scaling factors for spatial dimension.
        /// @param src_desc Source memory descriptor.
        /// @param dst_desc Destination memory descriptor.
        desc(prop_kind aprop_kind, algorithm aalgorithm,
             const std::vector<float> &factors, const memory::desc &src_desc,
             const memory::desc &dst_desc) {
            if (!factors.empty()) {
                memory::validate_dims(factors, src_desc.data.ndims - 2);
            }
            error::wrap_c_api(zendnn_resampling_forward_desc_init(&data,
                              zendnn::convert_to_c(aprop_kind),
                              convert_to_c(aalgorithm), factors.data(),
                              &src_desc.data, &dst_desc.data),
                              "could not create a resampling forward descriptor");
        }
    };

    /// Primitive descriptor for a resampling forward propagation primitive.
    struct primitive_desc : public zendnn::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a resampling forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a resampling forward propagation
        /// primitive.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                       bool allow_empty = false)
            : zendnn::primitive_desc(
                  &adesc.data, nullptr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a resampling forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a resampling forward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param attr Primitive attributes to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                       const engine &aengine, bool allow_empty = false)
            : zendnn::primitive_desc(
                  &adesc.data, &attr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a resampling forward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a resampling forward
        ///     propagation primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : zendnn::primitive_desc(pd, zendnn::primitive::kind::resampling,
                                     zendnn::prop_kind::forward_training,
                                     zendnn::prop_kind::forward_inference) {}

        /// @copydoc zendnn::primitive_desc_base::src_desc()const
        memory::desc src_desc() const {
            return base::src_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const {
            return base::dst_desc(0);
        }
    };

    /// Default constructor. Produces an empty object.
    resampling_forward() = default;

    /// Constructs a resampling forward propagation primitive.
    /// @param pd Primitive descriptor for a resampling forward propagation
    ///     primitive.
    resampling_forward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a resampling forward propagation primitive from a cache
    ///     blob.
    /// @param pd Primitive descriptor for a resampling forward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    resampling_forward(
        const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// Resampling backward propagation primitive.
struct resampling_backward : public primitive {
    /// Descriptor for a resampling backward propagation primitive.
    struct desc {
        zendnn_resampling_desc_t data;

        /// Constructs a descriptor for a resampling backward propagation
        /// primitive using source and destination memory descriptors.
        ///
        /// @param aalgorithm resampling algorithm kind: either
        ///     #zendnn::algorithm::resampling_nearest, or
        ///     #zendnn::algorithm::resampling_linear
        /// @param diff_src_desc Diff source memory descriptor.
        /// @param diff_dst_desc Diff destination memory descriptor.
        desc(algorithm aalgorithm, const memory::desc &diff_src_desc,
             const memory::desc &diff_dst_desc) {
            error::wrap_c_api(zendnn_resampling_backward_desc_init(&data,
                              convert_to_c(aalgorithm), nullptr,
                              &diff_src_desc.data, &diff_dst_desc.data),
                              "could not create a resampling backward data descriptor");
        }

        /// Constructs a descriptor for resampling backward propagation
        /// primitive.
        ///
        /// @param aalgorithm resampling algorithm kind: either
        ///     #zendnn::algorithm::resampling_nearest, or
        ///     #zendnn::algorithm::resampling_linear
        /// @param factors Vector of scaling factors for spatial dimension.
        /// @param diff_src_desc Diff source memory descriptor.
        /// @param diff_dst_desc Diff destination memory descriptor.
        desc(algorithm aalgorithm, const std::vector<float> &factors,
             const memory::desc &diff_src_desc,
             const memory::desc &diff_dst_desc) {
            if (!factors.empty()) {
                memory::validate_dims(factors, diff_src_desc.data.ndims - 2);
            }
            error::wrap_c_api(zendnn_resampling_backward_desc_init(&data,
                              convert_to_c(aalgorithm), factors.data(),
                              &diff_src_desc.data, &diff_dst_desc.data),
                              "could not create a resampling backward data descriptor");
        }
    };

    /// Primitive descriptor for resampling backward propagation primitive.
    struct primitive_desc : public zendnn::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a resampling backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a resampling backward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a resampling forward
        ///     propagation primitive. It is used as a hint for deciding which
        ///     memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                       const resampling_forward::primitive_desc &hint_fwd_pd,
                       bool allow_empty = false)
            : zendnn::primitive_desc(&adesc.data, nullptr, aengine,
                                     hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a resampling backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a resampling backward propagation
        ///     primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a resampling forward
        ///     propagation primitive. It is used as a hint for deciding which
        ///     memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                       const engine &aengine,
                       const resampling_forward::primitive_desc &hint_fwd_pd,
                       bool allow_empty = false)
            : zendnn::primitive_desc(&adesc.data, &attr, aengine,
                                     hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a resampling backward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a resampling backward
        ///     propagation primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : zendnn::primitive_desc(pd, zendnn::primitive::kind::resampling,
                                     zendnn::prop_kind::backward_data) {}

        /// @copydoc zendnn::primitive_desc_base::diff_src_desc()const
        memory::desc diff_src_desc() const {
            return base::diff_src_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::diff_dst_desc()const
        memory::desc diff_dst_desc() const {
            return base::diff_dst_desc(0);
        }
    };

    /// Default constructor. Produces an empty object.
    resampling_backward() = default;

    /// Constructs a resampling backward propagation primitive.
    /// @param pd Primitive descriptor for a resampling backward propagation
    ///     primitive.
    resampling_backward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a resampling backward propagation primitive from a cache
    ///     blob.
    /// @param pd Primitive descriptor for a resampling backward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    resampling_backward(
        const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// @} zendnn_api_resampling

/// @addtogroup zendnn_api_pooling_v2 Pooling_v2
///
/// A primitive to perform max or average pooling with dilation.
///
/// @sa @ref dev_guide_pooling in developer guide
///
/// @{

/// Pooling v2 (dilated pooling) forward propagation primitive.
struct pooling_v2_forward : public primitive {
    /// Descriptor for a pooling forward propagation primitive.
    struct desc {
        zendnn_pooling_v2_desc_t data;

        /// Constructs a descriptor for pooling v2
        /// (dilated pooling) forward propagation primitive.
        ///
        /// Arrays @p strides, @p kernel, @p dilation, @p padding_l
        /// and @p padding_r contain values for spatial dimensions only and
        /// hence must have the same number of elements as there are spatial
        /// dimensions. The order of values is the same as in the tensor:
        /// depth (for 3D tensors), height (for 3D and 2D tensors), and width.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #zendnn::prop_kind::forward_training, and
        ///     #zendnn::prop_kind::forward_inference.
        /// @param aalgorithm Pooling algorithm kind: either
        ///     #zendnn::algorithm::pooling_max,
        ///     #zendnn::algorithm::pooling_avg_include_padding,
        ///     or #zendnn::algorithm::pooling_avg (same as
        ///     #zendnn::algorithm::pooling_avg_exclude_padding).
        /// @param src_desc Source memory descriptor.
        /// @param dst_desc Destination memory descriptor.
        /// @param strides Vector of strides for spatial dimension.
        /// @param kernel Vector of kernel spatial dimensions.
        /// @param dilation Array of dilations for spatial dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        desc(prop_kind aprop_kind, algorithm aalgorithm,
             const memory::desc &src_desc, const memory::desc &dst_desc,
             const memory::dims &strides, const memory::dims &kernel,
             const memory::dims &dilation, const memory::dims &padding_l,
             const memory::dims &padding_r) {
            memory::validate_dims(strides, src_desc.data.ndims - 2);
            memory::validate_dims(kernel, src_desc.data.ndims - 2);
            memory::validate_dims(padding_l, src_desc.data.ndims - 2);
            memory::validate_dims(padding_r, src_desc.data.ndims - 2);
            memory::validate_dims(dilation, src_desc.data.ndims - 2);
            error::wrap_c_api(
                zendnn_pooling_v2_forward_desc_init(&data,
                                                    zendnn::convert_to_c(aprop_kind),
                                                    convert_to_c(aalgorithm), &src_desc.data,
                                                    &dst_desc.data, &strides[0], &kernel[0],
                                                    &dilation[0], &padding_l[0], &padding_r[0]),
                "could not create a descriptor for a pooling forward "
                "propagation primitive");
        }
    };

    /// Primitive descriptor for a pooling forward propagation primitive.
    struct primitive_desc : public zendnn::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a pooling v2
        /// (dilated pooling) forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a pooling forward propagation primitive.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                       bool allow_empty = false)
            : zendnn::primitive_desc(
                  &adesc.data, nullptr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a pooling v2
        /// (dilated pooling) forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a pooling forward propagation primitive.
        /// @param aengine Engine to use.
        /// @param attr Primitive attributes to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                       const engine &aengine, bool allow_empty = false)
            : zendnn::primitive_desc(
                  &adesc.data, &attr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a pooling v2
        /// (dilated pooling) forward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a pooling forward
        ///     propagation primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : zendnn::primitive_desc(pd, zendnn::primitive::kind::pooling_v2,
                                     zendnn::prop_kind::forward_training,
                                     zendnn::prop_kind::forward_inference) {}

        /// @copydoc zendnn::primitive_desc_base::src_desc()const
        memory::desc src_desc() const {
            return base::src_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const {
            return base::dst_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const {
            return base::workspace_desc();
        }
    };

    /// Default constructor. Produces an empty object.
    pooling_v2_forward() = default;

    /// Constructs a pooling v2 (dilated pooling) forward
    /// propagation primitive.
    /// @param pd Primitive descriptor for a pooling v2
    /// (dilated pooling) forward propagation primitive.
    pooling_v2_forward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a pooling v2 (dilated pooling) forward
    /// propagation primitive from a cache blob.
    /// @param pd Primitive descriptor for a pooling v2
    /// (dilated pooling) forward propagation primitive.
    /// @param cache_blob Cache blob.
    pooling_v2_forward(
        const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// Pooling v2 (dilated pooling) backward propagation primitive.
struct pooling_v2_backward : public primitive {
    /// Descriptor for a pooling backward propagation primitive.
    struct desc {
        zendnn_pooling_v2_desc_t data;

        /// Constructs a descriptor for pooling v2 (dilated pooling) backward
        /// propagation primitive.
        ///
        /// Arrays @p strides, @p kernel, @p dilation, @p padding_l
        /// and @p padding_r contain values for spatial dimensions only and
        /// hence must have the same number of elements as there are spatial
        /// dimensions. The order of values is the same as in the tensor:
        /// depth (for 3D tensors), height (for 3D and 2D tensors), and width.
        ///
        /// @param aalgorithm Pooling algorithm kind: either
        ///     #zendnn::algorithm::pooling_max,
        ///     #zendnn::algorithm::pooling_avg_include_padding,
        ///     or #zendnn::algorithm::pooling_avg (same as
        ///     #zendnn::algorithm::pooling_avg_exclude_padding).
        /// @param diff_src_desc Diff source memory descriptor.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param strides Vector of strides for spatial dimension.
        /// @param kernel Vector of kernel spatial dimensions.
        /// @param dilation Array of dilations for spatial dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        desc(algorithm aalgorithm, const memory::desc &diff_src_desc,
             const memory::desc &diff_dst_desc, const memory::dims &strides,
             const memory::dims &kernel, const memory::dims &dilation,
             const memory::dims &padding_l, const memory::dims &padding_r) {
            memory::validate_dims(strides, diff_src_desc.data.ndims - 2);
            memory::validate_dims(kernel, diff_src_desc.data.ndims - 2);
            memory::validate_dims(padding_l, diff_src_desc.data.ndims - 2);
            memory::validate_dims(padding_r, diff_src_desc.data.ndims - 2);
            memory::validate_dims(dilation, diff_src_desc.data.ndims - 2);
            error::wrap_c_api(
                zendnn_pooling_v2_backward_desc_init(&data,
                        convert_to_c(aalgorithm), &diff_src_desc.data,
                        &diff_dst_desc.data, &strides[0], &kernel[0],
                        &dilation[0], &padding_l[0], &padding_r[0]),
                "could not create a descriptor for a pooling backward "
                "propagation primitive");
        }
    };

    /// Primitive descriptor for a pooling v2 (dilated pooling) backward
    /// propagation primitive.
    struct primitive_desc : public zendnn::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a pooling v2
        /// (dilated pooling) backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a pooling backward propagation primitive.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a pooling forward
        ///     propagation primitive. It is used as a hint for deciding which
        ///     memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                       const pooling_v2_forward::primitive_desc &hint_fwd_pd,
                       bool allow_empty = false)
            : zendnn::primitive_desc(&adesc.data, nullptr, aengine,
                                     hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a pooling v2
        /// (dilated pooling) backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a pooling backward propagation primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a pooling forward
        ///     propagation primitive. It is used as a hint for deciding which
        ///     memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                       const engine &aengine,
                       const pooling_v2_forward::primitive_desc &hint_fwd_pd,
                       bool allow_empty = false)
            : zendnn::primitive_desc(&adesc.data, &attr, aengine,
                                     hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a pooling v2
        /// (dilated pooling) backward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a pooling backward
        ///     propagation primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : zendnn::primitive_desc(pd, zendnn::primitive::kind::pooling_v2,
                                     zendnn::prop_kind::backward_data) {}

        /// @copydoc zendnn::primitive_desc_base::src_desc()const
        memory::desc diff_src_desc() const {
            return base::diff_src_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::diff_dst_desc()const
        memory::desc diff_dst_desc() const {
            return base::diff_dst_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const {
            return base::workspace_desc();
        }
    };

    /// Default constructor. Produces an empty object.
    pooling_v2_backward() = default;

    /// Constructs a pooling v2 (dilated pooling) backward
    /// propagation primitive.
    /// @param pd Primitive descriptor for a pooling backward propagation
    ///     primitive.
    pooling_v2_backward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a pooling v2 (dilated pooling) backward
    /// propagation primitive from a cache blob.
    /// @param pd Primitive descriptor for a pooling backward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    pooling_v2_backward(
        const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// @} zendnn_api_pooling_v2

/// @addtogroup zendnn_api_prelu PReLU
///
/// PReLU primitive
/// A primitive to perform PReLU (leaky ReLU with trainable alpha parameter)
///
/// @sa @ref dev_guide_prelu in developer guide
///
/// @{

/// PReLU forward propagation primitive.
struct prelu_forward : public primitive {
    /// Descriptor for a PReLU forward propagation primitive.
    struct desc {
        zendnn_prelu_desc_t data;

        /// Constructs a descriptor for a PReLU forward propagation
        /// primitive.
        ///
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #zendnn::prop_kind::forward_training, and
        ///     #zendnn::prop_kind::forward_inference.
        /// @param data_desc Source and destination memory descriptors.
        /// @param weight_desc Alpha parameters memory descriptor.
        desc(prop_kind aprop_kind, const memory::desc &data_desc,
             const memory::desc &weight_desc) {
            error::wrap_c_api(zendnn_prelu_forward_desc_init(&data,
                              zendnn::convert_to_c(aprop_kind),
                              &data_desc.data, &weight_desc.data),
                              "could not create a descriptor for a prelu forward "
                              "propagation primitive");
        }
    };

    /// Primitive descriptor for a PReLU forward propagation primitive.
    struct primitive_desc : public zendnn::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a PReLU forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a PReLU forward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                       bool allow_empty = false)
            : zendnn::primitive_desc(
                  &adesc.data, nullptr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a PReLU forward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a PReLU forward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param attr Primitive attributes to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                       const engine &aengine, bool allow_empty = false)
            : zendnn::primitive_desc(
                  &adesc.data, &attr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a prelu forward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a prelu forward
        ///     propagation primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : zendnn::primitive_desc(pd, zendnn::primitive::kind::prelu,
                                     zendnn::prop_kind::forward_training,
                                     zendnn::prop_kind::forward_inference) {}

        /// @copydoc zendnn::primitive_desc_base::src_desc()const
        memory::desc src_desc() const {
            return base::src_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const {
            return base::dst_desc(0);
        }
    };

    /// Default constructor. Produces an empty object.
    prelu_forward() = default;

    /// Constructs a prelu forward propagation primitive.
    /// @param pd Primitive descriptor for a prelu forward propagation
    ///     primitive.
    prelu_forward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a prelu forward propagation primitive from a cache blob.
    /// @param pd Primitive descriptor for a prelu forward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    prelu_forward(
        const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// PReLU backward propagation primitive.
struct prelu_backward : public primitive {
    /// Descriptor for a PReLU backward propagation primitive.
    struct desc {
        zendnn_prelu_desc_t data;

        /// Constructs a descriptor for a PReLU backward propagation
        /// primitive.
        ///
        /// @param data_desc Source and destination memory descriptors.
        /// @param weight_desc Alpha parameters memory descriptor.
        /// @param diff_data_desc Diff source and destination memory
        ///     descriptors.
        /// @param diff_weights_desc Diff alpha parameters memory descriptor.
        desc(const memory::desc &data_desc, const memory::desc &weight_desc,
             const memory::desc &diff_data_desc,
             const memory::desc &diff_weights_desc) {
            error::wrap_c_api(
                zendnn_prelu_backward_desc_init(&data, &data_desc.data,
                                                &weight_desc.data, &diff_data_desc.data,
                                                &diff_weights_desc.data),
                "could not create a descriptor for a prelu backward "
                "propagation primitive");
        }
    };

    /// Primitive descriptor for prelu backward propagation.
    struct primitive_desc : public zendnn::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a PReLU backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a PReLU backward propagation
        ///     primitive.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a PReLU forward
        ///     propagation primitive. It is used as a hint for deciding which
        ///     memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                       const prelu_forward::primitive_desc &hint_fwd_pd,
                       bool allow_empty = false)
            : zendnn::primitive_desc(&adesc.data, nullptr, aengine,
                                     hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a PReLU backward
        /// propagation primitive.
        ///
        /// @param adesc Descriptor for a PReLU backward propagation
        ///     primitive.
        /// @param attr Primitive attributes to use.
        /// @param aengine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a PReLU forward
        ///     propagation primitive. It is used as a hint for deciding which
        ///     memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                       const engine &aengine,
                       const prelu_forward::primitive_desc &hint_fwd_pd,
                       bool allow_empty = false)
            : zendnn::primitive_desc(&adesc.data, &attr, aengine,
                                     hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a prelu backward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a prelu backward
        ///     propagation primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : zendnn::primitive_desc(pd, zendnn::primitive::kind::prelu,
                                     zendnn::prop_kind::backward_data) {}

        /// @copydoc zendnn::primitive_desc_base::src_desc()const
        memory::desc src_desc() const {
            return base::src_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::diff_src_desc()const
        memory::desc diff_src_desc() const {
            return base::diff_src_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::diff_dst_desc()const
        memory::desc diff_dst_desc() const {
            return base::diff_dst_desc(0);
        }
    };

    /// Default constructor. Produces an empty object.
    prelu_backward() = default;

    /// Constructs a prelu backward propagation primitive.
    /// @param pd Primitive descriptor for a prelu backward propagation
    ///     primitive.
    prelu_backward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a prelu backward propagation primitive from a cache blob.
    /// @param pd Primitive descriptor for a prelu backward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    prelu_backward(
        const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// @} zendnn_api_prelu

/// @addtogroup zendnn_api_reduction Reduction
///
/// A primitive to compute reduction operation on data tensor
/// using min, max, mul, sum, mean and norm_lp operations.
///
/// @sa @ref dev_guide_reduction in developer guide
///
/// @{

/// Reduction.
struct reduction : public primitive {
    /// Descriptor for reduction.
    struct desc {
        zendnn_reduction_desc_t data;

        /// Default constructor. Produces an empty object.
        desc() = default;

        /// Constructs a descriptor for a reduction primitive using algorithm
        /// specific parameters, source and destination memory descriptors.
        ///
        /// @note
        ///     Destination memory descriptor may be initialized with
        ///     #zendnn::memory::format_tag::any value of @p format_tag.
        ///
        /// @param aalgorithm reduction algorithm kind. Possible values:
        ///     #zendnn_reduction_max, #zendnn_reduction_min, #zendnn_reduction_sum,
        ///     #zendnn_reduction_mul, #zendnn_reduction_mean,
        ///     #zendnn_reduction_norm_lp_max, #zendnn_reduction_norm_lp_sum,
        ///     #zendnn_reduction_norm_lp_power_p_max,
        ///     #zendnn_reduction_norm_lp_power_p_sum.
        /// @param p algorithm specific parameter.
        /// @param eps algorithm specific parameter.
        /// @param src_desc Source memory descriptor.
        /// @param dst_desc Destination memory descriptor.
        desc(algorithm aalgorithm, const memory::desc &src_desc,
             const memory::desc &dst_desc, float p, float eps) {
            error::wrap_c_api(
                zendnn_reduction_desc_init(&data, convert_to_c(aalgorithm),
                                           &src_desc.data, &dst_desc.data, p, eps),
                "could not create a reduction descriptor");
        }
    };

    /// Primitive descriptor for a reduction primitive.
    struct primitive_desc : public zendnn::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a reduction primitive.
        ///
        /// @param adesc Descriptor for a reduction primitive.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                       bool allow_empty = false)
            : zendnn::primitive_desc(
                  &adesc.data, nullptr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a reduction primitive.
        ///
        /// @param adesc Descriptor for a reduction primitive.
        /// @param aengine Engine to use.
        /// @param attr Primitive attributes to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                       const engine &aengine, bool allow_empty = false)
            : zendnn::primitive_desc(
                  &adesc.data, &attr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a reduction primitive from a C
        /// API primitive descriptor that must have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a reduction primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : zendnn::primitive_desc(pd, zendnn::primitive::kind::reduction) {}

        /// @copydoc zendnn::primitive_desc_base::src_desc()const
        memory::desc src_desc() const {
            return base::src_desc(0);
        }

        /// @copydoc zendnn::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const {
            return base::dst_desc(0);
        }
    };

    /// Default constructor. Produces an empty object.
    reduction() = default;

    /// Constructs a reduction primitive.
    /// @param pd Primitive descriptor for a reduction primitive.
    reduction(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a reduction primitive from a cache blob.
    /// @param pd Primitive descriptor for a reduction primitive.
    /// @param cache_blob Cache blob.
    reduction(const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// @} zendnn_api_reduction

/* add new primitive */
/// @addtogroup zendnn_api_embedding_bag EmbeddingBag
///
/// A primitive to get embeding_bag using sum, mean and max operations.
///
/// @sa @ref dev_guide_embedding_bag in developer guide
///
/// @{

/// EmbeddingBag.
struct embedding_bag : public primitive {
    /// Descriptor for embedding_bag.
    struct desc {
        zendnn_embedding_bag_desc_t data;

        /// Default constructor. Produces an empty object.
        desc() = default;

        /// Constructs a descriptor for an embedding_bag primitive using
        /// algorithm specific parameters, source and destination memory
        /// descriptors.
        ///
        /// @note
        ///     Destination memory descriptor may be initialized with
        ///     #zendnn::memory::format_tag::any value of @p format_tag.
        ///     The primitive does not allocate memory for output and
        ///     it should be pre-allocated before calling the primitive.
        ///
        /// @param aprop_kind possible value forward_inference
        /// @param aalgorithm embedding_mag algorithm kind. Possible values:
        ///     embedding_bag_max, embedding_bag_sum or embedding_bag_mean,
        /// @param num_threads Parallel threads for the primitive
        /// (zero for default omp threads)
        /// @param input_desc Input (embedding table) memory descriptor.
        /// @param indices_desc Indices memory descriptor.
        /// @param offsets_desc Offsets memory descriptor.
        /// @param weights_desc Weights memory descriptor.
        /// @param padding_idx Padding Index.
        desc(prop_kind aprop_kind, algorithm aalgorithm,
             uint32_t num_threads,
             const memory::desc &input_desc,
             const memory::desc &indices_desc,
             const memory::desc &offsets_desc,
             const memory::desc &weights_desc,
             const memory::desc &dst_desc,
             int32_t            padding_idx,
             uint32_t           scatter_stride,
             uint32_t           scatter_offset) {
            error::wrap_c_api(
                zendnn_embedding_bag_desc_init(&data,
                                               convert_to_c(aprop_kind),
                                               convert_to_c(aalgorithm),
                                               num_threads,
                                               &input_desc.data,
                                               &indices_desc.data,
                                               &offsets_desc.data,
                                               &weights_desc.data,
                                               &dst_desc.data,
                                               padding_idx,
                                               scatter_stride,
                                               scatter_offset),
                "could not create an embedding_bag descriptor:1");
        }

        /// Constructs a descriptor for an embedding_bag primitive using
        /// algorithm specific parameters, source and destination memory
        /// descriptors.
        ///
        /// @note
        ///     Destination memory descriptor may be initialized with
        ///     #zendnn::memory::format_tag::any value of @p format_tag.
        ///     The primitive does not allocate memory for output and
        ///     it should be pre-allocated before calling the primitive.
        ///
        /// @param aprop_kind possible value forward_inference
        /// @param aalgorithm embedding_mag algorithm kind. Possible values:
        ///     embedding_bag_max, embedding_bag_sum or embedding_bag_mean,
        /// @param num_threads Parallel threads for the primitive
        /// (zero for default omp threads)
        /// @param input_desc Input (embedding table) memory descriptor.
        /// @param indices_desc Indices memory descriptor.
        /// @param offsets_desc Offsets memory descriptor.
        /// @param weights_desc Weights memory descriptor.
        /// @param padding_idx Padding Index.
        desc(prop_kind aprop_kind, algorithm aalgorithm,
             uint32_t num_threads,
             const memory::desc &input_desc,
             const memory::desc &indices_desc,
             const memory::desc &offsets_desc,
             const memory::desc &dst_desc,
             int32_t            padding_idx,
             uint32_t           scatter_stride,
             uint32_t           scatter_offset) {
            error::wrap_c_api(
                zendnn_embedding_bag_desc_init(&data,
                                               convert_to_c(aprop_kind),
                                               convert_to_c(aalgorithm),
                                               num_threads,
                                               &input_desc.data,
                                               &indices_desc.data,
                                               &offsets_desc.data,
                                               nullptr,
                                               &dst_desc.data,
                                               padding_idx,
                                               scatter_stride,
                                               scatter_offset),
                "could not create an embedding_bag descriptor:1");
        }

        /// Constructs a descriptor for an embedding_bag primitive using
        /// algorithm specific parameters, source and destination memory
        /// descriptors.
        ///
        /// @note
        ///     Destination memory descriptor may be initialized with
        ///     #zendnn::memory::format_tag::any value of @p format_tag.
        ///     The primitive does not allocate memory for output and
        ///     it should be pre-allocated before calling the primitive.
        ///
        /// @param aprop_kind possible value forward_inference
        /// @param aalgorithm embedding_mag algorithm kind. Possible values:
        ///     embedding_bag_max, embedding_bag_sum or embedding_bag_mean,
        /// @param num_threads Parallel threads for the primitive
        /// (zero for default omp threads)
        /// @param input_desc Input (embedding table) memory descriptor.
        /// @param indices_desc Indices memory descriptor.
        /// @param offsets_desc Offsets memory descriptor.
        /// @param weights_desc Weights memory descriptor.
        /// @param padding_idx Padding Index.
        desc(prop_kind aprop_kind, algorithm aalgorithm,
             uint32_t num_threads,
             const memory::desc &input_desc,
             const memory::desc &indices_desc,
             const memory::desc &offsets_desc,
             const memory::desc &weights_desc,
             const memory::desc &dst_desc,
             int32_t            padding_idx) {
            error::wrap_c_api(
                zendnn_embedding_bag_desc_init(&data,
                                               convert_to_c(aprop_kind),
                                               convert_to_c(aalgorithm),
                                               num_threads,
                                               &input_desc.data,
                                               &indices_desc.data,
                                               &offsets_desc.data,
                                               &weights_desc.data,
                                               &dst_desc.data,
                                               padding_idx,
                                               1,0),
                "could not create an embedding_bag descriptor:2");
        }

        /// Constructs a descriptor for an embedding_bag primitive using
        /// algorithm specific parameters, source and destination memory
        /// descriptors.
        ///
        /// @note
        ///     Destination memory descriptor may be initialized with
        ///     #zendnn::memory::format_tag::any value of @p format_tag.
        ///     The primitive does not allocate memory for output and
        ///     it should be pre-allocated before calling the primitive.
        ///
        /// @param aprop_kind possible value forward_inference
        /// @param aalgorithm embedding_mag algorithm kind. Possible values:
        ///     embedding_bag_max, embedding_bag_sum or embedding_bag_mean,
        /// @param num_threads Parallel threads for the primitive
        /// (zero for default  omp threads)
        /// @param input_desc Input (embedding table) memory descriptor.
        /// @param indices_desc Indices memory descriptor.
        /// @param offsets_desc Offsets memory descriptor.
        /// @param padding_idx Padding Index.
        desc(prop_kind aprop_kind, algorithm aalgorithm,
             uint32_t num_threads,
             const memory::desc &input_desc,
             const memory::desc &indices_desc,
             const memory::desc &offsets_desc,
             const memory::desc &dst_desc,
             int32_t            padding_idx) {
            error::wrap_c_api(
                zendnn_embedding_bag_desc_init(&data,
                                               convert_to_c(aprop_kind),
                                               convert_to_c(aalgorithm),
                                               num_threads,
                                               &input_desc.data,
                                               &indices_desc.data,
                                               &offsets_desc.data,
                                               nullptr,
                                               &dst_desc.data,
                                               padding_idx,
                                               1,0),
                "could not create an embedding_bag descriptor:3");
        }

        /// Constructs a descriptor for an embedding_bag primitive using
        /// algorithm specific parameters, source and destination memory
        /// descriptors.
        ///
        /// @note
        ///     Destination memory descriptor may be initialized with
        ///     #zendnn::memory::format_tag::any value of @p format_tag.
        ///     The primitive does not allocate memory for output and
        ///     it should be pre-allocated before calling the primitive.
        ///
        /// @param aprop_kind possible value forward_inference
        /// @param aalgorithm embedding_mag algorithm kind. Possible values:
        ///     embedding_bag_max, embedding_bag_sum or embedding_bag_mean,
        /// @param num_threads Parallel threads for the primitive
        ///     (zero for default omp threads)
        /// @param input_desc Input (embedding table) memory descriptor.
        /// @param indices_desc Indices memory descriptor.
        /// @param offsets_desc Offsets memory descriptor.
        /// @param weights_desc Weights memory descriptor. This can be omitted
        ///     if there are no weights.
        desc(prop_kind aprop_kind, algorithm aalgorithm,
             uint32_t num_threads,
             const memory::desc &input_desc,
             const memory::desc &indices_desc,
             const memory::desc &offsets_desc,
             const memory::desc &weights_desc,
             const memory::desc &dst_desc) {
            error::wrap_c_api(
                zendnn_embedding_bag_desc_init(&data,
                                               convert_to_c(aprop_kind),
                                               convert_to_c(aalgorithm),
                                               num_threads,
                                               &input_desc.data,
                                               &indices_desc.data,
                                               &offsets_desc.data,
                                               &weights_desc.data,
                                               &dst_desc.data,
                                               -1,1,0),
                "could not create an embedding_bag descriptor:4");
        }

        /// Constructs a descriptor for an embedding_bag primitive using
        /// algorithm specific parameters, source and destination memory
        /// descriptors.
        ///
        /// @note
        ///     Destination memory descriptor may be initialized with
        ///     #zendnn::memory::format_tag::any value of @p format_tag.
        ///     The primitive does not allocate memory for output and
        ///     it should be pre-allocated before calling the primitive.
        ///
        /// @param aprop_kind possible value forward_inference
        /// @param aalgorithm embedding_mag algorithm kind. Possible values:
        ///     embedding_bag_max, embedding_bag_sum or embedding_bag_mean,
        /// @param num_threads Parallel threads for the primitive
        ///     (zero for default omp threads)
        /// @param input_desc Input (embedding table) memory descriptor.
        /// @param indices_desc Indices memory descriptor.
        /// @param offsets_desc Offsets memory descriptor.
        desc(prop_kind aprop_kind, algorithm aalgorithm,
             uint32_t num_threads,
             const memory::desc &input_desc,
             const memory::desc &indices_desc,
             const memory::desc &offsets_desc,
             const memory::desc &dst_desc) {
            error::wrap_c_api(
                zendnn_embedding_bag_desc_init(&data,
                                               convert_to_c(aprop_kind),
                                               convert_to_c(aalgorithm),
                                               num_threads,
                                               &input_desc.data,
                                               &indices_desc.data,
                                               &offsets_desc.data,
                                               nullptr,
                                               &dst_desc.data,
                                               -1,1,0),
                "could not create an embedding_bag descriptor:5");

        }
    };

    /// Primitive descriptor for an embedding_bag primitive.
    struct primitive_desc : public zendnn::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an embedding_bag primitive.
        ///
        /// @param adesc Descriptor for an embedding_bag primitive.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const engine &aengine,
                       bool allow_empty = false)
            : zendnn::primitive_desc(
                  &adesc.data, nullptr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for an embedding_bag primitive.
        ///
        /// @param adesc Descriptor for an embedding_bag primitive.
        /// @param aengine Engine to use.
        /// @param attr Primitive attributes to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                       const engine &aengine, bool allow_empty = false)
            : zendnn::primitive_desc(
                  &adesc.data, &attr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for an embedding_bag primitive
        /// from a C API primitive descriptor that must have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a embedding_bag primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : zendnn::primitive_desc(pd, zendnn::primitive::kind::embedding_bag) {}

    };

    /// Default constructor. Produces an empty object.
    embedding_bag() = default;

    /// Constructs an embedding_bag primitive.
    /// @param pd Primitive descriptor for an embedding_bag primitive.
    embedding_bag(const primitive_desc &pd) : primitive(pd) {}
};

/// @} zendnn_api_embedding_bag

/* add new primitive */
/// @addtogroup zendnn_api_attention Attention
///
/// A primitive to perform Attention operation.
///
/// @sa @ref dev_guide_attention in developer guide
///
/// @{

/// Attention.
struct attention : public primitive {
    /// Descriptor for attention.
    struct desc {
        zendnn_attention_desc_t data;

        /// Default constructor. Produces an empty object.
        desc() = default;

        /// Constructs a descriptor for an attention primitive using
        /// algorithm specific parameters, source and destination memory
        /// descriptors.
        ///
        /// @note
        ///     Destination memory descriptor may be initialized with
        ///     #zendnn::memory::format_tag::any value of @p format_tag.
        ///     The primitive does not allocate memory for output and
        ///     it should be pre-allocated before calling the primitive.
        ///
        /// @param aprop_kind possible value forward_inference
        /// @param aalgorithm attention algorithm kind. Possible values:
        ///     #zendnn_multihead_attention, #zendnn_multihead_attention_flash_v1, #zendnn_multihead_attention_flash_v2
        ///     #zendnn_multiquery_attention
        ///     #zendnn_groupedquery_attention
        /// @param query_desc Input memory descriptor.
        /// @param key_desc Input memory descriptor.
        /// @param value_desc Input memory descriptor.
        /// @param weights_query_desc Input memory descriptor.
        /// @param weights_key_desc Input memory descriptor.
        /// @param weights_value_desc Input memory descriptor.
        /// @param bias_query_desc Input memory descriptor.
        /// @param bias_key_desc Input memory descriptor.
        /// @param bias_value_desc Input memory descriptor.
        /// @param mask_desc Input memory descriptor.
        /// @param dst_desc Destination memory descriptor.
        /// @param scale Scale factor. sqrt(head_size) or any custom float.
        /// @param num_heads Number of heads.
        /// @param num_threads Parallel threads for the primitive (zero for default
        ///              omp threads)
        desc(prop_kind aprop_kind, algorithm aalgorithm,
             const memory::desc &query_desc,
             const memory::desc &key_desc,
             const memory::desc &value_desc,
             const memory::desc &weights_query_desc,
             const memory::desc &weights_key_desc,
             const memory::desc &weights_value_desc,
             const memory::desc &bias_query_desc,
             const memory::desc &bias_key_desc,
             const memory::desc &bias_value_desc,
             const memory::desc &mask_desc,
             const memory::desc &dst_desc,
             float scale,
             uint32_t           num_heads,
             uint32_t           num_threads) {
            error::wrap_c_api(
                zendnn_attention_desc_init(&data,
                                           convert_to_c(aprop_kind),
                                           convert_to_c(aalgorithm),
                                           &query_desc.data,
                                           &key_desc.data,
                                           &value_desc.data,
                                           &weights_query_desc.data,
                                           &weights_key_desc.data,
                                           &weights_value_desc.data,
                                           &bias_query_desc.data,
                                           &bias_key_desc.data,
                                           &bias_value_desc.data,
                                           &mask_desc.data,
                                           &dst_desc.data,
                                           scale,
                                           num_heads,
                                           num_threads),
                "could not create an attention descriptor:1");
        }
    };

    /// Primitive descriptor for an attention primitive.
    struct primitive_desc : public zendnn::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an attention primitive.
        ///
        /// @param adesc Descriptor for an attention primitive.
        /// @param aengine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const desc &adesc, const primitive_attr &attr,
                       const engine &aengine,
                       bool allow_empty = false)
            : zendnn::primitive_desc(
                  &adesc.data, &attr, aengine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for an attention primitive
        /// from a C API primitive descriptor that must have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a attention primitive.
        primitive_desc(zendnn_primitive_desc_t pd)
            : zendnn::primitive_desc(pd, zendnn::primitive::kind::attention) {}

        /// @copydoc zendnn::primitive_desc_base::src_desc()const
        memory::desc src_desc() const {
            return query_md(query::src_md, 0);
        }

        /// @copydoc zendnn::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const {
            return query_md(query::dst_md, 0);
        }
    };

    /// Default constructor. Produces an empty object.
    attention() = default;

    /// Constructs an attention primitive.
    /// @param pd Primitive descriptor for an attention primitive.
    attention(const primitive_desc &pd) : primitive(pd) {}
};

/// @} zendnn_api_attention

/// @} zendnn_api_primitives

/// @addtogroup zendnn_api_service Service
///
/// A set of functions that aid in ZENDNN debugging and profiling.
///
/// @{

/// @copydoc zendnn_version_t
using version_t = zendnn_version_t;

/// Status values returned by the library functions.
enum class status {
    /// @copydoc zendnn_success
    success = zendnn_success,
    /// @copydoc zendnn_out_of_memory
    out_of_memory = zendnn_out_of_memory,
    /// @copydoc zendnn_invalid_arguments
    invalid_arguments = zendnn_invalid_arguments,
    /// @copydoc zendnn_unimplemented
    unimplemented = zendnn_unimplemented,
    /// @copydoc zendnn_iterator_ends
    iterator_ends = zendnn_iterator_ends,
    /// @copydoc zendnn_runtime_error
    runtime_error = zendnn_runtime_error,
    /// @copydoc zendnn_not_required
    not_required = zendnn_not_required,
};

/// @copydoc zendnn_set_verbose()
inline status set_verbose(int level) {
    return static_cast<status>(zendnn_set_verbose(level));
}

/// @copydoc zendnn_version()
inline const version_t *version() {
    return zendnn_version();
}

/// Returns the floating-point math mode that will be used by default
/// for all subsequently created primitives.
///
/// @returns Output FP math mode.
inline fpmath_mode get_default_fpmath_mode() {
    zendnn_fpmath_mode_t mode;
    error::wrap_c_api(zendnn_get_default_fpmath_mode(&mode),
                      "could not get a default fpmath mode");
    return static_cast<fpmath_mode>(mode);
}

/// @copydoc zendnn_set_default_fpmath_mode()
inline status set_default_fpmath_mode(fpmath_mode mode) {
    return static_cast<status>(
               zendnn_set_default_fpmath_mode(convert_to_c(mode)));
}

/// @copydoc zendnn_set_jit_dump()
inline status set_jit_dump(int enable) {
    return static_cast<status>(zendnn_set_jit_dump(enable));
}

/// @copydoc zendnn_set_jit_profiling_flags()
inline status set_jit_profiling_flags(unsigned flags) {
    return static_cast<status>(zendnn_set_jit_profiling_flags(flags));
}

/// @copydoc zendnn_set_jit_profiling_jitdumpdir()
inline status set_jit_profiling_jitdumpdir(const std::string &dir) {
    return static_cast<status>(zendnn_set_jit_profiling_jitdumpdir(dir.c_str()));
}

/// @copydoc zendnn_cpu_isa_t
enum class cpu_isa {
    /// @copydoc zendnn_cpu_isa_all
    all = zendnn_cpu_isa_all,
    /// @copydoc zendnn_cpu_isa_sse41
    sse41 = zendnn_cpu_isa_sse41,
    /// @copydoc zendnn_cpu_isa_avx
    avx = zendnn_cpu_isa_avx,
    /// @copydoc zendnn_cpu_isa_avx2
    avx2 = zendnn_cpu_isa_avx2,
    /// @copydoc zendnn_cpu_isa_avx512_mic
    avx512_mic = zendnn_cpu_isa_avx512_mic,
    /// @copydoc zendnn_cpu_isa_avx512_mic_4ops
    avx512_mic_4ops = zendnn_cpu_isa_avx512_mic_4ops,
    /// @copydoc zendnn_cpu_isa_avx512_core
    avx512_core = zendnn_cpu_isa_avx512_core,
    /// @copydoc zendnn_cpu_isa_avx512_core_vnni
    avx512_core_vnni = zendnn_cpu_isa_avx512_core_vnni,
    /// @copydoc zendnn_cpu_isa_avx512_core_bf16
    avx512_core_bf16 = zendnn_cpu_isa_avx512_core_bf16,
    /// @copydoc zendnn_cpu_isa_avx512_core_amx
    avx512_core_amx = zendnn_cpu_isa_avx512_core_amx,
    /// @copydoc zendnn_cpu_isa_avx2_vnni
    avx2_vnni = zendnn_cpu_isa_avx2_vnni,
};

/// @copydoc zendnn_set_max_cpu_isa()
inline status set_max_cpu_isa(cpu_isa isa) {
    return static_cast<status>(
               zendnn_set_max_cpu_isa(static_cast<zendnn_cpu_isa_t>(isa)));
}

/// @copydoc zendnn_get_effective_cpu_isa()
inline cpu_isa get_effective_cpu_isa() {
    return static_cast<cpu_isa>(zendnn_get_effective_cpu_isa());
}

/// @copydoc zendnn_cpu_isa_hints_t
enum class cpu_isa_hints {
    /// @copydoc zendnn_cpu_isa_no_hints
    no_hints = zendnn_cpu_isa_no_hints,
    /// @copydoc zendnn_cpu_isa_prefer_ymm
    prefer_ymm = zendnn_cpu_isa_prefer_ymm,
};

/// @copydoc zendnn_set_cpu_isa_hints()
inline status set_cpu_isa_hints(cpu_isa_hints isa_hints) {
    return static_cast<status>(zendnn_set_cpu_isa_hints(
                                   static_cast<zendnn_cpu_isa_hints_t>(isa_hints)));
}

/// @copydoc zendnn_get_cpu_isa_hints()
inline cpu_isa_hints get_cpu_isa_hints() {
    return static_cast<cpu_isa_hints>(zendnn_get_cpu_isa_hints());
}

/// @} zendnn_api_service

/// @addtogroup zendnn_api_primitive_cache Primitive Cache
///
/// A set of functions that provide primitive cache control.
///
/// @{

/// Returns the number of primitives that can be held in the primitive cache
/// at the same time.
inline int get_primitive_cache_capacity() {
    int result = 0;
    error::wrap_c_api(zendnn_get_primitive_cache_capacity(&result),
                      "could not get primitive cache capacity");
    return result;
}

/// @copydoc zendnn_set_primitive_cache_capacity(int capacity)
inline void set_primitive_cache_capacity(int capacity) {
    error::wrap_c_api(zendnn_set_primitive_cache_capacity(capacity),
                      "could not set primitive cache capacity");
}

/// @} zendnn_api_primitive_cache

/// @addtogroup zendnn_api_blas BLAS functions
///
/// A subset of Basic Linear Algebra (BLAS) functions that perform
/// matrix-matrix multiplication.
///
/// @{

/// @copydoc zendnn_sgemm()
inline status sgemm(char transa, char transb, zendnn_dim_t M, zendnn_dim_t N,
                    zendnn_dim_t K, float alpha, const float *A, zendnn_dim_t lda,
                    const float *B, zendnn_dim_t ldb, float beta, float *C, zendnn_dim_t ldc) {
    return static_cast<status>(zendnn_sgemm(
                                   transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc));
}

/// @copydoc zendnn_gemm_u8s8s32()
inline status gemm_u8s8s32(char transa, char transb, char offsetc,
                           zendnn_dim_t M,
                           zendnn_dim_t N, zendnn_dim_t K, float alpha, const uint8_t *A,
                           zendnn_dim_t lda, uint8_t ao, const int8_t *B, zendnn_dim_t ldb, int8_t bo,
                           float beta, int32_t *C, zendnn_dim_t ldc, const int32_t *co) {
    return static_cast<status>(zendnn_gemm_u8s8s32(transa, transb, offsetc, M, N,
                               K, alpha, A, lda, ao, B, ldb, bo, beta, C, ldc, co));
}

/// @copydoc zendnn_gemm_s8s8s32()
inline status gemm_s8s8s32(char transa, char transb, char offsetc,
                           zendnn_dim_t M,
                           zendnn_dim_t N, zendnn_dim_t K, float alpha, const int8_t *A,
                           zendnn_dim_t lda, int8_t ao, const int8_t *B, zendnn_dim_t ldb, int8_t bo,
                           float beta, int32_t *C, zendnn_dim_t ldc, const int32_t *co) {
    return static_cast<status>(zendnn_gemm_s8s8s32(transa, transb, offsetc, M, N,
                               K, alpha, A, lda, ao, B, ldb, bo, beta, C, ldc, co));
}

/// @} zendnn_api_blas

// implementation section

/// @cond DO_NOT_DOCUMENT_THIS
inline primitive::primitive(const_zendnn_primitive_desc_t c_pd) {
    zendnn_primitive_t result;
    error::wrap_c_api(zendnn_primitive_create(&result, c_pd),
                      "could not create a primitive");
    reset(result);
}

inline primitive::primitive(const_zendnn_primitive_desc_t c_pd,
                            const std::vector<uint8_t> &cache_blob) {
    zendnn_primitive_t result;
    size_t size = cache_blob.size();
    const uint8_t *cache_blob_data = cache_blob.data();
    error::wrap_c_api(zendnn_primitive_create_from_cache_blob(
                          &result, c_pd, size, cache_blob_data),
                      "could not create a primitive from a cache blob");
    reset(result);
}

inline primitive::primitive(const primitive_desc &pd) : primitive(pd.get()) {}
inline primitive::primitive(
    const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
    : primitive(pd.get(), cache_blob) {}

inline void primitive::execute(const stream &astream,
                               const std::unordered_map<int, memory> &args) const {
    std::vector<zendnn_exec_arg_t> c_args;
    c_args.reserve(args.size());
    for (const auto &a : args)
        c_args.push_back({a.first, a.second.get(true)});

    error::wrap_c_api(zendnn_primitive_execute(get(), astream.get(),
                      (int)c_args.size(), c_args.data()),
                      "could not execute a primitive");
}



/// @endcond

#undef ZENDNN_DEFINE_BITMASK_OPS

} // namespace zendnn

/// @} zendnn_api

#endif /* ZENDNN_HPP */
