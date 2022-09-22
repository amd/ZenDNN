/*******************************************************************************
* Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
* Notified per clause 4(b) of the license.
*******************************************************************************/

/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#ifndef ZENDNN_OCL_HPP
#define ZENDNN_OCL_HPP

#include "zendnn.hpp"

/// @cond DO_NOT_DOCUMENT_THIS
#include <algorithm>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "zendnn_ocl.h"

#include <CL/cl.h>
/// @endcond

/// @addtogroup zendnn_api
/// @{

namespace zendnn {

/// @addtogroup zendnn_api_interop Runtime interoperability API
/// API extensions to interact with the underlying run-time.
/// @{

/// @addtogroup zendnn_api_ocl_interop OpenCL interoperability API
/// API extensions to interact with the underlying OpenCL run-time.
///
/// @sa @ref dev_guide_opencl_interoperability in developer guide
/// @{

/// OpenCL interoperability namespace
namespace ocl_interop {

/// Memory allocation kind.
enum class memory_kind {
    /// USM (device, shared, host, or unknown) memory allocation kind.
    usm = zendnn_ocl_interop_usm,
    /// Buffer memory allocation kind - default.
    buffer = zendnn_ocl_interop_buffer,
};

/// Converts a memory allocation kind enum value from C++ API to C API type.
///
/// @param akind C++ API memory allocation kind enum value.
/// @returns Corresponding C API memory allocation kind enum value.
inline zendnn_ocl_interop_memory_kind_t convert_to_c(memory_kind akind) {
    return static_cast<zendnn_ocl_interop_memory_kind_t>(akind);
}

/// Constructs an engine from OpenCL device and context objects.
///
/// @param device The OpenCL device that this engine will encapsulate.
/// @param context The OpenCL context (containing the device) that this
///     engine will use for all operations.
/// @returns An engine.
inline engine make_engine(cl_device_id device, cl_context context) {
    zendnn_engine_t c_engine;
    error::wrap_c_api(
            zendnn_ocl_interop_engine_create(&c_engine, device, context),
            "could not create an engine");
    return engine(c_engine);
}

/// Returns OpenCL context associated with the engine.
///
/// @param aengine An engine.
/// @returns Underlying OpenCL context.
inline cl_context get_context(const engine &aengine) {
    cl_context context = nullptr;
    error::wrap_c_api(
            zendnn_ocl_interop_engine_get_context(aengine.get(), &context),
            "could not get an OpenCL context from an engine");
    return context;
}

/// Returns OpenCL device associated with the engine.
///
/// @param aengine An engine.
/// @returns Underlying OpenCL device.
inline cl_device_id get_device(const engine &aengine) {
    cl_device_id device = nullptr;
    error::wrap_c_api(zendnn_ocl_interop_get_device(aengine.get(), &device),
            "could not get an OpenCL device from an engine");
    return device;
}

/// Constructs an execution stream for the specified engine and OpenCL queue.
///
/// @param aengine Engine to create the stream on.
/// @param queue OpenCL queue to use for the stream.
/// @returns An execution stream.
inline stream make_stream(const engine &aengine, cl_command_queue queue) {
    zendnn_stream_t c_stream;
    error::wrap_c_api(
            zendnn_ocl_interop_stream_create(&c_stream, aengine.get(), queue),
            "could not create a stream");
    return stream(c_stream);
}

/// Returns OpenCL queue object associated with the execution stream.
///
/// @param astream An execution stream.
/// @returns Underlying OpenCL queue.
inline cl_command_queue get_command_queue(const stream &astream) {
    cl_command_queue queue = nullptr;
    error::wrap_c_api(
            zendnn_ocl_interop_stream_get_command_queue(astream.get(), &queue),
            "could not get an OpenCL command queue from a stream");
    return queue;
}

/// Returns the OpenCL memory object associated with the memory object.
///
/// @param amemory A memory object.
/// @returns Underlying OpenCL memory object.
inline cl_mem get_mem_object(const memory &amemory) {
    cl_mem mem_object;
    error::wrap_c_api(
            zendnn_ocl_interop_memory_get_mem_object(amemory.get(), &mem_object),
            "could not get OpenCL buffer object from a memory object");
    return mem_object;
}

/// Sets the OpenCL memory object associated with the memory object.
///
/// For behavioral details see memory::set_data_handle().
///
/// @param amemory A memory object.
/// @param mem_object OpenCL cl_mem object to use as the underlying
///     storage. It must have at least get_desc().get_size() bytes
///     allocated.
inline void set_mem_object(memory &amemory, cl_mem mem_object) {
    error::wrap_c_api(
            zendnn_ocl_interop_memory_set_mem_object(amemory.get(), mem_object),
            "could not set OpenCL buffer object from a memory object");
}

/// Returns the memory allocation kind associated with a memory object.
///
/// @param amemory A memory object.
///
/// @returns The underlying memory allocation kind of the memory object.
inline memory_kind get_memory_kind(const memory &amemory) {
    zendnn_ocl_interop_memory_kind_t ckind;
    error::wrap_c_api(
            zendnn_ocl_interop_memory_get_memory_kind(amemory.get(), &ckind),
            "could not get memory kind");
    return static_cast<memory_kind>(ckind);
}

/// Creates a memory object.
///
/// Unless @p handle is equal to ZENDNN_MEMORY_NONE or ZENDNN_MEMORY_ALLOCATE, the
/// constructed memory object will have the underlying buffer set. In this
/// case, the buffer will be initialized as if:
/// - zendnn::memory::set_data_handle() had been called, if @p memory_kind is
///   equal to zendnn::ocl_interop::memory_kind::usm, or
/// - zendnn::ocl_interop::set_mem_object() has been called, if @p memory_kind is
///   equal to zendnn::ocl_interop::memory_kind::buffer.
///
/// @param memory_desc Memory descriptor.
/// @param aengine Engine to use.
/// @param kind Memory allocation kind to specify the type of handle.
/// @param handle Handle of the memory buffer to use as an underlying storage.
///     - A USM pointer to the user-allocated buffer. In this case the library
///       doesn't own the buffer. Requires @p memory_kind to be equal to
///       zendnn::ocl_interop::memory_kind::usm.
///     - An OpenCL buffer. In this case the library doesn't own the buffer.
///       Requires @p memory_kind be equal to be equal to
///       zendnn::ocl_interop::memory_kind::buffer.
///     - The ZENDNN_MEMORY_ALLOCATE special value. Instructs the library to
///       allocate the buffer that corresponds to the memory allocation kind
///       @p memory_kind for the memory object. In this case the library
///       owns the buffer.
///     - The ZENDNN_MEMORY_NONE specific value. Instructs the library to
///       create memory object without an underlying buffer.
///
/// @returns Created memory object.
inline memory make_memory(const memory::desc &memory_desc,
        const engine &aengine, memory_kind kind,
        void *handle = ZENDNN_MEMORY_ALLOCATE) {
    zendnn_memory_t c_memory;
    error::wrap_c_api(
            zendnn_ocl_interop_memory_create(&c_memory, &memory_desc.data,
                    aengine.get(), convert_to_c(kind), handle),
            "could not create a memory");
    return memory(c_memory);
}

/// Constructs a memory object from an OpenCL buffer.
///
/// @param memory_desc Memory descriptor.
/// @param aengine Engine to use.
/// @param mem_object An OpenCL buffer to use.
///
/// @returns Created memory object.
inline memory make_memory(const memory::desc &memory_desc,
        const engine &aengine, cl_mem mem_object) {
    memory amemory(memory_desc, aengine, ZENDNN_MEMORY_NONE);
    set_mem_object(amemory, mem_object);
    return amemory;
}

} // namespace ocl_interop

/// @} zendnn_api_ocl_interop

/// @} zendnn_api_interop

} // namespace zendnn

/// @} zendnn_api

#endif
