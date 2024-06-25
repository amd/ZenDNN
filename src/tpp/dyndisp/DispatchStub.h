/*****************************************************************************
* Modifications Copyright (c) 2024 Advanced Micro Devices, Inc.
* All rights reserved.
* Notified per clause 4(b) of the license.
******************************************************************************/
/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/

#ifndef _DISPATCH_STUD_H_
#define _DISPATCH_STUD_H_

#include <atomic>
#include <type_traits>

// using namespace c10;

// Implements instruction set specific function dispatch.
//
// Kernels that may make use of specialized instruction sets (e.g. AVX2) are
// compiled multiple times with different compiler flags (e.g. -mavx2). A
// DispatchStub contains a table of function pointers for a kernel. At runtime,
// the fastest available kernel is chosen based on the features reported by
// cpuinfo.
//
// Example:
//
// In csrc/cpu/aten/MyKernel.h:
//   using fn_type = void(*)(const Tensor& x);
//   IPEX_DECLARE_DISPATCH(fn_type, stub);
//
// In csrc/cpu/aten/MyKernel.cpp
//   IPEX_DEFINE_DISPATCH(stub);
//
// In csrc/cpu/aten/kernels/MyKernel.cpp:
//   namespace {
//     // use anonymous namespace so that different cpu versions won't conflict
//     void kernel(const Tensor& x) { ... }
//   }
//   IPEX_REGISTER_DISPATCH(stub, &kernel);
//
// To call:
//   stub(kCPU, tensor);
//
// TODO: CPU instruction set selection should be folded into whatever
// the main dispatch mechanism is.

// ignore warnings about DispatchStub::DEFAULT, AVX, AVX2 defined elsewhere
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundefined-var-template"
#endif

namespace zendnn {
namespace tpp {
namespace cpu {

enum class DeviceType : int8_t {
  CPU = 0,
  CUDA = 1, // CUDA.
  MKLDNN = 2, // Reserved for explicit MKLDNN
  OPENGL = 3, // OpenGL
  OPENCL = 4, // OpenCL
  IDEEP = 5, // IDEEP.
  HIP = 6, // AMD HIP
  FPGA = 7, // FPGA
  MAIA = 8, // ONNX Runtime / Microsoft
  XLA = 9, // XLA / TPU
  Vulkan = 10, // Vulkan
  Metal = 11, // Metal
  XPU = 12, // XPU
  MPS = 13, // MPS
  Meta = 14, // Meta (tensors with no data)
  HPU = 15, // HPU / HABANA
  VE = 16, // SX-Aurora / NEC
  Lazy = 17, // Lazy Tensors
  IPU = 18, // Graphcore IPU
  MTIA = 19, // Meta training and inference devices
  PrivateUse1 = 20, // PrivateUse1 device
  // NB: If you add more devices:
  //  - Change the implementations of DeviceTypeName and isValidDeviceType
  //    in DeviceType.cpp
  //  - Change the number below
  COMPILE_TIME_MAX_DEVICE_TYPES = 21,
};

// ISA level number should order by required compiler version.
enum class CPUCapability {
  DEFAULT = 0,
  AVX2 = 1,
  AVX2_VNNI = 2,
  AVX512 = 3,
  AVX512_VNNI = 4, // gcc 9.2+
  AVX512_BF16 = 5, // gcc 10.3+
  AMX = 6, // gcc 11.2+
  AVX512_FP16 = 7, // gcc 12.1+
  NUM_OPTIONS
};

const char* CPUCapabilityToString(CPUCapability isa);
CPUCapability _get_highest_cpu_support_isa_level();
CPUCapability _get_highest_binary_support_isa_level();

bool check_not_sync_onednn_isa_level();

CPUCapability get_cpu_capability();

template <typename FnPtr, typename T>
struct DispatchStub;

/**
 * The sole purpose of this class is to outline methods that don't need to be
 * specialized or otherwise inlined and duplicated (by the compiler due to
 * template expansion), since it causes size bloat if there are a significant
 * number of specialization of the DispatchStub<> class.
 */
struct DispatchStubImpl {
  void* get_call_ptr(
      DeviceType device_type,
      void* DEFAULT
#ifdef HAVE_AVX512_FP16_CPU_DEFINITION
      ,
      void* AVX512_FP16
#endif
#ifdef HAVE_AMX_CPU_DEFINITION
      ,
      void* AMX
#endif
#ifdef HAVE_AVX512_BF16_CPU_DEFINITION
      ,
      void* AVX512_BF16
#endif
#ifdef HAVE_AVX512_VNNI_CPU_DEFINITION
      ,
      void* AVX512_VNNI
#endif
#ifdef HAVE_AVX512_CPU_DEFINITION
      ,
      void* AVX512
#endif
#ifdef HAVE_AVX2_VNNI_CPU_DEFINITION
      ,
      void* AVX2_VNNI
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
      ,
      void* AVX2
#endif
  );

  /**
   * The CPU Dispatch actual method is chosen in decreasing order of preference
   * by DispatchStubImpl::choose_cpu_impl() in case none is found by
   * DispatchStubImpl::get_call_ptr() in cpu_dispatch_ptr.
   */
  void* choose_cpu_impl(
      void* DEFAULT
#ifdef HAVE_AVX512_FP16_CPU_DEFINITION
      ,
      void* AVX512_FP16
#endif
#ifdef HAVE_AMX_CPU_DEFINITION
      ,
      void* AMX
#endif
#ifdef HAVE_AVX512_BF16_CPU_DEFINITION
      ,
      void* AVX512_BF16
#endif
#ifdef HAVE_AVX512_VNNI_CPU_DEFINITION
      ,
      void* AVX512_VNNI
#endif
#ifdef HAVE_AVX512_CPU_DEFINITION
      ,
      void* AVX512
#endif
#ifdef HAVE_AVX2_VNNI_CPU_DEFINITION
      ,
      void* AVX2_VNNI
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
      ,
      void* AVX2
#endif
  );

// Fixing dispatch error in Windows debug builds.
// See https://github.com/pytorch/pytorch/issues/22681 for more details.
#if defined(_MSC_VER) && defined(_DEBUG)
  std::atomic<void*> cpu_dispatch_ptr;
  void* xpu_dispatch_ptr;
#else
  std::atomic<void*> cpu_dispatch_ptr{nullptr};
  void* xpu_dispatch_ptr = nullptr;
#endif
};

template <typename rT, typename T, typename... Args>
struct DispatchStub<rT (*)(Args...), T> {
  using FnPtr = rT (*)(Args...);

  DispatchStub() = default;
  DispatchStub(const DispatchStub&) = delete;
  DispatchStub& operator=(const DispatchStub&) = delete;

 private:
  FnPtr get_call_ptr(DeviceType device_type) {
    return reinterpret_cast<FnPtr>(impl.get_call_ptr(
        device_type,
        reinterpret_cast<void*>(DEFAULT)
#ifdef HAVE_AVX512_FP16_CPU_DEFINITION
            ,
        reinterpret_cast<void*>(AVX512_FP16)
#endif
#ifdef HAVE_AMX_CPU_DEFINITION
            ,
        reinterpret_cast<void*>(AMX)
#endif
#ifdef HAVE_AVX512_BF16_CPU_DEFINITION
            ,
        reinterpret_cast<void*>(AVX512_BF16)
#endif
#ifdef HAVE_AVX512_VNNI_CPU_DEFINITION
            ,
        reinterpret_cast<void*>(AVX512_VNNI)
#endif
#ifdef HAVE_AVX512_CPU_DEFINITION
            ,
        reinterpret_cast<void*>(AVX512)
#endif
#ifdef HAVE_AVX2_VNNI_CPU_DEFINITION
            ,
        reinterpret_cast<void*>(AVX2_VNNI)
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
            ,
        reinterpret_cast<void*>(AVX2)
#endif
            ));
  }

 public:
  template <typename... ArgTypes>
  rT operator()(DeviceType device_type, ArgTypes&&... args) {
    FnPtr call_ptr = get_call_ptr(device_type);
    return (*call_ptr)(std::forward<ArgTypes>(args)...);
  }

  void set_xpu_dispatch_ptr(FnPtr fn_ptr) {
    impl.xpu_dispatch_ptr = reinterpret_cast<void*>(fn_ptr);
  }

  static FnPtr DEFAULT;
#ifdef HAVE_AVX512_FP16_CPU_DEFINITION
  static FnPtr AVX512_FP16;
#endif
#ifdef HAVE_AMX_CPU_DEFINITION
  static FnPtr AMX;
#endif
#ifdef HAVE_AVX512_BF16_CPU_DEFINITION
  static FnPtr AVX512_BF16;
#endif
#ifdef HAVE_AVX512_VNNI_CPU_DEFINITION
  static FnPtr AVX512_VNNI;
#endif
#ifdef HAVE_AVX512_CPU_DEFINITION
  static FnPtr AVX512;
#endif
#ifdef HAVE_AVX2_VNNI_CPU_DEFINITION
  static FnPtr AVX2_VNNI;
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
  static FnPtr AVX2;
#endif

 private:
  DispatchStubImpl impl;
};

// namespace {
// template <typename FnPtr, typename T>
// struct RegisterCUDADispatch {
//   RegisterCUDADispatch(DispatchStub<FnPtr, T>& stub, FnPtr value) {
//     stub.set_cuda_dispatch_ptr(value);
//   }
// };

// template <typename FnPtr, typename T>
// struct RegisterHIPDispatch {
//   RegisterHIPDispatch(DispatchStub<FnPtr, T>& stub, FnPtr value) {
//     // TODO: make this point at hip_dispatch_ptr
//     stub.set_cuda_dispatch_ptr(value);
//   }
// };
// } // anonymous namespace

// Compiler will complain if you put things like std::tuple<Tensor, Tensor> in
// the `fn` argument of IPEX_DECLARE_DISPATCH. Some possible workarounds, e.g.,
// adding parentheses and using helper struct to get rid of the parentheses, do
// not work with MSVC. So do a `using`-declaration if you need to pass in such
// `fn`, e.g., grid_sampler_2d_backward_cpu_kernel in GridSampleKernel.h.
#define IPEX_DECLARE_DISPATCH(fn, name)    \
  struct name : DispatchStub<fn, name> {   \
    name() = default;                      \
    name(const name&) = delete;            \
    name& operator=(const name&) = delete; \
  };                                       \
  extern struct name name

#define IPEX_DEFINE_DISPATCH(name) struct name name

#undef IPEX_REGISTER_ARCH_DISPATCH
#define IPEX_REGISTER_ARCH_DISPATCH(name, arch, fn) \
  template <>                                       \
  decltype(fn) DispatchStub<decltype(fn), struct name>::arch = fn;

#ifdef HAVE_AVX512_CPU_DEFINITION
#define IPEX_REGISTER_AVX512_DISPATCH(name, fn) \
  IPEX_REGISTER_ARCH_DISPATCH(name, AVX512, fn)
#else
#define IPEX_REGISTER_AVX512_DISPATCH(name, fn)
#endif

#ifdef HAVE_AVX2_CPU_DEFINITION
#define IPEX_REGISTER_AVX2_DISPATCH(name, fn) \
  IPEX_REGISTER_ARCH_DISPATCH(name, AVX2, fn)
#else
#define IPEX_REGISTER_AVX2_DISPATCH(name, fn)
#endif

#undef IPEX_REGISTER_NO_CPU_DISPATCH
#define IPEX_REGISTER_NO_CPU_DISPATCH(name, fn_type)                        \
  IPEX_REGISTER_ARCH_DISPATCH(name, DEFAULT, static_cast<fn_type>(nullptr)) \
  IPEX_REGISTER_AVX512_DISPATCH(name, static_cast<fn_type>(nullptr))        \
  IPEX_REGISTER_AVX2_DISPATCH(name, static_cast<fn_type>(nullptr))          \
  IPEX_REGISTER_VSX_DISPATCH(name, static_cast<fn_type>(nullptr))

/*
ToDo: Fix warning: "REGISTER_HIP_DISPATCH" redefined to stock pytorch.
-----------------------------------------------
#define REGISTER_CUDA_DISPATCH(name, fn)                                   \
  static RegisterCUDADispatch<decltype(fn), struct name> name##__register( \
      name, fn);

#define REGISTER_HIP_DISPATCH(name, fn)                                   \
  static RegisterHIPDispatch<decltype(fn), struct name> name##__register( \
      name, fn);
*/

// NB: This macro must be used in an actual 'cu' file; if you try using
// it from a 'cpp' file it will not work!
// #if defined(__CUDACC__)
// #define IPEX_REGISTER_DISPATCH(name, fn) REGISTER_CUDA_DISPATCH(name, fn)
// #elif defined(__HIPCC__)
// // TODO: cut this over to HIP dispatch once we stop pretending that CUDA
// // is HIP in the PyTorch HIPify build.
// #define IPEX_REGISTER_DISPATCH(name, fn) REGISTER_CUDA_DISPATCH(name, fn)
// // #define IPEX_REGISTER_DISPATCH(name, fn) REGISTER_HIP_DISPATCH(name, fn)
// #elif defined(CPU_CAPABILITY)

#if defined(CPU_CAPABILITY)
#define IPEX_REGISTER_DISPATCH(name, fn) \
  IPEX_REGISTER_ARCH_DISPATCH(name, CPU_CAPABILITY, fn)
#undef IPEX_REGISTER_NO_AVX512_DISPATCH
#define IPEX_REGISTER_NO_AVX512_DISPATCH(name, fn_type) \
  IPEX_REGISTER_AVX512_DISPATCH(name, static_cast<fn_type>(nullptr))
#endif

} // namespace cpu
} // namespace tpp
} // namespace zendnn

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#endif
