/*****************************************************************************
* Copyright (c) 2024 Advanced Micro Devices, Inc.
* All rights reserved.
******************************************************************************/
#ifndef _ZENTPP_DEFS_HPP_
#define _ZENTPP_DEFS_HPP_

/* DispatchStub.h, DispatchStub.cpp */
// #define HAVE_AVX512_FP16_CPU_DEFINITION
// #define HAVE_AMX_CPU_DEFINITION
// #define HAVE_AVX512_BF16_CPU_DEFINITION
// #define HAVE_AVX512_VNNI_CPU_DEFINITION
// #define HAVE_AVX512_CPU_DEFINITION
// #define HAVE_AVX2_VNNI_CPU_DEFINITION
// #define HAVE_AVX2_CPU_DEFINITION

#define CPU_CAPABILITY    DEFAULT

/* cpu_features.cpp */
// #define ENABLE_XCR_CHECK
// #define CPU_FEATURE_EXEC

/* TPPGEMM.h, TPPGEMM.cpp TPPGEMMKrnl.cpp */
#define USE_LIBXSMM

/* TPPGEMMKrnl.h */
// #define NO_PARLOOPER

/* timing.h */
// #define PROFILE_TPP
// #define DEBUG_TRACE_TPP

/* rtm.h */
// #define RTM_DEBUG

/* optim.cpp */
// #define ENABLE_RTM

#endif
