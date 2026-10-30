/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 *******************************************************************************/

#ifndef ZENDNNL_GTEST_GROUP_MATMUL_KBLOCK_TEST_HOOKS_HPP
#define ZENDNNL_GTEST_GROUP_MATMUL_KBLOCK_TEST_HOOKS_HPP

namespace group_matmul_test {

/// Execute one public group-matmul shape and compare every output against an
/// independent double-accumulation reference.
void run_kblock_case(int num_threads, int M, int N, int K, int nr,
        int kblock_mode, bool with_bias, bool f32_dst, int lda_pad, int ldc_pad,
        bool expect_kblock, const char *label);

} // namespace group_matmul_test

#endif
