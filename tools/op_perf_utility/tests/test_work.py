# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
"""The matmul work model: FLOPs and compulsory bytes, worked by hand."""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from op_perf_utility.matmul.work import MatmulShape, from_problem, work  # noqa: E402


class WorkTest(unittest.TestCase):
    def test_bf16_decode(self):
        w = work(MatmulShape(m=4, n=4096, k=4096, threads=16))
        self.assertEqual(w.flops, 2.0 * 4 * 4096 * 4096)
        self.assertEqual(w.wei_bytes, 4096 * 4096 * 2)
        self.assertEqual(w.src_bytes, 4 * 4096 * 2)
        self.assertEqual(w.dst_bytes, 4 * 4096 * 2)
        self.assertEqual(w.wei_scale_bytes + w.src_scale_bytes + w.bias_bytes, 0)
        self.assertEqual(w.total_bytes, 4096 * 4096 * 2 + 2 * 4 * 4096 * 2)

    def test_s4_groups_gated_batch(self):
        s = MatmulShape(m=8, n=8192, k=4096, batch=2, wei="s4", wei_group=128, wei_scale_bytes=2,
                        src_scale_bytes=4, gated=True, bias="f32")
        w = work(s)
        self.assertEqual(w.wei_bytes, 2 * 8192 * 4096 * 0.5)
        self.assertEqual(w.wei_scale_bytes, 2 * 8192 * 32 * 2)
        self.assertEqual(w.src_scale_bytes, 2 * 8 * 4)
        self.assertEqual(w.dst_bytes, 2 * 8 * 4096 * 2)  # gated: N / 2 columns
        self.assertEqual(w.bias_bytes, 8192 * 4)  # one bias the batch shares, as BenchDNN's {1, 1, N}
        self.assertEqual(w.flops, 2.0 * 2 * 8 * 8192 * 4096)
        self.assertEqual(work(MatmulShape(m=8, n=8192, k=4096, batch=2, bias="f32", binary=1,
                                          per_batch_operands=True)).bias_bytes, 2 * 8192 * 4)

    def test_scale_granularity(self):
        k = 4096
        # One weight scale for the tensor; source scales per group of 128 columns.
        w = work(MatmulShape(m=8, n=4096, k=k, wei="s8", wei_scale_bytes=4, wei_scale_tensor=True,
                             src_scale_bytes=4, src_group=128))
        self.assertEqual((w.wei_scale_bytes, w.src_scale_bytes), (4, 8 * 32 * 4))
        self.assertEqual(work(MatmulShape(m=8, n=4096, k=k, src="u8", wei="s8", src_scale_bytes=4,
                                          src_scale_tensor=True)).src_scale_bytes, 4)
        with self.assertRaises(ValueError):
            MatmulShape(m=1, n=1, k=k, src_group=100)
        with self.assertRaises(ValueError):
            MatmulShape(m=1, n=1, k=k, wei="s8", wei_group=128, wei_scale_tensor=True)
        s = MatmulShape.parse("m=8,n=4096,k=4096,wei=s8,src_scale_bytes=4,src_group=128,wei_scale_tensor=1")
        self.assertEqual((s.src_group, s.wei_scale_tensor), (128, True))

    def test_binary_post_ops(self):
        # Each binary post-op reads a tensor of the output's shape (N / 2 wide when gated).
        w = work(MatmulShape(m=8, n=4096, k=4096, binary=2, binary_dt="bf16"))
        self.assertEqual(w.post_bytes, 2 * 8 * 4096 * 2)
        self.assertEqual(w.total_bytes - w.post_bytes, work(MatmulShape(m=8, n=4096, k=4096)).total_bytes)
        # A gated call's binary operands index its [M, N] pre-activation.
        self.assertEqual(work(MatmulShape(m=8, n=4096, k=4096, gated=True, binary=1)).post_bytes, 8 * 4096 * 4)
        # One binary operand the batch shares.
        self.assertEqual(work(MatmulShape(m=8, n=4096, k=4096, batch=4, binary=1)).post_bytes, 8 * 4096 * 4)
        s = MatmulShape.parse("m=8,n=4096,k=4096,binary=1,binary_dt=f16")
        self.assertEqual((s.binary, s.binary_dt), (1, "f16"))
        self.assertIn("+1 f16 binary", s.label())
        with self.assertRaises(ValueError):
            MatmulShape(m=1, n=1, k=1, binary=1, binary_dt="q8")

    def test_validation(self):
        with self.assertRaises(ValueError):
            MatmulShape(m=0, n=1, k=1)
        with self.assertRaises(ValueError):
            MatmulShape(m=1, n=1, k=100, wei="s4", wei_group=64)
        with self.assertRaises(ValueError):
            MatmulShape(m=1, n=3, k=1, gated=True)
        with self.assertRaises(ValueError):
            MatmulShape(m=1, n=1, k=1, wei="q8")

    def test_parse(self):
        s = MatmulShape.parse("m=4, n=4096, k=4096, wei=s8, gated=1", threads=16)
        self.assertEqual((s.wei, s.wei_scale_bytes, s.gated, s.threads), ("s8", 4, True, 16))
        with self.assertRaises(ValueError):
            MatmulShape.parse("m=4,n=4096")
        with self.assertRaises(ValueError):
            MatmulShape.parse("m=4,n=4,k=4,colour=red")

    def test_from_problem(self):
        p = SimpleNamespace(dtype="s8", m=8, n=4096, k=4096, threads=32, src="bf16", dst="bf16", has_bias=True,
                            epilogue="store", dynamic=True, group=0, scale_bytes=0, name="q")
        s = from_problem(p)
        self.assertEqual((s.wei, s.wei_scale_bytes, s.src_scale_bytes, s.bias, s.wei_group), ("s8", 4, 4, "f32", 0))
        p4 = SimpleNamespace(dtype="s4", m=8, n=4096, k=4096, threads=32, src="bf16", dst="bf16", has_bias=False,
                             epilogue="gated", dynamic=True, group=128, scale_bytes=2, name="")
        s4 = from_problem(p4)
        self.assertEqual((s4.wei_group, s4.wei_scale_bytes, s4.gated), (128, 2, True))
        # One group over K is one scale per column.
        self.assertEqual(from_problem(SimpleNamespace(**dict(vars(p4), group=4096))).wei_group, 0)
        # A per-group s8 problem that gives its group: N x K / G scales.
        g8 = from_problem(SimpleNamespace(**dict(vars(p), group=128, scale_bytes=2)))
        self.assertEqual((g8.wei_group, g8.wei_scale_bytes), (128, 2))
        self.assertEqual(work(g8).wei_scale_bytes, 4096 * 32 * 2)

    def test_quantized_source_copy(self):
        # A bf16 source quantized in the call: its u8 copy is written once and read once.
        s = MatmulShape(m=8, n=4096, k=4096, wei="s8", wei_scale_bytes=4, src_scale_bytes=4)
        w = work(s)
        self.assertEqual(w.src_quant_bytes, 8 * 4096)
        self.assertEqual(w.write_bytes, w.dst_bytes + 8 * 4096)
        plain = work(MatmulShape(m=8, n=4096, k=4096, wei="s8", wei_scale_bytes=4))
        self.assertEqual(w.total_bytes - plain.total_bytes, 8 * 4 + 2 * 8 * 4096)
        # Its buffers hold the copy once: the traffic counts it twice.
        self.assertEqual(w.total_bytes - w.footprint_bytes, 8 * 4096)
        # A u8 source needs no copy, nor do weights the call does not quantize for.
        self.assertEqual(work(MatmulShape(m=8, n=4096, k=4096, src="u8", wei="s8", src_scale_bytes=4)).src_quant_bytes,
                         0)
        self.assertEqual(work(MatmulShape(m=8, n=4096, k=4096, src_scale_bytes=4)).src_quant_bytes, 0)


if __name__ == "__main__":
    unittest.main()
