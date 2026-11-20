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
"""benchdnn matmul results: its CSV and its printed lines, and the report command."""

from __future__ import annotations

import csv
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import EXAMPLE  # noqa: E402

from op_perf_utility import cli  # noqa: E402
from op_perf_utility.matmul.benchdnn_ingest import read_benchdnn  # noqa: E402

CSV_TEXT = (
    "Layer Number, BS, M, K, N, Iterations, Data type, Bias Enabled, Bias Data type, Post Operation, "
    "PostOp Data type, Kernel name, isWeightsConst, isTransA, isTransB, Alpha, Beta, Weight Scale Granularity, "
    "Weight Group Size, Weight Scale Data type, Warmup iterations, Src Dynamic Quant, Src Scale Granularity, "
    "Src Group Size, Src Scale Data type, Total time (ms) (all iters), Avg time (ms), GFLOPS, % of Total\n"
    "0, 1, 4, 4096, 4096, 100, bf16:bf16:bf16, 0, , , , native, 1, 0, 0, 1.00, 0.00, , , , 20, 0, , , , "
    "10.00, 0.10, 1342.18, 0.00\n"
    "0, 2, 8, 4096, 4096, 100, bf16:s4:bf16, 1, f32, , , native, 1, 0, 0, 1.00, 0.00, per-group, 128, bf16, "
    "20, 1, per-token, 0, f32, 20.00, 0.20, 1342.18, 0.00\n"
    "0, 1, 16, 4096, 4096, 100, f32:f32:f32, 1, f32, binary_add:relu, bf16, native, 1, 0, 0, 1.00, 0.00, , , , "
    "20, 0, , , , 30.00, 0.30, 1342.18, 0.00\n")

LINES_TEXT = ("benchdnn matmul\n"
              "4, 4096, 4096, 100, bf16:s8:bf16, true, f32, , , none, true, false, false, 1.0, 0.0, per-channel, "
              ", f32, 20, true, per-token, 0, f32, 12.5\n"
              "8, 4096, 4096, 100, bf16:s4:bf16, false, , binary_mul:binary_add, f16, none, true, false, false, 1.0, "
              "0.0, per-group, 128, bf16, 20, true, per-token, 0, bf16, 25.0\n"
              "not, a, result\n")


class IngestTest(unittest.TestCase):
    def test_csv(self):
        ms = read_benchdnn(CSV_TEXT, threads=16)
        self.assertEqual(len(ms), 3)
        a, b, c = ms
        self.assertEqual((c.shape.wei, c.shape.binary, c.shape.binary_dt, c.shape.bias), ("f32", 1, "bf16", "f32"))
        self.assertEqual((a.shape.binary, b.shape.binary), (0, 0))
        self.assertEqual((a.shape.m, a.shape.n, a.shape.k, a.shape.wei, a.time_ms), (4, 4096, 4096, "bf16", 0.10))
        s = b.shape
        self.assertEqual((s.batch, s.wei, s.wei_group, s.wei_scale_bytes, s.src_scale_bytes, s.bias),
                         (2, "s4", 128, 2, 4, "f32"))
        self.assertEqual(b.iters, 100)

    def test_csv_times_under_the_printed_precision(self):
        # 100 calls of 4 us: the average prints as 0.00, the total as 0.40.
        tiny = CSV_TEXT.split("\n")[0] + (
            "\n0, 1, 1, 64, 64, 100, bf16:bf16:bf16, 0, , , , native, 1, 0, 0, 1.00, 0.00, , , , 20, 0, , , , "
            "0.40, 0.00, 0.00, 0.00\n")
        (m,) = read_benchdnn(tiny, threads=16)
        self.assertAlmostEqual(m.time_ms, 0.004)

    def test_csv_summary_rows_and_granularity(self):
        head = CSV_TEXT.split("\n")[0]
        rows = (
            # A pipeline's Summary rows, before its layers: one of two layers, one of one.
            "\nSummary, 1, 4, 4096, 4096:11008, 100, bf16:s8:bf16, 0, , , , native, 1, 0, 0, 1.00, 0.00, , , , 20,"
            " 0, , , , 50.00, 0.50, 0.00, 0.00"
            "\nSummary, 1, 4, 4096, 4096, 100, bf16:bf16:bf16, 0, , , , native, 1, 0, 0, 1.00, 0.00, , , , 20,"
            " 0, , , , 10.00, 0.10, 0.00, 0.00"
            # Per-tensor weight scales, source scales per group of 128 columns.
            "\n0, 1, 4, 4096, 4096, 100, bf16:s8:bf16, 0, , , , native, 1, 0, 0, 1.00, 0.00, per-tensor, , f32, 20,"
            " 1, per-group, 128, f32, 10.00, 0.10, 0.00, 0.00"
            # A u8 source with a static per-tensor scale.
            "\n1, 1, 4, 4096, 4096, 100, u8:s8:bf16, 0, , , , native, 1, 0, 0, 1.00, 0.00, per-channel, , f32, 20,"
            " 0, per-tensor, , f32, 10.00, 0.10, 0.00, 0.00\n")
        a, b = read_benchdnn(head + rows, threads=16)
        self.assertEqual((a.shape.wei_scale_tensor, a.shape.src_group, a.shape.src_scale_bytes), (True, 128, 4))
        self.assertEqual((b.shape.src_scale_tensor, b.shape.src_scale_bytes, b.shape.wei_scale_tensor),
                         (True, 4, False))

    def test_batched_lines(self):
        # A batched call's line starts with its batch size.
        text = ("8, 16, 4096, 4096, 100, bf16:bf16:bf16, false, , , , none, true, false, false, 1.0, 0.0, "
                "per-channel, , f32, 20, false, , 0, f32, 40.0\n")
        (m,) = read_benchdnn(text, threads=32)
        self.assertEqual((m.shape.batch, m.shape.m, m.shape.k, m.shape.n, m.shape.wei, m.iters), (8, 16, 4096, 4096,
                                                                                                  "bf16", 100))
        self.assertAlmostEqual(m.time_ms, 0.4)

    def test_lines(self):
        m, m4 = read_benchdnn(LINES_TEXT, threads=32)
        s = m.shape
        self.assertEqual((s.m, s.k, s.n, s.wei, s.bias, s.wei_scale_bytes, s.src_scale_bytes),
                         (4, 4096, 4096, "s8", "f32", 4, 4))
        self.assertAlmostEqual(m.time_ms, 0.125)
        # The input fields of the line: per-group s4 scales in bf16, two binary post-ops in f16.
        s = m4.shape
        self.assertEqual((s.wei, s.bias, s.wei_group, s.wei_scale_bytes, s.src_scale_bytes, s.binary, s.binary_dt),
                         ("s4", "", 128, 2, 2, 2, "f16"))

    def test_file_and_cli_report(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "timings.csv")
            with open(path, "w", encoding="utf-8") as f:
                f.write(CSV_TEXT)
            self.assertEqual(len(read_benchdnn(path, 16)), 3)
            out = os.path.join(d, "report.csv")
            self.assertEqual(cli.main(["report", "--ceilings", EXAMPLE, "--benchdnn", path, "--format", "csv",
                                       "-o", out]), 0)
            with open(out, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["bound"], "l3")
            self.assertGreater(float(rows[0]["efficiency_pct"]), 0)


if __name__ == "__main__":
    unittest.main()
