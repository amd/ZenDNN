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
"""Matmul bounds on the synthetic machine: memory-bound decode, compute-bound prefill."""

from __future__ import annotations

import contextlib
import io
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import EXAMPLE, example, example_doc  # noqa: E402

from op_perf_utility import cli  # noqa: E402
from op_perf_utility.common.ceilings import Ceilings  # noqa: E402
from op_perf_utility.matmul.bounds import compute_path, matmul_bound, usable_threads  # noqa: E402
from op_perf_utility.matmul.work import MatmulShape  # noqa: E402


class BoundTest(unittest.TestCase):
    def setUp(self):
        self.c = example()

    def test_decode_is_l3_bound_warm_and_dram_bound_cold(self):
        s = MatmulShape(m=4, n=4096, k=4096, threads=16)
        b = matmul_bound(s, self.c)
        nbytes = 4096 * 4096 * 2 + 2 * 4 * 4096 * 2
        self.assertEqual(b.level, "l3")  # 33.6 MB > half the team's 16 MiB L2, < 80 % of 2 x 32 MiB
        self.assertAlmostEqual(b.memory_ms, nbytes / 640e6)
        self.assertAlmostEqual(b.compute_ms, 2.0 * 4 * 4096 * 4096 / 5120e6)
        self.assertEqual(b.bound, "l3")
        self.assertEqual(b.attainable_ms, b.memory_ms)
        cold = matmul_bound(s, self.c, cold=True)
        self.assertEqual(cold.level, "dram")
        self.assertAlmostEqual(cold.memory_ms, nbytes / 120e6)

    def test_prefill_is_compute_bound(self):
        b = matmul_bound(MatmulShape(m=4096, n=4096, k=4096, threads=16), self.c)
        self.assertEqual(b.level, "dram")
        self.assertEqual(b.bound, "compute")
        self.assertAlmostEqual(b.attainable_ms, 2.0 * 4096 ** 3 / 5120e6)

    def test_small_call_lives_in_l2(self):
        self.assertEqual(matmul_bound(MatmulShape(m=4, n=1024, k=1024, threads=16), self.c).level, "l2")

    def test_paths(self):
        c = self.c
        self.assertEqual(compute_path(MatmulShape(m=1, n=1, k=1, wei="s4"), c), "int8")
        self.assertEqual(compute_path(MatmulShape(m=1, n=1, k=1, wei="f16", src="f16", dst="f16"), c), "fp32")
        self.assertEqual(compute_path(MatmulShape(m=1, n=1, k=1, wei="f32", src="f32", dst="f32"), c), "fp32")

    def test_threads_scale_the_ceilings(self):
        b = matmul_bound(MatmulShape(m=4096, n=4096, k=4096, threads=4), self.c)
        self.assertAlmostEqual(b.compute_gflops, 5120.0 / 4)

    def test_efficiency_and_row(self):
        b = matmul_bound(MatmulShape(m=4, n=4096, k=4096, threads=16), self.c)
        self.assertAlmostEqual(b.efficiency(2 * b.attainable_ms), 0.5)
        self.assertIsNone(b.efficiency(None))
        self.assertIsNone(b.efficiency(0.0))
        row = b.row(2 * b.attainable_ms)
        self.assertAlmostEqual(row["efficiency_pct"], 50.0)
        self.assertAlmostEqual(row["gbps"], 320.0)
        self.assertIsNone(b.row()["efficiency_pct"])

    def test_residency_counts_each_buffer_once(self):
        # An s8 call quantizing its source: its 7 MiB of buffers fit half the team's 16 MiB L2,
        # while its traffic, the u8 copy written and read back, is 8.25 MiB.
        s = MatmulShape(m=640, n=1024, k=2048, wei="s8", wei_scale_bytes=4, src_scale_bytes=4, threads=16)
        b = matmul_bound(s, self.c)
        self.assertLessEqual(b.work.footprint_bytes, 8 << 20)
        self.assertGreater(b.work.total_bytes, 8 << 20)
        self.assertEqual(b.level, "l2")

    def test_smt_team_spans_its_cores_domains(self):
        # 16 threads on 8 cores of one 8-core domain: one domain's 32 MiB, not two.
        run = dict(example_doc()["run"], team_cores=8)
        c = example(run=run)
        self.assertEqual((c.domains(16), c.team_l3_bytes(16)), (1, 32 << 20))
        self.assertEqual(c.bandwidth_gbps("l3", 32), 640.0 * 2)  # 16 cores: two domains against one

    def test_tile_paths(self):
        doc = example_doc()
        doc["compute"]["int4"] = dict(doc["compute"]["int8"], instr_per_cycle=1.5, gflops_core=7680.0 / 16,
                                      gflops_team=7680.0)
        doc["compute"]["f16_widen"] = dict(doc["compute"]["fp32"], instr_per_cycle=1.8, gflops_core=2304.0 / 16,
                                           gflops_team=2304.0)
        c = Ceilings(doc)
        f16 = MatmulShape(m=4096, n=4096, k=4096, wei="f16", src="f16", dst="f16", threads=16)
        s4 = MatmulShape(m=4096, n=4096, k=4096, wei="s4", wei_group=128, wei_scale_bytes=2, threads=16)
        self.assertEqual((compute_path(s4, c), compute_path(f16, c)), ("int4", "f16_widen"))
        self.assertEqual(compute_path(MatmulShape(m=1, n=1, k=1, wei="s8"), c), "int8")
        self.assertAlmostEqual(matmul_bound(s4, c).compute_ms, 2.0 * 4096 ** 3 / 7680e6)
        self.assertAlmostEqual(matmul_bound(f16, c).compute_ms, 2.0 * 4096 ** 3 / 2304e6)
        # FP16 FMA, where the CPU has AVX512-FP16, before the F16 kernels' loop.
        doc["compute"]["fp16"] = dict(doc["compute"]["bf16"])
        self.assertEqual(compute_path(f16, Ceilings(doc)), "fp16")

    def test_stores_at_the_write_bandwidth(self):
        s = MatmulShape(m=4, n=4096, k=4096, threads=16)
        c = example(write_bandwidth=[{"level": "dram", "threads": 1, "gbps": 12.0},
                                     {"level": "dram", "threads": 16, "gbps": 60.0}])
        b = matmul_bound(s, c, cold=True)
        w = b.work
        self.assertEqual(b.write_gbps, 60.0)
        self.assertAlmostEqual(b.memory_ms, (w.total_bytes - w.dst_bytes) / 120e6 + w.dst_bytes / 60e6)
        self.assertGreater(b.memory_ms, matmul_bound(s, self.c, cold=True).memory_ms)
        # Two copies of 8 threads share the 16-thread write rate as well.
        two = matmul_bound(MatmulShape(m=4, n=4096, k=4096, threads=8), c, cold=True, instances=2)
        self.assertEqual((two.bandwidth_gbps, two.write_gbps), (60.0, 30.0))
        # A level without write points prices its stores as reads.
        warm = matmul_bound(s, c)
        self.assertEqual((warm.level, warm.write_gbps), ("l3", 0.0))
        self.assertAlmostEqual(warm.memory_ms, warm.work.total_bytes / 640e6)

    def test_threads_a_call_can_use(self):
        # One thread per 16 columns of a row of a batch item.
        self.assertEqual(usable_threads(MatmulShape(m=1, n=64, k=4096), 16), 4)
        self.assertEqual(usable_threads(MatmulShape(m=1, n=72, k=4096), 16), 5)
        self.assertEqual(usable_threads(MatmulShape(m=1, n=16, k=4096, batch=8), 16), 8)
        self.assertEqual(usable_threads(MatmulShape(m=4, n=4096, k=4096), 16), 16)
        # 4 of the 16 threads: L2 between the 1- and 16-thread points, bf16 at 4 x 320 GFLOP/s.
        b = matmul_bound(MatmulShape(m=1, n=64, k=4096, threads=16), self.c)
        self.assertEqual((b.threads, b.team, b.level, b.bound), (4, 16, "l2", "l2"))
        self.assertAlmostEqual(b.bandwidth_gbps, 96.0 + (1280.0 - 96.0) * 3 / 15)
        self.assertAlmostEqual(b.compute_gflops, 4 * 320.0)
        self.assertAlmostEqual(b.memory_ms, (64 * 4096 * 2 + 4096 * 2 + 64 * 2) / 332.8e6)
        # Cold: DRAM at 4 threads, between the 1- and 8-thread points.
        self.assertAlmostEqual(matmul_bound(MatmulShape(m=1, n=64, k=4096, threads=16), self.c,
                                            cold=True).bandwidth_gbps, 24.0 + 56.0 * 3 / 7)
        self.assertEqual((b.row()["threads"], b.row()["team"]), (4, 16))

    def test_instances_share_dram(self):
        s = MatmulShape(m=4, n=4096, k=4096, threads=8)
        one = matmul_bound(s, self.c, cold=True)
        self.assertAlmostEqual(one.bandwidth_gbps, 80.0)
        # Two copies of 8 threads: the 16-thread DRAM point, halved.
        two = matmul_bound(s, self.c, cold=True, instances=2)
        self.assertAlmostEqual(two.bandwidth_gbps, 60.0)
        self.assertTrue(two.dram_measured)
        self.assertEqual(two.row()["instances"], 2)
        # Four copies: 32 threads, above the probe's 16, keep the 16-thread bandwidth.
        four = matmul_bound(s, self.c, cold=True, instances=4)
        self.assertAlmostEqual(four.bandwidth_gbps, 30.0)
        self.assertFalse(four.dram_measured)
        # A call in L2 shares nothing.
        small = MatmulShape(m=4, n=1024, k=1024, threads=16)
        self.assertEqual(matmul_bound(small, self.c, instances=4).bandwidth_gbps,
                         matmul_bound(small, self.c).bandwidth_gbps)
        with self.assertRaises(ValueError):
            matmul_bound(s, self.c, instances=0)

    def test_cli_bound(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cli.main(["bound", "--ceilings", EXAMPLE, "--shape", "m=4096,n=4096,k=4096", "--format", "json"])
        self.assertIn('"bound": "compute"', out.getvalue())
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            cli.main(["bound", "--ceilings", EXAMPLE, "--shape", "m=4,n=4096,k=4096,threads=8", "--cold",
                      "--instances", "4", "--format", "json"])
        self.assertIn('"instances": 4', out.getvalue())
        self.assertIn("DRAM at 32 threads (1 bound) taken as at the probe's 16 threads", err.getvalue())
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                cli.main(["bound", "--ceilings", EXAMPLE, "--shape", "m=4,n=4096,k=4096", "--instances", "0"])


if __name__ == "__main__":
    unittest.main()
