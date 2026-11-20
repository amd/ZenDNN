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
"""The cost model's inputs from the synthetic machine, worked by hand."""

import contextlib
import copy
import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import EXAMPLE, example, example_doc  # noqa: E402

from op_perf_utility import cli  # noqa: E402
from op_perf_utility.matmul.calibrate import KEYS, calibrate  # noqa: E402


class TestCalibrate(unittest.TestCase):
    def test_the_synthetic_machine(self):
        # 16 threads at 2.5 GHz (3.0 on one core), 8 to a domain.
        cal = calibrate(example())
        self.assertEqual((cal.threads, cal.domain_threads), (16, 8))
        self.assertEqual(cal.common, {
            "cm_l2_bytes": 32,         # 1280 GB/s / 16 threads / 2.5 GHz
            "cm_l3_cycles": 64,        # 1024 / (320 GB/s / 8 threads / 2.5 GHz)
            "cm_mem_cycles": 341,      # 1024 / (120 GB/s / 16 threads / 2.5 GHz) = 341.3
            "cm_stream_cycles": 128,   # 1024 / (24 GB/s / 3.0 GHz)
        })
        self.assertEqual(cal.paths, {"fp32": {"cm_fma_pipes": 2, "cm_fma_lat": 4},
                                     "bf16": {"cm_fma_pipes": 2, "cm_fma_lat": 6},
                                     "int8": {"cm_fma_pipes": 2, "cm_fma_lat": 5}})
        self.assertEqual(cal.for_dtype("s4"), dict(cal.common, cm_fma_pipes=2, cm_fma_lat=5))
        # No FP16 FMA measured: f16 weights take the FP32 path.
        self.assertEqual(cal.path_of("f16"), "fp32")
        self.assertEqual(sorted(cal.sources), sorted(
            ["cm_l2_bytes", "cm_l3_cycles", "cm_mem_cycles", "cm_stream_cycles"]
            + ["%s@%s" % (k, p) for p in ("fp32", "bf16", "int8") for k in ("cm_fma_pipes", "cm_fma_lat")]))

    def test_team_and_domain(self):
        # 8 threads: L2 between the 1- and 16-thread points (648.5 GB/s), DRAM measured.
        cal = calibrate(example(), threads=8)
        self.assertEqual((cal.common["cm_l2_bytes"], cal.common["cm_mem_cycles"]), (32, 256))
        # A domain is at most the team.
        self.assertEqual(calibrate(example(), threads=4).domain_threads, 4)
        self.assertEqual(calibrate(example(), domain_threads=16).common["cm_l3_cycles"], 64)

    def test_ranges_and_paths(self):
        doc = copy.deepcopy(example_doc())
        doc["compute"]["fp16"] = dict(doc["compute"]["fp32"], flops_per_instr=64, instr_per_cycle=0.4,
                                      latency_cycles=200.0)
        cal = calibrate(example(compute=doc["compute"]))
        # Rounded into the keys' ranges.
        self.assertEqual(cal.paths["fp16"], {"cm_fma_pipes": 1, "cm_fma_lat": 64})
        self.assertEqual(cal.path_of("f16"), "fp16")
        self.assertEqual(cal.settings("int8"),
                         "cm_fma_pipes=2,cm_fma_lat=5,cm_l2_bytes=32,cm_l3_cycles=64,cm_mem_cycles=341,"
                         "cm_stream_cycles=128")
        self.assertEqual([c.split(":")[0] for c in cal.override().split(";")],
                         ["dtype=bf16", "dtype=s8", "dtype=s4", "dtype=f16"])

    def test_cli(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(cli.main(["calibrate", "--ceilings", EXAMPLE, "--format", "json"]), 0)
        rows = json.loads(out.getvalue())
        self.assertEqual({r["key"] for r in rows}, set(KEYS))
        self.assertIn({"key": "cm_l2_bytes", "path": "all", "value": 32, "builtin": 16,
                       "source": "L2 32.0 B/cycle per thread at 16 threads"}, rows)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "o.txt")
            self.assertEqual(cli.main(["calibrate", "--ceilings", EXAMPLE, "--format", "settings", "-o", path]), 0)
            with open(path) as f:
                self.assertTrue(f.read().startswith("dtype=bf16:cm_fma_pipes=2,cm_fma_lat=6,"))


if __name__ == "__main__":
    unittest.main()
