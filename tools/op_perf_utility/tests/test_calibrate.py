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
    def test_against_the_library_model(self):
        # Probe 1.2 prints the library's model: a path within 20% of it is priced right already;
        # s4 takes the int4 tile's path, f16 the widening loop's.  A latency off the model
        # (int8: 5 against 4) gives no cm_fma_lat, whose built-in 6 is a fitted floor: it goes
        # into the uarch value, with the memory terms in the library's form.
        doc = copy.deepcopy(example_doc())
        doc["uarch"]["dot"] = {p: {"ipc": 2.0, "latency": 6 if p == "bf16" else 4, "chains": 8}
                               for p in ("fp32", "bf16", "int8", "int4", "f16_widen")}
        doc["compute"]["int4"] = dict(doc["compute"]["int8"], instr_per_cycle=2.75, latency_cycles=4.0)
        doc["compute"]["f16_widen"] = dict(doc["compute"]["fp32"], instr_per_cycle=1.9)
        from op_perf_utility.common.ceilings import Ceilings

        cal = calibrate(Ceilings(doc, "model"))
        self.assertEqual(cal.paths, {"int4": {"cm_fma_pipes": 3}})
        self.assertEqual((cal.path_of("s4"), cal.path_of("u4"), cal.path_of("f16")), ("int4", "int4", "f16_widen"))
        self.assertEqual(cal.model["int8"], {"ipc": 2.0, "latency": 4})
        self.assertEqual(cal.measured["int4"], {"ipc": 2.75, "latency": 4.0})
        self.assertEqual(cal.plan_domain, 8)
        # mem_cycles: 341 cycles per KiB a thread at 8 threads to a domain, times the square root of 8.
        self.assertEqual(cal.uarch_override(),
                         "fp32_ipc=2,fp32_lat=4,bf16_ipc=2,bf16_lat=6,int8_ipc=2,int8_lat=5,int4_ipc=2.75,"
                         "int4_lat=4,f16_widen_ipc=1.9,f16_widen_lat=4,mem_cycles=964,stream_cycles=128")
        self.assertEqual(set(cal.to_dict()), {"threads", "domain_threads", "common", "paths", "measured", "model"})
        self.assertEqual(sorted(k for k in cal.sources if "@" in k), ["cm_fma_pipes@int4"])
        self.assertIn("model 2", cal.sources["cm_fma_pipes@int4"])
        # Without DRAM in the ceilings the value holds the paths alone.
        doc["bandwidth"] = [b for b in doc["bandwidth"] if b["level"] != "dram"]
        self.assertNotIn("mem_cycles", calibrate(Ceilings(doc, "no dram")).uarch_override())

    def test_the_campaigns_model(self):
        # The model to compare with is the one the library prices the campaign with: a probe that
        # measured int8 at 4 a cycle where its own model says 4 gives no correction alone, but
        # does against a campaign that prices int8 at 2.
        doc = copy.deepcopy(example_doc())
        doc["uarch"]["dot"] = {"int8": {"ipc": 4.0, "latency": 4, "chains": 16}}
        doc["compute"]["int8"] = dict(doc["compute"]["int8"], instr_per_cycle=4.0)
        from op_perf_utility.common.ceilings import Ceilings

        c = Ceilings(doc, "four")
        self.assertNotIn("int8", calibrate(c).paths)
        self.assertEqual(calibrate(c, model={"int8": (2.0, 4)}).paths["int8"], {"cm_fma_pipes": 4})
        self.assertEqual(calibrate(c, model={}).paths["int8"], {"cm_fma_pipes": 4})

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
        # A probe without the library's model: every measured path gets its pipes, no path a
        # cm_fma_lat.
        self.assertEqual(cal.paths, {"fp32": {"cm_fma_pipes": 2}, "bf16": {"cm_fma_pipes": 2},
                                     "int8": {"cm_fma_pipes": 2}})
        self.assertEqual(cal.for_dtype("s4"), dict(cal.common, cm_fma_pipes=2))
        # No FP16 FMA measured: f16 weights take the FP32 path.
        self.assertEqual(cal.path_of("f16"), "fp32")
        self.assertEqual(sorted(cal.sources), sorted(
            ["cm_l2_bytes", "cm_l3_cycles", "cm_mem_cycles", "cm_stream_cycles"]
            + ["cm_fma_pipes@%s" % p for p in ("fp32", "bf16", "int8")]))

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
        doc["compute"]["int4"] = dict(doc["compute"]["int8"], instr_per_cycle=0.001)
        cal = calibrate(example(compute=doc["compute"]))
        # Rounded into the keys' ranges, and the uarch value into the ranges the library takes.
        self.assertEqual(cal.paths["fp16"], {"cm_fma_pipes": 1})
        self.assertEqual(cal.path_of("f16"), "fp16")
        self.assertEqual(cal.settings("int8"),
                         "cm_fma_pipes=2,cm_l2_bytes=32,cm_l3_cycles=64,cm_mem_cycles=341,cm_stream_cycles=128")
        items = cal.uarch_override().split(",")
        self.assertIn("fp16_ipc=0.4", items)
        self.assertIn("fp16_lat=64", items)
        self.assertIn("int4_ipc=0.01", items)
        cal.common["cm_mem_cycles"] = 1 << 20
        self.assertIn("mem_cycles=1048576", cal.uarch_override().split(","))
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
                self.assertTrue(f.read().startswith("dtype=bf16:cm_fma_pipes=2,cm_l2_bytes=32,"))
            self.assertEqual(cli.main(["calibrate", "--ceilings", EXAMPLE, "--format", "uarch", "-o", path]), 0)
            with open(path) as f:
                self.assertEqual(f.read().strip(), "fp32_ipc=2,fp32_lat=4,bf16_ipc=2,bf16_lat=6,int8_ipc=2,int8_lat=5,"
                                 "mem_cycles=964,stream_cycles=128")


if __name__ == "__main__":
    unittest.main()
