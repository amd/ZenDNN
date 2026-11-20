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
"""The ceilings document: validation, lookups, machine matching and output."""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import EXAMPLE, example, example_doc  # noqa: E402

from op_perf_utility import cli  # noqa: E402
from op_perf_utility.common import report  # noqa: E402
from op_perf_utility.common.ceilings import Ceilings, CeilingsError, validate  # noqa: E402


class ValidateTest(unittest.TestCase):
    def test_example_is_valid(self):
        self.assertEqual(validate(example_doc()), [])

    def test_errors(self):
        doc = example_doc()
        doc["format"] = "other"
        doc["compute"]["avx"] = {"flops_per_instr": 1}
        doc["bandwidth"].append({"level": "l4", "threads": 1, "gbps": 1.0})
        doc["clock"] = {"ghz_1core": 0}
        errs = validate(doc)
        self.assertTrue(any("format" in e for e in errs))
        self.assertTrue(any("unknown path 'avx'" in e for e in errs))
        self.assertTrue(any("level must be one of" in e for e in errs))
        self.assertTrue(any("clock" in e for e in errs))
        with self.assertRaises(CeilingsError):
            Ceilings(doc)

    def test_probe_1_1_fields(self):
        doc = example_doc()
        doc["write_bandwidth"] = [{"level": "dram", "threads": 0, "gbps": 1.0}]
        doc["run"]["team_cores"] = -1
        errs = validate(doc)
        self.assertTrue(any("write_bandwidth[0]" in e for e in errs))
        self.assertTrue(any("team_cores" in e for e in errs))

    def test_load_missing_file(self):
        with self.assertRaises(CeilingsError):
            Ceilings.load(os.path.join(tempfile.gettempdir(), "no_such_ceilings.json"))


class LookupTest(unittest.TestCase):
    def setUp(self):
        self.c = example()

    def test_machine(self):
        c = self.c
        self.assertEqual(c.threads, 16)
        self.assertEqual(c.domain_threads, 8)
        self.assertEqual(c.clock_ghz(1), 3.0)
        self.assertEqual(c.clock_ghz(4), 2.5)
        self.assertEqual(c.team_l2_bytes(4), 4 << 20)
        self.assertEqual(c.team_l3_bytes(8), 32 << 20)
        self.assertEqual(c.team_l3_bytes(9), 64 << 20)

    def test_domain_falls_back_to_cpu(self):
        run = dict(example_doc()["run"])
        del run["domain_threads"]
        uarch = dict(example_doc()["uarch"], l3_domain_cores=0)
        self.assertEqual(example(run=run, uarch=uarch).domain_threads, 8)

    def test_compute(self):
        c = self.c
        self.assertEqual(c.compute_gflops("bf16", 16), 5120.0)
        self.assertEqual(c.compute_gflops("bf16", 4), 1280.0)
        self.assertFalse(c.has_compute("fp16"))
        with self.assertRaises(CeilingsError):
            c.compute_gflops("fp16", 16)
        self.assertEqual(c.instr_per_cycle("int8"), 2.0)
        self.assertEqual(c.latency_cycles("bf16"), 6.0)
        self.assertIsNone(c.latency_cycles("fp16"))

    def test_bandwidth(self):
        c = self.c
        self.assertEqual(c.bandwidth_gbps("dram", 8), 80.0)
        self.assertAlmostEqual(c.bandwidth_gbps("dram", 4), 24.0 + (80.0 - 24.0) * 3 / 7)
        self.assertEqual(c.bandwidth_gbps("dram", 32), 120.0)  # saturated
        self.assertEqual(c.bandwidth_gbps("l2", 32), 2560.0)  # private: per thread
        self.assertEqual(c.bandwidth_gbps("l3", 32), 1280.0)  # per domain
        with self.assertRaises(CeilingsError):
            example(bandwidth=[{"level": "l1", "threads": 1, "gbps": 1.0}]).bandwidth_gbps("dram", 1)

    def test_rates(self):
        c = self.c
        self.assertAlmostEqual(c.bytes_per_cycle_per_thread("l2", 16), 1280.0 / 16 / 2.5)
        self.assertAlmostEqual(c.cycles_per_kib_per_thread("dram", 1), 1024.0 / (24.0 / 3.0))

    def test_write_bandwidth(self):
        # Without write points the stores have no bandwidth of their own.
        self.assertFalse(self.c.has_level("dram", write=True))
        c = example(write_bandwidth=[{"level": "dram", "threads": 1, "gbps": 12.0},
                                     {"level": "dram", "threads": 16, "gbps": 60.0}])
        self.assertTrue(c.has_level("dram", write=True))
        self.assertAlmostEqual(c.bandwidth_gbps("dram", 4, write=True), 12.0 + 48.0 * 3 / 15)
        self.assertEqual(c.bandwidth_gbps("dram", 32, write=True), 60.0)
        self.assertEqual(c.measured_threads("dram", write=True), 16)
        self.assertEqual(c.bandwidth_gbps("dram", 16), 120.0)  # reads unchanged
        with self.assertRaises(CeilingsError):
            c.bandwidth_gbps("l2", 1, write=True)

    def test_smt_team(self):
        # 16 threads on 8 cores: two siblings a core share its pipes, L1 and L2.
        run = dict(example_doc()["run"], team_cores=8)
        c = example(run=run)
        self.assertEqual((c.team_cores, c.cores(16), c.cores(4), c.cores(3)), (8, 8, 2, 2))
        self.assertEqual(c.team_l2_bytes(16), 8 << 20)
        self.assertEqual(c.compute_gflops("bf16", 16), 5120.0)
        self.assertEqual(c.compute_gflops("bf16", 4), 5120.0 / 8 * 2)
        self.assertEqual(c.bandwidth_gbps("l2", 32), 2560.0)  # 16 cores against the probe's 8
        self.assertIn("on 8 cores", c.summary())
        self.assertEqual(self.c.cores(4), 4)  # without team_cores, a thread a core


class MatchTest(unittest.TestCase):
    def test_match(self):
        c = example()
        hw = {"vendor": "ExampleVendor", "family": 1, "model": 1, "l3_kib": 32768, "l3_domain_threads": 8,
              "isa": ["avx512f", "avx512bf16", "avx512vnni"], "logical_cpus": 999}
        self.assertEqual(c.match(hw), [])
        bad = c.match(dict(hw, model=2, isa=["avx512f"]))
        self.assertEqual(len(bad), 2)
        self.assertTrue(bad[0].startswith("model:"))
        self.assertTrue(bad[1].startswith("isa:"))
        # A field one side does not report is not compared.
        self.assertEqual(c.match(dict(hw, l2_kib=0, stepping=7)), [])

    def test_brief_holds_no_measurement(self):
        b = example().brief()
        self.assertEqual(set(b), {"probe_version", "created", "threads"})


class OutputTest(unittest.TestCase):
    def test_summary(self):
        s = example().summary()
        self.assertIn("compute bf16", s)
        self.assertIn("bandwidth dram", s)

    def test_render(self):
        rows = [{"a": 1, "b": 0.123456}, {"a": 22, "b": None}]
        t = report.render(rows)
        self.assertIn("0.1235", t)
        self.assertEqual(report.render(rows, "csv").splitlines(), ["a,b", "1,0.123456", "22,"])
        self.assertEqual(json.loads(report.render(rows, "json"))[1]["a"], 22)
        with self.assertRaises(ValueError):
            report.render(rows, "xml")

    def test_cli_show(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(cli.main(["show", EXAMPLE]), 0)
        self.assertIn("ExampleVendor", out.getvalue())


if __name__ == "__main__":
    unittest.main()
