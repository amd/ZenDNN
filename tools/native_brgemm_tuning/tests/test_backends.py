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
"""Backends: the API-log line, benchdnn / replay parsing, sessions, collect()."""

import gc
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import warnings
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import SCHEMA, UARCH  # noqa: E402

from native_brgemm_tuning import candidates as C  # noqa: E402
from native_brgemm_tuning import results as R  # noqa: E402
from native_brgemm_tuning.backends import BenchdnnBackend, ReplayBackend  # noqa: E402
from native_brgemm_tuning.backends import common, jobs  # noqa: E402
from native_brgemm_tuning.backends.benchdnn import bd_line, parse_result_line  # noqa: E402
from native_brgemm_tuning.problems import Problem  # noqa: E402

LINE = ("[API    ][verbose][12.5]:[MATMUL.NATIVE_BRGEMM] M=8 N=6144 K=4096 pack_nr=32 nr=32 mb=8 "
        "tiles=192 threads=32 col_groups=0 transA=0 gemv=none mr=6 row_unit=12 traversal=m_inner "
        "int8_rows=12/12 int8_cw=32/16 int8_pass=rule direct=1 dequant_bias=1 copy_block=64 "
        "source=override+table cfg=nr=32,int8_mr=8 team=32")
# An M = 1 call runs on the GEMV kernels of its type, which the line names.
GEMV_LINE = ("[API    ][verbose][0.0021]:[MATMUL.NATIVE_BRGEMM] M=1 N=4096 K=4096 pack_nr=32 nr=32 mb=1 "
             "tiles=128 threads=16 col_groups=0 transA=0 gemv=s8 mr=6 row_unit=12 traversal=m_inner "
             "int8_rows=12/12 int8_cw=32/16 int8_pass=rule direct=1 dequant_bias=0 copy_block=64 "
             "source=heuristic cfg=- team=16")


def log_line(p, fields, source, cfg):
    f = " ".join("%s=%s" % kv for kv in fields.items())
    return "[API    ][verbose][1.0]:[MATMUL.NATIVE_BRGEMM] M=%d N=%d K=%d pack_nr=32 %s transA=0 source=%s cfg=%s" % (
        p.m, p.n, p.k, f, source, cfg)


class TestApiLog(unittest.TestCase):
    def test_lines_of_each_weight_type(self):
        # The lines of the current planner: gemv=none on the tiles, the kernels of an f16 call,
        # the scheme, group size, order and pass budget of an s4 call, the pairing of a gated call.
        from native_brgemm_tuning import plan as PL
        s4 = ("[API    ][verbose][0.0037]:[MATMUL.NATIVE_BRGEMM] M=8 N=4096 K=4096 pack_nr=32 nr=32 mb=8 tiles=128 "
              "threads=16 col_groups=0 transA=0 gemv=none mr=6 row_unit=12 traversal=m_inner int8_rows=12/6 "
              "int8_cw=32/16 int8_scheme=per_group direct=1 wei=s4 gs=128 order=fused pass_budget=40960 "
              "copy_block=64 source=heuristic cfg=- team=16")
        f = common.parse_apilog_line(s4)
        self.assertLessEqual(set(f), set(PL.LOG_LINE_FIELDS))
        self.assertEqual(common.applied_status(f, ""), ("applied", ""))
        p = Problem("s4", 8, 4096, 4096, group=128)
        q = PL.builtin_plan(p, UARCH, SCHEMA)
        want = PL.plan_log_fields(q, p)
        self.assertEqual({k: want[k] for k in ("int8_scheme", "gs", "pass_budget", "gemv")},
                         {"int8_scheme": "per_group", "gs": "128", "pass_budget": str(q.int8_pass_budget),
                          "gemv": "none"})
        f16 = ("[API    ][verbose][0.0040]:[MATMUL.NATIVE_BRGEMM] M=8 N=4096 K=4096 pack_nr=32 nr=32 mb=8 kc=4096 "
               "kernels=f16 tiles=128 threads=16 col_groups=0 transA=0 gemv=none mr=12 row_unit=12 "
               "traversal=m_inner copy_block=64 source=heuristic cfg=- team=16")
        self.assertEqual(common.parse_apilog_line(f16)["kernels"], "f16")
        gated = LINE + " pairing=64"
        self.assertEqual(common.parse_apilog_line(gated)["pairing"], "64")

    def test_lines_of_the_gemv_kernels(self):
        # Every M = 1 call runs on the GEMV kernels of its type, which its line names next to
        # the plan of the tiles; the model names the same kernels, and the tuner keeps the
        # call out.
        from native_brgemm_tuning import plan as PL
        lines = {
            "bf16": "M=1 N=4096 K=4096 pack_nr=32 nr=32 mb=1 kc=4096 tiles=128 threads=16 col_groups=0 transA=0 "
                    "gemv=bf16 mr=12 row_unit=12 traversal=m_inner copy_block=64 source=heuristic cfg=- team=16",
            "f16": "M=1 N=4096 K=4096 pack_nr=32 nr=32 mb=1 kc=4096 kernels=f16 tiles=128 threads=16 col_groups=0 "
                   "transA=0 gemv=f16 mr=12 row_unit=12 traversal=m_inner copy_block=64 source=heuristic cfg=- "
                   "team=16",
            "s8": GEMV_LINE,
            "s4": "M=1 N=4096 K=4096 pack_nr=32 nr=32 mb=1 tiles=128 threads=16 col_groups=0 transA=0 gemv=s4 mr=6 "
                  "row_unit=12 traversal=m_inner int8_rows=12/6 int8_cw=32/16 int8_scheme=per_group direct=1 "
                  "wei=s4 gs=128 order=fused pass_budget=40960 copy_block=64 source=heuristic cfg=- team=16",
        }
        for dtype, text in lines.items():
            f = common.parse_apilog_line(text if text.startswith("[") else "[MATMUL.NATIVE_BRGEMM] " + text)
            self.assertLessEqual(set(f), set(PL.LOG_LINE_FIELDS), dtype)
            self.assertIn(f["gemv"], PL.GEMV_KINDS, dtype)
            p = Problem(dtype, 1, 4096, 4096, threads=16, group=128 if dtype == "s4" else 0)
            q = PL.builtin_plan(p, UARCH, SCHEMA)
            if dtype == "f16":
                q = replace(q, kind="f16")  # the line's CPU has no AVX512-FP16
            self.assertEqual(PL.plan_log_fields(q, p)["gemv"], f["gemv"], dtype)
            self.assertEqual(common.applied_status(f, ""), ("gemv", "the call ran on the %s GEMV kernels" % dtype))
            self.assertEqual(common.applied_status(f, "nr=16")[0], "gemv")
            # The same call at M = 8 runs on the tiles.
            self.assertEqual(PL.plan_log_fields(PL.builtin_plan(p.with_m(8), UARCH, SCHEMA), p.with_m(8))["gemv"],
                             "none", dtype)

    def test_line(self):
        f = common.parse_apilog_line(LINE)
        self.assertEqual((f["M"], f["N"], f["K"], f["source"], f["cfg"]),
                         ("8", "6144", "4096", "override+table", "nr=32,int8_mr=8"))
        self.assertIsNone(common.parse_apilog_line("[MATMUL.NATIVE_BRGEMM PACK] weight=0x1 K=4096 N=6144"))
        self.assertEqual(list(common.parse_apilog("x\n" + LINE + "\n" + LINE)), [(8, 6144, 4096)])

    def test_blocks_line(self):
        # A blocks plan names its unit width, in tiles and in columns, and the threads of a domain;
        # lines that differ only in the domain size are not one plan.
        blocks = LINE.replace(" col_groups=0 ", " col_groups=0 unit_tiles=1 nb=32 domain_threads=8 ")
        f = common.parse_apilog_line(blocks)
        self.assertEqual((f["unit_tiles"], f["nb"], f["domain_threads"]), ("1", "32", "8"))
        self.assertEqual(common.consistent([f, dict(f)]), f)
        self.assertIsNone(common.consistent([f, dict(f, domain_threads="4")]))

    def test_status(self):
        f = common.parse_apilog_line(LINE)
        self.assertEqual(common.applied_status(f, "nr=32,int8_mr=8")[0], "applied")
        self.assertEqual(common.applied_status(f, "nr=16")[0], "mismatch")
        self.assertEqual(common.applied_status(f, "")[0], "mismatch")
        self.assertEqual(common.applied_status(dict(f, source="heuristic", cfg="-"), "nr=16")[0], "dropped")
        self.assertEqual(common.applied_status(dict(f, source="heuristic", cfg="-"), "")[0], "applied")
        self.assertEqual(common.applied_status(None, "nr=16")[0], "no_log")
        old = {k: v for k, v in f.items() if k not in ("source", "cfg")}
        self.assertEqual(common.applied_status(old, "nr=16")[0], "no_log")
        g = common.parse_apilog_line(GEMV_LINE)
        self.assertEqual(common.applied_status(g, ""), ("gemv", "the call ran on the s8 GEMV kernels"))
        self.assertEqual(common.applied_status(g, "nr=16")[0], "gemv")
        self.assertIn("gemv", R.BAD_STATUSES)

    def test_padded_width(self):
        # The line names the call's N and, for N not a multiple of 32, the padded width the
        # plan and the filters use; a drop warning names the padded width.
        p = Problem("s8", 8, 4113, 4096)
        padded = LINE.replace(" N=6144 ", " N=4113 ").replace(" team=32", " team=32 padded_n=4128")
        self.assertEqual(list(common.parse_apilog(padded)), [(p.m, p.n, p.k)])
        self.assertEqual(common.parse_apilog_line(padded)["padded_n"], str(p.plan_n))
        reasons, _ = common.parse_warnings(
            "native BRGEMM: ZENDNNL_MATMUL_NATIVE_CONFIG clause 1 gives an illegal plan for M=8 N=4128 K=4096 "
            "threads=32 s8: mr is outside 1 .. 12; dropped\n")
        self.assertEqual(reasons, {(p.m, p.plan_n, p.k): "mr is outside 1 .. 12"})

    def test_warnings(self):
        text = ("[API    ][warning][1.0]:native BRGEMM: ZENDNNL_MATMUL_NATIVE_CONFIG clause 2 gives an illegal plan "
                "for M=1 N=6144 K=4096 threads=32 s8: nr is not 16 or 32 dividing pack_nr; dropped\n"
                "[API    ][warning][1.0]:native BRGEMM: tuning file entry b conflicts with a higher layer "
                "for M=8 N=6144 K=4096 threads=32 s8: no INT8 kernel for the call rows; dropped\n")
        reasons, lines = common.parse_warnings(text)
        self.assertEqual(reasons, {(1, 6144, 4096): "nr is not 16 or 32 dividing pack_nr",
                                   (8, 6144, 4096): "no INT8 kernel for the call rows"})
        self.assertEqual(len(lines), 2)


class TestBackendInterface(unittest.TestCase):
    def test_every_backend_prices_a_run_over_given_layers(self):
        # A caller may pass the layers of a process to any backend's cost(), as _over() does.
        from native_brgemm_tuning.backends import BACKENDS

        p = Problem("bf16", 8, 4096, 4096)
        for name, cls in sorted(dict(BACKENDS, base=common.Backend).items()):
            be = cls()
            self.assertEqual(list(inspect.signature(be.cost).parameters), ["p", "layers"], name)
            self.assertGreater(be.cost(p, 4), 0, name)
            self.assertEqual(be.cost(p, 0), be.cost(p), name)

    def test_pod_backends_time_the_pack_width_of_the_library(self):
        # The library packs every weight in blocks of 32 columns; a problem of another width
        # cannot be timed, nor its entries match a call.
        from native_brgemm_tuning.backends import BACKENDS

        for name, cls in sorted(BACKENDS.items()):
            be = cls()
            self.assertIsNone(be.supports(Problem("bf16", 8, 4096, 4096)), name)
            for pack in (16, 64):
                why = be.supports(Problem("bf16", 8, 4096, 4096, pack_nr=pack))
                if be.runs_on_pod:
                    self.assertEqual(why, "pack_nr %d: the library's calls pack every weight in blocks of 32 "
                                          "columns" % pack, name)
                else:
                    self.assertIsNone(why, name)
        with self.assertRaises(ValueError) as e:
            jobs.Session("never_written", ReplayBackend(), [Problem("bf16", 8, 4096, 4096, pack_nr=64)], {}, "x")
        self.assertIn("pack_nr 64", str(e.exception))

    def test_groups_keep_apart_the_problems_of_one_filter(self):
        # N 4113 and 4128 are planned and matched at 4128: one override cannot set both.
        probs = [Problem("bf16", m, n, 4096) for m in (8, 16) for n in (4096, 4113, 4128)]
        for be in (ReplayBackend(), BenchdnnBackend()):
            groups = be.groups(probs)
            self.assertEqual(sorted(p.key() for g in groups for p in g), sorted(p.key() for p in probs), be.name)
            for g in groups:
                self.assertEqual(len({C.problem_filters(p) for p in g}), len(g), (be.name, [p.label() for p in g]))
        # Without such a pair, the Ms of the same shapes still share one process.
        self.assertEqual([len(g) for g in ReplayBackend().groups([p for p in probs if p.n != 4113])], [4])
        self.assertEqual([len(g) for g in BenchdnnBackend().groups([p for p in probs if p.n != 4113])], [4])


class TestBenchdnn(unittest.TestCase):
    def test_lines_match_the_native_input_files(self):
        self.assertEqual(bd_line(Problem("bf16", 1, 4096, 4096), 100, 20),
                         "1, 4096, 4096, 100, bf16:bf16:bf16, false, f32, , , none, true, false, false, "
                         "1.0, 0.0, per-channel, , f32, 20")
        self.assertEqual(bd_line(Problem("s8", 8, 6144, 4096), 100, 20),
                         "8, 4096, 6144, 100, bf16:s8:bf16, false, f32, , , none, true, false, false, "
                         "1.0, 0.0, per-channel, , f32, 20, true, per-token, 0, f32")

    def test_result_line(self):
        r = parse_result_line("1, 4096, 4096, 100, bf16:bf16:bf16, 0, , , , none, 1, 0, 0, 1, 0, none, , , 20, 0, "
                              ", , , 2.63048")
        self.assertEqual((r["m"], r["n"], r["k"], r["wei"]), (1, 4096, 4096, "bf16"))
        self.assertAlmostEqual(r["time_ms"], 0.0263048)
        self.assertIsNone(parse_result_line("[TEST   ][error  ][11.2]:LOWOHA: Matmul execution failed."))

    def test_supports(self):
        be = BenchdnnBackend()
        self.assertIsNone(be.supports(Problem("s8", 1, 4096, 4096)))
        self.assertIsNone(be.supports(Problem("bf16", 1, 4096, 4096)))
        self.assertIsNotNone(be.supports(Problem("s8", 1, 4096, 4096, src="u8")))
        self.assertIsNotNone(be.supports(Problem("bf16", 1, 4096, 4096, threads=64)))
        # The two lines take bf16 weights and s8 weights with per-channel scales: an s4 or a
        # per-group s8 problem would run the per-channel s8 line.
        for p in (Problem("s4", 1, 4096, 4096, group=128), Problem("s4", 1, 4096, 4096, group=4096),
                  Problem("s8", 1, 4096, 4096, int8_scheme="per_group"), Problem("f16", 1, 4096, 4096, src="bf16"),
                  Problem("f32", 1, 4096, 4096, src="bf16")):
            self.assertIn("per-channel", be.supports(p), p.key())


class TestReplay(unittest.TestCase):
    def test_parsers(self):
        be = ReplayBackend()
        p = Problem("s8", 8, 6144, 4096)
        text = ("CONFIG algo=10 threads=32\n"
                "RESULT shape=n6144k4096 N=6144 K=4096 M=8 calls=120 median_ms=0.1234 mean_ms=0.13 min_ms=0.12 "
                "p90_ms=0.14\nCHECK shape=n6144k4096 N=6144 K=4096 M=8 hash=00AB12\n")
        self.assertEqual(be.parse_timed(text, [p])[p.key()]["time_ms"], 0.1234)
        self.assertEqual(be.parse_probe(text, [p])[p.key()]["hash"], "00ab12")

    def test_groups(self):
        be = ReplayBackend()
        probs = [Problem("s8", 1, 256 * (i + 1), 4096) for i in range(70)] + [Problem("bf16", 8, 4096, 4096)]
        groups = be.groups(probs)
        self.assertEqual(sorted(len(g) for g in groups), [1, 6, 64])
        same = be.groups([Problem("s8", m, n, 4096) for m in (1, 8, 16) for n in (4096, 6144)])
        self.assertEqual([len(g) for g in same], [6])
        self.assertIn("--ms 1,8,16", be.command(0, same[0], False))

    def test_group_caps(self):
        probs = [Problem("bf16", m, 4096 * (i + 1), 4096) for m in (1031, 4096) for i in range(8)]
        whole = ReplayBackend().groups(probs)
        capped = ReplayBackend(group_cost=10.0).groups(probs)
        self.assertEqual(sorted(map(id, sum(whole, []))), sorted(map(id, sum(capped, []))))
        self.assertGreater(len(capped), len(whole))
        be = ReplayBackend(group_cost=10.0)
        for g in capped:
            self.assertEqual(len({p.m for p in g}), 1)
            self.assertTrue(len(g) == 1 or sum(be.cost(p) for p in g) <= 10.0)
        gb = ReplayBackend(group_gb=1.0)
        timed = [p for p in probs if gb.supports(p) is None]
        self.assertEqual(len(timed), len(probs) - 2)
        for g in gb.groups(timed):
            self.assertLessEqual(gb.process_gb(g), 1.0)
        self.assertEqual(ReplayBackend(group_cost=1e9, group_gb=1e9).groups(probs), whole)
        self.assertEqual(ReplayBackend(group_cost=3.0).describe()["group_cost"], 3.0)

    def test_caps_count_the_layers_a_process_times(self):
        # The reviewer's two s8 shapes, 12.3 MB of weights: the cold floor times them over 21 layers.
        small = [Problem("s8", 1, 2880, 2880), Problem("s8", 1, 1408, 2816)]
        with self.assertRaises(ValueError):
            ReplayBackend(layers=1, cold_mb=256.0, group_gb=0.1)
        be = ReplayBackend(layers=1, cold_mb=256.0, group_gb=0.3)
        self.assertAlmostEqual(be.process_gb(small), 21 * (2880 * 2880 + 1408 * 2816) / 1e9)
        self.assertEqual([len(g) for g in be.groups(small)], [2])
        # bf16 weights of 33.6 and 134.2 MB at two Ms: a process of both shapes takes 2 layers
        # (336 MB), one of either shape 8 or 2 layers (268 MB).
        probs = [Problem("bf16", m, n, 4096) for m in (1, 8) for n in (4096, 16384)]
        be = ReplayBackend(layers=1, cold_mb=256.0, group_gb=0.27)
        self.assertGreater(be.process_gb(probs), 0.27)
        groups = be.groups(probs)
        self.assertEqual([len(g) for g in groups], [1, 1, 1, 1])
        for g in groups:
            self.assertLessEqual(be.process_gb(g), 0.27)
            self.assertIn("--layers %d " % (8 if g[0].n == 4096 else 2), be.command(0, g, False))
        # A problem whose process alone is above the cap: 117 MB takes 3 layers (352 MB).
        self.assertIn("above group_gb 0.27", be.supports(Problem("bf16", 8, 14336, 4096)))
        self.assertIsNone(be.supports(probs[0]))
        # The time cap prices a process at its layers too.
        timed = ReplayBackend(layers=1, cold_mb=256.0)
        self.assertGreater(timed.cost(small[0], timed.timed_layers(small)), 10 * timed.cost(small[0]))

    def test_small_shapes_take_more_layers(self):
        small = [Problem("s8", 1, 2880, 2880), Problem("s8", 1, 1408, 2816)]
        big = [Problem("bf16", 8, 32768, 4096)]
        be = ReplayBackend(layers=1, cold_mb=256.0)
        self.assertEqual(be.timed_layers(small), 21)
        self.assertEqual(be.timed_layers(big), 1)
        self.assertEqual(be.timed_layers([Problem("s4", 1, 2880, 2880, group=64)]), 62)
        self.assertIn("--layers 21 ", be.command(0, small, False))
        self.assertIn("--layers 1 ", be.command(0, small, True))
        self.assertIn("--layers 1 ", ReplayBackend(layers=1).command(0, small, False))
        self.assertEqual(be.describe()["cold_mb"], 256.0)

    def test_preamble_binds_the_first_cpus(self):
        lines = ReplayBackend(threads=4).preamble()
        trim = [line for line in lines if "head -n" in line]
        self.assertEqual(len(trim), 1)
        out = subprocess.check_output(["bash", "-c", "LIST=8,9,10,11,12,13; %s; echo $LIST" % trim[0]])
        self.assertEqual(out.decode().strip(), "8,9,10,11")
        self.assertIn('OMP_NUM_THREADS=4 KMP_AFFINITY="granularity=fine,explicit,proclist=[$LIST]"', lines[4])

    def test_supports(self):
        be = ReplayBackend()
        self.assertIsNone(be.supports(Problem("bf16", 8, 4096, 4096)))
        self.assertIsNotNone(be.supports(Problem("bf16", 8, 4096, 4096, has_bias=True, epilogue="bias")))
        self.assertIsNotNone(be.supports(Problem("bf16", 8, 4096, 4096, trans_a=True)))
        # s4 in groups or one group, f16 calls of f16 operands; not f32, per-group s8 or a bf16 dst of f16.
        self.assertIsNone(be.supports(Problem("s4", 8, 4096, 4096, group=128)))
        self.assertIsNone(be.supports(Problem("s4", 8, 4096, 4096, group=4096)))
        self.assertIsNone(be.supports(Problem("f16", 8, 4096, 4096)))
        self.assertIsNotNone(be.supports(Problem("f16", 8, 4096, 4096, dst="bf16")))
        self.assertIsNotNone(be.supports(Problem("f32", 8, 4096, 4096)))
        self.assertIsNotNone(be.supports(Problem("s8", 8, 4096, 4096, int8_scheme="per_group")))
        # The scales of an s4 problem are the backend's scale type.
        self.assertIsNotNone(be.supports(Problem("s4", 8, 4096, 4096, group=128, scale_bytes=4)))
        self.assertIsNone(ReplayBackend(scale_dt="f32").supports(Problem("s4", 8, 4096, 4096, group=128,
                                                                         scale_bytes=4)))

    def test_s4_and_f16_commands(self):
        be = ReplayBackend()
        probs = [Problem("s4", 8, n, k, group=g) for n, k, g in
                 ((4096, 4096, 128), (6144, 4096, 128), (4096, 14336, 14336), (4096, 4096, 4096), (2880, 2880, 64))]
        probs.append(Problem("f16", 8, 4096, 4096))
        groups = be.groups(probs)
        commands = sorted(be.command(i, g, False) for i, g in enumerate(groups))
        self.assertEqual(len(groups), 4)
        self.assertTrue(any("--wei s4" in c and "--group -1" in c and "n4096k14336" in c and "n4096k4096" in c
                            for c in commands))
        self.assertTrue(any("--wei s4" in c and "--group 128 " in c and "n6144k4096" in c for c in commands))
        self.assertTrue(any("--wei s4" in c and "--group 64 " in c for c in commands))
        self.assertTrue(any("--wei f16" in c and "--group" not in c for c in commands))
        # The override clause of an s4 problem selects its group size too.
        self.assertEqual(C.problem_filters(probs[0]), "dtype=s4,m=8,n=4096,k=4096,threads=32,group=128")
        with self.assertRaises(ValueError):
            be.command(0, [probs[0], probs[2]], False)

    def test_gated_commands(self):
        # The plugin's gated linear over a [gate | up] weight, in processes of its own.
        be = ReplayBackend()
        gated = [Problem(d, 8, 24576, 4096, epilogue="gated", group=128 if d == "s4" else 0)
                 for d in ("bf16", "s8", "s4")]
        for p in gated:
            self.assertIsNone(be.supports(p), p.dtype)
        self.assertIn("gated", be.supports(Problem("f16", 8, 24576, 4096, epilogue="gated")))
        self.assertIn("gated", be.supports(Problem("bf16", 8, 4097, 4096, epilogue="gated")))
        self.assertIsNotNone(be.supports(Problem("bf16", 8, 4096, 4096, epilogue="postops")))
        # A store problem of another shape, which would share the weight type's process.
        store = [Problem(p.dtype, 8, 4096, 4096, group=p.group) for p in gated]
        groups = be.groups(gated + store)
        self.assertEqual(sorted((g[0].dtype, g[0].epilogue, len(g)) for g in groups),
                         sorted((p.dtype, p.epilogue, 1) for p in gated + store))
        for g in groups:
            for probe in (False, True):
                self.assertEqual("--gated silu_and_mul" in be.command(0, g, probe), g[0].epilogue == "gated")
        with self.assertRaises(ValueError):
            be.command(0, [gated[1], store[1]], False)


class TestReplayBuild(unittest.TestCase):
    """replay/build.sh (and plan_dump/build.sh) link the dependency archives the build installed."""

    UTILS, DLP, DNN, XSMM, PLX = ("aoclutils/lib/libaoclutils.a", "aocldlp/lib/libaocl-dlp.a",
                                  "onednn/lib/libdnnl.a", "libxsmm/lib/libxsmm.a", "parlooper/lib/libparlooper.a")

    def build(self, archives, program="replay"):
        """<program>/build.sh on a build tree holding the zendnnl archive and archives (under install/deps),
        with a compiler that records its arguments: (exit status, stderr, arguments)."""
        root = tempfile.mkdtemp(prefix="nbt_build_")
        self.addCleanup(shutil.rmtree, root)
        b = os.path.join(root, "build")
        for rel in ["zendnnl/src/libzendnnl_archive.a"] + ["install/deps/" + a for a in archives]:
            os.makedirs(os.path.dirname(os.path.join(b, rel)), exist_ok=True)
            open(os.path.join(b, rel), "w").close()
            if rel.startswith("install/deps/"):
                os.makedirs(os.path.join(b, "install/deps", rel.split("/")[2], "include"), exist_ok=True)
        os.makedirs(os.path.join(b, "install/zendnnl/include"))
        cxx = os.path.join(root, "cxx")
        with open(cxx, "w") as f:
            f.write('#!/bin/bash\nprintf "%s\\n" "$@" > "$(dirname "$0")/args"\n')
        os.chmod(cxx, 0o755)
        script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), program, "build.sh")
        argv = [b, os.path.join(root, "out")] if program == "replay" else [root, os.path.join(root, "out"), b]
        r = subprocess.run(["bash", script] + argv, env=dict(os.environ, CXX=cxx),
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        args = []
        if os.path.exists(os.path.join(root, "args")):
            with open(os.path.join(root, "args")) as f:
                args = [a.replace(os.path.realpath(b) + "/install/deps/", "") for a in f.read().split("\n") if a]
        return r.returncode, r.stderr, args

    def whole(self, args):
        """The archives linked whole, in order."""
        return args[args.index("-Wl,--push-state,--whole-archive") + 1:args.index("-Wl,--pop-state")]

    def test_the_archives_a_build_installed(self):
        rc, _, args = self.build([self.UTILS, self.DLP, self.DNN, self.XSMM])
        self.assertEqual(rc, 0)
        self.assertEqual(self.whole(args), [self.UTILS, self.DLP, self.DNN, self.XSMM])
        self.assertIn("-Ilibxsmm/include", args)
        # The defaults leave LibXSMM out; a build can leave out oneDNN and the DLP library too.
        rc, _, args = self.build([self.UTILS, self.DLP, self.DNN])
        self.assertEqual((rc, self.whole(args)), (0, [self.UTILS, self.DLP, self.DNN]))
        self.assertNotIn("-Ilibxsmm/include", args)
        rc, _, args = self.build([self.UTILS, self.XSMM])
        self.assertEqual((rc, self.whole(args)), (0, [self.UTILS, self.XSMM]))
        rc, _, args = self.build([self.UTILS])
        self.assertEqual((rc, self.whole(args)), (0, [self.UTILS]))
        # The utilities archive is required.
        rc, err, args = self.build([self.DLP, self.DNN])
        self.assertEqual((rc, args), (1, []))
        self.assertIn("libaoclutils.a: build zendnnl first", err)

    def test_parlooper(self):
        # With LibXSMM and ParLooper on, the partitioner's threaded loops need libparlooper.a; no
        # installed zendnnl header includes ParLooper's.
        every = [self.UTILS, self.DLP, self.DNN, self.XSMM, self.PLX]
        for program in ("replay", "plan_dump"):
            rc, _, args = self.build(every, program)
            self.assertEqual((rc, self.whole(args)), (0, every), program)
        self.assertNotIn("-Iparlooper/include", args)


class TestSession(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="nbt_test_")
        self.addCleanup(shutil.rmtree, self.tmp)
        self.probs = [Problem("s8", 8, n, 4096) for n in (4096, 6144, 8192)]
        self.spaces = {p.key(): C.Space(p, SCHEMA, UARCH) for p in self.probs}
        self.assign = {p.key(): [self.spaces[p.key()].make({"nr": 16}), self.spaces[p.key()].make({"threads_cap": 16})]
                       for p in self.probs}
        self.defaults = {k: s.default for k, s in self.spaces.items()}

    def session(self, backend, sub="s", alone=False):
        return jobs.Session(os.path.join(self.tmp, sub), backend, self.probs, self.assign, "s001_0", repeats=2,
                            defaults=self.defaults, schema=SCHEMA, alone=alone)

    def test_order_and_clauses(self):
        s = self.session(ReplayBackend(binary="/opt/replay"))
        s.write()
        runs = s.manifest["runs"]
        self.assertEqual([(r["kind"], r["slot"]) for r in runs],
                         [("probe", 0), ("probe", 1), ("probe", 2)]
                         + [("timed", j) for j in (0, 1, 2, 0)] + [("timed", j) for j in (0, 2, 1, 0)])
        with open(os.path.join(s.dir, "cfg", "g0_s1.cfg")) as f:
            cfg = f.read().strip()
        self.assertEqual(cfg.split(";")[0], "dtype=s8,m=8,n=4096,k=4096,threads=32:nr=16")
        self.assertEqual(len(cfg.split(";")), 3)
        self.assertEqual(len(C.parse_override(cfg, SCHEMA)), 3)
        self.assertEqual(subprocess.call(["bash", "-n", s.script]), 0)
        self.assertEqual(s.launch_command("/tools/run_one.sh --cmd {dir} -- bash {script}"),
                         "/tools/run_one.sh --cmd %s -- bash %s" % (s.dir, s.script))

    def test_no_override_has_two_clauses_of_one_filter(self):
        # The library applies the first clause whose filters hold: a second clause of the same
        # filters would leave its candidate unapplied, a mismatch.
        probs = [Problem("bf16", m, n, 4096) for m in (8, 16) for n in (4113, 4128)]
        spaces = {p.key(): C.Space(p, SCHEMA, UARCH) for p in probs}
        assign = {p.key(): [spaces[p.key()].make({"nr": 16 if p.n == 4113 else 32, "kc": 1024})] for p in probs}
        for be in (ReplayBackend(binary="/opt/replay"), BenchdnnBackend(binary="/opt/benchdnn")):
            s = jobs.Session(os.path.join(self.tmp, be.name), be, probs, assign, "s001_0",
                             defaults={k: sp.default for k, sp in spaces.items()}, schema=SCHEMA)
            s.write()
            cfgs = sorted(os.listdir(os.path.join(s.dir, "cfg")))
            self.assertEqual(len(cfgs), len(s.groups), be.name)
            for name in cfgs:
                with open(os.path.join(s.dir, "cfg", name)) as f:
                    filters = [c.partition(":")[0] for c in f.read().strip().split(";")]
                self.assertEqual(len(filters), len(set(filters)), (be.name, name, filters))

    def test_benchdnn_session(self):
        s = self.session(BenchdnnBackend(binary="/opt/benchdnn"), "b")
        s.write()
        self.assertEqual(subprocess.call(["bash", "-n", s.script]), 0)
        with open(s.script) as f:
            script = f.read()
        self.assertIn("bdl inputs/s8_m8_n4096_k4096_", script)
        self.assertEqual(len(os.listdir(os.path.join(s.dir, "inputs"))), 6)
        with self.assertRaises(ValueError):
            jobs.Session(self.tmp, BenchdnnBackend(), [Problem("s8", 1, 4096, 4096, src="u8")], {}, "x")

    def test_collect_statuses(self):
        be = ReplayBackend(binary="/opt/replay")
        s = self.session(be)
        s.write()
        man = s.manifest
        p1, p2, p3 = self.probs
        times = {0: 1.0, 1: 0.8, 2: 0.9}
        for r in man["runs"]:
            slot = man["slots"]["0"][r["slot"]]
            lines = []
            for p in self.probs:
                if p.key() not in slot:
                    continue
                exp = dict(slot[p.key()]["expect"])
                text = slot[p.key()]["text"]
                if r["kind"] == "timed":
                    lines.append("RESULT shape=x N=%d K=4096 M=%d calls=90 median_ms=%.3f mean_ms=1 min_ms=1 p90_ms=1"
                                 % (p.n, p.m, times[r["slot"]]))
                    continue
                src, cfg, h = ("override", text, "aaaa") if text else ("heuristic", "-", "aaaa")
                if r["slot"] == 1 and p is p2:
                    src, cfg = "heuristic", "-"
                    lines.append("native BRGEMM: clause 2 gives an illegal plan for M=8 N=6144 K=4096 threads=32 "
                                 "s8: no kernel; dropped")
                if r["slot"] == 1 and p is p3:
                    cfg = "nr=32"
                if r["slot"] == 2 and p is p1:
                    h = "bbbb"
                    exp["tiles"] = "1"
                if r["slot"] == 2 and p is p3:
                    continue
                lines.append(log_line(p, exp, src, cfg))
                lines.append("CHECK shape=x N=%d K=4096 M=%d hash=%s" % (p.n, p.m, h))
            with open(os.path.join(s.dir, "out", r["id"] + ".txt"), "w") as f:
                f.write("\n".join(lines) + "\n")
        recs = jobs.collect(s.dir)
        st = {(r["pkey"], r["text"]): (r["status"], r["reason"], r["model_diff"]) for r in recs if r["kind"] == "probe"}
        self.assertEqual(st[(p1.key(), "nr=16")][:2], ("applied", ""))
        self.assertEqual(st[(p1.key(), "nr=16")][2], [])
        self.assertEqual(st[(p2.key(), "nr=16")][:2], ("dropped", "no kernel"))
        self.assertEqual(st[(p3.key(), "nr=16")][0], "mismatch")
        self.assertEqual(st[(p1.key(), "threads_cap=16")][0], "bits_changed")
        self.assertEqual(st[(p2.key(), "threads_cap=16")][0], "applied")
        self.assertEqual(st[(p3.key(), "threads_cap=16")][0], "no_log")
        self.assertEqual(st[(p1.key(), "")][0], "applied")
        summ = R.summarize(recs)
        self.assertAlmostEqual(summ[p1.key()].cands["nr=16"].ratio, 0.8)
        self.assertNotIn("threads_cap=16", summ[p1.key()].cands)
        self.assertEqual(sorted(summ[p3.key()].excluded), ["nr=16", "threads_cap=16"])
        self.assertEqual(sum(1 for r in recs if r["kind"] == "measure"), 2 * (4 * 3))

    def outputs(self, s, times, edit=None):
        """Every run's output of session s: a probe logs the plan its slot expects with the
        built-in hash (edit(slot, p, line) may change the line's fields, source and cfg), and
        a timed run takes times[slot] ms."""
        man = s.manifest
        for r in man["runs"]:
            slot = man["slots"][str(r["group"])][r["slot"]]
            lines = []
            for p in self.probs:
                if p.key() not in slot:
                    continue
                text = slot[p.key()]["text"]
                if r["kind"] == "timed":
                    lines.append("RESULT shape=x N=%d K=4096 M=%d calls=90 median_ms=%.3f mean_ms=1 min_ms=1 p90_ms=1"
                                 % (p.n, p.m, times[r["slot"]]))
                    continue
                line = {"fields": dict(slot[p.key()]["expect"]), "source": "override" if text else "heuristic",
                        "cfg": text or "-"}
                if edit:
                    edit(r["slot"], p, line)
                lines.append(log_line(p, line["fields"], line["source"], line["cfg"]))
                lines.append("CHECK shape=x N=%d K=4096 M=%d hash=aaaa" % (p.n, p.m))
            with open(os.path.join(s.dir, "out", r["id"] + ".txt"), "w") as f:
                f.write("\n".join(lines) + "\n")

    def test_a_plan_other_than_the_models_is_a_mismatch(self):
        # The library applied nr=16 for p1, with the built-in bits, but another tile count than
        # the plan model's: an entry would name a plan that was not timed.
        s = self.session(ReplayBackend(binary="/opt/replay"))
        s.write()
        p1, p2 = self.probs[:2]

        def other_tiles(j, p, line):
            if j == 1 and p is p1:
                line["fields"]["tiles"] = "1"

        self.outputs(s, {0: 1.0, 1: 0.8, 2: 0.9}, other_tiles)
        recs = jobs.collect(s.dir)
        probe = [r for r in recs if r["kind"] == "probe" and (r["pkey"], r["text"]) == (p1.key(), "nr=16")][0]
        self.assertEqual(probe["status"], "applied")
        self.assertTrue(probe["model_diff"])
        st, why = R.probe_status(recs)[(p1.key(), "nr=16")]
        self.assertEqual(st, "mismatch")
        self.assertIn("tiles: model ", why)
        summ = R.summarize(recs)
        self.assertNotIn("nr=16", summ[p1.key()].cands)
        self.assertTrue(summ[p1.key()].excluded["nr=16"].startswith("mismatch: the library ran another plan"))
        self.assertAlmostEqual(summ[p2.key()].cands["nr=16"].ratio, 0.8)
        self.assertAlmostEqual(summ[p1.key()].cands["threads_cap=16"].ratio, 0.9)

    def test_a_gated_probe_is_compared_at_its_pairing(self):
        # A gated call logs the plan its tiles ran: paired wider (pairing=64 for tiles of 16 or 32
        # columns), or the plan's own when it staged its rows (N not whole 32-column blocks).
        from native_brgemm_tuning import plan as PL

        self.probs = [Problem("s8", 8, n, 4096, epilogue="gated") for n in (24576, 28672, 4112)]
        self.spaces = {p.key(): C.Space(p, SCHEMA, UARCH) for p in self.probs}
        self.assign = {p.key(): [self.spaces[p.key()].make({"mb": 4})] for p in self.probs}
        self.defaults = {k: sp.default for k, sp in self.spaces.items()}
        paired, other, staged = self.probs

        def collect(pairing, alone=False):
            """The records of a session (alone: a slot per candidate, as confirm --alone runs) whose
            library pairs the halves of the first problem at pairing."""
            s = self.session(ReplayBackend(binary="/opt/replay"), "pairing_%s_%s" % (pairing, alone), alone)
            s.write()

            def library_line(j, p, line):
                q = self.defaults[p.key()].plan if j == 0 else self.assign[p.key()][0].plan
                w = pairing if p is paired else "staged" if p is staged else "64"
                if w == "staged":
                    line["fields"]["pairing"] = "staged"
                    return
                line["fields"] = dict(PL.plan_log_fields(PL.paired_plan(q, p, int(w)), p), pairing=w)
                if j > 0 and p is other:
                    line["fields"]["tiles"] = "1"

            self.outputs(s, {j: 0.8 if j else 1.0 for j in range(4)}, library_line)
            return jobs.collect(s.dir)

        recs = collect("64")
        probes = {(r["pkey"], r["text"]): r for r in recs if r["kind"] == "probe"}
        for p in self.probs:
            self.assertEqual(probes[(p.key(), "")]["model_diff"], [], p.n)
        probe = probes[(paired.key(), "mb=4")]
        self.assertEqual(probe["model_diff"], [])
        planned = PL.plan_log_fields(self.assign[paired.key()][0].plan, paired)
        self.assertIn("nr: model 32, log 64", PL.compare_fields(planned, probe["logged"]))
        self.assertTrue(probes[(other.key(), "mb=4")]["model_diff"])
        self.assertEqual(probes[(staged.key(), "mb=4")]["model_diff"], [])
        summ = R.summarize(recs)
        self.assertAlmostEqual(summ[paired.key()].cands["mb=4"].ratio, 0.8)
        self.assertAlmostEqual(summ[staged.key()].cands["mb=4"].ratio, 0.8)
        self.assertNotIn("mb=4", summ[other.key()].cands)
        # The model pairs the first problem's tiles at 64: a library that pairs them at 32, or stages
        # its rows, logs the plan as planned, which the fields alone would take for the model's.  So
        # does a slot of its own (confirm --alone).
        for pairing in ("32", "staged"):
            for alone in (False, True):
                recs = collect(pairing, alone)
                probes = {(r["pkey"], r["text"]): r for r in recs if r["kind"] == "probe"}
                for text in ("", "mb=4"):
                    self.assertEqual(probes[(paired.key(), text)]["model_diff"],
                                     ["pairing: model 64, log %s" % pairing], (pairing, alone, text))
                self.assertEqual(probes[(staged.key(), "mb=4")]["model_diff"], [], (pairing, alone))
                self.assertEqual(R.probe_status(recs)[(paired.key(), "mb=4")][0], "mismatch", (pairing, alone))
                self.assertNotIn("mb=4", R.summarize(recs)[paired.key()].cands, (pairing, alone))

    def test_a_built_in_run_at_another_plan_than_the_models_stops_its_problem(self):
        # The library pairs the built-in tiles of the first problem at 32 where the model pairs them at
        # 64, while the candidate's line is the model's.  The model's built-in plan, against which the
        # search de-duplicates, is not what the library runs: the problem's candidates are not judged,
        # and the search gives it no more.
        from native_brgemm_tuning import plan as PL
        from native_brgemm_tuning.search import Searcher, SearchSettings

        self.probs = [Problem("s8", 8, n, 4096, epilogue="gated") for n in (24576, 4112)]
        self.spaces = {p.key(): C.Space(p, SCHEMA, UARCH) for p in self.probs}
        self.assign = {p.key(): [self.spaces[p.key()].make({"mb": 4})] for p in self.probs}
        self.defaults = {k: sp.default for k, sp in self.spaces.items()}
        paired, staged = self.probs
        s = self.session(ReplayBackend(binary="/opt/replay"))
        s.write()

        def library_line(j, p, line):
            if p is staged:
                line["fields"]["pairing"] = "staged"
            elif j == 0:
                line["fields"]["pairing"] = "32"
            else:
                q = self.assign[p.key()][0].plan
                line["fields"] = dict(PL.plan_log_fields(PL.paired_plan(q, p, 64), p), pairing="64")

        self.outputs(s, {0: 1.0, 1: 0.8}, library_line)
        recs = jobs.collect(s.dir)
        probes = {(r["pkey"], r["text"]): r for r in recs if r["kind"] == "probe"}
        self.assertEqual(probes[(paired.key(), "")]["model_diff"], ["pairing: model 64, log 32"])
        self.assertEqual(probes[(paired.key(), "mb=4")]["model_diff"], [])
        status = R.probe_status(recs)
        self.assertEqual(status[(paired.key(), "")],
                         ("mismatch", "the library ran another plan than the plan model's: pairing: model 64, log 32"))
        self.assertEqual(status[(paired.key(), "mb=4")][0], "applied")
        summ = R.summarize(recs)
        self.assertEqual(summ[paired.key()].cands, {})
        self.assertEqual(summ[paired.key()].excluded["mb=4"],
                         "mismatch: the built-in run: the library ran another plan than the plan model's: "
                         "pairing: model 64, log 32")
        self.assertAlmostEqual(summ[staged.key()].cands["mb=4"].ratio, 0.8)
        db = R.ResultsDB(os.path.join(self.tmp, "results.jsonl"))
        db.append(recs)
        searcher = Searcher(self.probs, self.spaces, None, db, SearchSettings(strategy="grid"), log=lambda m: None)
        searcher.plan_round()
        self.assertEqual(sorted(searcher.held), [paired.key()])
        self.assertEqual(searcher.want(paired.key()), 0)
        self.assertGreater(searcher.want(staged.key()), 0)

    def test_a_probe_without_output_is_retried(self):
        # A session that left no output (collect --force records its probes as failed) is
        # proposed again; the second session's results count, in either order of the records.
        be = ReplayBackend(binary="/opt/replay")
        first = self.session(be, "a")
        first.write()
        lost = jobs.collect(first.dir)
        self.assertTrue(all(r["status"] == "failed" for r in lost))
        self.assertEqual(dict(R.measured_texts(lost)), {})
        again = jobs.Session(os.path.join(self.tmp, "b"), be, self.probs, self.assign, "s002_0", repeats=2,
                             defaults=self.defaults, schema=SCHEMA)
        again.write()
        self.outputs(again, {0: 1.0, 1: 0.8, 2: 0.9})
        ran = jobs.collect(again.dir)
        p1 = self.probs[0]
        for recs in (lost + ran, ran + lost):
            summ = R.summarize(recs)
            self.assertEqual(summ[p1.key()].excluded, {})
            self.assertAlmostEqual(summ[p1.key()].cands["nr=16"].ratio, 0.8)
        # A probe that answered with a rejection keeps the candidate out.
        dropped = [dict(r, session="s003_0", status="dropped", reason="no kernel") for r in ran
                   if r["kind"] == "probe" and (r["pkey"], r["text"]) == (p1.key(), "nr=16")]
        for recs in (lost + dropped + ran, lost + ran + dropped, ran + dropped + lost):
            self.assertEqual(R.summarize(recs)[p1.key()].excluded, {"nr=16": "dropped: no kernel"})

    def test_alone_slots_time_the_whole_group(self):
        # One candidate a slot, every run of the group timing all its problems (the others at their
        # built-in plans), and only the slot's problems recorded.
        s = jobs.Session(os.path.join(self.tmp, "alone"), ReplayBackend(binary="/opt/replay"), self.probs, self.assign,
                         "c001_0", phase="confirm", repeats=2, defaults=self.defaults, schema=SCHEMA, alone=True)
        s.write()
        man = s.manifest
        self.assertTrue(man["alone"])
        slots = man["slots"]["0"]
        self.assertEqual(sorted((k, v["text"]) for slot in slots[1:] for k, v in slot.items()),
                         sorted((p.key(), c.text) for p in self.probs for c in self.assign[p.key()]))
        for j in range(1, len(slots)):
            with open(os.path.join(s.dir, "cfg", "g0_s%d.cfg" % j)) as f:
                self.assertEqual(len(f.read().strip().split(";")), 1)
        keys = sorted(p.key() for p in self.probs)
        for r in man["runs"]:
            self.assertEqual((sorted(r["pkeys"]), r["record"]), (keys, list(slots[r["slot"]])))
        self.assertEqual(subprocess.call(["bash", "-n", s.script]), 0)
        for r in man["runs"]:
            slot, lines = slots[r["slot"]], []
            for k in r["pkeys"]:
                p, text = Problem.parse(k), slot.get(k, {}).get("text", "")
                if r["kind"] == "timed":
                    lines.append("RESULT shape=x N=%d K=4096 M=%d calls=90 median_ms=%.3f mean_ms=1 min_ms=1 p90_ms=1"
                                 % (p.n, p.m, 0.5 if text else 1.0))
                    continue
                lines.append(log_line(p, (slot.get(k) or slots[0][k])["expect"], "override" if text else "heuristic",
                                      text or "-"))
                lines.append("CHECK shape=x N=%d K=4096 M=%d hash=aaaa" % (p.n, p.m))
            with open(os.path.join(s.dir, "out", r["id"] + ".txt"), "w") as f:
                f.write("\n".join(lines) + "\n")
        recs = jobs.collect(s.dir)
        self.assertEqual(sum(1 for r in recs if r["kind"] == "probe"), 3 + 6)
        self.assertEqual(sum(1 for r in recs if r["kind"] == "measure"), 2 * (2 * 3 + 6))
        summ = R.summarize(recs, phases=["confirm"])
        for p in self.probs:
            self.assertEqual({t: round(c.ratio, 3) for t, c in summ[p.key()].cands.items()},
                             {c.text: 0.5 for c in self.assign[p.key()]})
        ev = jobs.PodEvaluator(ReplayBackend(binary="/opt/replay"), os.path.join(self.tmp, "ev"),
                               {p.key(): p for p in self.probs}, self.defaults, schema=SCHEMA, alone=True)
        with self.assertRaises(jobs.DryRun) as cm:
            ev(1, "confirm", self.assign)
        with open(os.path.join(cm.exception.dirs[0], "manifest.json")) as f:
            self.assertTrue(json.load(f)["alone"])

    def test_dry_run_and_pending(self):
        ev = jobs.PodEvaluator(ReplayBackend(binary="/opt/replay"), self.tmp, {p.key(): p for p in self.probs},
                               self.defaults, launcher=None, parallel=2, schema=SCHEMA)
        with self.assertRaises(jobs.DryRun) as cm:
            ev(1, "search", self.assign)
        self.assertEqual(len(cm.exception.dirs), 2)
        self.assertEqual(jobs.pending_sessions(self.tmp), sorted(cm.exception.dirs))
        for d in cm.exception.dirs:
            recs = jobs.collect(d)
            self.assertTrue(all(r["status"] == "failed" for r in recs))
            jobs.mark_collected(d, len(recs))
        self.assertEqual(jobs.pending_sessions(self.tmp), [])
        with open(os.path.join(cm.exception.dirs[0], "manifest.json")) as f:
            self.assertEqual(json.load(f)["backend"]["binary"], "/opt/replay")

    def test_pods_of_a_round(self):
        ev = jobs.PodEvaluator(ReplayBackend(binary="/opt/replay"), self.tmp, {p.key(): p for p in self.probs},
                               self.defaults, launcher="/tools/run_one.sh --pod {pod} --cmd {dir} -- bash {script}",
                               parallel=4, schema=SCHEMA, dry_run=True, pods=["pod-a", "pod-b"])
        # One session per pod.
        self.assertEqual(ev.parallel, 2)
        with self.assertRaises(jobs.DryRun) as cm:
            ev(1, "search", self.assign)
        self.assertEqual(len(cm.exception.dirs), 2)
        self.assertEqual([c.split()[2] for c in cm.exception.commands], ["pod-a", "pod-b"])
        # The launcher's environment names the session's pod.
        s = self.session(ReplayBackend(binary="/opt/replay"), "p")
        s.write()
        proc = s.launch("sh -c 'echo \"$POD {pod}\" > \"$1/pod.txt\"' sh {dir}", 60, "pod-c")
        self.assertEqual(proc.wait(), 0)
        with open(os.path.join(s.dir, "pod.txt")) as f:
            self.assertEqual(f.read().strip(), "pod-c pod-c")

    def test_parts_balance_backend_cost(self):
        be = ReplayBackend(binary="/opt/replay")
        probs = [Problem("bf16", m, 4096, 4096) for m in (1, 8, 32, 64)] + [Problem("bf16", 4096, 28672, 4096)]
        self.assertGreater(be.cost(probs[-1]), 10 * be.cost(probs[0]))
        self.assertGreater(be.cost(Problem("bf16", 4096, 4096, 4096)), be.cost(Problem("s8", 4096, 4096, 4096)))
        ev = jobs.PodEvaluator(be, self.tmp, {p.key(): p for p in probs}, {}, parallel=2)
        keys = [p.key() for p in probs]
        parts = ev._parts(keys, {k: [None] * 4 for k in keys})
        self.assertEqual(parts, [keys[:4], keys[4:]])


if __name__ == "__main__":
    unittest.main()


class TestVerify(unittest.TestCase):
    """tune.py verify: a tuning file's plans and output bits against the built-in plan."""

    def run_verify(self, file_line, epilogue="store", paired=True, config=None, pairing=None, n=4096):
        """collect_verify of a file with one entry (config, nr=16 by default) for M 8 of a weight of N
        n; a gated call's lines pair as the model pairs them (at 64, or staged for n not whole 32-column
        blocks), the M 8 line under the file at pairing when given (a width, or staged)."""
        from native_brgemm_tuning import export as E
        from native_brgemm_tuning import plan as PL

        be = ReplayBackend()
        probs = [Problem("bf16", 8, n, 4096, epilogue=epilogue), Problem("bf16", 16, n, 4096, epilogue=epilogue)]
        config = config or {"nr": 16}
        entries = [{"id": "e", "match": E.match_block(probs[0], 8, 8), "config": config}]
        plans = E.entry_plans(entries, probs, SCHEMA, UARCH)
        self.assertEqual([(r["problem"], r["source"], r["cfg"], r["builtin"]) for r in plans],
                         [(probs[0].key(), "table", C.format_settings(config, SCHEMA), False)])
        model = None if epilogue != "gated" else "staged" if n % 32 else "64"
        self.assertEqual(sorted(plans[0].get("paired", {})), ["64"] if model == "64" else [])
        self.assertEqual(plans[0].get("pairing"), model)
        if not paired:
            plans = [{k: v for k, v in r.items() if k != "paired"} for r in plans]
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d)
        jobs.write_verify(d, be, probs, os.path.join(d, "t.json"))
        with open(os.path.join(d, jobs.VERIFY_MANIFEST)) as f:
            man = json.load(f)
        self.assertEqual([r["kind"] for r in man["runs"]], ["builtin", "file"])
        with open(os.path.join(d, jobs.SCRIPT)) as f:
            self.assertIn("ZENDNNL_MATMUL_NATIVE_TUNING_FILE=", f.read())
        for r in man["runs"]:
            lines = []
            for k in r["pkeys"]:
                p = Problem.parse(k)
                q = PL.builtin_plan(p, UARCH, SCHEMA)
                source, cfg, digest = "heuristic", "-", "ab%d" % p.m
                if r["kind"] == "file":
                    source, cfg, q, digest = file_line(p, q, digest)
                fields = PL.plan_log_fields(q, p)
                if p.epilogue == "gated":
                    w = pairing if pairing and r["kind"] == "file" and p.m == 8 else PL.log_pairing(q, p)
                    ran = q if w == "staged" else PL.paired_plan(q, p, int(w))
                    fields = dict(PL.plan_log_fields(ran, p), pairing=w)
                lines.append(log_line(p, fields, source, cfg))
                lines.append("CHECK shape=x N=%d K=%d M=%d hash=%s" % (p.n, p.k, p.m, digest))
            with open(os.path.join(d, "out", r["id"] + ".txt"), "w") as f:
                f.write("\n".join(lines) + "\n")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ResourceWarning)
            rep = jobs.collect_verify(d, plans, be)
            gc.collect()
        self.assertEqual([str(w.message) for w in caught if issubclass(w.category, ResourceWarning)], [])
        return rep

    def test_the_file_names_its_plans(self):
        from native_brgemm_tuning import plan as PL

        def named(p, q, digest):
            if p.m != 8:
                return "heuristic", "-", q, digest
            return "table", "nr=16", PL.resolve(p, {"nr": 16}, UARCH, SCHEMA).plan, digest

        rep = self.run_verify(named)
        self.assertEqual((rep["problems"], rep["named"], rep["failed"]), (2, 1, 0))
        # A gated call logs the plan its tiles ran after pairing its halves at 64 columns: the
        # file names it (paired), and the plan as planned would fail on nr and tiles.
        rep = self.run_verify(named, "gated")
        self.assertEqual((rep["problems"], rep["named"], rep["failed"]), (2, 1, 0))
        rep = self.run_verify(named, "gated", paired=False)
        self.assertEqual(rep["failed"], 1)
        self.assertIn("nr: model 16, log 64", [r for r in rep["rows"] if not r["ok"]][0]["why"])

    def test_a_gated_line_at_another_pairing_fails(self):
        # The file's plans for the gated call at M 8 (tiles of 16 columns, or of 32 in M blocks of 4)
        # pair at 64.  A line that stages its rows, or pairs the tiles of 32 at 32, logs the plan as
        # planned: its fields alone would pass.
        from native_brgemm_tuning import plan as PL

        for config, pairings in (({"nr": 16}, ("staged",)), ({"mb": 4}, ("32", "staged"))):
            text = C.format_settings(config, SCHEMA)

            def named(p, q, digest, config=config, text=text):
                if p.m != 8:
                    return "heuristic", "-", q, digest
                return "table", text, PL.resolve(p, config, UARCH, SCHEMA).plan, digest

            self.assertEqual(self.run_verify(named, "gated", config=config)["failed"], 0, text)
            for pairing in pairings:
                rep = self.run_verify(named, "gated", config=config, pairing=pairing)
                self.assertEqual([r["why"] for r in rep["rows"] if not r["ok"]],
                                 [["pairing: model 64, log %s" % pairing]], (text, pairing))

    def test_an_entry_of_a_padded_width_names_its_calls(self):
        # N 4112 is planned and matched at 4128, which the entry's match holds: the file names the M 8
        # call, a store call and a gated one, which stages its rows (pairing=staged).
        from native_brgemm_tuning import plan as PL

        def named(p, q, digest):
            if p.m != 8:
                return "heuristic", "-", q, digest
            return "table", "mb=4", PL.resolve(p, {"mb": 4}, UARCH, SCHEMA).plan, digest

        for epilogue in ("store", "gated"):
            rep = self.run_verify(named, epilogue, config={"mb": 4}, n=4112)
            self.assertEqual((rep["problems"], rep["named"], rep["failed"]), (2, 1, 0), epilogue)

    def test_failures(self):
        from native_brgemm_tuning import plan as PL

        def other_cfg(p, q, digest):
            return ("table", "nr=32", q, digest) if p.m == 8 else ("heuristic", "-", q, digest)

        def other_bits(p, q, digest):
            if p.m == 8:
                return "table", "nr=16", PL.resolve(p, {"nr": 16}, UARCH, SCHEMA).plan, digest + "0"
            return "heuristic", "-", q, digest

        def unnamed_change(p, q, digest):
            if p.m == 8:
                return "table", "nr=16", PL.resolve(p, {"nr": 16}, UARCH, SCHEMA).plan, digest
            return "table", "nr=16", PL.resolve(p, {"nr": 16}, UARCH, SCHEMA).plan, digest

        for f, failed in ((other_cfg, 1), (other_bits, 1), (unnamed_change, 1)):
            self.assertEqual(self.run_verify(f)["failed"], failed, f.__name__)
