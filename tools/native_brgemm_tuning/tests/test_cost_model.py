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
"""The cost model's mirror and the model's choices, as the library's NativeCostModel tests
state them; the F16_PH tiles and the s4 rules of the planner."""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import SCHEMA  # noqa: E402

from native_brgemm_tuning import candidates as C  # noqa: E402
from native_brgemm_tuning import cost_model as CM  # noqa: E402
from native_brgemm_tuning import plan as PL  # noqa: E402
from native_brgemm_tuning.candidates import parse_settings  # noqa: E402
from native_brgemm_tuning.problems import Problem  # noqa: E402
from native_brgemm_tuning.schema import Schema, Uarch  # noqa: E402

# The library tests' hierarchies: 48 KiB L1d, 1 MiB L2, an L3 of 32 MiB per 8 cores or of
# 128 MiB per 32 cores.
DOMAINS8 = Uarch(l1d_bytes=48 << 10, l2_bytes=1 << 20, l3_bytes=32 << 20, family_l3_cores=1, l3_domain_cores=8)
DOMAINS32 = Uarch(l1d_bytes=48 << 10, l2_bytes=1 << 20, l3_bytes=128 << 20, family_l3_cores=1, l3_domain_cores=32)
DOMAINS16 = Uarch(l1d_bytes=48 << 10, l2_bytes=1 << 20, l3_bytes=32 << 20, family_l3_cores=1, l3_domain_cores=16)
BOTH = (DOMAINS8, DOMAINS32)


def params(text=""):
    return PL.params_from_config(parse_settings(text, SCHEMA) if text else {}, SCHEMA)


def plan(u, p, text=""):
    return PL.heuristic_plan(p, u, params(text), SCHEMA)


def bf16(m, n, k, threads=32):
    return Problem("bf16", m, n, k, threads=threads)


def int4(m, n, k, threads, gs, scale_bytes=2):
    return Problem("s4", m, n, k, threads=threads, group=gs, scale_bytes=scale_bytes)


def with_kernels(kind):
    """The fixture's schema as a build whose registry has (or lacks) the F16_PH kernels."""
    doc = dict(SCHEMA.doc)
    f16 = [k for k in doc["kernels"] if k["kind"] == "f16"]
    others = [k for k in doc["kernels"] if k["kind"] != "f16_ph"]
    if kind == "f16_ph":
        others += [dict(k, kind="f16_ph") for k in f16] + [dict(f16[-1], kind="f16_ph", nr=64)]
    return Schema(dict(doc, kernels=others), "%s with the %s kernels" % (SCHEMA.source, kind))


class TestPrices(unittest.TestCase):
    def test_memory_rate_and_tall_price(self):
        hp = params()
        self.assertEqual(CM.mem_cycles(DOMAINS8, hp), 2400.0 / math.sqrt(8.0))
        self.assertEqual(CM.mem_cycles(DOMAINS32, hp), 2400.0 / math.sqrt(32.0))
        self.assertEqual(CM.mem_cycles(DOMAINS32, params("domain_threads=8")), 2400.0 / math.sqrt(8.0))
        self.assertEqual(CM.mem_cycles(DOMAINS8, params("cm_mem_cycles=300")), 300.0)
        self.assertEqual((CM.tall_pct(DOMAINS8, hp), CM.tall_pct(DOMAINS32, hp)), (0.0, 25.0))
        # The strict comparison: a core streams slower than its share only below cm_stream_cycles;
        # the derived rate of 16 threads to a domain is the tie.
        self.assertEqual([CM.core_streams(u, hp) for u in (DOMAINS8, DOMAINS32)], [False, True])
        self.assertFalse(CM.core_streams(DOMAINS8, params("domain_threads=16")))

    def test_second_reads_only_where_a_domain_rereads(self):
        hp = params()

        def cost(u, threads, g):
            p = bf16(32, 4096, 4096, threads)
            q = plan(u, p, "cost_model=0,col_groups=%d" % g)
            self.assertEqual(q.col_groups, g)
            return CM.plan_cost(q, p, u, hp)

        mem32, mem8 = CM.mem_cycles(DOMAINS32, hp), CM.mem_cycles(DOMAINS8, hp)
        rereads = 8 * 256.0 * hp["cm_l3_cycles"]
        own = cost(DOMAINS32, 64, 32)
        self.assertEqual((own.stream_kib, own.stream), (1024.0, 1024.0 * mem32))
        two = cost(DOMAINS32, 64, 16)
        self.assertEqual((two.stream_kib, two.stream), (1024.0, 1024.0 * mem32 + rereads))
        self.assertEqual(cost(DOMAINS32, 32, 16).stream, 1024.0 * mem32 + rereads)
        spans = cost(DOMAINS8, 32, 16)
        self.assertEqual((spans.stream_kib, spans.stream), (2048.0, 2048.0 * mem8))

    def test_other_plans_are_not_priced(self):
        hp = params()
        for p in (Problem("f32", 32, 4096, 4096), Problem("s8", 32, 4096, 4096, int8_scheme="per_group")):
            self.assertEqual(CM.plan_cost(PL.rules_plan(p, DOMAINS8, hp, SCHEMA), p, DOMAINS8, hp), CM.PlanCost())


class TestBf16Choices(unittest.TestCase):
    def test_memory_rate_follows_the_l3_domain(self):
        self.assertEqual(plan(DOMAINS8, bf16(16, 151936, 2048)).mr, 12)
        self.assertEqual(plan(DOMAINS32, bf16(16, 151936, 2048)).mr, 8)
        self.assertEqual(plan(DOMAINS8, bf16(16, 151936, 2048), "cm_mem_cycles=424").mr, 8)

    def test_link_fills_the_idle_of_long_streams(self):
        for threads in (32, 64):
            self.assertEqual(plan(DOMAINS8, bf16(16, 2560, 9216, threads)).mr, 8)
            p = bf16(16, 248320, 4096, threads)
            self.assertEqual(plan(DOMAINS8, p), plan(DOMAINS8, p, "cost_model=0"))
            self.assertEqual(plan(DOMAINS32, p).mr, 8)
        for text in ("cm_stream_cycles=1048576", "cm_sync_tiles=1048576"):
            self.assertEqual(plan(DOMAINS8, bf16(16, 248320, 4096), text).mr, 8, text)

    def test_row_groups_keep_a_pinned_row_unit(self):
        p = bf16(55, 2560, 4096)
        q = plan(DOMAINS32, p)
        self.assertEqual((q.col_groups, q.mr, q.row_unit), (16, 8, 8))
        for unit in (1, 4):
            r = plan(DOMAINS32, p, "row_unit=%d" % unit)
            self.assertEqual((r.col_groups, r.mr, r.row_unit), (16, 8, unit))
            self.assertIsNone(PL.validate_plan(r, p, SCHEMA))
        # The auto value of row_unit is 12: setting it is leaving it unset.
        self.assertEqual(plan(DOMAINS32, p, "row_unit=12"), q)
        self.assertEqual(C.canonical({"row_unit": 12, "nr": 16}, SCHEMA, "bf16"), {"nr": 16})
        self.assertEqual(C.builtin_value(SCHEMA.key("row_unit")), 12)

    def test_the_model_needs_its_switch(self):
        for u in BOTH:
            p = bf16(16, 151936, 2048)
            self.assertEqual(plan(u, p, "cost_model=0"), PL.rules_plan(p, u, params(), SCHEMA))

    def test_short_blocks_need_a_longer_k_where_one_core_streams_slower(self):
        # The M 32 lm_heads of K 1536 and 2048 keep their K passes on the 32-thread domains at 32
        # threads and take the M blocks on the 8- and 16-thread ones; cm_block_k=0 or a lifted
        # limit give them back, and K 2560 takes them.  Teams of 8 to 28 and of 48 threads, and f16
        # plans, take the plan of no bound: the blocks at 16 threads for N 128256, K 2048.
        heads = ((262144, 1536), (151936, 2048), (128256, 2048))
        for n, k in heads:
            for threads in (8, 16, 24, 28, 48):
                p = bf16(32, n, k, threads)
                self.assertEqual(plan(DOMAINS32, p), plan(DOMAINS32, p, "cm_block_k=0"), (n, k, threads))
        t16 = plan(DOMAINS32, bf16(32, 128256, 2048, 16))
        self.assertEqual((t16.mb, t16.kc), (6, 2048))
        unbounded = params("cm_block_k=0")
        for kind in ("f16_ph", "f16"):
            schema = with_kernels(kind)
            for n, k in heads:
                p = Problem("f16", 32, n, k)
                f = PL.heuristic_plan(p, DOMAINS32, params(), schema)
                self.assertEqual(f, PL.heuristic_plan(p, DOMAINS32, unbounded, schema), (kind, n, k))
                if kind == "f16_ph" and n != 128256:
                    self.assertEqual((f.kind, f.mb, f.kc), ("f16_ph", 6, k), (n, k))
        for n, k in ((262144, 1536), (151936, 2048)):
            p = bf16(32, n, k)
            q = plan(DOMAINS32, p)
            self.assertEqual(q.mb, 32, (n, k))
            self.assertLess(q.kc, k, (n, k))
            for u in (DOMAINS8, DOMAINS16):
                b = plan(u, p)
                self.assertEqual((b.mb, b.kc), (6, k), (n, k, u.l3_domain_cores))
            for lift in ("cm_block_k=0", "cm_stream_cycles=424"):
                b = plan(DOMAINS32, p, lift)
                self.assertEqual((b.mb, b.kc), (6, k), (n, k, lift))
                self.assertIsNone(PL.validate_plan(b, p, SCHEMA), (n, k, lift))
        r = plan(DOMAINS32, bf16(32, 262144, 2560))
        self.assertEqual((r.mb, r.kc), (6, 2560))
        self.assertEqual(plan(DOMAINS32, bf16(32, 262144, 2560), "cm_block_k=2560").mb, 32)


class TestInt4Choices(unittest.TestCase):
    def test_row_blocks(self):
        for u in BOTH:
            for n, k, mb in ((6656, 19968, 64), (3840, 15360, 32)):
                p = int4(128, n, k, 32, 128)
                r = plan(u, p, "cost_model=0")
                self.assertEqual(r.m_blocks, 1)
                q = plan(u, p)
                self.assertEqual((q.mb, q.nr, q.threads, q.col_groups), (mb, r.nr, r.threads, 0), (n, u.l3_bytes))
                self.assertLess(CM.plan_cost(q, p, u, params()).cycles, 0.95 * CM.plan_cost(r, p, u, params()).cycles)
                for pin in ("mb=128", "col_groups=0", "schedule=slice", "cm_int4_min_m=0", "cm_int4_min_m=129",
                            "cm_gain_pct=100"):
                    self.assertEqual(plan(u, p, pin), plan(u, p, pin + ",cost_model=0"), pin)
            below = int4(47, 6656, 19968, 32, 128)
            self.assertEqual(plan(u, below), plan(u, below, "cost_model=0"))
            self.assertGreater(plan(u, below, "cm_int4_min_m=47").m_blocks, 1)

    def test_one_group_row_blocks_at_the_group_kernel_rows(self):
        # The row blocks of a weight of one group over K are those the model rates at the group
        # kernel's rows per call (the plan at int8_mr=6); the plan keeps the 12 rows per call of the
        # kernels of one group.
        blocked = 0
        for u in BOTH:
            for n, k in ((7680, 5120), (6656, 4096), (4608, 6656), (3840, 15360), (6656, 19968), (128256, 4096)):
                for m in (48, 55, 64, 80, 96, 112, 128):
                    p = int4(m, n, k, 32, k)
                    q, g = plan(u, p), plan(u, p, "int8_mr=6")
                    self.assertEqual((q.int8_rows_nv2, g.int8_rows_nv2), (12, 6), (m, n, k))
                    self.assertEqual((q.mb, q.m_blocks), (g.mb, g.m_blocks), (m, n, k, u.l3_domain_cores))
                    self.assertIsNone(PL.validate_plan(q, p, SCHEMA), (m, n, k))
                    blocked += q.m_blocks > 1
        self.assertGreater(blocked, 0)

    def test_one_group_takes_the_l2_share(self):
        def passes(q, k):
            quads = k // 4
            return quads // CM.int4_pass_quads(q, quads, 32, min(q.mb, 32), q.int8_rows_nv2)

        for u in BOTH:
            l2 = u.l2_bytes * params()["int8_l2_pct"] // 100
            dn = int4(32, 4096, 14336, 32, 14336)
            r = plan(u, dn, "cost_model=0")
            self.assertLess(r.int8_pass_budget, l2)
            self.assertGreater(passes(r, 14336), 1)
            for text in ("", "cm_pass_calls=1", "cm_pass_calls=3"):
                q = plan(u, dn, text)
                self.assertEqual((q.int8_pass_budget, passes(q, 14336)), (l2, 1), text)
            lm = plan(u, int4(128, 128256, 4096, 32, 4096))
            self.assertEqual((lm.int8_pass_budget, passes(lm, 4096)), (l2, 1))
            self.assertEqual(plan(u, int4(16, 4096, 14336, 32, 14336)).int8_pass_budget, l2)
            wide = plan(u, int4(32, 5120, 32768, 32, 32768))
            self.assertEqual(wide.int8_pass_budget, l2)
            self.assertGreater(passes(wide, 32768), 1)
            self.assertLess(plan(u, int4(128, 128256, 4096, 32, 128)).int8_pass_budget, l2)

    def test_s4_plans_and_lines(self):
        p = int4(16, 4096, 4096, 32, 128)
        q = PL.builtin_plan(p, DOMAINS8, SCHEMA)
        self.assertEqual((q.kind, q.int8_rows_nv1, q.int8_rows_nv2), ("int4", 12, 6))
        # One group over K: the kernels of one group, 12 rows per call at both widths.
        one = int4(16, 4096, 4096, 32, 4096)
        self.assertEqual(PL.plan_log_fields(PL.builtin_plan(one, DOMAINS8, SCHEMA), one)["int8_rows"], "12/12")
        # A decode plan's groups take batches of the int4_l1_pct share of L1.
        self.assertEqual(q.int8_pass_budget, min(DOMAINS8.l2_bytes // 2, DOMAINS8.l1d_bytes * 125 // 100))
        self.assertEqual(plan(DOMAINS8, p, "int4_l1_pct=0").int8_pass_budget, DOMAINS8.l2_bytes // 2)
        fields = PL.plan_log_fields(q, p)
        self.assertEqual((fields["int8_scheme"], fields["gs"], fields["pass_budget"]),
                         ("per_group", "128", str(q.int8_pass_budget)))
        self.assertNotIn("int8_pass", fields)
        self.assertNotIn("kc", fields)
        # Per-channel keys do not reach an s4 call.
        r = PL.resolve(p, {"int8_kc": 512}, DOMAINS8, SCHEMA)
        self.assertFalse(r.applied)
        self.assertEqual(PL.resolve(p, {"int8_kc": 512, "int8_mr": 4}, DOMAINS8, SCHEMA).plan.int8_rows_nv1, 4)


class TestF16(unittest.TestCase):
    def test_the_registry_decides_the_kernels(self):
        ph, wide = with_kernels("f16_ph"), with_kernels("f16")
        p = Problem("f16", 256, 4096, 4096)
        self.assertEqual(PL.builtin_plan(p, DOMAINS8, wide).kind, "f16")
        q = PL.builtin_plan(p, DOMAINS8, ph)
        self.assertEqual((q.kind, q.nr), ("f16_ph", 64))
        self.assertIsNone(PL.validate_plan(q, p, ph))
        self.assertEqual(PL.plan_log_fields(q, p)["kernels"], "f16_ph")
        # Below f16_wide_min_m, or with N not whole tiles of 64, F16_PH tiles take one block.
        self.assertEqual(PL.builtin_plan(p.with_m(191), DOMAINS8, ph).nr, 32)
        self.assertEqual(PL.builtin_plan(Problem("f16", 256, 4128, 4096), DOMAINS8, ph).nr, 32)
        q = PL.heuristic_plan(p, DOMAINS8, PL.params_from_config({"f16_wide_min_m": 1 << 30}, ph), ph)
        self.assertEqual(q.nr, 32)

    def test_f16_ph_passes_of_whole_blocks(self):
        ph = with_kernels("f16_ph")
        p = Problem("f16", 64, 4096, 4096)
        for kc in (300, 700, 1000):
            q = PL.heuristic_plan(p, DOMAINS8, PL.params_from_config({"kc": kc}, ph), ph)
            self.assertEqual(q.kc, kc // 256 * 256, kc)
        q = PL.heuristic_plan(p, DOMAINS8, PL.params_from_config({"kc": 100}, ph), ph)
        self.assertEqual(q.kc, 256)
        # The widening F16 kernels take any even K pass.
        f16 = with_kernels("f16")
        self.assertEqual(PL.heuristic_plan(p, DOMAINS8, PL.params_from_config({"kc": 300}, f16), f16).kc, 300)

    def test_alias_rule_skips_the_widened_rows(self):
        p = Problem("f16", 1024, 4096, 4096)
        f16, ph = with_kernels("f16"), with_kernels("f16_ph")
        self.assertEqual(PL.builtin_plan(p, DOMAINS8, f16).mr, 12)
        self.assertEqual(PL.builtin_plan(p, DOMAINS8, ph).mr, 8)


class TestGatedLines(unittest.TestCase):
    def test_paired_plans(self):
        p = Problem("bf16", 8, 4096, 4096, epilogue="gated")
        q = PL.builtin_plan(p, DOMAINS8, SCHEMA)
        logged = PL.plan_log_fields(PL.paired_plan(q, p, 64), p)
        logged["pairing"] = "64"
        self.assertEqual(PL.compare_with_log(q, p, logged), [])
        logged["pairing"] = "staged"
        self.assertNotEqual(PL.compare_with_log(q, p, logged), [])


if __name__ == "__main__":
    unittest.main()
