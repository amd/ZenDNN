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
"""The heuristic port (DESIGN 4.4) and its rules, validation (11.2), resolution and plan identity."""

import dataclasses
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plan_cases  # noqa: E402
from helpers import DESIGN, NEWER, NO_BLOCKS, OLDER, RULES_OFF, SCHEMA, UARCH  # noqa: E402

from native_brgemm_tuning import plan as PL  # noqa: E402
from native_brgemm_tuning.candidates import parse_settings  # noqa: E402
from native_brgemm_tuning.problems import Problem  # noqa: E402
from native_brgemm_tuning.schema import Uarch  # noqa: E402

# The cache hierarchies and shapes of the library's rule tests.
SMALL = Uarch(l1d_bytes=48 << 10, l2_bytes=1 << 20, l3_bytes=32 << 20, family_l3_cores=1, l3_domain_cores=8)
LARGE = Uarch(l1d_bytes=48 << 10, l2_bytes=1 << 20, l3_bytes=128 << 20, family_l3_cores=1, l3_domain_cores=32)
SHAPES = ((32, 64), (96, 4097), (256, 1536), (1024, 1000), (1536, 4096), (3072, 8192), (5120, 32768),
          (14336, 4096), (28672, 4096))
MS = (1, 7, 8, 12, 24, 32, 64, 65, 128, 193, 1031, 3077, 4096, 8192)
GRID = ("nr", "mr", "mb", "m_blocks", "n_tiles", "threads")
EVERY_RULE = RULES_OFF.replace("=0", "=1")
PASS = ("kc", "int8_pass_quads")
SCHEDULE = ("col_groups", "unit_tiles", "domain_threads")


def builtin(p):
    return PL.builtin_plan(p, UARCH, SCHEMA)


def with_cfg(p, **cfg):
    return PL.heuristic_plan(p, UARCH, PL.params_from_config(cfg, SCHEMA), SCHEMA)


def rules_off(p, **cfg):
    return with_cfg(p, **dict(parse_settings(RULES_OFF, SCHEMA), **cfg))


def plan_of(text, p, uarch=SMALL, schema=SCHEMA):
    return PL.heuristic_plan(p, uarch, PL.params_from_config(parse_settings(text, schema), schema), schema)


def only(rule):
    """Every rule switch off but rule's."""
    return RULES_OFF.replace(rule + "=0", rule + "=1")


def changed_fields(a, b):
    return {f.name for f in dataclasses.fields(a) if getattr(a, f.name) != getattr(b, f.name)}


class TestHeuristic(unittest.TestCase):
    def test_plans_without_the_rules(self):
        cases = [  # (problem, nr, mr, mb, threads, kc, col_groups) worked out from DESIGN 4.4
            (Problem("bf16", 1, 4096, 4096), 32, 12, 1, 32, 4096, 0),
            (Problem("bf16", 1, 256, 1536), 16, 6, 1, 16, 1536, 0),
            (Problem("bf16", 4096, 4096, 4096), 32, 12, 192, 32, 4096, 0),
            (Problem("bf16", 4096, 3840, 4096), 32, 12, 192, 32, 4096, 8),
            (Problem("bf16", 64, 4096, 14336), 32, 12, 64, 32, 8192, 0),
            (Problem("bf16", 64, 256, 4096), 16, 6, 36, 32, 4096, 0),
            (Problem("s8", 8, 6144, 4096), 32, 6, 8, 32, 4096, 0),
        ]
        for p, nr, mr, mb, threads, kc, cg in cases:
            q = rules_off(p)
            self.assertEqual((q.nr, q.mr, q.mb, q.threads, q.kc, q.col_groups), (nr, mr, mb, threads, kc, cg), p)
            self.assertIsNone(PL.validate_plan(q, p, SCHEMA), p)
            # A build without the rule keys plans so.
            self.assertEqual(PL.builtin_plan(p, UARCH, DESIGN), q, p)

    def test_int8_rows_are_clamped_to_the_kernels(self):
        q = builtin(Problem("s8", 8, 6144, 4096))
        self.assertEqual((q.int8_rows_nv1, q.int8_rows_nv2), (12, 12))
        self.assertEqual(q.int8_pass_budget, UARCH.l2_bytes // 2)

    def test_pins_and_parameters(self):
        p = Problem("bf16", 64, 4096, 14336)
        self.assertEqual(rules_off(p, kc_split="equal").kc, 7168)  # 14336 in two even passes
        self.assertEqual(rules_off(p, kc=1000).kc, 1000)
        self.assertEqual(rules_off(p, threads_cap=8).threads, 8)
        self.assertEqual(rules_off(Problem("bf16", 1, 256, 1536), min_tiles_per_thread=4).threads, 4)
        q = rules_off(Problem("bf16", 4096, 3840, 4096), schedule="slice")
        self.assertEqual(q.col_groups, 0)
        q = rules_off(Problem("bf16", 4096, 4096, 4096), schedule="row_groups")
        self.assertEqual(q.col_groups, 8)  # gcd(gcd(128, 32), the L3 domain 8)
        self.assertEqual(rules_off(Problem("bf16", 16, 4096, 4096), mb=100).mb, 16)  # clamped to M

    def test_l3_domain_from_the_topology(self):
        p = Problem("bf16", 4096, 4096, 4096)
        rows = PL.params_from_config({"schedule": "row_groups"}, SCHEMA)
        wide = UARCH.with_overrides("l3_domain_cores=32")
        self.assertEqual(PL.heuristic_plan(p, wide, rows, SCHEMA).col_groups, 32)  # gcd(gcd(128, 32), 32)
        self.assertEqual(PL.heuristic_plan(p, UARCH, rows, SCHEMA).col_groups, 8)
        pinned = PL.params_from_config({"schedule": "row_groups", "domain_threads": 8}, SCHEMA)
        self.assertEqual(PL.heuristic_plan(p, wide, pinned, SCHEMA).col_groups, 8)
        self.assertEqual(PL.heuristic_plan(p, UARCH.with_overrides("l3_domain_cores=8"), rows, SCHEMA),
                         PL.heuristic_plan(p, UARCH, rows, SCHEMA))

    def test_validation_rules(self):
        p = Problem("s8", 64, 4096, 4096)
        q = builtin(p)
        self.assertIsNone(PL.validate_plan(q, p, SCHEMA))
        broken = [dict(nr=64), dict(n_tiles=1), dict(mb=65), dict(mr=13), dict(threads=33),
                  dict(col_groups=3), dict(row_unit=0), dict(int8_cw_mask=2), dict(int8_rows_nv2=13),
                  dict(int8_pass_quads=-1), dict(scratch_floats=1)]
        for change in broken:
            r = dataclasses.replace(q, **change)
            if "nr" in change:
                r = dataclasses.replace(r, n_tiles=p.n // change["nr"], scratch_floats=r.mb * change["nr"])
            self.assertIsNotNone(PL.validate_plan(r, p, SCHEMA), change)
        b = Problem("bf16", 64, 4096, 4096)
        qb = builtin(b)
        odd_kc = dataclasses.replace(qb, kc=1001, scratch_floats=qb.mb * qb.nr)
        self.assertIsNotNone(PL.validate_plan(odd_kc, b, SCHEMA))
        self.assertIsNotNone(PL.validate_plan(dataclasses.replace(qb, nr=16, n_tiles=256, mr=8,
                                                                  scratch_floats=8 * 16), b, SCHEMA))
        q16 = PL.resolve(p, {"nr": 16}, UARCH, SCHEMA).plan
        self.assertEqual(q16.nr, 16)
        self.assertIsNone(PL.validate_plan(dataclasses.replace(q16, int8_rows_nv2=13), p,
                                           SCHEMA))  # widths above nr are not checked

    def test_resolve_drops_an_illegal_override(self):
        p = Problem("bf16", 8, 4096, 4096)
        r = PL.resolve(p, {"nr": 16, "tile_rows": 8}, UARCH, SCHEMA)
        self.assertFalse(r.applied)
        self.assertIn("kernel", r.reason)
        self.assertEqual(r.plan, builtin(p))
        r = PL.resolve(p, {"nr": 16}, UARCH, SCHEMA)
        self.assertTrue(r.applied)
        self.assertEqual(r.plan.nr, 16)

    def test_int8_pass_quads(self):
        q = dataclasses.replace(rules_off(Problem("s8", 64, 4096, 4096)), int8_pass_budget=4096)
        self.assertEqual(PL.int8_pass_quads(q, 1024, 32, 6, 12), 1024)  # one row call: all of K
        self.assertEqual(PL.int8_pass_quads(q, 1024, 32, 24, 12), 23)  # 4096 // (4 * 44) = 23 per pass: 45 passes
        self.assertEqual(PL.int8_pass_quads(dataclasses.replace(q, int8_pass_quads=100), 1024, 32, 24, 12), 100)

    def test_pinned_bf16_rows_stop_the_narrowing(self):
        p = Problem("bf16", 1, 256, 4096)  # 8 tiles of 32 for 32 threads: narrows to 16
        self.assertEqual(rules_off(p).nr, 16)
        self.assertEqual(rules_off(p, mr=8).nr, 32)  # no 8-row kernel at 16 columns
        self.assertEqual(rules_off(p, mr=6).nr, 16)
        self.assertEqual(rules_off(Problem("s8", 1, 256, 4096), mr=8).nr, 16)


class TestRules(unittest.TestCase):
    """The rules of the built-in plan, as the library's rule tests state them."""

    def test_rule_switches(self):
        def rules(text, schema=SCHEMA):
            return tuple(PL.rule_switches(PL.params_from_config(parse_settings(text, schema), schema)))

        self.assertEqual(PL.rule_switches(PL.BUILTIN_PARAMS), (True,) * 6)
        self.assertEqual(rules(""), (True,) * 6)
        self.assertEqual(rules(RULES_OFF), (False,) * 6)
        self.assertEqual(rules("l1_pass=0"), (True, False, True, True, True, True))
        self.assertEqual(rules(only("traffic_groups")), (False, False, True, False, False, False))
        self.assertEqual(rules("bf16_narrow_n=0,blocks=0"), (True, True, True, True, False, False))
        # The switches, the cost model's included, default to 1; a rule whose switch the schema lacks is
        # one its build does not have.
        self.assertEqual([SCHEMA.key(s).default for s in PL.RULE_SWITCHES], [1] * len(PL.RULE_SWITCHES))
        for s in (SCHEMA, OLDER, NO_BLOCKS, DESIGN):
            self.assertEqual([PL.builtin_params(s)[name] != 0 for name in PL.RULE_SWITCHES],
                             [name in s.key_map for name in PL.RULE_SWITCHES], s.source)

    def test_builtin_plan_of_each_schema(self):
        for p in (Problem("bf16", 32, 4096, 4096), Problem("s8", 32, 4096, 4096), Problem("bf16", 128, 1536, 4096),
                  Problem("bf16", 4096, 28672, 4096), Problem("bf16", 4096, 4096, 14336)):
            on, off = plan_of("", p), plan_of(RULES_OFF, p)
            self.assertNotEqual(on, off, p)
            self.assertEqual((PL.builtin_plan(p, SMALL, SCHEMA), plan_of(EVERY_RULE, p)), (on, on), p)
            self.assertEqual(PL.resolve(p, {}, SMALL, SCHEMA).plan, on, p)
            self.assertEqual(PL.builtin_plan(p, SMALL, DESIGN), off, p)

    def test_gain_bound_and_slice_makespan(self):
        self.assertEqual([PL.gain_bound(w, g) for w, g in ((1000, 5), (101, 5), (7, 0), (7, 100), (7, -3))],
                         [950, 96, 7, 0, 7])
        rng = random.Random(3)
        for _ in range(3000):
            nr, mb, n_tiles = rng.choice([16, 32]), rng.randint(1, 40), rng.randint(1, 12)
            M = rng.randint(1, 8 * mb)
            m = PL.ceil_div(M, mb)
            g = PL.Grid(nr, 6, mb, m, n_tiles)
            workers, by_bytes = rng.randint(1, 40), rng.random() < 0.5
            # The busiest slice of the m_inner order, tile by tile; the longer slices spread as the library does.
            busiest, t0, err = 0, 0, 0
            per, extra = divmod(m * n_tiles, workers)
            for _w in range(workers):
                err += extra
                size = per + (err >= workers)
                err -= workers if err >= workers else 0
                work = sum(PL.tile_work(nr, M - (m - 1) * mb if t % m == m - 1 else mb, by_bytes)
                           for t in range(t0, t0 + size))
                busiest, t0 = max(busiest, work), t0 + size
            self.assertEqual(PL.slice_makespan(g, M, workers, by_bytes), busiest, (g, M, workers, by_bytes))
            bound = rng.randint(1, busiest + 2)
            got = PL.slice_makespan(g, M, workers, by_bytes, bound)
            self.assertEqual(got < bound, busiest < bound)
            if busiest < bound:
                self.assertEqual(got, busiest)

    def test_worked_examples(self):
        def plan(text, m, n, k, dtype="bf16", uarch=SMALL):
            return plan_of(text, Problem(dtype, m, n, k), uarch)

        # L1 pass: eight equal passes over K=4096 for M=32; INT8, four passes of 256 quads.
        self.assertEqual(plan(RULES_OFF, 32, 4096, 4096).kc, 4096)
        self.assertEqual(plan("", 32, 4096, 4096).kc, 512)
        self.assertEqual(plan("l1_pass=0", 32, 4096, 4096).kc, 4096)
        self.assertEqual(plan("", 32, 4096, 4096).scratch_floats, 32 * 32)
        self.assertEqual(plan(RULES_OFF, 32, 4096, 4096, "s8").int8_pass_quads, 0)
        self.assertEqual(plan("", 32, 4096, 4096, "s8").int8_pass_quads, 256)
        # One call's rows per tile: no pass.
        self.assertEqual(plan("", 8, 4096, 4096).kc, 4096)
        self.assertEqual(plan("", 8, 4096, 4096, "s8").int8_pass_quads, 0)
        # Makespan: never narrower than the width of the narrowing.
        self.assertEqual(plan("", 1, 1536, 4096).nr, 32)
        self.assertEqual(plan("", 1, 1536, 4096, "s8").nr, 32)
        self.assertEqual(plan("", 1, 1536, 4096).mb, 1)
        self.assertEqual(plan(RULES_OFF, 128, 1536, 4096).mb, 128)
        self.assertEqual(plan("", 128, 1536, 4096).mb, 72)
        off, on = plan(RULES_OFF, 128, 256, 1536), plan("", 128, 256, 1536)
        self.assertEqual((off.nr, off.mb), (16, 66))
        self.assertEqual((on.nr, on.mr, on.mb, on.tiles), (32, 12, 36, 32))
        self.assertEqual(plan(RULES_OFF, 1031, 96, 5120).tiles, 33)
        n96 = plan("", 1031, 96, 5120)
        self.assertEqual((n96.mb, n96.tiles, n96.threads), (108, 30, 30))
        self.assertEqual(plan("", 1031, 256, 1536).mb, 132)
        # Traffic: row groups in each 8-core domain when the source overflows its L3.
        self.assertEqual(plan(RULES_OFF, 4096, 28672, 4096).col_groups, 0)
        self.assertEqual(plan("", 4096, 28672, 4096).col_groups, 8)
        self.assertEqual(plan("", 3077, 28672, 4096).col_groups, 0)
        self.assertEqual(plan("", 4096, 28672, 4096, uarch=LARGE).col_groups, 0)
        self.assertEqual(plan("traffic_groups=0", 4096, 28672, 4096).col_groups, 0)

    def test_resolution_log_and_signature_of_a_rule_plan(self):
        p = Problem("s8", 32, 4096, 4096)
        q = PL.builtin_plan(p, SMALL, SCHEMA)
        self.assertEqual(PL.plan_log_fields(q, p)["int8_pass"], "256")
        self.assertIn(("quads", 256), PL.effective_signature(q, p))
        r = PL.resolve(p, parse_settings(RULES_OFF, SCHEMA), SMALL, SCHEMA)
        self.assertTrue(r.applied)
        self.assertEqual(PL.plan_log_fields(r.plan, p)["int8_pass"], "rule")
        self.assertNotEqual(PL.effective_signature(r.plan, p), PL.effective_signature(q, p))

    def test_each_rule_changes_only_its_fields(self):
        rules = ((only("makespan"), set(GRID + PASS + ("col_groups", "scratch_floats"))),
                 (only("l1_pass"), set(PASS + ("scratch_floats",))),
                 (only("traffic_groups"), {"col_groups"}))
        changed = {r: 0 for r, _ in rules}
        for uarch in (SMALL, LARGE):
            for threads in (8, 32, 64):
                for n, k in SHAPES:
                    for m in MS:
                        for dtype in ("bf16", "s8"):
                            p = Problem(dtype, m, n, k, threads=threads)
                            base = plan_of(RULES_OFF, p, uarch)
                            for text, fields in rules:
                                diff = changed_fields(plan_of(text, p, uarch), base)
                                self.assertLessEqual(diff, fields, (text, p))
                                changed[text] += bool(diff)
                            self.assertEqual(plan_of("", p, uarch), plan_of(EVERY_RULE, p, uarch), p)
        self.assertTrue(all(changed.values()), changed)

    def test_legal_for_domains_and_teams(self):
        hp, blocks = PL.builtin_params(SCHEMA), 0
        for domain in (1, 3, 8, 64):
            for l3 in (8, 128):
                uarch = Uarch(l1d_bytes=48 << 10, l2_bytes=1 << 20, l3_bytes=l3 << 20, family_l3_cores=1,
                              l3_domain_cores=domain)
                for threads in (1, 7, 32, 33, 64, 256):
                    for n, k in SHAPES:
                        for m in MS:
                            for dtype in ("bf16", "s8"):
                                p = Problem(dtype, m, n, k, threads=threads)
                                q = PL.heuristic_plan(p, uarch, hp, SCHEMA)
                                self.assertIsNone(PL.validate_plan(q, p, SCHEMA), (domain, l3, p))
                                blocks += q.unit_tiles > 0
        self.assertGreater(blocks, 0)

    def test_keys_the_mirror_does_not_read(self):
        # A key of a newer build: in the schema, but no mirrored plan reads it.
        p = Problem("bf16", 32, 4096, 4096)
        self.assertFalse(PL.mirrored("newer_pct"))
        self.assertEqual(PL.builtin_params(NEWER)["newer_pct"], 50)
        self.assertEqual(PL.builtin_plan(p, SMALL, NEWER), PL.builtin_plan(p, SMALL, SCHEMA))
        r = PL.resolve(p, {"newer_pct": 10, "nr": 16}, SMALL, NEWER)
        self.assertEqual((r.applied, r.reason), (False, "newer_pct: not mirrored"))
        self.assertEqual(r.plan, PL.builtin_plan(p, SMALL, NEWER))


class TestBf16Rules(unittest.TestCase):
    """BF16 tiles of several 32-column slabs and the rules bf16_alias and
    bf16_narrow_n, as the library's rule tests state them."""

    def plan(self, text, m, n, k, dtype="bf16", uarch=SMALL, threads=32):
        return plan_of(text, Problem(dtype, m, n, k, threads=threads), uarch, SCHEMA)

    def test_wide_tiles(self):
        for pack in (32, 64):
            p = Problem("bf16", 64, 4096, 4096, pack_nr=pack)
            for nr in (64, 128):
                r = PL.resolve(p, {"nr": nr}, SMALL, SCHEMA)
                self.assertTrue(r.applied, (pack, nr, r.reason))
                self.assertEqual((r.plan.nr, r.plan.mr, r.plan.n_tiles), (nr, 12, 4096 // nr))
                self.assertIsNone(PL.validate_plan(r.plan, p, SCHEMA))
        s = Problem("s8", 64, 4096, 4096)
        self.assertFalse(PL.resolve(s, {"nr": 64}, SMALL, SCHEMA).applied)
        self.assertFalse(PL.resolve(Problem("bf16", 64, 4096, 4096), {"nr": 64}, SMALL, OLDER).applied)

    def test_wide_tiles_on_pack_blocks(self):
        # Each 32-column slab of a wide tile lies inside one pack block: blocks of whole slabs.
        for pack, legal in ((16, False), (48, False), (64, True), (128, True)):
            p = Problem("bf16", 64, 384, 4096, pack_nr=pack)
            for nr in (64, 128):
                self.assertEqual(PL.bf16_wide_width(nr, pack), legal, (pack, nr))
                r = PL.resolve(p, {"nr": nr}, SMALL, SCHEMA)
                self.assertEqual(r.applied, legal, (pack, nr, r.reason))
                q = dataclasses.replace(PL.resolve(p, {"nr": 16}, SMALL, SCHEMA).plan, nr=nr, n_tiles=384 // nr)
                q = dataclasses.replace(q, scratch_floats=PL.scratch_floats(q, p))
                self.assertEqual(PL.validate_plan(q, p, SCHEMA) is None, legal, (pack, nr))

    def test_alias_rows(self):
        # Above decode_m, BF16 source rows of K a multiple of 2048 take bf16_alias_mr rows per call.
        self.assertEqual(self.plan(RULES_OFF, 4096, 4096, 4096).mr, 12)
        self.assertEqual(self.plan("", 4096, 4096, 4096).mr, 8)
        self.assertEqual(self.plan("", 1031, 4096, 14336).mr, 8)
        self.assertEqual(self.plan("", 4096, 4096, 6656).mr, 12)
        self.assertEqual(self.plan("", 128, 4096, 4096).mr, 12)
        self.assertEqual(self.plan("", 4096, 4096, 4096, "s8").mr, self.plan(RULES_OFF, 4096, 4096, 4096, "s8").mr)
        self.assertEqual(self.plan("bf16_alias=0", 4096, 4096, 4096).mr, 12)
        self.assertEqual(self.plan("bf16_alias_mr=6", 4096, 4096, 4096).mr, 6)
        # A build without the rule.
        self.assertEqual(plan_of("", Problem("bf16", 4096, 4096, 4096), SMALL, OLDER).mr, 12)

    def test_narrow_n(self):
        # 32 threads over 8-core domains are 4 domains.
        self.assertEqual(self.plan(RULES_OFF, 4096, 64, 2560).nr, 32)
        n64 = self.plan("", 4096, 64, 2560)
        self.assertEqual((n64.nr, n64.n_tiles), (64, 1))
        self.assertEqual(n64, self.plan("nr=64", 4096, 64, 2560))
        self.assertEqual(self.plan("", 4096, 192, 2560).nr, 64)
        self.assertEqual(self.plan("", 4096, 128, 2560).nr, 128)
        self.assertEqual(self.plan("", 2048, 128, 4096).nr, 64)
        for m, n, k in ((1031, 128, 4096), (2048, 256, 4096), (4096, 512, 2560), (768, 128, 4096)):
            self.assertEqual(self.plan("", m, n, k).nr, 32, (m, n, k))
        rg = self.plan("", 4096, 64, 4096)
        self.assertEqual((rg.nr, rg.col_groups), (32, 2))
        self.assertEqual(self.plan("", 4096, 64, 2560, uarch=LARGE).nr, 32)
        self.assertEqual(self.plan("bf16_narrow_n=0", 4096, 64, 2560).nr, 32)
        self.assertEqual(self.plan("bf16_narrow_n_min_m=4097", 4096, 64, 2560).nr, 32)
        self.assertEqual(self.plan("", 4096, 64, 2560, "s8").nr, 32)
        # The domains are those of the team: 16 threads are 2, 64 threads 8.
        for n, threads, nr in ((64, 16, 64), (128, 16, 32), (256, 64, 128), (512, 64, 32), (64, 8, 32)):
            self.assertEqual(self.plan("", 4096, n, 2560, threads=threads).nr, nr, (n, threads))
        for m, n, k in ((4096, 64, 2560), (4096, 192, 2560), (4096, 128, 2560), (2048, 128, 4096)):
            p = Problem("bf16", m, n, k)
            self.assertIsNone(PL.validate_plan(self.plan("", m, n, k), p, SCHEMA))

    def test_each_rule_changes_only_its_fields(self):
        rules = ((only("bf16_alias"), set(GRID + PASS + ("scratch_floats",))),
                 (only("bf16_narrow_n"), set(GRID + PASS + ("col_groups", "scratch_floats"))))
        changed = {r: 0 for r, _ in rules}
        for uarch in (SMALL, LARGE):
            for threads in (8, 32, 64):
                for n, k in SHAPES + ((64, 2560), (128, 2560), (192, 2048)):
                    for m in MS:
                        for dtype in ("bf16", "s8"):
                            p = Problem(dtype, m, n, k, threads=threads)
                            base = plan_of(RULES_OFF, p, uarch, SCHEMA)
                            for text, fields in rules:
                                diff = changed_fields(plan_of(text, p, uarch, SCHEMA), base)
                                self.assertLessEqual(diff, fields, (text, p))
                                changed[text] += bool(diff)
                            q = plan_of("", p, uarch, SCHEMA)
                            self.assertEqual(q, plan_of(EVERY_RULE, p, uarch, SCHEMA), p)
                            self.assertIsNone(PL.validate_plan(q, p, SCHEMA), p)
        self.assertTrue(all(changed.values()), changed)


class TestBlocks(unittest.TestCase):
    """The blocks schedule and its rule, as the library's rule tests state them."""

    def plan(self, text, m, n, k, dtype="bf16", uarch=SMALL, threads=32):
        return plan_of(text, Problem(dtype, m, n, k, threads=threads), uarch, SCHEMA)

    @staticmethod
    def blocks(q):
        return q.unit_tiles == 1 and q.col_groups == 0 and q.domain_threads == 8

    def test_rule_examples(self):
        # 32 threads over 8-core domains are 4 domains.  BF16, K = 14336: a quarter of
        # the source is 28 MiB, the B slab 896 KiB.
        b = self.plan("", 4096, 4096, 14336)
        self.assertTrue(self.blocks(b))
        self.assertLess(b.kc, 14336)
        self.assertIsNone(PL.validate_plan(b, Problem("bf16", 4096, 4096, 14336), SCHEMA))
        for m, n, k in ((4096, 5120, 32768), (4096, 4096, 4096), (512, 4096, 14336)):
            self.assertFalse(self.blocks(self.plan("", m, n, k)), (m, n, k))
        self.assertTrue(self.blocks(self.plan("", 1024, 4096, 14336)))
        self.assertFalse(self.blocks(self.plan("bf16_blocks_min_m=2048", 1024, 4096, 14336)))
        self.assertEqual(self.plan("", 4096, 4096, 14336, uarch=LARGE).unit_tiles, 0)
        self.assertEqual(self.plan(RULES_OFF, 4096, 4096, 14336).unit_tiles, 0)
        self.assertEqual(self.plan("blocks=0", 4096, 4096, 14336).unit_tiles, 0)
        # INT8: the slice of K = 4096 takes it from int8_blocks_min_m rows.
        i = self.plan("", 4096, 4096, 4096, "s8")
        self.assertTrue(self.blocks(i))
        self.assertIsNone(PL.validate_plan(i, Problem("s8", 4096, 4096, 4096), SCHEMA))
        self.assertFalse(self.blocks(self.plan("", 1024, 4096, 4096, "s8")))
        self.assertTrue(self.blocks(self.plan("int8_blocks_min_m=1024", 1024, 4096, 4096, "s8")))
        self.assertFalse(self.blocks(self.plan("", 4096, 4096, 14336, "s8")))
        # A pinned schedule or col_groups keeps its own plan.
        self.assertEqual(self.plan("schedule=slice", 4096, 4096, 14336).unit_tiles, 0)
        self.assertEqual(self.plan("col_groups=8", 4096, 4096, 14336).col_groups, 8)
        # The narrow-N rule replaces only plans without the blocks schedule.
        narrow = self.plan("", 1024, 64, 9216)
        self.assertEqual((narrow.nr, narrow.unit_tiles), (32, 1))
        self.assertEqual(self.plan("blocks=0", 1024, 64, 9216).nr, 64)
        # A build without the schedule.
        self.assertEqual(plan_of("", Problem("bf16", 4096, 4096, 14336), SMALL, NO_BLOCKS).unit_tiles, 0)

    def test_schedule_and_unit_tiles(self):
        q = self.plan("schedule=blocks", 64, 4096, 4096)
        self.assertEqual((q.unit_tiles, q.col_groups, q.domain_threads, q.n_tiles), (1, 0, 8, 128))
        self.assertEqual(self.plan("schedule=blocks,domain_threads=16", 64, 4096, 4096).domain_threads, 16)
        self.assertEqual(self.plan("schedule=blocks,unit_tiles=4096", 4096, 4096, 4096).unit_tiles, 128)  # one K pass
        self.assertEqual(self.plan("unit_tiles=4", 64, 4096, 4096).unit_tiles, 0)  # a pin of the schedule only
        # BF16 K passes keep the accumulators of every tile of a unit; one K pass, those of a row call.
        k = self.plan("schedule=blocks,unit_tiles=4", 4096, 4096, 14336)
        self.assertEqual((k.kc, k.scratch_floats), (8192, k.mb * k.nr * 4))
        one = self.plan("schedule=blocks,unit_tiles=4", 4096, 4096, 4096)
        self.assertEqual((one.kc, one.scratch_floats), (4096, one.mr * one.nr))
        s = self.plan("schedule=blocks,unit_tiles=4", 4096, 4096, 4096, "s8")
        self.assertEqual(s.scratch_floats, s.mb * s.nr)
        for text in ("schedule=blocks,col_groups=0", "schedule=blocks,col_groups=8"):
            with self.assertRaises(ValueError):
                parse_settings(text, SCHEMA)
        with self.assertRaises(ValueError):
            parse_settings("schedule=blocks", NO_BLOCKS)
        p, b = Problem("bf16", 4096, 4096, 14336), self.plan("", 4096, 4096, 14336)
        for change in (dict(unit_tiles=129), dict(unit_tiles=-1), dict(domain_threads=0), dict(col_groups=8)):
            self.assertIn("unit_tiles", PL.validate_plan(dataclasses.replace(b, **change), p, SCHEMA), change)
        self.assertIn("scratch", PL.validate_plan(dataclasses.replace(b, unit_tiles=2), p, SCHEMA))

    def test_unit_bound_under_bf16_k_passes(self):
        # A unit keeps at most the tiles whose accumulators fit the bf16_l2_pct share of L2 (512 KiB of
        # 1 MiB: 21 row blocks of 192 x 32 floats), and at least one.
        for text, m, k, units in (("schedule=blocks,unit_tiles=4096", 4096, 14336, 21),
                                  ("unit_tiles=4096", 4096, 14336, 21),
                                  ("schedule=blocks,unit_tiles=4096", 64, 4096, 64),
                                  ("schedule=blocks,unit_tiles=4096,bf16_l2_pct=1", 4096, 14336, 1),
                                  ("schedule=blocks,unit_tiles=4", 4096, 14336, 4)):
            q = self.plan(text, m, 4096, k)
            self.assertLess(q.kc, k, text)
            self.assertEqual((q.unit_tiles, q.scratch_floats), (units, q.mb * q.nr * units), (text, m, k))
            self.assertLessEqual(q.scratch_floats * 4, max(SMALL.l2_bytes // 2, q.mb * q.nr * 4), (text, m, k))
        # One K pass keeps one row call's accumulators, and INT8 one tile's: the unit takes every tile.
        self.assertEqual(self.plan("schedule=blocks,unit_tiles=4096", 4096, 4096, 4096).unit_tiles, 128)
        self.assertEqual(self.plan("schedule=blocks,unit_tiles=4096", 4096, 4096, 14336, "s8").unit_tiles, 128)

    def test_rule_changes_only_the_schedule(self):
        changed = 0
        for uarch in (SMALL, LARGE):
            for threads in (8, 32, 64):
                for n, k in SHAPES + ((64, 9216), (4096, 14336)):
                    for m in MS:
                        for dtype in ("bf16", "s8"):
                            p = Problem(dtype, m, n, k, threads=threads)
                            diff = changed_fields(plan_of(only("blocks"), p, uarch), plan_of(RULES_OFF, p, uarch))
                            self.assertLessEqual(diff, set(SCHEDULE), p)
                            changed += bool(diff)
        self.assertGreater(changed, 0)

    def test_signature_and_log(self):
        p = Problem("bf16", 4096, 4096, 14336)
        b, sliced = self.plan("", 4096, 4096, 14336), self.plan("blocks=0", 4096, 4096, 14336)
        sig = PL.effective_signature(b, p)
        self.assertNotEqual(PL.effective_signature(sliced, p), sig)
        self.assertNotEqual(PL.effective_signature(self.plan("unit_tiles=2", 4096, 4096, 14336), p), sig)
        # The blocks walk takes its units from the domains' counters: the traversal does not apply.
        self.assertEqual(PL.effective_signature(self.plan("traversal=n_inner", 4096, 4096, 14336), p), sig)
        log = PL.plan_log_fields(b, p)
        self.assertEqual(log["unit_tiles"], "1")
        self.assertEqual(log["domain_threads"], str(b.domain_threads))
        self.assertNotIn("unit_tiles", PL.plan_log_fields(sliced, p))
        self.assertNotIn("domain_threads", PL.plan_log_fields(sliced, p))
        self.assertEqual(PL.compare_with_log(b, p, log), [])
        # The deal reads the domain size too: a blocks plan of another domain prints another line.
        other = dict(log, domain_threads=str(b.domain_threads * 2))
        self.assertEqual(PL.compare_with_log(b, p, other),
                         ["domain_threads: model %d, log %d" % (b.domain_threads, b.domain_threads * 2)])
        self.assertEqual(PL.compare_with_log(b, p, {k: v for k, v in log.items() if k != "unit_tiles"}),
                         ["unit_tiles: model 1, log 0"])
        self.assertEqual(PL.compare_with_log(sliced, p, dict(PL.plan_log_fields(sliced, p), unit_tiles="1")),
                         ["unit_tiles: model 0, log 1"])


class TestNarrowBlocks(unittest.TestCase):
    """The blocks schedule of narrow BF16 slices, as the library's rule tests state it."""

    def plan(self, text, m, n, k, uarch=SMALL, threads=32, dtype="bf16"):
        return plan_of(text, Problem(dtype, m, n, k, threads=threads), uarch, SCHEMA)

    @staticmethod
    def blocks(q):
        return q.unit_tiles == 1 and q.col_groups == 0 and q.domain_threads == 8 and q.nr == 32

    def test_rule_examples(self):
        # At most two 32-column tiles per 8-core domain: the slice's threads share
        # column tiles, so every domain reads the whole source; the blocks schedule
        # keeps the grid and deals the row blocks per domain.
        for m in (1024, 4096):
            for n in (96, 256):
                s, q = self.plan("blocks=0", m, n, 1536), self.plan("", m, n, 1536)
                self.assertEqual(s.col_groups, 0)
                self.assertTrue(self.blocks(q), (m, n))
                self.assertEqual((q.mb, q.mr, q.kc, q.threads, q.scratch_floats),
                                 (s.mb, s.mr, s.kc, s.threads, s.scratch_floats))
                self.assertIsNone(PL.validate_plan(q, Problem("bf16", m, n, 1536), SCHEMA))
        # The narrow-N rule's wider tile wins over it.
        n64 = self.plan("", 4096, 64, 2560)
        self.assertEqual((n64.unit_tiles, n64.nr), (0, 64))
        for text, m, n in (("", 4096, 288), ("col_groups=1", 4096, 256), ("", 512, 256)):
            self.assertEqual(self.plan(text, m, n, 1536).unit_tiles, 0, (text, m, n))
        self.assertFalse(self.blocks(self.plan("bf16_blocks_min_m=2048", 1024, 256, 1536)))
        self.assertEqual(self.plan("", 4096, 256, 1536, uarch=LARGE).unit_tiles, 0)
        self.assertEqual(self.plan("", 1024, 256, 1536, dtype="s8").unit_tiles, 0)
        self.assertTrue(self.blocks(self.plan("", 4096, 96, 1536, threads=16)))
        self.assertEqual(self.plan("", 4096, 256, 1536, threads=16).unit_tiles, 0)
        self.assertEqual(self.plan("", 4096, 256, 1536, threads=8).unit_tiles, 0)


class TestNoL3(unittest.TestCase):
    """A machine without an L3 (l3_bytes 0) takes none of the L3 rules."""

    def test_l3_rules_stay_off_without_an_l3(self):
        no_l3 = Uarch(l1d_bytes=48 << 10, l2_bytes=1 << 20, l3_bytes=0, family_l3_cores=1, l3_domain_cores=8)
        # The L3 row-group test (traffic_groups=0): 24 column tiles for 32 threads.
        p = Problem("bf16", 4096, 768, 4096)
        self.assertEqual(plan_of(RULES_OFF, p, SMALL).col_groups, 8)
        self.assertEqual(plan_of(RULES_OFF, p, no_l3).col_groups, 0)
        traffic = with_l3 = 0
        for dtype in ("bf16", "s8"):
            for m in (256, 1024, 4096):
                for n in (768, 4096):
                    for k in (2048, 8192):
                        q = Problem(dtype, m, n, k)
                        traffic += plan_of(only("traffic_groups"), q, SMALL).col_groups > 0
                        with_l3 += plan_of(only("blocks"), q, SMALL).unit_tiles > 0
                        self.assertEqual(plan_of(only("traffic_groups"), q, no_l3).col_groups, 0)
                        self.assertEqual(plan_of(only("blocks"), q, no_l3).unit_tiles, 0)
        self.assertGreater(traffic, 0)
        self.assertGreater(with_l3, 0)


class TestAnyN(unittest.TestCase):
    """A weight of N not a multiple of 32 is planned, matched and logged at its padded width."""

    def test_problem_takes_the_padded_width(self):
        for n in (1, 8, 31, 32, 33, 72, 4096, 4097, 151937):
            for threads in (1, 8, 32):
                for dtype in ("bf16", "s8"):
                    p = Problem(dtype, 33, n, 256, threads=threads)
                    self.assertEqual(p.plan_n, -(-n // 32) * 32)
                    q = builtin(p)
                    self.assertIsNone(PL.validate_plan(q, p, SCHEMA), (n, threads, dtype))
                    self.assertEqual(q.n_tiles * q.nr, p.plan_n)
        # A multiple of 32 is not padded to a wider pack's blocks.
        self.assertEqual(Problem("bf16", 1, 96, 64, pack_nr=64).plan_n, 96)
        self.assertEqual(Problem("bf16", 1, 100, 64, pack_nr=64).plan_n, 128)
        self.assertEqual(plan_cases.Case("-", 49152, 1 << 20, 32 << 20, 8, 8, "s8", "per_channel", 4113, 4096, 8,
                                         32, 32, 0, 0).problem().plan_n, 4128)


class TestPerGroup(unittest.TestCase):
    """Per-group INT8 calls: the group kernel's rows, no INT8 L1 pass, and the keys of the scheme."""

    def test_group_tiles(self):
        g, c = Problem("s8", 32, 4096, 4096, int8_scheme="per_group"), Problem("s8", 32, 4096, 4096)
        qg, qc = builtin(g), builtin(c)
        self.assertEqual((qg.int8_rows_nv1, qg.int8_rows_nv2),
                         (SCHEMA.int8_max_mr(1, "per_group"), SCHEMA.int8_max_mr(2, "per_group")))
        self.assertEqual((qc.int8_rows_nv1, qc.int8_rows_nv2), (12, 12))
        self.assertLess(qg.int8_rows_nv2, 12)
        self.assertGreater(qc.int8_pass_quads, 0)
        self.assertEqual(qg.int8_pass_quads, 0)
        self.assertEqual(qg.int8_scheme, "per_group")
        self.assertIsNone(PL.validate_plan(qg, g, SCHEMA))
        # The group kernel serves fewer rows per call at 32 columns than the per-channel one.
        rows = qg.int8_rows_nv2 + 1
        wide = dataclasses.replace(qg, int8_rows_nv2=rows)
        self.assertIsNone(PL.validate_plan(dataclasses.replace(qc, int8_rows_nv2=rows), c, SCHEMA))
        self.assertIn("scheme", PL.validate_plan(wide, g, SCHEMA))

    def test_keys_of_the_scheme(self):
        g = Problem("s8", 32, 4096, 4096, int8_scheme="per_group")
        cfg = {"int8_kc": 64, "l1_pass": 0, "int8_mr": 4, "kc": 512}
        self.assertEqual(PL.scheme_config(cfg, g, SCHEMA), {"int8_mr": 4, "kc": 512})
        self.assertEqual(PL.scheme_config(cfg, Problem("s8", 32, 4096, 4096), SCHEMA), cfg)
        r = PL.resolve(g, {"int8_kc": 64, "l1_pass": 0}, UARCH, SCHEMA)
        self.assertFalse(r.applied)
        self.assertIn("per_group", r.reason)
        r = PL.resolve(g, cfg, UARCH, SCHEMA)
        self.assertTrue(r.applied)
        self.assertEqual((r.plan.int8_rows_nv1, r.plan.int8_pass_quads), (4, 0))

    def test_problem_identity(self):
        g, c = Problem("s8", 8, 4096, 4096, int8_scheme="per_group"), Problem("s8", 8, 4096, 4096)
        self.assertEqual(Problem.parse(g.key()), g)
        self.assertIn(",int8_scheme=per_group", g.key())
        self.assertNotIn("int8_scheme", c.key())
        self.assertNotIn("int8_scheme", c.to_dict())
        self.assertEqual(Problem.from_dict(g.to_dict()), g)
        with self.assertRaises(ValueError):
            Problem("bf16", 1, 32, 32, int8_scheme="per_group")
        with self.assertRaises(ValueError):
            Problem("s8", 1, 32, 32, int8_scheme="per_row")


class TestLibraryPlans(unittest.TestCase):
    """plan.py against the library's plans for plan_cases.py's random problems, machines and configs."""

    def test_plans_match_the_library(self):
        fx = plan_cases.load_fixture()
        cs = plan_cases.cases(fx["seed"], fx["count"])
        self.assertEqual(plan_cases.case_counts(cs), fx["cases"])
        # Every kind of case, and every weight type, is a sizable share of the cases.
        self.assertGreater(min(fx["cases"].values()), 1000)
        got = plan_cases.digests([plan_cases.tuner_line(c, SCHEMA) for c in cs], fx["chunk"])
        bad = [i for i, (a, b) in enumerate(zip(got, fx["sha256"])) if a != b]
        self.assertEqual((len(got), bad), (len(fx["sha256"]), []),
                         "cases %s differ from the library (plan_cases.py compare <dump> shows the fields)" % (
                             ", ".join("%d-%d" % (i * fx["chunk"], (i + 1) * fx["chunk"] - 1) for i in bad)))


class TestSignature(unittest.TestCase):
    def sig(self, p, **cfg):
        return PL.effective_signature(PL.resolve(p, cfg, UARCH, SCHEMA).plan, p)

    def test_equivalent_configs_share_a_signature(self):
        p = Problem("bf16", 8, 4096, 4096)
        base = self.sig(p)
        for cfg in (dict(block_rows=384), dict(traversal="n_inner"), dict(copy_block=128), dict(row_unit=24),
                    dict(tall_rows=8), dict(l3_pct=50)):
            self.assertEqual(self.sig(p, **cfg), base, cfg)  # M=8 < mr: one row block, one call
        s = Problem("s8", 8, 6144, 4096)
        self.assertEqual(self.sig(s, int8_mr=8), self.sig(s))  # 8 rows: one call either way
        self.assertEqual(self.sig(s, int8_l2_pct=100), self.sig(s))  # one call per tile: no K pass

    def test_different_work_differs(self):
        p = Problem("bf16", 8, 4096, 4096)
        base = self.sig(p)
        for cfg in (dict(nr=16), dict(threads_cap=16), dict(mb=4)):
            self.assertNotEqual(self.sig(p, **cfg), base, cfg)
        s = Problem("s8", 64, 6144, 4096)
        self.assertNotEqual(self.sig(s, int8_cw=[16]), self.sig(s))
        self.assertNotEqual(self.sig(s, direct_store=False), self.sig(s))
        self.assertNotEqual(self.sig(s, int8_mr=4), self.sig(s))

    def test_compare_with_log(self):
        p = Problem("s8", 8, 6144, 4096)
        q = builtin(p)
        logged = dict(PL.plan_log_fields(q, p))
        self.assertEqual(PL.compare_with_log(q, p, logged), [])
        logged["nr"] = "16"
        self.assertEqual(len(PL.compare_with_log(q, p, logged)), 1)


if __name__ == "__main__":
    unittest.main()
