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
"""Candidates: override formatting, static checks, the bits filter, legality, de-duplication."""

import itertools
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import DESIGN, NEWER, NO_BLOCKS, OLDER, RULES_OFF, SCHEMA, UARCH  # noqa: E402

from native_brgemm_tuning import candidates as C  # noqa: E402
from native_brgemm_tuning import plan as PL  # noqa: E402
from native_brgemm_tuning.plan import bf16_wide_width, validate_plan  # noqa: E402
from native_brgemm_tuning.problems import Problem  # noqa: E402

PROBLEMS = [Problem("bf16", 1, 4096, 4096), Problem("bf16", 16, 6144, 4096), Problem("bf16", 2048, 3840, 4096),
            Problem("s8", 8, 28672, 4096), Problem("s8", 512, 4096, 14336)]
# The switches and constants of the rules both dtypes read.
RULE_KEYS = ["makespan", "min_gain_pct", "idle_pct", "decode_m", "l1_pass", "l1_pct", "l1_pass_max_mb",
             "traffic_groups", "blocks"]


class TestOverrideText(unittest.TestCase):
    def test_settings_in_schema_order(self):
        cfg = {"int8_cw": [16, 32], "mb": 24, "row_split": False, "nr": 16, "schedule": "row_groups"}
        self.assertEqual(C.format_settings(cfg, SCHEMA), "nr=16,mb=24,schedule=row_groups,row_split=0,int8_cw=32/16")
        self.assertEqual(C.format_settings({}, SCHEMA), "")

    def test_round_trip_every_type(self):
        cfg = {"nr": 16, "kc": 1024, "schedule": "slice", "traversal": "n_inner", "row_split": False,
               "kc_split": "equal", "int8_cw": [16], "direct_store": False, "bf16_copy_par_min": 1 << 40}
        text = C.format_settings(cfg, SCHEMA)
        self.assertEqual(C.parse_settings(text, SCHEMA), cfg)
        self.assertEqual(C.parse_settings(" NR = 16 , Row_Split=off ", SCHEMA), {"nr": 16, "row_split": False})

    def test_bad_settings(self):
        for text in ("nr=24", "foo=1", "nr16", "schedule=slice,col_groups=4", "schedule=row_groups,col_groups=0",
                     "int8_cw=32", "min_pass_k=255", "int8_kc=6", "mb=0", "nr=256", "int8_cw=64/32/16",
                     "int8_mr_wide=4"):
            with self.assertRaises(ValueError, msg=text):
                C.parse_settings(text, SCHEMA)

    def test_clauses(self):
        p = Problem("s8", 1, 4096, 4096)
        text = C.format_override([(C.problem_filters(p), "nr=16,int8_mr=8"), ("", "mb=4")])
        self.assertEqual(text, "dtype=s8,m=1,n=4096,k=4096,threads=32:nr=16,int8_mr=8;mb=4")
        parsed = C.parse_override(text, SCHEMA)
        self.assertEqual(parsed[0], ({"dtype": "s8", "m": "1", "n": "4096", "k": "4096", "threads": "32"},
                                     {"nr": 16, "int8_mr": 8}))
        self.assertEqual(parsed[1], ({}, {"mb": 4}))
        self.assertEqual(C.problem_filters(p, 1, 16), "dtype=s8,m=1-16,n=4096,k=4096,threads=32")
        self.assertEqual(C.problem_filters(p, 513, None), "dtype=s8,m=513-,n=4096,k=4096,threads=32")
        # A call of N not a multiple of 32 matches n on its padded width.
        self.assertEqual(C.problem_filters(Problem("bf16", 8, 4113, 4096)), "dtype=bf16,m=8,n=4128,k=4096,threads=32")
        with self.assertRaises(ValueError):
            C.format_override([("m=%d" % i, "nr=16") for i in range(C.MAX_CLAUSES + 1)])


class TestSpace(unittest.TestCase):
    def test_default_keys_keep_the_bits(self):
        s8 = C.search_keys(SCHEMA, "s8")
        self.assertNotIn("bias_in_dequant", s8)
        self.assertNotIn("kc", s8)
        self.assertIn("int8_cw", s8)
        self.assertIn("bias_in_dequant", C.search_keys(SCHEMA, "s8", allow_bit_changing=True))
        bf = C.search_keys(SCHEMA, "bf16")
        self.assertIn("kc", bf)
        self.assertFalse([k for k in bf if k.startswith("int8")])
        p = Problem("s8", 8, 4096, 4096)
        with self.assertRaises(ValueError):
            C.Space(p, SCHEMA, UARCH, ["bias_in_dequant"])
        self.assertIsNone(C.Space(p, SCHEMA, UARCH).make({"bias_in_dequant": False}))
        allowed = C.Space(p, SCHEMA, UARCH, allow_bit_changing=True)
        self.assertIsNotNone(allowed.make({"bias_in_dequant": False}))

    def test_every_candidate_is_legal(self):
        for p in PROBLEMS:
            sp = C.Space(p, SCHEMA, UARCH)
            self.assertTrue(all(w <= p.pack_nr or (p.bf16 and bf16_wide_width(w, p.pack_nr))
                                for w in sp.choices["nr"]))
            self.assertTrue(all(1 <= v <= p.m for v in sp.choices["mb"]))
            for c in itertools.islice(sp.iter_grid(2), 600):
                self.assertIsNone(C.check_config(c.config, SCHEMA), c)
                self.assertIsNone(validate_plan(c.plan, p, SCHEMA), c)
                self.assertTrue(all(SCHEMA.key(k).same_bits(p.dtype) for k in c.config), c)
                self.assertEqual(sp.from_text(c.text).sig, c.sig)

    def test_illegal_configs_are_dropped(self):
        b = C.Space(Problem("bf16", 8, 256, 4096), SCHEMA, UARCH)  # nr narrows to 16: 6-row kernels only
        self.assertIsNone(b.make({"tile_rows": 8}))
        self.assertIsNone(C.Space(Problem("s8", 8, 256, 4096), SCHEMA, UARCH).make({"nr": 64}))  # BF16 only
        narrow = C.Space(Problem("bf16", 8, 256, 4096, pack_nr=16), SCHEMA, UARCH)
        self.assertIsNone(narrow.make({"nr": 32}))  # wider than pack_nr
        self.assertIsNone(b.make({"schedule": "slice", "col_groups": 2}))
        self.assertIsNotNone(b.make({"tile_rows": 4}))

    def test_keys_of_another_int8_scheme_are_not_searched(self):
        p = Problem("s4", 16, 4096, 4096, group=128)
        sp = C.Space(p, SCHEMA, UARCH, ["int8_kc", "int4_l1_pct", "int8_mr", "bias_in_dequant"],
                     allow_bit_changing=True)
        self.assertEqual(sp.keys, ["int4_l1_pct", "int8_mr"])
        self.assertIn("int8_kc", C.Space(Problem("s8", 16, 4096, 4096), SCHEMA, UARCH, ["int8_kc"]).keys)

    def test_canonical_drops_built_in_values_and_other_dtypes(self):
        sp = C.Space(Problem("bf16", 64, 4096, 4096), SCHEMA, UARCH)
        c = sp.make({"tile_rows": 6, "nr": 16, "int8_mr": 4, "schedule": "auto"})
        self.assertEqual(c.config, {"nr": 16})
        self.assertIs(sp.make({"tile_rows": 6}), sp.default)

    def test_dedup_by_plan(self):
        for p in PROBLEMS:
            sp = C.Space(p, SCHEMA, UARCH)
            level1 = list(sp.iter_grid(1))
            uniq = C.unique(level1, sp)
            sigs = [c.sig for c in uniq]
            self.assertEqual(len(sigs), len(set(sigs)))
            self.assertNotIn(sp.default.sig, sigs)
            self.assertLess(len(uniq), len(level1), p)
        sp = C.Space(Problem("bf16", 1, 4096, 4096), SCHEMA, UARCH)
        texts = [c.text for c in C.unique(sp.iter_grid(1), sp)]
        self.assertNotIn("block_rows=96", texts)  # M=1: the block is one row either way
        self.assertIn("nr=16", texts)

    def test_neighbors_and_sampling(self):
        sp = C.Space(Problem("s8", 64, 4096, 4096), SCHEMA, UARCH)
        c = sp.make({"nr": 16, "int8_mr": 8})
        for n in sp.neighbors(c):
            changed = [k for k in set(n.config) | set(c.config) if n.config.get(k) != c.config.get(k)]
            self.assertEqual(len(changed), 1, n)
        a = [x.text for x in (sp.sample(random.Random(5)) for _ in range(30)) if x]
        b = [x.text for x in (sp.sample(random.Random(5)) for _ in range(30)) if x]
        self.assertEqual(a, b)

    def test_keep_threads(self):
        p = Problem("bf16", 8, 1536, 4096)  # 48 tiles of 32 columns: all 32 threads
        free = C.Space(p, SCHEMA, UARCH, ["nr", "threads_cap", "min_tiles_per_thread", "tall_rows"])
        kept = C.Space(p, SCHEMA, UARCH, ["nr", "threads_cap", "min_tiles_per_thread", "tall_rows"],
                       keep_threads=True)
        everything = C.unique(free.iter_grid(1), free)
        self.assertTrue(any(c.plan.threads < 32 for c in everything))
        texts = [c.text for c in C.unique(kept.iter_grid(1), kept)]
        self.assertEqual(texts, [c.text for c in everything if c.plan.threads == 32])
        self.assertIn("nr=16", texts)
        self.assertNotIn("nr=128", texts)  # 12 tiles
        self.assertIsNone(kept.make({"threads_cap": 16}))

    def test_value_overrides(self):
        vals = C.parse_values(["mb=8|16", "int8_cw=32/16|16"], SCHEMA)
        self.assertEqual(vals, {"mb": [8, 16], "int8_cw": [[32, 16], [16]]})
        sp = C.Space(Problem("s8", 64, 4096, 4096), SCHEMA, UARCH, ["mb", "int8_cw"], vals)
        self.assertEqual(sp.choices["mb"], [8, 16])
        self.assertEqual(sp.size(), 9)
        with self.assertRaises(ValueError):
            C.parse_values(["nr=24"], SCHEMA)


class TestRuleSpace(unittest.TestCase):
    def test_rule_keys_are_tuning_dimensions(self):
        for dtype in ("bf16", "s8"):
            keys = C.search_keys(SCHEMA, dtype)
            self.assertLessEqual(set(RULE_KEYS), set(keys), dtype)
            older = C.search_keys(DESIGN, dtype)
            self.assertFalse(set(RULE_KEYS) & set(older), dtype)
        self.assertNotIn("min_pass_k", C.search_keys(DESIGN, "s8"))
        p = Problem("s8", 32, 4096, 4096)
        sp = C.Space(p, SCHEMA, UARCH)
        # Each rule switch at the value other than its default: the rule off.
        self.assertEqual([sp.choices[k] for k in ("makespan", "l1_pass", "traffic_groups", "blocks")], [[0]] * 4)
        self.assertEqual(sp.choices["decode_m"], [0, 32])
        self.assertEqual(sp.choices["min_pass_k"], [128, 512, 1024])
        self.assertEqual(C.Space(Problem("s8", 128, 4096, 4096), SCHEMA, UARCH).choices["decode_m"], [0])
        for k in RULE_KEYS:
            self.assertTrue(all(SCHEMA.key(k).check(v) is None for v in sp.choices[k]), k)

    def test_candidates_of_rules_off(self):
        p = Problem("bf16", 32, 4096, 4096)
        sp = C.Space(p, SCHEMA, UARCH)
        off_cfg = C.parse_settings(RULES_OFF, SCHEMA)
        off = sp.make(off_cfg)
        self.assertEqual((off.plan.kc, sp.default.plan.kc), (4096, 512))
        self.assertIs(sp.make(C.parse_settings(RULES_OFF.replace("=0", "=1"), SCHEMA)), sp.default)
        # A rule off, or a constant of a rule that is off: the plan without the rule.
        for cfg in ({"l1_pass": 0}, {"makespan": 0, "l1_pass": 0, "traffic_groups": 0},
                    dict(off_cfg, l1_pct=50), dict(off_cfg, decode_m=0)):
            self.assertEqual(sp.make(cfg).sig, off.sig, cfg)
        tuned = sp.make({"l1_pct": 50})
        self.assertEqual((tuned.text, tuned.plan.kc), ("l1_pct=50", 274))
        self.assertIs(sp.make({"l1_pass": 1}), sp.default)
        self.assertEqual(sp.from_text("l1_pct=50,l1_pass=0").text, "l1_pass=0,l1_pct=50")
        uniq = C.unique(sp.iter_grid(1), sp)
        self.assertIn(off.sig, [c.sig for c in uniq])  # first as kc=4096, the same plan
        self.assertIn("l1_pct=50", [c.text for c in uniq])

    def test_every_rule_candidate_is_legal(self):
        for s in (SCHEMA, OLDER):
            for p in PROBLEMS + [Problem("bf16", 128, 1536, 4096), Problem("bf16", 1031, 96, 5120)]:
                sp = C.Space(p, s, s.uarch, [k for k in C.search_keys(s, p.dtype) if k in RULE_KEYS + ["nr", "mr"]])
                n = 0
                for c in sp.iter_grid(2):
                    self.assertIsNone(validate_plan(c.plan, p, s), c)
                    self.assertTrue(all(s.key(k).same_bits(p.dtype) for k in c.config), c)
                    n += 1
                self.assertGreater(n, 100, (s.source, p))


class TestBf16RuleSpace(unittest.TestCase):
    def test_wide_widths_and_rule_values(self):
        b, s = Problem("bf16", 4096, 4096, 4096), Problem("s8", 4096, 4096, 4096)
        self.assertEqual(C.value_choices("nr", b, SCHEMA), [16, 32, 64, 128])
        self.assertEqual(C.value_choices("nr", b, OLDER), [16, 32])
        self.assertEqual(C.value_choices("nr", s, SCHEMA), [16, 32])
        self.assertEqual(C.value_choices("nr", Problem("bf16", 64, 4096, 4096, pack_nr=64), SCHEMA),
                         [16, 32, 64, 128])
        # Slabs of a wide tile lie inside pack blocks: none on 16-column blocks.
        self.assertEqual(C.value_choices("nr", Problem("bf16", 64, 4096, 4096, pack_nr=16), SCHEMA), [16])
        for key, values in (("bf16_alias", [0]), ("bf16_alias_mr", [4, 6, 12]), ("bf16_narrow_n", [0]),
                            ("bf16_narrow_n_min_m", [1])):
            self.assertEqual(C.value_choices(key, b, SCHEMA), values, key)
            self.assertIn(key, C.search_keys(SCHEMA, "bf16"))
            self.assertNotIn(key, C.search_keys(SCHEMA, "s8"))
            self.assertNotIn(key, C.search_keys(OLDER, "bf16"))
        sp = C.Space(b, SCHEMA, UARCH, ["nr", "bf16_alias", "bf16_alias_mr"])
        texts = [c.text for c in C.unique(sp.iter_grid(1), sp)]
        # bf16_alias_mr=12 keeps tall_rows: the plan of bf16_alias=0.
        self.assertEqual(texts, ["nr=16", "nr=64", "nr=128", "bf16_alias=0", "bf16_alias_mr=4", "bf16_alias_mr=6"])


class TestBlocksSpace(unittest.TestCase):
    def test_values(self):
        b, s = Problem("bf16", 4096, 4096, 14336), Problem("s8", 4096, 4096, 4096)
        self.assertEqual(C.value_choices("schedule", b, SCHEMA), ["slice", "row_groups", "blocks"])
        self.assertEqual(C.value_choices("schedule", b, NO_BLOCKS), ["slice", "row_groups"])
        self.assertEqual([C.value_choices(k, b, SCHEMA) for k in ("unit_tiles", "blocks")], [[2, 4], [0]])
        for key, dtype, other in (("bf16_blocks_min_m", "bf16", "s8"), ("int8_blocks_min_m", "s8", "bf16")):
            self.assertEqual(C.value_choices(key, b if dtype == "bf16" else s, SCHEMA), [1], key)
            self.assertIn(key, C.search_keys(SCHEMA, dtype))
            self.assertNotIn(key, C.search_keys(SCHEMA, other))
            self.assertNotIn(key, C.search_keys(NO_BLOCKS, dtype))
        self.assertIsNotNone(C.check_config({"schedule": "blocks", "col_groups": 0}, SCHEMA))
        # Where the rule takes the call, schedule=blocks is the built-in plan and blocks=0 the slice.
        for p in (b, s):
            sp = C.Space(p, SCHEMA, UARCH, ["schedule", "unit_tiles", "blocks"])
            self.assertEqual(sp.default.plan.unit_tiles, 1)
            got = [(c.text, c.plan.unit_tiles) for c in C.unique(sp.iter_grid(2), sp)]
            self.assertEqual(got, [("schedule=slice", 0), ("schedule=row_groups", 0), ("unit_tiles=2", 2),
                                   ("unit_tiles=4", 4)], p)

    def test_every_candidate_is_legal(self):
        keys = ["schedule", "unit_tiles", "blocks", "bf16_blocks_min_m", "int8_blocks_min_m", "col_groups", "nr",
                "mb", "domain_threads"]
        for p in PROBLEMS + [Problem("bf16", 4096, 4096, 14336), Problem("bf16", 1024, 64, 9216)]:
            sp = C.Space(p, SCHEMA, UARCH, [k for k in keys if SCHEMA.key(k).applies_to(p.dtype)])
            blocks = 0
            for c in sp.iter_grid(2):
                self.assertIsNone(validate_plan(c.plan, p, SCHEMA), c)
                blocks += c.plan.unit_tiles > 0
            self.assertGreater(blocks, 10, p)


class TestUnmirroredKeys(unittest.TestCase):
    """A key of a newer build, which the plan mirror does not read, is neither searched nor measured."""

    def test_not_searched_and_dropped(self):
        self.assertEqual(C.parse_settings("newer_pct=10", NEWER), {"newer_pct": 10})
        for p in PROBLEMS:
            self.assertNotIn("newer_pct", C.search_keys(NEWER, p.dtype))
            self.assertEqual(PL.builtin_plan(p, NEWER.uarch, NEWER), PL.builtin_plan(p, NEWER.uarch, SCHEMA), p)
            # Named on purpose, its candidates resolve as dropped.
            sp = C.Space(p, NEWER, NEWER.uarch, ["newer_pct"], {"newer_pct": [10]})
            self.assertEqual(list(sp.iter_grid(1)), [])
            self.assertIsNone(sp.make({"newer_pct": 10, "nr": 16}))


if __name__ == "__main__":
    unittest.main()
