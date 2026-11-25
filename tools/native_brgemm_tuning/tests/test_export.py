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
"""Export: the tuning-file format of DESIGN section 6.2, the gates, the validator, the rules."""

import copy
import json
import os
import sys
import unittest
import unittest.mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import DESIGN, HAVE_NUMPY, HW, NO_NB, OLDER, SCHEMA, UARCH, session_records, strict_json  # noqa: E402

from native_brgemm_tuning import candidates as C  # noqa: E402
from native_brgemm_tuning import export as E  # noqa: E402
from native_brgemm_tuning import plan as PL  # noqa: E402
from native_brgemm_tuning import results as R  # noqa: E402
from native_brgemm_tuning.problems import (  # noqa: E402
    MClass, Problem, Shape, parse_m_classes, problems_from_shapes, rule_features)

CLASSES = parse_m_classes("1,2-16:8/16,129-512:256/512")
SHAPES = [(4096, 4096), (6144, 4096)]
PROBLEMS = [Problem("bf16", m, n, k) for n, k in SHAPES for m in (1, 8, 16, 256, 512)]


def planted(p, c, confirm=True):
    if c.text == "nr=16":
        return 0.8 if p.m <= 16 else 1.05
    if c.text == "threads_cap=16":
        return 0.99 if p.m >= 256 else 1.1
    if c.text == "block_rows=96":
        return 0.85 if p.m >= 256 else 1.02
    return 1.0


class Fixture:
    def __init__(self, problems=PROBLEMS, texts=("nr=16", "threads_cap=16", "block_rows=96"), ratio=planted,
                 allow_bit_changing=False, schema=SCHEMA):
        self.problems = {p.key(): p for p in problems}
        self.spaces = {k: C.Space(p, schema, UARCH, allow_bit_changing=allow_bit_changing)
                       for k, p in self.problems.items()}
        self.batch = {k: [sp.from_text(t) for t in texts] for k, sp in self.spaces.items()}
        self.records = session_records(self.problems, self.batch, ratio, "s001")
        self.ratio = ratio

    def winners(self, classes=CLASSES):
        return E.select_winners(self.records, list(self.problems.values()), classes, self.spaces)

    def confirm(self, winners, ratio=None):
        batch = E.confirm_batch(winners, self.spaces)
        self.records += session_records(self.problems, batch, ratio or self.ratio, "c001", phase="confirm",
                                        repeats=3)
        E.attach_confirmation(winners, self.records, self.spaces)


class TestEntries(unittest.TestCase):
    def test_file_matches_the_design_format(self):
        fx = Fixture()
        w = fx.winners()
        fx.confirm(w)
        entries, skipped = E.build_entries(w, fx.spaces, SCHEMA, UARCH)
        doc = E.build_doc(entries, None, SCHEMA, meta={"note": "test"})
        self.assertEqual(doc["format"], "zendnnl-native-brgemm-tuning")
        self.assertIs(type(doc["version"]), int)
        self.assertEqual(doc["version"], 2)
        self.assertEqual(doc["hardware"], {k: HW[k] for k in E.HW_DEFAULT_KEYS})
        self.assertEqual(list(doc["hardware"]), list(E.HW_DEFAULT_KEYS))
        self.assertEqual(E.validate_doc(doc, SCHEMA), ([], []))
        self.assertEqual(json.loads(json.dumps(doc)), doc)
        ids = [e["id"] for e in entries]
        self.assertEqual(ids, ["bf16-n4096-k4096-m2-16-t32", "bf16-n4096-k4096-m129-512-t32",
                               "bf16-n6144-k4096-m2-16-t32", "bf16-n6144-k4096-m129-512-t32"])
        e = entries[0]
        self.assertEqual(set(e), {"id", "match", "config"})
        self.assertEqual(e["match"], {"dtype": "bf16", "src": "bf16", "dst": "bf16", "trans_a": False,
                                      "dynamic": False, "has_bias": False, "epilogue": "store", "n": 4096,
                                      "k": 4096, "threads": 32, "pack_nr": 32, "m": {"min": 2, "max": 16}})
        cfg = e["config"]
        self.assertEqual(cfg["nr"], 16)
        self.assertIs(cfg["row_split"], True)
        self.assertEqual(cfg["schedule"], "auto")
        self.assertEqual(cfg["threads_cap"], 32)
        self.assertEqual(cfg["domain_threads"], UARCH.plan_domain)
        self.assertEqual(list(cfg), [k.name for k in SCHEMA.keys if k.name in cfg])
        self.assertEqual(entries[1]["config"]["block_rows"], 96)
        # The class of M = 1, which the GEMV kernels run, has no entry.
        self.assertEqual(skipped, [{"id": "bf16-n%d-k4096-m1-t32" % n, "reason": "M = 1 calls run on the GEMV kernels"}
                                   for n in (4096, 6144)])
        for m, layer in ((12, "table:bf16-n4096-k4096-m2-16-t32"), (300, "table:bf16-n4096-k4096-m129-512-t32")):
            got, layers = E.resolve_config(doc, PROBLEMS[0].with_m(m), SCHEMA, UARCH)
            self.assertEqual(layers, [layer])
        self.assertEqual(E.resolve_config(doc, PROBLEMS[0].with_m(100), SCHEMA, UARCH), ({}, []))

    def test_full_config_reproduces_the_measured_plan(self):
        for schema, auto in ((SCHEMA, {"mb", "kc", "nb"}), (DESIGN, {"mb"})):
            fx = Fixture(schema=schema)
            w = [x for x in fx.winners() if x.entry_id() == "bf16-n4096-k4096-m2-16-t32"][0]
            full = E.full_config(w, fx.spaces, schema, UARCH)
            for p in w.problems:
                sp = fx.spaces[p.key()]
                self.assertEqual(sp.make(full).sig, sp.from_text(w.text).sig)
                self.assertEqual(set(sp.keys) - set(full), auto, schema.source)
            # An auto pin whose plan value differs across the class's Ms stays auto: kc of the L1 pass at
            # M=16 only; nb of plans without the blocks schedule.
            kcs = [fx.spaces[p.key()].from_text(w.text).plan.kc for p in w.problems]
            self.assertEqual((kcs, full.get("kc")), ([4096, 1024], None) if "kc" in auto else ([4096] * 2, 4096))

    def test_full_config_pins_the_unit_width_of_a_blocks_plan(self):
        # BF16 K = 14336 at M 2048 and 4096: the blocks rule takes both built-in plans, of 32-column
        # tiles.  A blocks plan pins its unit as nb, its columns; on a schema without nb, as unit_tiles;
        # a winner's unit_tiles stays its unit alone.
        probs = [Problem("bf16", m, 4096, 14336) for m in (2048, 4096)]
        classes = parse_m_classes("1025-4096:2048/4096")
        for schema, text, nb, units, schedule in ((SCHEMA, "nb=128", 128, None, "auto"),
                                                 (SCHEMA, "unit_tiles=2", None, 2, "auto"),
                                                 (SCHEMA, "kc=4096", 32, None, "auto"),
                                                 (SCHEMA, "schedule=slice", None, None, "slice"),
                                                 (NO_NB, "unit_tiles=2", None, 2, "auto"),
                                                 (NO_NB, "kc=4096", None, 1, "auto")):
            fx = Fixture(probs, texts=(text,), ratio=lambda p, c: 0.9 if c.text else 1.0, schema=schema)
            self.assertEqual([fx.spaces[p.key()].default.plan.unit_tiles for p in probs], [1, 1])
            w = [x for x in fx.winners(classes) if x.text]
            self.assertEqual([x.text for x in w], [text])
            full = E.full_config(w[0], fx.spaces, schema, UARCH)
            self.assertEqual((full.get("nb"), full.get("unit_tiles"), full["schedule"]), (nb, units, schedule),
                             (schema.source, text))
        # The unit is pinned as the key the search took: unit_tiles for --keys naming it but not nb.
        fx = Fixture(probs, texts=("kc=4096",), ratio=lambda p, c: 0.9 if c.text else 1.0, schema=SCHEMA)
        w = [x for x in fx.winners(classes) if x.text]
        for keys, nb, units in ((["schedule", "unit_tiles", "kc"], None, 1), (["schedule", "nb", "kc"], 32, None),
                                (["schedule", "nb", "unit_tiles", "kc"], 32, None), (["schedule", "kc"], None, None)):
            spaces = {k: C.Space(p, SCHEMA, UARCH, keys) for k, p in fx.problems.items()}
            full = E.full_config(w[0], spaces, SCHEMA, UARCH)
            self.assertEqual((full.get("nb"), full.get("unit_tiles"), full["kc"]), (nb, units, 4096), keys)

    def test_full_config_keeps_the_unit_key_the_winner_sets(self):
        # F16 calls take the F16_PH tiles of 64 columns at M 256 and 512, of 32 at M 129 to 191
        # (f16_wide_min_m).  With nb and unit_tiles searched and nr not, an nb pin of 128 beside a
        # winning unit_tiles=2 would take 4 tiles at M 129: the entry keeps the winner's unit key
        # alone, and its unit is the winner's at every M of the class.
        probs = [Problem("f16", m, 4096, 4096) for m in (256, 512)]
        classes = parse_m_classes("129-512:256/512")
        keys = ["schedule", "nb", "unit_tiles", "kc"]
        for text, nb, units in (("schedule=blocks,unit_tiles=2", None, 2), ("nb=128,schedule=blocks", 128, None)):
            fx = Fixture(probs, texts=(text,), ratio=lambda p, c: 0.9 if c.text else 1.0)
            w = [x for x in fx.winners(classes) if x.text]
            self.assertEqual([x.text for x in w], [text])
            spaces = {k: C.Space(p, SCHEMA, UARCH, keys) for k, p in fx.problems.items()}
            full = E.full_config(w[0], spaces, SCHEMA, UARCH)
            self.assertEqual((full.get("nb"), full.get("unit_tiles"), full["schedule"]), (nb, units, "blocks"), text)
            won = C.parse_settings(text, SCHEMA)
            tiles = {}
            for m in E.class_ms(classes[0]):
                p = probs[0].with_m(m)
                entry, winner = (PL.resolve(p, cfg, UARCH, SCHEMA).plan for cfg in (full, won))
                self.assertEqual((entry.nr, entry.unit_tiles), (winner.nr, winner.unit_tiles), (text, m))
                tiles[entry.nr] = entry.unit_tiles
            self.assertEqual(tiles, {32: 2, 64: 2} if units else {32: 4, 64: 2}, text)

    def check_rule_entries(self, schema):
        """Entries of a class won by the L1 pass off and of one won by nr=16 under the built-in rules."""
        probs = [Problem("bf16", m, 4096, 4096) for m in (8, 32, 64)]
        classes = parse_m_classes("1-16:8,17-64:32/64")
        text = "l1_pass=0"

        def ratio(p, c):
            return {text: 0.85, "nr=16": 0.9 if p.m <= 16 else 1.0}.get(c.text, 1.0)

        fx = Fixture(probs, texts=(text, "nr=16"), ratio=ratio, schema=schema)
        w = fx.winners(classes)
        fx.confirm(w)
        entries, skipped = E.build_entries(w, fx.spaces, schema, UARCH)
        self.assertEqual(skipped, [])
        cfg = {e["id"]: e["config"] for e in entries}
        tuned, kept = cfg["bf16-n4096-k4096-m17-64-t32"], cfg["bf16-n4096-k4096-m1-16-t32"]  # one plan at M=8
        # The entry of class 1-16 leaves M = 1 to the GEMV kernels.
        self.assertEqual({e["id"]: e["match"]["m"] for e in entries},
                         {"bf16-n4096-k4096-m1-16-t32": {"min": 2, "max": 16},
                          "bf16-n4096-k4096-m17-64-t32": {"min": 17, "max": 64}})
        # Each rule switch at its built-in value unless the winner sets it, so no other layer turns a
        # rule on or off.
        switches = tuple(k for k in PL.RULE_SWITCHES if k in schema.key_map and schema.key(k).applies_to("bf16"))
        self.assertEqual([tuned[k] for k in switches + ("l1_pct",)],
                         [int(k != "l1_pass") for k in switches] + [PL.builtin_params(schema)["l1_pct"]])
        self.assertEqual([kept[k] for k in ("nr",) + switches], [16] + [1] * len(switches))
        rule = {"l1_pass": 1, "l1_pct": 50}
        doc = E.build_doc(entries, {"kind": "ordered", "list": [{"when": [], "config": rule}]}, schema)
        self.assertEqual(E.validate_doc(doc, schema), ([], []))
        self.assertTrue(E.validate_doc(json.loads(json.dumps(doc).replace('"m_inner"', '"M_inner"')), schema)[0])

        def plan(p):
            cfg, _ = E.resolve_config(doc, p, schema, UARCH)
            return PL.resolve(p, E.config_from_json(cfg, schema), UARCH, schema).plan

        def plan_of(text, p):
            return PL.resolve(p, C.parse_settings(text, schema), UARCH, schema).plan

        for p in probs[1:]:
            self.assertEqual(plan(p), plan_of(text, p), p.m)
        self.assertEqual((plan(probs[1]).kc, plan_of("", probs[1]).kc), (4096, 512))
        self.assertEqual(plan(probs[0]), plan_of("nr=16", probs[0]))
        self.assertEqual(plan(probs[0].with_m(100)), plan_of(C.format_settings(rule, schema), probs[0].with_m(100)))
        self.assertEqual(E.validate_doc(doc, DESIGN)[0][0],
                         "entry 0 (bf16-n4096-k4096-m1-16-t32): unknown config key 'makespan'")
        # With the switches among the searched keys, an entry pins them at their built-in value.
        spaces = {k: C.Space(p, schema, UARCH, ["nr"] + list(switches)) for k, p in fx.problems.items()}
        full = E.full_config([x for x in w if x.text == "nr=16"][0], spaces, schema, UARCH)
        self.assertEqual(full, dict({"nr": 16}, **{k: 1 for k in switches}))

    def test_entries_pin_the_rule_switches(self):
        self.check_rule_entries(SCHEMA)

    def test_entries_of_a_build_without_the_bf16_rules(self):
        self.check_rule_entries(OLDER)

    def test_m1_class_takes_the_gemv_keys(self):
        # The class of M = 1 alone, whose calls run on the GEMV kernels: a winner of their keys gets
        # an entry at M 1, its config the GEMV keys at their measured or built-in values; a winner of
        # the tiles' keys none.  A class that measures M 1 next to another M is refused.
        def gemv(p, c):
            return 0.8 if c.text == "gemv_blocks=1" else 0.9 if c.text == "nr=16" else 1.0

        probs = [Problem("bf16", 1, 4096, 4096), Problem("bf16", 8, 4096, 4096)]
        fx = Fixture(probs, texts=("gemv_blocks=1",), ratio=gemv)
        w = fx.winners(parse_m_classes("1,2-16:8"))
        fx.confirm(w)
        entries, skipped = E.build_entries(w, fx.spaces, SCHEMA, UARCH)
        self.assertEqual([e["id"] for e in entries], ["bf16-n4096-k4096-m1-t32"])
        self.assertEqual(entries[0]["match"]["m"], {"min": 1, "max": 1})
        self.assertEqual(entries[0]["config"], {"gemv": True, "gemv_blocks": 1, "gemv_one_thread_pct": 25})
        self.assertEqual([s["id"] for s in skipped], ["bf16-n4096-k4096-m2-16-t32"])
        doc = E.build_doc(entries, None, SCHEMA)
        self.assertEqual(E.validate_doc(doc, SCHEMA), ([], []))
        cfg, layers = E.resolve_config(doc, probs[0], SCHEMA, UARCH)
        q = PL.resolve(probs[0], E.config_from_json(cfg, SCHEMA), UARCH, SCHEMA).plan
        self.assertEqual((layers, q.gemv, q.gemv_call_blocks), (["table:bf16-n4096-k4096-m1-t32"], True, 1))
        self.assertEqual(E.resolve_config(doc, probs[1], SCHEMA, UARCH), ({}, []))
        tiles = Fixture(probs[:1], texts=("nr=16",), ratio=gemv)
        w = tiles.winners(parse_m_classes("1"))
        tiles.confirm(w)
        self.assertEqual(E.build_entries(w, tiles.spaces, SCHEMA, UARCH),
                         ([], [{"id": "bf16-n4096-k4096-m1-t32", "reason": "M = 1 calls run on the GEMV kernels"}]))
        # A cap on the plan's threads gives the GEMV kernels 8 of 32, another schedule: its winner
        # gets an entry at M 1, with the cap.
        capped = Fixture(probs[:1], texts=("threads_cap=8",), ratio=lambda p, c: 0.8)
        w = capped.winners(parse_m_classes("1"))
        capped.confirm(w)
        entries, skipped = E.build_entries(w, capped.spaces, SCHEMA, UARCH)
        self.assertEqual(([e["id"] for e in entries], skipped), (["bf16-n4096-k4096-m1-t32"], []))
        self.assertEqual(entries[0]["config"]["threads_cap"], 8)
        cfg, _ = E.resolve_config(E.build_doc(entries, None, SCHEMA), probs[0], SCHEMA, UARCH)
        self.assertEqual(PL.resolve(probs[0], E.config_from_json(cfg, SCHEMA), UARCH, SCHEMA).plan.gemv_threads, 8)
        for text in ("1-16:1/8", "1-4:1"):
            with self.assertRaises(ValueError, msg=text):
                parse_m_classes(text)
        self.assertEqual(parse_m_classes("1-16:8")[0].reps, (8,))

    def test_a_problem_at_m1_is_a_class_of_its_own(self):
        # A shape whose M range is 1 alone meets the class 1-16 at M 1 (its representative 8 taken down
        # to 1): that problem is judged and exported as a class of M 1 alone, so a winner of the GEMV
        # keys gets an entry at M 1, never one over M 2-16, which runs on the tiles; a problem at M 8 of
        # the same shape keeps the class, from M 2.
        classes = parse_m_classes("1-16:8")
        probs = problems_from_shapes([Shape("bf16", 4096, 4096, m_lo=1, m_hi=1)], classes)
        self.assertEqual([p.m for p in probs], [1])

        def ratio(p, c):
            return 0.8 if (c.text == "gemv_blocks=1") == (p.m == 1) else 1.0

        for extra in ([], [Problem("bf16", 8, 4096, 4096)]):
            fx = Fixture(probs + extra, texts=("gemv_blocks=1", "nr=16"), ratio=ratio)
            w = fx.winners(classes)
            self.assertEqual([(x.mclass.label(), [p.m for p in x.problems]) for x in w],
                             [("1", [1])] + [("1-16", [8])] * len(extra))
            fx.confirm(w)
            entries, skipped = E.build_entries(w, fx.spaces, SCHEMA, UARCH)
            self.assertEqual([(e["id"], e["match"]["m"], e["config"].get("gemv_blocks", e["config"].get("nr")))
                              for e in entries],
                             [("bf16-n4096-k4096-m1-t32", {"min": 1, "max": 1}, 1)]
                             + [("bf16-n4096-k4096-m1-16-t32", {"min": 2, "max": 16}, 16)] * len(extra))
            self.assertEqual(skipped, [])
            doc = E.build_doc(entries, None, SCHEMA)
            self.assertEqual(E.resolve_config(doc, probs[0].with_m(4), SCHEMA, UARCH)[1],
                             ["table:bf16-n4096-k4096-m1-16-t32"] if extra else [])

    def test_noise_and_confirmation_gates(self):
        fx = Fixture(texts=("threads_cap=16", "nr=16"))
        w = fx.winners()
        big = [x for x in w if x.mclass.lo == 129]
        self.assertTrue(all(not x.significant and "noise" in x.reason for x in big))
        entries, skipped = E.build_entries(w, fx.spaces, SCHEMA, UARCH)
        self.assertEqual(entries, [])
        self.assertTrue(all("confirmation" in s["reason"] or "noise" in s["reason"] or "GEMV" in s["reason"]
                            for s in skipped))
        self.assertEqual(len(E.build_entries(w, fx.spaces, SCHEMA, UARCH, require_confirm=False)[0]), 2)
        fx.confirm(w, ratio=lambda p, c: 1.0)
        entries, skipped = E.build_entries(w, fx.spaces, SCHEMA, UARCH)
        self.assertEqual(entries, [])
        self.assertTrue(any("disagrees" in s["reason"] for s in skipped))

    def test_confirmation_falls_back_to_the_next_confirmed_candidate(self):
        def two(p, c):
            return {"nr=16": 0.80, "kc=1024": 0.90}.get(c.text, 1.0) if p.m <= 16 else 1.0

        fx = Fixture(texts=("nr=16", "kc=1024"), ratio=two)
        w = [x for x in fx.winners() if x.entry_id() == "bf16-n4096-k4096-m2-16-t32"]
        self.assertEqual([t for t, _ in w[0].alternatives], ["nr=16", "kc=1024"])
        fx.confirm(w, ratio=lambda p, c: 1.0)
        self.assertFalse(w[0].confirmed)
        self.assertTrue(E.confirmed_before(w[0], fx.records, fx.spaces))
        alt = {k: [fx.spaces[k].from_text("kc=1024")] for k in fx.problems if fx.problems[k].m in (8, 16)
               and fx.problems[k].n == 4096}
        fx.records += session_records(fx.problems, alt, two, "c002", phase="confirm", repeats=3)
        w = [x for x in fx.winners() if x.entry_id() == "bf16-n4096-k4096-m2-16-t32"]
        E.attach_confirmation(w, fx.records, fx.spaces)
        self.assertTrue(w[0].confirmed)
        self.assertEqual(w[0].text, "kc=1024")
        self.assertIn("next confirmed candidate", w[0].reason)
        self.assertAlmostEqual(w[0].ratio, 0.90, places=6)
        entries, _ = E.build_entries(w, fx.spaces, SCHEMA, UARCH)
        self.assertEqual(entries[0]["config"]["kc"], 1024)

    def test_fallback_winners(self):
        def two(p, c):
            return {"nr=16": 0.80, "kc=1024": 0.90}.get(c.text, 1.0) if p.m <= 16 else 1.0

        fx = Fixture(texts=("nr=16", "kc=1024"), ratio=two)
        w = [x for x in fx.winners() if x.entry_id() == "bf16-n4096-k4096-m2-16-t32"]
        self.assertEqual([x.text for x in E.fallback_winners(w, fx.records, fx.spaces)], ["nr=16"])
        fx.confirm(w, ratio=lambda p, c: 1.0)
        w = [x for x in fx.winners() if x.entry_id() == "bf16-n4096-k4096-m2-16-t32"]
        fb = E.fallback_winners(w, fx.records, fx.spaces)
        self.assertEqual([x.text for x in fb], ["kc=1024"])
        fx.confirm(fb, ratio=two)
        w = [x for x in fx.winners() if x.entry_id() == "bf16-n4096-k4096-m2-16-t32"]
        self.assertEqual(E.fallback_winners(w, fx.records, fx.spaces), [])
        self.assertTrue(w[0].confirmed)
        self.assertEqual(w[0].text, "kc=1024")

    def test_exclude_entries(self):
        entries = [{"id": i} for i in ("a", "b", "c", "d")]
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_exclude_test.txt")
        with open(path, "w") as f:
            f.write("# slower in validation\nc 1.05\n")
        self.addCleanup(os.remove, path)
        kept, dropped = E.exclude_entries(entries, ["a, x", "@" + path])
        self.assertEqual([e["id"] for e in kept], ["b", "d"])
        self.assertEqual(dropped, [{"id": "a", "reason": "excluded (--exclude)"},
                                   {"id": "c", "reason": "excluded (--exclude)"}])

    def test_bits_gate(self):
        p = Problem("s8", 8, 4096, 4096, has_bias=True)
        fx = Fixture([p], texts=("bias_in_dequant=0",), ratio=lambda p, c: 0.8, allow_bit_changing=True)
        w = fx.winners(parse_m_classes("1-16:8"))
        fx.confirm(w)
        entries, skipped = E.build_entries(w, fx.spaces, SCHEMA, UARCH)
        self.assertEqual(entries, [])
        self.assertIn("accuracy", skipped[0]["reason"])
        entries, _ = E.build_entries(w, fx.spaces, SCHEMA, UARCH, allow_bit_changing=True)
        self.assertIs(entries[0]["config"]["bias_in_dequant"], False)

    def test_cross_measurement_across_the_ms_of_a_class(self):
        p8, p16 = Problem("bf16", 8, 4096, 4096), Problem("bf16", 16, 4096, 4096)
        cls = parse_m_classes("2-16:8/16")
        plan = {(8, "nr=16"): 0.8, (16, "threads_cap=16"): 0.85, (16, "nr=16"): 0.82, (8, "threads_cap=16"): 0.95}
        fx = Fixture([p8, p16], texts=())
        sp = fx.spaces
        first = {p8.key(): [sp[p8.key()].from_text("nr=16")], p16.key(): [sp[p16.key()].from_text("threads_cap=16")]}
        fx.records = session_records(fx.problems, first, lambda p, c: plan[(p.m, c.text)], "s001")
        self.assertEqual(fx.winners(cls)[0].text, "")
        cross = E.cross_batch(fx.records, [p8, p16], cls, sp)
        self.assertEqual({k: [c.text for c in v] for k, v in cross.items()},
                         {p8.key(): ["threads_cap=16"], p16.key(): ["nr=16"]})
        fx.records += session_records(fx.problems, cross, lambda p, c: plan[(p.m, c.text)], "s002")
        self.assertEqual(E.cross_batch(fx.records, [p8, p16], cls, sp), {})
        w = fx.winners(cls)[0]
        self.assertEqual((w.text, w.significant), ("nr=16", True))
        self.assertAlmostEqual(w.ratio, (0.8 * 0.82) ** 0.5)

    def test_ms_with_the_built_in_plan_are_neutral(self):
        p12, p16 = Problem("bf16", 12, 4096, 4096), Problem("bf16", 16, 4096, 4096)
        fx = Fixture([p12, p16], texts=())
        sp = fx.spaces
        self.assertEqual(sp[p12.key()].from_text("mb=12").sig, sp[p12.key()].default.sig)
        fx.records = session_records(fx.problems, {p16.key(): [sp[p16.key()].from_text("mb=12")]},
                                     lambda p, c: 0.9, "s001")
        w = fx.winners(parse_m_classes("9-16:12/16"))
        self.assertEqual((w[0].text, w[0].significant), ("mb=12", True))
        self.assertEqual(w[0].ratios, {p12.key(): 1.0, p16.key(): 0.9})
        self.assertEqual(list(E.confirm_batch(w, sp)), [p16.key()])
        fx.ratio = lambda p, c: 0.9
        fx.confirm(w)
        entries, _ = E.build_entries(w, sp, SCHEMA, UARCH)
        self.assertEqual(entries[0]["match"]["m"], {"min": 9, "max": 16})

    def test_winner_must_be_legal_over_its_class(self):
        # At M 2: a problem at M 1 is a class of its own.
        p = Problem("bf16", 2, 512, 4096)
        cls = MClass(2, 512, (2,))
        self.assertIsNotNone(C.Space(p, SCHEMA, UARCH).from_text("col_groups=32"))
        self.assertFalse(E.legal_over_class("col_groups=32", p, cls, SCHEMA, UARCH))  # n_tiles 16 at M 256
        self.assertTrue(E.legal_over_class("threads_cap=16", p, cls, SCHEMA, UARCH))
        fx = Fixture([p], texts=("col_groups=32", "threads_cap=16"),
                     ratio=lambda p, c: 0.7 if c.text == "col_groups=32" else 0.9)
        w = fx.winners([cls])
        self.assertEqual(w[0].text, "threads_cap=16")
        self.assertEqual(E.class_ms(MClass(1, 1, (1,))), [1])
        self.assertEqual(E.class_ms(MClass(2, 16, (16,))), list(range(2, 17)))
        big = E.class_ms(MClass(513, None, (1024,)))
        self.assertEqual((big[0], big[-1]), (513, 16384))


class TestValidator(unittest.TestCase):
    def setUp(self):
        self.doc = {"format": E.FORMAT, "version": 2, "hardware": {"isa": ["avx512bf16"], "l2_kib": {"min": 512}},
                    "entries": [{"id": "x", "match": {"dtype": "bf16", "n": 4096, "m": {"min": 1, "max": 16}},
                                 "config": {"nr": 16, "int8_cw": [32, 16], "row_split": False}}],
                    "rules": {"kind": "ordered", "list": [{"when": [["m", "<=", 16]], "config": {"nr": 16}}]}}

    def errors(self, mutate):
        d = copy.deepcopy(self.doc)
        mutate(d)
        return E.validate_doc(d, SCHEMA)[0]

    def test_valid(self):
        self.assertEqual(E.validate_doc(self.doc, SCHEMA), ([], []))
        d = dict(self.doc, extra=1)
        self.assertEqual(E.validate_doc(d, SCHEMA)[0], [])
        self.assertEqual(len(E.validate_doc(d, SCHEMA)[1]), 1)

    def test_errors(self):
        bad = [
            lambda d: d.update(format="other"),
            lambda d: d.update(version="2"),
            lambda d: d.update(version=True),
            lambda d: d.update(version=1),
            lambda d: d.update(version=3),
            lambda d: d.update(version=2.0),
            lambda d: d.update(hardware={"signature": "v1;isa=none"}),
            lambda d: d["hardware"].update(cpus=3),
            lambda d: d["entries"][0]["match"].update(foo=1),
            lambda d: d["entries"][0]["match"].update(n="4096"),
            lambda d: d["entries"][0]["match"].update(dtype="u4"),
            lambda d: d["entries"][0]["match"].update(trans_a=0),
            lambda d: d["entries"][0]["config"].update(nr=24),
            lambda d: d["entries"][0]["config"].update(row_split=1),
            lambda d: d["entries"][0]["config"].update(schedule="slice", col_groups=4),
            lambda d: d["entries"][0]["config"].update(int8_cw=[32]),
            lambda d: d["entries"][0]["config"].update(unknown_key=1),
            lambda d: d["rules"].update(kind="forest"),
            lambda d: d["rules"]["list"][0]["when"].append(["bogus", "<", 1]),
            lambda d: d["rules"]["list"][0]["when"].append(["m", "=~", 1]),
            lambda d: d["rules"]["list"][0]["when"].extend([["m", ">", 0]] * 32),
            lambda d: d["rules"]["list"][0].update(config={"nr": 24}),
        ]
        for i, mutate in enumerate(bad):
            self.assertTrue(self.errors(mutate), i)

    def test_strict_json_within_the_loader_bounds(self):
        # The loader ignores the whole file when it cannot parse it or the file exceeds a bound.
        for x in (float("nan"), float("inf"), -float("inf")):
            errs = self.errors(lambda d: d.update(meta={"rules_report": {"geomean_ratio": x}}))
            self.assertEqual(len(errs), 1, x)
            self.assertIn("not strict JSON", errs[0])
        self.assertIn("not strict JSON", self.errors(lambda d: d["entries"][0].update(id="x\udcff"))[0])
        self.assertEqual(self.errors(lambda d: d.update(meta={"rules_report": {"geomean_ratio": None}})), [])
        self.assertEqual(strict_json(E.file_bytes(self.doc).decode("utf-8")), self.doc)

        def nested(levels):
            v = []
            for _ in range(levels - 1):
                v = [v]
            return v

        # meta.x is below the document and meta.
        self.assertEqual(self.errors(lambda d: d.update(meta={"x": nested(E.MAX_JSON_NESTING - 2)})), [])
        self.assertEqual(self.errors(lambda d: d.update(meta={"x": nested(E.MAX_JSON_NESTING - 1)})),
                         ["the file nests arrays and objects deeper than 128 levels"])
        self.assertEqual(E.json_shape('{"a": "x,[{\\"", "b": [1, 2, {"c": [], "d": "]}\\\\"}]}'), (4, 9))
        data = E.file_bytes(self.doc)
        values = E.json_shape(data.decode("utf-8"))[1]
        with unittest.mock.patch.object(E, "MAX_JSON_VALUES", values):
            with unittest.mock.patch.object(E, "MAX_BYTES", len(data)):
                self.assertEqual(self.errors(lambda d: None), [])
        with unittest.mock.patch.object(E, "MAX_JSON_VALUES", values - 1):
            self.assertEqual(self.errors(lambda d: None), ["the file has more than %d values" % (values - 1)])
        # The size of the file as it is written, not of a compact text.
        with unittest.mock.patch.object(E, "MAX_BYTES", len(json.dumps(self.doc))):
            self.assertEqual(self.errors(lambda d: None), ["the file is larger than 64 MiB"])

    def test_tree_bounds(self):
        node = {"config": {}}
        for _ in range(70):
            node = {"feature": "m", "le": 1, "then": node, "else": {"config": {}}}
        self.assertTrue(self.errors(lambda d: d.update(rules={"kind": "tree", "root": node})))
        leaf = {"config": {"nr": 24}}
        self.assertTrue(self.errors(lambda d: d.update(rules={"kind": "tree", "root": leaf})))

    def test_loader_simulation(self):
        p = Problem("bf16", 8, 4096, 4096)
        doc = {"format": E.FORMAT, "version": 2,
               "entries": [{"id": "a", "match": {"n": 4096, "m": {"max": 16}}, "config": {"tile_rows": 8}},
                           {"id": "b", "match": {"n": 4096}, "config": {"tile_rows": 4}}],
               "rules": {"kind": "tree", "root": {"feature": "m", "le": 16,
                                                  "then": {"config": {"nr": 16, "tile_rows": 4}},
                                                  "else": {"config": {}}}}}
        cfg, layers = E.resolve_config(doc, p, SCHEMA, UARCH)
        self.assertEqual(cfg, {"nr": 16, "tile_rows": 8})
        self.assertEqual(layers, ["rules", "table:a"])
        self.assertEqual(E.resolve_config(doc, p.with_m(64), SCHEMA, UARCH)[1], ["table:b"])
        # The loader reads version 2 files only.
        self.assertEqual(E.resolve_config(dict(doc, version=1), p, SCHEMA, UARCH), (None, []))
        doc["hardware"] = {"l2_kib": 1}
        self.assertEqual(E.resolve_config(doc, p, SCHEMA, UARCH), (None, []))
        self.assertEqual(E.hardware_mismatch({"cores": {"min": HW["cores"] + 1}}, SCHEMA), "cores")
        self.assertIsNone(E.hardware_mismatch(E.hardware_block(SCHEMA, "fields"), SCHEMA))
        doc.update(hardware=E.hardware_block(SCHEMA))
        self.assertEqual(E.resolve_config(doc, p, SCHEMA, UARCH)[1], ["rules", "table:a"])
        doc["hardware"]["model"] = 3
        self.assertEqual(E.resolve_config(doc, p, SCHEMA, UARCH), (None, []))

    def test_paired_keys_merge_together(self):
        p = Problem("bf16", 4096, 4096, 4096)

        def doc(rule, entry):
            return {"format": E.FORMAT, "version": 2,
                    "entries": [{"id": "e", "match": {"n": 4096}, "config": entry}],
                    "rules": {"kind": "ordered", "list": [{"when": [], "config": rule}]}}

        # A layer that sets either key of a pair (schedule and col_groups; nb and unit_tiles)
        # replaces both, as merge_native_config does.
        cases = (({"schedule": "row_groups", "nr": 16}, {"col_groups": 0}, {"nr": 16, "col_groups": 0}),
                 ({"col_groups": 4}, {"schedule": "slice"}, {"schedule": "slice"}),
                 ({"schedule": "row_groups"}, {"mb": 96}, {"schedule": "row_groups", "mb": 96}),
                 ({"unit_tiles": 4, "nr": 16}, {"nb": 128}, {"nr": 16, "nb": 128}),
                 ({"nb": 128}, {"unit_tiles": 2}, {"unit_tiles": 2}),
                 ({"nb": 64}, {"mb": 96}, {"nb": 64, "mb": 96}))
        for rule, entry, merged in cases:
            cfg, layers = E.resolve_config(doc(rule, entry), p, SCHEMA, UARCH)
            self.assertEqual((cfg, layers), (merged, ["rules", "table:e"]), (rule, entry))
            self.assertIsNone(C.check_config(E.config_from_json(cfg, SCHEMA), SCHEMA))
        self.assertEqual(E.merge_config({"schedule": "slice", "nr": 16}, {}), {"schedule": "slice", "nr": 16})


def machine(drop=(), **fields):
    """SCHEMA with its machine's hardware fields changed (or dropped: not reported)."""
    s = copy.copy(SCHEMA)
    s.hw_match = {k: v for k, v in dict(SCHEMA.hw_match, **fields).items() if k not in drop}
    return s


class TestInt4Entries(unittest.TestCase):
    def test_s4_entries_match_their_group(self):
        p = Problem("s4", 16, 4096, 4096, group=128)
        m = E.match_block(p, 9, 16)
        self.assertEqual((m["dtype"], m["group"]), ("s4", 128))
        self.assertTrue(E.match_entry(m, p))
        self.assertFalse(E.match_entry(m, Problem("s4", 16, 4096, 4096, group=4096)))
        self.assertNotIn("group", E.match_block(Problem("s8", 16, 4096, 4096), 9, 16))
        w = E.Winner(p.shape_key(), MClass(9, 16, (16,)), [p])
        self.assertEqual(w.entry_id(), "s4-g128-n4096-k4096-m9-16-t32")
        doc = {"format": E.FORMAT, "version": 2, "entries": [{"id": w.entry_id(), "match": m, "config": {"nr": 16}}]}
        self.assertEqual(E.validate_doc(doc, SCHEMA)[0], [])


class TestHardware(unittest.TestCase):
    def test_entries_match_the_padded_width(self):
        p = Problem("s8", 8, 4113, 4096)
        self.assertEqual(E.match_block(p, 1, 16)["n"], 4128)
        self.assertEqual(E.match_block(Problem("s8", 8, 4096, 4096), 1, 16)["n"], 4096)
        # The library matches n at the padded width: the entry holds for every N padded to 4128, and the
        # file's plans and loader simulation name those problems.
        entry = {"id": "e", "match": E.match_block(p, 1, 16), "config": {"mb": 4}}
        doc = {"format": E.FORMAT, "version": 2, "entries": [entry]}
        for n, named in ((4113, True), (4100, True), (4128, True), (4096, False), (4160, False)):
            q = Problem("s8", 8, n, 4096)
            self.assertEqual(E.match_entry(entry["match"], q), named, n)
            self.assertEqual(E.resolve_config(doc, q, SCHEMA, UARCH),
                             ({"mb": 4}, ["table:e"]) if named else ({}, []), n)
            plans = E.entry_plans([entry], [q], SCHEMA, UARCH)
            self.assertEqual([r["problem"] for r in plans], [q.key()] if named else [], n)

    def test_match_modes(self):
        self.assertEqual(E.hardware_block(SCHEMA, "none"), {})
        doc = E.build_doc([], None, SCHEMA, "none")
        self.assertEqual((doc["version"], "hardware" in doc), (2, False))
        self.assertEqual(E.validate_doc(doc, SCHEMA), ([], []))
        with self.assertRaises(ValueError):
            E.hardware_block(SCHEMA, "signature")

    def test_one_file_for_machines_of_a_family(self):
        a = machine(model=2, smt=1, l3_threads=8, cores=256, logical_cpus=256)
        b = machine(model=3, l3_kib=65536, l3_cores=16, smt=1, l3_threads=16, cores=256, logical_cpus=256)
        for_a, for_b = E.hardware_block(a), E.hardware_block(b)
        self.assertIsNone(E.hardware_mismatch(for_a, a))
        self.assertEqual(E.hardware_mismatch(for_a, b), "model")
        self.assertIsNone(E.hardware_mismatch(for_b, b))
        self.assertEqual(E.hardware_mismatch(for_b, a), "model")
        both = E.hardware_block(a, values=E.parse_hw_set(["model=2|3", "l3_kib=32768-65536", "l3_cores=8|16"]))
        self.assertEqual((both["model"], both["l3_kib"], both["l3_cores"]),
                         ([2, 3], {"min": 32768, "max": 65536}, [8, 16]))
        for m in (a, b):
            self.assertIsNone(E.hardware_mismatch(both, m))
        # The same part with SMT on, or with fewer cores, keeps the default block.
        for other in (machine(model=2, smt=2, l3_threads=16, cores=256, logical_cpus=512),
                      machine(model=2, smt=1, l3_threads=8, cores=128, logical_cpus=128)):
            self.assertIsNone(E.hardware_mismatch(for_a, other))
        self.assertEqual(E.hardware_mismatch(E.hardware_block(a, keys=list(E.HW_DEFAULT_KEYS) + ["cores"]),
                                             machine(model=2, cores=128)), "cores")
        doc = E.build_doc([], None, a, hw_values=E.parse_hw_set(["model=2|3"]))
        self.assertEqual(E.validate_doc(doc, SCHEMA), ([], []))

    def test_keys_and_fields_the_machine_does_not_report(self):
        self.assertEqual(E.hardware_block(SCHEMA, keys=["family", "smt", "cores"], values={"model": {"max": 15}}),
                         {"family": HW["family"], "model": {"max": 15}, "smt": HW["smt"], "cores": HW["cores"]})
        old = machine(drop=("vendor", "family", "model", "stepping"))
        with self.assertRaises(ValueError) as e:
            E.hardware_block(old)
        self.assertIn("vendor, family, model", str(e.exception))
        self.assertEqual(E.hardware_block(old, keys=["isa", "l2_kib", "l3_cores"]),
                         {"isa": ["avx512f", "avx512bf16", "avx512vnni", "avx512fp16"], "l2_kib": HW["l2_kib"],
                          "l3_cores": HW["l3_cores"]})
        self.assertEqual(E.hardware_mismatch({"family": {"min": 0}}, old), "family")
        self.assertEqual(E.hardware_mismatch({"smt": {"max": 2}}, machine(drop=("smt",))), "smt")
        self.assertEqual(E.hardware_mismatch({"family": 26, "frequency": 3}, SCHEMA), "frequency")
        with self.assertRaises(ValueError):
            E.hardware_block(SCHEMA, keys=["frequency"])

    def test_isa_subset_vendor_lists_and_order(self):
        self.assertIsNone(E.hardware_mismatch({"isa": "avx512vnni"}, SCHEMA))
        self.assertIsNone(E.hardware_mismatch({"isa": []}, SCHEMA))
        no_vnni = machine(isa=["avx512f", "avx512bf16"])
        self.assertIsNone(E.hardware_mismatch({"isa": ["avx512bf16"]}, no_vnni))
        self.assertEqual(E.hardware_mismatch({"isa": ["avx512bf16", "avx512vnni"]}, no_vnni), "isa")
        self.assertIsNone(E.hardware_mismatch({"vendor": ["OtherVendor0", HW["vendor"]]}, SCHEMA))
        self.assertEqual(E.hardware_mismatch({"vendor": "OtherVendor0"}, SCHEMA), "vendor")
        model = HW["model"]
        self.assertIsNone(E.hardware_mismatch({"model": {"min": model - 1, "max": model + 14}, "l3_cores": [8, 32]},
                                              SCHEMA))
        self.assertEqual(E.hardware_mismatch({"model": {"min": model + 1}}, SCHEMA), "model")
        self.assertEqual(E.hardware_mismatch({"smt": HW["smt"], "model": model + 1}, SCHEMA), "model")

    def test_validator(self):
        good = {"format": E.FORMAT, "version": 2,
                "hardware": E.hardware_block(SCHEMA, values={"model": {"min": 0}, "vendor": ["A", "B"],
                                                             "isa": "avx512f", "cores": [64, 128]})}
        self.assertEqual(E.validate_doc(good, SCHEMA), ([], []))
        for hw in ({"signature": "v1;isa=none"}, {"frequency": 3}, {"model": {"min": 5, "max": 2}},
                   {"model": {}}, {"model": {"lo": 1}}, {"model": []}, {"model": [1, "2"]}, {"model": "2"},
                   {"model": 2.0}, {"model": True}, {"model": None}, {"vendor": 5}, {"vendor": []},
                   {"isa": ["amx"]}, {"isa": {"avx512f": True}}, {"isa": ["avx512f", 3]}):
            self.assertTrue(E.validate_doc(dict(good, hardware=hw), SCHEMA)[0], hw)

    def test_parse_hw_set(self):
        self.assertEqual(E.parse_hw_set(["model=0-15", "l3_kib=32768-", "cores=-128", "l3_cores=8|16", "family=26",
                                         "vendor=A|B", "isa=avx512bf16+avx512vnni"]),
                         {"model": {"min": 0, "max": 15}, "l3_kib": {"min": 32768}, "cores": {"max": 128},
                          "l3_cores": [8, 16], "family": 26, "vendor": ["A", "B"],
                          "isa": ["avx512bf16", "avx512vnni"]})
        for bad in ("model", "model=", "frequency=3", "model=15-0", "model=-", "model=a", "model=1|x", "isa=amx",
                    "vendor=A|"):
            with self.assertRaises(ValueError):
                E.parse_hw_set([bad])


@unittest.skipUnless(HAVE_NUMPY, "numpy is not installed")
class TestRules(unittest.TestCase):
    def test_tree_and_ordered(self):
        from native_brgemm_tuning.rules import eval_rules, train_rules

        probs = [Problem("bf16", m, n, 4096) for n in (2048, 4096, 6144, 8192) for m in (1, 8, 16, 256, 512)]
        fx = Fixture(probs, texts=("nr=16", "block_rows=96"))
        summ = R.summarize(fx.records)
        feats = [f for f in SCHEMA.features]

        def fn(p):
            return rule_features(p, UARCH, SCHEMA.hw)

        def config_of(t):
            return E.config_json(C.parse_settings(t, SCHEMA), SCHEMA) if t else {}

        out = {}
        for kind in ("tree", "ordered"):
            rules, rep = train_rules(probs, summ, feats, fn, config_of, kind=kind, min_leaf=2, folds=4)
            doc = {"format": E.FORMAT, "version": 2, "rules": rules}
            self.assertEqual(E.validate_doc(doc, SCHEMA), ([], []))
            out[kind] = [eval_rules(rules, fn(p)) or {} for p in probs]
            self.assertEqual(rep["problems"], 16)
            self.assertEqual(rep["heldout"]["slower_than_builtin"], 0)
            self.assertAlmostEqual(rep["heldout"]["geomean_ratio"], rep["heldout"]["oracle_geomean_ratio"])
        self.assertEqual(out["tree"], out["ordered"])
        # An M = 1 call runs on the GEMV kernels: the rules learn nothing from it and set nothing there.
        for p, cfg in zip(probs, out["tree"]):
            self.assertEqual(cfg, {} if p.m == 1 else {"nr": 16} if p.m <= 16 else {"block_rows": 96}, p.label())

    def test_rules_set_nothing_at_m_1(self):
        from native_brgemm_tuning.rules import eval_rules, train_rules

        # nr=16 wins at every M, M = 1 included.
        probs = [Problem("bf16", m, n, 4096) for n in (2048, 4096, 6144) for m in (1, 2, 8)]
        fx = Fixture(probs, texts=("nr=16",), ratio=lambda p, c: 0.8)

        def fn(p):
            return rule_features(p, UARCH, SCHEMA.hw)

        for kind in ("tree", "ordered"):
            rules, _ = train_rules(probs, R.summarize(fx.records), SCHEMA.features, fn,
                                   lambda t: E.config_json(C.parse_settings(t, SCHEMA), SCHEMA) if t else {},
                                   kind=kind, min_leaf=2, folds=3)
            for p in probs:
                self.assertEqual(eval_rules(rules, fn(p)) or {}, {} if p.m == 1 else {"nr": 16}, (kind, p.label()))
            if kind == "tree":
                self.assertEqual((rules["root"]["feature"], rules["root"]["le"], rules["root"]["then"]),
                                 ("m", 1, {"config": {}}))
            else:
                self.assertTrue(rules["list"])
                self.assertTrue(all(r["when"][0] == ["m", ">=", 2] for r in rules["list"]))
            doc = {"format": E.FORMAT, "version": 2, "rules": rules}
            self.assertEqual(E.validate_doc(doc, SCHEMA), ([], []))
            for m in (1, 2):
                cfg, layers = E.resolve_config(doc, probs[0].with_m(m), SCHEMA, UARCH)
                self.assertEqual((cfg, layers), ({}, []) if m == 1 else ({"nr": 16}, ["rules"]), (kind, m))

    def test_nothing_to_learn(self):
        from native_brgemm_tuning.rules import train_rules

        fx = Fixture(texts=("threads_cap=16",), ratio=lambda p, c: 1.0)
        rules, rep = train_rules(PROBLEMS, R.summarize(fx.records), SCHEMA.features,
                                 lambda p: rule_features(p, UARCH, SCHEMA.hw), lambda t: {})
        self.assertIsNone(rules)
        self.assertIn("skipped", rep)


if __name__ == "__main__":
    unittest.main()
