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
"""The tuner against the planner's schema: every key, match key, name, rule feature and
comparison, kernel kind, plan-line field and hardware field the planner's schema dump
gives is one the tuner reads, mirrors and writes.

The planner's own test, NativeConfigSchema.TheTunerFollowsThePlanner, compares its dump
with fixtures/schema_dump.json; these tests compare the fixture with the tuner's code.
ZENDNNL_NATIVE_SCHEMA names another dump (a JSON file or the log of the SchemaDump gtest,
for example of the build under test), which must then agree with the fixture and pass
the same checks.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import SCHEMA  # noqa: E402

from native_brgemm_tuning import candidates as C  # noqa: E402
from native_brgemm_tuning import export as E  # noqa: E402
from native_brgemm_tuning import plan as PL  # noqa: E402
from native_brgemm_tuning import problems as P  # noqa: E402
from native_brgemm_tuning import rules as R  # noqa: E402
from native_brgemm_tuning import schema as S  # noqa: E402

# Problems of every weight type the planner serves, at a decode and a prefill M.
PROBLEMS = [P.Problem(d, m, 4096, 4096, group=128 if d == "s4" else 0)
            for d in P.DTYPES for m in (8, 2048)]
# The schema members that do not depend on the machine of the dump.
SHARED = ("format", "version", "keys", "match_keys", "features", "rule_ops", "tuning_file", "plan_line",
          "plan_line_gemv")


def schemas():
    """The fixture and, when ZENDNNL_NATIVE_SCHEMA names one, another dump."""
    out = [SCHEMA]
    path = os.environ.get("ZENDNNL_NATIVE_SCHEMA")
    if path:
        out.append(S.load_schema(path))
    return out


class TestKeys(unittest.TestCase):
    def test_every_key_is_mirrored_at_its_builtin_value(self):
        for s in schemas():
            self.assertEqual([k.name for k in s.keys], list(PL.BUILTIN_PARAMS), s.source)
            for k in s.keys:
                self.assertIn(k.type, S.KEY_TYPES, (s.source, k.name))
                builtin = k.default if k.default is not None else k.auto
                self.assertEqual(PL.BUILTIN_PARAMS[k.name], builtin, (s.source, k.name))
                if k.default is None:
                    self.assertIn(k.name, PL.AUTO_KEYS, (s.source, k.name))

    def test_every_key_can_be_searched_where_it_applies(self):
        for s in schemas():
            for k in s.keys:
                served = [p for p in PROBLEMS if k.applies_to(p.dtype)]
                self.assertTrue(served, (s.source, k.name))
                values = [v for p in served for v in C.value_choices(k.name, p, s)]
                self.assertTrue(values, "%s: no search values for %s" % (s.source, k.name))
                for v in values:
                    self.assertIsNone(k.check(v), (s.source, k.name, v))

    def test_scheme_masks(self):
        for s in schemas():
            for k in s.keys:
                self.assertTrue(set(k.int8_schemes) <= set(S.INT8_SCHEMES), (s.source, k.name))
                if k.applies == "bf16":
                    self.assertEqual(k.int8_schemes, [], (s.source, k.name))


class TestNamesAndFeatures(unittest.TestCase):
    def test_match_keys_and_their_names(self):
        for s in schemas():
            mk = s.match_keys
            self.assertEqual(list(mk), list(E.MATCH_NUMERIC) + list(E.MATCH_ENUM) + list(E.MATCH_BOOL), s.source)
            self.assertEqual([k for k, v in mk.items() if v == "range"], list(E.MATCH_NUMERIC), s.source)
            self.assertEqual([k for k, v in mk.items() if v == "bool"], list(E.MATCH_BOOL), s.source)
            # The rule feature epilogue is the index of the problem's name in this order.
            self.assertEqual((mk["dtype"], mk["src"], mk["dst"], mk["epilogue"]),
                             (list(P.DTYPES), list(P.SRC_TYPES), list(P.DST_TYPES), list(P.EPILOGUES)), s.source)
            self.assertEqual(list(S.WEIGHT_DTYPES), list(P.DTYPES))
            for p in PROBLEMS:
                for k in E.MATCH_NUMERIC + E.MATCH_ENUM + E.MATCH_BOOL:
                    self.assertTrue(hasattr(p, k), k)

    def test_rule_features_and_comparisons(self):
        for s in schemas():
            for p in PROBLEMS:
                self.assertEqual(list(P.rule_features(p, s.uarch, s.hw)), list(s.features), (s.source, p.dtype))
            self.assertEqual(list(s.doc.get("rule_ops", [])), list(R.OPS), s.source)

    def test_tuning_file_names(self):
        for s in schemas():
            tf = s.doc.get("tuning_file", {})
            self.assertEqual((tf.get("format"), tf.get("version")), (E.FORMAT, E.VERSION), s.source)
            self.assertEqual(tf.get("hardware_keys"), list(S.HW_KEYS), s.source)
            self.assertEqual(tf.get("isa"), list(S.ISA_ORDER), s.source)

    def test_plan_line_fields(self):
        for s in schemas():
            line = s.doc.get("plan_line", [])
            self.assertEqual(sorted(line), sorted(PL.LOG_LINE_FIELDS), s.source)
            self.assertTrue(set(PL.LOG_PLAN_FIELDS) <= set(line), s.source)

    def test_plan_line_gemv_values(self):
        # The GEMV kernels the line names, and the ones the model gives the problems of
        # every weight type and INT8 scheme at M = 1 (and F16_PH plans of f16 calls).
        for s in schemas():
            self.assertEqual(s.doc.get("plan_line_gemv"), list(PL.GEMV_KINDS), s.source)
            given = set()
            for p in PROBLEMS + [P.Problem("s8", 8, 4096, 4096, int8_scheme="per_group")]:
                for kind in ("f16", "f16_ph") if p.f16 else (s.kind_of(p.dtype),):
                    q = PL.Plan(kind=kind)
                    self.assertEqual(PL.gemv_kind(q, p), "none", (s.source, p.label()))
                    given.add(PL.gemv_kind(q, p.with_m(1)))
            self.assertEqual(given, set(PL.GEMV_KINDS) - {"none"}, s.source)


class TestKernelsAndMachine(unittest.TestCase):
    def test_kernel_kinds(self):
        for s in schemas():
            for c in s.kernels:
                self.assertTrue(c.known, (s.source, c.kind))
                self.assertEqual(c.k_unit, S.KernelCaps.K_UNITS[c.kind], (s.source, c.kind, c.nr))
            # Every weight type plans for kernels of a kind the dump lists.
            for d in P.DTYPES:
                self.assertTrue(s.nr_widths(s.kind_of(d)), (s.source, d))

    def test_uarch_and_hardware_fields(self):
        for s in schemas():
            self.assertTrue(set(S.Uarch.FIELDS) <= set(s.doc.get("uarch", {})), s.source)
            self.assertEqual(set(s.doc.get("hw", {})), set(SCHEMA.doc["hw"]), s.source)
            self.assertTrue(set(s.hw_match) <= set(S.HW_KEYS), s.source)

    def test_another_dump_agrees_with_the_fixture(self):
        if len(schemas()) < 2:
            self.skipTest("ZENDNNL_NATIVE_SCHEMA is not set")
        other = schemas()[1]
        for member in SHARED:
            self.assertEqual(other.doc.get(member), SCHEMA.doc.get(member), member)
        mine = {(c.kind, c.nr, c.schemes): (c.mr_mask, c.max_mr, c.k_unit) for c in SCHEMA.kernels}
        kinds = {c.kind for c in other.kernels}
        for c in other.kernels:
            self.assertEqual(mine.get((c.kind, c.nr, c.schemes)), (c.mr_mask, c.max_mr, c.k_unit), (c.kind, c.nr))
        # A kind the other machine has none of may be the fixture machine's.
        theirs = {(c.kind, c.nr, c.schemes) for c in other.kernels}
        for key in mine:
            self.assertTrue(key in theirs or key[0] not in kinds, key)


if __name__ == "__main__":
    unittest.main()
