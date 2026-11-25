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
"""Schema loading: the fixture, the SchemaDump markers, normalization, keys, registry, hardware."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import BF16_RULE_KEYS, BLOCKS_RULE_KEYS, DESIGN, NO_BLOCKS, OLDER, RULES_OFF, SCHEMA  # noqa: E402

from native_brgemm_tuning import plan as PL  # noqa: E402
from native_brgemm_tuning import schema as S  # noqa: E402
from native_brgemm_tuning.candidates import parse_settings  # noqa: E402
from native_brgemm_tuning.plan import AUTO_KEYS, BUILTIN_PARAMS, RULE_SWITCHES, builtin_params  # noqa: E402
from native_brgemm_tuning.problems import Problem  # noqa: E402

DESIGN_KEYS = ["nr", "mr", "mb", "kc", "col_groups", "schedule", "traversal", "threads_cap",
               "min_tiles_per_thread", "row_unit", "tile_rows", "tall_rows", "block_rows", "min_tile_cols",
               "row_split", "min_pass_k", "bf16_l2_pct", "kc_split", "l3_pct", "domain_threads", "int8_mr",
               "int8_cw", "int8_l2_pct", "int8_kc", "direct_store", "bias_in_dequant",
               "copy_block", "bf16_copy_par_min", "int8_copy_par_min"]
# The keys of the rules, after the design's keys.
RULE_KEYS = ["makespan", "min_gain_pct", "idle_pct", "decode_m", "l1_pass", "l1_pct", "l1_pass_max_mb",
             "traffic_groups"] + list(BF16_RULE_KEYS) + list(BLOCKS_RULE_KEYS)
# The cost model's switch, margin and inputs, after the rules' keys.
COST_KEYS = ["cost_model", "cm_gain_pct", "cm_short_rows", "cm_fma_pipes", "cm_fma_lat", "cm_call_cycles",
             "cm_tall_pct", "cm_l2_bytes", "cm_l3_cycles", "cm_mem_cycles", "cm_prefetch_kib", "cm_overlap_pct",
             "cm_reuse_pct", "cm_stream_cycles", "cm_sync_tiles", "cm_group_slab_pct", "cm_group_l3_pct",
             "cm_group_skew_pct", "cm_group_rows_pct", "cm_drift_kib", "cm_pass_kib", "cm_pass_calls",
             "cm_block_kib", "cm_block_k", "cm_int8_l1_pct", "cm_int4_min_m"]
# The keys of an M = 1 call, after the cost model's.
M1_KEYS = ["gemv", "gemv_blocks", "gemv_one_thread_pct"]
# The keys of the planner's schema, in its order: the design's, nb after mb, unit_tiles after
# traversal, int4_l1_pct after int8_kc, f16_wide_min_m after the narrow-N rule, then the cost model's
# and those of an M = 1 call.
SCHEMA_KEYS = (DESIGN_KEYS[:3] + ["nb"] + DESIGN_KEYS[3:7] + ["unit_tiles"] + DESIGN_KEYS[7:24] + ["int4_l1_pct"]
               + DESIGN_KEYS[24:] + RULE_KEYS[:12] + ["f16_wide_min_m"] + RULE_KEYS[12:] + COST_KEYS + M1_KEYS)
DESIGN_FIXTURE = os.path.join(os.path.dirname(S.FIXTURE), "schema_design.json")


class TestFixture(unittest.TestCase):
    def test_fixture_layout(self):
        # refresh_fixtures.sh writes the fixture through tune.py schema --extract.
        with open(S.FIXTURE, encoding="utf-8") as f:
            text = f.read()
        self.assertEqual(S.schema_text(json.loads(text)), text)

    def test_keys_in_design_order(self):
        # The blocks schedule's pins: nb, its unit in columns, follows mb; unit_tiles, its unit in
        # tiles, follows traversal.
        self.assertEqual([k.name for k in SCHEMA.keys], SCHEMA_KEYS)
        self.assertEqual([k.name for k in DESIGN.keys], DESIGN_KEYS)
        self.assertEqual(SCHEMA.key("schedule").values, ["auto", "slice", "row_groups", "blocks"])

    def test_defaults_match_the_heuristic_parameters(self):
        for s in (SCHEMA, DESIGN, OLDER, NO_BLOCKS):
            hp = builtin_params(s)
            for k in s.keys:
                if k.default is None:
                    self.assertIn(k.name, AUTO_KEYS)
                    self.assertEqual(hp[k.name], k.auto, k.name)
                else:
                    self.assertIsNone(k.auto, k.name)
                    self.assertEqual(hp[k.name], k.default, k.name)
            # A rule switch the schema lacks is off: its build has no such rule.
            self.assertEqual([hp[k] for k in RULE_SWITCHES if k not in s.key_map],
                             [0] * sum(k not in s.key_map for k in RULE_SWITCHES), s.source)
        # The mirror reads every key of the schema, in its order; each rule switch defaults to 1 (on).
        self.assertEqual(builtin_params(SCHEMA), BUILTIN_PARAMS)
        self.assertEqual(list(BUILTIN_PARAMS), [k.name for k in SCHEMA.keys])
        self.assertTrue(all(PL.mirrored(k.name) for k in SCHEMA.keys))
        self.assertEqual([(SCHEMA.key(k).default, SCHEMA.key(k).min, SCHEMA.key(k).max) for k in RULE_SWITCHES],
                         [(1, 0, 1)] * len(RULE_SWITCHES))
        # row_unit's auto value is the rules' unit of 12 rows, a legal setting: setting it pins nothing.
        self.assertEqual((SCHEMA.key("row_unit").default, SCHEMA.key("row_unit").auto), (None, 12))
        self.assertEqual([builtin_params(SCHEMA)[k] for k in ("nb", "unit_tiles") + BLOCKS_RULE_KEYS],
                         [0, 0, 1, 1024, 3072])
        self.assertEqual((SCHEMA.key("nb").min, SCHEMA.key("nb").max, SCHEMA.key("nb").check(24) is None),
                         (16, 1 << 20, False))

    def test_bits_columns(self):
        changing = [k.name for k in SCHEMA.keys if not k.same_bits("s8")]
        self.assertEqual(changing, ["bias_in_dequant"])
        self.assertTrue(SCHEMA.key("kc").applies_to("bf16"))
        self.assertFalse(SCHEMA.key("kc").applies_to("s8"))
        self.assertFalse(SCHEMA.key("int8_cw").applies_to("bf16"))
        bf16_only = list(BF16_RULE_KEYS) + ["bf16_blocks_min_m"]
        both = [k for k in RULE_KEYS if k not in bf16_only + ["int8_blocks_min_m"]]
        for name in both + ["min_pass_k", "nb", "unit_tiles"]:
            self.assertTrue(SCHEMA.key(name).applies_to("s8") and SCHEMA.key(name).applies_to("bf16"), name)
        for name in bf16_only:
            self.assertTrue(SCHEMA.key(name).applies_to("bf16") and not SCHEMA.key(name).applies_to("s8"), name)
        self.assertTrue(SCHEMA.key("int8_blocks_min_m").applies_to("s8"))
        self.assertFalse(SCHEMA.key("int8_blocks_min_m").applies_to("bf16"))
        self.assertFalse(DESIGN.key("min_pass_k").applies_to("s8"))
        # The families a key applies to, which a key of a dump without them takes from its
        # applies; a problem's families are those of its format's routes.
        self.assertEqual(SCHEMA.key("kc").families, ("bf16", "f32", "f16", "f16_ph"))
        self.assertEqual(SCHEMA.key("int4_l1_pct").families, ("int8", "int4"))
        self.assertEqual(DESIGN.key("kc").families, SCHEMA.key("kc").families)
        older = S.Schema(dict(SCHEMA.doc, keys=[{f: v for f, v in d.items() if f != "families"}
                                                for d in SCHEMA.doc["keys"]]))
        self.assertEqual([k.families for k in older.keys], [k.families for k in SCHEMA.keys])
        self.assertTrue(SCHEMA.key("kc").applies_to("f16_ph") and SCHEMA.key("kc").applies_to("f16"))
        self.assertFalse(SCHEMA.key("kc").applies_to("int4") or SCHEMA.key("kc").applies_to("s4"))
        self.assertEqual((S.WEIGHT_DTYPES, S.INT8_DTYPES, S.FLOAT_DTYPES),
                         (("bf16", "s8", "f32", "s4", "f16"), ("s8", "s4"), ("bf16", "f32", "f16")))

    def test_older_schema(self):
        self.assertEqual(S.schema_diff(DESIGN, SCHEMA),
                         ["key %s only in %s" % (k, SCHEMA.source) for k in SCHEMA_KEYS if k not in DESIGN_KEYS]
                         + ["key nr: values [16, 32] vs [16, 32, 64, 128]",
                            "key schedule: values ['auto', 'slice', 'row_groups'] vs "
                            + "['auto', 'slice', 'row_groups', 'blocks']",
                            "key row_unit: default 12 vs None", "key row_unit: auto None vs 12",
                            "key min_pass_k: applies 'bf16' vs 'both'",
                            "kernel bf16 nr 64: None vs (4095, 12, 2)", "kernel bf16 nr 128: None vs (4095, 12, 2)"]
                         + ["kernel f16 nr %d: None vs (4095, 12, 1)" % nr for nr in (16, 32, 64, 128)]
                         + ["kernel f16_ph nr %d: None vs (4095, 12, 1)" % nr for nr in SCHEMA.nr_widths("f16_ph")]
                         + ["kernel f32 nr %d: None vs (4095, 12, 1)" % nr for nr in (16, 32, 64, 128)]
                         # Tiles of 64 and 128 columns have the rows of their 32-column slab.
                         + [line for nr in (16, 32, 64, 128) for line in (
                             "kernel int4 nr %d (per_channel): None vs (4095, 12, 8)" % nr,
                             "kernel int4 nr %d (per_group): None vs %s" % (
                                 nr, (4095, 12, 8) if nr == 16 else (63, 6, 8)))]
                         + ["kernel int8 nr 16 (per_channel): None vs (4095, 12, 4)",
                            "kernel int8 nr 16 (per_channel/per_group): (4095, 12, 4) vs None",
                            "kernel int8 nr 16 (per_group): None vs (4095, 12, 4)",
                            "kernel int8 nr 32 (per_channel): None vs (4095, 12, 4)",
                            "kernel int8 nr 32 (per_channel/per_group): (4095, 12, 4) vs None",
                            "kernel int8 nr 32 (per_group): None vs (63, 6, 4)"]
                         + [line for nr in (64, 128) for line in (
                             "kernel int8 nr %d (per_channel): None vs (4095, 12, 4)" % nr,
                             "kernel int8 nr %d (per_group): None vs (63, 6, 4)" % nr)])
        self.assertEqual(S.schema_diff(NO_BLOCKS, SCHEMA)[-1],
                         "key schedule: values ['auto', 'slice', 'row_groups'] vs "
                         + "['auto', 'slice', 'row_groups', 'blocks']")
        with self.assertRaises(ValueError):
            parse_settings("makespan=1", DESIGN)
        self.assertEqual(parse_settings("l1_pass=0,l1_pct=50", SCHEMA), {"l1_pass": 0, "l1_pct": 50})
        # A build without the rule keys plans without the rules.
        for p in (Problem("bf16", 32, 4096, 4096), Problem("s8", 32, 4096, 4096), Problem("bf16", 128, 1536, 4096)):
            off = PL.resolve(p, parse_settings(RULES_OFF, SCHEMA), DESIGN.uarch, SCHEMA).plan
            self.assertNotEqual(off, PL.builtin_plan(p, DESIGN.uarch, SCHEMA), p)
            self.assertEqual(PL.builtin_plan(p, DESIGN.uarch, DESIGN), off, p)

    def test_registry(self):
        self.assertTrue(SCHEMA.covers("bf16", 6, 16))
        self.assertFalse(SCHEMA.covers("bf16", 7, 16))
        self.assertTrue(SCHEMA.covers("bf16", 12, 32))
        self.assertTrue(SCHEMA.covers("bf16", 12, 128))  # tiles of 32-column slabs
        self.assertIsNone(SCHEMA.caps("bf16", 256))
        # Every type's tiles of 64 and 128 columns, with the rows and schemes of the slab.
        self.assertTrue(SCHEMA.covers("int8", 12, 64))
        self.assertTrue(SCHEMA.covers("int8", 6, 128, "per_group"))
        self.assertFalse(SCHEMA.covers("int8", 7, 128, "per_group"))
        self.assertIsNone(SCHEMA.caps("int8", 256))
        s8, s8_group = SCHEMA.route_of("s8"), SCHEMA.route_of("s8", "per_group")
        self.assertEqual((SCHEMA.route_max_mr(s8, 3), SCHEMA.route_max_mr(s8, 4)), (0, 12))
        self.assertEqual((SCHEMA.route_max_mr(s8, 2), SCHEMA.route_max_mr(s8_group, 2)), (12, 6))
        # A float plan carries the rows of the per-channel INT8 kernels.
        self.assertEqual(SCHEMA.route_max_mr(SCHEMA.route_of("bf16"), 2), 12)
        for kind in ("bf16", "int8", "int4", "f32", "f16", "f16_ph"):
            self.assertEqual(SCHEMA.nr_widths(kind), [16, 32, 64, 128], kind)
        self.assertEqual(S.kernel_kind("f32"), "f32")
        # s4 weights run the INT4 group kernels (rows 1 .. 12 at 16 columns, 1 .. 6 at 32), a weight of one
        # group over K the kernels of one group (1 .. 12 at both); f16 weights the F16_PH kernels where the
        # registry has them (also two blocks wide), else the F16 kernels.
        s4, s4_one = SCHEMA.route_of("s4", "per_group"), SCHEMA.route_of("s4", "per_group", one_group=True)
        self.assertEqual((S.kernel_kind("s4"), SCHEMA.route_max_mr(s4, 1), SCHEMA.route_max_mr(s4, 2)), ("int4", 12, 6))
        self.assertEqual((SCHEMA.route_max_mr(s4_one, 1), SCHEMA.route_max_mr(s4_one, 2)), (12, 12))
        self.assertEqual(SCHEMA.caps("int4", 128, "per_channel").max_mr, 12)
        self.assertEqual(SCHEMA.caps("int4", 32, "per_group").schemes, ("per_group",))
        self.assertEqual(SCHEMA.caps("int4", 32, "per_channel").schemes, ("per_channel",))
        self.assertTrue(SCHEMA.covers("int4", 12, 32, "per_channel"))
        self.assertFalse(SCHEMA.covers("int4", 7, 32, "per_group"))
        self.assertEqual((s4_one.schemes, s4.schemes), (("per_channel",), ("per_group",)))
        self.assertEqual((s4_one.cls, s4.cls, SCHEMA.route_of("s8").cls), ("group", "group", "channel"))
        # A build without the kernels of one group runs every s4 weight on the group kernels.
        group_only = S.Schema(dict(SCHEMA.doc, kernels=[
            k for k in SCHEMA.doc["kernels"] if not (k["kind"] == "int4" and k["int8_schemes"] == ["per_channel"])]))
        self.assertEqual((group_only.route_of("s4", "per_group", one_group=True).schemes,
                          group_only.caps("int4", 32, "per_channel")), (("per_group",), None))
        # Without a scheme, caps and covers take what the library's caps(kind, nr) takes: the INT4
        # group kernels and the per-channel INT8 kernels, with or without the kernels of one group.
        for s in (SCHEMA, group_only):
            self.assertEqual(s.caps("int4", 32), s.caps("int4", 32, "per_group"))
            self.assertEqual(s.caps("int4", 32).schemes, ("per_group",))
            self.assertEqual((s.covers("int4", 6, 32), s.covers("int4", 7, 32)), (True, False))
            self.assertEqual(s.caps("int8", 32), s.caps("int8", 32, "per_channel"))
        # A family that lists no scheme serves every scheme, for the routes as for caps.
        unlisted = S.Schema(dict(SCHEMA.doc, kernels=[
            dict(k, int8_schemes=[]) if k["kind"] == "int4" else k for k in group_only.doc["kernels"]]))
        self.assertEqual(unlisted.route_of("s4", "per_group", one_group=True).schemes, ("per_channel",))
        self.assertEqual(unlisted.caps("int4", 32, "per_channel"), unlisted.caps("int4", 32))
        self.assertEqual(SCHEMA.nr_widths("f16"), [16, 32, 64, 128])
        kinds = {c.kind for c in SCHEMA.kernels}
        self.assertEqual(SCHEMA.kind_of("f16"), "f16_ph" if "f16_ph" in kinds else "f16")
        if "f16_ph" in kinds:
            self.assertEqual(SCHEMA.nr_widths("f16_ph"), [16, 32, 64, 128])
        self.assertEqual(set(kinds) - set(S.KERNEL_KINDS), set())
        # The INT8 kernels of a scheme at a width take 1 .. max_mr rows.
        self.assertTrue(SCHEMA.covers("int8", 12, 32, "per_channel"))
        self.assertTrue(SCHEMA.covers("int8", 6, 32, "per_group"))
        self.assertFalse(SCHEMA.covers("int8", 0, 16, "per_channel"))
        self.assertTrue(SCHEMA.covers("int8", 12, 64, "per_channel"))
        self.assertFalse(SCHEMA.covers("int8", 1, 48, "per_channel"))

    def test_hardware_and_uarch(self):
        # The fixture's example machine: L3 domains of 8 cores and 32 MiB, AVX512-FP16.
        self.assertEqual((SCHEMA.hw.l1d_kib, SCHEMA.hw.l2_kib, SCHEMA.hw.l3_kib), (48, 1024, 32768))
        self.assertEqual(SCHEMA.hw.isa, ["avx512f", "avx512bf16", "avx512vnni", "avx512fp16"])
        self.assertTrue(SCHEMA.uarch.complete)
        # Its compute model, the class default of a CPU the library's table does not hold, and
        # each family's path.
        self.assertEqual(SCHEMA.uarch.dot_source, "default")
        self.assertEqual(SCHEMA.uarch.dot["bf16"], S.Dot(2.0, 6))
        self.assertEqual(SCHEMA.uarch.dot["int8"].chains, 12)
        self.assertEqual({n: f.path for n, f in SCHEMA.families.items()},
                         {"bf16": "bf16", "int8": "int8", "f32": "fp32", "int4": "int4", "f16": "f16_widen",
                          "f16_ph": "fp16"})
        self.assertEqual((SCHEMA.uarch.l3_bytes, SCHEMA.uarch.family_l3_cores, SCHEMA.uarch.l3_domain_cores),
                         (32 << 20, 8, 8))
        self.assertEqual((SCHEMA.hw.vendor, SCHEMA.hw.family, SCHEMA.hw.model, SCHEMA.hw.stepping),
                         ("ExampleVendor", 1, 1, 0))
        self.assertTrue(SCHEMA.doc["uarch"]["avx512fp16"])

    def test_machine_model(self):
        # A dump of a build without the model: its fma_ports on every path, the class default's
        # latencies; with it, its paths (the class default for a path it lacks).
        old = S.Uarch.from_dict({"l1d_bytes": 49152, "fma_ports": 1})
        self.assertEqual((old.dot_source, old.dot["int8"], old.dot["bf16"]),
                         ("fma_ports", S.Dot(1.0, 6), S.Dot(1.0, 6)))
        new = S.Uarch.from_dict({"fma_ports": 2, "dot": {"int8": {"ipc": 3.8, "latency": 4, "chains": 16}},
                                 "dot_source": "default"})
        self.assertEqual((new.dot["int8"], new.dot["int8"].chains, new.dot["bf16"]),
                         (S.Dot(3.8, 4), 16, S.Dot(2.0, 6)))
        # --uarch takes the paths as ZENDNNL_MATMUL_NATIVE_UARCH does; fma_ports sets every path.
        u = new.with_overrides("int8_ipc=4, int8_lat=5,\tbf16_ipc=1")
        self.assertEqual((u.dot["int8"], u.dot["bf16"].ipc, u.fma_ports, u.dot_source), (S.Dot(4.0, 5), 1.0, 1, "env"))
        self.assertEqual(new.dot["int8"], S.Dot(3.8, 4))
        self.assertTrue(all(d.ipc == 1.0 for d in new.with_overrides("fma_ports=1").dot.values()))
        # What the library refuses: a number that is not finite or out of its range, an empty item,
        # a space inside an item.
        for bad in ("int8_ipc=0", "int9_ipc=2", "int8_lat=2.5", "int8_lat=65", "int8_ipc=x", "dot=1",
                    "bf16_lat=inf", "int8_lat=1e10", "int8_lat=nan", "fp32_ipc=1e400", "mem_cycles=inf",
                    "mem_cycles=1e400", "stream_cycles=-1e300", "int8_ipc=0x4", "int8_ipc=4,", "int8_ipc=4,,bf16_ipc=1",
                    "int8_ipc =4", "int8_ipc= 4", "family_l3_cores= 8"):
            with self.assertRaises(S.SchemaError, msg=bad):
                new.with_overrides(bad)
        self.assertEqual(S.Uarch.from_dict(new.to_dict()).dot, new.dot)
        # The path of a problem's kernels and its model.
        self.assertEqual(SCHEMA.path_of(Problem("s4", 8, 4096, 4096, group=128, scale_bytes=2)), "int4")
        self.assertEqual(SCHEMA.path_of(Problem("bf16", 8, 4096, 4096)), "bf16")
        self.assertEqual(SCHEMA.dot_of(Problem("bf16", 8, 4096, 4096)), S.Dot(2.0, 6))

    def test_with_machine(self):
        with open(S.FIXTURE) as f:
            fixture = json.load(f)
        dump = json.loads(json.dumps(fixture))
        dump["hw"].update(vendor="OtherVendor", model=9, l3_kib=65536)
        dump["hw_match"].update(vendor="OtherVendor", model=9, l3_kib=65536)
        dump["uarch"]["l3_bytes"] = 64 << 20
        kept = S.with_machine(dump, fixture)
        self.assertEqual(kept, fixture)
        self.assertEqual(S.schema_text(kept), S.schema_text(fixture))
        # The dump's fields and order; a field the example machine lacks needs a value first.
        dump["uarch"]["new_field"] = 1
        with self.assertRaises(S.SchemaError):
            S.with_machine(dump, fixture)

    def test_hardware_fields(self):
        with open(S.FIXTURE) as f:
            self.assertEqual(SCHEMA.hw_match, json.load(f)["hw_match"])
        # The design fixture has no hw_match: the fields come from its hardware and uarch blocks.
        self.assertEqual(DESIGN.hw_match, {
            "vendor": "AuthenticAMD", "family": 26, "model": 2, "stepping": 1,
            "isa": ["avx512f", "avx512bf16", "avx512vnni"], "l1d_kib": 48, "l2_kib": 1024, "l3_kib": 32768,
            "l3_cores": 8, "smt": 2, "l3_threads": 16, "cores": 128, "logical_cpus": 256})
        self.assertEqual(list(DESIGN.hw_match), list(S.HW_KEYS))
        with open(DESIGN_FIXTURE) as f:
            doc = json.load(f)
        # A dump's hw_match is taken as it is.
        dumped = {"vendor": "ExampleVendor", "family": 1, "model": 2, "isa": ["avx512f"], "l3_cores": 16}
        self.assertEqual(S.Schema(dict(doc, hw_match=dumped)).hw_match, dumped)
        # A dump without the identity: a zero count left out, the plan's domain from the topology.
        hw = {k: v for k, v in doc["hw"].items() if k not in ("vendor", "family", "model", "stepping")}
        old = S.Schema(dict(doc, hw=dict(hw, smt=0, l3_domain_threads=32), uarch={"family_l3_cores": 8}))
        self.assertNotIn("vendor", old.hw_match)
        self.assertNotIn("family", old.hw_match)
        self.assertNotIn("smt", old.hw_match)
        self.assertEqual(old.hw_match["l3_cores"], 32)
        self.assertEqual(old.hw_match["l3_threads"], 32)
        # An unknown identity is left out; a zero stepping is kept.
        unknown = S.Schema(dict(doc, hw=dict(doc["hw"], family=-1, model=-1, stepping=0)))
        self.assertNotIn("family", unknown.hw_match)
        self.assertEqual(unknown.hw_match["stepping"], 0)
        self.assertIn("hardware fields:", SCHEMA.summary())


class TestKeys(unittest.TestCase):
    def test_parse_format(self):
        self.assertTrue(SCHEMA.key("row_split").parse("On"))
        self.assertFalse(SCHEMA.key("row_split").parse("false"))
        self.assertEqual(SCHEMA.key("row_split").format(False), "0")
        self.assertEqual(SCHEMA.key("int8_cw").parse("16/32"), [32, 16])
        self.assertEqual(SCHEMA.key("int8_cw").format([16, 32]), "32/16")
        self.assertEqual(SCHEMA.key("schedule").parse(" Row_Groups "), "row_groups")
        for key, text in (("nr", "16x"), ("row_split", "2"), ("int8_cw", "a/16")):
            with self.assertRaises(ValueError):
                SCHEMA.key(key).parse(text)

    def test_static_ranges(self):
        bad = {"nr": 256, "mr": 13, "min_pass_k": 255, "int8_kc": 6, "int8_cw": [32, 64], "min_tile_cols": 64,
               "schedule": "rows", "row_split": 1}
        for key, v in bad.items():
            self.assertIsNotNone(SCHEMA.key(key).check(v), key)
        good = {"nr": 128, "mr": 12, "min_pass_k": 256, "int8_kc": 8, "int8_cw": [16], "min_tile_cols": 32,
                "schedule": "slice", "row_split": False, "bf16_copy_par_min": 1 << 40}
        for key, v in good.items():
            self.assertIsNone(SCHEMA.key(key).check(v), key)

    def test_json_types(self):
        self.assertEqual(SCHEMA.key("int8_cw").to_json([16, 32, 16]), [32, 16])
        self.assertIs(SCHEMA.key("direct_store").to_json(0), False)
        with self.assertRaises(ValueError):
            SCHEMA.key("nr").from_json(True)
        with self.assertRaises(ValueError):
            SCHEMA.key("direct_store").from_json(1)


class TestLoading(unittest.TestCase):
    def _write(self, text):
        f = tempfile.NamedTemporaryFile("w", suffix=".log", delete=False)
        f.write(text)
        f.close()
        self.addCleanup(os.unlink, f.name)
        return f.name

    def test_gtest_log_with_markers(self):
        with open(S.FIXTURE) as f:
            body = f.read()
        log = "[ RUN      ] NativeConfigSchema.SchemaDump\n%s\n%s\n%s\n[       OK ]\n" % (
            S.BEGIN_MARK, body, S.END_MARK)
        s = S.load_schema(self._write(log))
        self.assertEqual(len(s.keys), len(SCHEMA.keys))
        with self.assertRaises(S.SchemaError):
            S.load_schema(self._write("%s\n{}\n" % S.BEGIN_MARK))

    def test_normalized_spellings(self):
        doc = {
            "keys": {"nr": {"type": "int", "values": [16, 32], "default": "auto", "dtype": "both"},
                     "bias_in_dequant": {"type": "bool", "default": "1", "dtype": "s8", "bits_same": False}},
            "registry": [{"kind": "s8", "nr": 32, "mr_mask": [1, 2, 3, 4, 5, 6]}],
            "hardware": {"l1d_kib": 48, "l2_kib": 1024, "l3_kib": 32768, "smt": 2, "l3_domain_threads": 16},
        }
        s = S.Schema(doc)
        self.assertEqual(s.key("bias_in_dequant").applies, "int8")
        self.assertFalse(s.key("bias_in_dequant").same_bits("s8"))
        self.assertIs(s.key("bias_in_dequant").default, True)
        self.assertEqual(s.caps("int8", 32).max_mr, 6)
        self.assertIsNone(s.uarch.family_l3_cores)  # not in the hardware block
        self.assertEqual(s.uarch.plan_domain, 8)  # l3_domain_threads 16 / smt 2
        self.assertTrue(s.uarch.complete)
        self.assertEqual(s.uarch.with_overrides("family_l3_cores=8").family_l3_cores, 8)
        with self.assertRaises(S.SchemaError):
            s.uarch.with_overrides("cores=8")

    def test_int8_kernels_per_scheme(self):
        # The fixture's per-channel and per-group families, at 16 and 32 columns and tiles of 64 and 128.
        self.assertEqual(SCHEMA.nr_widths("int8"), [16, 32, 64, 128])
        self.assertEqual(SCHEMA.caps("int8", 32).schemes, ("per_channel",))
        self.assertEqual(SCHEMA.caps("int8", 32, "per_group").schemes, ("per_group",))
        self.assertTrue(SCHEMA.covers("int8", 12, 32))
        self.assertFalse(SCHEMA.covers("int8", 7, 32, "per_group"))
        self.assertTrue(SCHEMA.covers("bf16", 12, 32, "per_group"))  # BF16 kernels serve no INT8 scheme
        self.assertIn("schemes=per_group", SCHEMA.summary())
        with open(S.FIXTURE) as f:
            doc = json.load(f)
        no_group32 = [k for k in doc["kernels"]
                      if not (k["nr"] == 32 and k["kind"] == "int8" and k["int8_schemes"] == ["per_group"])]
        s = S.Schema(dict(doc, kernels=no_group32))
        self.assertEqual(S.schema_diff(SCHEMA, s), ["kernel int8 nr 32 (per_group): (63, 6, 4) vs None"])
        self.assertEqual(s.route_max_mr(s.route_of("s8", "per_group"), 2), 0)
        # A family that names no scheme serves every scheme.
        older = S.Schema(dict(doc, kernels=[dict(k, int8_schemes=[]) for k in doc["kernels"]
                                            if "per_group" not in k["int8_schemes"]]))
        self.assertEqual(older.route_max_mr(older.route_of("s8", "per_group"), 2), 12)

    def test_l3_domain_of_the_plan(self):
        hw = dict(SCHEMA.doc["hw"], l3_domain_threads=64, smt=2)
        u = S.Uarch.from_hw(S.Hardware(hw))
        self.assertEqual((u.l3_domain_cores, u.plan_domain), (32, 32))
        self.assertEqual(u.physical_cores, S.Hardware(hw).physical_cores)
        self.assertTrue(u.complete)
        self.assertEqual(S.Uarch.from_dict({"physical_cores": 32}).physical_cores, 32)
        self.assertEqual(S.Uarch.from_dict({}).physical_cores, 0)  # an older dump: one L2 a thread
        no_topology = S.Uarch.from_hw(S.Hardware(dict(hw, l3_domain_threads=0)))
        self.assertEqual(no_topology.l3_domain_cores, 0)
        self.assertIsNone(no_topology.plan_domain)  # neither field: the heuristic cannot run
        self.assertFalse(no_topology.complete)
        self.assertEqual(no_topology.with_overrides("family_l3_cores=8").plan_domain, 8)
        self.assertEqual(S.Uarch.from_dict({"family_l3_cores": 8, "l3_domain_cores": 32}).plan_domain, 32)

    def test_diff(self):
        with open(S.FIXTURE) as f:
            doc = json.load(f)
        doc["keys"][0]["values"] = [16]
        diff = S.schema_diff(SCHEMA, S.Schema(doc, "changed"))
        self.assertTrue(any("nr" in d and "values" in d for d in diff), diff)


if __name__ == "__main__":
    unittest.main()
