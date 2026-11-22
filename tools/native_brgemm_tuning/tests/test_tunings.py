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
"""The tuning files of tunings/ against the current planner's keys, values, match keys and plans."""

import glob
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import SCHEMA  # noqa: E402

from native_brgemm_tuning import export as E  # noqa: E402
from native_brgemm_tuning import schema as schema_mod  # noqa: E402
from native_brgemm_tuning.problems import Problem  # noqa: E402

TUNINGS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tunings")


def tuning_files():
    """The files of the machine types and of the models (tunings/models/<model>/)."""
    return sorted(glob.glob(os.path.join(TUNINGS, "**", "*.json"), recursive=True))


def machine_of(doc):
    """(schema, uarch) of the machine a file's plans are for: the fixture's keys and kernels, with the
    caches and L3 domain of the file's hardware block, and the F16_PH kernels only where the file's plans
    name them (the block's ISA does not tell)."""
    hw = doc["hardware"]
    f16_ph = any(pl["plan"].get("kernels") == "f16_ph" for pl in doc["meta"]["plans"])
    kernels = [k for k in SCHEMA.doc["kernels"] if f16_ph or k["kind"] != "f16_ph"]
    uarch = schema_mod.Uarch.from_dict(dict(SCHEMA.doc["uarch"], l1d_bytes=hw["l1d_kib"] << 10,
                                            l2_bytes=hw["l2_kib"] << 10, l3_bytes=hw["l3_kib"] << 10,
                                            l3_domain_cores=hw["l3_cores"]))
    return schema_mod.Schema(dict(SCHEMA.doc, kernels=kernels), SCHEMA.source), uarch


class TestTuningFiles(unittest.TestCase):
    def test_the_files_of_every_model_are_checked(self):
        models = os.path.join(TUNINGS, "models")
        names = sorted(d for d in os.listdir(models) if os.path.isdir(os.path.join(models, d)))
        self.assertTrue(names)
        checked = {os.path.relpath(path, models).split(os.sep)[0] for path in tuning_files()
                   if path.startswith(models + os.sep)}
        self.assertEqual(sorted(checked), names)

    def test_every_file_is_valid_and_names_its_plans(self):
        files = tuning_files()
        self.assertTrue(files)
        for path in files:
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
            self.assertEqual(E.validate_doc(doc, SCHEMA)[0], [], path)
            # Matched by what the planner reads of the machine.
            self.assertEqual(sorted(doc["hardware"]), ["isa", "l1d_kib", "l2_kib", "l3_cores", "l3_kib"], path)
            # tune.py verify checks the plan each entry names.
            self.assertEqual({pl["id"] for pl in doc["meta"]["plans"]}, {e["id"] for e in doc["entries"]}, path)
            # An M = 1 call runs on the GEMV kernels, which take none of an entry's tiles.
            low = [e["id"] for e in doc["entries"] if e["match"].get("m", {}).get("min", 1) < 2]
            self.assertEqual(low, [], path)

    def test_every_entry_names_the_plan_of_the_planner(self):
        # The CI holds plan.py to the library's planner: a planner change that moves an entry's plan
        # fails here until the file is tuned again.
        for path in tuning_files():
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
            schema, uarch = machine_of(doc)
            recs = doc["meta"]["plans"]
            probs = {r["problem"]: Problem.parse(r["problem"]) for r in recs}
            got = {(g["id"], g["problem"]): g for g in E.entry_plans(doc["entries"], list(probs.values()), schema,
                                                                     uarch)}
            self.assertEqual(set(got), {(r["id"], r["problem"]) for r in recs}, path)
            for r in recs:
                g = got[(r["id"], r["problem"])]
                # A field the record lacks, of a line before it, is not compared.
                self.assertEqual(({k: g["plan"].get(k) for k in r["plan"]}, g["source"], g["cfg"], g["builtin"]),
                                 (r["plan"], r["source"], r["cfg"], r["builtin"]), (path, r["id"], r["problem"]))
                for w, fields in r.get("paired", {}).items():
                    self.assertEqual({k: g["paired"][w].get(k) for k in fields}, fields, (path, r["id"], w))
                if "pairing" in r:
                    self.assertEqual(g.get("pairing"), r["pairing"], (path, r["id"], r["problem"]))


if __name__ == "__main__":
    unittest.main()
