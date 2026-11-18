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
"""tune.py end to end on the host: a synthetic campaign, and a replay dry run."""

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import HAVE_NUMPY, HW, SCHEMA, strict_json  # noqa: E402

from native_brgemm_tuning import export as E  # noqa: E402
from native_brgemm_tuning import tune  # noqa: E402

SHAPES = "model\tname\tN\tK\twei\tcalls\tm_lo\tm_hi\n" + "".join(
    "m%d\tw%d\t%d\t%d\t%s\t100\t1\t4096\n" % (i, i, n, k, w)
    for i, (n, k, w) in enumerate([(4096, 4096, "bf16"), (6144, 4096, "bf16"), (14336, 4096, "s8"),
                                   (4096, 14336, "s8"), (262144, 4096, "bf16")]))


def run(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = tune.main([str(a) for a in argv])
    return rc, out.getvalue()


class TestCli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="nbt_cli_")
        self.addCleanup(shutil.rmtree, self.tmp)
        self.shapes = os.path.join(self.tmp, "shapes.tsv")
        with open(self.shapes, "w") as f:
            f.write(SHAPES.replace("w4\t", "lm_head\t"))

    def test_synthetic_campaign(self):
        wd = os.path.join(self.tmp, "syn")
        rc, out = run("init", "--workdir", wd, "--shapes", self.shapes, "--m-classes", "1,2-16:8/16,129-512:256",
                      "--backend", "synthetic", "--noise", "0.002")
        self.assertEqual(rc, 0)
        self.assertIn("16 problems", out)  # lm_head left out
        strategy = "guided" if HAVE_NUMPY else "random"
        self.assertEqual(run("search", "--workdir", wd, "--strategy", strategy, "--budget", 12, "--batch", 6)[0], 0)
        self.assertEqual(run("confirm", "--workdir", wd)[0], 0)
        out_file = os.path.join(wd, "tuning.json")
        rc, out = run("export", "--workdir", wd, "-o", out_file, "--rules", "tree" if HAVE_NUMPY else "none",
                      "--min-leaf", 2)
        self.assertEqual(rc, 0, out)
        with open(out_file) as f:
            doc = json.load(f)
        self.assertEqual(E.validate_doc(doc, SCHEMA)[0], [])
        self.assertEqual(doc["version"], 2)
        self.assertEqual(doc["hardware"], E.hardware_block(SCHEMA))
        with open(os.path.join(wd, "export_report.json")) as f:
            report = json.load(f)
        self.assertEqual(report["entries"], len(doc.get("entries", [])))
        rc, out = run("check", out_file, "--workdir", wd)
        self.assertEqual(rc, 0)
        self.assertEqual(len(out.strip().splitlines()), 16)
        self.assertEqual(run("status", "--workdir", wd)[0], 0)
        # A widened block, and one for another model.
        wide_file, other_file = (os.path.join(wd, n) for n in ("wide.json", "other.json"))
        lo = HW["model"] - HW["model"] % 16
        self.assertEqual(run("export", "--workdir", wd, "-o", wide_file, "--rules", "none", "--hw-keys",
                             "family,model,l3_cores,smt", "--hw-set", "model=%d-%d" % (lo, lo + 15),
                             "--hw-set", "l3_cores=8|32")[0], 0)
        with open(wide_file) as f:
            self.assertEqual(json.load(f)["hardware"],
                             {"family": HW["family"], "model": {"min": lo, "max": lo + 15}, "l3_cores": [8, 32],
                              "smt": HW["smt"]})
        self.assertEqual(run("export", "--workdir", wd, "-o", other_file, "--rules", "none", "--hw-set",
                             "model=%d" % (HW["model"] + 1))[0], 0)
        rc, out = run("check", other_file, "--workdir", wd)
        self.assertIn("hardware key model does not hold on this machine (%d)" % HW["model"], out)
        self.assertEqual(out.count("hardware mismatch"), 16)
        for bad in (["--format-version", 1], ["--hw-set", "model=x"],
                    ["--hw-keys", "frequency"], ["--hw-match", "signature"]):
            with self.assertRaises(SystemExit):
                run("export", "--workdir", wd, "-o", os.path.join(wd, "bad.json"), "--rules", "none", *bad)
        # Merge: the entries of two files of one hardware block, one file again.
        table_file = os.path.join(wd, "table.json")
        self.assertEqual(run("export", "--workdir", wd, "-o", table_file, "--rules", "none")[0], 0)
        with open(table_file) as f:
            table = json.load(f)
        entries = table.get("entries", [])
        self.assertGreater(len(entries), 1)
        parts = []
        for i, chunk in enumerate((entries[::2], entries[1::2])):
            parts.append(os.path.join(wd, "part%d.json" % i))
            with open(parts[-1], "w") as f:
                json.dump(dict(table, entries=chunk), f)
        merged = os.path.join(wd, "merged.json")
        rc, out = run("merge", "-o", merged, *parts)
        self.assertEqual(rc, 0, out)
        with open(merged) as f:
            doc = json.load(f)
        self.assertEqual(doc["entries"], entries)
        self.assertEqual((doc["hardware"], [m["entries"] for m in doc["meta"]["merged"]]),
                         (table["hardware"], [len(entries[::2]), len(entries[1::2])]))
        self.assertEqual(E.validate_doc(doc, SCHEMA)[0], [])
        # The plans the inputs name, once; the inputs by file name, without their plans.
        self.assertEqual(doc["meta"]["plans"], table["meta"]["plans"])
        self.assertEqual([(m["file"], "plans" in m["meta"]) for m in doc["meta"]["merged"]],
                         [("part0.json", False), ("part1.json", False)])
        other = [dict(pl, plan=dict(pl["plan"], nr="16")) for pl in table["meta"]["plans"]]
        with self.assertRaises(ValueError):
            E.merge_docs([("a", dict(table, entries=entries[::2])),
                          ("b", dict(table, entries=entries[1::2], meta=dict(table["meta"], plans=other)))])
        for bad in ([parts[0], parts[0]], [parts[0], wide_file], [parts[0], out_file]):
            with self.assertRaises(SystemExit):
                run("merge", "-o", os.path.join(wd, "bad.json"), *bad)
        moved = dict(entries[0], id="moved")
        self.assertFalse(E._overlap(dict(moved["match"], n=moved["match"]["n"] + 32), moved["match"]))
        with self.assertRaises(ValueError):
            E.merge_docs([("a", dict(table, entries=[entries[0]])), ("b", dict(table, entries=[moved]))])

    @unittest.skipUnless(HAVE_NUMPY, "numpy is not installed")
    def test_rules_of_one_shape_export_strict_json(self):
        # No fold can hold out the only shape, so the rules have no held-out ratio.
        shapes = os.path.join(self.tmp, "one.tsv")
        with open(shapes, "w") as f:
            f.write("\n".join(SHAPES.splitlines()[:2]) + "\n")
        wd = os.path.join(self.tmp, "one")
        self.assertEqual(run("init", "--workdir", wd, "--shapes", shapes, "--m-classes",
                             "2-16:4/8/16,129-512:256/512", "--backend", "synthetic", "--noise", "0.002")[0], 0)
        self.assertEqual(run("search", "--workdir", wd, "--strategy", "guided", "--budget", 12, "--batch", 6)[0], 0)
        self.assertEqual(run("confirm", "--workdir", wd)[0], 0)
        out_file = os.path.join(wd, "tuning.json")
        rc, out = run("export", "--workdir", wd, "-o", out_file, "--rules", "tree", "--min-leaf", 2)
        self.assertEqual(rc, 0, out)
        with open(out_file, encoding="utf-8") as f:
            doc = strict_json(f.read())
        self.assertEqual(doc["rules"]["kind"], "tree")
        held_out = doc["meta"]["rules_report"]["heldout"]
        self.assertEqual((held_out["shapes"], held_out["measured"], held_out["geomean_ratio"],
                          held_out["oracle_geomean_ratio"]), (1, 0, None, None))
        self.assertIn("rules held out: no measured problem (1 shapes in 1 folds), so no held-out ratio", out)
        self.assertEqual(E.validate_doc(doc, SCHEMA)[0], [])
        with open(os.path.join(wd, "export_report.json"), encoding="utf-8") as f:
            self.assertEqual(strict_json(f.read())["rules"]["heldout"], held_out)

    def test_search_overrides_and_exclude(self):
        wd = os.path.join(self.tmp, "ov")
        self.assertEqual(run("init", "--workdir", wd, "--shapes", self.shapes, "--m-classes", "1,2-16:8/16",
                             "--backend", "synthetic", "--noise", "0.001")[0], 0)
        rc, out = run("search", "--workdir", wd, "--strategy", "grid", "--max-keys", 1, "--budget", 3, "--batch", 3,
                      "--ms", 8, "--keys", "kc,int8_kc,nr", "--values", "kc=512|1024")
        self.assertEqual(rc, 0, out)
        with open(os.path.join(wd, "results.jsonl")) as f:
            recs = [json.loads(line) for line in f]
        texts = {r["text"] for r in recs if r["kind"] == "measure" and r["text"]}
        self.assertEqual({r["problem"]["m"] for r in recs}, {8})
        self.assertTrue(texts and all(t.split("=")[0] in ("kc", "int8_kc", "nr") for t in texts), texts)
        self.assertFalse({t for t in texts if t.startswith("kc=") and t not in ("kc=512", "kc=1024")})
        with open(os.path.join(wd, "campaign.json")) as f:
            self.assertIsNone(json.load(f)["keys"])
        self.assertEqual(run("confirm", "--workdir", wd, "--only-new")[0], 0)
        rc, out = run("confirm", "--workdir", wd, "--only-new")
        self.assertIn("nothing to confirm", out)
        out_file = os.path.join(wd, "t.json")
        self.assertEqual(run("export", "--workdir", wd, "-o", out_file, "--rules", "none",
                             "--exclude", "bf16-n4096-k4096-m1-t32")[0], 0)

    def test_replay_dry_run(self):
        wd = os.path.join(self.tmp, "rp")
        rc, out = run("init", "--workdir", wd, "--shapes", self.shapes, "--dtypes", "s8", "--m-classes", "1,2-16:16",
                      "--backend", "replay", "--replay-bin", "/opt/replay", "--preload", "/opt/a.so:/opt/b.so")
        self.assertEqual(rc, 0)
        with open(os.path.join(wd, "campaign.json")) as f:
            cold = json.load(f)["backend"]["cold_mb"]
        u = SCHEMA.uarch
        self.assertEqual(cold, 8.0 * u.l3_bytes * -(-32 // u.plan_domain) / 1e6)
        rc, out = run("search", "--workdir", wd, "--strategy", "grid", "--budget", 2, "--batch", 2)
        self.assertIn("sessions written, not launched", out)
        with self.assertRaises(SystemExit):
            run("search", "--workdir", wd, "--strategy", "grid", "--budget", 2, "--batch", 2)
        self.assertIn("no outputs yet", run("collect", "--workdir", wd)[1])
        rc, out = run("collect", "--workdir", wd, "--force")
        self.assertIn("probes not applied", out)
        rc, out = run("search", "--workdir", wd, "--strategy", "grid", "--budget", 2, "--batch", 2)
        self.assertIn("round 2", out)  # failed candidates are proposed again
        with self.assertRaises(SystemExit):
            run("init", "--workdir", wd, "--shapes", self.shapes, "--backend", "synthetic")

    def test_replay_memory_cap_at_the_timed_layers(self):
        # bf16 4096 x 4096 (33.6 MB) takes 8 layers (268 MB) of a 256 MB floor, 6144 x 4096 6 (302 MB).
        args = ["init", "--workdir", os.path.join(self.tmp, "cap"), "--shapes", self.shapes, "--dtypes", "bf16",
                "--m-classes", "1", "--backend", "replay", "--replay-bin", "/opt/replay", "--layers", 1,
                "--cold-mb", 256]
        with self.assertRaises(SystemExit) as cm:
            run(*args, "--group-gb", 0.1)
        self.assertIn("cold_mb=256 MB", str(cm.exception))
        with self.assertRaises(SystemExit) as cm:
            run(*args, "--group-gb", 0.26)
        self.assertIn("allocates 0.268 GB of weights (8 layers), above group_gb 0.26", str(cm.exception))
        rc, out = run(*args, "--group-gb", 0.27)
        self.assertEqual(rc, 0, out)
        self.assertIn(": 1 problems (1 skipped)", out)
        self.assertIn("N=6144 K=4096: a timed process of this problem alone allocates 0.302 GB", out)

    def test_pod_backends_take_the_pack_width_of_the_library(self):
        for backend in (["--backend", "replay", "--replay-bin", "/opt/replay"],
                        ["--backend", "benchdnn", "--benchdnn", "/opt/benchdnn"]):
            with self.assertRaises(SystemExit) as cm:
                run("init", "--workdir", os.path.join(self.tmp, "p64"), "--shapes", self.shapes, "--dtypes", "bf16",
                    "--m-classes", "2-16:16", "--pack-nr", 64, *backend)
            self.assertIn("pack_nr 64: the library's calls pack every weight in blocks of 32 columns",
                          str(cm.exception))
        rc, out = run("init", "--workdir", os.path.join(self.tmp, "p64"), "--shapes", self.shapes, "--dtypes", "bf16",
                      "--m-classes", "2-16:16", "--pack-nr", 64, "--backend", "synthetic")
        self.assertEqual(rc, 0, out)

    def test_benchdnn_keeps_the_problems_its_lines_run(self):
        shapes = os.path.join(self.tmp, "wei.tsv")
        with open(shapes, "w") as f:
            f.write("model\tname\tN\tK\twei\tcalls\tm_lo\tm_hi\n" + "".join(
                "m\t%s\t4096\t4096\t%s\t100\t1\t16\n" % (w, w) for w in ("bf16", "s8", "s4", "f16")))
        wd = os.path.join(self.tmp, "bd")
        rc, out = run("init", "--workdir", wd, "--shapes", shapes, "--m-classes", "1,2-16:16",
                      "--backend", "benchdnn", "--benchdnn", "/opt/benchdnn")
        self.assertEqual(rc, 0, out)
        with open(os.path.join(wd, "campaign.json")) as f:
            self.assertEqual(sorted({p["dtype"] for p in json.load(f)["problems"]}), ["bf16", "s8"])
        self.assertIn(": 4 problems (4 skipped)", out)
        self.assertEqual(out.count("per-channel"), 4)

    def test_pods(self):
        wd = os.path.join(self.tmp, "pods")
        rc, out = run("init", "--workdir", wd, "--shapes", self.shapes, "--m-classes", "1,2-16:16",
                      "--backend", "replay", "--replay-bin", "/opt/replay", "--parallel", 3, "--pods", "pod-a,pod-b",
                      "--launcher", "/tools/run_one.sh --pod {pod} --cmd {dir} -- bash {script}")
        self.assertEqual(rc, 0, out)
        with open(os.path.join(wd, "campaign.json")) as f:
            self.assertEqual(json.load(f)["pods"], ["pod-a", "pod-b"])
        rc, out = run("search", "--workdir", wd, "--strategy", "grid", "--budget", 1, "--batch", 1, "--pods", "pod-c",
                      "--dry-run")
        self.assertEqual([line.split()[2] for line in out.splitlines() if "/tools/run_one.sh" in line], ["pod-c"])


if __name__ == "__main__":
    unittest.main()
