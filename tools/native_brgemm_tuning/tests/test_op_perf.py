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
"""The op performance utility in a campaign: probe jobs, the machine check, the bounds and
efficiencies of status and export, and meta.op_perf."""

import contextlib
import copy
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import MACHINE, SCHEMA, UARCH, session_records  # noqa: E402

from native_brgemm_tuning import calibration  # noqa: E402
from native_brgemm_tuning import export as E  # noqa: E402
from native_brgemm_tuning import op_perf, tune  # noqa: E402
from native_brgemm_tuning import problems as problems_mod  # noqa: E402
from native_brgemm_tuning.backends.benchdnn import BenchdnnBackend  # noqa: E402
from native_brgemm_tuning.backends.replay import ReplayBackend  # noqa: E402
from native_brgemm_tuning.candidates import Candidate, Space  # noqa: E402
from native_brgemm_tuning.problems import Problem  # noqa: E402
from native_brgemm_tuning.results import ResultsDB, summarize  # noqa: E402
from native_brgemm_tuning.search import SearchSettings, Searcher  # noqa: E402
from op_perf_utility.common.ceilings import Ceilings  # noqa: E402
from op_perf_utility.tests.helpers import example_doc  # noqa: E402

SHAPES = "model\tname\tN\tK\twei\tcalls\tm_lo\tm_hi\n" + "".join(
    "m\tw%d\t%d\t%d\t%s\t100\t1\t4096\n" % (i, n, k, w)
    for i, (n, k, w) in enumerate([(4096, 4096, "bf16"), (14336, 4096, "s8")]))
FAKE_PROBE = """#!/bin/bash
while [ $# -gt 0 ]; do [ "$1" = --out ] && out=$2; shift; done
if [ -f "%(src)s" ]; then cp "%(src)s" "$out" && echo "probe: wrote $out"; else echo "probe: failed"; exit 1; fi
"""


def ceilings_doc(threads=32, **cpu):
    """The utility's synthetic machine with the CPU fields of the tests' schema, its L3 domain
    and a team of threads."""
    doc = copy.deepcopy(example_doc())
    hw = SCHEMA.hw
    if hw is not None:
        doc["cpu"].update({k: getattr(hw, k) for k in ("vendor", "family", "model", "stepping", "isa", "l1d_kib",
                                                      "l2_kib", "l3_kib", "logical_cpus", "smt", "physical_cores",
                                                      "l3_domain_threads")})
    fp16 = (SCHEMA.doc.get("uarch") or {}).get("avx512fp16")
    if isinstance(fp16, bool):
        doc["cpu"]["avx512fp16"] = fp16
    doc["cpu"].update(cpu)
    domain = SCHEMA.uarch.plan_domain
    doc["uarch"]["l3_domain_cores"] = domain
    doc["run"].update(threads=threads, domain_threads=domain)
    return doc


def run(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = tune.main([str(a) for a in argv])
    return rc, out.getvalue()


class TestMachine(unittest.TestCase):
    def test_match(self):
        self.assertEqual(op_perf.machine_mismatch(Ceilings(ceilings_doc()), SCHEMA, 32), [])
        bad = op_perf.machine_mismatch(Ceilings(ceilings_doc(16)), SCHEMA, 32)
        self.assertEqual(bad, ["threads: probe 16, campaign 32"])
        if SCHEMA.hw is not None and SCHEMA.hw.model is not None:
            bad = op_perf.machine_mismatch(Ceilings(ceilings_doc(model=SCHEMA.hw.model + 1)), SCHEMA, 32)
            self.assertEqual(bad, ["model: probe %d, schema %d" % (SCHEMA.hw.model + 1, SCHEMA.hw.model)])
        doc = ceilings_doc()
        doc["uarch"]["l3_domain_cores"] = SCHEMA.uarch.plan_domain + 1
        self.assertIn("L3 domain: probe %d cores, campaign %d" % (SCHEMA.uarch.plan_domain + 1,
                                                                  SCHEMA.uarch.plan_domain),
                      op_perf.machine_mismatch(Ceilings(doc), SCHEMA, 32))
        fp16 = (SCHEMA.doc.get("uarch") or {}).get("avx512fp16")
        if isinstance(fp16, bool):
            self.assertEqual(op_perf.machine_mismatch(Ceilings(ceilings_doc(avx512fp16=not fp16)), SCHEMA, 32),
                             ["avx512fp16: probe %s, schema %s" % (not fp16, fp16)])

    def test_cold_weights(self):
        self.assertTrue(op_perf.cold_weights({"backend": "replay", "cold_mb": 1000.0}))
        self.assertFalse(op_perf.cold_weights({"backend": "replay", "cold_mb": 0}))
        self.assertFalse(op_perf.cold_weights({"backend": "benchdnn"}))

    def test_efficiency_rows(self):
        ceil = Ceilings(ceilings_doc())
        p = Problem.parse("dtype=bf16,m=1,n=4096,k=4096,threads=32")
        q = Problem.parse("dtype=bf16,m=512,n=4096,k=4096,threads=32")
        probs = {x.key(): x for x in (p, q)}
        fast = Candidate({"kc": 512}, "kc=512", None, ())
        recs = session_records(probs, {p.key(): [fast], q.key(): []}, lambda _p, c: 0.5, "s1")
        summ = summarize(recs)
        bnds = op_perf.bounds([p, q], ceil, cold=True)
        op_perf.attach(summ, bnds)
        self.assertEqual(bnds[p.key()].bound, "dram")
        self.assertEqual(bnds[q.key()].bound, "compute")
        ps = summ[p.key()]
        self.assertEqual(ps.bound, "dram")
        self.assertAlmostEqual(ps.builtin_efficiency, bnds[p.key()].attainable_ms / 1.0)
        self.assertAlmostEqual(ps.best_efficiency(0.02), bnds[p.key()].attainable_ms / 0.5)
        rows = {r["problem"]: r for r in op_perf.efficiency_rows(summ, [p, q], bnds, 0.02)}
        self.assertEqual(rows[p.key()]["best"], "kc=512")
        self.assertAlmostEqual(rows[p.key()]["best_pct"], round(200 * bnds[p.key()].attainable_ms, 1))
        self.assertEqual(rows[q.key()]["best_pct"], rows[q.key()]["builtin_pct"])
        m = op_perf.meta(rows.values(), ceil, True)
        self.assertEqual(set(m), {"probe", "cold_weights", "builtin_pct_geomean", "best_pct_geomean", "problems"})
        self.assertEqual(m["probe"], {"probe_version": "1.0", "created": "2026-01-01T00:00:00Z", "threads": 32})
        self.assertEqual(set(m["problems"][0]), {"problem", "bound", "builtin_pct", "best_pct", "best_status"})
        self.assertEqual((rows[p.key()]["best_status"], rows[q.key()]["best_status"]), ("unconfirmed", ""))
        # A confirmation that holds the candidate, and one where it does not hold.
        held = summarize(session_records(probs, {p.key(): [fast]}, lambda _p, c: 0.6, "c1", "confirm"),
                         phases=["confirm"])
        r = op_perf.efficiency_rows(summ, [p], bnds, 0.02, held)[0]
        self.assertEqual((r["best"], r["best_status"]), ("kc=512", "confirmed"))
        self.assertAlmostEqual(r["best_ms"], 0.6)
        lost = summarize(session_records(probs, {p.key(): [fast]}, lambda _p, c: 1.0, "c2", "confirm"),
                         phases=["confirm"])
        r = op_perf.efficiency_rows(summ, [p], bnds, 0.02, lost)[0]
        self.assertEqual((r["best"], r["best_status"], r["best_ms"]), ("kc=512", "rejected", 1.0))
        # A confirmation that times the built-in plan at 2 ms and the winner at 1 ms: the winner
        # takes 1 ms, not half of the search's 1 ms.
        slow = session_records(probs, {p.key(): [fast]}, lambda _p, c: 0.5, "c3", "confirm")
        for rec in slow:
            if rec["kind"] == "measure":
                rec["time_ms"] *= 2.0
        r = op_perf.efficiency_rows(summ, [p], bnds, 0.02, summarize(slow, phases=["confirm"]))[0]
        self.assertEqual(r["best_status"], "confirmed")
        self.assertAlmostEqual(r["best_ms"], 1.0)

    def test_every_weight_type(self):
        doc = ceilings_doc()
        with_fp16 = copy.deepcopy(doc)
        with_fp16["compute"]["fp16"] = dict(doc["compute"]["fp32"], flops_per_instr=64, gflops_core=384.0,
                                            gflops_team=5120.0)
        probs = [Problem("bf16", 16, 4096, 4096), Problem("s8", 16, 4096, 4096),
                 Problem("s8", 16, 4096, 4096, src="u8"), Problem("s4", 16, 4096, 4096, group=128),
                 Problem("s4", 16, 4096, 4096, group=4096, scale_bytes=4), Problem("f16", 16, 4096, 4096),
                 Problem("f32", 16, 4096, 4096)]
        for d, f16_path in ((doc, "fp32"), (with_fp16, "fp16")):
            bnds = op_perf.bounds(probs, Ceilings(d), cold=True)
            self.assertEqual([bnds[p.key()].path for p in probs],
                             ["bf16", "int8", "int8", "int8", "int8", f16_path, "fp32"])
        w = [bnds[p.key()].work for p in probs]
        self.assertEqual((w[3].wei_scale_bytes, w[4].wei_scale_bytes), (4096 * 32 * 2, 4096 * 4))
        self.assertEqual((w[1].src_scale_bytes, w[2].src_bytes, w[2].src_scale_bytes), (16 * 4, 16 * 4096, 0))
        self.assertEqual((w[5].src_bytes, w[6].src_bytes), (16 * 4096 * 2, 16 * 4096 * 4))
        recs = session_records({p.key(): p for p in probs}, {p.key(): [] for p in probs}, lambda p, c: 1.0, "s1")
        rows = op_perf.efficiency_rows(summarize(recs), probs, bnds, 0.02)
        self.assertTrue(len(rows) == len(probs) and all(r["builtin_pct"] for r in rows))


HEAD_PROBLEMS = [Problem("bf16", m, 4096, 4096) for m in (16, 32, 64)]
HEAD_KEYS = ["nr", "threads_cap", "traversal", "mb", "tile_rows", "min_tiles_per_thread", "copy_block"]


class TestHeadroom(unittest.TestCase):
    """Built-in plans take 1 ms; the problems' bounds put them at 95%, 20% and 50% efficiency."""

    def search(self, ratio=lambda p, c: 1.0, **kw):
        tmp = tempfile.mkdtemp(prefix="nbt_head_")
        self.addCleanup(shutil.rmtree, tmp)
        a, b, c = (p.key() for p in HEAD_PROBLEMS)
        probs = {p.key(): p for p in HEAD_PROBLEMS}
        spaces = {p.key(): Space(p, SCHEMA, UARCH, HEAD_KEYS) for p in HEAD_PROBLEMS}
        settings = SearchSettings(strategy="random", attainable={a: 0.95, b: 0.2, c: 0.5}, **kw)
        logs, counts = [], {k: 0 for k in probs}

        def evaluate(rnd, phase, batch):
            for k, v in batch.items():
                counts[k] += len(v)
                self.assertLessEqual(len(v), settings.batch)
            return session_records(probs, batch, ratio, "s%03d" % rnd, phase)

        db = ResultsDB(os.path.join(tmp, "r.jsonl"))
        Searcher(HEAD_PROBLEMS, spaces, evaluate, db, settings, log=logs.append).run()
        return [counts[k] for k in (a, b, c)], logs

    def test_off_by_default(self):
        self.assertFalse(SearchSettings(attainable={"x": 1.0}).headroom)
        self.assertEqual(self.search(budget=4, batch=2)[0], [4, 4, 4])

    def test_skip(self):
        counts, logs = self.search(budget=6, batch=3, skip_at=0.9)
        # One scouting candidate each, then nothing more for the problem at 95%.
        self.assertEqual(counts, [1, 6, 6])
        self.assertIn("no more candidates for %s: built-in plan at 95%% of its bound" % HEAD_PROBLEMS[0].label(),
                      logs)

    def test_stop(self):
        # The third problem's candidates halve its time: 0.5 ms reaches its 0.5 ms bound.
        counts, logs = self.search(lambda p, c: 0.5 if p.m == 64 else 1.0, budget=6, batch=3, stop_at=0.97)
        self.assertEqual(counts, [6, 6, 1])
        self.assertIn("no more candidates for %s: best plan at 100%% of its bound" % HEAD_PROBLEMS[2].label(), logs)
        # A built-in plan already there is the best plan: the problem stops after its scout.
        counts, _ = self.search(budget=6, batch=3, stop_at=0.9)
        self.assertEqual(counts, [1, 6, 6])

    def test_prioritize(self):
        # Headroom 0.05, 0.8 and 0.5 ms of 1.35: 24 candidates split 1 (raised to 2), 14 and 9.
        counts, _ = self.search(budget=8, batch=4, prioritize=True)
        self.assertEqual(counts, [2, 14, 9])
        # Calls weigh the headroom: ten times the calls give the first problem 0.5 ms of 1.8.
        a = HEAD_PROBLEMS[0].key()
        counts, _ = self.search(budget=8, batch=4, prioritize=True, weights={a: 10})
        self.assertEqual(counts, [7, 11, 7])
        # Relative: the headroom alone, whatever the calls.
        counts, _ = self.search(budget=8, batch=4, prioritize=True, prioritize_by="relative", weights={a: 10})
        self.assertEqual(counts, [2, 14, 9])

    def test_calls_at_each_m(self):
        rows = [(1, 16, 100), (17, 0, 5)]
        self.assertEqual([problems_mod.calls_at(rows, m) for m in (1, 16, 17, 4096)], [100, 100, 5, 5])
        self.assertEqual(problems_mod.calls_at([(1, 16, 100)], 64), 0)
        tmp = tempfile.mkdtemp(prefix="nbt_calls_")
        self.addCleanup(shutil.rmtree, tmp)
        shapes = os.path.join(tmp, "shapes.tsv")
        with open(shapes, "w") as f:
            f.write("model\tname\tN\tK\twei\tcalls\tm_lo\tm_hi\n"
                    "m\tqkv\t4096\t4096\tbf16\t100\t1\t16\n"
                    "m\tqkv\t4096\t4096\tbf16\t5\t17\t0\n")
        wd = os.path.join(tmp, "syn")
        self.assertEqual(run("init", "--workdir", wd, "--shapes", shapes, "--m-classes", "2-16:16,17-64:64",
                             "--backend", "synthetic")[0], 0)
        with open(os.path.join(wd, "campaign.json")) as f:
            weights = json.load(f)["weights"]
        # A model trace's rows give each problem the calls at its M, not the shape's 105.
        self.assertEqual({Problem.parse(k).m: v for k, v in weights.items()}, {16: 100, 64: 5})

    def test_per_group_s8(self):
        from native_brgemm_tuning.candidates import builtin_plan
        from native_brgemm_tuning.problems import rule_features

        p = Problem.parse("dtype=s8,m=8,n=4096,k=4096,int8_scheme=per_group,group=128")
        self.assertEqual((p.group, p.scale_bytes), (128, 4))
        self.assertEqual((Problem.parse(p.key()), Problem.from_dict(p.to_dict())), (p, p))
        self.assertIn("s8 g128", p.label())
        # The library's group feature, and its plan, are an s4 weight's only.
        plain = Problem.parse("dtype=s8,m=8,n=4096,k=4096,int8_scheme=per_group")
        self.assertEqual(rule_features(p, UARCH, SCHEMA.hw), rule_features(plain, UARCH, SCHEMA.hw))
        self.assertEqual(builtin_plan(p, UARCH, SCHEMA), builtin_plan(plain, UARCH, SCHEMA))
        with self.assertRaises(ValueError):
            Problem.parse("dtype=s8,m=8,n=4096,k=4096,group=128")  # a per-channel s8 weight has no group
        with self.assertRaises(ValueError):
            Problem.parse("dtype=s8,m=8,n=4096,k=4096,int8_scheme=per_group,group=100")
        # The bound prices its scales per group of 128 rows.
        bnd = op_perf.bounds([p, plain], Ceilings(ceilings_doc()), cold=True)
        self.assertEqual(bnd[p.key()].work.wei_scale_bytes, 4096 * 32 * 4)
        self.assertEqual(bnd[plain.key()].work.wei_scale_bytes, 4096 * 4)

    def test_cli(self):
        tmp = tempfile.mkdtemp(prefix="nbt_head_cli_")
        self.addCleanup(shutil.rmtree, tmp)
        shapes = os.path.join(tmp, "shapes.tsv")
        with open(shapes, "w") as f:
            f.write(SHAPES.replace("\t100\t", "\t250\t", 1))
        wd = os.path.join(tmp, "syn")
        self.assertEqual(run("init", "--workdir", wd, "--shapes", shapes, "--m-classes", "1,2-16:16",
                             "--backend", "synthetic")[0], 0)
        with open(os.path.join(wd, "campaign.json")) as f:
            weights = json.load(f)["weights"]
        self.assertEqual(sorted(weights.values()), [100, 100, 250, 250])
        with self.assertRaises(SystemExit):
            run("search", "--workdir", wd, "--prioritize")
        src = os.path.join(tmp, "c.json")
        with open(src, "w") as f:
            json.dump(ceilings_doc(), f)
        self.assertEqual(run("probe", "--workdir", wd, "--ceilings", src)[0], 0)
        rc, out = run("search", "--workdir", wd, "--strategy", "random", "--budget", 4, "--batch", 2,
                      "--prioritize", "--skip-at-pct", 99.9, "--stop-at-pct", 99.9)
        self.assertEqual(rc, 0, out)
        self.assertIn("round 1 (random): 4 candidates over 4 problems", out)


# The utility's synthetic machine calibrated for a 32-thread campaign (see TestCalibration.test_cli).
CAL = {"cm_fma_pipes": 2, "cm_fma_lat": 6, "cm_l2_bytes": 32, "cm_l3_cycles": 64, "cm_mem_cycles": 683,
       "cm_stream_cycles": 128}
# The tests' machine (helpers.UARCH) for a campaign, whatever machine the schema describes.
UARCH_TEXT = "l1d_bytes=49152,l2_bytes=1048576,l3_bytes=33554432,family_l3_cores=8,num_cores=32,l3_domain_cores=8"
# On it, the calibrated values change the plans of the bf16 M = 16 problems, not these others.
CHANGED = [Problem("bf16", 16, 4096, 4096), Problem("bf16", 16, 4096, 14336)]
UNCHANGED = [Problem("bf16", 8, 4096, 4096), Problem("s8", 256, 4096, 4096)]


def features(p):
    from native_brgemm_tuning.problems import rule_features

    return rule_features(p, UARCH, SCHEMA.hw)


class TestCalibration(unittest.TestCase):
    def test_space_base(self):
        p = CHANGED[0]
        sp = Space(p, SCHEMA, UARCH, HEAD_KEYS, base=CAL)
        plain = Space(p, SCHEMA, UARCH, HEAD_KEYS)
        self.assertIsNone(plain.base_candidate)
        self.assertEqual(sp.default.sig, plain.default.sig)
        # cm_fma_lat=6 is the built-in value: the base leaves it out.
        self.assertEqual(sp.base_candidate.text,
                         "cm_fma_pipes=2,cm_l2_bytes=32,cm_l3_cycles=64,cm_mem_cycles=683,cm_stream_cycles=128")
        self.assertEqual(next(sp.iter_grid(1)).text, sp.base_candidate.text)
        c = sp.make({"nr": 16})
        self.assertEqual(c.config, dict(sp.base_candidate.config, nr=16))
        self.assertEqual(sp.from_text(c.text).sig, c.sig)
        for q in UNCHANGED:
            self.assertIsNone(Space(q, SCHEMA, UARCH, [], base=CAL).base_candidate, q.label())
        # A base key set back to its built-in value stays in the text, so the text reads back
        # to the candidate that was timed.
        reset = sp.make({"cm_l2_bytes": 16})
        self.assertIn("cm_l2_bytes=16", reset.text)
        back = sp.from_text(reset.text)
        self.assertEqual((back.config, back.text, back.sig), (reset.config, reset.text, reset.sig))

    def test_decide(self):
        probs = CHANGED + UNCHANGED
        plain = {p.key(): Space(p, SCHEMA, UARCH, []) for p in probs}
        batch = calibration.ab_batch(probs, plain, {"bf16": CAL, "s8": CAL})
        self.assertEqual(sorted(batch), sorted(p.key() for p in CHANGED))

        def verdict(ratio, keys=None):
            b = {k: v for k, v in batch.items() if keys is None or k in keys}
            recs = session_records({p.key(): p for p in probs}, b, ratio, "c001_0", "calibrate")
            return calibration.decide(probs, batch, summarize(recs, phases=["calibrate"]), 0.02, 0.01)

        v = verdict(lambda p, c: 0.9)
        self.assertTrue(v["accepted"], v["why"])
        self.assertEqual((v["changed"], v["geomean_changed"], v["geomean_ratio"]), (2, 0.9, round(0.81 ** 0.25, 4)))
        v = verdict(lambda p, c: 0.8 if p.k == 4096 else 1.1)
        self.assertEqual((v["accepted"], v["slower"]), (False, [CHANGED[1].key()]))
        v = verdict(lambda p, c: 0.9, keys=[CHANGED[0].key()])
        self.assertEqual((v["accepted"], v["missing"]), (False, [CHANGED[1].key()]))
        v = verdict(lambda p, c: 0.995)
        self.assertEqual((v["accepted"], v["why"]),
                         (False, ["geomean ratio 0.9950 over the 2 changed problems above 0.9900"]))
        # 5% on the 2 changed problems among 34: 0.997 over all of them, kept.
        recs = session_records({p.key(): p for p in probs}, batch, lambda p, c: 0.95, "c001_0", "calibrate")
        extra = [Problem("bf16", 8, 4096 + 32 * i, 4096) for i in range(30)]
        v = calibration.decide(probs + extra, batch, summarize(recs, phases=["calibrate"]), 0.02, 0.01)
        self.assertTrue(v["accepted"], v["why"])
        self.assertGreater(v["geomean_ratio"], 0.99)

    def test_seed_values(self):
        cfgs = {"bf16": {"cm_fma_lat": 6, "cm_l2_bytes": 40}, "s8": {"cm_fma_lat": 4, "cm_l2_bytes": 40}}
        self.assertEqual(calibration.seed_values(["kc=512|1024", "cm_l2_bytes=8|32"], cfgs, SCHEMA),
                         ["kc=512|1024", "cm_fma_lat=3|4|5|6|7", "cm_l2_bytes=16|20|40|80"])

    def test_rules(self):
        from native_brgemm_tuning.rules import eval_rules

        cfgs = {"bf16": {"cm_fma_lat": 6, "cm_l2_bytes": 40}, "s8": {"cm_fma_lat": 4, "cm_l2_bytes": 40}}
        bf, s8, s4 = (Problem("bf16", 8, 4096, 4096), Problem("s8", 8, 4096, 4096),
                      Problem("s4", 8, 4096, 4096, group=128))
        trained = {"kind": "tree", "root": {"feature": "m", "le": 1, "then": {"config": {}}, "else": {
            "feature": "k", "le": 8192, "then": {"config": {}}, "else": {"id": "c1", "config": {"nr": 16}}}}}
        ordered = {"kind": "ordered", "list": [{"id": "c1", "when": [["m", ">=", 2], ["k", ">", 8192]],
                                                "config": {"nr": 16}}]}
        for rules in (None, trained, ordered):
            r = calibration.with_rules(rules, cfgs, 32)
            self.assertEqual(E._check_rules(r, SCHEMA), [])
            self.assertEqual(eval_rules(r, features(bf)), cfgs["bf16"])
            self.assertEqual(eval_rules(r, features(s8)), cfgs["s8"])
            # No s4 problem in the campaign, and M = 1 runs on the GEMV kernels.
            self.assertIn(eval_rules(r, features(s4)), ({}, None))
            self.assertIn(eval_rules(r, features(Problem("bf16", 1, 4096, 4096))), ({}, None))
            # The values were measured on the campaign's team of 32: other teams keep the rules.
            for t in (16, 31, 33, 64):
                self.assertIn(eval_rules(r, features(Problem("bf16", 8, 4096, 4096, threads=t))), ({}, None), t)
            if rules is not None:
                self.assertEqual(eval_rules(r, features(Problem("s8", 8, 4096, 14336))), dict(cfgs["s8"], nr=16))
                self.assertEqual(eval_rules(r, features(Problem("s8", 8, 4096, 14336, threads=64))), {"nr": 16})
                # A type without calibration keeps the rule's own config.
                self.assertEqual(eval_rules(r, features(Problem("s4", 8, 4096, 14336, group=128))), {"nr": 16})
                # A BF16-only calibration reaches no other type.
                only = calibration.with_rules(rules, {"bf16": cfgs["bf16"]}, 32)
                self.assertEqual(E._check_rules(only, SCHEMA), [])
                self.assertEqual(eval_rules(only, features(Problem("bf16", 8, 4096, 14336))),
                                 dict(cfgs["bf16"], nr=16))
                self.assertEqual(eval_rules(only, features(Problem("s8", 8, 4096, 14336))), {"nr": 16})
        # A trained leaf's calibrated copies are named for it, as an ordered rule's variants are.
        ids = set()

        def leaves(n):
            if "feature" in n:
                leaves(n["then"])
                leaves(n["else"])
            elif n.get("id"):
                ids.add(n["id"])

        leaves(calibration.with_rules(trained, cfgs, 32)["root"])
        self.assertEqual(ids, {"c1", "c1_bf16", "c1_s8", "calibration_bf16", "calibration_s8"})
        self.assertEqual({r["id"] for r in calibration.with_rules(ordered, cfgs, 32)["list"]},
                         {"c1", "c1_bf16", "c1_s8", "calibration_bf16", "calibration_s8"})
        every = {"bf16": {"cm_fma_lat": 6}, "s8": {"cm_fma_lat": 4}, "s4": {"cm_fma_lat": 5}, "f16": {"cm_fma_lat": 3}}
        r = calibration.with_rules(None, every, 32)
        self.assertEqual(E._check_rules(r, SCHEMA), [])
        for p, d in ((bf, "bf16"), (s8, "s8"), (s4, "s4"), (Problem("f16", 8, 4096, 4096), "f16")):
            self.assertEqual(eval_rules(r, features(p)), every[d], d)
        self.assertIn(eval_rules(r, features(Problem("f32", 8, 4096, 4096))), ({}, None))
        from op_perf_utility.matmul.calibrate import calibrate

        cal = calibrate(Ceilings(ceilings_doc()))
        self.assertEqual(sorted(calibration.configs(cal, ["bf16", "s4", "f16", "f32"])), ["bf16", "f16", "s4"])
        self.assertNotIn("cm_l2_bytes", calibration.configs(cal, ["bf16"], calibration.DEFAULT_KEYS)["bf16"])

    def test_plans_and_verify_under_rules(self):
        probs = [CHANGED[0], UNCHANGED[0]]
        entries = [{"id": "e", "match": E.match_block(probs[0], 16, 16), "config": {"nr": 16}}]
        rules = calibration.with_rules(None, {"bf16": {"cm_l2_bytes": 40}}, 32)
        plans = E.entry_plans(entries, probs, SCHEMA, UARCH, rules)
        self.assertEqual([(r["id"], r["problem"], r["source"], r["cfg"]) for r in plans],
                         [("e", probs[0].key(), "rules+table", "nr=16,cm_l2_bytes=40"),
                          ("rules", probs[1].key(), "rules", "cm_l2_bytes=40")])
        self.assertEqual(E.entry_plans(entries, probs, SCHEMA, UARCH)[0]["source"], "table")
        # What the library logs under the file against what the file names.
        from native_brgemm_tuning import plan as PL
        from native_brgemm_tuning.backends import jobs

        tmp = tempfile.mkdtemp(prefix="nbt_verify_")
        self.addCleanup(shutil.rmtree, tmp)
        be = ReplayBackend()

        def check(ruled_source):
            d = os.path.join(tmp, ruled_source)
            jobs.write_verify(d, be, probs, os.path.join(tmp, "t.json"))
            with open(os.path.join(d, jobs.VERIFY_MANIFEST)) as f:
                man = json.load(f)
            named = {r["problem"]: r for r in plans}
            for r in man["runs"]:
                lines = []
                for k in r["pkeys"]:
                    p = Problem.parse(k)
                    fields, source, cfg = PL.plan_log_fields(PL.builtin_plan(p, UARCH, SCHEMA), p), "heuristic", "-"
                    if r["kind"] == "file":
                        source = named[k]["source"] if k != probs[1].key() else ruled_source
                        cfg = named[k]["cfg"] if source != "heuristic" else "-"
                        fields = named[k]["plan"] if source != "heuristic" else fields
                    lines.append("[MATMUL.NATIVE_BRGEMM] M=%d N=%d K=%d source=%s cfg=%s %s" % (
                        p.m, p.n, p.k, source, cfg, " ".join("%s=%s" % kv for kv in fields.items())))
                    lines.append("CHECK shape=x N=%d K=%d M=%d hash=ab" % (p.n, p.k, p.m))
                with open(os.path.join(d, "out", r["id"] + ".txt"), "w") as f:
                    f.write("\n".join(lines) + "\n")
            return jobs.collect_verify(d, plans, be)

        rep = check("rules")
        self.assertEqual((rep["failed"], rep["named"], rep["ruled"]), (0, 1, 1), rep["rows"])
        self.assertEqual(check("heuristic")["failed"], 1)

    def test_cli(self):
        tmp = tempfile.mkdtemp(prefix="nbt_cal_")
        self.addCleanup(shutil.rmtree, tmp)
        shapes = os.path.join(tmp, "shapes.tsv")
        with open(shapes, "w") as f:
            f.write(SHAPES.replace("14336\t4096\ts8", "4096\t14336\tbf16"))
        wd = os.path.join(tmp, "syn")
        self.assertEqual(run("init", "--workdir", wd, "--shapes", shapes, "--m-classes", "8,16", "--uarch", UARCH_TEXT,
                             "--backend", "synthetic", "--noise", "0.001")[0], 0)
        with self.assertRaises(SystemExit):
            run("calibrate", "--workdir", wd)  # no ceilings yet
        doc = ceilings_doc()
        doc["uarch"]["l3_domain_cores"] = doc["run"]["domain_threads"] = 8
        src = os.path.join(tmp, "c.json")
        with open(src, "w") as f:
            json.dump(doc, f)
        self.assertEqual(run("probe", "--workdir", wd, "--ceilings", src, "--force")[0], 0)
        rc, out = run("calibrate", "--workdir", wd)
        self.assertEqual(rc, 0, out)
        self.assertIn("bf16: cm_fma_pipes=2,cm_fma_lat=6,cm_l3_cycles=64,cm_mem_cycles=683,cm_stream_cycles=128\n",
                      out)  # no cm_l2_bytes unless named
        with open(os.path.join(wd, "campaign.json")) as f:
            parallel = json.load(f)["parallel"]
        rc, out = run("calibrate", "--workdir", wd, "--keys", ",".join(calibration.KEYS), "--parallel", parallel + 2)
        self.assertEqual(rc, 0, out)
        self.assertIn("calibration round 1: 2 of 4 problems change plan", out)
        with open(os.path.join(wd, "campaign.json")) as f:
            camp = json.load(f)
        # The A/B's --parallel is its own; the campaign keeps its setting.
        self.assertEqual(camp["parallel"], parallel)
        cal = camp["calibration"]
        self.assertEqual((cal["problems"], cal["changed"]), (4, 2))
        # The campaign's 32 threads share the 120 GB/s DRAM saturates at: 1024 / (120 / 32 / 2.5).
        self.assertEqual(cal["configs"]["bf16"], {"cm_fma_pipes": 2, "cm_fma_lat": 6, "cm_l2_bytes": 32,
                                                   "cm_l3_cycles": 64, "cm_mem_cycles": 683, "cm_stream_cycles": 128})
        # Whatever the synthetic A/B said, as if it kept them: the search starts from them.
        camp["calibration"]["accepted"] = True
        camp["values"] = calibration.seed_values([], cal["configs"], SCHEMA)
        with open(os.path.join(wd, "campaign.json"), "w") as f:
            json.dump(camp, f)
        rc, out = run("search", "--workdir", wd, "--strategy", "grid", "--budget", 2, "--batch", 2)
        self.assertEqual(rc, 0, out)
        with open(os.path.join(wd, "results.jsonl")) as f:
            texts = {json.loads(line)["text"] for line in f if '"phase": "search"' in line}
        base_text = "cm_fma_pipes=2,cm_l2_bytes=32,cm_l3_cycles=64,cm_mem_cycles=683,cm_stream_cycles=128"
        self.assertIn(base_text, texts)
        self.assertTrue(all(base_text.split(",")[1] in t for t in texts if t))
        with self.assertRaises(SystemExit):
            run("calibrate", "--workdir", wd)  # after the search
        out_file = os.path.join(wd, "t.json")
        rc, out = run("export", "--workdir", wd, "-o", out_file, "--rules", "none", "--no-confirm")
        self.assertEqual(rc, 0, out)
        with open(out_file) as f:
            doc = json.load(f)
        self.assertEqual(E.validate_doc(doc, SCHEMA)[0], [])
        self.assertEqual(eval_rules_of(doc, Problem("bf16", 8, 4096, 4096)),
                         E.config_json(cal["configs"]["bf16"], SCHEMA))
        self.assertTrue(doc["meta"]["op_perf"]["calibration"]["accepted"])
        self.assertEqual(doc["meta"]["op_perf"]["calibration"]["configs"], cal["configs"])
        self.assertTrue(all("cm_l2_bytes" in e["config"] for e in doc.get("entries", [])))
        rc, out = run("export", "--workdir", wd, "-o", out_file, "--rules", "none", "--no-confirm",
                      "--no-calibration-rule")
        self.assertEqual(rc, 0, out)
        with open(out_file) as f:
            self.assertNotIn("rules", json.load(f))
        # A rejected calibration's values are no settings of the file: its meta keeps the verdict.
        with open(os.path.join(wd, "campaign.json")) as f:
            camp = json.load(f)
        camp["calibration"]["accepted"] = False
        with open(os.path.join(wd, "campaign.json"), "w") as f:
            json.dump(camp, f)
        rc, out = run("export", "--workdir", wd, "-o", out_file, "--rules", "none", "--no-confirm")
        self.assertEqual(rc, 0, out)
        with open(out_file) as f:
            doc = json.load(f)
        self.assertNotIn("rules", doc)
        self.assertEqual(doc["meta"]["op_perf"]["calibration"],
                         {k: v for k, v in camp["calibration"].items()
                          if k in ("accepted", "geomean_ratio", "geomean_changed", "problems", "changed",
                                   "calib_min_gain")})


def eval_rules_of(doc, p):
    from native_brgemm_tuning.rules import eval_rules

    return eval_rules(doc["rules"], features(p))


class TestProbeJobs(unittest.TestCase):
    def test_pinning_of_the_timed_runs(self):
        self.assertIn("export OMP_NUM_THREADS=8 OMP_PROC_BIND=close OMP_PLACES=cores",
                      BenchdnnBackend(threads=8).probe_env())
        if shutil.which("bash") is None:
            self.skipTest("no bash")
        lines = ReplayBackend(threads=2).probe_env() + ['echo "$OMP_NUM_THREADS $OMP_PROC_BIND $OMP_PLACES"']
        out = subprocess.run(["bash", "-c", "\n".join(lines)], capture_output=True, text=True, check=True).stdout
        # The first two CPUs of this process's cpuset, one place each (one if it has one CPU).
        self.assertRegex(out.strip(), r"^2 close \{\d+\}(,\{\d+\})?$")


class TestProbeCli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="nbt_opp_")
        self.addCleanup(shutil.rmtree, self.tmp)
        self.shapes = os.path.join(self.tmp, "shapes.tsv")
        with open(self.shapes, "w") as f:
            f.write(SHAPES)
        self.src = os.path.join(self.tmp, "measured.json")
        self.probe = os.path.join(self.tmp, "probe")
        with open(self.probe, "w") as f:
            f.write(FAKE_PROBE % {"src": self.src})
        os.chmod(self.probe, 0o755)

    def measured(self, doc):
        with open(self.src, "w") as f:
            json.dump(doc, f)

    def test_local_probe_status_and_export(self):
        wd = os.path.join(self.tmp, "syn")
        self.assertEqual(run("init", "--workdir", wd, "--shapes", self.shapes, "--m-classes", "1,2-16:8/16",
                             "--backend", "synthetic", "--noise", "0.002")[0], 0)
        self.measured(ceilings_doc())
        rc, out = run("probe", "--workdir", wd, "--probe-bin", self.probe)
        self.assertEqual(rc, 0, out)
        self.assertIn("probe on this host: exit 0", out)
        with open(os.path.join(wd, "ceilings.json")) as f:
            self.assertEqual(json.load(f), ceilings_doc())
        with open(os.path.join(wd, "campaign.json")) as f:
            camp = json.load(f)
        self.assertEqual((camp["probe_bin"], camp["ceilings"]["mismatch"]), (self.probe, []))
        self.assertEqual(run("search", "--workdir", wd, "--strategy", "random", "--budget", 4, "--batch", 4)[0], 0)
        rc, out = run("status", "--workdir", wd)
        self.assertIn("ceilings of 2026-01-01T00:00:00Z (32 threads, warm weights)", out)
        self.assertEqual(out.count("-bound, built-in"), 6)
        out_file = os.path.join(wd, "tuning.json")
        rc, out = run("export", "--workdir", wd, "-o", out_file, "--rules", "none", "--no-confirm")
        self.assertEqual(rc, 0, out)
        with open(out_file) as f:
            doc = json.load(f)
        self.assertEqual(E.validate_doc(doc, SCHEMA)[0], [])
        opm = doc["meta"]["op_perf"]
        self.assertEqual(len(opm["problems"]), 6)
        self.assertFalse(opm["cold_weights"])
        # Percentages and bound kinds only: no measured ceiling or time.
        self.assertIsNone(re.search(r"gbps|gflops|ghz|_ms|bandwidth|clock", json.dumps(opm)))
        with open(os.path.join(wd, "export_report.json")) as f:
            report = json.load(f)
        self.assertEqual(len(report["op_perf"]["rows"]), 6)
        self.assertTrue(all(r["attainable_ms"] > 0 for r in report["op_perf"]["rows"]))

    def test_refuses_another_machine_or_team(self):
        wd = os.path.join(self.tmp, "syn")
        self.assertEqual(run("init", "--workdir", wd, "--shapes", self.shapes, "--m-classes", "1",
                             "--backend", "synthetic")[0], 0)
        self.measured(ceilings_doc(16))
        rc, out = run("probe", "--workdir", wd, "--probe-bin", self.probe)
        self.assertEqual(rc, 1)
        self.assertIn("threads: probe 16, campaign 32", out)
        self.assertFalse(os.path.exists(os.path.join(wd, "ceilings.json")))
        for how in (["--probe-bin", self.probe], ["--ceilings", self.src]):
            rc, out = run("probe", "--workdir", wd, *how, "--force")
            self.assertEqual(rc, 0, out)
            with open(os.path.join(wd, "campaign.json")) as f:
                self.assertEqual(json.load(f)["ceilings"]["mismatch"], ["threads: probe 16, campaign 32"], how)
            os.remove(os.path.join(wd, "ceilings.json"))
        # A probe that wrote nothing.
        os.remove(self.src)
        rc, out = run("probe", "--workdir", wd)
        self.assertEqual(rc, 1)
        self.assertIn("wrote no ceilings: probe: failed", out)

    def test_pods_dry_run_then_collect(self):
        wd = os.path.join(self.tmp, "rp")
        rc, out = run("init", "--workdir", wd, "--shapes", self.shapes, "--m-classes", "1,2-16:16",
                      "--backend", "replay", "--replay-bin", "/opt/replay", "--pods", "pod-a,pod-b",
                      "--launcher", "/tools/run_one.sh --pod {pod} --cmd {dir} -- bash {script}", *MACHINE)
        self.assertEqual(rc, 0, out)
        rc, out = run("probe", "--workdir", wd, "--probe-bin", self.probe, "--dry-run", "--quick")
        self.assertEqual(rc, 0, out)
        self.assertEqual([line.split()[2] for line in out.splitlines() if "/tools/run_one.sh" in line],
                         ["pod-a", "pod-b"])
        with open(os.path.join(wd, "probe", "pod-a", "job.sh")) as f:
            job = f.read()
        self.assertIn("OMP_PLACES=", job)
        self.assertIn('--out "$S/ceilings.json" --threads 32 --quick', job)
        with self.assertRaises(SystemExit):
            run("probe", "--workdir", wd)
        # A probe still running has its output file and no exit status yet.
        open(os.path.join(wd, "probe", "pod-a", "probe.txt"), "w").close()
        rc, out = run("collect", "--workdir", wd)
        self.assertEqual(out.count("the probe has not finished"), 2, out)
        self.assertNotIn("installed the ceilings", out)
        # pod-a's job ran, pod-b's failed, as the launcher would leave them.
        self.measured(ceilings_doc())
        subprocess.call(["bash", os.path.join(wd, "probe", "pod-a", "job.sh")], stdout=subprocess.DEVNULL)
        os.remove(self.src)
        subprocess.call(["bash", os.path.join(wd, "probe", "pod-b", "job.sh")], stdout=subprocess.DEVNULL)
        rc, out = run("collect", "--workdir", wd)
        self.assertIn("pod-b wrote no ceilings: probe: failed", out)
        self.assertIn("installed the ceilings of %s" % os.path.join(wd, "probe", "pod-a", "ceilings.json"), out)
        camp = tune.Campaign(wd)
        self.assertTrue(camp.cold)
        self.assertEqual({b.level for b in camp.bounds.values()}, {"dram"})
        self.assertEqual(run("collect", "--workdir", wd)[1].strip(), "no pending session")

    def test_launched_through_the_launcher(self):
        wd = os.path.join(self.tmp, "launch")
        self.assertEqual(run("init", "--workdir", wd, "--shapes", self.shapes, "--m-classes", "1",
                             "--backend", "benchdnn", "--benchdnn", "/opt/benchdnn",
                             "--launcher", "bash {script}", *MACHINE)[0], 0)
        self.measured(ceilings_doc())
        rc, out = run("probe", "--workdir", wd, "--probe-bin", self.probe)
        self.assertEqual(rc, 0, out)
        self.assertIn("probe pod: launcher exit 0", out)
        self.assertTrue(os.path.exists(os.path.join(wd, "ceilings.json")))


if __name__ == "__main__":
    unittest.main()
