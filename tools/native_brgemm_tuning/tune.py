#!/usr/bin/env python3
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
"""Native BRGEMM tuner CLI: shapes and M classes -> problems -> search ->
results store -> confirmation -> tuning file.  See README.md.

  tune.py init     --workdir D --shapes S.tsv --backend replay --replay-bin B ...
  tune.py search   --workdir D [--strategy guided --budget 32 --batch 8] [--dry-run]
  tune.py collect  --workdir D            (sessions run by hand after --dry-run)
  tune.py confirm  --workdir D
  tune.py export   --workdir D -o tuning.json
  tune.py check    tuning.json [--workdir D | --problem dtype=s8,m=1,n=4096,k=4096]
  tune.py verify   tuning.json --workdir D   (the file against the built-in plan on a pod)
  tune.py merge    -o tuning.json a.json b.json ...   (the entries of several campaigns' files)
  tune.py status   --workdir D
  tune.py schema | problems | space   (inspection)
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from typing import Any, Dict, List, Optional

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from native_brgemm_tuning import candidates as cand_mod  # noqa: E402
from native_brgemm_tuning import export as exp  # noqa: E402
from native_brgemm_tuning import problems as prob_mod  # noqa: E402
from native_brgemm_tuning import results as res_mod  # noqa: E402
from native_brgemm_tuning import schema as schema_mod  # noqa: E402
from native_brgemm_tuning.backends import BACKENDS  # noqa: E402
from native_brgemm_tuning.backends import jobs  # noqa: E402
from native_brgemm_tuning.plan import plan_log_fields, resolve  # noqa: E402
from native_brgemm_tuning.problems import Problem  # noqa: E402
from native_brgemm_tuning.search import Searcher, SearchSettings  # noqa: E402

CAMPAIGN = "campaign.json"
RESULTS = "results.jsonl"


class Campaign:
    """The settings of one tuning campaign (workdir/campaign.json) and what they build."""

    def __init__(self, workdir: str):
        self.workdir = os.path.abspath(workdir)
        path = os.path.join(self.workdir, CAMPAIGN)
        if not os.path.exists(path):
            sys.exit("no %s in %s: run tune.py init first" % (CAMPAIGN, self.workdir))
        with open(path, encoding="utf-8") as f:
            self.c = json.load(f)
        self.schema = schema_mod.load_schema(self.c.get("schema"), self.c.get("uarch") or None)
        if self.schema.uarch is None or not self.schema.uarch.complete:
            sys.exit("the schema lacks uarch fields: rerun init with --uarch family_l3_cores=<n>[,l2_bytes=..]")
        self.uarch = self.schema.uarch
        self.problems = [Problem.from_dict(d) for d in self.c["problems"]]
        self.by_key = {p.key(): p for p in self.problems}
        self.classes = prob_mod.parse_m_classes(self.c["m_classes"])
        self.db = res_mod.ResultsDB(os.path.join(self.workdir, RESULTS))
        self._spaces: Optional[Dict[str, cand_mod.Space]] = None

    @property
    def spaces(self) -> Dict[str, cand_mod.Space]:
        if self._spaces is None:
            values = cand_mod.parse_values(self.c.get("values") or [], self.schema)
            self._spaces = {}
            for p in self.problems:
                keys = cand_mod.search_keys(self.schema, p.dtype, self.c["allow_bit_changing"],
                                            self.c.get("keys") or None)
                self._spaces[p.key()] = cand_mod.Space(p, self.schema, self.uarch, keys, values,
                                                       self.c["allow_bit_changing"],
                                                       self.c.get("keep_threads", False))
        return self._spaces

    def backend(self) -> Any:
        return jobs.make_backend(self.c["backend"])

    def evaluator(self, repeats: int, dry_run: bool) -> Any:
        be = self.backend()
        defaults = {k: s.default for k, s in self.spaces.items()}
        if not be.runs_on_pod:
            return be.evaluator(self.by_key, defaults, repeats)
        return jobs.PodEvaluator(be, self.workdir, self.by_key, defaults, self.c.get("launcher"), repeats,
                                 self.c.get("parallel", 1), self.c.get("timeout_s", 7200), self.schema,
                                 dry_run, log=log, pods=self.c.get("pods") or ())


def log(msg: str) -> None:
    print(msg, flush=True)


def _write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")


def _write_tuning_file(path: str, doc: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(exp.file_bytes(doc))


def _pods(text: Optional[str]) -> List[str]:
    return [p.strip() for p in (text or "").split(",") if p.strip()]


def _no_pending(camp: Campaign, force: bool) -> None:
    pend = jobs.pending_sessions(camp.workdir)
    if pend and not force:
        sys.exit("sessions not collected yet (run them, then tune.py collect):\n  " + "\n  ".join(pend))


def _report_dry_run(e: jobs.DryRun, camp: Campaign) -> None:
    log("sessions written, not launched:")
    for cmd in e.commands:
        log("  " + cmd)
    log("after they ran: tune.py collect --workdir %s, then rerun this command" % camp.workdir)


# ---------------------------------------------------------------- commands

def cmd_schema(a: argparse.Namespace) -> int:
    if a.extract:
        with open(a.extract, encoding="utf-8") as f:
            doc = json.loads(schema_mod.extract_schema_json(f.read()))
        out = a.output or "schema.json"
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(schema_mod.schema_text(doc))
        log("wrote %s" % (a.output or "schema.json"))
        return 0
    s = schema_mod.load_schema(a.schema, a.uarch)
    log(s.summary())
    if a.compare:
        diff = schema_mod.schema_diff(s, schema_mod.load_schema(a.compare))
        log("\n".join(diff) if diff else "no differences in keys and kernels")
    return 0


def _read_problems(a: argparse.Namespace, threads: int) -> List[Problem]:
    out: List[Problem] = []
    if a.shapes:
        dtypes = a.dtypes.split(",") if a.dtypes else None
        shapes = prob_mod.read_shapes_tsv(a.shapes, dtypes, min_calls=a.min_calls, group=a.s4_group)
        # An s4 weight's scales take the bytes of the backend's scale type.
        scale_bytes = 4 if getattr(a, "scale_dt", "bf16") == "f32" else 2
        out += prob_mod.problems_from_shapes(shapes, prob_mod.parse_m_classes(a.m_classes), threads=threads,
                                             pack_nr=a.pack_nr, scale_bytes=scale_bytes)
    if a.problems_file:
        with open(a.problems_file, encoding="utf-8") as f:
            out += [Problem.from_dict(d) for d in json.load(f)]
    for text in a.problem or []:
        p = Problem.parse(text)
        out.append(p if "threads=" in text else Problem.from_dict(dict(p.to_dict(), threads=threads)))
    return out


def cmd_problems(a: argparse.Namespace) -> int:
    probs = _read_problems(a, a.threads)
    for p in probs:
        log(p.key())
    if a.output:
        _write_json(a.output, [p.to_dict() for p in probs])
    log("%d problems" % len(probs))
    return 0


def cmd_space(a: argparse.Namespace) -> int:
    s = schema_mod.load_schema(a.schema, a.uarch)
    p = Problem.parse(a.problem)
    keys = cand_mod.search_keys(s, p.dtype, a.allow_bit_changing, a.keys.split(",") if a.keys else None)
    sp = cand_mod.Space(p, s, s.uarch, keys, cand_mod.parse_values(a.values or [], s), a.allow_bit_changing,
                        a.keep_threads)
    log("problem %s\nbuilt-in plan: %s" % (p.label(), plan_log_fields(sp.default.plan, p)))
    for k in sp.keys:
        log("  %-20s %s" % (k, "/".join(s.key(k).format(v) for v in sp.choices[k])))
    log("grid size before legality and de-duplication: %.3g" % sp.size())
    shown = 0
    for c in cand_mod.unique(sp.iter_grid(a.max_keys), sp):
        if shown >= a.list:
            break
        log("  %s" % c.text)
        shown += 1
    return 0


def _cold_mb(uarch: schema_mod.Uarch, threads: int) -> float:
    """Eight times the L3 of the domains a team of threads spans."""
    domain = uarch.plan_domain or uarch.num_cores or 1
    return 8.0 * uarch.l3_bytes * -(-threads // domain) / 1e6


def cmd_init(a: argparse.Namespace) -> int:
    wd = os.path.abspath(a.workdir)
    if os.path.exists(os.path.join(wd, CAMPAIGN)) and not a.force:
        sys.exit("%s exists (use --force to replace it)" % os.path.join(wd, CAMPAIGN))
    s = schema_mod.load_schema(a.schema, a.uarch)
    if s.uarch is None or not s.uarch.complete:
        sys.exit("the schema lacks uarch fields: pass --uarch family_l3_cores=<n>[,l2_bytes=..]")
    if a.backend == "replay":
        if not a.replay_bin:
            sys.exit("--backend replay needs --replay-bin (built with replay/build.sh)")
        be: Dict[str, Any] = {"backend": "replay", "binary": os.path.abspath(a.replay_bin), "threads": a.threads,
                              "preload": a.preload or "", "layers": a.layers, "iters": a.iters,
                              "warmup": a.warmup, "budget_ms": a.budget_ms, "scale_dt": a.scale_dt,
                              "group_cost": a.group_cost, "group_gb": a.group_gb,
                              "cold_mb": a.cold_mb if a.cold_mb >= 0 else _cold_mb(s.uarch, a.threads)}
    elif a.backend == "benchdnn":
        if not a.benchdnn:
            sys.exit("--backend benchdnn needs --benchdnn <binary>")
        be = {"backend": "benchdnn", "binary": os.path.abspath(a.benchdnn), "threads": a.threads,
              "per_line": not a.whole_file, "iters_scale": a.iters_scale}
    else:
        be = {"backend": "synthetic", "threads": a.threads, "noise": a.noise, "seed": a.seed}
    try:
        backend = jobs.make_backend(be)
    except ValueError as e:
        sys.exit("%s backend: %s" % (a.backend, e))
    probs, skipped = [], []
    for p in _read_problems(a, a.threads):
        why = backend.supports(p)
        (skipped if why else probs).append((p, why) if why else p)
    if not probs:
        sys.exit("no problem the %s backend can time%s" % (a.backend, "".join(
            "\n  %s: %s" % (p.label(), why) for p, why in skipped[:5])))
    if a.keys:
        for k in a.keys.split(","):
            s.key(k)
    cand_mod.parse_values(a.values or [], s)
    camp = {
        "schema": os.path.abspath(a.schema) if a.schema else None, "uarch": a.uarch or "",
        "backend": be, "problems": [p.to_dict() for p in probs], "m_classes": a.m_classes,
        "keys": a.keys.split(",") if a.keys else None, "values": a.values or [],
        "allow_bit_changing": a.allow_bit_changing, "keep_threads": a.keep_threads,
        "launcher": a.launcher, "pods": _pods(a.pods), "repeats": a.repeats,
        "confirm_repeats": a.confirm_repeats, "parallel": a.parallel, "timeout_s": a.timeout,
        "min_gain": a.min_gain, "z": a.z,
    }
    _write_json(os.path.join(wd, CAMPAIGN), camp)
    for p, why in skipped:
        log("skipped %s: %s" % (p.label(), why))
    log("wrote %s: %d problems (%d skipped), backend %s, schema %s" % (
        os.path.join(wd, CAMPAIGN), len(probs), len(skipped), a.backend, s.source))
    if a.backend != "synthetic" and not a.launcher:
        log("no --launcher: search / confirm will write sessions and stop (dry run)")
    return 0


def cmd_search(a: argparse.Namespace) -> int:
    camp = Campaign(a.workdir)
    _no_pending(camp, a.ignore_pending)
    # Overrides for this search only (campaign.json keeps its keys and values).
    if a.keys:
        for k in a.keys.split(","):
            camp.schema.key(k)
        camp.c["keys"] = a.keys.split(",")
    if a.values:
        cand_mod.parse_values(a.values, camp.schema)
        camp.c["values"] = a.values
    if a.parallel:
        camp.c["parallel"] = a.parallel
    if a.pods is not None:
        camp.c["pods"] = _pods(a.pods)
    probs = camp.problems
    if a.ms:
        want = {int(m) for m in a.ms.split(",")}
        probs = [p for p in probs if p.m in want]
        if not probs:
            sys.exit("no problem of the campaign has M in %s" % a.ms)
    st = SearchSettings(strategy=a.strategy, budget=a.budget, batch=a.batch, max_keys=a.max_keys,
                        p_change=a.p_change, init=a.init, pool=a.pool, explore=a.explore, kappa=a.kappa,
                        surrogate=a.surrogate, seed=a.seed, z=camp.c["z"], seeds=a.seeds,
                        min_gain=camp.c["min_gain"])
    ev = camp.evaluator(camp.c["repeats"], a.dry_run)
    searcher = Searcher(probs, camp.spaces, ev, camp.db, st, log=log)
    try:
        n = searcher.run(a.rounds)
    except jobs.DryRun as e:
        _report_dry_run(e, camp)
        return 0
    log("%d round(s) done; results in %s" % (n, camp.db.path))
    return cmd_status(a)


def cmd_collect(a: argparse.Namespace) -> int:
    camp = Campaign(a.workdir)
    dirs = a.sessions or jobs.pending_sessions(camp.workdir)
    if not dirs:
        log("no pending session")
        return 0
    be = camp.backend()
    for d in dirs:
        out_dir = os.path.join(d, "out")
        if not (os.path.isdir(out_dir) and os.listdir(out_dir)) and not a.force:
            log("skipped %s: no outputs yet (did the job run? --force collects it as failed)" % d)
            continue
        recs = jobs.collect(d, be)
        sid = os.path.basename(os.path.normpath(d))
        for r in recs:
            if sid[1:4].isdigit():
                r["round"] = int(sid[1:4])
        camp.db.append(recs)
        jobs.mark_collected(d, len(recs))
        bad = sum(1 for r in recs if r["kind"] == "probe" and res_mod.probe_verdict(r)[0] != res_mod.APPLIED)
        log("collected %s: %d records, %d probes not applied" % (d, len(recs), bad))
    return 0


def _winners(camp: Campaign) -> List[exp.Winner]:
    return exp.select_winners(camp.db.load(), camp.problems, camp.classes, camp.spaces, camp.c["min_gain"],
                              camp.c["z"])


def _next_round(camp: Campaign, phase: str) -> int:
    rounds = [int(r.get("round", 0)) for r in camp.db.load() if r.get("phase") == phase]
    return max(rounds) + 1 if rounds else 1


def cmd_confirm(a: argparse.Namespace) -> int:
    camp = Campaign(a.workdir)
    _no_pending(camp, a.ignore_pending)
    if a.parallel:
        camp.c["parallel"] = a.parallel
    if a.pods is not None:
        camp.c["pods"] = _pods(a.pods)
    cross = exp.cross_batch(camp.db.load(), camp.problems, camp.classes, camp.spaces, a.top,
                            camp.c["min_gain"], camp.c["z"])
    if cross:
        rnd = _next_round(camp, "search")
        log("round %d: the best candidates of each M at the other Ms of their class (%d candidates)" % (
            rnd, sum(len(v) for v in cross.values())))
        try:
            recs = camp.evaluator(camp.c["repeats"], a.dry_run)(rnd, "search", cross)
        except jobs.DryRun as e:
            _report_dry_run(e, camp)
            return 0
        for r in recs:
            r["round"] = rnd
        camp.db.append(recs)
    winners = _winners(camp)
    if a.fallback:
        winners = exp.fallback_winners(winners, camp.db.load(), camp.spaces, camp.c["min_gain"], camp.c["z"])
    elif a.only_new:
        recs = camp.db.load()
        winners = [w for w in winners if not exp.confirmed_before(w, recs, camp.spaces, camp.c["z"])]
    batch = exp.confirm_batch(winners, camp.spaces)
    if not batch:
        log("no winner beats the built-in plan beyond the noise; nothing to confirm")
        return 0
    rnd = _next_round(camp, "confirm")
    try:
        recs = camp.evaluator(camp.c["confirm_repeats"], a.dry_run)(rnd, "confirm", batch)
    except jobs.DryRun as e:
        _report_dry_run(e, camp)
        return 0
    for r in recs:
        r["round"] = rnd
    camp.db.append(recs)
    log("confirmation round %d: %d candidates" % (rnd, sum(len(v) for v in batch.values())))
    return 0


def cmd_export(a: argparse.Namespace) -> int:
    camp = Campaign(a.workdir)
    recs = camp.db.load()
    winners = _winners(camp)
    exp.attach_confirmation(winners, recs, camp.spaces, camp.c["min_gain"], camp.c["z"])
    entries, skipped = exp.build_entries(winners, camp.spaces, camp.schema, camp.uarch,
                                         a.allow_bit_changing, not a.no_confirm)
    if a.exclude:
        entries, dropped = exp.exclude_entries(entries, a.exclude)
        skipped += dropped
    rules, rules_report = None, {"skipped": "not requested"}
    if a.rules != "none":
        from native_brgemm_tuning.problems import rule_features
        from native_brgemm_tuning.rules import train_rules

        summ = res_mod.summarize(recs, phases=["search"], z=camp.c["z"])
        feats = [f for f in camp.schema.features if f in rule_features(camp.problems[0], camp.uarch, camp.schema.hw)]

        def config_of(text: str) -> Dict[str, Any]:
            return exp.config_json(cand_mod.parse_settings(text, camp.schema), camp.schema) if text else {}

        rules, rules_report = train_rules(camp.problems, summ, feats,
                                          lambda p: rule_features(p, camp.uarch, camp.schema.hw), config_of,
                                          kind=a.rules, max_depth=a.max_depth, min_leaf=a.min_leaf, folds=a.folds,
                                          min_gain=camp.c["min_gain"])
        if rules is not None and not a.allow_bit_changing:
            texts = [t for t in rules_report["classes"] if t]
            bad = [t for t in texts if any(not camp.schema.key(k).same_bits(d)
                                           for k in cand_mod.parse_settings(t, camp.schema) for d in prob_mod.DTYPES
                                           if camp.schema.key(k).applies_to(d))]
            if bad:
                rules, rules_report["skipped"] = None, "classes with bit-changing keys: %s" % bad
    hw_keys = [k.strip() for k in a.hw_keys.split(",") if k.strip()] if a.hw_keys is not None else None
    try:
        hw_values = exp.parse_hw_set(a.hw_set or [])
        # The plans the entries name, which tune.py verify checks against the library's lines.
        plans = exp.entry_plans(entries, camp.problems, camp.schema, camp.uarch)
        doc = exp.build_doc(entries, rules, camp.schema, a.hw_match,
                            {"backend": camp.c["backend"]["backend"], "m_classes": camp.c["m_classes"],
                             "min_gain": camp.c["min_gain"], "z": camp.c["z"], "rules_report": rules_report,
                             "plans": plans},
                            hw_keys, hw_values)
    except ValueError as e:
        sys.exit("export: %s" % e)
    errs, warns = exp.validate_doc(doc, camp.schema)
    if errs:
        sys.exit("the exported file fails validation:\n  " + "\n  ".join(errs))
    _write_tuning_file(a.output, doc)
    report = {"entries": len(entries), "skipped": skipped, "winners": [w.to_dict() for w in winners],
              "rules": rules_report, "warnings": warns}
    _write_json(os.path.join(camp.workdir, "export_report.json"), report)
    log("wrote %s: %d entries, rules %s; %d classes skipped (export_report.json)" % (
        a.output, len(entries), rules["kind"] if rules else "none (%s)" % rules_report.get("skipped"), len(skipped)))
    if rules is not None:
        ho = rules_report["heldout"]
        if ho["measured"]:
            log("rules held out: geomean ratio %.3f (oracle %.3f) over %d measured problems, %d slower than "
                "built-in" % (ho["geomean_ratio"], ho["oracle_geomean_ratio"], ho["measured"],
                              ho["slower_than_builtin"]))
        else:
            log("rules held out: no measured problem (%d shapes in %d folds), so no held-out ratio" % (
                ho["shapes"], ho["folds"]))
    return 0


def cmd_check(a: argparse.Namespace) -> int:
    with open(a.file, encoding="utf-8") as f:
        doc = json.load(f)
    camp = Campaign(a.workdir) if a.workdir else None
    s = camp.schema if camp else schema_mod.load_schema(a.schema, a.uarch)
    errs, warns = exp.validate_doc(doc, s)
    for w in warns:
        log("warning: " + w)
    for e in errs:
        log("error: " + e)
    mism = exp.hardware_mismatch(doc.get("hardware", {}), s)
    if mism:
        log("hardware key %s does not hold on this machine (%s): the loader ignores the file there" % (
            mism, json.dumps(s.hw_match[mism]) if mism in s.hw_match else "not reported"))
    probs = (camp.problems if camp else []) + [Problem.parse(t) for t in a.problem or []]
    for p in probs:
        cfg, layers = exp.resolve_config(doc, p, s, s.uarch)
        if cfg is None:
            log("%s: hardware mismatch" % p.label())
            continue
        conf = exp.config_from_json(cfg, s)
        r = resolve(p, conf, s.uarch, s)
        log("%s: %s -> %s%s" % (p.label(), "+".join(layers) or "built-in", cand_mod.format_settings(conf, s) or "-",
                                "" if r.applied else "  (illegal plan, merged layers dropped: %s)" % r.reason))
    return 1 if errs else 0


def cmd_verify(a: argparse.Namespace) -> int:
    """Runs a tuning file the tuner wrote against the built-in plan on a pod (or reads such a run)."""
    camp = Campaign(a.workdir)
    with open(a.file, encoding="utf-8") as f:
        doc = json.load(f)
    plans = doc.get("meta", {}).get("plans")
    if plans is None:
        sys.exit("%s names no plans (meta.plans): export it with this tool first" % a.file)
    sdir = os.path.abspath(a.dir or os.path.join(camp.workdir, "verify"))
    be = camp.backend()
    if not os.path.exists(os.path.join(sdir, jobs.VERIFY_MANIFEST)):
        probs = [p for p in camp.problems if not a.ms or p.m in {int(m) for m in a.ms.split(",")}]
        script = jobs.write_verify(sdir, be, probs, a.file)
        launcher = camp.c.get("launcher")
        if a.dry_run or not launcher:
            log("verification written, not launched: <launcher> %s" % script)
            return 0
        pods = camp.c.get("pods") or [""]
        cmd = launcher.format(dir=shlex.quote(sdir), script=shlex.quote(script), pod=shlex.quote(pods[0]))
        env = dict(os.environ, TIMEOUT=str(camp.c.get("timeout_s", 7200)), THREADS=str(be.threads),
                   LOG=os.path.join(sdir, "launcher.log"))
        if pods[0]:
            env["POD"] = pods[0]
        rc = subprocess.call(shlex.split(cmd), env=env)
        log("verification run: launcher exit %d" % rc)
    report = jobs.collect_verify(sdir, plans, be)
    _write_json(os.path.join(sdir, "verify_report.json"), report)
    for r in report["rows"]:
        if not r["ok"]:
            log("  %s: %s" % (r["problem"], "; ".join(r["why"])))
    log("%d problems, %d named by entries (%d of them the built-in plan), %d failed: %s" % (
        report["problems"], report["named"], report["named_builtin"], report["failed"],
        os.path.join(sdir, "verify_report.json")))
    return 1 if report["failed"] else 0


def cmd_merge(a: argparse.Namespace) -> int:
    s = schema_mod.load_schema(a.schema, a.uarch)
    docs = []
    for path in a.files:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        errs, _ = exp.validate_doc(doc, s)
        if errs:
            sys.exit("%s fails validation:\n  %s" % (path, "\n  ".join(errs)))
        docs.append((os.path.abspath(path), doc))
    try:
        doc = exp.merge_docs(docs)
    except ValueError as e:
        sys.exit("merge: %s" % e)
    errs, _ = exp.validate_doc(doc, s)
    if errs:
        sys.exit("the merged file fails validation:\n  " + "\n  ".join(errs))
    _write_tuning_file(a.output, doc)
    log("wrote %s: %d entries from %d files" % (a.output, len(doc.get("entries", [])), len(docs)))
    return 0


def cmd_status(a: argparse.Namespace) -> int:
    camp = Campaign(a.workdir)
    recs = camp.db.load()
    summ = res_mod.summarize(recs, phases=["search"], z=camp.c["z"])
    status = res_mod.probe_status(recs)
    bad: Dict[str, int] = {}
    for st, _ in status.values():
        if st != res_mod.APPLIED:
            bad[st] = bad.get(st, 0) + 1
    log("%d problems, %d with results; probes not applied: %s; pending sessions: %d" % (
        len(camp.problems), len(summ), bad or "none", len(jobs.pending_sessions(camp.workdir))))
    for k, ps in sorted(summ.items(), key=lambda kv: kv[0]):
        b = ps.best(camp.c["min_gain"], significant_only=True)
        p = camp.by_key.get(k)
        log("  %-40s built-in %.4f ms (noise %.3f), %3d candidates, best %s" % (
            p.label() if p else k[:40], ps.default_ms, ps.default_noise, len(ps.cands),
            "%s (%.3f)" % (b.text, b.ratio) if b else "-"))
    return 0


# ---------------------------------------------------------------- arguments

def _common_schema(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--schema", help="SchemaDump JSON or gtest log (default: fixtures/schema_dump.json)")
    sp.add_argument("--uarch", help="UarchParams overrides, e.g. family_l3_cores=8,l2_bytes=1048576")


def _common_problems(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--shapes", help="TSV with model, name, N, K, wei, calls, m_lo, m_hi")
    sp.add_argument("--problems-file", help="JSON list of problems")
    sp.add_argument("--problem", action="append", help="dtype=s8,m=1,n=4096,k=4096[,...] (repeatable)")
    sp.add_argument("--dtypes", help="weight types to keep, e.g. s8 or bf16,s8,s4,f16")
    sp.add_argument("--s4-group", type=int, default=128,
                    help="group size along K of the s4 shapes whose TSV row names none (-1: one group over K)")
    sp.add_argument("--m-classes", default=prob_mod.DEFAULT_M_CLASSES, help="lo[-[hi]][:rep/rep],...")
    sp.add_argument("--min-calls", type=int, default=0)
    sp.add_argument("--pack-nr", type=int, default=32)
    sp.add_argument("--threads", type=int, default=32)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("schema", help="show, compare or extract the schema")
    _common_schema(sp)
    sp.add_argument("--compare", help="another schema to diff against")
    sp.add_argument("--extract", help="gtest log holding the SchemaDump output")
    sp.add_argument("-o", "--output")
    sp.set_defaults(fn=cmd_schema)

    sp = sub.add_parser("problems", help="list the problems of a shape list and M classes")
    _common_problems(sp)
    sp.add_argument("-o", "--output")
    sp.set_defaults(fn=cmd_problems)

    sp = sub.add_parser("space", help="show one problem's candidate space")
    _common_schema(sp)
    sp.add_argument("--problem", required=True)
    sp.add_argument("--keys")
    sp.add_argument("--values", action="append", help="key=a|b (repeatable)")
    sp.add_argument("--allow-bit-changing", action="store_true")
    sp.add_argument("--keep-threads", action="store_true", help="no plan with fewer threads than the built-in one")
    sp.add_argument("--max-keys", type=int, default=1)
    sp.add_argument("--list", type=int, default=40)
    sp.set_defaults(fn=cmd_space)

    sp = sub.add_parser("init", help="create a campaign")
    sp.add_argument("--workdir", required=True)
    sp.add_argument("--force", action="store_true")
    _common_schema(sp)
    _common_problems(sp)
    sp.add_argument("--backend", choices=sorted(BACKENDS), required=True)
    sp.add_argument("--replay-bin")
    sp.add_argument("--preload", help="replay: colon-separated libraries to preload (tcmalloc, libiomp5)")
    sp.add_argument("--layers", type=int, default=4)
    sp.add_argument("--iters", type=int, default=30)
    sp.add_argument("--warmup", type=int, default=2)
    sp.add_argument("--budget-ms", type=int, default=1500)
    sp.add_argument("--scale-dt", default="bf16")
    sp.add_argument("--group-cost", type=float, default=0.0,
                    help="replay: most estimated seconds per process run of a group, at the layers it times "
                         "(0: no cap)")
    sp.add_argument("--group-gb", type=float, default=0.0,
                    help="replay: most weight GB one process allocates, every layer it times (0: no cap; "
                         "at least --cold-mb; a problem whose process alone is above it is skipped)")
    sp.add_argument("--cold-mb", type=float, default=-1.0,
                    help="replay: least MB of distinct weights a timed process cycles through, raising its layers "
                         "(default: eight times the L3 of the domains the team spans; 0: --layers as given)")
    sp.add_argument("--benchdnn")
    sp.add_argument("--whole-file", action="store_true", help="benchdnn: all lines of a run in one process")
    sp.add_argument("--iters-scale", type=float, default=1.0)
    sp.add_argument("--noise", type=float, default=0.02, help="synthetic: log-normal noise")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--keys", help="searched keys (default: every key that keeps the bits)")
    sp.add_argument("--values", action="append", help="key=a|b (repeatable)")
    sp.add_argument("--allow-bit-changing", action="store_true",
                    help="search keys that change output bits (entries then need a model-level accuracy check)")
    sp.add_argument("--keep-threads", action="store_true",
                    help="leave out plans with fewer threads than the built-in plan (a smaller team changes the "
                         "thread team of the calls around it)")
    sp.add_argument("--launcher", help="e.g. '<tools>/run_one.sh --cmd {dir} -- bash {script}'")
    sp.add_argument("--pods", help="pods of the sessions of a round, comma separated: session i runs with "
                                   "POD=<i-th pod> ({pod} in --launcher), at most one session per pod")
    sp.add_argument("--repeats", type=int, default=2)
    sp.add_argument("--confirm-repeats", type=int, default=4)
    sp.add_argument("--parallel", type=int, default=1, help="sessions (pod claims) per round")
    sp.add_argument("--timeout", type=int, default=7200, help="seconds per session")
    sp.add_argument("--min-gain", type=float, default=0.02)
    sp.add_argument("--z", type=float, default=2.0)
    sp.set_defaults(fn=cmd_init)

    sp = sub.add_parser("search", help="run search rounds")
    sp.add_argument("--workdir", required=True)
    sp.add_argument("--strategy", choices=("grid", "random", "guided", "local"), default="guided")
    sp.add_argument("--seeds", type=int, default=2,
                    help="local: best significant candidates per problem whose neighbours are proposed")
    sp.add_argument("--budget", type=int, default=32)
    sp.add_argument("--batch", type=int, default=8)
    sp.add_argument("--rounds", type=int)
    sp.add_argument("--max-keys", type=int)
    sp.add_argument("--p-change", type=float)
    sp.add_argument("--init", type=int, default=8)
    sp.add_argument("--pool", type=int, default=400)
    sp.add_argument("--explore", type=float, default=0.25)
    sp.add_argument("--kappa", type=float, default=1.0)
    sp.add_argument("--surrogate", choices=("auto", "gbt", "forest", "knn"), default="auto")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--ms", help="search only the problems at these Ms, e.g. 16,32")
    sp.add_argument("--keys", help="searched keys and their order for this search (default: the campaign's)")
    sp.add_argument("--values", action="append", help="key=a|b for this search (repeatable)")
    sp.add_argument("--parallel", type=int, help="sessions of this search (default: the campaign's)")
    sp.add_argument("--pods", help="pods of this search's sessions (default: the campaign's)")
    sp.add_argument("--dry-run", action="store_true", help="write the sessions and stop")
    sp.add_argument("--ignore-pending", action="store_true")
    sp.set_defaults(fn=cmd_search)

    sp = sub.add_parser("collect", help="read sessions run by hand into the results store")
    sp.add_argument("--workdir", required=True)
    sp.add_argument("--force", action="store_true", help="collect sessions without outputs (as failed)")
    sp.add_argument("sessions", nargs="*")
    sp.set_defaults(fn=cmd_collect)

    sp = sub.add_parser("confirm", help="re-run every class winner against the built-in plan")
    sp.add_argument("--workdir", required=True)
    sp.add_argument("--top", type=int, default=3,
                    help="best candidates of each M first measured at the other Ms of its class")
    sp.add_argument("--only-new", action="store_true",
                    help="confirm only winners without confirmation results yet")
    sp.add_argument("--fallback", action="store_true",
                    help="for winners no confirmed candidate covers: confirm their best alternative not run yet")
    sp.add_argument("--parallel", type=int, help="sessions of each round (default: the campaign's)")
    sp.add_argument("--pods", help="pods of the sessions (default: the campaign's)")
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--ignore-pending", action="store_true")
    sp.set_defaults(fn=cmd_confirm)

    sp = sub.add_parser("export", help="write the tuning file")
    sp.add_argument("--workdir", required=True)
    sp.add_argument("-o", "--output", required=True)
    sp.add_argument("--rules", choices=("tree", "ordered", "none"), default="tree")
    sp.add_argument("--max-depth", type=int, default=4)
    sp.add_argument("--min-leaf", type=int, default=3)
    sp.add_argument("--folds", type=int, default=5)
    sp.add_argument("--hw-match", choices=("fields", "none"),
                    help="hardware block: fields (the default) or none (a file for any CPU)")
    sp.add_argument("--hw-keys", help="hardware keys to write (default %s)" % ",".join(
        exp.HW_DEFAULT_KEYS))
    sp.add_argument("--hw-set", action="append",
                    help="key=value written instead of the machine's value, e.g. model=0-15, "
                         "l3_kib=32768|131072, cores=64-, vendor=A|B, isa=avx512bf16+avx512vnni (repeatable)")
    sp.add_argument("--no-confirm", action="store_true", help="export winners without a confirmation run")
    sp.add_argument("--allow-bit-changing", action="store_true")
    sp.add_argument("--exclude", action="append",
                    help="entry ids to leave out (comma separated, or @file with one id per line; repeatable)")
    sp.set_defaults(fn=cmd_export)

    sp = sub.add_parser("check", help="validate a tuning file and show what it gives problems")
    sp.add_argument("file")
    sp.add_argument("--workdir")
    _common_schema(sp)
    sp.add_argument("--problem", action="append")
    sp.set_defaults(fn=cmd_check)

    sp = sub.add_parser("verify", help="run a tuning file against the built-in plan on a pod and check its plans")
    sp.add_argument("file")
    sp.add_argument("--workdir", required=True)
    sp.add_argument("--dir", help="the verification's directory (default <workdir>/verify)")
    sp.add_argument("--ms", help="verify only the problems at these Ms")
    sp.add_argument("--dry-run", action="store_true", help="write the job and stop")
    sp.set_defaults(fn=cmd_verify)

    sp = sub.add_parser("merge", help="one tuning file of the entries of several files")
    sp.add_argument("files", nargs="+")
    sp.add_argument("-o", "--output", required=True)
    _common_schema(sp)
    sp.set_defaults(fn=cmd_merge)

    sp = sub.add_parser("status", help="results so far")
    sp.add_argument("--workdir", required=True)
    sp.set_defaults(fn=cmd_status)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return a.fn(a)
    except BrokenPipeError:
        devnull = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(devnull, sys.stdout.fileno())
        finally:
            os.close(devnull)
        return 0


if __name__ == "__main__":
    sys.exit(main())
