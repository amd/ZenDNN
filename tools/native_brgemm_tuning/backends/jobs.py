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
"""Pod sessions: many candidates per pod claim.

A Session takes candidates per problem and writes one directory:

  job.sh         the pod-side script, run as <launcher> (e.g.
                 run_one.sh --cmd <dir> -- bash <dir>/job.sh)
  manifest.json  problems, slots, runs and the plan each probe should log
  cfg/           one ZENDNNL_MATMUL_NATIVE_CONFIG value per (group, slot)
  inputs/        backend input files
  out/           one output file per run (written on the pod)

Problems are split into groups (one process each, see Backend.groups).  Slot 0
of a group is the built-in plan (no override); slot j holds the j-th candidate
of each problem as one clause per problem, so one process times many problems.
Runs: one probe per (group, slot) with ZENDNNL_API_LOG_LEVEL=4 (the source,
cfg and plan each call used, and the replay output hash), then the timed runs
with the log off, per repeat A S1 .. Sk A (slots reversed on odd repeats).
collect() turns the outputs into result records.

write_probe() writes a job that measures the machine's ceilings with the team of
the backend's timed runs (tools/op_perf_utility/probe), and write_verify() one
that runs a tuning file against the built-in plan; both go through the same
launcher.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..candidates import Candidate, format_override, problem_filters
from ..plan import compare_fields, compare_pairing, log_pairing, paired_log_fields, plan_log_fields
from ..problems import Problem
from ..results import utc_now
from .common import Backend, applied_status, parse_warnings

MANIFEST = "manifest.json"
SCRIPT = "job.sh"
COLLECTED = "collected.json"


def make_backend(desc: Dict[str, Any]) -> Backend:
    """The backend a manifest describes."""
    from . import BACKENDS

    params = dict(desc)
    name = params.pop("backend")
    return BACKENDS[name](**params)


class Session:
    """One pod session of one backend."""

    def __init__(self, sdir: str, backend: Backend, problems: Sequence[Problem],
                 assignments: Dict[str, List[Candidate]], session_id: str, phase: str = "search",
                 repeats: int = 2, probe: bool = True, defaults: Optional[Dict[str, Candidate]] = None,
                 schema: Any = None, alone: bool = False):
        self.dir = os.path.abspath(sdir)
        self.backend = backend
        self.session_id = session_id
        self.phase = phase
        self.repeats = max(1, repeats)
        self.probe = probe
        self.schema = schema
        # One candidate a slot: the other problems of its process run their built-in plans.
        self.alone = alone
        bad = {p.key(): backend.supports(p) for p in problems}
        bad = {k: v for k, v in bad.items() if v}
        if bad:
            raise ValueError("backend %s cannot time: %s" % (backend.name, "; ".join(
                "%s (%s)" % (k, v) for k, v in sorted(bad.items())[:5])))
        self.problems = {p.key(): p for p in problems}
        self.assign = {k: list(v) for k, v in assignments.items() if k in self.problems}
        self.defaults = defaults or {}
        self.groups = backend.groups(list(problems))
        self.manifest: Dict[str, Any] = {}

    @property
    def script(self) -> str:
        return os.path.join(self.dir, SCRIPT)

    def _bits_same(self, p: Problem, cand: Candidate) -> bool:
        if self.schema is None:
            return True
        return all(self.schema.key(k).same_bits(p.dtype) for k in cand.config)

    def build(self) -> Dict[str, Any]:
        """The manifest: slots, cfg values, expected log fields (a gated call's also at the wider
        width its tiles pair at, and its pairing=) and the run order.  Alone, slot j of a group
        holds its j-th (problem, candidate) pair only, and every run of the group times all its
        problems (the process of slot 0) but records those of its slot."""
        slots: Dict[str, List[Dict[str, Any]]] = {}
        cfgs: Dict[str, str] = {}
        for gid, probs in enumerate(self.groups):
            n = max([len(self.assign.get(p.key(), [])) for p in probs] + [0])
            gslots: List[Dict[str, Any]] = [{
                p.key(): {"text": "", "bits_same": True,
                          "expect": plan_log_fields(self.defaults[p.key()].plan, p)
                          if p.key() in self.defaults else {},
                          "expect_paired": paired_log_fields(self.defaults[p.key()].plan, p)
                          if p.key() in self.defaults else {},
                          "expect_pairing": log_pairing(self.defaults[p.key()].plan, p)
                          if p.key() in self.defaults else None}
                for p in probs}]
            if self.alone:
                for p in probs:
                    for c in self.assign.get(p.key(), []):
                        if c.text:
                            cfgs["cfg/g%d_s%d.cfg" % (gid, len(gslots))] = format_override(
                                [(problem_filters(p), c.text)])
                            gslots.append({p.key(): {"text": c.text, "bits_same": self._bits_same(p, c),
                                                     "expect": plan_log_fields(c.plan, p),
                                                     "expect_paired": paired_log_fields(c.plan, p),
                                                     "expect_pairing": log_pairing(c.plan, p)}})
                n = 0
            for j in range(n):
                slot: Dict[str, Any] = {}
                clauses = []
                for p in probs:
                    cands = self.assign.get(p.key(), [])
                    if j < len(cands) and cands[j].text:
                        c = cands[j]
                        slot[p.key()] = {"text": c.text, "bits_same": self._bits_same(p, c),
                                         "expect": plan_log_fields(c.plan, p),
                                         "expect_paired": paired_log_fields(c.plan, p),
                                         "expect_pairing": log_pairing(c.plan, p)}
                        clauses.append((problem_filters(p), c.text))
                if not clauses:
                    continue
                cfgs["cfg/g%d_s%d.cfg" % (gid, len(gslots))] = format_override(clauses)
                gslots.append(slot)
            slots[str(gid)] = gslots
        runs: List[Dict[str, Any]] = []

        def add(kind: str, gid: int, j: int, rep: int) -> None:
            run = {"id": "r%04d" % (len(runs) + 1), "kind": kind, "group": gid, "slot": j,
                   "repeat": rep, "pos": len(runs), "pkeys": list(slots[str(gid)][j]),
                   "cfg": "cfg/g%d_s%d.cfg" % (gid, j) if j else None}
            if self.alone:
                run["record"], run["pkeys"] = run["pkeys"], [p.key() for p in self.groups[gid]]
            runs.append(run)

        if self.probe:
            for gid in range(len(self.groups)):
                for j in range(len(slots[str(gid)])):
                    add("probe", gid, j, -1)
        for rep in range(self.repeats):
            for gid in range(len(self.groups)):
                order = list(range(1, len(slots[str(gid)])))
                if rep % 2:
                    order.reverse()
                for j in [0] + order + [0]:
                    add("timed", gid, j, rep)
        self.manifest = {
            "session": self.session_id, "phase": self.phase, "created": utc_now(), "alone": self.alone,
            "backend": self.backend.describe(), "repeats": self.repeats,
            "problems": {k: p.to_dict() for k, p in self.problems.items()},
            "groups": [[p.key() for p in g] for g in self.groups],
            "slots": slots, "cfgs": cfgs, "runs": runs,
        }
        return self.manifest

    def write(self) -> str:
        """Writes the session directory; returns the job script path."""
        if os.path.exists(os.path.join(self.dir, MANIFEST)):
            raise FileExistsError("session %s exists; outputs of two sessions must not mix" % self.dir)
        man = self.manifest or self.build()
        for sub in ("cfg", "inputs", "out"):
            os.makedirs(os.path.join(self.dir, sub), exist_ok=True)
        for rel, text in man["cfgs"].items():
            with open(os.path.join(self.dir, rel), "w", encoding="utf-8") as f:
                f.write(text + "\n")
        for gid, probs in enumerate(self.groups):
            self.backend.write_inputs(self.dir, gid, probs)
        lines = [
            "#!/bin/bash",
            "# native_brgemm_tuning session %s (%s, phase %s): %d runs, probes first, then the built-in"
            % (self.session_id, self.backend.name, self.phase, len(man["runs"])),
            "# plan and the candidates interleaved (A B .. B A) and repeated %d times." % self.repeats,
            "set -uo pipefail",
            'S=$(cd "$(dirname "$0")" && pwd)',
            'mkdir -p "$S/out"',
        ]
        lines += self.backend.preamble()
        lines += [
            "echo \"host=$(hostname) cpus=$(awk '/Cpus_allowed_list/{print $2}' /proc/self/status)"
            + " start=$(date -u '+%F %T UTC')\"",
            "run() {",
            "    local i=$1 cfg=$2 lvl=$3",
            "    shift 3",
            "    (",
            "        export ZENDNNL_MATMUL_ALGO=10",
            '        [ "$cfg" = - ] || export ZENDNNL_MATMUL_NATIVE_CONFIG="$(cat "$S/$cfg")"',
            '        [ "$lvl" = - ] || export ZENDNNL_API_LOG_LEVEL="$lvl"',
            '        "$@"',
            '    ) > "$S/out/$i.txt" 2>&1',
            "    echo \"run $i rc=$? $(date -u '+%T')\"",
            "}",
        ]
        problems = self.problems
        for r in man["runs"]:
            probs = [problems[k] for k in r["pkeys"]]
            cmd = self.backend.command(r["group"], probs, r["kind"] == "probe")
            lines.append("run %s %s %s %s" % (r["id"], r["cfg"] or "-", "4" if r["kind"] == "probe" else "-", cmd))
        lines.append("echo \"done $(date -u '+%F %T UTC')\"")
        with open(self.script, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        os.chmod(self.script, 0o755)
        with open(os.path.join(self.dir, MANIFEST), "w", encoding="utf-8") as f:
            json.dump(man, f, indent=1, sort_keys=True)
        return self.script

    def launch_command(self, launcher: str, pod: str = "") -> str:
        """launcher with {dir}, {script} and {pod} filled in."""
        return launch_command(self.dir, self.script, launcher, pod)

    def launch(self, launcher: str, timeout_s: int = 7200, pod: str = "") -> subprocess.Popen:
        """Starts the launcher (one pod claim for the whole session); does not wait.
        A pod, when given, is also the launcher's POD."""
        return launch_job(self.dir, self.script, launcher, self.backend.threads, timeout_s, pod)


def launch_command(sdir: str, script: str, launcher: str, pod: str = "") -> str:
    """launcher with {dir}, {script} and {pod} filled in."""
    return launcher.format(dir=shlex.quote(sdir), script=shlex.quote(script), pod=shlex.quote(pod))


def launch_job(sdir: str, script: str, launcher: str, threads: int, timeout_s: int = 7200,
               pod: str = "") -> subprocess.Popen:
    """Starts launcher on the job script of sdir (output in sdir/launch.out); does not wait."""
    env = dict(os.environ, TIMEOUT=str(timeout_s), THREADS=str(threads), LOG=os.path.join(sdir, "launcher.log"))
    if pod:
        env["POD"] = pod
    with open(os.path.join(sdir, "launch.out"), "w", encoding="utf-8") as out:
        return subprocess.Popen(shlex.split(launch_command(sdir, script, launcher, pod)), env=env, stdout=out,
                                stderr=subprocess.STDOUT)


def collect(sdir: str, backend: Optional[Backend] = None) -> List[Dict[str, Any]]:
    """Result records of a session directory whose job has run."""
    with open(os.path.join(sdir, MANIFEST), encoding="utf-8") as f:
        man = json.load(f)
    backend = backend or make_backend(man["backend"])
    problems = {k: Problem.from_dict(d) for k, d in man["problems"].items()}
    slots = man["slots"]
    records: List[Dict[str, Any]] = []
    hashes: Dict[tuple, Optional[str]] = {}
    for r in man["runs"]:
        gslots = slots[str(r["group"])]
        slot = gslots[r["slot"]]
        probs = [problems[k] for k in r["pkeys"]]
        # The problems a run times but does not record (alone: those at their built-in plans).
        skip = set(r["pkeys"]) - set(r.get("record", r["pkeys"]))
        base = {"session": man["session"], "phase": man["phase"], "backend": man["backend"]["backend"],
                "run": r["id"], "group": r["group"], "slot": r["slot"]}
        path = os.path.join(sdir, "out", r["id"] + ".txt")
        text = None
        if os.path.exists(path):
            with open(path, encoding="utf-8", errors="replace") as f:
                text = f.read()
        if r["kind"] == "probe":
            info = backend.parse_probe(text, probs) if text is not None else {}
            reasons, warns = parse_warnings(text) if text is not None else ({}, [])
            for p in probs:
                k = p.key()
                if k in skip:
                    continue
                s = slot[k]
                i = info.get(k, {})
                logged = i.get("logged")
                if text is None:
                    st, why = "failed", "no output (the run did not happen)"
                else:
                    st, why = applied_status(logged, s["text"])
                    if st == "dropped":
                        # The warning names the planner's N.
                        why = reasons.get((p.m, p.plan_n, p.k), why)
                    if i.get("inconsistent"):
                        st, why = "mismatch", "the calls of this problem logged different plans"
                want = s.get("expect_paired", {}).get((logged or {}).get("pairing", ""), s.get("expect", {}))
                diff = (compare_pairing(s.get("expect_pairing"), logged) + compare_fields(want, logged, p.bf16)
                        if logged and st == "applied" else [])
                hashes[(k, r["group"], r["slot"])] = i.get("hash")
                records.append(dict(base, kind="probe", pkey=k, problem=p.to_dict(), text=s["text"],
                                    status=st, reason=why, logged=logged, model_diff=diff,
                                    hash=i.get("hash"), bits_same=s.get("bits_same", True),
                                    warnings=warns[:8]))
        else:
            res = backend.parse_timed(text, probs) if text is not None else {}
            for p in probs:
                k = p.key()
                if k in skip:
                    continue
                got = res.get(k)
                records.append(dict(base, kind="measure", pkey=k, problem=p.to_dict(), text=slot[k]["text"],
                                    time_ms=got["time_ms"] if got else None, stats=got.get("stats", {}) if got else {},
                                    repeat=r["repeat"], pos=r["pos"], status="ok" if got else "failed"))
    for rec in records:
        if rec["kind"] != "probe" or not rec["text"] or rec["status"] != "applied":
            continue
        h0, h = hashes.get((rec["pkey"], rec["group"], 0)), rec.get("hash")
        if h0 and h and h != h0:
            if rec["bits_same"]:
                rec["status"], rec["reason"] = "bits_changed", "output hash %s, built-in %s" % (h, h0)
            else:
                rec["bits"] = "changed"
    return records


def mark_collected(sdir: str, n: int) -> None:
    with open(os.path.join(sdir, COLLECTED), "w", encoding="utf-8") as f:
        json.dump({"records": n, "ts": utc_now()}, f)


def pending_sessions(workdir: str) -> List[str]:
    """Session directories written but not collected yet."""
    root = os.path.join(workdir, "sessions")
    if not os.path.isdir(root):
        return []
    return sorted(os.path.join(root, d) for d in os.listdir(root)
                  if os.path.exists(os.path.join(root, d, MANIFEST))
                  and not os.path.exists(os.path.join(root, d, COLLECTED)))


class DryRun(Exception):
    """Sessions were written and not launched."""

    def __init__(self, dirs: List[str], commands: List[str]):
        super().__init__("%d session(s) written, not launched" % len(dirs))
        self.dirs = dirs
        self.commands = commands


class PodEvaluator:
    """evaluate(round, phase, batch): up to `parallel` sessions (one pod claim each) per round.
    With pods, session i of a round runs on pods[i] (the launcher's POD and {pod}), and a
    round has at most one session per pod."""

    def __init__(self, backend: Backend, workdir: str, problems: Dict[str, Problem],
                 defaults: Dict[str, Candidate], launcher: Optional[str] = None, repeats: int = 2,
                 parallel: int = 1, timeout_s: int = 7200, schema: Any = None, dry_run: bool = False,
                 log: Any = print, pods: Sequence[str] = (), alone: bool = False):
        self.backend = backend
        self.alone = alone
        self.workdir = workdir
        self.problems = problems
        self.defaults = defaults
        self.launcher = launcher
        self.repeats = repeats
        self.pods = [p for p in pods if p]
        self.parallel = max(1, min(parallel, len(self.pods)) if self.pods else parallel)
        self.timeout_s = timeout_s
        self.schema = schema
        self.dry_run = dry_run or not launcher
        self.log = log

    def _parts(self, keys: List[str], batch: Dict[str, List[Candidate]]) -> List[List[str]]:
        """keys in order, cut into up to `parallel` parts of about equal backend cost."""
        n = min(self.parallel, len(keys))
        weight = {k: len(batch[k]) * self.backend.cost(self.problems[k]) for k in keys}
        total = sum(weight.values())
        parts: List[List[str]] = [[]]
        acc = 0.0
        for k in keys:
            if len(parts) < n and parts[-1] and acc + weight[k] / 2 > total * len(parts) / n:
                parts.append([])
            parts[-1].append(k)
            acc += weight[k]
        return [p for p in parts if p]

    def __call__(self, round_id: int, phase: str, batch: Dict[str, List[Candidate]]) -> List[Dict[str, Any]]:
        probs = self.problems
        keys = sorted((k for k in batch if batch[k]),
                      key=lambda k: (probs[k].dtype, probs[k].m, probs[k].n, probs[k].k))
        sessions = []
        for i, part in enumerate(self._parts(keys, batch)):
            base = sid = "%s%03d_%d" % (phase[0], round_id, i)
            n = 1
            while os.path.exists(os.path.join(self.workdir, "sessions", sid)):
                n += 1
                sid = "%s_%d" % (base, n)
            s = Session(os.path.join(self.workdir, "sessions", sid), self.backend, [probs[k] for k in part],
                        {k: batch[k] for k in part}, sid, phase=phase, repeats=self.repeats,
                        defaults=self.defaults, schema=self.schema, alone=self.alone)
            s.write()
            sessions.append(s)
        pods = [self.pods[i % len(self.pods)] if self.pods else "" for i in range(len(sessions))]
        commands = [s.launch_command(self.launcher, pod) if self.launcher else "<launcher> " + s.script
                    for s, pod in zip(sessions, pods)]
        if self.dry_run:
            raise DryRun([s.dir for s in sessions], commands)
        running = [(s, s.launch(self.launcher, self.timeout_s, pod)) for s, pod in zip(sessions, pods)]
        records: List[Dict[str, Any]] = []
        for s, proc in running:
            rc = proc.wait()
            recs = collect(s.dir, self.backend)
            mark_collected(s.dir, len(recs))
            self.log("session %s: launcher exit %d, %d records" % (s.session_id, rc, len(recs)))
            records += recs
        return records


PROBE_MANIFEST = "probe.json"
PROBE_OUT = "ceilings.json"
PROBE_DONE = "probe.rc"  # the probe's exit status, written when it ends


def write_probe(sdir: str, backend: Backend, probe_bin: str, pod: str = "", quick: bool = False,
                seconds: float = 0.0, repeats: int = 0) -> str:
    """A job that runs the op performance utility's probe (tools/op_perf_utility/probe) with
    the team and pinning of the backend's timed runs, writing sdir/ceilings.json.  Returns the
    job script; collect_probe() reads its output."""
    sdir = os.path.abspath(sdir)
    if os.path.exists(os.path.join(sdir, PROBE_MANIFEST)):
        raise FileExistsError("probe %s exists" % sdir)
    os.makedirs(sdir, exist_ok=True)
    args = ["--out", '"$S/%s"' % PROBE_OUT, "--threads", str(backend.threads)]
    if quick:
        args.append("--quick")
    if seconds > 0:
        args += ["--seconds", "%g" % seconds]
    if repeats > 0:
        args += ["--repeats", str(repeats)]
    lines = ["#!/bin/bash",
             "# native_brgemm_tuning: the machine's ceilings with the team of the %s backend's timed runs"
             % backend.name,
             "set -uo pipefail", 'S=$(cd "$(dirname "$0")" && pwd)'] + backend.probe_env() + [
        "echo \"host=$(hostname) cpus=$(awk '/Cpus_allowed_list/{print $2}' /proc/self/status)"
        + " start=$(date -u '+%F %T UTC')\"",
        '%s %s > "$S/probe.txt" 2>&1' % (shlex.quote(os.path.abspath(probe_bin)), " ".join(args)),
        'rc=$?',
        # probe.txt exists from the probe's start: collect waits for this.
        'echo "$rc" > "$S/%s"' % PROBE_DONE,
        'echo "probe rc=$rc"',
        "echo \"done $(date -u '+%F %T UTC')\"",
    ]
    script = os.path.join(sdir, SCRIPT)
    with open(script, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(script, 0o755)
    with open(os.path.join(sdir, PROBE_MANIFEST), "w", encoding="utf-8") as f:
        json.dump({"probe": os.path.abspath(probe_bin), "pod": pod, "backend": backend.describe(),
                   "threads": backend.threads, "quick": quick, "created": utc_now()}, f, indent=1)
    return script


def collect_probe(sdir: str) -> Tuple[Optional[str], str]:
    """The ceilings file a probe job wrote (None when it wrote none) and the last line the
    probe printed."""
    tail = ""
    out = os.path.join(sdir, "probe.txt")
    if os.path.exists(out):
        with open(out, encoding="utf-8", errors="replace") as f:
            lines = [line.strip() for line in f if line.strip()]
        tail = lines[-1] if lines else ""
    path = os.path.join(sdir, PROBE_OUT)
    return (path if os.path.exists(path) else None), tail


def pending_probes(workdir: str) -> List[str]:
    """Probe directories written but not collected yet."""
    root = os.path.join(workdir, "probe")
    if not os.path.isdir(root):
        return []
    return sorted(os.path.join(root, d) for d in os.listdir(root)
                  if os.path.exists(os.path.join(root, d, PROBE_MANIFEST))
                  and not os.path.exists(os.path.join(root, d, COLLECTED)))


VERIFY_MANIFEST = "verify.json"


def write_verify(sdir: str, backend: Backend, problems: Sequence[Problem], tuning_file: str) -> str:
    """A pod job that runs every problem's process twice with the plan lines and output hashes
    logged: with the built-in plan, then with ZENDNNL_MATMUL_NATIVE_TUNING_FILE=tuning_file.
    Returns the job script; collect_verify() reads the outputs."""
    sdir = os.path.abspath(sdir)
    if os.path.exists(os.path.join(sdir, VERIFY_MANIFEST)):
        raise FileExistsError("verification %s exists" % sdir)
    groups = backend.groups(list(problems))
    for sub in ("inputs", "out"):
        os.makedirs(os.path.join(sdir, sub), exist_ok=True)
    for gid, probs in enumerate(groups):
        backend.write_inputs(sdir, gid, probs)
    lines = ["#!/bin/bash", "# native_brgemm_tuning verification of %s" % tuning_file, "set -uo pipefail",
             'S=$(cd "$(dirname "$0")" && pwd)', 'mkdir -p "$S/out"'] + backend.preamble() + [
        "run() {",
        "    local i=$1 file=$2",
        "    shift 2",
        "    (",
        "        export ZENDNNL_MATMUL_ALGO=10 ZENDNNL_API_LOG_LEVEL=4",
        '        [ "$file" = - ] || export ZENDNNL_MATMUL_NATIVE_TUNING_FILE="$file"',
        '        "$@"',
        '    ) > "$S/out/$i.txt" 2>&1',
        "    echo \"run $i rc=$? $(date -u '+%T')\"",
        "}",
    ]
    runs = []
    for gid, probs in enumerate(groups):
        cmd = backend.command(gid, probs, True)
        for kind, f in (("builtin", "-"), ("file", shlex.quote(os.path.abspath(tuning_file)))):
            rid = "g%d_%s" % (gid, kind)
            lines.append("run %s %s %s" % (rid, f, cmd))
            runs.append({"id": rid, "group": gid, "kind": kind, "pkeys": [p.key() for p in probs]})
    lines.append("echo \"done $(date -u '+%F %T UTC')\"")
    script = os.path.join(sdir, SCRIPT)
    with open(script, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(script, 0o755)
    with open(os.path.join(sdir, VERIFY_MANIFEST), "w", encoding="utf-8") as f:
        json.dump({"file": os.path.abspath(tuning_file), "backend": backend.describe(), "runs": runs,
                   "problems": {p.key(): p.to_dict() for p in problems}, "created": utc_now()}, f, indent=1)
    return script


def collect_verify(sdir: str, plans: Sequence[Dict[str, Any]], backend: Optional[Backend] = None) -> Dict[str, Any]:
    """Per problem: the logged plan and output hash with the built-in plan and with the file, and the
    checks: the file's line names the layers, config and plan (and a gated call's pairing=) the file
    names for it (plans: the export's meta.plans, for the problems of its entries and those its rules
    alone give a config), a problem it names nothing for keeps the built-in plan, and the output bits
    are the built-in plan's."""
    with open(os.path.join(sdir, VERIFY_MANIFEST), encoding="utf-8") as f:
        man = json.load(f)
    backend = backend or make_backend(man["backend"])
    problems = {k: Problem.from_dict(d) for k, d in man["problems"].items()}
    named = {r["problem"]: r for r in plans}
    got: Dict[str, Dict[str, Any]] = {k: {} for k in problems}
    for r in man["runs"]:
        path = os.path.join(sdir, "out", r["id"] + ".txt")
        text = ""
        if os.path.exists(path):
            with open(path, encoding="utf-8", errors="replace") as f:
                text = f.read()
        info = backend.parse_probe(text, [problems[k] for k in r["pkeys"]]) if text else {}
        for k in r["pkeys"]:
            got[k][r["kind"]] = info.get(k, {})
    rows, bad = [], 0
    for k, p in sorted(problems.items()):
        b, f = got[k].get("builtin", {}), got[k].get("file", {})
        lb, lf = b.get("logged") or {}, f.get("logged") or {}
        why = []
        want = named.get(k)
        if not lb or not lf:
            why.append("no plan line")
        elif want is not None:
            layers = set(lf.get("source", "").split("+"))
            if want["source"] != "heuristic" and (not set(want["source"].split("+")) <= layers
                                                  or lf.get("cfg") != want["cfg"]):
                why.append("source=%s cfg=%s, the file names source=%s cfg=%s" % (
                    lf.get("source"), lf.get("cfg"), want["source"], want["cfg"]))
            why += compare_pairing(want.get("pairing"), lf)
            why += compare_fields(want.get("paired", {}).get(lf.get("pairing", ""), want["plan"]), lf)
        else:
            if lf.get("source") != lb.get("source") or lf.get("cfg") != lb.get("cfg"):
                why.append("no entry names it, yet source=%s cfg=%s" % (lf.get("source"), lf.get("cfg")))
            why += compare_fields(lb, lf)
        same_bits = b.get("hash") is not None and b.get("hash") == f.get("hash")
        if not same_bits:
            why.append("output hash %s, built-in %s" % (f.get("hash"), b.get("hash")))
        bad += bool(why)
        ruled = want is not None and want["source"] == "rules"
        rows.append({"problem": k, "entry": want["id"] if want and not ruled else None, "rules": ruled,
                     "builtin_plan": bool(want["builtin"]) if want else True, "hash": f.get("hash"),
                     "same_bits": same_bits, "ok": not why, "why": why})
    return {"file": man["file"], "problems": len(rows), "failed": bad,
            "named": sum(r["entry"] is not None for r in rows),
            "named_builtin": sum(r["entry"] is not None and r["builtin_plan"] for r in rows),
            "ruled": sum(r["rules"] for r in rows), "rows": rows}
