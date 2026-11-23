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
"""A/B gate of the native BRGEMM: two builds against each other.

ab_corpus (ab_corpus.cpp, one binary per build from build.sh) runs a fixed corpus on the library it
holds.  This script runs the binaries of two builds and compares them:

  run   the calls (the bytes of dst after each call and of the weight buffer, the
        [MATMUL.NATIVE_BRGEMM] plan lines and the other log lines of each case), the layouts (packed
        and unpacked bytes) and the layout rule's verdicts; exit status 1 when anything differs, 2
        when a mode compares nothing (no mode, or an A run without a case or record).
  size  the library's code: the text of the static archive and of its matmul_native objects, and
        the kernel functions (the names only in A and only in B); exit status 1 when the archive
        text moves by more than --budget percent, or with --same-kernels when the kernel functions
        differ, 2 when a build tree lacks an object of its archive or its kernel objects define no
        function.
  pack  the prepack time of the layouts and the unpack time of the bf16 and f16 layouts, the two
        binaries in turn: per case the median over the rounds and the B / A ratio, and the geomean
        of the packs and of the unpacks; exit status 2 when a round of A or B times other cases
        than the first round of A, or none.

Usage:
  ab.py run <binary-a> <binary-b> [--out DIR] [--modes calls,layout,rules] [--threads N]
            [--filter TEXT] [--show N]
  ab.py size <build-a> <build-b> [--budget PCT] [--same-kernels]
  ab.py pack <binary-a> <binary-b> [--rounds N] [--reps N] [--threads N] [--filter TEXT]
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import subprocess
import sys
from collections import Counter
from typing import Dict, List, Optional, Sequence, Set, Tuple

MODES = ("calls", "layout", "rules")
# A log line's [module][level], then its [time].
LOG_PREFIX_RE = re.compile(r"^(\[[^\]]*\]\[[^\]]*\])\[[^\]]*\]:")
POINTER_RE = re.compile(r"0x[0-9a-fA-F]+")
PLAN_TAG = "[MATMUL.NATIVE_BRGEMM] "
RECORD_KINDS = {"layout": ("LAYOUT", "UNPACK"), "rules": ("RULE",)}
NATIVE_DIR = os.sep + "matmul_native" + os.sep
KERNEL_DIR = os.sep + os.path.join("matmul_native", "brgemm", "kernel") + os.sep


def corpus_env(threads: int) -> Dict[str, str]:
    """The environment of a corpus run: the caller's without its ZENDNNL_* settings, the native BRGEMM
    selected for the process (weight cache mode 2 then rewrites weights in place), verbose log lines
    and a fixed thread count."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("ZENDNNL_")}
    env.update({"ZENDNNL_MATMUL_ALGO": "10", "ZENDNNL_API_LOG_LEVEL": "4", "OMP_NUM_THREADS": str(threads)})
    return env


def run_corpus(binary: str, mode: str, threads: int, extra: Sequence[str] = ()) -> str:
    """The output of one corpus run; raises when the run fails or stops early."""
    proc = subprocess.run([binary, mode, *extra], env=corpus_env(threads), stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, universal_newlines=True)
    if proc.returncode != 0 or (mode in MODES and f"SUMMARY {mode} " not in proc.stdout):
        tail = "\n".join(proc.stdout.splitlines()[-5:])
        raise RuntimeError(f"{binary} {mode}: exit status {proc.returncode}\n{tail}")
    return proc.stdout


def log_body(line: str) -> Tuple[str, str]:
    """A log line as its [module][level] and its text, the time dropped and the pointers masked."""
    m = LOG_PREFIX_RE.match(line)
    head, text = (m.group(1), line[m.end():]) if m else ("", line)
    return head, POINTER_RE.sub("0x?", text)


def parse_calls(text: str) -> Dict[str, Dict[str, object]]:
    """The cases of a calls run, in order: per case its CASE fields (st, out, again, wei), its plan lines
    and its other log lines (each with its module and level)."""
    cases: Dict[str, Dict[str, object]] = {}
    plan: List[str] = []
    log: List[str] = []
    open_case = False
    for line in text.splitlines():
        if line.startswith("BEGIN "):
            plan, log, open_case = [], [], True
        elif line.startswith("CASE "):
            parts = line.split()
            fields = dict(p.split("=", 1) for p in parts[2:])
            cases[parts[1]] = {"fields": fields, "plan": plan, "log": log}
            plan, log, open_case = [], [], False
        elif open_case:
            head, body = log_body(line)
            (plan if body.startswith(PLAN_TAG) else log).append(head + body)
    return cases


def parse_records(text: str, kinds: Sequence[str]) -> Dict[Tuple[str, str], str]:
    """The records of a layout or rules run, in order: per (kind, case) the rest of its line."""
    records: Dict[Tuple[str, str], str] = {}
    for line in text.splitlines():
        kind, _, rest = line.partition(" ")
        if kind in kinds:
            name, _, fields = rest.partition(" ")
            records[(kind, name)] = fields
    return records


def token_diff(x: str, y: str) -> str:
    """Two lines of the same fields as their differing fields, else whole."""
    tx, ty = x.split(" "), y.split(" ")
    if len(tx) != len(ty):
        return f"{x} -> {y}"
    return ", ".join(f"{p} -> {q}" for p, q in zip(tx, ty) if p != q)


def line_diff(what: str, a: Sequence[str], b: Sequence[str]) -> List[str]:
    """The first difference between two lists of log lines."""
    if list(a) == list(b):
        return []
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return [f"{what} line {i + 1}: {token_diff(x, y)}"]
    return [f"{what} lines {len(a)} -> {len(b)}"]


def keys_in_order(a: Dict, b: Dict) -> List:
    return list(a) + [k for k in b if k not in a]


def diff_calls(a: Dict[str, Dict[str, object]], b: Dict[str, Dict[str, object]]) -> List[str]:
    """One line per case whose fields, plan lines or log lines differ, or that one run lacks."""
    out = []
    for name in keys_in_order(a, b):
        if name not in b or name not in a:
            out.append(f"{name}: only in {'A' if name in a else 'B'}")
            continue
        ca, cb = a[name], b[name]
        fa, fb = ca["fields"], cb["fields"]
        notes = [f"{k} {fa.get(k)} -> {fb.get(k)}" for k in keys_in_order(fa, fb) if fa.get(k) != fb.get(k)]
        notes += line_diff("plan", ca["plan"], cb["plan"])
        notes += line_diff("log", ca["log"], cb["log"])
        if notes:
            out.append(f"{name}: " + "; ".join(notes))
    return out


def diff_records(a: Dict[Tuple[str, str], str], b: Dict[Tuple[str, str], str]) -> List[str]:
    """One line per record that differs or that one run lacks."""
    out = []
    for key in keys_in_order(a, b):
        kind, name = key
        if key not in b or key not in a:
            out.append(f"{kind} {name}: only in {'A' if key in a else 'B'}")
        elif a[key] != b[key]:
            out.append(f"{kind} {name}: {a[key]} -> {b[key]}")
    return out


def summarize_calls(cases: Dict[str, Dict[str, object]]) -> str:
    served = sum(1 for c in cases.values() if c["fields"].get("st") == "1")
    plans = sum(len(c["plan"]) for c in cases.values())
    return f"{len(cases)} cases, {served} served, {plans} plan lines"


def summarize_records(records: Dict[Tuple[str, str], str]) -> str:
    counts: Dict[str, int] = {}
    for kind, _ in records:
        counts[kind] = counts.get(kind, 0) + 1
    served = sum(1 for (kind, _), f in records.items() if kind == "RULE" and " st=1 " in f" {f} ")
    text = ", ".join(f"{n} {kind.lower()}" for kind, n in counts.items())
    return text + (f" ({served} served)" if "RULE" in counts else "")


def compare_mode(mode: str, text_a: str, text_b: str) -> Tuple[str, List[str], int]:
    """What the A run of a mode holds (a summary and the number of its cases or records), and the
    differences of the B run from it."""
    if mode == "calls":
        a, b = parse_calls(text_a), parse_calls(text_b)
        return summarize_calls(a), diff_calls(a, b), len(a)
    ra, rb = parse_records(text_a, RECORD_KINDS[mode]), parse_records(text_b, RECORD_KINDS[mode])
    return summarize_records(ra), diff_records(ra, rb), len(ra)


def cmd_run(args: argparse.Namespace) -> int:
    modes = [m for m in args.modes.split(",") if m]
    unknown = [m for m in modes if m not in MODES]
    if unknown or not modes:
        what = f"unknown mode {unknown[0]}" if unknown else "no mode"
        print(f"{what} (modes: {', '.join(MODES)})", file=sys.stderr)
        return 2
    extra = ["--filter", args.filter] if args.filter else []
    failed = empty = False
    for mode in modes:
        texts = []
        for label, binary in (("a", args.binary_a), ("b", args.binary_b)):
            text = run_corpus(binary, mode, args.threads, extra)
            if args.out:
                os.makedirs(args.out, exist_ok=True)
                with open(os.path.join(args.out, f"{label}.{mode}.txt"), "w") as f:
                    f.write(text)
            texts.append(text)
        summary, diffs, held = compare_mode(mode, texts[0], texts[1])
        if not held:
            print(f"{mode}: the A run holds no case{' for this filter' if args.filter else ''}: "
                  "nothing compared")
            empty = True
            continue
        verdict = "identical" if not diffs else f"{len(diffs)} differ"
        print(f"{mode}: {summary}: {verdict}")
        for d in diffs[:args.show]:
            print(f"  {d}")
        if len(diffs) > args.show:
            print(f"  ... {len(diffs) - args.show} more")
        failed = failed or bool(diffs)
    return 2 if empty else 1 if failed else 0


def parse_size_totals(text: str) -> Tuple[int, int, int]:
    """text, data and bss of the (TOTALS) line of size -t output (Berkeley format)."""
    for line in text.splitlines():
        if line.rstrip().endswith("(TOTALS)"):
            text_bytes, data, bss = (int(v) for v in line.split()[:3])
            return text_bytes, data, bss
    raise RuntimeError("no (TOTALS) line in the output of size")


def size_totals(paths: Sequence[str]) -> Tuple[int, int, int]:
    """text, data and bss of the objects or archives at paths."""
    totals = [0, 0, 0]
    for i in range(0, len(paths), 256):
        out = subprocess.run(["size", "-t", *paths[i:i + 256]], stdout=subprocess.PIPE, check=True,
                             universal_newlines=True).stdout
        for j, v in enumerate(parse_size_totals(out)):
            totals[j] += v
    return totals[0], totals[1], totals[2]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def archive_members(archive: str) -> Counter:
    """The members of a static archive (GNU ar format) as (name, size, digest of the bytes) with their
    counts."""
    members: Counter = Counter()
    with open(archive, "rb") as f:
        if f.read(8) != b"!<arch>\n":
            raise RuntimeError(f"{archive} is not an ar archive")
        long_names = b""
        while True:
            header = f.read(60)
            if not header:
                break
            if len(header) != 60 or header[58:] != b"`\n":
                raise RuntimeError(f"{archive}: bad member header at byte {f.tell() - len(header)}")
            name, size = header[:16].decode().rstrip(), int(header[48:58])
            data = f.read(size + size % 2)[:size]
            if name in ("/", "/SYM64/"):
                continue
            if name == "//":
                long_names = data
                continue
            if name.startswith("/"):
                start = int(name[1:])
                name = long_names[start:long_names.index(b"/\n", start)].decode()
            members[(name.rstrip("/"), size, digest(data))] += 1
    return members


def objects_under(build: str, part: str) -> List[str]:
    """The objects of the zendnnl archive in a build tree whose path holds part: those of the archive
    target that the archive holds, by name, size and bytes (not the shared library's copies of a
    --shared build, nor the objects of sources removed or moved since).  Raises when the tree lacks
    an object of the archive, or when copies of one in and out of part leave its place open."""
    src = os.path.join(build, "zendnnl", "src")
    archive = os.path.join(src, "libzendnnl_archive.a")
    members = archive_members(archive)
    sizes = {(name, size) for name, size, _ in members}
    root = os.path.join(src, "CMakeFiles", "zendnnl_archive.dir")
    paths: Dict[Tuple[str, int, str], List[str]] = {}
    for dirpath, dirnames, files in os.walk(root):
        dirnames.sort()
        for f in sorted(files):
            path = os.path.join(dirpath, f)
            size = os.path.getsize(path)
            if not f.endswith(".o") or (f, size) not in sizes:
                continue
            with open(path, "rb") as obj:
                paths.setdefault((f, size, digest(obj.read())), []).append(path)
    found = []
    for key, count in members.items():
        held = paths.get(key, [])
        if len(held) < count:
            raise RuntimeError(f"{archive} holds {key[0]} ({key[1]} bytes), which no object under {root} "
                               "matches: rebuild the tree")
        inside = [p for p in held if part in os.sep + os.path.relpath(p, root)]
        if len(held) > count and 0 < len(inside) < len(held):
            raise RuntimeError(f"{key[0]} of {archive}: objects in and out of {part} hold its bytes "
                               f"({', '.join(held)}): remove the stale ones")
        found += inside[:count]
    return sorted(found)


def text_symbols(objects: Sequence[str]) -> Set[str]:
    """The distinct functions the objects define (nm: text symbols, weak ones included)."""
    names = set()
    for i in range(0, len(objects), 256):
        out = subprocess.run(["nm", "--defined-only", *objects[i:i + 256]], stdout=subprocess.PIPE,
                             check=True, universal_newlines=True).stdout
        for line in out.splitlines():
            parts = line.split()
            if len(parts) == 3 and parts[1] in ("T", "t", "W", "w"):
                names.add(parts[2])
    return names


def size_report(build: str) -> Dict[str, object]:
    """The text of the archive and of its matmul_native objects, and the kernel functions; raises when
    the kernel objects define none."""
    archive = os.path.join(build, "zendnnl", "src", "libzendnnl_archive.a")
    if not os.path.isfile(archive):
        raise RuntimeError(f"no {archive}: build zendnnl first")
    kernels = text_symbols(objects_under(build, KERNEL_DIR))
    if not kernels:
        raise RuntimeError(f"no kernel function in the objects of {archive} under {KERNEL_DIR}")
    return {"library text": size_totals([archive])[0],
            "matmul_native text": size_totals(objects_under(build, NATIVE_DIR))[0],
            "kernel symbols": kernels}


def percent(a: int, b: int) -> float:
    return 100.0 * (b - a) / a if a else 0.0


def cmd_size(args: argparse.Namespace) -> int:
    a, b = size_report(args.build_a), size_report(args.build_b)
    print(f"{'size':20} {'A':>12} {'B':>12} {'delta':>10}")
    for key in ("library text", "matmul_native text"):
        delta = b[key] - a[key]
        print(f"{key:20} {a[key]:12d} {b[key]:12d} {delta:+10d} ({percent(a[key], b[key]):+.3f}%)")
    ka, kb = a["kernel symbols"], b["kernel symbols"]
    print(f"{'kernel symbols':20} {len(ka):12d} {len(kb):12d} {len(kb) - len(ka):+10d}")
    for label, names in (("only in A", ka - kb), ("only in B", kb - ka)):
        for name in sorted(names):
            print(f"  {label}: {name}")
    status = 0
    moved = abs(percent(a["library text"], b["library text"]))
    if args.budget is not None and moved > args.budget:
        print(f"library text moved {moved:.3f}%, above the budget of {args.budget}%")
        status = 1
    if args.same_kernels and ka != kb:
        print(f"kernel symbols differ: {len(ka - kb)} only in A, {len(kb - ka)} only in B")
        status = 1
    return status


PACK_RE = re.compile(r"^(?P<kind>PACK|UNPACK) (?P<name>\S+) median_ms=(?P<median>[\d.eE+-]+) "
                     r"min_ms=(?P<min>[\d.eE+-]+)")


def parse_pack(text: str) -> Dict[str, float]:
    """Per pack and unpack of a layout case in a pack run ("pack <case>", "unpack <case>"), its median
    time in ms."""
    return {f"{m['kind'].lower()} {m['name']}": float(m["median"])
            for m in map(PACK_RE.match, text.splitlines()) if m}


def median(values: Sequence[float]) -> float:
    v = sorted(values)
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def cmd_pack(args: argparse.Namespace) -> int:
    if args.rounds < 1:
        print("--rounds must be at least 1", file=sys.stderr)
        return 2
    extra = ["--reps", str(args.reps)] + (["--filter", args.filter] if args.filter else [])
    runs: Dict[str, List[Dict[str, float]]] = {"a": [], "b": []}
    for _ in range(args.rounds):
        for label, binary in (("a", args.binary_a), ("b", args.binary_b)):
            runs[label].append(parse_pack(run_corpus(binary, "pack", args.threads, extra)))
    cases = set(runs["a"][0])
    if not cases:
        print(f"pack: the A run holds no case{' for this filter' if args.filter else ''}: nothing timed")
        return 2
    for label, rounds in runs.items():
        for i, times in enumerate(rounds):
            if set(times) != cases:
                odd = sorted(cases ^ set(times))
                print(f"pack: round {i + 1} of {label.upper()} times other cases than round 1 of A "
                      f"({len(times)} against {len(cases)}, {odd[0]} in one of them only)")
                return 2
    print(f"{'case':40} {'A ms':>10} {'B ms':>10} {'B/A':>7}")
    ratios: Dict[str, List[float]] = {}
    for name in runs["a"][0]:
        ta = median([r[name] for r in runs["a"]])
        tb = median([r[name] for r in runs["b"]])
        ratios.setdefault(name.split()[0], []).append(tb / ta)
        print(f"{name:40} {ta:10.3f} {tb:10.3f} {tb / ta:7.3f}")
    for kind, r in ratios.items():
        print(f"{kind}: geomean B/A {math.exp(sum(map(math.log, r)) / len(r)):.3f} over {len(r)} cases")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="compare the corpus of two builds")
    run.add_argument("binary_a")
    run.add_argument("binary_b")
    run.add_argument("--out", help="directory for the outputs of the runs (a.<mode>.txt, b.<mode>.txt)")
    run.add_argument("--modes", default=",".join(MODES))
    run.add_argument("--threads", type=int, default=16)
    run.add_argument("--filter", help="cases whose name holds this text only")
    run.add_argument("--show", type=int, default=20, help="differences listed per mode")
    size = sub.add_parser("size", help="compare the code size of two builds")
    size.add_argument("build_a")
    size.add_argument("build_b")
    size.add_argument("--budget", type=float, help="largest change of the library text, percent")
    size.add_argument("--same-kernels", action="store_true",
                      help="exit status 1 when the kernel symbols of the two builds differ")
    pack = sub.add_parser("pack", help="compare the pack and unpack time of two builds")
    pack.add_argument("binary_a")
    pack.add_argument("binary_b")
    pack.add_argument("--rounds", type=int, default=3)
    pack.add_argument("--reps", type=int, default=15)
    pack.add_argument("--threads", type=int, default=len(os.sched_getaffinity(0)),
                      help="threads of the packs (default: the CPUs this process may run on)")
    pack.add_argument("--filter", help="cases whose name holds this text only")
    args = ap.parse_args(argv)
    try:
        return {"run": cmd_run, "size": cmd_size, "pack": cmd_pack}[args.cmd](args)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as e:
        print(e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
