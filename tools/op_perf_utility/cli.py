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
"""Op performance utility CLI (run from tools/: python -m op_perf_utility ...).

  show      ceilings.json                       the machine the probe measured
  bound     --ceilings C --op matmul --shape S [--cold] [--instances N]  one call's bound
  report    --ceilings C --op matmul --benchdnn OUT [--cold] [--instances N] [--format table|csv|json]
  calibrate --ceilings C [--threads N]          the native BRGEMM cost model's hardware inputs

A shape is "m=..,n=..,k=..[,batch=..,src=bf16,wei=s8,dst=bf16,bias=f32,wei_group=..,
wei_scale_bytes=..,src_scale_bytes=..,gated=0,binary=0,binary_dt=f32,threads=..]".
See README.md.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, List, Optional

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from op_perf_utility.common import report  # noqa: E402
from op_perf_utility.common.ceilings import Ceilings, CeilingsError  # noqa: E402

OPS = ("matmul",)


def _ceilings(path: str) -> Ceilings:
    try:
        return Ceilings.load(path)
    except CeilingsError as e:
        sys.exit("ceilings: %s" % e)


def _write(text: str, out: Optional[str]) -> None:
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        sys.stdout.write(text if text.endswith("\n") else text + "\n")


def cmd_show(a: argparse.Namespace) -> int:
    print(_ceilings(a.ceilings).summary())
    return 0


def cmd_bound(a: argparse.Namespace) -> int:
    from op_perf_utility.matmul.bounds import matmul_bound
    from op_perf_utility.matmul.work import MatmulShape

    c = _ceilings(a.ceilings)
    try:
        s = MatmulShape.parse(a.shape, threads=a.threads or c.threads)
    except ValueError as e:
        sys.exit("shape: %s" % e)
    try:
        b = matmul_bound(s, c, cold=a.cold, instances=a.instances)
    except ValueError as e:
        sys.exit("bound: %s" % e)
    _dram_note([b], c)
    _write(report.render([b.row(a.measured_ms)], a.format), a.output)
    return 0


def _dram_note(bounds: List[Any], c: Ceilings) -> None:
    """Say when a DRAM share is taken beyond the probe's largest measured thread count."""
    over = [b for b in bounds if not b.dram_measured]
    if over:
        t = max(b.instances * b.threads for b in over)
        print("note: DRAM at %d threads (%d bound%s) taken as at the probe's %d threads: probe the instances' "
              "CPUs together to measure it" % (t, len(over), "" if len(over) == 1 else "s",
                                               c.measured_threads("dram")), file=sys.stderr)


def cmd_report(a: argparse.Namespace) -> int:
    from op_perf_utility.matmul.benchdnn_ingest import read_benchdnn
    from op_perf_utility.matmul.bounds import matmul_bound

    c = _ceilings(a.ceilings)
    if a.instances < 1:
        sys.exit("report: instances must be positive, not %d" % a.instances)
    bounds, rows = [], []
    for path in a.benchdnn:
        for meas in read_benchdnn(path, threads=a.threads or c.threads):
            bounds.append(matmul_bound(meas.shape, c, cold=a.cold, instances=a.instances))
            rows.append(bounds[-1].row(meas.time_ms))
    if not rows:
        sys.exit("no matmul result in %s" % ", ".join(a.benchdnn))
    _dram_note(bounds, c)
    _write(report.render(rows, a.format), a.output)
    return 0


def cmd_calibrate(a: argparse.Namespace) -> int:
    from op_perf_utility.matmul.calibrate import calibrate

    c = _ceilings(a.ceilings)
    cal = calibrate(c, threads=a.threads or c.threads, domain_threads=a.domain_threads or None)
    if a.format == "settings":
        _write(cal.override(), a.output)
    elif a.format == "uarch":
        _write(cal.uarch_override(), a.output)
    else:
        _write(report.render(cal.rows(), a.format, ["key", "path", "value", "builtin", "source"]), a.output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("show", help="summarize a ceilings file")
    sp.add_argument("ceilings")
    sp.set_defaults(fn=cmd_show)

    sp = sub.add_parser("bound", help="the bound of one call")
    sp.add_argument("--ceilings", required=True)
    sp.add_argument("--op", choices=OPS, default="matmul")
    sp.add_argument("--shape", required=True)
    sp.add_argument("--threads", type=int, help="threads of the call (default: the probe's team)")
    sp.add_argument("--cold", action="store_true", help="weights come from DRAM whatever their size")
    sp.add_argument("--instances", type=int, default=1,
                    help="copies of the call running at once over the same memory, sharing its DRAM bandwidth")
    sp.add_argument("--measured-ms", type=float, help="a measured time to rate against the bound")
    sp.add_argument("--format", choices=report.FORMATS, default="table")
    sp.add_argument("-o", "--output")
    sp.set_defaults(fn=cmd_bound)

    sp = sub.add_parser("report", help="bounds and efficiency of benchdnn results")
    sp.add_argument("--ceilings", required=True)
    sp.add_argument("--op", choices=OPS, default="matmul")
    sp.add_argument("--benchdnn", action="append", required=True,
                    help="benchdnn output: its CSV file or its printed lines (repeatable)")
    sp.add_argument("--threads", type=int, help="threads of the runs (default: the probe's team)")
    sp.add_argument("--cold", action="store_true", help="the runs read cold weights")
    sp.add_argument("--instances", type=int, default=1,
                    help="copies of each run going at once over the same memory, sharing its DRAM bandwidth")
    sp.add_argument("--format", choices=report.FORMATS, default="table")
    sp.add_argument("-o", "--output")
    sp.set_defaults(fn=cmd_report)

    sp = sub.add_parser("calibrate", help="the native BRGEMM cost model's hardware inputs from the ceilings")
    sp.add_argument("--ceilings", required=True)
    sp.add_argument("--threads", type=int, help="the tuned team (default: the probe's)")
    sp.add_argument("--domain-threads", type=int, help="threads of one L3 domain (default: the probe's)")
    sp.add_argument("--format", choices=report.FORMATS + ("settings", "uarch"), default="table",
                    help="settings: a ZENDNNL_MATMUL_NATIVE_CONFIG of one clause per weight type; uarch: a "
                    "ZENDNNL_MATMUL_NATIVE_UARCH of the measured paths and, where the probe measured DRAM, "
                    "the memory terms (values to A/B, as the settings)")
    sp.add_argument("-o", "--output")
    sp.set_defaults(fn=cmd_calibrate)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
