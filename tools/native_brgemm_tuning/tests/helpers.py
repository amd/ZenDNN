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
"""Shared test fixtures: the dumped schemas and records of a fake session."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import sys
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from native_brgemm_tuning import schema as schema_mod  # noqa: E402
from native_brgemm_tuning.candidates import Candidate  # noqa: E402
from native_brgemm_tuning.problems import Problem  # noqa: E402

HAVE_NUMPY = importlib.util.find_spec("numpy") is not None

SCHEMA = schema_mod.load_schema()
# The machine the tests' plans are worked out for (8-core L3 domains of 32 MiB), whichever
# machine the schema fixture was dumped on.
UARCH = schema_mod.Uarch(l1d_bytes=48 << 10, l2_bytes=1 << 20, l3_bytes=32 << 20, family_l3_cores=8,
                         num_cores=32, l3_domain_cores=8)
# The fixture's machine, as the hardware blocks of a tuning file read it.
HW = SCHEMA.hw_match
# Written from the design: the schema of an older build, without the rule keys, the
# blocks schedule and hw_match.
DESIGN = schema_mod.load_schema(os.path.join(os.path.dirname(schema_mod.FIXTURE), "schema_design.json"))

# Every rule switch off.
RULES_OFF = "makespan=0,l1_pass=0,traffic_groups=0,bf16_alias=0,bf16_narrow_n=0,blocks=0,cost_model=0"

BF16_RULE_KEYS = ("bf16_alias", "bf16_alias_mr", "bf16_narrow_n", "bf16_narrow_n_min_m")
BLOCKS_RULE_KEYS = ("blocks", "bf16_blocks_min_m", "int8_blocks_min_m")


def without_bf16_rules(s: schema_mod.Schema) -> schema_mod.Schema:
    """s as a build without BF16 tiles of 64 and 128 columns and without the
    rules bf16_alias and bf16_narrow_n."""
    doc = dict(s.doc)
    keys = [dict(k, values=[16, 32]) if k["name"] == "nr" else k for k in doc["keys"]
            if k["name"] not in BF16_RULE_KEYS]
    kernels = [k for k in doc["kernels"] if k["kind"] != "bf16" or k["nr"] <= 32]
    return schema_mod.Schema(dict(doc, keys=keys, kernels=kernels), s.source + " without the BF16 rules")


def without_blocks(s: schema_mod.Schema) -> schema_mod.Schema:
    """s as a build without the blocks schedule, its unit_tiles pin and its rule."""
    keys = [dict(k, values=[v for v in k["values"] if v != "blocks"]) if k["name"] == "schedule" else k
            for k in s.doc["keys"] if k["name"] not in ("unit_tiles",) + BLOCKS_RULE_KEYS]
    return schema_mod.Schema(dict(s.doc, keys=keys), s.source + " without the blocks schedule")


def with_newer_key(s: schema_mod.Schema) -> schema_mod.Schema:
    """s with newer_pct, a key of a newer build that the plan mirror does not read."""
    newer = {"name": "newer_pct", "type": "int", "default": 50, "min": 0, "max": 100, "applies": "both",
             "int8_schemes": ["per_channel", "per_group"], "bits": "same", "kind": "param"}
    return schema_mod.Schema(dict(s.doc, keys=s.doc["keys"] + [newer]), s.source + " with a newer key")


# The schemas of builds without those rules, and of a newer build.
OLDER = without_bf16_rules(SCHEMA)
NO_BLOCKS = without_blocks(SCHEMA)
NEWER = with_newer_key(SCHEMA)


def strict_json(text: str) -> Any:
    """text parsed as the library's loader parses a tuning file: NaN and infinities are not JSON."""
    def refuse(name: str) -> Any:
        raise ValueError("%s is not a JSON number" % name)

    return json.loads(text, parse_constant=refuse)


def session_records(problems: Dict[str, Problem], batch: Dict[str, List[Candidate]],
                    ratio: Callable[[Problem, Candidate], float], session: str, phase: str = "search",
                    repeats: int = 2, noise: float = 0.0, seed: int = 0,
                    status: Optional[Callable[[Problem, Candidate], str]] = None) -> List[Dict[str, Any]]:
    """Probe and measure records of one session (A B .. B A per repeat), times = ratio x 1 ms."""
    rng = random.Random("%s/%s" % (seed, session))
    out: List[Dict[str, Any]] = []
    for k, cands in batch.items():
        p = problems[k]
        base = {"session": session, "phase": phase, "backend": "test", "pkey": k, "problem": p.to_dict()}
        out.append(dict(base, kind="probe", text="", status="applied", reason=""))
        for c in cands:
            out.append(dict(base, kind="probe", text=c.text, status=status(p, c) if status else "applied",
                            reason=""))
        for rep in range(repeats):
            seq = [None] + list(cands if rep % 2 == 0 else reversed(cands)) + [None]
            for c in seq:
                t = (ratio(p, c) if c is not None else 1.0) * math.exp(rng.gauss(0.0, noise))
                out.append(dict(base, kind="measure", text=c.text if c else "", time_ms=t, repeat=rep))
    return out
