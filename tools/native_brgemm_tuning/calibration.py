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
"""The cost model's hardware inputs from the machine's ceilings (tune.py calibrate).

op_perf_utility.matmul.calibrate maps the probe's measurements onto the cm_* keys per
compute path; configs() gives each weight type of the campaign whose plans the cost model
prices its values.  tune.py calibrate times every problem whose plan they change against
the built-in plan (ab_batch(), phase "calibrate") and decide() keeps them when the
geometric mean of those problems' ratios gains at least calib_min_gain and none is slower
beyond the noise.  Kept, they are the base of every candidate of the search
(candidates.Space base), the search tries values around them (seed_values()), and the
export carries them in every entry and in rules (with_rules()) for the calls of the
problems without an entry, from M = 2."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

from .candidates import Candidate, Space, builtin_value, parse_values
from .export import merge_config
from .problems import FORMATS, Problem, format_families
from .results import ProblemSummary
from .rules import MIN_RULE_M, OPS

# The keys op_perf_utility.matmul.calibrate derives (its KEYS).
KEYS = ("cm_fma_pipes", "cm_l2_bytes", "cm_l3_cycles", "cm_mem_cycles", "cm_stream_cycles")
# The keys tune.py calibrate tries unless told otherwise: at the probe's L2 streaming rate,
# cm_l2_bytes made BF16 plans slower on every machine it was tried on (a call's slab re-reads
# share the core with its FMAs).
DEFAULT_KEYS = tuple(k for k in KEYS if k != "cm_l2_bytes")
# The weight types whose plans the cost model prices (native_plan_cost: BF16, F16, INT8
# with per-channel scales and s4); an f32 plan takes none of the inputs.
PRICED = ("bf16", "f16", "s8", "s4")


def type_features(formats: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """The rule features that tell weight types apart, per format of the dump (rule_features):
    esz, the bytes of a weight element, and bf16, 1 for a weight on the BF16 kernels."""
    return {f["name"]: {"esz": f["elem_bits"] / 8.0, "bf16": 1.0 if "bf16" in format_families(f) else 0.0}
            for f in formats}


def _holds(when: Sequence[Sequence[Any]], feats: Dict[str, float]) -> bool:
    return all(OPS[op](feats[f], v) for f, op, v in when)


def type_when(formats: Sequence[Dict[str, Any]]) -> Dict[str, List[List[Any]]]:
    """Per weight type, rule conditions that hold for its problems and for no other type's: esz
    above the midpoint to the next smaller element size, the bf16 feature where another type has
    its element size, then esz up to the midpoint to the next larger one where a larger type
    meets the conditions before it.  A type that another type matches in esz and bf16 has none."""
    feats = type_features(formats)
    sizes = sorted({v["esz"] for v in feats.values()})
    out = {}
    for d, v in feats.items():
        same = [o for o, w in feats.items() if o != d and w["esz"] == v["esz"]]
        if any(feats[o]["bf16"] == v["bf16"] for o in same):
            continue
        i = sizes.index(v["esz"])
        when: List[List[Any]] = []
        if i > 0:
            when.append(["esz", ">", (sizes[i - 1] + v["esz"]) / 2])
        if same:
            when.append(["bf16", ">", 0.5] if v["bf16"] > 0.5 else ["bf16", "<=", 0.5])
        if any(w["esz"] > v["esz"] and _holds(when, w) for w in feats.values()):
            when.append(["esz", "<=", (v["esz"] + sizes[i + 1]) / 2])
        out[d] = when
    return out


def type_splits(formats: Sequence[Dict[str, Any]]) -> Any:
    """The weight types as a tree of splits, {"feature", "le", "then", "else"} with then the types
    of feature <= le, each the most even of bf16 at 0.5 and esz at the midpoints between element
    sizes (on a tie the first of them, in that order); a leaf is a type's name, or the list of the
    types no split tells apart."""
    feats = type_features(formats)
    sizes = sorted({v["esz"] for v in feats.values()})
    splits = [("bf16", 0.5)] + [("esz", (a + b) / 2) for a, b in zip(sizes, sizes[1:])]

    def build(names: List[str]) -> Any:
        best = None
        for feature, le in splits:
            then = [d for d in names if feats[d][feature] <= le]
            other = [d for d in names if feats[d][feature] > le]
            if then and other and (best is None or max(len(then), len(other)) < best[0]):
                best = (max(len(then), len(other)), feature, le, then, other)
        if best is None:
            return names[0] if len(names) == 1 else names
        return {"feature": best[1], "le": best[2], "then": build(best[3]), "else": build(best[4])}

    return build(list(feats))


# The rule conditions that tell each weight type's problems from the others' and the types'
# tree of splits, from the formats of the dump.
TYPE_WHEN = type_when(list(FORMATS.values()))
TYPE_SPLITS = type_splits(list(FORMATS.values()))


def _type_when(dtype: str) -> List[List[Any]]:
    if dtype not in TYPE_WHEN:
        raise ValueError("weight type %s: no condition on esz and bf16 tells its problems from another type's"
                         % dtype)
    return TYPE_WHEN[dtype]


def configs(cal: Any, dtypes: Sequence[str], keys: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, int]]:
    """Weight type -> its cm_* values (only keys, when given), for the types the cost model
    prices."""
    out = {}
    for d in sorted(set(dtypes) & set(PRICED)):
        v = cal.for_dtype(d)
        out[d] = {k: v[k] for k in KEYS if k in v and (keys is None or k in keys)}
    return out


def ab_batch(problems: Sequence[Problem], spaces: Dict[str, Space],
             cfgs: Dict[str, Dict[str, int]]) -> Dict[str, List[Candidate]]:
    """Per problem whose plan the values change: the candidate of its weight type's values
    (spaces without a base)."""
    out = {}
    for p in problems:
        sp = spaces[p.key()]
        c = sp.make(dict(cfgs.get(p.dtype, {})))
        if c is not None and c.sig != sp.default.sig:
            out[p.key()] = [c]
    return out


def decide(problems: Sequence[Problem], batch: Dict[str, List[Candidate]], summ: Dict[str, ProblemSummary],
           min_gain: float, calib_min_gain: float) -> Dict[str, Any]:
    """The A/B's verdict: per problem the ratio of the calibrated values to the built-in plan
    (1 for a problem whose plan they do not change), their geometric means over the changed
    problems (the one the gain is judged on) and over all, the problems slower beyond the
    noise (ratio - 1 above max(min_gain, noise)) and those without a result."""
    rows, logs, missing, slower = [], [], [], []
    for p in problems:
        k = p.key()
        if k not in batch:
            rows.append({"problem": k, "changed": False, "ratio": 1.0, "noise": 0.0})
            logs.append(0.0)
            continue
        cs = summ[k].cands.get(batch[k][0].text) if k in summ else None
        if cs is None or not cs.ratios:
            missing.append(k)
            rows.append({"problem": k, "changed": True, "ratio": None, "noise": None})
            continue
        r, nz = cs.ratio, cs.noise
        rows.append({"problem": k, "changed": True, "ratio": round(r, 4), "noise": round(nz, 4)})
        logs.append(math.log(r))
        if r - 1.0 > max(min_gain, nz):
            slower.append(k)
    changed = [math.log(row["ratio"]) for row in rows if row["changed"] and row["ratio"]]
    gm = math.exp(sum(logs) / len(logs)) if logs else 1.0
    gm_changed = math.exp(sum(changed) / len(changed)) if changed else 1.0
    why = []
    if not batch:
        why.append("the values change no problem's plan")
    if missing:
        why.append("%d problems without a result" % len(missing))
    if slower:
        why.append("%d problems slower beyond the noise" % len(slower))
    if batch and gm_changed > 1.0 - calib_min_gain:
        why.append("geomean ratio %.4f over the %d changed problems above %.4f" % (
            gm_changed, len(batch), 1.0 - calib_min_gain))
    return {"accepted": not why, "why": why, "geomean_ratio": round(gm, 4),
            "geomean_changed": round(gm_changed, 4), "problems": len(problems), "changed": len(batch),
            "slower": slower, "missing": missing, "calib_min_gain": calib_min_gain, "rows": rows}


def seed_values(items: Sequence[str], cfgs: Dict[str, Dict[str, int]], schema: Any) -> List[str]:
    """The campaign's --values items with the calibrated keys' lists replaced: their values,
    halved and doubled (one off for the pipes), and the built-in value."""
    keys = [k for k in KEYS if any(k in c for c in cfgs.values())]
    out = [it for it in items if it.partition("=")[0].strip().lower() not in keys]
    for k in keys:
        spec = schema.key(k)
        vals = set()
        for c in cfgs.values():
            if k not in c:
                continue
            v = c[k]
            near = (v - 1, v + 1) if k == "cm_fma_pipes" else (int(round(v / 2.0)), 2 * v)
            vals |= {v} | {x for x in near if spec.check(x) is None and x >= 1}
        b = builtin_value(spec)
        if b is not None:
            vals.add(b)
        out.append("%s=%s" % (k, "|".join(str(x) for x in sorted(vals))))
    parse_values(out, schema)
    return out


def _leaf(cfg: Dict[str, Any], dtype: str) -> Dict[str, Any]:
    return {"id": "calibration_%s" % dtype, "config": dict(cfg)} if cfg else {"config": {}}


def _split(feature: str, le: float, then: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
    if "feature" not in then and "feature" not in other and then.get("config") == other.get("config"):
        return then
    return {"feature": feature, "le": le, "then": then, "else": other}


def type_tree(cfgs: Dict[str, Dict[str, Any]], over: Optional[Dict[str, Any]] = None,
              rule_id: Optional[str] = None) -> Dict[str, Any]:
    """A rule subtree that gives each weight type cfgs[type] (with over merged on top); a type
    the campaign has no problem of gets over alone. The subtree of a rule's leaf (rule_id)
    names its leaves for the rule: <rule_id>_<type> for a calibrated type, rule_id otherwise.
    The splits are TYPE_SPLITS'; ValueError for a calibrated type no split tells apart."""
    over = over or {}

    def leaf(d: str) -> Dict[str, Any]:
        if d not in cfgs:
            return dict({"config": dict(over)}, **({"id": rule_id} if rule_id else {}))
        out = _leaf(merge_config(cfgs[d], over), d)
        return dict(out, id="%s_%s" % (rule_id, d)) if rule_id and out["config"] else out

    def node(n: Any) -> Dict[str, Any]:
        if isinstance(n, str):
            return leaf(n)
        if isinstance(n, list):
            if any(d in cfgs for d in n):
                raise ValueError("weight types %s: no split on esz and bf16 tells them apart" % ", ".join(n))
            return leaf(n[0])
        return _split(n["feature"], n["le"], node(n["then"]), node(n["else"]))

    return node(TYPE_SPLITS)


def _on_team(threads: int, inside: Dict[str, Any], outside: Dict[str, Any]) -> Dict[str, Any]:
    """A rule subtree that is inside for calls on threads threads and outside for the others."""
    return {"feature": "threads", "le": threads - 1, "then": outside,
            "else": {"feature": "threads", "le": threads, "then": inside, "else": outside}}


def with_rules(rules: Optional[Dict[str, Any]], cfgs: Dict[str, Dict[str, Any]], threads: int) -> Dict[str, Any]:
    """rules that give every call of the campaign's team (threads) from M = MIN_RULE_M its
    weight type's calibrated values, which hold for that team only: a tree of them alone, or
    the trained rules with them under every leaf's (tree) or rule's config (ordered: each
    calibrated type's variant of a rule ahead of the rule, which the other types and teams
    keep, then one rule per weight type for the calls no rule takes)."""
    if rules is None:
        return {"kind": "tree", "root": {"feature": "m", "le": MIN_RULE_M - 1, "then": {"config": {}},
                                         "else": _on_team(threads, type_tree(cfgs), {"config": {}})}}
    if rules["kind"] == "tree":
        def graft(n: Dict[str, Any]) -> Dict[str, Any]:
            if "feature" in n:
                return dict(n, then=graft(n["then"]), **{"else": graft(n["else"])})
            return type_tree(cfgs, n.get("config", {}), n.get("id"))

        root = rules["root"]
        return dict(rules, root=dict(root, **{"else": _on_team(threads, graft(root["else"]), root["else"])}))
    team = [["threads", "==", threads]]
    out = []
    for r in rules["list"]:
        # The first rule that holds wins: a calibrated type's variant, then the rule itself for
        # every other type and team.
        own = r.get("config", {})
        for d, cal in sorted(cfgs.items()):
            cfg = merge_config(cal, own)
            if cal and cfg != own:
                out.append(dict(r, id="%s_%s" % (r.get("id", "rule"), d),
                                when=r.get("when", []) + _type_when(d) + team, config=cfg))
        out.append(r)
    for d, cfg in sorted(cfgs.items()):
        if cfg:
            out.append({"id": "calibration_%s" % d, "when": [["m", ">=", MIN_RULE_M]] + _type_when(d) + team,
                        "config": dict(cfg)})
    return {"kind": "ordered", "list": out}
