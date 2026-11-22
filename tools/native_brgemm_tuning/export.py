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
"""The tuning file (DESIGN section 6.2): export, validation, loader simulation.

Entries: one per problem class, a shape (every problem field but M) and an M
class.  Its winner is the candidate with the lowest geometric-mean ratio to
the built-in plan over the class's measured Ms, among the candidates that
beat the built-in plan beyond the noise at every M where they change the plan
(search phase; a candidate counts at an M through any measured candidate with
the same plan there, and cross_batch measures each M's best candidates at the
class's other Ms).  A winner must be legal at every M of its class, so the
loader never drops the entry for part of the range.  The entry is written only
when the confirmation runs (phase "confirm", the winner against the built-in
plan again) agree at every such M, and only with keys that keep the output
bits unless bit-changing keys are allowed (approval decision 6: those need a
model-level accuracy check first).

The entry's config is the full knob set the measurement fixed (DESIGN section
5: table entries set every knob, rules fill only what they leave open): the
winner's settings; the built-in value of every other searched key that has
one (the rule switches among them, so no lower layer turns a measured rule on
or off); threads_cap = the problem's threads and domain_threads = the plan's L3
domain (l3_domain_cores, else family_l3_cores; equal to "none" / auto); and an
auto pin's resolved value where it is legal for every M of the class and
reproduces the measured plan (nr, mr, kc; the unit of a blocks plan as the key
of nb and unit_tiles the winner sets, else as nb, nr times its unit tiles, or
as unit_tiles where the search took unit_tiles and not nb; col_groups only
when 0; mb only for a single-M class).

The file is version 2: its hardware block matches CPU fields (the vendor,
family and model, the ISA, the caches and the plan's L3 domain of the
machine the schema was dumped on, not its core count or SMT setting), each
key a value, a list or a {"min", "max"} range.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import TOOL_NAME
from .candidates import Candidate, Space, builtin_value, check_config
from .problems import MClass, Problem, class_of, rule_features
from .results import measured_texts, summarize, utc_now
from .rules import OPS, eval_rules
from .schema import HW_KEYS, ISA_ORDER, Schema

FORMAT = "zendnnl-native-brgemm-tuning"
VERSION = 2
TOP_KEYS = ("format", "version", "hardware", "meta", "entries", "rules")
# The version 2 hardware keys written by default: what decides how a plan
# runs, without the core count and the SMT setting.
HW_DEFAULT_KEYS = ("vendor", "family", "model", "isa", "l1d_kib", "l2_kib", "l3_kib", "l3_cores")
MATCH_NUMERIC = ("m", "n", "k", "threads", "pack_nr", "group")
MATCH_ENUM = ("dtype", "src", "dst", "epilogue")
MATCH_BOOL = ("trans_a", "dynamic", "has_bias")
MAX_ENTRIES = 65536
MAX_TREE_NODES = 65535
MAX_TREE_DEPTH = 64
MAX_ORDERED = 4096
MAX_CONDITIONS = 32
MAX_BYTES = 64 << 20
MAX_JSON_NESTING = 128
MAX_JSON_VALUES = 1 << 21


def _geomean(xs: Sequence[float]) -> float:
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def hardware_block(schema: Schema, mode: Optional[str] = None, keys: Optional[Sequence[str]] = None,
                   values: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """"hardware" of a file for the schema's machine.

    Mode "fields" (the default): the keys (HW_DEFAULT_KEYS unless given) at the
    machine's values (schema.hw_match), then values (key -> JSON value) on top; a key in
    values is written even when keys leaves it out.  Mode "none": no block, a file for
    any CPU.
    """
    mode = mode or "fields"
    if mode == "none":
        return {}
    if mode != "fields":
        raise ValueError("unknown hardware match mode %r (fields, none)" % mode)
    names = list(keys) if keys is not None else list(HW_DEFAULT_KEYS)
    names += [k for k in (values or {}) if k not in names]
    unknown = [k for k in names if k not in HW_KEYS]
    if unknown:
        raise ValueError("unknown hardware keys %s (%s)" % (", ".join(unknown), ", ".join(HW_KEYS)))
    missing = [k for k in names if k not in schema.hw_match and k not in (values or {})]
    if missing:
        raise ValueError("the schema gives no hardware %s (a dump from a build without hw_match): dump the "
                         "schema again, or leave them out with --hw-keys" % ", ".join(missing))
    return {k: (values or {}).get(k, schema.hw_match.get(k)) for k in HW_KEYS if k in names}


def parse_hw_set(items: Sequence[str]) -> Dict[str, Any]:
    """{key: JSON value} of "key=value" items for version 2 hardware keys: vendor takes a
    name or names joined by "|", isa features joined by "+", a number a value, "lo-hi",
    "lo-" (lo or more), "-hi" (hi or less) or values joined by "|"."""
    out: Dict[str, Any] = {}
    for item in items:
        key, eq, text = (t.strip() for t in item.partition("="))
        if not eq or key not in HW_KEYS or not text:
            raise ValueError("bad hardware setting %r: key=value with a key of %s" % (item, ", ".join(HW_KEYS)))
        if key == "vendor":
            names = [t.strip() for t in text.split("|")]
            if not all(names):
                raise ValueError("bad hardware setting %r: an empty vendor" % item)
            out[key] = names[0] if len(names) == 1 else names
        elif key == "isa":
            out[key] = [t.strip() for t in text.split("+")]
            bad = [f for f in out[key] if f not in ISA_ORDER]
            if bad:
                raise ValueError("bad hardware setting %r: unknown ISA features %s (%s)" % (
                    item, ", ".join(bad), ", ".join(ISA_ORDER)))
        else:
            try:
                if "|" in text:
                    out[key] = [int(t) for t in text.split("|")]
                elif "-" in text:
                    lo, _, hi = text.partition("-")
                    rng = {}
                    if lo.strip():
                        rng["min"] = int(lo)
                    if hi.strip():
                        rng["max"] = int(hi)
                    if not rng or rng.get("min", 0) > rng.get("max", rng.get("min", 0)):
                        raise ValueError(text)
                    out[key] = rng
                else:
                    out[key] = int(text)
            except ValueError:
                raise ValueError("bad hardware setting %r: a number, lo-hi, lo-, -hi or values joined by |" % item)
    return out


def config_json(config: Dict[str, Any], schema: Schema) -> Dict[str, Any]:
    return {k.name: k.to_json(config[k.name]) for k in schema.keys if k.name in config}


def config_from_json(obj: Dict[str, Any], schema: Schema) -> Dict[str, Any]:
    """The config of a JSON config object; ValueError on an unknown key or a wrong type."""
    out = {}
    for name, v in obj.items():
        if name not in schema.key_map:
            raise ValueError("unknown config key %r" % name)
        out[name] = schema.key_map[name].from_json(v)
    return out


def match_block(p: Problem, m_lo: int, m_hi: Optional[int]) -> Dict[str, Any]:
    m: Dict[str, int] = {"min": m_lo}
    if m_hi is not None:
        m["max"] = m_hi
    out = {"dtype": p.dtype, "src": p.src, "dst": p.dst, "trans_a": p.trans_a, "dynamic": bool(p.dynamic),
           "has_bias": p.has_bias, "epilogue": p.epilogue, "n": p.plan_n, "k": p.k, "threads": p.threads,
           "pack_nr": p.pack_nr, "m": m}
    if p.int4:
        out["group"] = p.group
    return out


@dataclass
class Winner:
    shape_key: str
    mclass: MClass
    problems: List[Problem]
    text: str = ""
    ratios: Dict[str, float] = field(default_factory=dict)
    significant: bool = False
    confirm: Dict[str, float] = field(default_factory=dict)
    confirmed: bool = False
    reason: str = ""
    # Every significant candidate, best first, with its search ratio per problem: the
    # confirmation falls back to the next one when a better one is not confirmed.
    alternatives: List[Tuple[str, Dict[str, float]]] = field(default_factory=list)

    @property
    def ratio(self) -> float:
        return _geomean(list(self.ratios.values()))

    def entry_id(self) -> str:
        p = self.problems[0]
        group = "-g%d" % p.group if p.int4 else ""
        return "%s%s-n%d-k%d-m%s-t%d" % (p.dtype, group, p.n, p.k, self.mclass.label(), p.threads)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.entry_id(), "class": self.mclass.label(), "ms": [p.m for p in self.problems],
                "text": self.text, "ratio": self.ratio if self.ratios else None, "ratios": self.ratios,
                "significant": self.significant, "confirm": self.confirm, "confirmed": self.confirmed,
                "reason": self.reason}


def class_ms(c: MClass, unbounded_to: int = 16384) -> List[int]:
    """The Ms a class entry must be legal at: all of a short range, else its ends,
    the first 64 Ms and the powers of two and their halves in between."""
    hi = c.hi if c.hi is not None else max(unbounded_to, 4 * c.lo)
    if hi - c.lo <= 64:
        return list(range(c.lo, hi + 1))
    ms = set(range(c.lo, c.lo + 65)) | {hi}
    v = 1
    while v <= hi:
        ms |= {x for x in (v, v + v // 2) if c.lo <= x <= hi}
        v *= 2
    return sorted(ms)


def legal_over_class(text: str, p: Problem, c: MClass, schema: Schema, uarch: Any) -> bool:
    """True when the candidate resolves to a legal plan at every M class_ms(c) holds,
    so the loader never drops the entry for part of its range."""
    from .candidates import parse_settings
    from .plan import resolve

    cfg = parse_settings(text, schema) if text else {}
    return all(resolve(p.with_m(m), cfg, uarch, schema).applied for m in class_ms(c))


def _class_groups(problems: Sequence[Problem], classes: List[MClass], have: Any) -> List[Tuple[MClass, List[Problem]]]:
    """(class, its problems) for every shape and class with results at one M at least."""
    groups: Dict[Tuple[str, int], List[Problem]] = {}
    for p in problems:
        c = class_of(p.m, classes)
        if c is not None:
            groups.setdefault((p.shape_key(), classes.index(c)), []).append(p)
    return [(classes[ci], sorted(probs, key=lambda p: p.m)) for (_, ci), probs in sorted(groups.items())
            if any(p.key() in have for p in probs)]


def _measured_by_sig(ps: Any, sp: Space) -> Dict[tuple, Any]:
    """Signature -> CandStat of the candidates measured for one problem."""
    out: Dict[tuple, Any] = {}
    if ps is None:
        return out
    for t, cs in sorted(ps.cands.items()):
        c = sp.from_text(t) if cs.ratios else None
        if c is not None:
            out.setdefault(c.sig, cs)
    return out


def _per_m(text: str, probs: List[Problem], spaces: Dict[str, Space],
           maps: Dict[str, Dict[tuple, Any]]) -> Optional[Dict[str, Any]]:
    """pkey -> the CandStat of text's plan at each M (None where it is the built-in
    plan); None when text is illegal or its plan is unmeasured at some M."""
    per: Dict[str, Any] = {}
    for p in probs:
        sp = spaces[p.key()]
        c = sp.from_text(text)
        if c is None:
            return None
        if c.sig == sp.default.sig:
            per[p.key()] = None
            continue
        cs = maps[p.key()].get(c.sig)
        if cs is None:
            return None
        per[p.key()] = cs
    return per


def select_winners(records: List[Dict[str, Any]], problems: Sequence[Problem], classes: List[MClass],
                   spaces: Dict[str, Space], min_gain: float = 0.02, z: float = 2.0,
                   check_range: bool = True) -> List[Winner]:
    """The winner of every (shape, M class) with search results.

    A candidate counts at an M through any measured candidate with the same
    plan there; at an M where it is the built-in plan it is neutral.  It must
    gain beyond the noise at every M where it changes the plan (and change it
    at one at least), and with check_range be legal at every M of the class.
    """
    summ = summarize(records, phases=["search"], z=z)
    out = []
    for mclass, probs in _class_groups(problems, classes, summ):
        w = Winner(probs[0].shape_key(), mclass, probs)
        maps = {p.key(): _measured_by_sig(summ.get(p.key()), spaces[p.key()]) for p in probs}
        texts = sorted({t for p in probs if p.key() in summ for t, cs in summ[p.key()].cands.items() if cs.ratios})
        rows = {}
        for t in texts:
            per = _per_m(t, probs, spaces, maps)
            if per is None or all(v is None for v in per.values()):
                continue
            if check_range and not legal_over_class(t, probs[0], mclass, spaces[probs[0].key()].schema,
                                                    spaces[probs[0].key()].uarch):
                continue
            rows[t] = per
        if not rows:
            w.reason = "no candidate measured at every M of the class and legal over its range"
            out.append(w)
            continue

        def score(t: str) -> float:
            return _geomean([cs.ratio if cs is not None else 1.0 for cs in rows[t].values()])

        sig = [t for t, per in rows.items() if all(cs.significant(min_gain) for cs in per.values() if cs is not None)]
        best = min(sig or sorted(rows), key=score)
        w.text = best
        w.ratios = {k: cs.ratio if cs is not None else 1.0 for k, cs in rows[best].items()}
        w.significant = bool(sig)
        w.alternatives = [(t, {k: cs.ratio if cs is not None else 1.0 for k, cs in rows[t].items()})
                          for t in sorted(sig, key=lambda t: (score(t), t))]
        if not sig:
            w.reason = "no candidate beats the built-in plan beyond the noise at every M (best %s, ratio %.3f)" % (
                best, w.ratio)
        out.append(w)
    return out


def _changed(w: Winner, spaces: Dict[str, Space]) -> List[Tuple[Problem, Candidate]]:
    """The Ms where the winner's plan differs from the built-in plan, with its candidate there."""
    out = []
    for p in w.problems:
        sp = spaces[p.key()]
        c = sp.from_text(w.text)
        if c is not None and c.sig != sp.default.sig:
            out.append((p, c))
    return out


def _confirmation(w: Winner, summ: Dict[str, Any], spaces: Dict[str, Space],
                  min_gain: float) -> Tuple[bool, Dict[str, float], str]:
    """(confirmed, confirm ratio per problem, reason) of w.text from the confirm-phase summary."""
    ratios: Dict[str, float] = {}
    for p, _ in _changed(w, spaces):
        ps = summ.get(p.key())
        cs = ps.cands.get(w.text) if ps else None
        if cs is None or not cs.ratios:
            return False, ratios, "no confirmation run at M=%d" % p.m
        ratios[p.key()] = cs.ratio
        if not cs.significant(min_gain):
            return False, ratios, "the confirmation run disagrees at M=%d (ratio %.3f, noise %.3f)" % (
                p.m, cs.ratio, cs.noise)
    return True, ratios, ""


def attach_confirmation(winners: List[Winner], records: List[Dict[str, Any]], spaces: Dict[str, Space],
                        min_gain: float = 0.02, z: float = 2.0) -> None:
    """Sets confirmed / reason from the records of phase "confirm".  When the best
    candidate is not confirmed, the next significant candidate (in search order)
    that is confirmed takes its place."""
    summ = summarize(records, phases=["confirm"], z=z)
    for w in winners:
        if not w.text or not w.significant:
            continue
        best_text, best_ratios = w.text, w.ratios
        ok, conf, why = _confirmation(w, summ, spaces, min_gain)
        first_conf, first_why = conf, why
        if not ok:
            for t, r in w.alternatives:
                if t == best_text:
                    continue
                w.text, w.ratios = t, r
                ok, conf, alt_why = _confirmation(w, summ, spaces, min_gain)
                if ok:
                    why = "%s not confirmed (%s); next confirmed candidate used" % (best_text, first_why)
                    break
        if not ok:
            w.text, w.ratios, conf, why = best_text, best_ratios, first_conf, first_why
        w.confirmed, w.confirm, w.reason = ok, conf, why if not ok else ("" if w.text == best_text else why)


def confirmed_before(w: Winner, records: List[Dict[str, Any]], spaces: Dict[str, Space], z: float = 2.0,
                     summ: Optional[Dict[str, Any]] = None) -> bool:
    """True when w.text already has confirm-phase results at every M where it changes the plan."""
    summ = summ if summ is not None else summarize(records, phases=["confirm"], z=z)
    for p, _ in _changed(w, spaces):
        ps = summ.get(p.key())
        cs = ps.cands.get(w.text) if ps else None
        if cs is None or not cs.ratios:
            return False
    return True


def fallback_winners(winners: List[Winner], records: List[Dict[str, Any]], spaces: Dict[str, Space],
                     min_gain: float = 0.02, z: float = 2.0) -> List[Winner]:
    """The significant winners that no confirmed candidate covers yet, each set to its
    best alternative without confirmation results (winners left without one are dropped)."""
    attach_confirmation(winners, records, spaces, min_gain, z)
    summ = summarize(records, phases=["confirm"], z=z)
    out = []
    for w in winners:
        if not w.significant or w.confirmed:
            continue
        for t, r in w.alternatives:
            w.text, w.ratios = t, r
            if not confirmed_before(w, records, spaces, z, summ):
                out.append(w)
                break
    return out


def confirm_batch(winners: List[Winner], spaces: Dict[str, Space]) -> Dict[str, List[Candidate]]:
    """The candidates to re-run against the built-in plan: every significant winner
    at the Ms where it changes the plan."""
    out: Dict[str, List[Candidate]] = {}
    for w in winners:
        if w.text and w.significant:
            for p, c in _changed(w, spaces):
                out[p.key()] = [c]
    return out


def cross_batch(records: List[Dict[str, Any]], problems: Sequence[Problem], classes: List[MClass],
                spaces: Dict[str, Space], top: int = 3, min_gain: float = 0.02,
                z: float = 2.0) -> Dict[str, List[Candidate]]:
    """For classes with several Ms: the best `top` candidates of each M, at the
    class's other Ms where their plan is not measured yet (so a class winner can
    be judged at all its Ms)."""
    search = [r for r in records if r.get("phase") == "search"]
    summ = summarize(search, z=z)
    tried = measured_texts(search)
    out: Dict[str, List[Candidate]] = {}
    for mclass, probs in _class_groups(problems, classes, summ):
        if len(probs) < 2:
            continue
        pool = set()
        for p in probs:
            ps = summ.get(p.key())
            good = [c for c in ps.cands.values() if c.ratios and c.significant(min_gain)] if ps else []
            pool |= {c.text for c in sorted(good, key=lambda c: c.ratio)[:top]}
        for p in probs:
            sp = spaces[p.key()]
            known = {c.sig for c in (sp.from_text(t) for t in tried.get(p.key(), ())) if c is not None}
            known.add(sp.default.sig)
            for t in sorted(pool):
                c = sp.from_text(t)
                if c is not None and c.sig not in known:
                    known.add(c.sig)
                    out.setdefault(p.key(), []).append(c)
    return out


def full_config(w: Winner, spaces: Dict[str, Space], schema: Schema, uarch: Any) -> Dict[str, Any]:
    """The full knob set of a winner (see the module text)."""
    sps = [spaces[p.key()] for p in w.problems]
    cands = [sp.from_text(w.text) for sp in sps]
    if any(c is None for c in cands):
        raise ValueError("%s is not legal at every M of %s" % (w.text, w.entry_id()))
    base = dict(cands[0].config)
    cfg = dict(base)
    single_m = w.mclass.hi == w.mclass.lo
    # nb replaces unit_tiles: an nb pin beside a winner's unit_tiles would take other tiles
    # at an M of the class where nr differs from the measured Ms.
    unit_key = next((k for k in ("nb", "unit_tiles") if k in base), "nb" if "nb" in sps[0].keys else "unit_tiles")
    for key in sps[0].keys:
        if key in cfg:
            continue
        spec = schema.key(key)
        if builtin_value(spec) is not None:
            cfg[key] = builtin_value(spec)
        elif key == "threads_cap":
            cfg[key] = w.problems[0].threads
        elif key == "domain_threads" and uarch.plan_domain:
            cfg[key] = uarch.plan_domain
        elif key in ("nr", "mr", "kc", "col_groups", "mb", unit_key):
            vals = {c.plan.nr * c.plan.unit_tiles if key == "nb" else getattr(c.plan, key) for c in cands}
            if len(vals) != 1:
                continue
            v = vals.pop()
            if (key == "col_groups" and v != 0) or (key == "mb" and not single_m) or spec.check(v):
                continue
            trial = dict(cfg, **{key: v})
            made = [sp.make(trial) for sp in sps]
            if all(m is not None and m.sig == c.sig for m, c in zip(made, cands)):
                cfg[key] = v
    made = [sp.make(cfg) for sp in sps]
    if not all(m is not None and m.sig == c.sig for m, c in zip(made, cands)):
        return base
    return cfg


def build_entries(winners: List[Winner], spaces: Dict[str, Space], schema: Schema, uarch: Any,
                  allow_bit_changing: bool = False,
                  require_confirm: bool = True) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """(entries, skipped classes with the reason)."""
    entries, skipped = [], []
    for w in winners:
        why = w.reason
        # An M = 1 call runs on the GEMV kernels, which take none of an entry's tiles: an entry
        # starts at M 2.
        m_lo = max(w.mclass.lo, 2)
        if w.mclass.hi is not None and w.mclass.hi < m_lo:
            skipped.append({"id": w.entry_id(), "reason": "M = 1 calls run on the GEMV kernels"})
            continue
        if w.text and w.significant and not w.confirmed and require_confirm and not why:
            why = "no confirmation run yet (tune.py confirm)"
        if w.text and w.significant and (w.confirmed or not require_confirm):
            cfg = full_config(w, spaces, schema, uarch)
            bad = [k for k in cfg if not schema.key(k).same_bits(w.problems[0].dtype)]
            if bad and not allow_bit_changing:
                why = "changes output bits (%s): needs a model-level accuracy check" % ",".join(bad)
            else:
                p = w.problems[0]
                entries.append({"id": w.entry_id(), "match": match_block(p, m_lo, w.mclass.hi),
                                "config": config_json(cfg, schema)})
                continue
        skipped.append({"id": w.entry_id(), "reason": why or "no winner"})
    entries.sort(key=lambda e: (e["match"]["dtype"], e["match"]["n"], e["match"]["k"], e["match"]["m"]["min"]))
    return entries, skipped


def layered_plan(p: Problem, layers: Sequence[Tuple[str, Dict[str, Any]]], schema: Schema,
                 uarch: Any) -> Tuple[str, Dict[str, Any], Any]:
    """resolve_layers: (the source names joined by "+", the applied config, the plan) of the
    configs of layers, lowest precedence first (rules, then table).  Each layer is taken
    without the keys the problem's INT8 scheme does not apply, and a layer left with no key
    is not applied; the merged config stands when its plan is legal, else, from the highest
    layer down, each layer stays only if the plan with it under the layers kept above it is."""
    from .plan import builtin_plan, resolve, scheme_config

    cfgs = [(n, scheme_config(c, p, schema)) for n, c in layers]
    cfgs = [(n, c) for n, c in cfgs if c]
    merged: Dict[str, Any] = {}
    for _, c in cfgs:
        merged = merge_config(merged, c)
    if cfgs:
        r = resolve(p, merged, uarch, schema)
        if r.applied:
            return "+".join(n for n, _ in cfgs), merged, r.plan
    kept: Dict[str, Any] = {}
    names: List[str] = []
    plan = builtin_plan(p, uarch, schema)
    for n, c in reversed(cfgs):
        trial = merge_config(c, kept)
        r = resolve(p, trial, uarch, schema)
        if r.applied:
            kept, plan = trial, r.plan
            names.insert(0, n)
    return "+".join(names) or "heuristic", kept, plan


def entry_plans(entries: List[Dict[str, Any]], problems: Sequence[Problem], schema: Schema,
                uarch: Any, rules: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """The plan the file names for each problem an entry matches, and with rules for each other
    problem a rule gives a config: the [MATMUL.NATIVE_BRGEMM] fields the library logs for it
    under the file (source= the layers applied, cfg= their config as the call applies it, the
    plan's fields, and for a gated call its pairing= and, where its tiles pair wider, the fields of
    the paired plan, paired), and whether that is the built-in plan."""
    from .candidates import format_settings
    from .plan import builtin_plan, log_pairing, paired_log_fields, plan_log_fields

    def rule_config(p: Problem) -> Dict[str, Any]:
        r = eval_rules(rules, rule_features(p, uarch, schema.hw)) if rules else None
        return config_from_json(r, schema) if r else {}

    out = []

    def record(rid: str, p: Problem, layers: List[Tuple[str, Dict[str, Any]]]) -> None:
        source, cfg, plan = layered_plan(p, layers, schema, uarch)
        fields = plan_log_fields(plan, p)
        out.append({"id": rid, "problem": p.key(), "source": source,
                    "cfg": (format_settings(cfg, schema) or "-") if source != "heuristic" else "-",
                    "plan": fields, "builtin": fields == plan_log_fields(builtin_plan(p, uarch, schema), p)})
        paired = paired_log_fields(plan, p)
        if paired:
            out[-1]["paired"] = paired
        pairing = log_pairing(plan, p)
        if pairing:
            out[-1]["pairing"] = pairing

    named = set()
    for e in entries:
        cfg = config_from_json(e.get("config", {}), schema)
        for p in problems:
            if match_entry(e.get("match", {}), p):
                named.add(p.key())
                record(e.get("id", ""), p, [("rules", rule_config(p)), ("table", cfg)])
    for p in problems:
        if rules and p.key() not in named and rule_config(p):
            record("rules", p, [("rules", rule_config(p))])
    return out


def exclude_entries(entries: List[Dict[str, Any]],
                    items: Sequence[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """(kept entries, skipped records) without the entry ids items name: comma-separated
    ids, or "@file" with one id per line (first word; "#" lines ignored)."""
    drop = set()
    for item in items:
        if item.startswith("@"):
            with open(item[1:], encoding="utf-8") as f:
                drop |= {t.split()[0] for t in f if t.strip() and not t.startswith("#")}
        else:
            drop |= {t.strip() for t in item.split(",") if t.strip()}
    return ([e for e in entries if e["id"] not in drop],
            [{"id": e["id"], "reason": "excluded (--exclude)"} for e in entries if e["id"] in drop])


def build_doc(entries: List[Dict[str, Any]], rules: Optional[Dict[str, Any]], schema: Schema,
              hw_mode: Optional[str] = None, meta: Optional[Dict[str, Any]] = None,
              hw_keys: Optional[Sequence[str]] = None, hw_values: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    doc: Dict[str, Any] = {"format": FORMAT, "version": VERSION}
    hw = hardware_block(schema, hw_mode, hw_keys, hw_values)
    if hw:
        doc["hardware"] = hw
    doc["meta"] = dict(meta or {}, tool=TOOL_NAME, created=utc_now())
    if entries:
        doc["entries"] = entries
    if rules:
        doc["rules"] = rules
    return doc


def _overlap(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """True when some problem matches both match blocks (a key absent from one block matches anything)."""
    for k in set(a) | set(b):
        if k not in a or k not in b:
            continue
        va, vb = a[k], b[k]
        if isinstance(va, dict) or isinstance(vb, dict):
            lo = max(v.get("min", -math.inf) if isinstance(v, dict) else v for v in (va, vb))
            hi = min(v.get("max", math.inf) if isinstance(v, dict) else v for v in (va, vb))
            if lo > hi:
                return False
        elif isinstance(va, list) or isinstance(vb, list):
            if not set(va if isinstance(va, list) else [va]) & set(vb if isinstance(vb, list) else [vb]):
                return False
        elif va != vb:
            return False
    return True


def merge_docs(docs: Sequence[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """One file of the entries of several files (name, document) with one hardware block and no rules; the
    entries in the order build_entries sorts them, the inputs' plans (meta.plans) in meta.plans and the rest
    of their meta blocks under meta.merged.  ValueError when the hardware blocks differ, a file has rules, two
    entries share an id, two entries match one problem (the loader would apply only the first), or two files
    name different plans for one problem."""
    if not docs:
        raise ValueError("no file to merge")
    hw = docs[0][1].get("hardware", {})
    entries: List[Dict[str, Any]] = []
    for name, d in docs:
        if not is_current(d):
            raise ValueError("%s is not a version %d file" % (name, VERSION))
        if d.get("hardware", {}) != hw:
            raise ValueError("%s has another hardware block than %s" % (name, docs[0][0]))
        if d.get("rules"):
            raise ValueError("%s has rules: merge takes table entries only" % name)
        entries += d.get("entries", [])
    ids = [e.get("id") for e in entries]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise ValueError("entries %s appear more than once" % ", ".join(map(str, dup)))
    for i, a in enumerate(entries):
        for b in entries[i + 1:]:
            if _overlap(a.get("match", {}), b.get("match", {})):
                raise ValueError("entries %s and %s match the same problems" % (a.get("id"), b.get("id")))
    entries.sort(key=lambda e: (e["match"]["dtype"], e["match"]["n"], e["match"]["k"], e["match"]["m"]["min"]))
    plans: Dict[str, Dict[str, Any]] = {}
    for name, d in docs:
        for pl in d.get("meta", {}).get("plans", []):
            if plans.setdefault(pl["problem"], pl) != pl:
                raise ValueError("%s names another plan for %s than an earlier file" % (name, pl["problem"]))
    doc: Dict[str, Any] = {"format": FORMAT, "version": VERSION}
    if hw:
        doc["hardware"] = hw
    doc["meta"] = {"tool": TOOL_NAME, "created": utc_now(),
                   "merged": [{"file": os.path.basename(name), "entries": len(d.get("entries", [])),
                               "meta": {k: v for k, v in d.get("meta", {}).items() if k != "plans"}}
                              for name, d in docs]}
    if any("plans" in d.get("meta", {}) for _, d in docs):
        doc["meta"]["plans"] = list(plans.values())
    if entries:
        doc["entries"] = entries
    return doc


# ---------------------------------------------------------------- validation

def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _num_or_range(v: Any, ints: bool) -> bool:
    ok = (lambda x: isinstance(x, int) and not isinstance(x, bool)) if ints else _is_num
    if ok(v):
        return True
    if isinstance(v, dict):
        return bool(v) and set(v) <= {"min", "max"} and all(ok(x) for x in v.values())
    return False


def _int_values(v: Any) -> bool:
    """An integer, a non-empty list of integers or {"min", "max"} (either bound) with min <= max."""
    def is_int(x: Any) -> bool:
        return isinstance(x, int) and not isinstance(x, bool)

    if is_int(v):
        return True
    if isinstance(v, list):
        return bool(v) and all(is_int(x) for x in v)
    if isinstance(v, dict):
        return (bool(v) and set(v) <= {"min", "max"} and all(is_int(x) for x in v.values())
                and v.get("min", v.get("max")) <= v.get("max", v.get("min")))
    return False


def _check_hardware(hw: Dict[str, Any]) -> List[str]:
    """What makes a hardware block unusable for the loader."""
    errs: List[str] = []
    for k, val in hw.items():
        if k not in HW_KEYS:
            errs.append("hardware.%s is not a version 2 hardware key" % k)
        elif k == "vendor":
            names = [val] if isinstance(val, str) else val
            if not (isinstance(names, list) and names and all(isinstance(x, str) for x in names)):
                errs.append("hardware.vendor is not a string or a non-empty list of strings")
        elif k == "isa":
            names = [val] if isinstance(val, str) else val
            if not (isinstance(names, list) and all(isinstance(x, str) for x in names)):
                errs.append("hardware.isa is not a string or a list of strings")
            elif any(x not in ISA_ORDER for x in names):
                errs.append("hardware.isa names features the loader does not know: %s" % ", ".join(
                    x for x in names if x not in ISA_ORDER))
        elif not _int_values(val):
            errs.append("hardware.%s is not an integer, a list of integers or {min, max}" % k)
    return errs


def _check_config_obj(obj: Any, schema: Schema, where: str) -> List[str]:
    if not isinstance(obj, dict):
        return ["%s: config is not an object" % where]
    try:
        cfg = config_from_json(obj, schema)
    except ValueError as e:
        return ["%s: %s" % (where, e)]
    why = check_config(cfg, schema)
    return ["%s: %s" % (where, why)] if why else []


def _check_rules(rules: Any, schema: Schema) -> List[str]:
    if not isinstance(rules, dict) or rules.get("kind") not in ("tree", "ordered"):
        return ["rules: kind must be tree or ordered"]
    feats = set(schema.features)
    errs: List[str] = []
    if rules["kind"] == "tree":
        count = [0]

        def walk(n: Any, depth: int) -> None:
            count[0] += 1
            if depth > MAX_TREE_DEPTH:
                errs.append("rules: the tree is deeper than %d" % MAX_TREE_DEPTH)
                return
            if count[0] > MAX_TREE_NODES:
                return
            if not isinstance(n, dict):
                errs.append("rules: a node is not an object")
            elif "feature" in n:
                if n["feature"] not in feats:
                    errs.append("rules: unknown feature %r" % n["feature"])
                if not _is_num(n.get("le")) or "then" not in n or "else" not in n:
                    errs.append("rules: malformed inner node %r" % {k: n[k] for k in n if k not in ("then", "else")})
                    return
                walk(n["then"], depth + 1)
                walk(n["else"], depth + 1)
            else:
                errs.extend(_check_config_obj(n.get("config"), schema, "rules leaf %s" % n.get("id", "")))

        walk(rules.get("root"), 1)
        if count[0] > MAX_TREE_NODES:
            errs.append("rules: more than %d nodes" % MAX_TREE_NODES)
        return errs
    lst = rules.get("list")
    if not isinstance(lst, list) or len(lst) > MAX_ORDERED:
        return ["rules: list missing or longer than %d" % MAX_ORDERED]
    for i, r in enumerate(lst):
        when = r.get("when", []) if isinstance(r, dict) else None
        if not isinstance(when, list) or len(when) > MAX_CONDITIONS:
            errs.append("rules: rule %d has a malformed when or more than %d conditions" % (i, MAX_CONDITIONS))
            continue
        for c in when:
            if not (isinstance(c, list) and len(c) == 3 and c[0] in feats and c[1] in OPS and _is_num(c[2])):
                errs.append("rules: rule %d has a bad condition %r" % (i, c))
        errs.extend(_check_config_obj(r.get("config"), schema, "rules rule %d" % i))
    return errs


def validate_doc(doc: Any, schema: Schema) -> Tuple[List[str], List[str]]:
    """(errors, warnings) of a tuning-file object against DESIGN section 6.2."""
    errs: List[str] = []
    warns: List[str] = []
    if not isinstance(doc, dict):
        return ["the file is not a JSON object"], warns
    if doc.get("format") != FORMAT:
        errs.append("format must be %r" % FORMAT)
    if not is_current(doc):
        errs.append("version must be the integer %d" % VERSION)
    for k in doc:
        if k not in TOP_KEYS:
            warns.append("unknown top-level key %r (ignored by the loader)" % k)
    hw = doc.get("hardware", {})
    if not isinstance(hw, dict):
        errs.append("hardware is not an object")
    else:
        errs.extend(_check_hardware(hw))
    entries = doc.get("entries", [])
    if not isinstance(entries, list) or len(entries) > MAX_ENTRIES:
        errs.append("entries is not a list of at most %d" % MAX_ENTRIES)
        entries = []
    enum_values = {k: v for k, v in schema.match_keys.items() if isinstance(v, list)}
    for i, e in enumerate(entries):
        where = "entry %d (%s)" % (i, e.get("id", "") if isinstance(e, dict) else "")
        if not isinstance(e, dict) or not isinstance(e.get("match", {}), dict):
            errs.append("%s: not an object with a match object" % where)
            continue
        for k, val in e.get("match", {}).items():
            if k in MATCH_NUMERIC:
                good = _num_or_range(val, ints=True) or (isinstance(val, list) and val and all(
                    isinstance(x, int) and not isinstance(x, bool) for x in val))
            elif k in MATCH_ENUM:
                names = val if isinstance(val, list) else [val]
                good = bool(names) and all(isinstance(x, str) and x in enum_values.get(k, [x]) for x in names)
            elif k in MATCH_BOOL:
                good = isinstance(val, bool)
            else:
                errs.append("%s: unknown match key %r" % (where, k))
                continue
            if not good:
                errs.append("%s: bad match value %s=%r" % (where, k, val))
        errs.extend(_check_config_obj(e.get("config", {}), schema, where))
    if "rules" in doc:
        errs.extend(_check_rules(doc["rules"], schema))
    try:
        data = file_bytes(doc)
    except ValueError as e:
        errs.append("the file is not strict JSON, which the loader parses whole or ignores: %s" % e)
        return errs, warns
    depth, values = json_shape(data.decode("utf-8"))
    if len(data) > MAX_BYTES:
        errs.append("the file is larger than 64 MiB")
    if depth > MAX_JSON_NESTING:
        errs.append("the file nests arrays and objects deeper than %d levels" % MAX_JSON_NESTING)
    if values > MAX_JSON_VALUES:
        errs.append("the file has more than %d values" % MAX_JSON_VALUES)
    return errs, warns


def file_bytes(doc: Any) -> bytes:
    """The file export and merge write for doc: strict JSON in UTF-8, the only text the loader parses.
    ValueError for a NaN or infinite number or a string with an unpaired surrogate, which it refuses."""
    return (json.dumps(doc, indent=1, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


_JSON_STRING = re.compile(r'"(?:[^"\\]|\\.)*"')


def json_shape(text: str) -> Tuple[int, int]:
    """(nesting depth, values) of JSON text as the loader bounds them before it parses: the deepest nesting
    of arrays and objects, and the opening brackets and commas outside strings, plus one."""
    bare = _JSON_STRING.sub('""', text)
    depth = deepest = 0
    for c in re.findall(r"[\[\]{}]", bare):
        depth += 1 if c in "[{" else -1
        deepest = max(deepest, depth)
    return deepest, 1 + sum(bare.count(c) for c in "[{,")


# ---------------------------------------------------------------- loader simulation

def is_current(doc: Dict[str, Any]) -> bool:
    """The file has the version the loader reads; the loader ignores any other."""
    v = doc.get("version")
    return isinstance(v, int) and not isinstance(v, bool) and v == VERSION


def _in(val: Any, x: Any) -> bool:
    if isinstance(val, dict):
        return ("min" not in val or x >= val["min"]) and ("max" not in val or x <= val["max"])
    if isinstance(val, list):
        return x in val
    return x == val


def _mismatch(hw_block: Dict[str, Any], cpu: Dict[str, Any]) -> Optional[str]:
    """The first key of a hardware block that does not hold on the machine with the given
    hw_match: an unknown key first, then the keys in the loader's order.  A key the machine
    does not report does not hold; isa holds when the machine has every feature it names."""
    for k in hw_block:
        if k not in HW_KEYS:
            return k
    for k in HW_KEYS:
        if k not in hw_block:
            continue
        val = hw_block[k]
        if k == "isa":
            if not set([val] if isinstance(val, str) else val) <= set(cpu.get("isa", [])):
                return k
        elif k not in cpu:
            return k
        elif k == "vendor":
            if cpu[k] not in ([val] if isinstance(val, str) else val):
                return k
        elif not _in(val, cpu[k]):
            return k
    return None


def hardware_mismatch(hw_block: Dict[str, Any], schema: Schema) -> Optional[str]:
    """The first hardware key that does not match the schema's machine, or None."""
    if not hw_block:
        return None
    return _mismatch(hw_block, schema.hw_match)


def match_entry(match: Dict[str, Any], p: Problem) -> bool:
    """True when the entry's match block holds for p; its n is the width the library matches, the
    padded width of a weight not whole 32-column blocks (plan_n), as match_block writes it."""
    for k, val in match.items():
        x = p.plan_n if k == "n" else bool(p.dynamic) if k == "dynamic" else getattr(p, k)
        if not _in(val, x):
            return False
    return True


def merge_config(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """merge_native_config: the keys overlay sets replace base's; schedule and col_groups choose one
    schedule together, and nb and unit_tiles one unit width, so an overlay that sets either key of a
    pair replaces both."""
    out = dict(base)
    for pair in (("schedule", "col_groups"), ("nb", "unit_tiles")):
        if any(k in overlay for k in pair):
            for k in pair:
                out.pop(k, None)
    out.update(overlay)
    return out


def resolve_config(doc: Dict[str, Any], p: Problem, schema: Schema,
                   uarch: Any) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """The merged config the file gives p (rules, then the first matching entry), and the layers used.

    None when the file is not version 2 or the hardware does not match.
    """
    if not is_current(doc) or hardware_mismatch(doc.get("hardware", {}), schema):
        return None, []
    cfg: Dict[str, Any] = {}
    layers: List[str] = []
    if doc.get("rules"):
        r = eval_rules(doc["rules"], rule_features(p, uarch, schema.hw))
        if r:
            cfg = merge_config(cfg, r)
            layers.append("rules")
    for e in doc.get("entries", []):
        if match_entry(e.get("match", {}), p):
            cfg = merge_config(cfg, e.get("config", {}))
            layers.append("table:%s" % e.get("id", ""))
            break
    return cfg, layers
