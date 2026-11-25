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
"""Learned rules: a cost-sensitive decision tree over the rule features.

The rules learn from, and apply to, the calls from M = 2: an M = 1 call runs on
the GEMV kernels, whose keys come from entries.  The classes are a few measured configs (the winners of most
problems) plus the built-in plan.  cost[i, c] is config c's time on problem i
relative to the built-in plan (1 for the built-in plan); a cell never measured costs
`unknown` (slightly above 1, so leaves prefer configs measured on their
problems).  The tree splits on "feature <= threshold" (features of DESIGN
section 6.2, as the loader evaluates them) to minimize the weighted cost when
each leaf applies its one best config; a leaf whose best is the built-in plan
exports an empty config (no opinion).  evaluate() is K-fold by shape (dtype,
N, K): fit on the other folds, apply to the held-out problems, and report the
geometric-mean ratio to the built-in plan on measured cells (None when no
held-out cell is measured, as with a single shape, which no fold can hold out).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # eval_rules and the JSON forms need no numpy; training does
    np = None

from .problems import Problem
from .results import ProblemSummary


class CostTree:
    def __init__(self, max_depth: int = 4, min_leaf: int = 3, min_gain: float = 1e-3):
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.min_gain = min_gain  # relative cost reduction a split must bring

    def fit(self, X: np.ndarray, C: np.ndarray, w: Optional[np.ndarray] = None) -> "CostTree":
        if np is None:
            raise RuntimeError("training rules needs numpy")
        self.X, self.C = X, C
        self.w = np.ones(len(X)) if w is None else w
        self.root = self._build(np.arange(len(X)), 0)
        return self

    def _leaf(self, idx: np.ndarray) -> Tuple[int, float]:
        tot = (self.w[idx, None] * self.C[idx]).sum(axis=0)
        c = int(np.argmin(tot))
        return c, float(tot[c])

    def _build(self, idx: np.ndarray, depth: int) -> Dict[str, Any]:
        cls, cost = self._leaf(idx)
        node: Dict[str, Any] = {"class": cls, "cost": cost, "n": int(len(idx))}
        if depth >= self.max_depth or len(idx) < 2 * self.min_leaf:
            return node
        best = None
        for f in range(self.X.shape[1]):
            order = idx[np.argsort(self.X[idx, f], kind="stable")]
            xs = self.X[order, f]
            cum = np.cumsum(self.w[order, None] * self.C[order], axis=0)
            pos = np.nonzero(xs[:-1] < xs[1:])[0]
            pos = pos[(pos + 1 >= self.min_leaf) & (len(idx) - pos - 1 >= self.min_leaf)]
            if not len(pos):
                continue
            s = cum[pos].min(axis=1) + (cum[-1] - cum[pos]).min(axis=1)
            j = int(np.argmin(s))
            if s[j] < cost * (1.0 - self.min_gain) and (best is None or s[j] < best[0]):
                best = (float(s[j]), f, float(0.5 * (xs[pos[j]] + xs[pos[j] + 1])))
        if best is None:
            return node
        _, f, thr = best
        mask = self.X[idx, f] <= thr
        then, other = self._build(idx[mask], depth + 1), self._build(idx[~mask], depth + 1)
        if "feature" not in then and "feature" not in other and then["class"] == other["class"]:
            return node
        node.update(feature=f, le=thr, then=then, other=other)
        return node

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = np.empty(len(X), dtype=int)
        for i, x in enumerate(X):
            n = self.root
            while "feature" in n:
                n = n["then"] if x[n["feature"]] <= n["le"] else n["other"]
            out[i] = n["class"]
        return out

    def size(self) -> Tuple[int, int]:
        """(nodes, depth)."""
        def walk(n: Dict[str, Any], d: int) -> Tuple[int, int]:
            if "feature" not in n:
                return 1, d
            a, da = walk(n["then"], d + 1)
            b, db = walk(n["other"], d + 1)
            return 1 + a + b, max(da, db)
        return walk(self.root, 0)


# The loader applies the rules to every call, and they learn the plans of the tiles: an M = 1
# call runs on the GEMV kernels, which take none of them (an entry of M 1 sets their keys), so
# no rule may reach it.
MIN_RULE_M = 2


def tree_json(tree: CostTree, features: Sequence[str], configs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """{"kind": "tree", "root": ...} of DESIGN section 6.2, under a root split that gives
    calls below MIN_RULE_M no config."""
    def node(n: Dict[str, Any]) -> Dict[str, Any]:
        if "feature" not in n:
            leaf: Dict[str, Any] = {"config": dict(configs[n["class"]])}
            if configs[n["class"]]:
                leaf = {"id": "c%d" % n["class"], "config": dict(configs[n["class"]])}
            return leaf
        return {"feature": features[n["feature"]], "le": n["le"], "then": node(n["then"]), "else": node(n["other"])}
    return {"kind": "tree", "root": {"feature": "m", "le": MIN_RULE_M - 1, "then": {"config": {}},
                                     "else": node(tree.root)}}


def ordered_json(tree: CostTree, features: Sequence[str], configs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The ordered form: one rule per leaf with an opinion, conditions from its path after
    m >= MIN_RULE_M."""
    rules: List[Dict[str, Any]] = []

    def walk(n: Dict[str, Any], when: List[List[Any]]) -> None:
        if "feature" not in n:
            if configs[n["class"]]:
                rules.append({"id": "c%d" % n["class"], "when": when, "config": dict(configs[n["class"]])})
            return
        f = features[n["feature"]]
        walk(n["then"], when + [[f, "<=", n["le"]]])
        walk(n["other"], when + [[f, ">", n["le"]]])

    walk(tree.root, [["m", ">=", MIN_RULE_M]])
    return {"kind": "ordered", "list": rules}


OPS = {"<": lambda a, b: a < b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b,
       ">=": lambda a, b: a >= b, "==": lambda a, b: a == b, "!=": lambda a, b: a != b}


def eval_rules(rules: Dict[str, Any], feats: Dict[str, float]) -> Optional[Dict[str, Any]]:
    """The config the rules give for features feats (None: no rule holds)."""
    if rules.get("kind") == "tree":
        n = rules["root"]
        while "feature" in n:
            n = n["then"] if float(feats[n["feature"]]) <= float(n["le"]) else n["else"]
        return n.get("config", {})
    for r in rules.get("list", []):
        if all(OPS[op](float(feats[f]), float(v)) for f, op, v in r.get("when", [])):
            return r.get("config", {})
    return None


def choose_classes(summaries: Dict[str, ProblemSummary], min_gain: float, max_classes: int) -> List[str]:
    """"" (built-in) and the texts that win the most problems beyond the noise."""
    score: Dict[str, float] = {}
    for ps in summaries.values():
        b = ps.best(min_gain, significant_only=True)
        if b is not None:
            score[b.text] = score.get(b.text, 0.0) + (1.0 - b.ratio)
    return [""] + [t for t, _ in sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))[:max_classes]]


def cost_matrix(pkeys: Sequence[str], summaries: Dict[str, ProblemSummary], classes: Sequence[str],
                unknown: float = 1.05,
                predict: Optional[Callable[[str, str], Optional[float]]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """(cost, measured) of every problem and class; see the module text."""
    C = np.full((len(pkeys), len(classes)), unknown)
    M = np.zeros_like(C, dtype=bool)
    for i, k in enumerate(pkeys):
        ps = summaries.get(k)
        for j, t in enumerate(classes):
            if not t:
                C[i, j], M[i, j] = 1.0, True
            elif ps is not None and t in ps.cands and ps.cands[t].ratios:
                C[i, j], M[i, j] = ps.cands[t].ratio, True
            elif predict is not None:
                v = predict(k, t)
                if v is not None:
                    C[i, j] = v
    return C, M


def _geomean(xs: Sequence[float]) -> Optional[float]:
    """None for no value: the report goes into the tuning file, which holds strict JSON only."""
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def evaluate(X: np.ndarray, C: np.ndarray, M: np.ndarray, groups: Sequence[Any], folds: int,
             make_tree: Callable[[], CostTree]) -> Dict[str, Any]:
    """Held-out quality: K folds over groups (whole shapes held out together)."""
    uniq = sorted(set(groups), key=str)
    fold_of = {g: i % folds for i, g in enumerate(uniq)}
    realized, oracle, unmeasured, regress = [], [], 0, 0
    for f in range(min(folds, len(uniq))):
        test = np.array([fold_of[g] == f for g in groups])
        if test.all() or not test.any():
            continue
        tree = make_tree().fit(X[~test], C[~test])
        pred = tree.predict(X[test])
        for i, c in zip(np.nonzero(test)[0], pred):
            if not M[i, c]:
                unmeasured += 1
                continue
            realized.append(C[i, c])
            oracle.append(C[i][M[i]].min())
            regress += C[i, c] > 1.0
    return {"shapes": len(uniq), "folds": min(folds, len(uniq)), "held_out": len(realized) + unmeasured,
            "measured": len(realized), "unmeasured": unmeasured,
            "geomean_ratio": _geomean(realized), "oracle_geomean_ratio": _geomean(oracle),
            "slower_than_builtin": int(regress)}


def train_rules(problems: Sequence[Problem], summaries: Dict[str, ProblemSummary],
                features: Sequence[str], feature_fn: Callable[[Problem], Dict[str, float]],
                configs: Callable[[str], Dict[str, Any]], kind: str = "tree", max_depth: int = 4,
                min_leaf: int = 3, folds: int = 5, max_classes: int = 12, min_gain: float = 0.02,
                unknown: float = 1.05) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """(rules JSON or None, report).  configs(text) gives a class's JSON config; the rules
    learn from, and apply to, the problems from M = MIN_RULE_M."""
    probs = [p for p in problems if p.key() in summaries and p.m >= MIN_RULE_M]
    pkeys = [p.key() for p in probs]
    classes = choose_classes({k: summaries[k] for k in pkeys}, min_gain, max_classes)
    report: Dict[str, Any] = {"problems": len(pkeys), "classes": classes}
    if len(classes) < 2 or len(pkeys) < 2 * min_leaf:
        report["skipped"] = "no config beats the built-in plan beyond the noise" if len(classes) < 2 \
            else "too few problems"
        return None, report
    if np is None:
        raise RuntimeError("training rules needs numpy (or export with --rules none)")
    X = np.array([[feature_fn(p)[f] for f in features] for p in probs], dtype=float)
    C, M = cost_matrix(pkeys, summaries, classes, unknown)

    def make() -> CostTree:
        return CostTree(max_depth=max_depth, min_leaf=min_leaf)

    tree = make().fit(X, C)
    pred = tree.predict(X)
    nodes, depth = tree.size()
    report.update(nodes=nodes, depth=depth,
                  train_geomean_ratio=_geomean([C[i, c] for i, c in enumerate(pred) if M[i, c]]),
                  train_oracle_geomean_ratio=_geomean([C[i][M[i]].min() for i in range(len(pkeys))]),
                  heldout=evaluate(X, C, M, [(p.dtype, p.n, p.k) for p in probs], folds, make))
    cfgs = [configs(t) for t in classes]

    def leaves(n: Dict[str, Any]) -> List[int]:
        return [n["class"]] if "feature" not in n else leaves(n["then"]) + leaves(n["other"])

    if not any(cfgs[c] for c in leaves(tree.root)):
        report["skipped"] = "the tree has no leaf with an opinion"
        return None, report
    return (tree_json(tree, features, cfgs) if kind == "tree" else ordered_json(tree, features, cfgs)), report
