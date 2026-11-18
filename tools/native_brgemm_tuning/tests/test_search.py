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
"""The search loop on a synthetic cost function with a known optimum."""

import math
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import HAVE_NUMPY, SCHEMA, UARCH, session_records  # noqa: E402

from native_brgemm_tuning import candidates as C  # noqa: E402
from native_brgemm_tuning import results as R  # noqa: E402
from native_brgemm_tuning.backends import SyntheticBackend  # noqa: E402
from native_brgemm_tuning.problems import Problem  # noqa: E402
from native_brgemm_tuning.search import SearchSettings, Searcher  # noqa: E402

PROBLEMS = [Problem("bf16", m, n, 4096) for m in (1, 16, 64) for n in (4096, 6144)] + \
           [Problem("s8", m, 14336, 4096) for m in (8, 256)]
KEYS = ["nr", "threads_cap", "traversal", "mb", "tile_rows", "min_tiles_per_thread", "copy_block"]


def ratio(p, c):
    """Planted cost: a narrow tile and half the team help, everything else costs a little."""
    q = c.plan
    r = 1.0
    r *= 0.85 if q.nr == 16 else 1.0
    r *= 0.80 if q.threads == p.threads // 2 else (1.0 if q.threads == p.threads else 1.1)
    r *= 1.0 + 0.01 * len([k for k in c.config if k not in ("nr", "threads_cap")])
    return r


class Harness:
    def __init__(self, strategy, budget=24, batch=8, seed=0, **kw):
        self.tmp = tempfile.mkdtemp(prefix="nbt_search_")
        self.db = R.ResultsDB(os.path.join(self.tmp, "r.jsonl"))
        self.problems = {p.key(): p for p in PROBLEMS}
        self.spaces = {p.key(): C.Space(p, SCHEMA, UARCH, KEYS) for p in PROBLEMS}
        self.settings = SearchSettings(strategy=strategy, budget=budget, batch=batch, seed=seed, **kw)
        self.calls = []

    def evaluate(self, rnd, phase, batch):
        self.calls.append(batch)
        return session_records(self.problems, batch, ratio, "s%03d" % rnd, phase)

    def searcher(self):
        return Searcher(PROBLEMS, self.spaces, self.evaluate, self.db, self.settings, log=lambda m: None)

    def best(self):
        return {k: ps.best().ratio for k, ps in R.summarize(self.db.load()).items()}

    def close(self):
        shutil.rmtree(self.tmp)


def optimum(p):
    sp = C.Space(p, SCHEMA, UARCH, KEYS)
    return min([ratio(p, c) for c in C.unique(sp.iter_grid(2), sp)] + [1.0])


class TestSearch(unittest.TestCase):
    def check_run(self, h):
        h.searcher().run()
        for k in h.problems:
            texts = [c.text for b in h.calls for c in b.get(k, [])]
            sigs = [h.spaces[k].from_text(t).sig for t in texts]
            self.assertLessEqual(len(texts), h.settings.budget)
            self.assertEqual(len(sigs), len(set(sigs)), k)
            self.assertNotIn(h.spaces[k].default.sig, sigs)
        self.assertTrue(all(len(v) <= h.settings.batch for b in h.calls for v in b.values()))

    def test_grid_takes_single_key_changes_first(self):
        h = Harness("grid", budget=1000, batch=50, max_keys=1)
        self.addCleanup(h.close)
        self.check_run(h)
        for k, sp in h.spaces.items():
            measured = {c.text for b in h.calls for c in b.get(k, [])}
            self.assertEqual(measured, {c.text for c in C.unique(sp.iter_grid(1), sp)})
        h2 = Harness("grid", budget=12, batch=5)
        self.addCleanup(h2.close)
        self.check_run(h2)
        for k, sp in h2.spaces.items():
            seq = [len(c.config) for b in h2.calls for c in b.get(k, [])]
            self.assertEqual(seq, sorted(seq), k)
            n1 = len(C.unique(sp.iter_grid(1), sp))
            self.assertEqual(seq[:min(n1, 12)], [1] * min(n1, 12), k)

    def test_local_refines_the_significant_winners(self):
        h = Harness("grid", budget=6, batch=6, max_keys=1)
        self.addCleanup(h.close)
        h.searcher().run(max_rounds=1)
        summ = R.summarize(h.db.load())
        seeds = {}
        for k, ps in summ.items():
            good = sorted((c for c in ps.cands.values() if c.significant(0.02)), key=lambda c: c.ratio)
            seeds[k] = [h.spaces[k].from_text(c.text) for c in good[:2]]
        h.settings = SearchSettings(strategy="local", budget=100, batch=5, seeds=2)
        h.calls = []
        h.searcher().run(max_rounds=1)
        batch = h.calls[0]
        self.assertTrue(batch)
        for k, cands in batch.items():
            self.assertTrue(seeds[k], k)
            near = {c.sig for s in seeds[k] for c in h.spaces[k].neighbors(s)}
            for c in cands:
                self.assertIn(c.sig, near)
            self.assertLessEqual(len(cands), 5)
        self.assertFalse([k for k, s in seeds.items() if not s and k in batch])
        measured = {(k, h.spaces[k].from_text(t).sig) for k, ps in summ.items() for t in ps.cands}
        self.assertFalse([c for k, v in batch.items() for c in v if (k, c.sig) in measured])

    def test_random_respects_budget(self):
        h = Harness("random", budget=20, batch=8, seed=4)
        self.addCleanup(h.close)
        self.check_run(h)
        self.assertEqual(len(h.calls), 3)

    def test_resume(self):
        h = Harness("random", budget=16, batch=8, seed=1)
        self.addCleanup(h.close)
        self.assertEqual(h.searcher().run(max_rounds=1), 1)
        s2 = h.searcher()
        self.assertEqual(s2.round, 2)
        self.assertEqual(s2.count, {k: len(h.calls[0].get(k, [])) for k in h.problems})
        self.assertTrue(any(n == 8 for n in s2.count.values()))
        s2.run()
        for k in h.problems:
            texts = [c.text for b in h.calls for c in b.get(k, [])]
            self.assertEqual(len(texts), len(set(texts)))
            self.assertLessEqual(len(texts), 16)

    @unittest.skipUnless(HAVE_NUMPY, "numpy is not installed")
    def test_guided_reaches_the_planted_optimum(self):
        h = Harness("guided", budget=24, batch=8, seed=2, init=8, surrogate="forest")
        self.addCleanup(h.close)
        self.check_run(h)
        best = h.best()
        for k, p in h.problems.items():
            self.assertLessEqual(best[k], optimum(p) * 1.03, p.label())

    @unittest.skipUnless(HAVE_NUMPY, "numpy is not installed")
    def test_guided_not_worse_than_random(self):
        def geo(h):
            b = h.best()
            return math.exp(sum(math.log(v) for v in b.values()) / len(b))

        g, r = Harness("guided", budget=16, batch=8, seed=3, init=8, surrogate="knn"), Harness("random", 16, 8, 3)
        self.addCleanup(g.close)
        self.addCleanup(r.close)
        g.searcher().run()
        r.searcher().run()
        self.assertLessEqual(geo(g), geo(r) * 1.01)

    def test_synthetic_backend_end_to_end(self):
        tmp = tempfile.mkdtemp(prefix="nbt_syn_")
        self.addCleanup(shutil.rmtree, tmp)
        db = R.ResultsDB(os.path.join(tmp, "r.jsonl"))
        spaces = {p.key(): C.Space(p, SCHEMA, UARCH) for p in PROBLEMS}
        be = SyntheticBackend(noise=0.01, seed=1)
        ev = be.evaluator({p.key(): p for p in PROBLEMS}, {k: s.default for k, s in spaces.items()})
        Searcher(PROBLEMS, spaces, ev, db, SearchSettings(strategy="random", budget=8, batch=8),
                 log=lambda m: None).run()
        summ = R.summarize(db.load())
        self.assertEqual(len(summ), len(PROBLEMS))
        self.assertTrue(all(len(ps.cands) == 8 and ps.default_ms > 0 for ps in summ.values()))


@unittest.skipUnless(HAVE_NUMPY, "numpy is not installed")
class TestSurrogates(unittest.TestCase):
    def test_fit_a_smooth_function(self):
        import numpy as np

        from native_brgemm_tuning.surrogate import make_surrogate

        rng = np.random.default_rng(0)
        X = rng.uniform(-1, 1, (400, 5))
        f = lambda A: (A[:, 0] - 0.3) ** 2 + 0.5 * np.abs(A[:, 1])  # noqa: E731
        y = f(X) + 0.02 * rng.normal(size=400)
        Xt = rng.uniform(-1, 1, (200, 5))
        for kind in ("forest", "knn"):
            mu, sd = make_surrogate(kind).fit(X, y).predict(Xt)
            self.assertLess(np.sqrt(((mu - f(Xt)) ** 2).mean()), 0.5 * f(Xt).std(), kind)
            self.assertEqual(sd.shape, (200,))
        with self.assertRaises(ValueError):
            make_surrogate("tree")


if __name__ == "__main__":
    unittest.main()
