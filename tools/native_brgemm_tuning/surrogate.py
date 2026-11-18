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
"""Surrogate models of log time for the model-guided search.

make_surrogate("auto") gives gradient-boosted trees from scikit-learn when it
is installed, else a small random forest written with numpy; "forest", "knn"
(distance-weighted k nearest neighbours) and "gbt" pick one.  predict()
returns a mean and a spread per row (the spread of bagged models), which the
search turns into a lower confidence bound.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def have_sklearn() -> bool:
    try:
        import sklearn.ensemble  # noqa: F401
    except ImportError:
        return False
    return True


class RegressionTree:
    """CART regression tree on numpy arrays (squared error, quantile thresholds)."""

    def __init__(self, max_depth: int = 8, min_leaf: int = 2, max_features: float = 1.0,
                 n_thresholds: int = 24, rng: Optional[np.random.Generator] = None):
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.max_features = max_features
        self.n_thresholds = n_thresholds
        self.rng = rng or np.random.default_rng(0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RegressionTree":
        self.feat, self.thr, self.left, self.right, self.value = [], [], [], [], []
        stack = [(np.arange(len(y)), 0, self._new(float(y.mean())))]
        while stack:
            idx, depth, node = stack.pop()
            if depth >= self.max_depth or len(idx) < 2 * self.min_leaf:
                continue
            split = self._best_split(X[idx], y[idx])
            if split is None:
                continue
            f, t = split
            mask = X[idx, f] <= t
            li, ri = idx[mask], idx[~mask]
            ln, rn = self._new(float(y[li].mean())), self._new(float(y[ri].mean()))
            self.feat[node], self.thr[node], self.left[node], self.right[node] = f, t, ln, rn
            stack += [(li, depth + 1, ln), (ri, depth + 1, rn)]
        self.feat_a = np.array(self.feat)
        self.thr_a = np.array(self.thr)
        self.left_a = np.array(self.left)
        self.right_a = np.array(self.right)
        self.value_a = np.array(self.value)
        return self

    def _new(self, v: float) -> int:
        for a, x in ((self.feat, -1), (self.thr, 0.0), (self.left, -1), (self.right, -1), (self.value, v)):
            a.append(x)
        return len(self.value) - 1

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Optional[Tuple[int, float]]:
        n, d = X.shape
        k = max(1, int(round(self.max_features * d)))
        feats = self.rng.choice(d, size=k, replace=False) if k < d else np.arange(d)
        base = float(((y - y.mean()) ** 2).sum())
        best, best_sse = None, base - 1e-12
        for f in feats:
            order = np.argsort(X[:, f], kind="stable")
            xs, ys = X[order, f], y[order]
            c1, c2 = np.cumsum(ys), np.cumsum(ys * ys)
            pos = np.nonzero(xs[:-1] < xs[1:])[0]
            pos = pos[(pos + 1 >= self.min_leaf) & (n - pos - 1 >= self.min_leaf)]
            if not len(pos):
                continue
            if len(pos) > self.n_thresholds:
                pos = pos[np.linspace(0, len(pos) - 1, self.n_thresholds).astype(int)]
            nl = pos + 1.0
            nr = n - nl
            sl, sr = c1[pos], c1[-1] - c1[pos]
            sse = (c2[pos] - sl * sl / nl) + ((c2[-1] - c2[pos]) - sr * sr / nr)
            i = int(np.argmin(sse))
            if sse[i] < best_sse:
                best_sse = float(sse[i])
                best = (int(f), float(0.5 * (xs[pos[i]] + xs[pos[i] + 1])))
        return best

    def predict(self, X: np.ndarray) -> np.ndarray:
        node = np.zeros(len(X), dtype=int)
        for _ in range(self.max_depth + 1):
            inner = self.feat_a[node] >= 0
            if not inner.any():
                break
            ni = node[inner]
            go_left = X[inner, self.feat_a[ni]] <= self.thr_a[ni]
            node[inner] = np.where(go_left, self.left_a[ni], self.right_a[ni])
        return self.value_a[node]


class ForestSurrogate:
    """Bagged regression trees with feature subsampling."""

    def __init__(self, n_trees: int = 32, max_depth: int = 8, min_leaf: int = 2,
                 max_features: float = 0.6, seed: int = 0):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.max_features = max_features
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ForestSurrogate":
        rng = np.random.default_rng(self.seed)
        self.trees = []
        for _ in range(self.n_trees):
            idx = rng.integers(0, len(y), len(y))
            t = RegressionTree(self.max_depth, self.min_leaf, self.max_features, rng=rng)
            self.trees.append(t.fit(X[idx], y[idx]))
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        P = np.stack([t.predict(X) for t in self.trees])
        return P.mean(axis=0), P.std(axis=0)


class KNNSurrogate:
    """Distance-weighted k nearest neighbours on standardized features."""

    def __init__(self, k: int = 7):
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNSurrogate":
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0)
        self.sd[self.sd == 0] = 1.0
        self.X = (X - self.mu) / self.sd
        self.y = y
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Z = (X - self.mu) / self.sd
        d = np.sqrt(((Z[:, None, :] - self.X[None, :, :]) ** 2).sum(axis=2))
        k = min(self.k, len(self.y))
        nn = np.argpartition(d, k - 1, axis=1)[:, :k]
        dn = np.take_along_axis(d, nn, axis=1)
        w = 1.0 / (dn + 1e-6)
        yn = self.y[nn]
        mean = (w * yn).sum(axis=1) / w.sum(axis=1)
        var = (w * (yn - mean[:, None]) ** 2).sum(axis=1) / w.sum(axis=1)
        return mean, np.sqrt(var)


class SklearnSurrogate:
    """Bagged gradient-boosted trees (scikit-learn)."""

    def __init__(self, n_models: int = 5, seed: int = 0):
        self.n_models = n_models
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnSurrogate":
        from sklearn.ensemble import GradientBoostingRegressor

        rng = np.random.default_rng(self.seed)
        self.models = []
        for i in range(self.n_models):
            idx = rng.integers(0, len(y), len(y))
            m = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                          subsample=0.8, random_state=self.seed + i)
            self.models.append(m.fit(X[idx], y[idx]))
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        P = np.stack([m.predict(X) for m in self.models])
        return P.mean(axis=0), P.std(axis=0)


def make_surrogate(kind: str = "auto", seed: int = 0):
    """auto: scikit-learn trees when installed, else the numpy forest."""
    if kind == "auto":
        kind = "gbt" if have_sklearn() else "forest"
    if kind == "gbt":
        if not have_sklearn():
            raise RuntimeError("scikit-learn is not installed; use --surrogate forest or knn")
        return SklearnSurrogate(seed=seed)
    if kind == "forest":
        return ForestSurrogate(seed=seed)
    if kind == "knn":
        return KNNSurrogate()
    raise ValueError("unknown surrogate %r (auto, gbt, forest, knn)" % kind)
