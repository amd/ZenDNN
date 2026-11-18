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
"""Problems (the native_problem_t fields a plan depends on), shape lists, M
classes and the rule features of DESIGN section 6.2."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Weight types (native_problem_t::wei_dt), sources and destinations, as the match keys name them.
DTYPES = ("bf16", "s8", "f32", "s4", "f16")
SRC_TYPES = ("bf16", "f32", "u8", "s8", "f16")
DST_TYPES = ("f32", "bf16", "f16", "u8", "s8", "s32")
EPILOGUES = ("store", "bias", "postops", "gated")
DST_BYTES = {"f32": 4, "bf16": 2, "f16": 2, "u8": 1, "s8": 1, "s32": 4}
INT8_SCHEMES = ("per_channel", "per_group")
# Bytes of one weight scale of an s4 problem: f32, or bf16 / f16.
SCALE_BYTES = (2, 4)
# Columns of one block of the native layout: a weight of N not a multiple of it is
# held, planned and matched at N rounded up to whole pack_nr blocks.
LAYOUT_PACK_NR = 32

DEFAULT_M_CLASSES = (
    "1,2-4:4,5-8:8,9-16:16,17-32:32,33-64:64,65-128:128,129-512:256/512,"
    "513-2048:1024/2048,2049-:4096"
)


def _bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "on", "yes"):
        return True
    if s in ("0", "false", "off", "no"):
        return False
    raise ValueError("not a bool: %r" % (v,))


@dataclass(frozen=True)
class Problem:
    """One matmul as the planner sees it (native_problem_t).  dtype is the weight type
    (bf16, s8, f32, s4 or f16); n is the call's N (plan_n the width the planner and the
    tuning file see); an s8 problem's int8_scheme is that of its weight scales, an s4
    problem's is per_group.  group is an s4 weight's group size along K (K for one group
    over K: {1, N} scales) and scale_bytes the bytes of one of its weight scales; both are
    0 for any other weight.  src and dst default to the weight's type for f16 and f32
    weights, to bf16 otherwise."""

    dtype: str
    m: int
    n: int
    k: int
    threads: int = 32
    pack_nr: int = 32
    src: Optional[str] = None
    dst: Optional[str] = None
    trans_a: bool = False
    dynamic: Optional[bool] = None
    has_bias: bool = False
    epilogue: str = "store"
    int8_scheme: Optional[str] = None
    group: int = 0
    scale_bytes: int = 0
    name: str = field(default="", compare=False)

    FIELDS = ("dtype", "m", "n", "k", "threads", "pack_nr", "src", "dst", "trans_a",
              "dynamic", "has_bias", "epilogue", "int8_scheme", "group", "scale_bytes")

    def __post_init__(self) -> None:
        if self.dtype not in DTYPES:
            raise ValueError("weight type %r is not one of %s" % (self.dtype, ", ".join(DTYPES)))
        if min(self.m, self.n, self.k, self.threads, self.pack_nr) < 1:
            raise ValueError("M, N, K, threads and pack_nr must be positive: %r" % (self,))
        own = self.dtype if self.dtype in ("f16", "f32") else "bf16"
        if self.src is None:
            object.__setattr__(self, "src", own)
        if self.dst is None:
            object.__setattr__(self, "dst", own)
        if self.src not in SRC_TYPES or self.dst not in DST_TYPES:
            raise ValueError("bad src / dst type in %r" % (self,))
        if self.epilogue not in EPILOGUES:
            raise ValueError("epilogue %r is not one of %s" % (self.epilogue, EPILOGUES))
        if self.int8_scheme is None:
            object.__setattr__(self, "int8_scheme", "per_group" if self.dtype == "s4" else "per_channel")
        if self.int8_scheme not in INT8_SCHEMES or (self.dtype != "s8" and self.int8_scheme != (
                "per_group" if self.dtype == "s4" else "per_channel")):
            raise ValueError("INT8 scheme %r: an s8 weight takes one of %s, an s4 weight per_group" % (
                self.int8_scheme, INT8_SCHEMES))
        if self.dtype == "s4":
            if self.scale_bytes == 0:
                object.__setattr__(self, "scale_bytes", 2)
            if self.group < 1 or self.group > self.k or self.k % self.group:
                raise ValueError("an s4 weight's group size %d does not divide K=%d" % (self.group, self.k))
            if self.scale_bytes not in SCALE_BYTES:
                raise ValueError("an s4 weight's scales take %s bytes, not %d" % (SCALE_BYTES, self.scale_bytes))
        elif self.group or self.scale_bytes:
            raise ValueError("group and scale_bytes are for s4 weights only: %r" % (self,))
        if self.dynamic is None:
            object.__setattr__(self, "dynamic", self.int8 and self.src in ("bf16", "f32"))

    @property
    def bf16(self) -> bool:
        return self.dtype == "bf16"

    @property
    def f32(self) -> bool:
        return self.dtype == "f32"

    @property
    def f16(self) -> bool:
        return self.dtype == "f16"

    @property
    def int8(self) -> bool:
        """An s8 or s4 weight: the INT8 tile and its plan fields."""
        return self.dtype in ("s8", "s4")

    @property
    def int4(self) -> bool:
        return self.dtype == "s4"

    @property
    def plan_n(self) -> int:
        """The N of the planner's problem (make_native_problem): N rounded up to whole
        pack_nr blocks when it is not a multiple of the layout's blocks."""
        if self.n % LAYOUT_PACK_NR == 0 or self.pack_nr <= 0:
            return self.n
        return -(-self.n // self.pack_nr) * self.pack_nr

    @property
    def esz(self) -> int:
        """Bytes of a source element the kernels read (the planner's elem_bytes)."""
        return 4 if self.f32 else (2 if self.bf16 or self.f16 else 1)

    @property
    def wei_esz(self) -> float:
        """Bytes of a weight element: esz, half a byte for s4."""
        return 0.5 if self.int4 else float(self.esz)

    @property
    def row_byte_units(self) -> int:
        """Weight columns a source row weighs in the makespan's bytes: 2 over s4 weights."""
        return 2 if self.int4 else 1

    def key(self) -> str:
        """Identity string; Problem.parse(p.key()) == p."""
        return "dtype=%s,m=%d,%s" % (self.dtype, self.m, self._rest())

    def shape_key(self) -> str:
        """Identity without M: the problems one table entry can cover."""
        return "dtype=%s,%s" % (self.dtype, self._rest())

    def _rest(self) -> str:
        rest = "n=%d,k=%d,threads=%d,pack_nr=%d,src=%s,dst=%s,trans_a=%d,dynamic=%d,has_bias=%d,epilogue=%s" % (
            self.n, self.k, self.threads, self.pack_nr, self.src, self.dst, self.trans_a,
            self.dynamic, self.has_bias, self.epilogue)
        # A per-channel problem's identity names no scheme, as its stored records do.
        if self.int8_scheme != "per_channel":
            rest += ",int8_scheme=" + self.int8_scheme
        if self.int4:
            rest += ",group=%d,scale_bytes=%d" % (self.group, self.scale_bytes)
        return rest

    def label(self) -> str:
        wei = "%s g%d" % (self.dtype, self.group) if self.int4 else self.dtype
        return "%s%s M=%d N=%d K=%d" % (self.name + " " if self.name else "", wei, self.m, self.n, self.k)

    def with_m(self, m: int) -> "Problem":
        return replace(self, m=m)

    def to_dict(self) -> Dict[str, Any]:
        d = {f: getattr(self, f) for f in self.FIELDS}
        if self.int8_scheme == "per_channel" or self.int4:
            del d["int8_scheme"]
        if not self.int4:
            del d["group"], d["scale_bytes"]
        if self.name:
            d["name"] = self.name
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Problem":
        kw = {f: d[f] for f in cls.FIELDS if f in d}
        for f in ("m", "n", "k", "threads", "pack_nr", "group", "scale_bytes"):
            if f in kw:
                kw[f] = int(kw[f])
        for f in ("trans_a", "dynamic", "has_bias"):
            if f in kw and kw[f] is not None:
                kw[f] = _bool(kw[f])
        return cls(name=str(d.get("name", "")), **kw)

    @classmethod
    def parse(cls, text: str) -> "Problem":
        """"dtype=s8,m=1,n=6144,k=4096[,threads=32,...]" (',' or ':' separated)."""
        d: Dict[str, Any] = {}
        for item in filter(None, (t.strip() for t in text.replace(":", ",").split(","))):
            k, eq, v = item.partition("=")
            if not eq:
                raise ValueError("problem field %r has no value" % item)
            d[k.strip().lower()] = v.strip()
        unknown = set(d) - set(cls.FIELDS) - {"name"}
        if unknown:
            raise ValueError("unknown problem fields: %s" % ", ".join(sorted(unknown)))
        return cls.from_dict(d)


@dataclass
class Shape:
    """A weight of a model: (dtype, N, K) with the M range and call count it saw."""

    dtype: str
    n: int
    k: int
    name: str = ""
    calls: int = 0
    m_lo: int = 1
    m_hi: int = 0  # 0: unbounded
    models: Tuple[str, ...] = ()
    # s4: the group size along K (K: one group); 0 for other weights.
    group: int = 0

    @property
    def key(self) -> Tuple[str, int, int, int]:
        return (self.dtype, self.n, self.k, self.group)


def read_shapes_tsv(path: str, dtypes: Optional[Iterable[str]] = None,
                    skip_names: Iterable[str] = ("lm_head",), min_calls: int = 0,
                    group: int = 128) -> List[Shape]:
    """Unique (dtype, N, K, group) of a TSV with the columns model, name, N, K, wei, calls, m_lo,
    m_hi and, optionally, group (an s4 weight's group size along K, -1 for one group over K;
    group by default when the column is absent or empty).

    Rows of one (dtype, N, K, group) merge: calls add up and the M range widens.
    """
    want = set(dtypes) if dtypes else None
    skip = set(skip_names)
    merged: Dict[Tuple[str, int, int, int], Shape] = {}
    with open(path, encoding="utf-8") as f:
        header: Optional[List[str]] = None
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            cols = line.split("\t")
            if header is None:
                header = [c.strip().lower() for c in cols]
                need = {"name", "n", "k", "wei"}
                if not need <= set(header):
                    raise ValueError("%s: the header lacks %s" % (path, ", ".join(sorted(need - set(header)))))
                continue
            row = dict(zip(header, cols))
            dtype = row["wei"].strip()
            if dtype not in DTYPES or (want and dtype not in want):
                continue
            if row["name"].strip() in skip:
                continue
            calls = int(row.get("calls", "0") or 0)
            if calls < min_calls:
                continue
            n, k = int(row["n"]), int(row["k"])
            gs = 0
            if dtype == "s4":
                gs = int((row.get("group") or "").strip() or group)
                gs = k if gs == -1 else gs
            key = (dtype, n, k, gs)
            lo = int(row.get("m_lo", "1") or 1)
            hi = int(row.get("m_hi", "0") or 0)
            model = row.get("model", "").strip()
            s = merged.get(key)
            if s is None:
                merged[key] = Shape(dtype, n, k, row["name"].strip(), calls, lo, hi,
                                    (model,) if model else (), gs)
                continue
            s.calls += calls
            s.m_lo = min(s.m_lo, lo)
            s.m_hi = 0 if (s.m_hi == 0 or hi == 0) else max(s.m_hi, hi)
            if model and model not in s.models:
                s.models = s.models + (model,)
            if row["name"].strip() not in s.name.split("/"):
                s.name = s.name + "/" + row["name"].strip()
    return sorted(merged.values(), key=lambda s: (s.dtype, s.n, s.k, s.group))


@dataclass(frozen=True)
class MClass:
    """M from lo to hi inclusive (hi None: no bound), measured at the reps."""

    lo: int
    hi: Optional[int]
    reps: Tuple[int, ...]

    def contains(self, m: int) -> bool:
        return m >= self.lo and (self.hi is None or m <= self.hi)

    def label(self) -> str:
        if self.hi == self.lo:
            return str(self.lo)
        return "%d-%s" % (self.lo, "" if self.hi is None else self.hi)


def parse_m_classes(text: str) -> List[MClass]:
    """"lo[-[hi]][:rep/rep...]" items, comma separated; reps default to hi (or lo)."""
    out: List[MClass] = []
    for item in filter(None, (t.strip() for t in text.split(","))):
        rng, _, reps = item.partition(":")
        lo_s, dash, hi_s = rng.partition("-")
        lo = int(lo_s)
        hi: Optional[int] = (int(hi_s) if hi_s else None) if dash else lo
        if lo < 1 or (hi is not None and hi < lo):
            raise ValueError("bad M class %r" % item)
        rep = tuple(int(r) for r in reps.split("/")) if reps else ((hi,) if hi else (lo,))
        if any(not (r >= lo and (hi is None or r <= hi)) for r in rep):
            raise ValueError("M class %r has a representative outside its range" % item)
        out.append(MClass(lo, hi, rep))
    out.sort(key=lambda c: c.lo)
    for a, b in zip(out, out[1:]):
        if a.hi is None or a.hi >= b.lo:
            raise ValueError("M classes %s and %s overlap" % (a.label(), b.label()))
    return out


def class_of(m: int, classes: List[MClass]) -> Optional[MClass]:
    for c in classes:
        if c.contains(m):
            return c
    return None


def problems_from_shapes(shapes: Iterable[Shape], classes: List[MClass], threads: int = 32,
                         pack_nr: int = 32, **attrs: Any) -> List[Problem]:
    """One problem per shape and representative M of every class the shape's M range meets."""
    out: List[Problem] = []
    seen = set()
    for s in shapes:
        hi = s.m_hi if s.m_hi > 0 else None
        for c in classes:
            if hi is not None and c.lo > hi or c.hi is not None and c.hi < s.m_lo:
                continue
            ms = [r for r in c.reps if r >= s.m_lo and (hi is None or r <= hi)]
            if not ms:
                ms = [max(s.m_lo, min(c.reps[0], hi if hi is not None else c.reps[0]))]
            for m in ms:
                extra = dict(attrs)
                if s.dtype == "s4":
                    extra["group"] = s.group
                elif "scale_bytes" in extra:
                    del extra["scale_bytes"]
                p = Problem(s.dtype, m, s.n, s.k, threads=threads, pack_nr=pack_nr,
                            name=s.name, **extra)
                if p.key() not in seen:
                    seen.add(p.key())
                    out.append(p)
    return out


def rule_features(p: Problem, uarch: Any, hw: Any = None) -> Dict[str, float]:
    """The rule features of the tuning file's rules for p (problem_features: doubles, as the
    loader evaluates them).  esz is the bytes of a weight element (0.5 for s4), a_bytes
    counts the source in the bytes of the weight's tile (4 for f32, 2 for bf16 and f16, 1
    for an s8 source), b_bytes and slab_l2 the weight at esz; group is the s4 group size.

    uarch gives l2 / l3; hw (schema.Hardware) gives l3_threads / smt / cores (0 when None).
    """
    src_esz = float(p.esz)
    esz = p.wei_esz
    n = float(p.plan_n)
    a_bytes = float(p.m) * p.k * src_esz
    b_bytes = n * p.k * esz
    l2 = float(uarch.l2_bytes)
    l3 = float(uarch.l3_bytes)
    return {
        "m": float(p.m), "n": n, "k": float(p.k),
        "threads": float(p.threads), "pack_nr": float(p.pack_nr),
        "bf16": 1.0 if p.bf16 else 0.0,
        "dynamic": float(bool(p.dynamic)), "trans_a": float(p.trans_a), "has_bias": float(p.has_bias),
        "epilogue": float(EPILOGUES.index(p.epilogue)),
        "dst_bytes": float(DST_BYTES[p.dst]), "esz": float(esz),
        "a_bytes": a_bytes, "b_bytes": b_bytes,
        "col_tiles": n / p.pack_nr,
        "m_per_thread": float(p.m) / p.threads, "n_per_thread": n / p.threads,
        "flops_per_thread": 2.0 * p.m * n * p.k / p.threads,
        "l2": l2, "l3": l3,
        "l3_threads": float(hw.l3_domain_threads) if hw else 0.0,
        "smt": float(hw.smt) if hw else 0.0,
        "cores": float(hw.physical_cores) if hw else 0.0,
        "a_l3": a_bytes / l3 if l3 else 0.0, "b_l3": b_bytes / l3 if l3 else 0.0,
        "slab_l2": float(p.pack_nr) * p.k * esz / l2 if l2 else 0.0,
        "group": float(p.group),
    }
