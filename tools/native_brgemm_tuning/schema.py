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
"""Schema of the native BRGEMM planner, as printed by the SchemaDump gtest.

The gtest prints the schema between the lines BEGIN NATIVE TUNING SCHEMA and
END NATIVE TUNING SCHEMA: the config keys (type, range, default or auto
value, dtype, bits), the match keys, the rule features, the kernel registry
and the hardware, with the machine's value of every version 2 hardware key
(hw_match).  load_schema() reads that JSON, a gtest log that holds it, or
fixtures/schema_dump.json (the gtest's output, with an example machine's
hardware, uarch and hw_match: with_machine()) when no path is given;
fixtures/schema_design.json, written from the design, is the schema of an
older build, without the rule keys, the blocks schedule and hw_match.  Field
names are normalized, so small spelling differences of the dump (a key dict
instead of a list, "bits_same" instead of "bits", a "registry" instead of
"kernels") load the same; a dump without hw_match gets the fields its
hardware and uarch give.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

BEGIN_MARK = "BEGIN NATIVE TUNING SCHEMA"
END_MARK = "END NATIVE TUNING SCHEMA"
FIXTURE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fixtures", "schema_dump.json"
)

WEIGHT_DTYPES = ("bf16", "s8", "f32", "s4", "f16")
INT8_SCHEMES = ("per_channel", "per_group")
# The INT8 scheme of a problem that names none: per-channel weight scales.
INT8_SCHEME = "per_channel"
# The planner's kernel kinds (native_kernel_kind_t): int4 is the kernels of an s4 weight
# (the group kernels, and those of a weight of one group over K), f16 the kernels of an
# f16 weight over a source widened to f32, f16_ph the kernels that multiply-add in f16.
KERNEL_KINDS = ("bf16", "int8", "f32", "int4", "f16", "f16_ph")
# Weight types whose tiles are INT8 tiles (the keys the schema says apply to int8).
INT8_DTYPES = ("s8", "s4")
ISA_ORDER = ("avx512f", "avx512bf16", "avx512vnni")
# Version 2 hardware keys, in the order the loader checks them.
HW_KEYS = ("vendor", "family", "model", "stepping", "isa", "l1d_kib", "l2_kib", "l3_kib", "l3_cores", "smt",
           "l3_threads", "cores", "logical_cpus")
KEY_TYPES = ("int", "int64", "bool", "enum", "list")
BOOL_TRUE = ("1", "true", "on")
BOOL_FALSE = ("0", "false", "off")


class SchemaError(ValueError):
    """The schema cannot be used."""


def kernel_kind(dtype: str) -> str:
    """Kernel kind of a weight type (or a kind's own name): bf16, int8 (s8), int4 (s4), f32,
    f16; an f16 problem runs the registry's F16 kind (Schema.f16_kind)."""
    if dtype in KERNEL_KINDS:
        return dtype
    if dtype == "s8":
        return "int8"
    if dtype == "s4":
        return "int4"
    raise SchemaError("unknown weight type %r" % (dtype,))


def _norm_applies(v: Any) -> str:
    if isinstance(v, (list, tuple)):
        kinds = {_norm_applies(x) for x in v}
        return "both" if {"bf16", "int8"} <= kinds else (kinds.pop() if kinds else "none")
    v = str(v).lower()
    if v in ("both", "all", "any"):
        return "both"
    if v in ("s8", "int8", "s4", "int4"):
        return "int8"
    if v in ("bf16", "f16", "f32"):
        return "bf16"
    return "none"


def _norm_bits(d: Dict[str, Any]) -> Any:
    if "bits" in d:
        b = d["bits"]
        if isinstance(b, dict):
            return {kernel_kind(k): str(v) for k, v in b.items()}
        if isinstance(b, bool):
            return "same" if b else "changes"
        return str(b)
    for name in ("bits_same", "same_bits"):
        if name in d:
            return "same" if d[name] else "changes"
    return "same"


def _max_covered(mask: int) -> int:
    r = 0
    while mask & (1 << r):
        r += 1
    return r


class KeySpec:
    """One config key (DESIGN section 4.2): type, legal values, default, bits."""

    def __init__(self, d: Dict[str, Any], order: int):
        try:
            self.name = str(d["name"])
        except KeyError:
            raise SchemaError("a key without a name: %r" % (d,))
        self.type = str(d.get("type", "int"))
        if self.type not in KEY_TYPES:
            raise SchemaError("key %s: unknown type %r" % (self.name, self.type))
        self.order = order
        self.applies = _norm_applies(d.get("applies", d.get("dtype", "both")))
        self.bits = _norm_bits(d)
        self.kind = str(d.get("kind", "param"))
        self.values = list(d["values"]) if d.get("values") is not None else None
        rng = d.get("range")
        self.min = d.get("min", rng[0] if rng else None)
        self.max = d.get("max", rng[1] if rng else None)
        self.multiple_of = int(d.get("multiple_of", d.get("step", 1)) or 1)
        # The INT8 schemes whose calls apply the key; a dump without the field: every scheme.
        self.int8_schemes = [str(s) for s in d.get("int8_schemes", INT8_SCHEMES)]
        self.must_contain = list(d.get("must_contain", []))
        if self.type == "list" and not self.must_contain:
            self.must_contain = [16] if self.name == "int8_cw" else []
        self.default = self._typed_default(d.get("default"))
        # heuristic_params_t's value of a key left at auto.
        self.auto = d.get("auto") if self.default is None else None

    def _typed_default(self, raw: Any) -> Any:
        """The default as a value of the key; None when a rule decides (auto)."""
        if raw is None:
            return None
        if self.type == "bool":
            return raw if isinstance(raw, bool) else self.parse(str(raw))
        if self.type == "list":
            return self.parse("/".join(str(x) for x in raw)) if isinstance(raw, list) else self.parse(str(raw))
        if self.type == "enum":
            return str(raw)
        if isinstance(raw, bool):
            return int(raw)
        if isinstance(raw, int):
            return raw
        s = str(raw).strip()
        return int(s) if re.fullmatch(r"[+-]?\d+", s) else None

    def applies_to(self, dtype: str) -> bool:
        """The key changes the plan of a problem of weight type dtype: "int8" keys apply to
        the INT8 tiles (s8, s4), "bf16" keys to the others (bf16, f16, f32)."""
        if self.applies == "both":
            return True
        int8 = dtype in INT8_DTYPES or dtype in ("int8", "int4")
        return self.applies == ("int8" if int8 else "bf16")

    def same_bits(self, dtype: str) -> bool:
        """True when changing the key keeps the output bits for dtype."""
        if isinstance(self.bits, dict):
            kind = "int8" if dtype in INT8_DTYPES else ("bf16" if dtype in ("f16", "f32") else kernel_kind(dtype))
            return self.bits.get(kind, self.bits.get(kernel_kind(dtype))) == "same"
        return self.bits == "same"

    def parse(self, text: str) -> Any:
        """The value of text in the override grammar; ValueError when malformed."""
        s = text.strip()
        if self.type == "bool":
            low = s.lower()
            if low in BOOL_TRUE:
                return True
            if low in BOOL_FALSE:
                return False
            raise ValueError("%s: %r is not a bool" % (self.name, text))
        if self.type in ("int", "int64"):
            if not re.fullmatch(r"[+-]?\d+", s):
                raise ValueError("%s: %r is not an integer" % (self.name, text))
            return int(s)
        if self.type == "enum":
            return s.lower()
        parts = [p.strip() for p in s.split("/")]
        if not parts or any(not re.fullmatch(r"\d+", p) for p in parts):
            raise ValueError("%s: %r is not a list such as 64/32/16" % (self.name, text))
        return sorted({int(p) for p in parts}, reverse=True)

    def format(self, v: Any) -> str:
        """The value as format_native_config writes it."""
        if self.type == "bool":
            return "1" if v else "0"
        if self.type == "list":
            return "/".join(str(w) for w in sorted(set(v), reverse=True))
        return str(v)

    def to_json(self, v: Any) -> Any:
        """The value as a tuning-file config takes it (DESIGN section 6.2)."""
        if self.type == "bool":
            return bool(v)
        if self.type == "list":
            return sorted({int(w) for w in v}, reverse=True)
        if self.type == "enum":
            return str(v)
        return int(v)

    def from_json(self, j: Any) -> Any:
        """The value of a tuning-file JSON value; ValueError on a wrong JSON type."""
        if self.type == "bool":
            if not isinstance(j, bool):
                raise ValueError("%s takes a JSON boolean" % self.name)
            return j
        if self.type == "list":
            if not isinstance(j, list) or any(isinstance(w, bool) or not isinstance(w, int) for w in j):
                raise ValueError("%s takes an array of integers" % self.name)
            return sorted(set(j), reverse=True)
        if self.type == "enum":
            if not isinstance(j, str):
                raise ValueError("%s takes a string" % self.name)
            return j
        if isinstance(j, bool) or not isinstance(j, int):
            raise ValueError("%s takes a JSON integer" % self.name)
        return j

    def check(self, v: Any) -> Optional[str]:
        """Why v is outside the key's static range (section 11.1), or None."""
        if self.type == "bool":
            return None if isinstance(v, bool) else "%s takes 0 or 1" % self.name
        if self.type in ("int", "int64"):
            if isinstance(v, bool) or not isinstance(v, int):
                return "%s takes an integer" % self.name
            if self.values is not None and v not in self.values:
                return "%s=%d is not one of %s" % (self.name, v, "/".join(map(str, self.values)))
            if (self.min is not None and v < self.min) or (self.max is not None and v > self.max):
                return "%s=%d is outside %s .. %s" % (self.name, v, self.min, self.max)
            if self.multiple_of > 1 and v % self.multiple_of:
                return "%s=%d is not a multiple of %d" % (self.name, v, self.multiple_of)
            return None
        if self.type == "enum":
            return None if v in (self.values or ()) else "%s=%s is not one of %s" % (
                self.name, v, "/".join(self.values or ()))
        if not isinstance(v, (list, tuple)) or not v:
            return "%s takes a non-empty list" % self.name
        if self.values is not None and any(w not in self.values for w in v):
            return "%s lists a width outside %s" % (self.name, "/".join(map(str, self.values)))
        if any(w not in v for w in self.must_contain):
            return "%s must contain %s" % (self.name, "/".join(map(str, self.must_contain)))
        return None


class KernelCaps:
    """What the micro-kernels of one kind offer at one tile width (native_kernel_caps_t).
    An INT8 or INT4 family serves the schemes it lists; one that lists none serves every
    scheme. A kind the tuner does not know (of a newer build) is kept with known False."""

    K_UNITS = {"bf16": 2, "int8": 4, "int4": 8, "f32": 1, "f16": 1, "f16_ph": 1}

    def __init__(self, d: Dict[str, Any]):
        kind = str(d.get("kind", "bf16"))
        self.known = kind in KERNEL_KINDS or kind in ("s8", "s4")
        self.kind = kernel_kind(kind) if self.known else kind
        self.nr = int(d["nr"])
        self.schemes = tuple(str(s) for s in d.get("int8_schemes") or ())
        mask = d.get("mr_mask", 0)
        if isinstance(mask, (list, tuple)):
            mask = sum(1 << (int(r) - 1) for r in mask)
        elif isinstance(mask, str):
            mask = int(mask, 0)
        self.mr_mask = int(mask)
        self.max_mr = int(d.get("max_mr", _max_covered(self.mr_mask)))
        self.k_unit = int(d.get("k_unit", self.K_UNITS.get(self.kind, 1)))

    def covers(self, mr: int) -> bool:
        """A kernel exists for every row count 1 .. mr."""
        if mr < 1 or mr > 32:
            return False
        full = (1 << mr) - 1
        return (self.mr_mask & full) == full

    def serves(self, scheme: str) -> bool:
        return not self.schemes or scheme in self.schemes

    @property
    def served(self) -> Tuple[str, ...]:
        """The INT8 schemes the family serves; none for the other kinds."""
        if self.kind not in ("int8", "int4"):
            return ()
        return tuple(s for s in INT8_SCHEMES if self.serves(s))


def _identity(v: Any) -> Optional[int]:
    """A CPUID family / model / stepping of the dump; None when absent or negative (unknown)."""
    return int(v) if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else None


def _count(v: Any) -> int:
    """A count or size of the dump; 0 when absent (unknown)."""
    return int(v) if isinstance(v, int) and not isinstance(v, bool) and v > 0 else 0


class Hardware:
    """native_hw() of the machine the schema was dumped on."""

    def __init__(self, d: Dict[str, Any]):
        self.vendor = str(d["vendor"]) if d.get("vendor") else None
        self.family = _identity(d.get("family"))
        self.model = _identity(d.get("model"))
        self.stepping = _identity(d.get("stepping"))
        self._isa = [str(f) for f in d.get("isa") or []]
        self.l1d_kib = _count(d.get("l1d_kib"))
        self.l2_kib = _count(d.get("l2_kib"))
        self.l3_kib = _count(d.get("l3_kib"))
        self.logical_cpus = _count(d.get("logical_cpus"))
        self.smt = _count(d.get("smt"))
        self.physical_cores = _count(d.get("physical_cores"))
        self.l3_domain_threads = _count(d.get("l3_domain_threads"))

    @property
    def isa(self) -> List[str]:
        return list(self._isa)

    def match_fields(self, uarch: Optional["Uarch"]) -> Dict[str, Any]:
        """The value of every version 2 hardware key this dump gives, as the loader reads
        the machine: l3_cores is the plan's L3 domain (the physical cores of one L3, else
        family_l3_cores), a zero count and an unknown identity are left out."""
        smt = self.smt if self.smt > 0 else 1
        domain = uarch.l3_domain_cores if uarch is not None else 0
        if not domain and self.l3_domain_threads > 0:
            domain = max(1, self.l3_domain_threads // smt)
        if not domain and uarch is not None:
            domain = uarch.family_l3_cores or 0
        given = {"vendor": self.vendor, "family": self.family, "model": self.model, "stepping": self.stepping,
                 "isa": self.isa, "l1d_kib": self.l1d_kib, "l2_kib": self.l2_kib,
                 "l3_kib": self.l3_kib, "l3_cores": domain, "smt": self.smt,
                 "l3_threads": self.l3_domain_threads, "cores": self.physical_cores,
                 "logical_cpus": self.logical_cpus}
        return {k: given[k] for k in HW_KEYS
                if k == "isa" or (k in ("vendor", "family", "model", "stepping") and given[k] is not None)
                or (isinstance(given[k], int) and given[k] > 0)}


class Uarch:
    """The UarchParams fields the built-in heuristic and its cost model read."""

    FIELDS = ("l1d_bytes", "l2_bytes", "l3_bytes", "family_l3_cores", "num_cores", "l3_domain_cores", "fma_ports")

    def __init__(self, l1d_bytes: int = 32768, l2_bytes: int = 1048576,
                 l3_bytes: int = 33554432, family_l3_cores: Optional[int] = 1,
                 num_cores: int = 1, l3_domain_cores: int = 0, fma_ports: int = 2):
        self.l1d_bytes = l1d_bytes
        self.l2_bytes = l2_bytes
        self.l3_bytes = l3_bytes
        # The cores per L3 the CPU family gives.
        self.family_l3_cores = family_l3_cores
        self.num_cores = num_cores
        # Physical cores of one L3 domain; 0 when unknown (the plan then uses family_l3_cores).
        self.l3_domain_cores = l3_domain_cores
        # FMA pipes of a core (cm_fma_pipes 0).
        self.fma_ports = fma_ports

    @property
    def plan_domain(self) -> Optional[int]:
        """The L3 domain the built-in plan uses when domain_threads is unset."""
        return self.l3_domain_cores if self.l3_domain_cores else self.family_l3_cores

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Uarch":
        u = cls()
        for k in cls.FIELDS:
            if d.get(k) is not None:
                setattr(u, k, int(d[k]))
        return u

    @classmethod
    def from_hw(cls, hw: Hardware) -> "Uarch":
        """l1d / l2 / l3 and the L3 domain from the hardware; family_l3_cores is not in it (None)."""
        domain = max(1, hw.l3_domain_threads // (hw.smt or 1)) if hw.l3_domain_threads > 0 else 0
        return cls(l1d_bytes=hw.l1d_kib * 1024, l2_bytes=hw.l2_kib * 1024, l3_bytes=hw.l3_kib * 1024,
                   family_l3_cores=None, l3_domain_cores=domain)

    def with_overrides(self, text: str) -> "Uarch":
        """A copy with "field=value,..." applied (e.g. "family_l3_cores=8")."""
        u = Uarch(**{k: getattr(self, k) for k in self.FIELDS})
        for item in filter(None, (t.strip() for t in text.split(","))):
            k, eq, v = item.partition("=")
            if not eq or k.strip() not in self.FIELDS:
                raise SchemaError("bad uarch override %r (fields: %s)" % (item, ", ".join(self.FIELDS)))
            setattr(u, k.strip(), int(v))
        return u

    @property
    def complete(self) -> bool:
        """Every field the heuristic reads is known; the L3 domain may come from either field."""
        return self.plan_domain is not None and all(
            getattr(self, k) is not None for k in self.FIELDS if k not in ("family_l3_cores", "l3_domain_cores"))

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.FIELDS}


class Schema:
    """Keys, match keys, features, kernel registry and hardware of one dump."""

    def __init__(self, doc: Dict[str, Any], source: str = "<dict>"):
        if not isinstance(doc, dict):
            raise SchemaError("%s: the schema is not a JSON object" % source)
        self.doc = doc
        self.source = source
        self.origin = str(doc.get("origin", ""))
        keys = doc.get("keys")
        if isinstance(keys, dict):
            keys = [dict(v, name=k) for k, v in keys.items()]
        if not keys:
            raise SchemaError("%s: the schema has no keys" % source)
        self.keys = [KeySpec(d, i) for i, d in enumerate(keys)]
        self.key_map = {k.name: k for k in self.keys}
        mk = doc.get("match_keys", {})
        self.match_keys = {k: "any" for k in mk} if isinstance(mk, list) else dict(mk)
        feats = doc.get("features", [])
        self.features = [f["name"] if isinstance(f, dict) else str(f) for f in feats]
        kernels = doc.get("kernels") or doc.get("registry") or doc.get("kernel_caps") or []
        self.kernels = [KernelCaps(d) for d in kernels]
        hw = doc.get("hw") or doc.get("hardware") or doc.get("native_hw")
        self.hw = Hardware(hw) if hw else None
        if doc.get("uarch"):
            self.uarch: Optional[Uarch] = Uarch.from_dict(doc["uarch"])
        elif self.hw is not None:
            self.uarch = Uarch.from_hw(self.hw)
        else:
            self.uarch = None
        # The machine's value of every version 2 hardware key the dump gives.
        match = doc.get("hw_match")
        if isinstance(match, dict):
            self.hw_match: Dict[str, Any] = dict(match)
        else:
            self.hw_match = self.hw.match_fields(self.uarch) if self.hw is not None else {}

    def key(self, name: str) -> KeySpec:
        try:
            return self.key_map[name]
        except KeyError:
            raise SchemaError("unknown config key %r" % (name,))

    def caps(self, kind: str, nr: int, scheme: Optional[str] = None) -> Optional[KernelCaps]:
        """native_kernel_registry_t::caps: the kernels of kind at width nr; for INT8 and INT4
        the family that serves scheme (the INT4 kernels of one group over K serve per_channel,
        the INT4 group kernels per_group).  Without a scheme, as the library's caps(kind, nr):
        the per-channel INT8 kernels, the INT4 group kernels."""
        if scheme is None:
            scheme = "per_group" if kind == "int4" else INT8_SCHEME
        for c in self.kernels:
            if c.kind == kind and c.nr == nr and (kind not in ("int8", "int4") or c.serves(scheme)):
                return c
        return None

    def covers(self, kind: str, mr: int, nr: int, scheme: Optional[str] = None) -> bool:
        """native_kernel_registry_t::covers (with the scheme for INT8 and INT4): a kernel for
        every row count 1 .. mr at nr."""
        c = self.caps(kind, nr, scheme)
        return c is not None and 1 <= mr <= c.max_mr

    def int8_serves(self, mr: int, nr: int, scheme: str = INT8_SCHEME) -> bool:
        """native_kernel_registry_t::covers_int8: the family of scheme at width nr takes
        mr rows per call (1 .. its max_mr)."""
        c = self.caps("int8", nr, scheme)
        return c is not None and 1 <= mr <= c.max_mr

    def nr_widths(self, kind: str) -> List[int]:
        return sorted({c.nr for c in self.kernels if c.kind == kind})

    def int8_max_mr(self, nv: int, scheme: str = INT8_SCHEME) -> int:
        """Rows of the widest INT8 kernel of scheme at nr 16 nv (native_int8_ukernel_max_mr
        for per-channel calls, the group kernel's for per-group calls)."""
        c = self.caps("int8", 16 * nv, scheme)
        return c.max_mr if c else 0

    def int4_max_mr(self, nv: int, scheme: str = "per_group") -> int:
        """Rows of the INT4 kernels of scheme at nr 16 nv: per_group those of the group
        kernels (native_int4_group_ukernel_max_mr), per_channel those of the kernels of one
        group over K (native_int4_one_group_ukernel_max_mr)."""
        c = self.caps("int4", 16 * nv, scheme)
        return c.max_mr if c else 0

    def int4_scheme(self, one_group: bool) -> str:
        """The scheme of the INT4 kernels an s4 weight takes: per_channel for a weight of one
        group over K where the registry has the kernels of one group, else per_group (a
        build without them runs the group kernels for every s4 weight)."""
        if one_group and any(c.kind == "int4" and c.serves("per_channel") for c in self.kernels):
            return "per_channel"
        return "per_group"

    def f16_kind(self) -> str:
        """native_kernel_registry_t::f16_kind: f16_ph where the registry has those kernels."""
        return "f16_ph" if any(c.kind == "f16_ph" for c in self.kernels) else "f16"

    def kind_of(self, dtype: str) -> str:
        """The kernels a problem of weight type dtype plans for."""
        return self.f16_kind() if dtype == "f16" else kernel_kind(dtype)

    def summary(self) -> str:
        lines = ["schema %s (%s)" % (self.source, self.origin or "dump")]
        lines.append("hardware fields: %s" % (json.dumps(self.hw_match) if self.hw_match else "none"))
        lines.append("uarch: %s" % (self.uarch.to_dict() if self.uarch else "none"))
        for k in self.keys:
            rng = ("/".join(map(str, k.values)) if k.values is not None
                   else "%s .. %s" % (k.min, k.max))
            lines.append("  %-20s %-5s default=%-10s %-22s applies=%-4s bits=%s" % (
                k.name, k.type, "auto" if k.default is None else k.format(k.default),
                rng, k.applies, k.bits))
        for c in self.kernels:
            lines.append("  kernel %-4s nr=%-3d max_mr=%-2d mask=%#x k_unit=%d%s" % (
                c.kind, c.nr, c.max_mr, c.mr_mask, c.k_unit,
                " schemes=%s" % "/".join(c.schemes) if c.schemes else ""))
        return "\n".join(lines)


def schema_diff(a: Schema, b: Schema) -> List[str]:
    """Differences of keys and kernels between two schemas (e.g. fixture vs dump)."""
    out = []
    names_a, names_b = [k.name for k in a.keys], [k.name for k in b.keys]
    for n in names_a:
        if n not in b.key_map:
            out.append("key %s only in %s" % (n, a.source))
    for n in names_b:
        if n not in a.key_map:
            out.append("key %s only in %s" % (n, b.source))
    common = [n for n in names_a if n in b.key_map]
    if common != [n for n in names_b if n in a.key_map]:
        out.append("key order differs")
    for n in common:
        ka, kb = a.key_map[n], b.key_map[n]
        for attr in ("type", "default", "auto", "min", "max", "values", "multiple_of", "applies", "bits"):
            if getattr(ka, attr) != getattr(kb, attr):
                out.append("key %s: %s %r vs %r" % (n, attr, getattr(ka, attr), getattr(kb, attr)))
    ka = {(c.kind, c.nr, "/".join(c.served)): (c.mr_mask, c.max_mr, c.k_unit) for c in a.kernels}
    kb = {(c.kind, c.nr, "/".join(c.served)): (c.mr_mask, c.max_mr, c.k_unit) for c in b.kernels}
    for k in sorted(set(ka) | set(kb)):
        if ka.get(k) != kb.get(k):
            scheme = " (%s)" % k[2] if k[2] else ""
            out.append("kernel %s nr %d%s: %r vs %r" % (k[0], k[1], scheme, ka.get(k), kb.get(k)))
    return out


def extract_schema_json(text: str) -> str:
    """The JSON between the SchemaDump markers of a gtest log, or text itself."""
    b = text.find(BEGIN_MARK)
    if b < 0:
        return text
    e = text.find(END_MARK, b)
    if e < 0:
        raise SchemaError("%s without %s" % (BEGIN_MARK, END_MARK))
    body = text[b + len(BEGIN_MARK):e]
    i, j = body.find("{"), body.rfind("}")
    if i < 0 or j < i:
        raise SchemaError("no JSON object between the schema markers")
    return body[i:j + 1]


# The blocks that describe the machine a dump was made on.
MACHINE_BLOCKS = ("hw", "uarch", "hw_match")


def with_machine(doc: Dict[str, Any], machine: Dict[str, Any]) -> Dict[str, Any]:
    """doc with the values of machine's hw, uarch and hw_match, in doc's fields and order.

    fixtures/schema_dump.json describes an example machine whatever machine its dump
    came from: the planner's test compares only these blocks' field names."""
    out = dict(doc)
    missing: List[str] = []
    for block in MACHINE_BLOCKS:
        if block not in doc:
            continue
        values = machine.get(block) or {}
        missing += ["%s.%s" % (block, k) for k in doc[block] if k not in values]
        out[block] = {k: values.get(k) for k in doc[block]}
    if missing:
        raise SchemaError("the example machine has no %s: give it a value" % ", ".join(missing))
    return out


def schema_text(doc: Dict[str, Any], width: int = 112) -> str:
    """A schema document in the layout of fixtures/schema_dump.json: an object's
    entries and a list's objects one per line, a list of names wrapped at width."""
    def compact(v: Any) -> str:
        return json.dumps(v, separators=(", ", ": "))

    def member(name: str, v: Any) -> List[str]:
        head = "  %s: " % json.dumps(name)
        if isinstance(v, dict) and v:
            items = list(v.items())
            return [head + "{"] + ["    %s: %s%s" % (json.dumps(k), compact(x), "," if i < len(items) - 1 else "")
                                   for i, (k, x) in enumerate(items)] + ["  }"]
        if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            return [head + "["] + ["    %s%s" % (compact(x), "," if i < len(v) - 1 else "")
                                   for i, x in enumerate(v)] + ["  ]"]
        if isinstance(v, list) and v:
            rows: List[List[Any]] = [[]]
            for x in v:
                if rows[-1] and len("    " + ", ".join(json.dumps(y) for y in rows[-1] + [x]) + ",") > width:
                    rows.append([])
                rows[-1].append(x)
            return [head + "["] + ["    %s%s" % (", ".join(json.dumps(x) for x in r), "," if i < len(rows) - 1 else "")
                                   for i, r in enumerate(rows)] + ["  ]"]
        return [head + compact(v)]

    out = ["{"]
    items = list(doc.items())
    for i, (k, v) in enumerate(items):
        lines = member(k, v)
        if i < len(items) - 1:
            lines[-1] += ","
        out += lines
    return "\n".join(out + ["}"]) + "\n"


def load_schema(path: Optional[str] = None, uarch_overrides: Optional[str] = None) -> Schema:
    """The schema of a JSON file or gtest log; fixtures/schema_dump.json when path is None."""
    path = path or FIXTURE
    with open(path, encoding="utf-8") as f:
        text = f.read()
    try:
        doc = json.loads(extract_schema_json(text))
    except json.JSONDecodeError as e:
        raise SchemaError("%s: %s" % (path, e))
    s = Schema(doc, source=path)
    if uarch_overrides:
        s.uarch = (s.uarch or Uarch(family_l3_cores=None)).with_overrides(uarch_overrides)
    return s
