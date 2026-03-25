#!/usr/bin/env python3
"""
Analyze benchdnn matmul output for AMD EPYC (Zen 4 / Zen 5).
Auto-detects architecture from [ARCH] line in perf output; falls back to Zen 5.

Three modes:
  Single file:  analyze_benchmark.py <file>
  Compare:      analyze_benchmark.py --compare <file1> <file2> ...
  Perf:         analyze_benchmark.py --perf [flags] <file>
"""

import sys
import os
import re
import argparse
from math import gcd

# ── Architecture constants ─────────────────────────────────────────────────
ARCH_CONSTANTS = {
    "zen4": dict(name="Zen4 (Genoa)", family=0x19,
                 cpu_freq_ghz=3.7,
                 l1d_kb=32, l2_kb=1024, l3_mb_per_ccd=32,
                 l2_bw_b_per_cycle=32, dispatch_width=6,
                 l3_bw_gbs=80.0, dram_bw_gbs=40.0),
    "zen5": dict(name="Zen5 (Turin)", family=0x1A,
                 cpu_freq_ghz=4.1,
                 l1d_kb=48, l2_kb=1024, l3_mb_per_ccd=32,
                 l2_bw_b_per_cycle=64, dispatch_width=8,
                 l3_bw_gbs=120.0, dram_bw_gbs=50.0),
}

def set_arch(arch_key="zen5"):
    """Set global constants from the named architecture."""
    global CPU_FREQ_GHZ, L2_BW_B_PER_CYCLE, L1D_KB, L2_KB
    global L3_BW_GBS, DRAM_BW_GBS, ARCH_NAME, L3_KB
    ac = ARCH_CONSTANTS.get(arch_key, ARCH_CONSTANTS["zen5"])
    CPU_FREQ_GHZ       = ac["cpu_freq_ghz"]
    L2_BW_B_PER_CYCLE  = ac["l2_bw_b_per_cycle"]
    L1D_KB             = ac["l1d_kb"]
    L2_KB              = ac["l2_kb"]
    L3_KB              = ac["l3_mb_per_ccd"] * 1024   # 32 MB → 32768 KB
    L3_BW_GBS          = ac["l3_bw_gbs"]
    DRAM_BW_GBS        = ac["dram_bw_gbs"]
    ARCH_NAME          = ac["name"]
    _compute_ops_per_cycle()

SIMD_WIDTH   = 512
FMA_PER_CORE = 2
OPS_PER_FMA  = 2

DTYPE_BITS = {"fp32": 32, "bf16": 16, "int8": 8}
COMPUTE = {}
DOUBLE_PUMP = 1

def _compute_ops_per_cycle():
    """Recompute OPS/cycle after arch change. Zen 4 double-pumps 512-bit ops
    through 256-bit datapaths (2 cycles per op → half throughput vs Zen 5)."""
    global COMPUTE, DOUBLE_PUMP
    DOUBLE_PUMP = 0.5 if "Zen4" in ARCH_NAME else 1
    COMPUTE.clear()
    for dt, bits in DTYPE_BITS.items():
        COMPUTE[dt] = int((SIMD_WIDTH // bits) * FMA_PER_CORE * OPS_PER_FMA * DOUBLE_PUMP)

set_arch("zen5")   # default until [ARCH] detected

ELEM_BYTES = {"u8":1, "s8":1, "s4":0.5, "bf16":2, "f16":2, "f32":4}

ALGO_NAMES = {
    "1": "DLP-Blocked", "2": "DLP-2", "3": "DLP-Reorder",
    "10": "Native-GEMM", "11": "Native-BRGEMM",
}

# ── Shared helpers ──────────────────────────────────────────────────────────

def esz(tag):
    return ELEM_BYTES.get(tag.lower(), 4)

def classify(dt):
    p = dt.lower().split(":")
    s, w = p[0], p[1] if len(p) > 1 else p[0]
    if s in ("u8","s8") or w in ("s8","u8","s4"): return "int8"
    if "bf16" in s or "bf16" in w:                 return "bf16"
    return "fp32"

def peak_gops(cls):
    return COMPUTE[cls] * CPU_FREQ_GHZ

def l2_peak_gbs():
    return L2_BW_B_PER_CYCLE * CPU_FREQ_GHZ

def ws_bytes(m, k, n, dt):
    p = dt.lower().split(":")
    return int(m*k*esz(p[0]) + k*n*esz(p[1]) + m*n*esz(p[2] if len(p)>2 else "f32"))

def zone_label(ws_kb):
    """Fine-grained cache residency zone based on working set size.

    Returns one of: L1, L1end, L2^, L2, L2end, L3^, L3, L3end, DRAM^, DRAM.
    Thresholds:
      0-80%   of level capacity  → comfortably resident
      80-100% of level capacity  → approaching capacity wall  ("end")
      100-120% of level capacity → just spilled from level    ("^" cliff)
    """
    if   ws_kb <= L1D_KB * 0.80: return "L1"
    elif ws_kb <= L1D_KB:        return "L1end"
    elif ws_kb <= L1D_KB * 1.20: return "L2^"
    elif ws_kb <= L2_KB  * 0.80: return "L2"
    elif ws_kb <= L2_KB:         return "L2end"
    elif ws_kb <= L2_KB  * 1.20: return "L3^"
    elif ws_kb <= L3_KB  * 0.80: return "L3"
    elif ws_kb <= L3_KB:         return "L3end"
    elif ws_kb <= L3_KB  * 1.20: return "DRAM^"
    else:                        return "DRAM"

def roof_bw(ws_kb):
    """Blended roofline BW ceiling (GB/s) based on working set size.

    Uses a weighted-average model: fraction of WS that fits in the faster
    cache level is served at that speed; the rest from the next level.
    """
    l2_bw = l2_peak_gbs()

    if ws_kb <= L2_KB:
        return l2_bw
    elif ws_kb <= L3_KB:
        frac_l2 = min(L2_KB / ws_kb, 1.0)
        return frac_l2 * l2_bw + (1.0 - frac_l2) * L3_BW_GBS
    else:
        frac_l3 = min(L3_KB / ws_kb, 1.0)
        return frac_l3 * L3_BW_GBS + (1.0 - frac_l3) * DRAM_BW_GBS

CORES_PER_CCD = 8

def _scaled_shared_bw(nt):
    """Per-core share of L3 and DRAM BW under thread contention."""
    return L3_BW_GBS / min(nt, CORES_PER_CCD), DRAM_BW_GBS / nt

def blended_bw_l2(pf_l2_pct, pf_l3_pct, pf_dr_pct, nt=1):
    """Effective per-core BW through L2 port (from HW counters)."""
    l2_bw = l2_peak_gbs()
    l3_bw, dram_bw = _scaled_shared_bw(nt)
    return (pf_l2_pct / 100.0 * l2_bw +
            pf_l3_pct / 100.0 * l3_bw +
            pf_dr_pct / 100.0 * dram_bw)

def blended_bw_l3(pf_l3_pct, pf_dr_pct, nt=1):
    """Effective per-core BW through L3 port (from HW counters)."""
    l3_bw, dram_bw = _scaled_shared_bw(nt)
    l3_frac = pf_l3_pct / (pf_l3_pct + pf_dr_pct) * 100 if (pf_l3_pct + pf_dr_pct) > 0 else 100
    dr_frac = 100 - l3_frac
    return (l3_frac / 100.0 * l3_bw + dr_frac / 100.0 * dram_bw)

def enrich(rows):
    """Add derived metrics to parsed rows (shared across all modes)."""
    for r in rows:
        pk       = peak_gops(r["cls"])
        peak_l2  = l2_peak_gbs()
        g        = gcd(r["k"], r["n"]) or 1
        wt_dt    = r["dt"].split(":")[1] if ":" in r["dt"] else r["dt"]

        r["peak_gops_val"] = pk
        r["comp_pct"]      = r["gflops"] / pk * 100 if pk else 0
        r["l2_bw_pct"]     = r["bw"] / peak_l2 * 100 if peak_l2 else 0
        r["kn_ratio"]      = f"{r['k']//g}:{r['n']//g}"
        r.setdefault("b_kb", r["k"] * r["n"] * esz(wt_dt) / 1024)
        r["ws_kb"]         = r["ws"] / 1024
        r["zone"]          = zone_label(r["ws_kb"])
        r["roof_bw_val"]   = roof_bw(r["ws_kb"])

        is_ovhd = r["zone"] in ("L1", "L1end")
        r["is_overhead"]   = is_ovhd
        r["bw_eff_pct"]    = (r["bw"] / r["roof_bw_val"] * 100
                              if r["roof_bw_val"] > 0 and not is_ovhd else 0)
    return rows

# ── Parsers ─────────────────────────────────────────────────────────────────

def parse(path):
    """Parse benchdnn timing-only output."""
    with open(path) as f:
        lines = f.readlines()

    hdr = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith("M") and "GFLOPS" in ln:
            hdr = i; break
    if hdr is None:
        return []

    rows = []
    for ln in lines[hdr+1:]:
        s = ln.strip()
        if not s or s.startswith("Timing"): continue
        c = s.split()
        if len(c) < 15: continue
        try:
            m, k, n = int(c[0]), int(c[1]), int(c[2])
            iters, dt = int(c[3]), c[4]
            gflops, avg_ms, tot_ms = float(c[-1]), float(c[-2]), float(c[-3])
        except (ValueError, IndexError):
            continue

        ws  = ws_bytes(m, k, n, dt)
        flp = 2.0 * m * k * n
        ai  = flp / ws if ws else 0
        bw  = (ws * iters / (tot_ms / 1000)) / 1e9 if tot_ms else 0

        rows.append(dict(
            m=m, k=k, n=n, iters=iters, dt=dt, cls=classify(dt),
            tot_ms=tot_ms, avg_ms=avg_ms, gflops=gflops,
            ws=ws, ai=ai, bw=bw))
    return rows

def extract_algo(filename):
    m = re.search(r'algo(\d+)', os.path.basename(filename))
    return m.group(1) if m else os.path.basename(filename)

def parse_perf_raw(path):
    """Parse perf counter data (internal or external format)."""
    with open(path) as f:
        content = f.read()

    if '=== SHAPE' in content:
        blocks = re.split(r'=== SHAPE \d+/\d+ ===', content)
    else:
        lines = content.split('\n')
        blocks = []
        current_block = []
        for line in lines:
            csv_fields = [x.strip() for x in line.split(',')]
            is_csv = (len(csv_fields) >= 19 and csv_fields[0].isdigit()
                      and not line.strip().startswith("M"))
            if is_csv and current_block:
                blocks.append('\n'.join(current_block))
                current_block = []
            current_block.append(line)
        if current_block:
            blocks.append('\n'.join(current_block))

    rows = []
    for block in blocks:
        if not block.strip():
            continue

        m = k = n = iters = 0
        dt = ""
        gflops = tot_ms = 0.0
        counters = {}

        for line in block.split('\n'):
            line = line.strip()

            if line.startswith("INPUT:"):
                fields = [x.strip() for x in line[6:].split(',')]
                if len(fields) >= 5:
                    try:
                        m, k, n, iters = int(fields[0]), int(fields[1]), int(fields[2]), int(fields[3])
                        dt = fields[4]
                    except ValueError:
                        pass
                continue

            csv_fields = [x.strip() for x in line.split(',')]
            if len(csv_fields) >= 19 and not line.startswith("M"):
                try:
                    tm = float(csv_fields[-1])
                    if tm > 0:
                        if m == 0:
                            f0 = csv_fields[0]
                            if not f0.isdigit():
                                continue
                            m_val = int(f0)
                            k_val = int(csv_fields[1])
                            n_str = csv_fields[2]
                            if ':' in n_str:
                                n_str = n_str.split(':')[0]
                            n_val = int(n_str)
                            iters_val = int(csv_fields[3])
                            m, k, n, iters = m_val, k_val, n_val, iters_val
                            dt = csv_fields[4]
                        tot_ms = tm
                        flops = 2.0 * m * k * n * iters
                        gflops = flops / (tot_ms * 1e6)
                except (ValueError, IndexError):
                    pass
                continue

            if line.startswith("[PERF]"):
                continue

            if line.startswith("[ARCH]"):
                arch_m = re.search(r'(Zen\d)', line, re.IGNORECASE)
                if arch_m:
                    akey = arch_m.group(1).lower()
                    if akey in ARCH_CONSTANTS:
                        set_arch(akey)
                continue

            ctr_match = re.match(r'([\d,]+)\s+(\S+)', line)
            if not ctr_match:
                continue
            try:
                val = int(ctr_match.group(1).replace(',', ''))
            except ValueError:
                continue
            ename = ctr_match.group(2)
            if   ename == 'L1-dcache-loads':        counters['l1_ld'] = val
            elif ename == 'L1-dcache-load-misses':  counters['l1_miss'] = val
            elif ename == 'rFF70':                  counters['l2pf_hit'] = val
            elif ename == 'rFF71':                  counters['l2pf_l3hit'] = val
            elif ename == 'rFF72':                  counters['l2pf_dram'] = val
            elif ename == 'rF064':                  counters['l2_hit'] = val
            elif ename == 'r0864':                  counters['l2_miss'] = val
            elif ename == 'r0F45':                  counters['dtlb_l2_hit'] = val
            elif ename == 'rF045':                  counters['dtlb_l2_miss'] = val
            elif ename == 'r00C0':                  counters['ret_insn'] = val
            elif ename == 'r0076':                  counters['cycles'] = val
            elif ename == 'r20AE':                  counters['fp_reg_stall'] = val
            elif ename == 'r40AE':                  counters['fp_sched_stall'] = val
            elif ename == 'r02AE':                  counters['lq_stall'] = val
            elif ename == 'r20AF':                  counters['retire_stall'] = val

        if m == 0 or k == 0 or n == 0:
            continue

        ws = ws_bytes(m, k, n, dt)
        avg_ms = tot_ms / iters if iters > 0 else 0
        bw = (ws * iters / (tot_ms / 1000)) / 1e9 if tot_ms > 0 else 0
        ai = (2.0 * m * k * n) / ws if ws > 0 else 0
        cls = classify(dt)
        wt_dt = dt.split(":")[1] if ":" in dt else dt
        b_kb = k * n * esz(wt_dt) / 1024

        l1_ld      = counters.get('l1_ld', 0)
        l1_miss    = counters.get('l1_miss', 0)
        l2pf_hit   = counters.get('l2pf_hit', 0)
        l2pf_l3hit = counters.get('l2pf_l3hit', 0)
        l2pf_dram  = counters.get('l2pf_dram', 0)
        l2_hit     = counters.get('l2_hit', 0)
        l2_miss    = counters.get('l2_miss', 0)

        l1m_pct = 100.0 * l1_miss / l1_ld if l1_ld > 0 else 0
        l2_all_hit  = l2_hit + l2pf_hit
        l2_all_miss = l2_miss + l2pf_l3hit + l2pf_dram
        l2_all      = l2_all_hit + l2_all_miss
        l2m_pct = 100.0 * l2_all_miss / l2_all if l2_all > 0 else 0
        l3_accesses = l2_miss + l2pf_l3hit + l2pf_dram
        l3_misses   = l2pf_dram
        l3m_pct = 100.0 * l3_misses / l3_accesses if l3_accesses > 0 else 0
        pf_total = l2pf_hit + l2pf_l3hit + l2pf_dram
        pf_l2 = 100.0 * l2pf_hit    / pf_total if pf_total > 0 else 0
        pf_l3 = 100.0 * l2pf_l3hit  / pf_total if pf_total > 0 else 0
        pf_dr = 100.0 * l2pf_dram   / pf_total if pf_total > 0 else 0

        l2_bw_measured = (l1_miss * 64) / (tot_ms / 1000) / 1e9 if tot_ms > 0 else 0
        l3_traffic = l2_miss + l2pf_l3hit + l2pf_dram
        l3_bw_measured = (l3_traffic * 64) / (tot_ms / 1000) / 1e9 if tot_ms > 0 else 0

        rows.append(dict(
            m=m, k=k, n=n, iters=iters, dt=dt, cls=cls,
            tot_ms=tot_ms, avg_ms=avg_ms, gflops=gflops,
            ws=ws, ai=ai, bw=bw, b_kb=b_kb,
            l2_bw_meas=l2_bw_measured, l3_bw_meas=l3_bw_measured,
            l1m_pct=l1m_pct, l2m_pct=l2m_pct, l3m_pct=l3m_pct,
            pf_l2=pf_l2, pf_l3=pf_l3, pf_dr=pf_dr))
    return rows

# ── Output: shared ──────────────────────────────────────────────────────────

def print_arch_header(mode_label="Single Core Analysis"):
    peak_l2 = l2_peak_gbs()
    print(f"\n{'='*120}")
    print(f"  AMD EPYC ({ARCH_NAME}) — {mode_label}")
    print(f"  CPU freq: {CPU_FREQ_GHZ} GHz | L1d: {L1D_KB} KB | L2: {L2_KB} KB"
          f" | L3: {L3_KB//1024} MB/CCD | L2 BW: {L2_BW_B_PER_CYCLE} B/cycle")
    print(f"  NOTE: Actual freq under sustained AVX-512 may be ~3-5% lower than max boost.")
    print(f"{'='*120}")
    print(f"  Compute peaks/core:")
    for label, key in [("FP32","fp32"), ("BF16","bf16"), ("INT8 VNNI","int8")]:
        print(f"    {label:>10}: {peak_gops(key):>8.2f} GOPS  ({COMPUTE[key]} OPS/cycle)")
    print(f"  L2 BW/core    : {peak_l2:.2f} GB/s  ({L2_BW_B_PER_CYCLE} B/cycle)")

def print_formulas():
    peak_l2 = l2_peak_gbs()
    W = SIMD_WIDTH

    print(f"\n{'='*120}")
    print(f"  Formulas")
    print(f"{'='*120}")

    print(f"""
  1. Theoretical Peak (GOPS per core)
     ------------------------------------
     Peak = SIMD_Width/DataType_Size x Num_Cores x FMA_Per_Core x OPS_Per_FMA x DoublePump x Freq

     Constants ({ARCH_NAME}):
       SIMD_Width           = {W} bits (native {W}-bit per pipe)
       Num_Cores            = 1
       FMA_Per_Core         = {FMA_PER_CORE} ({FMA_PER_CORE} independent FMA pipes)
       OPS_Per_FMA          = {OPS_PER_FMA} (1 multiply + 1 add)
       DoublePumpMultiplier = {DOUBLE_PUMP} (Turin=1 native | Genoa=0.5 double-pumped)

     INT8: {W}/8  x {FMA_PER_CORE} x {OPS_PER_FMA} x {DOUBLE_PUMP} = {COMPUTE['int8']} OPS/cycle  x {CPU_FREQ_GHZ} GHz = {peak_gops('int8'):.2f} GOPS
     BF16: {W}/16 x {FMA_PER_CORE} x {OPS_PER_FMA} x {DOUBLE_PUMP} = {COMPUTE['bf16']} OPS/cycle  x {CPU_FREQ_GHZ} GHz = {peak_gops('bf16'):.2f} GOPS
     FP32: {W}/32 x {FMA_PER_CORE} x {OPS_PER_FMA} x {DOUBLE_PUMP} = {COMPUTE['fp32']} OPS/cycle  x {CPU_FREQ_GHZ} GHz = {peak_gops('fp32'):.2f} GOPS

  2. Measured GOPS
     -------------
     FLOPS_per_op  = 2 x M x K x N
     GOPS_measured = FLOPS_per_op x iters / total_time_sec / 1e9

  3. Compute Efficiency
     ----------------------
     Efficiency% = (GOPS_measured / Peak_GOPS) x 100

  4. L2 Bandwidth
     ------------
     L2_peak     = L2_bytes_per_cycle x CPU_freq_GHz
                 = {L2_BW_B_PER_CYCLE} B/cycle x {CPU_FREQ_GHZ} GHz = {peak_l2:.2f} GB/s
     WorkSet     = M*K*sizeof(src) + K*N*sizeof(wt) + M*N*sizeof(dst)
                   u8:s8:f32      -> M*K*1 + K*N*1 + M*N*4  bytes
                   bf16:bf16:bf16 -> M*K*2 + K*N*2 + M*N*2  bytes
     BW_measured = WorkSet x iters / total_time_sec / 1e9   (GB/s)

  5. Arithmetic Intensity (Roofline)
     ------------------------------------
     AI = FLOPS_per_op / WorkSet  =  2*M*K*N / bytes   (FLOPS/byte)
     If AI > roofline_knee -> compute-bound (limited by peak GOPS)
     If AI < roofline_knee -> memory-bound  (limited by L2 BW)
     Roofline knee = Peak_GOPS / L2_peak_GBs
       INT8 : {peak_gops('int8'):.2f} / {peak_l2:.2f} = {peak_gops('int8')/peak_l2:.1f} FLOPS/byte
       BF16 : {peak_gops('bf16'):.2f} / {peak_l2:.2f} = {peak_gops('bf16')/peak_l2:.1f} FLOPS/byte
       FP32 : {peak_gops('fp32'):.2f} / {peak_l2:.2f} = {peak_gops('fp32')/peak_l2:.1f} FLOPS/byte
""")

def print_column_guide(show_perf=False):
    """Print Column Guide matching fine-grained zone definitions and blended BW model."""
    l2_bw = l2_peak_gbs()

    print(f"\n{'='*120}")
    print(f"  Column Guide")
    print(f"{'='*120}")
    print(f"""\
  K:N      = Shape ratio (K dimension : N dimension)
  AI(F/B)  = Arithmetic Intensity (FLOPS/byte) = 2*M*K*N / WorkSet
  B(KB)    = B-matrix (weights) size in KB  (K x N x elem_size)
  WS(KB)   = Full working set in KB  (A + B + C = M*K*src + K*N*wt + M*N*dst)
             (For GEMV M=1, WS ≈ B. For GEMM M>1, WS includes all matrices.)
  Zone     = Cache residency based on full working set WS = A+B+C.
        L1      = WS fits in L1d (≤{L1D_KB}KB). Overhead-dominated.
        L1end   = WS 80-100% of L1d. Approaching L1 capacity wall.
        L2^     = WS 100-120% of L1d. Just spilled from L1 — L1 cliff.
        L2      = WS comfortably in L2 (<80% of {L2_KB//1024}MB). Full L2 fill BW.
        L2end   = WS 80-100% of L2. Approaching L2 capacity wall —
                    conflict evictions as WS crowds L2 associativity.
        L3^     = WS 100-120% of L2. Just spilled from L2 — cache cliff.
                    L3 fill latency (~40 cycles) not fully hidden by prefetcher.
        L3      = WS comfortably in L3. Blended L2+L3 bandwidth.
        L3end   = WS 80-100% of L3/CCD ({L3_KB//1024}MB). Approaching L3 capacity wall.
        DRAM^   = WS 100-120% of L3. Just spilled from L3 — DRAM cliff.
        DRAM    = WS well beyond L3. Blended L3+DRAM bandwidth.
  GOPS     = Measured throughput = 2*M*K*N x iters / time / 1e9
  Comp%    = Compute Efficiency = GOPS / Peak_GOPS x 100
  BW GB/s  = Operational bandwidth = WorkSet x iters / time / 1e9
             Timing-derived: assumes every byte of WS is transferred each iteration.
             Available in all modes (no perf counters needed).
  L2 BW%   = BW GB/s / L2_peak x 100 — timing-based L2 utilization estimate.
             Compare with L2BW%m (--perf) to validate: for L2-resident shapes they
             should be close; divergence means cache reuse or WS model mismatch.
  RoofBW   = Roofline bandwidth ceiling (GB/s) — the bandwidth ceiling for this shape.
             Uses a blended model: the fraction of WS that fits in the faster
             cache is served at that speed; the rest from the next level.
               L1, L1end  : ovhd (overhead-dominated, not BW-limited)
               L2^        : L2 peak BW (data just left L1, fully in L2)
               L2, L2end  : L2 peak BW ({l2_bw:.0f} GB/s)
               L3^, L3, L3end (example: WS=2MB, L2={L2_KB//1024}MB):
                   frac_L2 = L2_size / WS = {L2_KB}KB / 2048KB = 0.50 (50%)
                   frac_L3 = 1 - frac_L2 = 0.50 (50%)
                   RoofBW  = frac_L2 x L2_BW + frac_L3 x L3_BW
                           = 0.50 x {l2_bw:.0f} + 0.50 x {L3_BW_GBS:.0f} = {0.5*l2_bw + 0.5*L3_BW_GBS:.0f} GB/s
                 As WS grows, frac_L2 shrinks → RoofBW drops toward L3 speed:
                   WS≈1.1MB (L3^):   {L2_KB/(L2_KB*1.1):.0%} L2 + {1 - L2_KB/(L2_KB*1.1):.0%} L3 → ~{L2_KB/(L2_KB*1.1)*l2_bw + (1-L2_KB/(L2_KB*1.1))*L3_BW_GBS:.0f} GB/s
                   WS=2MB   (L3):    50% L2 + 50% L3 → ~{0.5*l2_bw + 0.5*L3_BW_GBS:.0f} GB/s
                   WS=8MB   (L3):    12% L2 + 88% L3 → ~{0.12*l2_bw + 0.88*L3_BW_GBS:.0f} GB/s
                   WS=26MB  (L3end):  4% L2 + 96% L3 → ~{0.04*l2_bw + 0.96*L3_BW_GBS:.0f} GB/s
               DRAM^, DRAM (example: WS=64MB, L3={L3_KB//1024}MB):
                   frac_L3  = L3_size / WS = {L3_KB//1024}MB / 64MB = 0.50 (50%)
                   frac_DRAM = 1 - frac_L3 = 0.50 (50%)
                   RoofBW   = frac_L3 x L3_BW + frac_DRAM x DRAM_BW
                            = 0.50 x {L3_BW_GBS:.0f} + 0.50 x {DRAM_BW_GBS:.0f} = {0.5*L3_BW_GBS + 0.5*DRAM_BW_GBS:.0f} GB/s
             Larger WS within a zone → lower RoofBW (more spill to slower level).
  BWEff%   = BW Efficiency = BW GB/s / RoofBW x 100
             How close the kernel's timing-derived bandwidth is to the roofline
             bandwidth ceiling for this shape's cache residency zone.
               >80%: near-optimal | 50-80%: room to improve | <50%: overhead-limited
             THIS IS THE KEY METRIC for evaluating kernel quality.

  ── Comparing modes: without vs with perf counters ───────────────────────────
  ┌────────────┬──────────────────────────────────────────────────────────────┐
  │ Column     │ Source and meaning                                          │
  ├────────────┼──────────────────────────────────────────────────────────────┤
  │ BW GB/s    │ WorkSet × iters / time  (timing-derived, both modes)        │
  │ L2 BW%     │ BW GB/s / L2_peak × 100  (timing estimate, both modes)     │
  │ BWEff%     │ BW GB/s / RoofBW × 100   (key metric, both modes)          │
  │ L2BW%m     │ L1_misses × 64B / time / L2_peak  (HW counters, --perf)   │
  └────────────┴──────────────────────────────────────────────────────────────┘
  L2 BW% (timing) vs L2BW%m (counters):
    For L2-resident GEMV: should be close (every iter reloads B from L2).
    For L3/DRAM shapes: may diverge — timing assumes full WS transfer,
      counters measure actual L1 miss traffic through L2 port.
    Large gap → cache reuse across iterations or WS model inaccuracy.""")

    if show_perf:
        peak_l2 = l2_peak_gbs()
        print(f"""
{'='*120}
  HW Counter Columns — AMD {ARCH_NAME} PMU (perf stat counting mode)
{'='*120}

  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │                     AMD EPYC ({ARCH_NAME}) Cache Hierarchy                     │
  │                                                                                 │
  │   ┌──────┐     ┌───────────┐     ┌───────────┐     ┌───────────┐     ┌──────┐  │
  │   │ DRAM │────→│ L3 ({L3_KB//1024}MB) │────→│ L2 ({L2_KB//1024}MB)  │────→│ L1d({L1D_KB}KB) │────→│ Regs │  │
  │   │      │     │  per CCD  │     │ per core  │     │ per core  │     │      │  │
  │   └──────┘     └───────────┘     └─────┬─────┘     └─────┬─────┘     └──────┘  │
  │                                        │                  │                     │
  │                                   L2 Prefetcher      L1 Prefetcher              │
  │                                   (tracked here)     (not tracked)              │
  │                                                                                 │
  │   Data flows LEFT→RIGHT.  The L2 prefetcher is part of the core's L2            │
  │   cache controller. It detects streaming patterns and proactively issues         │
  │   requests to bring data into L2 BEFORE the core demands it.                    │
  │   PMU events tell us WHERE the prefetcher found the data (L2/L3/DRAM).          │
  └─────────────────────────────────────────────────────────────────────────────────┘

  ── Cache Miss Rates ────────────────────────────────────────────────────────────

    L1miss  = L1 Data Cache Miss Rate
              = L1_misses / L1_loads × 100
              Source: L1-dcache-loads, L1-dcache-load-misses

    L2miss  = L2 Total Miss Rate (demand + prefetch combined)
              = (DC_miss + PF_miss_L2_hit_L3 + PF_miss_L2_L3)
                / (DC_hit + DC_miss + PF_hit_L2 + PF_miss_L2_hit_L3 + PF_miss_L2_L3) × 100
              Source: PMCx064 (demand), PMCx070-072 (prefetcher)

    L3miss  = L3 Miss Rate (of accesses that reached L3)
              = PF_miss_L2_L3 / (DC_miss + PF_miss_L2_hit_L3 + PF_miss_L2_L3) × 100
              Approximation using prefetcher DRAM misses as numerator.

  ── L2 Prefetcher Source Breakdown (PF→L2 + PF→L3 + PF→DR ≈ 100%) ─────────────

    PF→L2   = Prefetcher found data already in L2  (PMCx070: L2PfHitL2)
    PF→L3   = Prefetcher fetched data from L3      (PMCx071: L2PfMissL2HitL3)
    PF→DR   = Prefetcher fetched data from DRAM    (PMCx072: L2PfMissL2L3)

  ── L2 BW% (HW counter-based) ────────────────────────────────────────────────

    L2 BW (GB/s) = L1_dcache_load_misses × 64 bytes / total_time / num_threads
    L2 BW%m      = L2 BW / L2_peak_per_core × 100

    L2 peak = {peak_l2:.0f} GB/s ({L2_BW_B_PER_CYCLE} B/cycle × {CPU_FREQ_GHZ} GHz)

  ── Per-shape BW Roofline (Eff / Meas / %Eff columns, shown with -c) ────────────

    L2Eff = PF→L2% × L2_peak + PF→L3% × L3_BW + PF→DRAM% × DRAM_BW
    L2Meas = L1_misses × 64B / time / threads  (actual L2 port throughput)
    L2%Eff = L2Meas / L2Eff × 100

  ── Interpreting by B-matrix size ──────────────────────────────────────────────

    ┌────────────┬────────┬────────┬────────┬────────┬────────┬──────────────────┐
    │  B size    │ L1miss │ L2miss │ L3miss │ PF→L2  │ PF→L3  │ Performance      │
    ├────────────┼────────┼────────┼────────┼────────┼────────┼──────────────────┤
    │ < {L1D_KB} KB    │ Low    │ Low    │ Low    │ Med    │ Low    │ Overhead-limited │
    │ {L1D_KB}KB–{L2_KB//1024}MB   │ Rising │ LOW    │ Low    │ HIGH   │ Low    │ ★ PEAK BW/GOPS  │
    │ {L2_KB//1024}MB–{L3_KB//1024}MB   │ High   │ HIGH   │ Low    │ Low    │ HIGH   │ L3 streaming     │
    │ > {L3_KB//1024} MB    │ High   │ High   │ HIGH   │ Low    │ Low    │ DRAM bottleneck  │
    └────────────┴────────┴────────┴────────┴────────┴────────┴──────────────────┘
""")

# ── Single-file analysis ────────────────────────────────────────────────────

def print_section(title, rows):
    """Print a table section with enriched metrics."""
    print(f"\n{'='*155}")
    print(f"  {title}")
    print(f"{'='*155}")

    h = (f"{'M':>5} {'K':>6} {'N':>6} {'K:N':>6}  {'DType':>16}"
         f"  {'WS(KB)':>8} {'Zone':>5}  {'Tot(ms)':>10}"
         f"  {'GOPS':>10}  {'Comp%':>7}"
         f"  {'BW GB/s':>9}  {'L2 BW%':>7}  {'RoofBW':>7}  {'BWEff%':>7}"
         f"  {'AI(F/B)':>8}  {'B(KB)':>8}")
    print(h)
    print("-" * len(h))

    peak_l2 = l2_peak_gbs()
    for r in rows:
        l2_bw_pct = r["bw"] / peak_l2 * 100 if peak_l2 else 0

        if r["is_overhead"]:
            roof_str = f"{'ovhd':>7}"
            eff_str  = f"{'':>7}"
        else:
            roof_str = f"{r['roof_bw_val']:>6.0f}"
            eff_str  = f"{min(r['bw_eff_pct'], 999):>5.0f}%"

        print(f"{r['m']:>5} {r['k']:>6} {r['n']:>6} {r['kn_ratio']:>6}"
              f"  {r['dt']:>16}"
              f"  {r['ws_kb']:>8.1f} {r['zone']:>5}  {r['tot_ms']:>10.2f}"
              f"  {r['gflops']:>10.2f}  {r['comp_pct']:>6.2f}%"
              f"  {r['bw']:>9.2f}  {l2_bw_pct:>6.1f}%  {roof_str}  {eff_str}"
              f"  {r['ai']:>8.1f}  {r['b_kb']:>8.1f}")

def single_file_analysis(path):
    rows = enrich(parse(path))
    if not rows:
        sys.exit("No data rows parsed.")

    int8_gemm = [r for r in rows if r["cls"] == "int8" and r["m"] > 1]
    bf16_gemm = [r for r in rows if r["cls"] == "bf16" and r["m"] > 1]
    int8_gemv = [r for r in rows if r["cls"] == "int8" and r["m"] == 1]
    bf16_gemv = [r for r in rows if r["cls"] == "bf16" and r["m"] == 1]

    print_arch_header("Single Core Analysis")
    print_formulas()

    if int8_gemm: print_section("INT8 GEMM  (u8:s8:f32)", int8_gemm)
    if bf16_gemm: print_section("BF16 GEMM  (bf16:bf16:bf16)", bf16_gemm)
    if int8_gemv: print_section("INT8 GEMV  (M=1, u8:s8:f32)", int8_gemv)
    if bf16_gemv: print_section("BF16 GEMV  (M=1, bf16:bf16:bf16)", bf16_gemv)

    print_column_guide(show_perf=False)

# ── Multi-algo comparison ───────────────────────────────────────────────────

def compare_algos(files):
    peak_l2 = l2_peak_gbs()

    algo_data = {}
    for fpath in files:
        algo_id = extract_algo(fpath)
        rows = enrich(parse(fpath))
        if not rows:
            print(f"WARNING: no data parsed from {fpath}", file=sys.stderr)
            continue
        algo_data[algo_id] = {}
        for r in rows:
            key = (r["m"], r["k"], r["n"], r["dt"])
            algo_data[algo_id][key] = r

    algo_ids = sorted(algo_data.keys(), key=lambda x: int(x) if x.isdigit() else 999)
    if not algo_ids:
        sys.exit("No data parsed from any file.")

    all_keys = set()
    for ad in algo_data.values():
        all_keys.update(ad.keys())
    all_keys = sorted(all_keys, key=lambda x: (x[3], x[0], x[1]*x[2], x[1], x[2]))

    dtypes_in_data = sorted(set(k[3] for k in all_keys))
    has_gemv = any(k[0] == 1 for k in all_keys)
    has_gemm = any(k[0] > 1 for k in all_keys)

    print_arch_header("Multi-Algo Comparison")

    col_w = 9 + 1 + 7 + 1 + 7 + 2  # width of per-algo sub-columns

    def _print_compare_table(title, shape_keys):
        if not shape_keys:
            return
        print(f"\n{'='*160}")
        print(f"  {title}")
        print(f"{'='*160}")

        h = f"{'M':>5} {'K':>6} {'N':>6} {'K:N':>8} {'Zone':>5} {'WS(KB)':>8} {'B(KB)':>8}"
        for aid in algo_ids:
            name = ALGO_NAMES.get(aid, f"A{aid}")
            label = f"--- {name} ---"
            h += f"  {label:>{col_w}}"
        h += f"  {'Best':>12}"
        print(h)

        sub = f"{'':>5} {'':>6} {'':>6} {'':>8} {'':>5} {'':>8} {'':>8}"
        for _ in algo_ids:
            sub += f"  {'Tot(ms)':>9} {'Eff%':>7} {'GB/s':>7}"
        sub += f"  {'':>12}"
        print(sub)
        print("-" * len(h))

        for key in shape_keys:
            m, k, n, dt = key
            ref = None
            for aid in algo_ids:
                ref = algo_data[aid].get(key)
                if ref:
                    break
            if not ref:
                continue

            line = (f"{m:>5} {k:>6} {n:>6} {ref['kn_ratio']:>8}"
                    f" {ref['zone']:>5} {ref['ws_kb']:>8.1f} {ref['b_kb']:>8.1f}")

            best_eff = -1
            best_algo = ""
            for aid in algo_ids:
                r = algo_data[aid].get(key)
                if r:
                    eff = r["bw_eff_pct"] if not r["is_overhead"] else r["l2_bw_pct"]
                    line += f"  {r['tot_ms']:>9.2f} {eff:>6.1f}% {r['bw']:>7.1f}"
                    if eff > best_eff:
                        best_eff = eff
                        best_algo = ALGO_NAMES.get(aid, f"A{aid}")
                else:
                    line += f"  {'---':>9} {'---':>7} {'---':>7}"
            line += f"  {best_algo:>12}"
            print(line)

    gemv_keys = [k for k in all_keys if k[0] == 1]
    gemm_keys = [k for k in all_keys if k[0] > 1]

    if gemv_keys:
        _print_compare_table("GEMV (M=1) — Multi-Algo L2 BW Comparison", gemv_keys)
    if gemm_keys:
        _print_compare_table("GEMM (M>1) — Multi-Algo Comparison", gemm_keys)

    # ── Summary: average BW efficiency per algo per zone ──
    print(f"\n{'='*120}")
    print(f"  Summary: Average L2BWEff% by Zone")
    print(f"{'='*120}")

    zones_seen = []
    for z in ["L1","L1end","L2^","L2","L2end","L3^","L3","L3end","DRAM^","DRAM"]:
        if any(algo_data[aid].get(key, {}).get("zone") == z
               for key in all_keys for aid in algo_ids):
            zones_seen.append(z)

    h2 = f"{'Zone':>8} {'Count':>6}"
    for aid in algo_ids:
        h2 += f"  {ALGO_NAMES.get(aid, f'A{aid}'):>12}"
    h2 += f"  {'Best Algo':>12}"
    print(h2)
    print("-" * len(h2))

    for zn in zones_seen:
        sums = {aid: [] for aid in algo_ids}
        cnt = 0
        for key in all_keys:
            for aid in algo_ids:
                r = algo_data[aid].get(key)
                if r and r["zone"] == zn:
                    if cnt == 0 or key not in [(sk[0], sk[1], sk[2], sk[3])
                                               for sk in list(all_keys)[:cnt]]:
                        cnt_this = True
                    eff = r["bw_eff_pct"] if not r["is_overhead"] else r["l2_bw_pct"]
                    sums[aid].append(eff)
        cnt = max(len(v) for v in sums.values()) if sums else 0

        line = f"{zn:>8} {cnt:>6}"
        best_avg = -1
        best_a = ""
        for aid in algo_ids:
            avg = sum(sums[aid]) / len(sums[aid]) if sums[aid] else 0
            line += f"  {avg:>11.1f}%"
            if avg > best_avg:
                best_avg = avg
                best_a = ALGO_NAMES.get(aid, f"A{aid}")
        line += f"  {best_a:>12}"
        print(line)

    # ── Peak BW per algo ──
    print(f"\n{'='*120}")
    print(f"  Peak BW per Algo (best single shape)")
    print(f"{'='*120}")
    for aid in algo_ids:
        name = ALGO_NAMES.get(aid, f"A{aid}")
        best_r = None
        best_pct = 0
        for key, r in algo_data[aid].items():
            pct = r["bw_eff_pct"] if not r["is_overhead"] else r["l2_bw_pct"]
            if pct > best_pct:
                best_pct = pct
                best_r = r
        if best_r:
            print(f"  {name:<15}: {best_pct:>6.1f}% L2BWEff  ({best_r['bw']:.1f} GB/s)  "
                  f"at M={best_r['m']}, K={best_r['k']}, N={best_r['n']}  [{best_r['zone']}]")

    # ── Win count ──
    print(f"\n{'='*120}")
    print(f"  Win Count (which algo has highest L2BWEff% per shape)")
    print(f"{'='*120}")
    wins = {aid: 0 for aid in algo_ids}
    total = 0
    for key in all_keys:
        total += 1
        best_pct = -1
        best_aid = ""
        for aid in algo_ids:
            r = algo_data[aid].get(key)
            if r:
                pct = r["bw_eff_pct"] if not r["is_overhead"] else r["l2_bw_pct"]
                if pct > best_pct:
                    best_pct = pct
                    best_aid = aid
        if best_aid:
            wins[best_aid] += 1

    for aid in algo_ids:
        name = ALGO_NAMES.get(aid, f"A{aid}")
        pct = wins[aid] / total * 100 if total else 0
        print(f"  {name:<15}: {wins[aid]:>3}/{total} shapes  ({pct:.0f}%)")

    print_column_guide(show_perf=False)

# ── Per-shape perf stat analysis ────────────────────────────────────────────

def perf_analysis(path, verbose=False, bottleneck=False, num_threads=1, clevel=None):
    rows = enrich(parse_perf_raw(path))
    if not rows:
        sys.exit("No data parsed from perf raw file.")

    peak_l2 = l2_peak_gbs()
    nt = num_threads
    mode = "Single Core" if nt == 1 else f"{nt}-Thread (per-core normalized)"

    print_arch_header(f"{mode} Analysis + HW Counters")
    if nt > 1:
        print(f"  Threads: {nt} | GOPS = system total | Comp%, L2BW%, L2BW%m = per-core (÷ {nt})")
    print_formulas()

    gemv = [r for r in rows if r["m"] == 1]
    gemm = [r for r in rows if r["m"] > 1]

    CL = clevel.upper() if clevel else None
    S = "│"

    for label, subset in [("GEMV (M=1)", gemv), ("GEMM (M>1)", gemm)]:
        if not subset:
            continue

        sub_dtypes = sorted(set(r["cls"] for r in subset))
        dtype_label = "/".join(c.upper() for c in sub_dtypes)
        print(f"\n{'='*170}")
        print(f"  {dtype_label} {label}  —  Timing + L2/L3 HW Counters (perf stat)")
        print(f"{'='*170}")

        if verbose:
            h = (f"  {'M':>5} {'K':>6} {'N':>6} {'K:N':>6} {'DType':>14}"
                 f" {'WS(KB)':>8} {'Zone':>5}"
                 f" {'Tot(ms)':>9} {'GOPS':>8} {'Comp%':>7}"
                 f" {'BW GB/s':>8} {'L2 BW%':>7} {'RoofBW':>7} {'BWEff%':>7}"
                 f" {'AI':>5} {'B(KB)':>8}"
                 f"  {S} {'L1miss':>7} {'L2miss':>7} {'L3miss':>7}"
                 f" {'PF_L2':>7} {'PF_L3':>7} {'PF_DR':>7}"
                 f" {'L2BW%m':>7}")
        else:
            h = (f"  {'M':>5} {'K':>6} {'N':>6} {'DType':>14}"
                 f" {'WS(KB)':>8} {'Zone':>5}"
                 f" {'GOPS':>8} {'Comp%':>7} {'RoofBW':>7} {'BWEff%':>7}"
                 f"  {S} {'L1miss':>7} {'L2miss':>7} {'L3miss':>7}"
                 f" {'PF_L2':>7} {'PF_L3':>7} {'PF_DR':>7}"
                 f" {'L2BW%m':>7}")
        if CL:
            eff_hdr = CL + "Eff"
            meas_hdr = CL + "Meas"
            pct_hdr = CL + "%Eff"
            h += f"  {S} {eff_hdr+' GB/s':>12} {meas_hdr+' GB/s':>12} {pct_hdr:>7}"
        if bottleneck:
            h += f"  {S} Bottleneck"
        print(h)
        print("  " + "─" * (len(h) - 2))

        for r in subset:
            pk = r["peak_gops_val"]
            roofline_knee = pk / peak_l2 if peak_l2 > 0 else 1.0

            gops_pc   = r["gflops"] / nt
            bw_pc     = r["bw"] / nt
            l2bw_m_pc = r["l2_bw_meas"] / nt

            eff       = gops_pc / pk * 100 if pk else 0
            l2bw_m_pct = l2bw_m_pc / peak_l2 * 100 if peak_l2 else 0

            l2_bw_pct = bw_pc / peak_l2 * 100 if peak_l2 else 0

            if r["is_overhead"]:
                roof_str = f"{'ovhd':>7}"
                eff_roof_str = f"{'':>7}"
            else:
                roof_str = f"{r['roof_bw_val']:>6.0f}"
                eff_roof = min(bw_pc / r['roof_bw_val'] * 100, 999) if r['roof_bw_val'] > 0 else 0
                eff_roof_str = f"{eff_roof:>5.0f}%"

            b_kb = r["b_kb"]

            # Bottleneck commentary
            if bottleneck:
                if b_kb < L1D_KB:
                    comment = (f"Overhead-dominated: B={b_kb:.0f}KB in L1d ({L1D_KB}KB), "
                               f"dispatch/timing overhead > compute.")
                elif b_kb <= 512 and r["pf_l2"] >= 80:
                    if l2bw_m_pct >= 65:
                        comment = (f"L2 BW-bound (near-optimal): B={b_kb:.0f}KB in L2, "
                                   f"PF→L2={r['pf_l2']:.0f}%, L2 port at {l2bw_m_pct:.0f}%.")
                    else:
                        comment = (f"L2 BW-bound (scaling): B={b_kb:.0f}KB, "
                                   f"PF→L2={r['pf_l2']:.0f}%, L2 port at {l2bw_m_pct:.0f}%.")
                elif b_kb <= L2_KB and r["l2m_pct"] <= 25:
                    comment = (f"L2-L3 transition: B={b_kb:.0f}KB at L2 edge, "
                               f"PF→L2 dropping to {r['pf_l2']:.0f}%.")
                elif r["pf_l3"] >= 40 and r["l2m_pct"] >= 30:
                    if l2bw_m_pct >= 50:
                        comment = (f"L3 BW-bound (PF effective): B={b_kb:.0f}KB in L3, "
                                   f"PF→L3={r['pf_l3']:.0f}%.")
                    elif l2bw_m_pct >= 38:
                        comment = (f"L3 latency-bound: B={b_kb:.0f}KB, "
                                   f"PF→L3={r['pf_l3']:.0f}%, L2 port at {l2bw_m_pct:.0f}%.")
                    else:
                        comment = (f"L3 BW-limited: B={b_kb:.0f}KB, "
                                   f"PF→L3={r['pf_l3']:.0f}%, L2 port at {l2bw_m_pct:.0f}%.")
                elif r["pf_dr"] >= 10:
                    comment = (f"DRAM BW-bound: B={b_kb:.0f}KB > L3, "
                               f"PF→DRAM={r['pf_dr']:.0f}%.")
                elif r["ai"] >= roofline_knee:
                    comment = (f"Compute-bound: AI={r['ai']:.1f} > knee ({roofline_knee:.1f}).")
                else:
                    comment = (f"Memory-bound: AI={r['ai']:.1f}, "
                               f"B={b_kb:.0f}KB, L2 port at {l2bw_m_pct:.0f}%.")

            # Per-shape counter-based BW roofline (only with -c)
            roof_suffix = ""
            if CL:
                l1_resident = b_kb < L1D_KB and r["l1m_pct"] < 5.0
                l2_resident = b_kb < L2_KB and r["l2m_pct"] < 5.0

                if CL == "L3":
                    if l1_resident:
                        roof_suffix = f"  {S} {'':>12} {'L1':>12} {'bound':>7}"
                    elif l2_resident:
                        roof_suffix = f"  {S} {'':>12} {'L2':>12} {'bound':>7}"
                    else:
                        eff_bw = blended_bw_l3(r["pf_l3"], r["pf_dr"], nt)
                        meas_bw = r["l3_bw_meas"] / nt
                        pct_eff = meas_bw / eff_bw * 100 if eff_bw > 0 else 0
                        roof_suffix = f"  {S} {eff_bw:>12.1f} {meas_bw:>12.1f} {pct_eff:>6.1f}%"
                else:
                    if l1_resident:
                        roof_suffix = f"  {S} {'':>12} {'L1':>12} {'bound':>7}"
                    else:
                        eff_bw = blended_bw_l2(r["pf_l2"], r["pf_l3"], r["pf_dr"], nt)
                        meas_bw = l2bw_m_pc
                        pct_eff = meas_bw / eff_bw * 100 if eff_bw > 0 else 0
                        roof_suffix = f"  {S} {eff_bw:>12.1f} {meas_bw:>12.1f} {pct_eff:>6.1f}%"

            if verbose:
                row = (f"  {r['m']:>5} {r['k']:>6} {r['n']:>6} {r['kn_ratio']:>6} {r['dt']:>14}"
                       f" {r['ws_kb']:>8.1f} {r['zone']:>5}"
                       f" {r['tot_ms']:>9.2f} {gops_pc:>8.2f} {eff:>6.2f}%"
                       f" {bw_pc:>8.2f} {l2_bw_pct:>6.1f}% {roof_str} {eff_roof_str}"
                       f" {r['ai']:>5.1f} {r['b_kb']:>8.1f}"
                       f"  {S} {r['l1m_pct']:>6.1f}% {r['l2m_pct']:>6.1f}% {r['l3m_pct']:>6.1f}%"
                       f" {r['pf_l2']:>6.1f}% {r['pf_l3']:>6.1f}% {r['pf_dr']:>6.1f}%"
                       f" {l2bw_m_pct:>6.1f}%"
                       f"{roof_suffix}")
            else:
                row = (f"  {r['m']:>5} {r['k']:>6} {r['n']:>6} {r['dt']:>14}"
                       f" {r['ws_kb']:>8.1f} {r['zone']:>5}"
                       f" {gops_pc:>8.2f} {eff:>6.2f}% {roof_str} {eff_roof_str}"
                       f"  {S} {r['l1m_pct']:>6.1f}% {r['l2m_pct']:>6.1f}% {r['l3m_pct']:>6.1f}%"
                       f" {r['pf_l2']:>6.1f}% {r['pf_l3']:>6.1f}% {r['pf_dr']:>6.1f}%"
                       f" {l2bw_m_pct:>6.1f}%"
                       f"{roof_suffix}")
            if bottleneck:
                row += f"  {S} {comment}"
            print(row)

    print_column_guide(show_perf=True)

# ── CLI ─────────────────────────────────────────────────────────────────────

def build_parser():
    prog = os.path.basename(sys.argv[0])
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Analyze benchdnn matmul output — AMD EPYC (Zen 4 / Zen 5)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
modes:
  {prog} <file>                         Single-file timing analysis
  {prog} --compare <f1> <f2> ...        Multi-algo comparison
  {prog} --perf [flags] <file>          HW counter analysis (perf stat)

examples:
  {prog} build/benchmark_bf16_algo11_1c.txt
  {prog} --compare build/benchmark_algo1_1c.txt build/benchmark_algo11_1c.txt
  {prog} --perf -v -b build/benchmark_perf_raw.txt
  {prog} --perf -t 64 -c L2 build/benchmark_64c_perf_raw.txt
""")

    # mode selection
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--compare", action="store_true",
                      help="Multi-algo comparison: side-by-side metrics for multiple files")
    mode.add_argument("--perf", action="store_true",
                      help="HW counter analysis: merge timing + AMD Zen PMU counters")

    # shared options
    parser.add_argument("--arch", choices=["zen4", "zen5"], default=None,
                        help="CPU architecture (default: auto-detect from [ARCH] line, or zen5)")

    # hardware parameter overrides (override --arch defaults)
    hw = parser.add_argument_group("hardware overrides (override --arch defaults)")
    hw.add_argument("--freq", type=float, default=None, metavar="GHZ",
                    help="CPU frequency in GHz (e.g. 2.7 for boost-off, 4.121 for boost)")
    hw.add_argument("--l2-bw", type=int, default=None, metavar="B/CYC",
                    help="L2 fill bandwidth in bytes/cycle (Zen5=64, Zen4=32)")
    hw.add_argument("--l1d", type=int, default=None, metavar="KB",
                    help="L1d cache size in KB (Zen5=48, Zen4=32)")
    hw.add_argument("--l2", type=int, default=None, metavar="KB",
                    help="L2 cache size in KB (default: 1024)")
    hw.add_argument("--l3", type=int, default=None, metavar="MB",
                    help="L3 cache size per CCD in MB (default: 32)")
    hw.add_argument("--l3-bw", type=float, default=None, metavar="GB/S",
                    help="L3 BW per core in GB/s (Zen5=120, Zen4=80)")
    hw.add_argument("--dram-bw", type=float, default=None, metavar="GB/S",
                    help="DRAM BW per core in GB/s (Zen5=50, Zen4=40)")

    # perf-specific options
    perf_opts = parser.add_argument_group("perf options (only with --perf)")
    perf_opts.add_argument("-v", "--verbose", action="store_true",
                           help="Show all columns (Tot(ms), K:N, AI, B(KB), BW GB/s)")
    perf_opts.add_argument("-b", "--bottleneck", action="store_true",
                           help="Append per-shape bottleneck commentary")
    perf_opts.add_argument("-t", "--threads", type=int, default=1, metavar="N",
                           help="OMP threads used during benchmark (default: 1)")
    perf_opts.add_argument("-c", "--cache-level", choices=["L2", "L3"], default=None,
                           metavar="LEVEL",
                           help="Cache-level BW roofline: L2 or L3 (adds Eff/Meas/%%Eff columns)")

    # positional: input file(s)
    parser.add_argument("files", nargs="+", metavar="FILE",
                        help="Benchmark output file(s)")

    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.arch:
        set_arch(args.arch)

    # Apply hardware overrides (after --arch sets defaults)
    global CPU_FREQ_GHZ, L2_BW_B_PER_CYCLE, L1D_KB, L2_KB, L3_KB
    global L3_BW_GBS, DRAM_BW_GBS
    if args.freq is not None:     CPU_FREQ_GHZ = args.freq; _compute_ops_per_cycle()
    if args.l2_bw is not None:    L2_BW_B_PER_CYCLE = args.l2_bw
    if args.l1d is not None:      L1D_KB = args.l1d
    if args.l2 is not None:       L2_KB = args.l2
    if args.l3 is not None:       L3_KB = args.l3 * 1024
    if args.l3_bw is not None:    L3_BW_GBS = args.l3_bw
    if args.dram_bw is not None:  DRAM_BW_GBS = args.dram_bw

    if args.compare:
        if len(args.files) < 2:
            parser.error("--compare requires at least two files")
        compare_algos(args.files)
    elif args.perf:
        if len(args.files) != 1:
            parser.error("--perf requires exactly one file")
        perf_analysis(args.files[0],
                      verbose=args.verbose,
                      bottleneck=args.bottleneck,
                      num_threads=args.threads,
                      clevel=args.cache_level)
    else:
        if len(args.files) != 1:
            parser.error("single-file mode requires exactly one file")
        single_file_analysis(args.files[0])

if __name__ == "__main__":
    main()
