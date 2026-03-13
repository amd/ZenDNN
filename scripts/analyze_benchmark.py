#!/usr/bin/env python3
"""
Analyze benchdnn matmul output for AMD EPYC (Zen 4 / Zen 5).
Auto-detects architecture from [ARCH] line in perf output; falls back to Zen 5.

Usage:
    Single file:   python3 analyze_benchmark.py <raw_output_file>
    Multi-algo:    python3 analyze_benchmark.py --compare <file1> <file2> ...
                   python3 analyze_benchmark.py --compare build/benchmark_custom_algo*_1c.txt
"""

import sys
import os
import re
from math import gcd

# ── Architecture constants ─────────────────────────────────────────────────
# Selected at startup from [ARCH] line or --arch flag; default = Zen 5.
ARCH_CONSTANTS = {
    "zen4": dict(name="Zen4", family=0x19,
                 cpu_freq_ghz=3.7,
                 l1d_kb=32, l2_kb=1024, l3_mb_per_ccd=32,
                 l2_bw_b_per_cycle=32, dispatch_width=6,
                 l3_bw_gbs=80.0, dram_bw_gbs=40.0),
    "zen5": dict(name="Zen5", family=0x1A,
                 cpu_freq_ghz=4.121,
                 l1d_kb=48, l2_kb=1024, l3_mb_per_ccd=32,
                 l2_bw_b_per_cycle=64, dispatch_width=8,
                 l3_bw_gbs=120.0, dram_bw_gbs=50.0),
}

def set_arch(arch_key="zen5"):
    """Set global constants from the named architecture."""
    global CPU_FREQ_GHZ, L2_BW_B_PER_CYCLE, L1D_KB, L2_KB
    global L3_BW_GBS, DRAM_BW_GBS, ARCH_NAME
    ac = ARCH_CONSTANTS.get(arch_key, ARCH_CONSTANTS["zen5"])
    CPU_FREQ_GHZ = ac["cpu_freq_ghz"]
    L2_BW_B_PER_CYCLE = ac["l2_bw_b_per_cycle"]
    L1D_KB = ac["l1d_kb"]
    L2_KB = ac["l2_kb"]
    L3_BW_GBS = ac["l3_bw_gbs"]
    DRAM_BW_GBS = ac["dram_bw_gbs"]
    ARCH_NAME = ac["name"]

set_arch("zen5")  # default until [ARCH] detected

SIMD_WIDTH   = 512
FMA_PER_CORE = 2
OPS_PER_FMA  = 2
DOUBLE_PUMP  = 1

DTYPE_BITS = {"fp32": 32, "bf16": 16, "int8": 8}

COMPUTE = {}
for dt, bits in DTYPE_BITS.items():
    COMPUTE[dt] = (SIMD_WIDTH // bits) * FMA_PER_CORE * OPS_PER_FMA * DOUBLE_PUMP

ELEM_BYTES = {"u8":1, "s8":1, "s4":0.5, "bf16":2, "f16":2, "f32":4}

ALGO_NAMES = {
    "1": "DLP-Blocked", "2": "DLP-2", "3": "DLP-Reorder",
    "10": "Native-GEMM", "11": "Native-BRGEMM",
}

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

CORES_PER_CCD = 8   # AMD Zen 5 Turin: 8 cores share one L3 CCD

def _scaled_shared_bw(nt):
    """Per-core share of L3 and DRAM BW under thread contention.
    L3 is shared by up to 8 cores per CCD; DRAM is shared system-wide."""
    l3_contention = min(nt, CORES_PER_CCD)
    dram_contention = nt
    return L3_BW_GBS / l3_contention, DRAM_BW_GBS / dram_contention

def blended_bw_l2(pf_l2_pct, pf_l3_pct, pf_dr_pct, nt=1):
    """Effective per-core BW through L2 port based on data source.
    L2 is private (no contention). L3/DRAM scale with thread count."""
    l2_bw = l2_peak_gbs()
    l3_bw, dram_bw = _scaled_shared_bw(nt)
    return (pf_l2_pct / 100.0 * l2_bw +
            pf_l3_pct / 100.0 * l3_bw +
            pf_dr_pct / 100.0 * dram_bw)

def blended_bw_l3(pf_l3_pct, pf_dr_pct, nt=1):
    """Effective per-core BW through L3 port based on data source.
    Only considers traffic that actually reaches L3 (L2 misses).
    Scales by thread contention on shared L3/DRAM."""
    l3_bw, dram_bw = _scaled_shared_bw(nt)
    l3_frac = pf_l3_pct / (pf_l3_pct + pf_dr_pct) * 100 if (pf_l3_pct + pf_dr_pct) > 0 else 100
    dr_frac = 100 - l3_frac
    return (l3_frac / 100.0 * l3_bw +
            dr_frac / 100.0 * dram_bw)

def cache_level(b_kb):
    if b_kb <= L1D_KB:   return "L1d"
    if b_kb <= L2_KB:    return "L2"
    if b_kb <= 32*1024:  return "L3"
    return "DRAM"

def parse(path):
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

# ── Single-file analysis ─────────────────────────────────────────────

def print_section(title, rows):
    peak_l2 = l2_peak_gbs()

    print(f"\n{'='*135}")
    print(f"  {title}")
    print(f"{'='*135}")

    h = (f"{'M':>5} {'K':>6} {'N':>6} {'K:N':>6}  {'DType':>16}  {'Tot(ms)':>10}  "
         f"{'GOPS':>10}  {'Peak GOPS':>10}  {'Comp%':>7}  "
         f"{'BW GB/s':>9}  {'L2 Peak':>9}  {'L2 BW%':>7}  "
         f"{'AI(F/B)':>8}  {'B(KB)':>9}")
    print(h)
    print("-" * len(h))
    for r in rows:
        pk     = peak_gops(r["cls"])
        eff    = r["gflops"] / pk * 100 if pk else 0
        l2_pct = r["bw"] / peak_l2 * 100 if peak_l2 else 0
        g = gcd(r["k"], r["n"]) or 1
        ratio = f"{r['k']//g}:{r['n']//g}"
        wt_dt = r["dt"].split(":")[1] if ":" in r["dt"] else r["dt"]
        b_kb = r["k"] * r["n"] * esz(wt_dt) / 1024
        print(f"{r['m']:>5} {r['k']:>6} {r['n']:>6} {ratio:>6}  {r['dt']:>16}  "
              f"{r['tot_ms']:>10.2f}  "
              f"{r['gflops']:>10.2f}  {pk:>10.2f}  {eff:>6.2f}%  "
              f"{r['bw']:>9.2f}  {peak_l2:>9.2f}  {l2_pct:>6.2f}%  "
              f"{r['ai']:>8.1f}  {b_kb:>9.1f}")

def print_formulas():
    peak_l2 = l2_peak_gbs()
    W = SIMD_WIDTH

    print(f"\n{'='*110}")
    print(f"  Formulas")
    print(f"{'='*110}")

    print(f"""
  1. Theoretical Peak (GOPS per core)
     ------------------------------------
     Peak = SIMD_Width/DataType_Size x Num_Cores x FMA_Per_Core x OPS_Per_FMA x DoublePump x Freq

     Constants (Zen 5):
       SIMD_Width           = {W} bits (Turin: native {W}-bit per pipe)
       Num_Cores            = 1
       FMA_Per_Core         = {FMA_PER_CORE} ({FMA_PER_CORE} independent FMA pipes)
       OPS_Per_FMA          = {OPS_PER_FMA} (1 multiply + 1 add)
       DoublePumpMultiplier = {DOUBLE_PUMP} (Turin=1 native | Genoa=0.5 double-pumped)

     INT8: {W}/8 x 1 x {FMA_PER_CORE} x {OPS_PER_FMA} x {DOUBLE_PUMP} = {W//8} x {FMA_PER_CORE*OPS_PER_FMA} = {COMPUTE['int8']} OPS/cycle x {CPU_FREQ_GHZ} GHz = {peak_gops('int8'):.2f} GOPS
     BF16: {W}/16 x 1 x {FMA_PER_CORE} x {OPS_PER_FMA} x {DOUBLE_PUMP} = {W//16} x {FMA_PER_CORE*OPS_PER_FMA} = {COMPUTE['bf16']} OPS/cycle x {CPU_FREQ_GHZ} GHz = {peak_gops('bf16'):.2f} GOPS
     FP32: {W}/32 x 1 x {FMA_PER_CORE} x {OPS_PER_FMA} x {DOUBLE_PUMP} = {W//32} x {FMA_PER_CORE*OPS_PER_FMA} = {COMPUTE['fp32']} OPS/cycle x {CPU_FREQ_GHZ} GHz = {peak_gops('fp32'):.2f} GOPS

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
                   (Zen 5: {L2_BW_B_PER_CYCLE} B/cycle fill bandwidth, 2x Zen 4's 32 B/cycle)
     WorkSet     = M*K*sizeof(src) + K*N*sizeof(wt) + M*N*sizeof(dst)
                   u8:s8:f32      -> M*K*1 + K*N*1 + M*N*4  bytes
                   bf16:bf16:bf16 -> M*K*2 + K*N*2 + M*N*2  bytes
     BW_measured = WorkSet x iters / total_time_sec / 1e9   (GB/s)
     BW_util%    = (BW_measured / L2_peak) x 100

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

def print_column_guide():
    print(f"\n{'='*110}")
    print(f"  Column Guide")
    print(f"{'='*110}")
    print("""\
  K:N     = Shape ratio (K dimension : N dimension)
  AI(F/B) = Arithmetic Intensity  (FLOPS/byte)
  B(KB)   = B-matrix (weights) size in KB  (K x N x elem_size)
  GOPS    = measured throughput = 2*M*K*N x iters / time / 1e9
  Peak GOPS = theoretical compute peak for this data type
  Comp%   = Compute Efficiency = GOPS / Peak_GOPS x 100
  BW GB/s = operational bandwidth = WorkSet x iters / time / 1e9
  L2 Peak = theoretical L2 peak bandwidth (GB/s)
  L2 BW%  = L2 Bandwidth Efficiency = BW / L2_Peak x 100
  L2Eff   = L2 effective achievable BW (GB/s) through L2 port for this shape
            = PF->L2% x L2_peak + PF->L3% x L3_BW + PF->DRAM% x DRAM_BW
            Per-shape L2 roofline: adapts to where data actually lives.
            L2-resident: ~264 GB/s | L3: ~120 GB/s | DRAM: ~50 GB/s
  L2Meas  = L2 measured BW (GB/s) = L1_misses x 64B / time / threads
            Actual data throughput rate through the L2->L1 port.
  L2%Eff  = L2Meas / L2Eff x 100  — kernel efficiency vs L2 port roofline
            >80%: near-optimal | 50-80%: room to improve | <50%: overhead-limited
""")

def single_file_analysis(path):
    rows = parse(path)
    if not rows:
        sys.exit("No data rows parsed.")

    int8_gemm = [r for r in rows if r["cls"] == "int8" and r["m"] > 1]
    bf16_gemm = [r for r in rows if r["cls"] == "bf16" and r["m"] > 1]
    int8_gemv = [r for r in rows if r["cls"] == "int8" and r["m"] == 1]
    bf16_gemv = [r for r in rows if r["cls"] == "bf16" and r["m"] == 1]

    peak_l2 = l2_peak_gbs()

    print(f"\n{'='*110}")
    print(f"  AMD EPYC 9B45 (Zen 5 / Turin) - Single Core Analysis")
    print(f"  CPU freq: {CPU_FREQ_GHZ} GHz (max boost) | L1d: {L1D_KB} KB | L2: {L2_KB} KB (16-way)")
    print(f"  NOTE: Actual freq under sustained AVX-512 may be ~3-5% lower than max boost.")
    print(f"{'='*110}")
    print(f"  Compute peaks/core:")
    for label, key in [("FP32","fp32"), ("BF16","bf16"), ("INT8 VNNI","int8")]:
        print(f"    {label:>10}: {peak_gops(key):>8.2f} GOPS  ({COMPUTE[key]} OPS/cycle)")
    print(f"  L2 BW/core    : {peak_l2:.2f} GB/s  ({L2_BW_B_PER_CYCLE} B/cycle)")

    print_formulas()

    if int8_gemm: print_section("INT8 GEMM  (u8:s8:f32)", int8_gemm)
    if bf16_gemm: print_section("BF16 GEMM  (bf16:bf16:bf16)", bf16_gemm)
    if int8_gemv: print_section("INT8 GEMV  (M=1, u8:s8:f32)", int8_gemv)
    if bf16_gemv: print_section("BF16 GEMV  (M=1, bf16:bf16:bf16)", bf16_gemv)

    print_column_guide()

# ── Multi-algo comparison ────────────────────────────────────────────

def compare_algos(files):
    peak_l2 = l2_peak_gbs()

    algo_data = {}
    for fpath in files:
        algo_id = extract_algo(fpath)
        rows = parse(fpath)
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

    print(f"\n{'='*160}")
    print(f"  BF16 GEMV - Multi-Algo L2 Bandwidth Comparison (Single Core, AMD EPYC 9B45)")
    print(f"  L2 Peak BW = {peak_l2:.2f} GB/s  ({L2_BW_B_PER_CYCLE} B/cycle x {CPU_FREQ_GHZ} GHz)")
    print(f"{'='*160}")

    algo_headers = ""
    for aid in algo_ids:
        name = ALGO_NAMES.get(aid, f"Algo{aid}")
        algo_headers += f"  {'Tot(ms)':>9} {'BW%':>7} {'GB/s':>7}"
    col_w = 9 + 1 + 7 + 1 + 7 + 2

    print()
    h = f"{'K':>6} {'N':>6} {'K:N':>8} {'Cache':>5} {'B(KB)':>8}"
    for aid in algo_ids:
        name = ALGO_NAMES.get(aid, f"A{aid}")
        label = f"--- {name} ---"
        h += f"  {label:>{col_w}}"
    h += f"  {'Best':>12}"
    print(h)

    sub = f"{'':>6} {'':>6} {'':>8} {'':>5} {'':>8}"
    for aid in algo_ids:
        sub += f"  {'Tot(ms)':>9} {'BW%':>7} {'GB/s':>7}"
    sub += f"  {'':>12}"
    print(sub)
    print("-" * len(h))

    for key in all_keys:
        m, k, n, dt = key
        if m != 1:
            continue

        g = gcd(k, n) or 1
        ratio = f"{k//g}:{n//g}"
        wt_dt = dt.split(":")[1] if ":" in dt else dt
        b_kb = k * n * esz(wt_dt) / 1024
        clvl = cache_level(b_kb)

        line = f"{k:>6} {n:>6} {ratio:>8} {clvl:>5} {b_kb:>8.1f}"

        best_bw_pct = -1
        best_algo = ""
        for aid in algo_ids:
            r = algo_data[aid].get(key)
            if r:
                l2_pct = r["bw"] / peak_l2 * 100
                line += f"  {r['tot_ms']:>9.2f} {l2_pct:>6.1f}% {r['bw']:>7.1f}"
                if l2_pct > best_bw_pct:
                    best_bw_pct = l2_pct
                    best_algo = ALGO_NAMES.get(aid, f"A{aid}")
            else:
                line += f"  {'---':>9} {'---':>7} {'---':>7}"

        line += f"  {best_algo:>12}"
        print(line)

    # ── Summary: average L2 BW% per algo per cache level ──────────
    print(f"\n{'='*100}")
    print(f"  Summary: Average L2 BW% by Cache Level")
    print(f"{'='*100}")

    levels = ["L1d", "L2", "L3"]
    h2 = f"{'Cache Level':>12} {'Count':>6}"
    for aid in algo_ids:
        name = ALGO_NAMES.get(aid, f"A{aid}")
        h2 += f"  {name:>12}"
    h2 += f"  {'Best Algo':>12}"
    print(h2)
    print("-" * len(h2))

    for lvl in levels:
        sums = {aid: [] for aid in algo_ids}
        cnt = 0
        for key in all_keys:
            m, k, n, dt = key
            if m != 1: continue
            wt_dt = dt.split(":")[1] if ":" in dt else dt
            b_kb = k * n * esz(wt_dt) / 1024
            if cache_level(b_kb) != lvl: continue
            cnt += 1
            for aid in algo_ids:
                r = algo_data[aid].get(key)
                if r:
                    sums[aid].append(r["bw"] / peak_l2 * 100)

        line = f"{lvl:>12} {cnt:>6}"
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

    # ── Peak BW% per algo ──────────
    print(f"\n{'='*100}")
    print(f"  Peak L2 BW% per Algo (best single shape)")
    print(f"{'='*100}")
    for aid in algo_ids:
        name = ALGO_NAMES.get(aid, f"A{aid}")
        best_r = None
        best_pct = 0
        for key, r in algo_data[aid].items():
            pct = r["bw"] / peak_l2 * 100
            if pct > best_pct:
                best_pct = pct
                best_r = r
        if best_r:
            print(f"  {name:<15}: {best_pct:>6.1f}% L2 BW  ({best_r['bw']:.1f} GB/s)  "
                  f"at K={best_r['k']}, N={best_r['n']}")

    # ── Win count ──────────
    print(f"\n{'='*100}")
    print(f"  Win Count (which algo has highest L2 BW% per shape)")
    print(f"{'='*100}")
    wins = {aid: 0 for aid in algo_ids}
    total = 0
    for key in all_keys:
        m, k, n, dt = key
        if m != 1: continue
        total += 1
        best_pct = -1
        best_aid = ""
        for aid in algo_ids:
            r = algo_data[aid].get(key)
            if r:
                pct = r["bw"] / peak_l2 * 100
                if pct > best_pct:
                    best_pct = pct
                    best_aid = aid
        if best_aid:
            wins[best_aid] += 1

    for aid in algo_ids:
        name = ALGO_NAMES.get(aid, f"A{aid}")
        pct = wins[aid] / total * 100 if total else 0
        print(f"  {name:<15}: {wins[aid]:>3}/{total} shapes  ({pct:.0f}%)")

# ── Per-shape perf stat analysis ──────────────────────────────────────

def parse_perf_raw(path):
    """Parse perf counter data from either:
    1. External format (_perf_raw.txt from -p): delimited by '=== SHAPE N/M ==='
    2. Internal format (from -P / --perf-counters): inline CSV + raw counters
    """
    with open(path) as f:
        content = f.read()

    # Detect format: external has '=== SHAPE' delimiters, internal doesn't
    if '=== SHAPE' in content:
        blocks = re.split(r'=== SHAPE \d+/\d+ ===', content)
    else:
        # Internal format: split into blocks at each benchdnn CSV line
        # A CSV line has 19+ comma-separated fields starting with a digit
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

            # Parse INPUT line for M,K,N (external format)
            if line.startswith("INPUT:"):
                fields = [x.strip() for x in line[6:].split(',')]
                if len(fields) >= 5:
                    try:
                        m, k, n, iters = int(fields[0]), int(fields[1]), int(fields[2]), int(fields[3])
                        dt = fields[4]
                    except ValueError:
                        pass
                continue

            # Parse benchdnn CSV line (has 19+ comma fields with total_ms at end)
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

            # Skip [PERF] summary line (derived values, we recompute from raw)
            if line.startswith("[PERF]"):
                continue

            # [ARCH] line: auto-detect architecture from benchdnn output.
            # Profile is stored for future use but current analysis always
            # computes cache-profile metrics. TLB/stalls event names are
            # parsed into counters[] for raw display but profile-specific
            # derived output is not yet implemented.
            if line.startswith("[ARCH]"):
                arch_m = re.search(r'(Zen\d)', line, re.IGNORECASE)
                if arch_m:
                    akey = arch_m.group(1).lower()
                    if akey in ARCH_CONSTANTS:
                        set_arch(akey)
                continue

            # Parse perf stat counter lines: "117,953,177      L1-dcache-loads ..."
            ctr_match = re.match(r'([\d,]+)\s+(\S+)', line)
            if not ctr_match:
                continue
            try:
                val = int(ctr_match.group(1).replace(',', ''))
            except ValueError:
                continue
            ename = ctr_match.group(2)
            # ── cache profile events ──
            if ename == 'L1-dcache-loads':
                counters['l1_ld'] = val
            elif ename == 'L1-dcache-load-misses':
                counters['l1_miss'] = val
            elif ename == 'rFF70':
                counters['l2pf_hit'] = val
            elif ename == 'rFF71':
                counters['l2pf_l3hit'] = val
            elif ename == 'rFF72':
                counters['l2pf_dram'] = val
            elif ename in ('r0764', 'r0F64', 'rF064'):
                counters['l2_hit'] = val
            elif ename in ('r3864', 'r1064', 'r0864'):
                counters['l2_miss'] = val
            # ── tlb profile events ──
            elif ename == 'r0F45':
                counters['dtlb_l2_hit'] = val
            elif ename == 'rF045':
                counters['dtlb_l2_miss'] = val
            # ── stalls + tlb: IPC events ──
            elif ename == 'r00C0':
                counters['ret_insn'] = val
            elif ename == 'r0076':
                counters['cycles'] = val
            # ── stalls profile events ──
            elif ename == 'r20AE':
                counters['fp_reg_stall'] = val
            elif ename == 'r40AE':
                counters['fp_sched_stall'] = val
            elif ename == 'r02AE':
                counters['lq_stall'] = val
            elif ename == 'r20AF':
                counters['retire_stall'] = val

        if m == 0 or k == 0 or n == 0:
            continue

        # Compute derived metrics
        ws = ws_bytes(m, k, n, dt)
        avg_ms = tot_ms / iters if iters > 0 else 0
        bw = (ws * iters / (tot_ms / 1000)) / 1e9 if tot_ms > 0 else 0
        ai = (2.0 * m * k * n) / ws if ws > 0 else 0
        cls = classify(dt)

        l1_ld = counters.get('l1_ld', 0)
        l1_miss = counters.get('l1_miss', 0)
        l2pf_hit = counters.get('l2pf_hit', 0)
        l2pf_l3hit = counters.get('l2pf_l3hit', 0)
        l2pf_dram = counters.get('l2pf_dram', 0)
        l2_hit = counters.get('l2_hit', 0)
        l2_miss = counters.get('l2_miss', 0)

        # L1 miss %
        l1m_pct = 100.0 * l1_miss / l1_ld if l1_ld > 0 else 0

        # L2 miss % (demand + prefetch combined)
        l2_all_hit = l2_hit + l2pf_hit
        l2_all_miss = l2_miss + l2pf_l3hit + l2pf_dram
        l2_all = l2_all_hit + l2_all_miss
        l2m_pct = 100.0 * l2_all_miss / l2_all if l2_all > 0 else 0

        # L3 miss % (of those that went to L3, how many missed)
        # L3 accesses = l2_miss (demand L2 miss → L3) + l2pf_l3hit + l2pf_dram
        l3_accesses = l2_miss + l2pf_l3hit + l2pf_dram
        l3_misses = l2pf_dram
        l3m_pct = 100.0 * l3_misses / l3_accesses if l3_accesses > 0 else 0

        # Prefetch source breakdown
        pf_total = l2pf_hit + l2pf_l3hit + l2pf_dram
        pf_l2 = 100.0 * l2pf_hit / pf_total if pf_total > 0 else 0
        pf_l3 = 100.0 * l2pf_l3hit / pf_total if pf_total > 0 else 0
        pf_dr = 100.0 * l2pf_dram / pf_total if pf_total > 0 else 0

        wt_dt = dt.split(":")[1] if ":" in dt else dt
        b_kb = k * n * esz(wt_dt) / 1024

        # Counter-based L2 BW: every L1 miss = 64-byte cacheline through L2
        l2_bw_measured = (l1_miss * 64) / (tot_ms / 1000) / 1e9 if tot_ms > 0 else 0

        # Counter-based L3 BW: every L2 miss = 64-byte cacheline through L3
        l3_traffic = l2_miss + l2pf_l3hit + l2pf_dram
        l3_bw_measured = (l3_traffic * 64) / (tot_ms / 1000) / 1e9 if tot_ms > 0 else 0

        rows.append(dict(
            m=m, k=k, n=n, iters=iters, dt=dt, cls=cls,
            tot_ms=tot_ms, avg_ms=avg_ms, gflops=gflops,
            ws=ws, ai=ai, bw=bw, b_kb=b_kb,
            l2_bw_meas=l2_bw_measured,
            l3_bw_meas=l3_bw_measured,
            l1m_pct=l1m_pct, l2m_pct=l2m_pct, l3m_pct=l3m_pct,
            pf_l2=pf_l2, pf_l3=pf_l3, pf_dr=pf_dr))

    return rows


def perf_analysis(path, verbose=False, bottleneck=False, num_threads=1, cache_level=None):
    rows = parse_perf_raw(path)
    if not rows:
        sys.exit("No data parsed from perf raw file.")

    peak_l2 = l2_peak_gbs()

    nt = num_threads
    mode = "Single Core" if nt == 1 else f"{nt}-Thread (per-core normalized)"

    print(f"\n{'='*110}")
    print(f"  AMD EPYC ({ARCH_NAME}) - {mode} Analysis + HW Counters")
    print(f"  CPU freq: {CPU_FREQ_GHZ} GHz | L1d: {L1D_KB} KB | L2: {L2_KB} KB"
          f" | L2 BW: {L2_BW_B_PER_CYCLE} B/cycle | L3: 32 MB/CCD")
    if nt > 1:
        print(f"  Threads: {nt} | GOPS = system total | Comp%, L2BW%, L2BW%m = per-core (÷ {nt})")
    print(f"  NOTE: Actual freq under sustained AVX-512 may be ~3-5% lower than max boost.")
    print(f"{'='*110}")
    dtypes_present = sorted(set(r["cls"] for r in rows))
    peaks_str = " | ".join(f"{c.upper()}: {peak_gops(c):.2f}" for c in dtypes_present)
    print(f"  Compute peak/core: {peaks_str} GOPS  |  L2 BW peak/core: {peak_l2:.2f} GB/s")

    print_formulas()

    gemv = [r for r in rows if r["m"] == 1]
    gemm = [r for r in rows if r["m"] > 1]

    for label, subset in [("GEMV (M=1)", gemv), ("GEMM (M>1)", gemm)]:
        if not subset:
            continue

        sub_dtypes = sorted(set(r["cls"] for r in subset))
        dtype_label = "/".join(c.upper() for c in sub_dtypes)
        print(f"\n{'='*170}")
        print(f"  {dtype_label} {label}  —  Timing + L2/L3 HW Counters (perf stat counting mode)")
        print(f"{'='*170}")

        CL = cache_level.upper() if cache_level else None
        S = "│"
        if verbose:
            h = (f"  {'M':>5} {'K':>6} {'N':>6} {'K:N':>6} {'DType':>14}"
                 f" {'Tot(ms)':>9} {'GOPS':>8} {'Comp%':>7} {'BW GB/s':>8} {'L2 BW%':>7}"
                 f" {'AI':>5} {'B(KB)':>8}"
                 f"  {S} {'L1miss':>7} {'L2miss':>7} {'L3miss':>7}"
                 f" {'PF_L2':>7} {'PF_L3':>7} {'PF_DR':>7}"
                 f" {'L2BW%m':>7}")
        else:
            h = (f"  {'M':>5} {'K':>6} {'N':>6} {'DType':>14}"
                 f" {'GOPS':>8} {'Comp%':>7} {'L2 BW%':>7}"
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
            # Per-row peak: each row uses its own dtype's compute peak
            pk = peak_gops(r["cls"])
            roofline_knee = pk / peak_l2 if peak_l2 > 0 else 1.0

            # Per-core normalization: divide system-wide metrics by num_threads
            # Ratios (L1miss%, L2miss%, PF→L2, etc.) are already per-core averages
            gops_per_core = r["gflops"] / nt
            bw_per_core = r["bw"] / nt
            l2bw_meas_per_core = r["l2_bw_meas"] / nt

            eff = gops_per_core / pk * 100 if pk else 0
            l2_pct = bw_per_core / peak_l2 * 100 if peak_l2 else 0
            l2bw_m_pct = l2bw_meas_per_core / peak_l2 * 100 if peak_l2 else 0
            g = gcd(r["k"], r["n"]) or 1
            ratio = f"{r['k']//g}:{r['n']//g}"

            b_kb = r["b_kb"]

            # Bottleneck classification — AMD Zen 5 microarchitecture:
            #   L1d=48KB, L2=1MB (64B/cyc fill BW), L3=32MB/CCD (~20-30 cyc latency)
            #   GEMV AI ≈ 2.0 FLOP/byte (INT8) < roofline knee (4.0) → always memory-bound
            if b_kb < 48:
                comment = (f"Overhead-dominated: B={b_kb:.0f}KB in L1d ({L1D_KB}KB), "
                           f"kernel too small — dispatch/timing overhead > compute. "
                           f"Batch multiple queries to amortize.")
            elif b_kb <= 512 and r["pf_l2"] >= 80:
                if l2bw_m_pct >= 65:
                    comment = (f"L2 BW-bound (near-optimal): B={b_kb:.0f}KB in L2 ({L2_KB}KB), "
                               f"PF confirms residency ({r['pf_l2']:.0f}%), "
                               f"L2 port at {l2bw_m_pct:.0f}% of {peak_l2:.0f} GB/s. "
                               f"Saturating L2 BW — gains need higher AI (batch M>1).")
                else:
                    comment = (f"L2 BW-bound (scaling): B={b_kb:.0f}KB in L2, "
                               f"PF->L2={r['pf_l2']:.0f}%, L2 port at {l2bw_m_pct:.0f}%. "
                               f"BW ramping with problem size.")
            elif b_kb <= 1024 and r["l2m_pct"] <= 25:
                comment = (f"L2-L3 transition: B={b_kb:.0f}KB at L2 edge ({L2_KB}KB), "
                           f"PF->L2 dropping to {r['pf_l2']:.0f}%, L2miss={r['l2m_pct']:.0f}%. "
                           f"PF switching to L3 source. "
                           f"Tile N into <512KB strips to stay in L2.")
            elif r["pf_l3"] >= 40 and r["l2m_pct"] >= 30:
                if l2bw_m_pct >= 50:
                    comment = (f"L3 BW-bound (PF effective): B={b_kb:.0f}KB in L3, "
                               f"PF streams {r['pf_l3']:.0f}% from L3, L2miss={r['l2m_pct']:.0f}%, "
                               f"L2 port at {l2bw_m_pct:.0f}%. "
                               f"PF hides L3 latency well. "
                               f"Tile N for L2 residency to recover peak BW.")
                elif l2bw_m_pct >= 38:
                    comment = (f"L3 latency-bound: B={b_kb:.0f}KB in L3, "
                               f"PF->L3={r['pf_l3']:.0f}%, L2miss={r['l2m_pct']:.0f}%, "
                               f"L2 port at {l2bw_m_pct:.0f}%. "
                               f"L3->L2 fill latency (~20-30 cyc) limits throughput. "
                               f"Deeper SW prefetch or N-tiling into L2 would help.")
                else:
                    comment = (f"L3 BW-limited: B={b_kb:.0f}KB in L3, "
                               f"PF->L3={r['pf_l3']:.0f}%, L2 port only {l2bw_m_pct:.0f}%. "
                               f"L3 BW or cross-CCX contention. "
                               f"NUMA-aware placement or smaller working set needed.")
            elif r["pf_dr"] >= 10:
                comment = (f"DRAM BW-bound: B={b_kb:.0f}KB > L3 (32MB), "
                           f"PF->DRAM={r['pf_dr']:.0f}%. "
                           f"Limited by DRAM BW (~50-80 GB/s). "
                           f"Tile or compress weights to fit in L3.")
            elif r["ai"] >= roofline_knee:
                comment = (f"Compute-bound: AI={r['ai']:.1f} > knee ({roofline_knee:.1f}). "
                           f"Limited by {pk:.0f} GOPS peak. "
                           f"Improve ILP/vectorization.")
            else:
                comment = (f"Memory-bound: AI={r['ai']:.1f} < knee ({roofline_knee:.1f}), "
                           f"B={b_kb:.0f}KB, L2 port at {l2bw_m_pct:.0f}%.")

            # Per-shape BW roofline (only when -c is specified)
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
                        meas_bw = l2bw_meas_per_core
                        pct_eff = meas_bw / eff_bw * 100 if eff_bw > 0 else 0
                        roof_suffix = f"  {S} {eff_bw:>12.1f} {meas_bw:>12.1f} {pct_eff:>6.1f}%"

            if verbose:
                row = (f"  {r['m']:>5} {r['k']:>6} {r['n']:>6} {ratio:>6} {r['dt']:>14}"
                       f" {r['tot_ms']:>9.2f} {r['gflops']:>8.2f} {eff:>6.2f}% {bw_per_core:>8.2f} {l2_pct:>6.2f}%"
                       f" {r['ai']:>5.1f} {r['b_kb']:>8.1f}"
                       f"  {S} {r['l1m_pct']:>6.1f}% {r['l2m_pct']:>6.1f}% {r['l3m_pct']:>6.1f}%"
                       f" {r['pf_l2']:>6.1f}% {r['pf_l3']:>6.1f}% {r['pf_dr']:>6.1f}%"
                       f" {l2bw_m_pct:>6.1f}%"
                       f"{roof_suffix}")
            else:
                row = (f"  {r['m']:>5} {r['k']:>6} {r['n']:>6} {r['dt']:>14}"
                       f" {r['gflops']:>8.2f} {eff:>6.2f}% {l2_pct:>6.2f}%"
                       f"  {S} {r['l1m_pct']:>6.1f}% {r['l2m_pct']:>6.1f}% {r['l3m_pct']:>6.1f}%"
                       f" {r['pf_l2']:>6.1f}% {r['pf_l3']:>6.1f}% {r['pf_dr']:>6.1f}%"
                       f" {l2bw_m_pct:>6.1f}%"
                       f"{roof_suffix}")
            if bottleneck:
                row += f"  {S} {comment}"
            print(row)

    print_column_guide()
    peak_l2 = l2_peak_gbs()
    print(f"""
{'='*120}
  HW Counter Columns — AMD Zen 5 PMU (perf stat counting mode)
{'='*120}

  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │                     AMD EPYC 9B45 (Zen 5) Cache Hierarchy                      │
  │                                                                                 │
  │   ┌──────┐     ┌───────────┐     ┌───────────┐     ┌───────────┐     ┌──────┐  │
  │   │ DRAM │────→│ L3 (32MB) │────→│ L2 (1MB)  │────→│ L1d(48KB) │────→│ Regs │  │
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
              % of all L1 load requests that missed L1 and went to L2.
              Source: L1-dcache-loads, L1-dcache-load-misses

    L2miss  = L2 Total Miss Rate (demand + prefetch combined)
              = (DC_miss + PF_miss_L2_hit_L3 + PF_miss_L2_L3)
                / (DC_hit + DC_miss + PF_hit_L2 + PF_miss_L2_hit_L3 + PF_miss_L2_L3) × 100
              % of ALL L2 accesses (demand + prefetch) that missed L2.
              Source: PMCx064 L2CacheReqStat (demand, excludes PF):
                        rF064 = umask 0xF0 = all DC hits (read+store)
                        r0864 = umask 0x08 = DC miss
                      PMCx070-072 (prefetcher):
                        rFF70 = PF hit L2, rFF71 = PF→L3, rFF72 = PF→DRAM

    L3miss  = L3 Miss Rate (of accesses that reached L3)
              = PF_miss_L2_L3 / (DC_miss + PF_miss_L2_hit_L3 + PF_miss_L2_L3) × 100
              Approximation: uses prefetcher DRAM misses as numerator.
              Demand L2 misses also go to L3 but we lack per-demand L3 hit/miss
              split (would need PMCx044 Any_DC_Fills_by_Data_Source for that).
              For GEMV workloads this is a reasonable proxy since most B-matrix
              traffic is prefetcher-driven.
              Source: rFF72 / (r0864 + rFF71 + rFF72)

  ── L2 Prefetcher Source Breakdown (PF→L2 + PF→L3 + PF→DR ≈ 100%) ─────────────

    PF→L2   = Prefetcher found data already in L2  (PMCx070: L2PfHitL2)
              High → B matrix fits in L2; data reused across iterations.

    PF→L3   = Prefetcher fetched data from L3 into L2  (PMCx071: L2PfMissL2HitL3)
              High → B matrix in L3; prefetcher streams L3→L2 ahead of core.

    PF→DR   = Prefetcher fetched data from DRAM  (PMCx072: L2PfMissL2L3)
              High → B matrix exceeds L3 (>32 MB); DRAM bandwidth bottleneck.

  ── L2 BW and L2 BW% (HW counter-based) ────────────────────────────────────

    L2 BW is measured directly from hardware counters, not estimated:

        L2 BW (GB/s) = L1_dcache_load_misses × 64 bytes / total_time / num_threads
        L2 BW%       = L2 BW / L2_peak_per_core × 100

    For multi-thread runs, perf aggregates L1 misses across all cores.
    We divide by num_threads to get per-core average BW.
    Comp% and L2 BW% are similarly per-core normalized.

    Every L1 miss results in a 64-byte cacheline transfer through L2 to L1.
    This is the actual measured data throughput rate through the L2 port,
    regardless of where data originally resides (L2 hit, L3 fill, or DRAM fill):

        L3/DRAM ──(fill)──→ L2 ──(64B per L1 miss)──→ L1 ──→ CPU
                              │
                 L2 BW measures this transfer rate.
                 Data flows through L2 even on an L2 miss (as a fill).

    L2 peak = {peak_l2:.0f} GB/s ({L2_BW_B_PER_CYCLE} B/cycle × {CPU_FREQ_GHZ} GHz)

    L2 BW% and L2miss CAN both be high simultaneously:
        L2 BW% = 45%  → decent throughput (data flowing fast through L2 pipe)
        L2miss = 85%  → data originates from L3 (not resident in L2)
        This means L3→L2 streaming is working well at ~45% of L2 port capacity.

  ── Interpreting by B-matrix size (K × N × element_bytes) ──────────────────────

    ┌────────────┬────────┬────────┬────────┬────────┬────────┬──────────────────┐
    │  B size    │ L1miss │ L2miss │ L3miss │ PF→L2  │ PF→L3  │ Performance      │
    ├────────────┼────────┼────────┼────────┼────────┼────────┼──────────────────┤
    │ < 48 KB    │ Low    │ Low    │ Low    │ Med    │ Low    │ Overhead-limited │
    │ 48KB–1MB   │ Rising │ LOW    │ Low    │ HIGH   │ Low    │ ★ PEAK BW/GOPS  │
    │ 1MB–32MB   │ High   │ HIGH   │ Low    │ Low    │ HIGH   │ L3 streaming     │
    │ > 32 MB    │ High   │ High   │ HIGH   │ Low    │ Low    │ DRAM bottleneck  │
    └────────────┴────────┴────────┴────────┴────────┴────────┴──────────────────┘

  ── L2 BW Roofline (L2Eff / L2Meas / L2%Eff columns) ────────────────────────────

    The achievable BW through the L2 port depends on WHERE data actually lives.
    We use HW prefetcher counters (PF→L2/L3/DRAM) to compute the effective BW:

        L2Eff = PF→L2% × L2_peak + PF→L3% × L3_BW + PF→DRAM% × DRAM_BW

    Per-core BW constants (empirical, AMD EPYC 9B45):
        L2 peak  = {L2_BW_B_PER_CYCLE} B/cycle × {CPU_FREQ_GHZ} GHz = {l2_peak_gbs():.0f} GB/s  (theoretical max)
        L3 eff   ≈ {L3_BW_GBS:.0f} GB/s   (measured from PF→L3 > 85% shapes)
        DRAM eff ≈ {DRAM_BW_GBS:.0f} GB/s   (measured from PF→DRAM > 70% shapes)

    L2Meas = L1_dcache_load_misses × 64 bytes / total_time / num_threads
             Every L1 miss = one 64B cacheline fill through L2 port.

    ┌───────────────┬─────────────┬────────────────────────────────────────────┐
    │ Data Location  │ L2Eff       │ Interpretation                            │
    ├───────────────┼─────────────┼────────────────────────────────────────────┤
    │ 100% in L2     │ 264 GB/s    │ Max: L2 fill port is the bottleneck       │
    │ 100% in L3     │ 120 GB/s    │ L3→L2 fill rate limits throughput         │
    │ 100% in DRAM   │  50 GB/s    │ DRAM BW wall, nothing kernel can do       │
    │ 50% L2, 50% L3 │ 192 GB/s    │ Transition: some data spills to L3        │
    └───────────────┴─────────────┴────────────────────────────────────────────┘

    L2%Eff = L2Meas / L2Eff × 100  — kernel efficiency at utilizing L2 port BW
      >80%  → near-optimal for this memory level
      50-80% → room for improvement (prefetch depth, tiling, instruction mix)
      <50%  → significant overhead (small shapes) or pipeline stalls
""")


# ── main ─────────────────────────────────────────────────────────────

def main():
    prog = sys.argv[0]
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        sys.exit(f"""
Analyze benchdnn matmul output — AMD EPYC 9B45 (Zen 5 / Turin)

MODES:
  {prog} <benchdnn_output_file>
      Single-file analysis: parses benchdnn timing output, computes GOPS,
      Comp%, L2 BW%, AI, and categorizes by INT8/BF16 GEMM/GEMV.

  {prog} --compare <file1> <file2> ...
      Multi-algo comparison: side-by-side L2 BW% for multiple algo outputs.
      Files should be named with 'algo<N>' for auto-detection (e.g. algo1, algo11).

  {prog} --perf [flags] <perf_raw_file>
      HW counter analysis: parses per-shape perf stat output from
      run_matmul_benchmark_sweep.sh -p, merges timing + AMD Zen 5 PMU
      counters (L1/L2/L3 miss rates, prefetcher source, measured L2 BW).

FLAGS (for --perf mode):
  -v          Verbose: show all columns including Tot(ms), K:N ratio,
              Arithmetic Intensity, B(KB), calculated BW GB/s.
              Default shows compact view with key metrics only.

  -b          Bottleneck: append a per-shape bottleneck analysis column
              with detailed commentary (overhead-dominated, L2 BW-bound,
              L3 latency-bound, etc.) based on cache counter patterns.

  -t N        Number of OMP threads used during the benchmark run.
              Default: 1 (single-core). For multi-thread perf data,
              perf stat aggregates counters across all cores. This flag
              normalizes Comp%, L2 BW%, and L2BW%m to per-core averages.

  -c LEVEL    Cache level for BW roofline analysis: L2 or L3 (off by default).
              L2: roofline based on L2→L1 port BW (from L1 miss counters).
                  Shows L2Eff/L2Meas/L2%Eff. L1-resident shapes show "L1 bound".
              L3: roofline based on L3→L2 port BW (from L2 miss counters).
                  Shows L3Eff/L3Meas/L3%Eff. L2-resident shapes show "L2 bound".

  -h, --help  Show this help message.

EXAMPLES:
  # Single-file timing analysis
  {prog} build/benchmark_bf16_algo1_64c.txt

  # Compare ALGO 1 vs 11
  {prog} --compare build/benchmark_bf16_algo1_64c.txt build/benchmark_bf16_algo11_64c.txt

  # Perf analysis — single thread, compact view
  {prog} --perf build/benchmark_gemv_int8_algo1_1c_perf_raw.txt

  # Perf analysis — single thread, all columns + bottleneck
  {prog} --perf -v -b build/benchmark_gemv_int8_algo1_1c_perf_raw.txt

  # Perf analysis — 64 threads, per-core normalized
  {prog} --perf -t 64 -b build/benchmark_bf16_algo11_64c_perf_raw.txt

WORKFLOW:
  1. Run benchmark:
     bash scripts/run_matmul_benchmark_sweep.sh -t 1 -i bf16 1

  2. Run benchmark + HW counters:
     sudo bash scripts/run_matmul_benchmark_sweep.sh -t 1 -p -i bf16 1

  3. Analyze:
     {prog} --perf -b build/..._perf_raw.txt
""")

    if sys.argv[1] == "--compare":
        if len(sys.argv) < 3:
            sys.exit("--compare requires at least one file")
        compare_algos(sys.argv[2:])
    elif sys.argv[1] == "--perf":
        args = sys.argv[2:]
        verbose = False
        bneck = False
        nt = 1
        clevel = None
        filepath = None
        i = 0
        while i < len(args):
            if args[i] == "-v":
                verbose = True
            elif args[i] == "-b":
                bneck = True
            elif args[i] == "-t":
                i += 1
                if i >= len(args):
                    sys.exit("-t requires a number")
                nt = int(args[i])
            elif args[i] == "-c":
                i += 1
                if i >= len(args):
                    sys.exit("-c requires L2 or L3")
                clevel = args[i].upper()
                if clevel not in ("L2", "L3"):
                    sys.exit(f"-c must be L2 or L3, got: {args[i]}")
            elif args[i] == "--arch":
                i += 1
                if i >= len(args):
                    sys.exit("--arch requires zen4 or zen5")
                akey = args[i].lower()
                if akey not in ARCH_CONSTANTS:
                    sys.exit(f"--arch must be zen4 or zen5, got: {args[i]}")
                set_arch(akey)
            else:
                filepath = args[i]
            i += 1
        if not filepath:
            sys.exit("--perf requires a perf_raw file")
        perf_analysis(filepath, verbose=verbose, bottleneck=bneck,
                      num_threads=nt, cache_level=clevel)
    else:
        single_file_analysis(sys.argv[1])

if __name__ == "__main__":
    main()
