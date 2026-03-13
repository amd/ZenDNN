# BenchDNN Performance Counter Support

## Overview

BenchDNN supports **in-process hardware performance counter collection** for matmul
operations using the Linux `perf_event_open()` API. This provides per-shape cache
behavior analysis without external profiling tools, measuring **only the matmul
iterations** (excluding warmup, tensor init, and process startup overhead).

**Supported architectures:** AMD Zen 4 (EPYC 9004 / Family 19h) and Zen 5 (EPYC 9005 / Family 1Ah).
Architecture is auto-detected at runtime via CPUID; constants (L2 bandwidth,
cache sizes) are selected accordingly to ensure uniform, comparable metrics.

## Counter Profiles

Three profiles are available, each optimized for a different analysis scenario.
The `tlb` and `stalls` profiles use 6 events, fitting within the AMD 6-PMC
hardware limit. The `cache` profile uses 7 events (2 HW_CACHE + 5 raw); the
kernel may time-share one PMC slot, but ratio metrics remain accurate since
all events experience the same proportional sampling.

| Profile | Events | Primary Use Case |
|:--------|:-------|:-----------------|
| **cache** (default) | L1 loads/misses, L2 demand hit/miss, L2 PF→L2/L3/DRAM | Cache hierarchy analysis, L2 BW utilization |
| **tlb** | L1 loads/misses, DTLB L2 hit/miss, retired insn, cycles | TLB pressure, IPC, page walk rate |
| **stalls** | Retired insn, cycles, FP reg/sched stalls, LQ stall, retire stall | Backend bottleneck identification |

## Execution Flow

```
  FOR EACH SHAPE in input file:
  ┌───────────────────────────────────────────────────────────────┐
  │ Phase 1: Setup (NOT MEASURED)                                 │
  │   Parse M,K,N,dtype → Allocate tensors                        │
  │   detect_zen_arch() → select ArchConstants (Zen4 or Zen5)     │
  │   Open perf fds for selected profile (disabled)               │
  ├───────────────────────────────────────────────────────────────┤
  │ Phase 2: Warmup (NOT MEASURED, counters disabled)             │
  │   for j = 1..warmup_iters: matmul_direct(...)                 │
  │   OMP threads created here inherit the perf fds               │
  ├───────────────────────────────────────────────────────────────┤
  │ perf_ctrs.reset() + enable()              ◄── START counting  │
  │ ╔═══════════════════════════════════════════════════════════╗  │
  │ ║ Phase 3: Measurement (COUNTERS ACTIVE)                   ║  │
  │ ║   for j = 1..iters:                                      ║  │
  │ ║     matmul_direct(M, K, N, A, B, C, ...)  ◄── measured   ║  │
  │ ║   Counters accumulate across ALL iters and ALL OMP threads║  │
  │ ╚═══════════════════════════════════════════════════════════╝  │
  │ perf_ctrs.disable() + read()              ◄── STOP counting  │
  ├───────────────────────────────────────────────────────────────┤
  │ Phase 4: Output                                               │
  │   Print timing (GOPS) + [PERF] derived + [ARCH] + raw ctrs   │
  └───────────────────────────────────────────────────────────────┘
```

## Architecture Support

### Auto-Detection

The CPU family is detected at runtime via CPUID (leaf 1):

| CPU Family | Architecture | L1d | L2 | L3/CCD | L2 Fill BW | Dispatch |
|:-----------|:-------------|:----|:---|:-------|:-----------|:---------|
| 0x19 (25) | Zen 4 | 32 KB | 1 MB | 32 MB | **32 B/cycle** | 6-wide |
| 0x1A (26) | Zen 5 | 48 KB | 1 MB | 32 MB | **64 B/cycle** | 8-wide |

The key difference for performance analysis is L2 fill bandwidth: Zen 5 doubled it
from 32 to 64 bytes/cycle. All derived metrics (L2BW%m) automatically use the
correct peak bandwidth for the detected architecture.

### PMU Event Compatibility

All PMC event codes are **identical** between Zen 4 and Zen 5:

| Event | Code | Zen 4 | Zen 5 | Description |
|:------|:-----|:------|:------|:------------|
| PMCx064 | rF064 / r0864 | ✓ | ✓ | L2 demand DC hit / miss |
| PMCx070 | rFF70 | ✓ | ✓ | L2 prefetcher hit in L2 |
| PMCx071 | rFF71 | ✓ | ✓ | L2 prefetcher miss L2, hit L3 |
| PMCx072 | rFF72 | ✓ | ✓ | L2 prefetcher miss L2+L3 (DRAM) |
| PMCx045 | r0F45 / rF045 | ✓ | ✓ | DTLB L2 hit / L2 miss (page walk) |
| PMCx0C0 | r00C0 | ✓ | ✓ | Retired instructions |
| PMCx076 | r0076 | ✓ | ✓ | CPU unhalted cycles |
| PMCx0AE | r20AE / r40AE / r02AE | ✓ | ✓ | Dispatch stalls (FP reg/sched, LQ) |
| PMCx0AF | r20AF | ✓ | ✓ | Retire token stall |

Source: LIKWID `perfmon_zen4_events.txt`, AMD PPR for Family 1Ah (Zen 5).

## Derived Metrics

### Cache Profile

| Metric | Formula | Description |
|:-------|:--------|:------------|
| **L1miss%** | `min(100, L1_misses / max(L1_loads, L1_misses) × 100)` | L1 data cache miss rate (clamped; AMD HW_CACHE may undercount loads) |
| **L2miss%** | `(DC_miss + PF→L3 + PF→DRAM) / (DC_hit + DC_miss + PF→L2 + PF→L3 + PF→DRAM) × 100` | Total L2 miss rate |
| **L3miss%** | `PF→DRAM / (DC_miss + PF→L3 + PF→DRAM) × 100` | L3 miss rate (prefetch proxy) |
| **PF→L2%** | `rFF70 / (rFF70+rFF71+rFF72) × 100` | Prefetcher found data in L2 |
| **PF→L3%** | `rFF71 / (rFF70+rFF71+rFF72) × 100` | Prefetcher fetched from L3 |
| **PF→DRAM%** | `rFF72 / (rFF70+rFF71+rFF72) × 100` | Prefetcher fetched from DRAM |
| **L2BW%m** | `(L1_misses/threads × 64B) / elapsed / L2_peak × 100` | Measured L2 BW utilization (arch-specific peak) |

### TLB Profile

| Metric | Formula | Description |
|:-------|:--------|:------------|
| **L1miss%** | `min(100, L1_misses / max(L1_loads, L1_misses) × 100)` | L1 data cache miss rate (clamped) |
| **DTLB_hit%** | `DTLB_L2_hit / L1_loads × 100` | L1 DTLB miss rate (resolved by L2 DTLB, ~7 cycle penalty) |
| **DTLB_walk%** | `DTLB_L2_miss / L1_loads × 100` | Full page walk rate (~100+ cycles, performance killer) |
| **IPC** | `retired_insn / unhalted_cycles` | Instructions per cycle (higher = better compute utilization) |
| **L2BW%m** | Same as cache profile | Measured L2 BW utilization |

### Stalls Profile

| Metric | Formula | Description |
|:-------|:--------|:------------|
| **IPC** | `retired_insn / unhalted_cycles` | Instructions per cycle |
| **FP_reg%** | `fp_regfile_stall_cycles / total_cycles × 100` | FP register file full (too many live ZMM registers) |
| **FP_sch%** | `fp_scheduler_stall_cycles / total_cycles × 100` | FP scheduler full (execution unit back-pressure) |
| **LQ%** | `load_queue_stall_cycles / total_cycles × 100` | Load queue full (too many outstanding loads) |
| **Retire%** | `retire_token_stall_cycles / total_cycles × 100` | Retire buffer full (instruction retirement bottleneck) |

## Usage

### In-process counters (recommended)

```bash
# Cache profile (default) — L1/L2 hit-miss analysis
sudo benchdnn --op=matmul --lowoha=true --perf-counters \
    --input_file=benchdnn/input/matmul/benchmark_gemv_sweep_bf16.txt

# TLB profile — page walk analysis + IPC
sudo benchdnn --op=matmul --lowoha=true --perf-counters=tlb \
    --input_file=benchdnn/input/matmul/benchmark_gemv_sweep_bf16.txt

# Stalls profile — dispatch bottleneck analysis
sudo benchdnn --op=matmul --lowoha=true --perf-counters=stalls \
    --input_file=benchdnn/input/matmul/benchmark_gemv_sweep_bf16.txt

# Via the benchmark script
sudo bash scripts/run_matmul_benchmark_sweep.sh -a 1 -t 1 -P -i bf16          # cache (default)
sudo bash scripts/run_matmul_benchmark_sweep.sh -a 1 -t 1 -P tlb -i bf16      # TLB + IPC
sudo bash scripts/run_matmul_benchmark_sweep.sh -a 1 -t 128 -P stalls -i bf16  # dispatch stalls
```

**Output per shape (cache profile):**
```
1, 512, 512, 100, bf16:bf16:bf16, ..., 0.231933
  [PERF]   36.5%    4.0%    0.0%  100.0%    0.0%    0.0%   43.5%
  [ARCH] Zen5 (Family 0x1A) profile=cache
          57327491      L1-dcache-loads
            454808      L1-dcache-load-misses
            685744      rF064
             11599      r0864
            214062      rFF70
               407      rFF71
                 6      rFF72
```

**Output per shape (tlb profile):**
```
1, 4096, 4096, 100, bf16:bf16:bf16, ..., 4.123456
  [PERF]   42.3%   0.15%  0.002%   2.31   38.2%
  [ARCH] Zen5 (Family 0x1A) profile=tlb
         123456789      L1-dcache-loads
          52300000      L1-dcache-load-misses
            185000      r0F45
              2500      rF045
         987654321      r00C0
         426739130      r0076
```

### External perf stat (alternative, cache events only)

```bash
sudo bash scripts/run_matmul_benchmark_sweep.sh -t 1 -p -i bf16 1
```

### Analyze results

```bash
python3 scripts/analyze_benchmark.py --perf output_file.txt
python3 scripts/analyze_benchmark.py --perf -v -b output_file.txt
python3 scripts/analyze_benchmark.py --perf -t 64 -b output_file.txt
```

The analyze script auto-detects the architecture from the `[ARCH]` line and
uses the corresponding constants (L2 BW, cache sizes). The current analysis
logic computes cache-profile metrics regardless of profile; tlb and stalls
counter names are parsed but profile-specific derived output is not yet
implemented in the analysis script.

## Implementation Details

### Files

| File | Purpose |
|:-----|:--------|
| `benchdnn/utils/perf_counters.hpp` | `PerfCounterGroup`, `ZenArch`, `PerfProfile`, `ArchConstants` |
| `benchdnn/utils/perf_counters.cpp` | CPUID detection, event tables, `perf_event_open()`, derive, print |
| `benchdnn/matmul/matmul_lowoha.cpp` | Integration into matmul measurement loop |
| `benchdnn/benchdnn.cpp` | `--perf-counters[=profile]` CLI flag |
| `scripts/analyze_benchmark.py` | Offline analysis with `--perf` mode |
| `scripts/run_matmul_benchmark_sweep.sh` | `-P [profile]` flag for internal perf |

### perf_event_open() Configuration

```c
attr.type     = PERF_TYPE_RAW;      // or PERF_TYPE_HW_CACHE for L1
attr.config   = 0xFF70;             // AMD raw event (umask << 8 | event)
attr.disabled = 1;                  // Start disabled, enable explicitly
attr.exclude_kernel = 1;            // User-space only
attr.exclude_hv = 1;                // Exclude hypervisor
attr.inherit = 1;                   // Child threads (OMP) inherit counter

fd = perf_event_open(&attr, getpid(), -1, -1, 0);
```

### Multi-thread Support

With `attr.inherit=1` and `pid=getpid()`, counters automatically aggregate across
all threads in the process (including OMP worker threads). The `derive()` function
normalizes to per-core using the thread count:

```
L2BW%m = (total_L1_misses / num_threads × 64B) / elapsed_sec / L2_peak × 100
```

Where `L2_peak` is architecture-specific: Zen 4 = 32 × freq, Zen 5 = 64 × freq.

Ratio metrics (L1miss%, L2miss%, PF→L2/L3/DRAM, DTLB%, stall%) are already
per-core averages since both numerator and denominator scale equally.

### Limitations

1. **AMD Zen 4/5 only**: Other CPU families show `[ARCH] Unknown` and use
   conservative defaults (32 B/cycle L2 BW). Events may not work on non-AMD CPUs.
2. **L3miss% is approximate**: Uses prefetcher DRAM misses as proxy.
3. **Requires elevated privileges**: `sudo` or `perf_event_paranoid <= 1`.
4. **Linux only**: Guarded by `#ifdef __linux__`; non-Linux builds compile but
   counters are disabled.
5. **CPU frequency**: Zen 4 uses 3.7 GHz (generic); Zen 5 uses 4.121 GHz
   (EPYC 9B45 specific). L2BW%m accuracy depends on matching the actual
   boost frequency of the deployed SKU.
6. **6 PMC limit**: Hardware exposes 6 programmable counters. The `tlb` and
   `stalls` profiles fit entirely within this limit. The `cache` profile
   uses 7 events and may be multiplexed by the kernel — absolute counts
   can be noisy but ratios remain valid. Run multiple profiles sequentially
   for comprehensive analysis.

### Interpreting Results for Matmul Optimization

| Observation | Likely Bottleneck | Action |
|:------------|:------------------|:-------|
| L1miss% > 50%, L2miss% < 5% | Good L2 reuse, L1 capacity miss | Tile to fit L1d (32/48KB) |
| L2miss% > 30%, PF→L3% > 50% | Data spilling to L3 | Reduce B-panel blocking (NB) |
| PF→DRAM% > 20% | Data exceeds L3 | Consider CCD-aware tiling |
| DTLB_walk% > 0.01% | Frequent page walks | Use 2MB hugepages |
| FP_sch% > 20% | FP execution units saturated | Reduce K-unroll factor |
| LQ% > 20% | Too many outstanding loads | Fewer prefetch instructions |
| IPC < 1.0 | Memory-bound | Focus on cache optimization |
| IPC > 3.0 | Compute-bound | Focus on VNNI/FMA throughput |
