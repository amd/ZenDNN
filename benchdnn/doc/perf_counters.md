# BenchDNN Performance Counter Support

## Overview

BenchDNN supports **in-process hardware performance counter collection** for matmul
operations using the Linux `perf_event_open()` API. This provides per-shape cache
behavior analysis without external profiling tools, measuring **only the matmul
iterations** (excluding warmup, tensor init, and process startup overhead).

## Execution Flow

```
  FOR EACH SHAPE in input file:
  ┌───────────────────────────────────────────────────────────────┐
  │ Phase 1: Setup (NOT MEASURED)                                 │
  │   Parse M,K,N,dtype → Allocate tensors                        │
  │   Open 7 perf fds (disabled) ← before warmup so OMP inherits │
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
  │   Print timing (GOPS) + [PERF] derived + raw counters         │
  └───────────────────────────────────────────────────────────────┘

  PerfCounterGroup API:  open() → reset() → enable() → [work] → disable() → read()
    • perf_event_open() with pid=getpid(), attr.inherit=1 (covers all OMP threads)
    • attr.exclude_kernel=1 (user-space events only), non-copyable, fds auto-closed
```

## AMD Zen 5 Cache Hierarchy & PMU Events

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│              AMD EPYC 9B45 (Zen 5 / Turin) — Cache Hierarchy & PMU Events      │
│                                                                                 │
│   ┌──────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│   │   DRAM   │───→│  L3 (32MB)  │───→│  L2 (1MB)   │───→│  L1d (48KB) │───→ CPU │
│   │ ~50-80   │    │  per CCD    │    │  per core   │    │  per core   │  Regs   │
│   │ GB/s     │    │  shared by  │    │  64 B/cyc   │    │             │         │
│   │          │    │  8 cores    │    │  fill BW    │    │             │         │
│   └──────────┘    └─────────────┘    └──────┬──────┘    └──────┬──────┘         │
│                                             │                  │                │
│                                        L2 Prefetcher      L1 Prefetcher         │
│                                        (HW, tracked)      (HW, not tracked)     │
│                                                                                 │
│   The L2 prefetcher is part of the core's L2 cache controller.                  │
│   It detects streaming access patterns and proactively issues                   │
│   requests to bring data into L2 BEFORE the core demands it.                    │
│                                                                                 │
│  ═══════════════════════════════════════════════════════════════════════════════  │
│   PMU Events Collected (7 counters)                                             │
│  ═══════════════════════════════════════════════════════════════════════════════  │
│                                                                                 │
│   ┌─ L1 Data Cache ─────────────────────────────────────────────────────────┐   │
│   │  L1-dcache-loads         PERF_TYPE_HW_CACHE   All L1 DC read accesses   │   │
│   │  L1-dcache-load-misses   PERF_TYPE_HW_CACHE   L1 DC read misses        │   │
│   │  → L1miss% = misses / loads × 100                                       │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│   ┌─ L2 Demand Requests (PMCx064 L2CacheReqStat) ──────────────────────────┐   │
│   │  Core requests (L1 misses → L2). Excludes L2 Prefetcher requests.       │   │
│   │  rF064 (umask 0xF0)   Bits 4-7: All DC Hits in L2 (read + store)       │   │
│   │  r0864 (umask 0x08)   Bit 3: DC Miss in L2                             │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│   ┌─ L2 Prefetcher Requests (PMCx070-072) ──────────────────────────────────┐   │
│   │  Speculative requests by HW prefetcher. WHERE did it find the data?      │   │
│   │  rFF70 (PMCx070)  PF hit in L2                              → PF→L2     │   │
│   │  rFF71 (PMCx071)  PF miss L2, hit L3                       → PF→L3     │   │
│   │  rFF72 (PMCx072)  PF miss L2 AND L3                        → PF→DRAM   │   │
│   │  PF→L2 + PF→L3 + PF→DRAM ≈ 100%  (mutually exclusive outcomes)         │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ═══════════════════════════════════════════════════════════════════════════════  │
│   Hardware PMC Note                                                             │
│  ═══════════════════════════════════════════════════════════════════════════════  │
│   AMD Zen 5 provides 6 programmable Core PMCs per core. We collect 7 events:    │
│   2 via PERF_TYPE_HW_CACHE (L1) + 5 via PERF_TYPE_RAW (L2 demand + PF).        │
│   The Linux kernel multiplexes when more events than PMCs are requested,        │
│   time-sharing one counter slot. Raw read() values are not auto-scaled;         │
│   however, ratios (miss rates, PF breakdown) remain accurate since all          │
│   events experience the same proportional undercounting.                         │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Derived Metrics

| Metric | Formula | Description |
|:-------|:--------|:------------|
| **L1miss%** | `L1_misses / L1_loads × 100` | L1 data cache miss rate |
| **L2miss%** | `(DC_miss + PF→L3 + PF→DRAM) / (DC_hit + DC_miss + PF→L2 + PF→L3 + PF→DRAM) × 100` | Total L2 miss rate (demand + prefetch combined) |
| **L3miss%** | `PF→DRAM / (DC_miss + PF→L3 + PF→DRAM) × 100` | L3 miss rate (approx, prefetch-based proxy) |
| **PF→L2%** | `rFF70 / (rFF70 + rFF71 + rFF72) × 100` | Prefetcher found data in L2 |
| **PF→L3%** | `rFF71 / (rFF70 + rFF71 + rFF72) × 100` | Prefetcher fetched from L3 |
| **PF→DRAM%** | `rFF72 / (rFF70 + rFF71 + rFF72) × 100` | Prefetcher fetched from DRAM |
| **L2BW%m** | `(L1_misses / threads × 64B) / elapsed_sec / L2_peak × 100` | Measured L2 bandwidth utilization (per-core) |

**Key relationships:**
- PMCx064 (demand) and PMCx070-072 (prefetch) are **non-overlapping** — AMD PPR confirms
  "*L2 Cache Request Outcomes (not including L2 Prefetch)*" for PMCx064.
- L2BW%m measures data throughput **through** L2, not L2 hit rate. Data flows through L2
  even on a miss (as an L3/DRAM fill). High L2miss% + high L2BW%m means effective L3 streaming.

## Usage

### In-process counters (recommended)

```bash
# Single-thread, INT8 GEMV
sudo benchdnn --op=matmul --lowoha=true --perf-counters \
    --input_file=benchdnn/input/matmul/benchmark_gemv_int8_sweep.txt

# Via the benchmark script
sudo bash scripts/run_matmul_benchmark_sweep.sh -a 1 -t 1 -P -i bf16
```

**Output per shape:**
```
1, 512, 512, 100, u8:s8:f32, ..., 0.231933
  [PERF]   36.5%    4.0%    0.0%  100.0%    0.0%    0.0%   43.5%
          57327491      L1-dcache-loads
            454808      L1-dcache-load-misses
            685744      rF064
             11599      r0864
            214062      rFF70
               407      rFF71
                 6      rFF72
```

### External perf stat (alternative)

```bash
# Each shape run as separate process under perf stat
sudo bash scripts/run_matmul_benchmark_sweep.sh -t 1 -p -i bf16 1
```

### Analyze results

```bash
# Compact view
python3 scripts/analyze_benchmark.py --perf output_file.txt

# All columns + bottleneck analysis
python3 scripts/analyze_benchmark.py --perf -v -b output_file.txt

# Multi-thread (per-core normalized)
python3 scripts/analyze_benchmark.py --perf -t 64 -b output_file.txt
```

The analyze script auto-detects both internal (`-P`) and external (`-p`) output formats.

## Implementation Details

### Files

| File | Purpose |
|:-----|:--------|
| `benchdnn/utils/perf_counters.hpp` | `PerfCounterGroup` class API |
| `benchdnn/utils/perf_counters.cpp` | Linux `perf_event_open()` wrapper, AMD Zen 5 event defs |
| `benchdnn/matmul/matmul_lowoha.cpp` | Integration into matmul measurement loop |
| `benchdnn/benchdnn.cpp` | `--perf-counters` CLI flag |
| `scripts/analyze_benchmark.py` | Offline analysis with `--perf` mode |
| `scripts/run_matmul_benchmark_sweep.sh` | `-P` flag for internal perf, `-p` for external |

### perf_event_open() Configuration

```c
attr.type     = PERF_TYPE_RAW;      // or PERF_TYPE_HW_CACHE for L1
attr.config   = 0xFF70;             // AMD raw event (umask << 8 | event)
attr.disabled = 1;                  // Start disabled, enable explicitly
attr.exclude_kernel = 1;            // User-space only
attr.exclude_hv = 1;                // Exclude hypervisor
attr.inherit = 1;                   // Child threads (OMP) inherit counter

fd = perf_event_open(&attr, getpid(), -1, -1, 0);
//                    pid=process  cpu=any  no_group  no_flags
```

### Multi-thread Support

With `attr.inherit=1` and `pid=getpid()`, counters automatically aggregate across
all threads in the process (including OMP worker threads). The `derive()` function
normalizes to per-core using the thread count:

```
L2BW%m = (total_L1_misses / num_threads × 64B) / elapsed_sec / L2_peak × 100
```

Ratio metrics (L1miss%, L2miss%, PF→L2/L3/DRAM) are already per-core averages
since both numerator and denominator scale equally with thread count.

### Limitations

1. **AMD Zen 5 specific**: Raw event codes (PMCx064, PMCx070-072) are AMD Family 1Ah.
   Other CPU families would need different event definitions.
2. **L3miss% is approximate**: Uses prefetcher DRAM misses as proxy. Demand L2 misses
   also go to L3 but we lack per-demand L3 hit/miss split (would need PMCx044).
3. **Requires elevated privileges**: `sudo` or `perf_event_paranoid <= 1`.
4. **Linux only**: Guarded by `#ifdef __linux__`; non-Linux builds compile but
   counters are disabled.
