# Op performance utility

Measures the compute and memory ceilings of the machine a run is on, and gives every
call of an op the best time it could take there (its *bound*), the resource that
bounds it (compute, or the L2, L3 or DRAM bandwidth) and its *efficiency* (bound /
measured time). The native BRGEMM tuner (`tools/native_brgemm_tuning`) uses it to spend
its search where the headroom is and to calibrate the planner's cost model on the
machine it tunes; benchdnn results can be rated against it directly.

Python 3.8 or newer, standard library only. The probe is C++ (GCC or Clang on Linux)
built against a ZenDNN build.

## Layout

| Path | Role |
|---|---|
| `common/ceilings.py` | The ceilings file (`op-perf-ceilings`, version 1): validation, lookups of compute and bandwidth by type, memory level and thread count, and the match against a tuning campaign's machine. |
| `common/report.py` | Rows as an aligned table, CSV or JSON. |
| `probe/probe.cpp`, `probe/build.sh` | The probe, built against a ZenDNN build, run on the target machine. |
| `matmul/` | The matmul work model (`work.py`), bounds (`bounds.py`), benchdnn results (`benchdnn_ingest.py`) and the native BRGEMM cost-model inputs (`calibrate.py`). |
| `cli.py` | `python -m op_perf_utility {show, bound, report, calibrate}` (from `tools/`). |
| `tests/` | Host unit tests on a synthetic machine (`tests/fixtures/example_ceilings.json`). |

## Measuring the ceilings

    bash tools/op_perf_utility/probe/build.sh <worktree or build dir> <out>/probe
    OMP_NUM_THREADS=32 OMP_PROC_BIND=close OMP_PLACES=cores <out>/probe --out ceilings.json
    cd tools && python3 -m op_perf_utility show ceilings.json

Run the probe the way the measured work runs: on the same machine, with the same
cpuset, thread count and pinning (the tuner's `tune.py probe` does this for a campaign).
It takes about 40 s (`--quick`: a few seconds, for a smoke test; `--seconds`,
`--repeats` per measurement; `--threads`, `--domain-threads` to override the team and
the L3 domain). One binary runs on every AVX-512 machine: each kernel carries its ISA
as a function attribute and runs only where the CPU has it. It measures

- the sustained clock, from a chain of dependent 1-cycle adds, on one core and on
  every thread at once;
- per type, the throughput (instructions per cycle, GFLOP/s of one core and of the
  team) of the dot product the native kernels use, with enough accumulators to hide
  its latency, and the latency from a dependent chain: FP32 FMA (`vfmadd231ps`), the
  BF16 dot product (`vdpbf16ps`), the INT8 VNNI dot product (`vpdpbusd`) and, where
  the CPU has AVX-512-FP16, FP16 FMA (`vfmadd231ph`);
- the inner loops of two tiles at their widest shapes, as the native kernels issue
  them: the INT4 tile (`int4`: 6 rows of 2 vectors, each K octet's nibbles unpacked into
  two planes, two source dwords broadcast a row, two `vpdpbusd` a vector) and, where
  the CPU has no AVX-512-FP16, the F16 kernels (`f16_widen`: 12 rows of 2 vectors, each
  vector widened with `vcvtph2ps`, a broadcast and an FMA a row and vector);
- the chain sweep: on one core, each path's instructions per cycle with 4, 8, 12, 16 and
  24 independent accumulator chains (the tiles at 2 to 12 rows of 2 vectors, the widening
  loop on every machine), a few tens of milliseconds a point (5 ms samples, the run's
  repeats): a path issues at its rate only with
  about ipc x latency chains in flight, which sizes a kernel's accumulators to the machine;
- the read and the write bandwidth of 1 thread, of one L3 domain's threads and of the
  team, each streaming its own first-touched buffer sized for L1, L2, its share of its
  domain's L3, and DRAM; the writes are whole-line stores, as an output tile's, which
  read their lines first. The buffers get the pages a plain allocation gets, as the buffers of
  the calls it bounds do (neither the tuner's replay nor benchdnn advises huge pages).
  `--hugepages` advises huge pages where a first fault gets one; on fragmented memory
  the faults of a full team can still take minutes to compact it. `run.hugepages`
  records whether the buffers were advised.

The CPU and caches come from the library's own detection (`native_hw()`,
`native_plan_uarch()`), named as the tuner's schema names them, and the cores of the
team from sysfs: two threads on the siblings of one core count once. So does the
library's machine model (`uarch.dot`: per path the dot products a cycle, the latency and
the chains it implies, and where the model comes from; `uarch.mem_cycles` and
`uarch.stream_cycles`, its memory terms, 0 where not set; `uarch.priced`, whether its
plans price kernels at the model's rates, which takes a model `ZENDNNL_MATMUL_NATIVE_UARCH`
sets, or else at 2 a cycle on every path): `show` prints it next to the measurement, and
the tuner flags a path whose price is off by more than 20%.

The library holds no machine's figures: without `ZENDNNL_MATMUL_NATIVE_UARCH` every CPU
takes the class default, the prices of a planner without a model. A machine's model comes
from this probe run on it, through `calibrate --format uarch` (below) or a tuning file
`tune.py calibrate` exports, and stays in its run directory.

### The ceilings file

| Block | Contents |
|---|---|
| `cpu` | vendor, family, model, stepping, the ISA features the tuning files match on, `avx512fp16`, cache sizes, logical CPUs, SMT, physical cores, threads per L3 domain |
| `uarch` | the planner's view: L1d, L2 and L3-domain bytes, cores per L3 domain, FMA ports, vector width, and (probe 1.2) the compute model per path `dot` with its `dot_source`, the memory terms `mem_cycles` and `stream_cycles`, and `priced` |
| `run` | threads, threads per L3 domain, the process's cpuset, the team's CPUs and cores, `OMP_PROC_BIND`, `OMP_PLACES`, repeats, seconds per measurement, whether the buffers were advised huge pages |
| `clock` | GHz on one core and on the team, with their spread |
| `compute` | per path (`fp32`, `bf16`, `int8`, `fp16`, `int4`, `f16_widen`): FLOPs per instruction, instructions per cycle (one core, team), latency, GFLOP/s (one core, team), spread |
| `chain_sweep` | (probe 1.2) the chain counts, and per path one core's instructions per cycle at each |
| `bandwidth` | per level (`l1`, `l2`, `l3`, `dram`) and thread count: bytes per thread, GB/s, spread |
| `write_bandwidth` | the same for stores |

Between the measured thread counts a bandwidth is interpolated; above the last, L1 and
L2 scale per core, L3 per domain and DRAM stays at its last value. A team whose threads
share cores (`run.team_cores` below its threads) takes that share at any thread count.
Probe 1.0 files have no `team_cores`, `int4`, `f16_widen` or `write_bandwidth`: their
teams take a core a thread, s4 and f16 calls the INT8 and FP32 dot products, and stores
are priced as reads.

## Matmul bounds

    python3 -m op_perf_utility bound  --ceilings C --shape m=16,n=4096,k=4096,wei=s8 [--cold] [--instances N] [--measured-ms T]
    python3 -m op_perf_utility report --ceilings C --benchdnn out.csv [--benchdnn out.txt] [--cold] [--instances N] [--format csv]

A shape is `m=..,n=..,k=..` with optional `batch`, `src`, `wei`, `dst` (`f32`, `bf16`,
`f16`, `s8`, `u8`, `s4`, `u4`), `bias`, `wei_group`, `wei_scale_bytes`,
`wei_scale_tensor`, `src_group`, `src_scale_bytes`, `src_scale_tensor`, `gated`,
`binary` (binary post-ops), `binary_dt`, `per_batch_operands` and `threads`.
`report` reads benchdnn's matmul CSV (its `Summary` rows skipped) or its printed result
lines (a batched call's batch size, the fields of a matmul input line, then the total):
types, bias, post-ops, the granularity (per tensor, channel or group) and types of the
weight and source scales, and source quantization. A call's time is its total over the
iterations, since both printed times have two decimals.

- **Work:** FLOPs = 2 M N K x batch. The compulsory bytes move each operand once: the
  weights and their scales (an s8 weight one scale per column, per group of K rows or
  per tensor, an s4 weight one per group of K rows), the source and its scales (per
  token, per group of K columns or per tensor) when the call quantizes it or an integer
  source has static scales, the output (N / 2 wide when gated), the bias (N) and the
  tensor each binary post-op reads (M x N: a gated call's applies before the activation
  halves it). The bias and the binary operands are shared by a batch, as BenchDNN gives
  them; `per_batch_operands` gives each batch item its own. A source the call quantizes
  is also written as u8 and read back by the tiles; the quantizing arithmetic is not
  counted.
- **Memory level:** DRAM for cold weights (as the tuner's replay runs time them).
  Otherwise L2 when the call's buffers, each counted once (a quantized source's copy
  among them), fit half the team's L2, L3 when they fit 80% of the L3 of the domains
  the team's cores span, else DRAM. Reads are priced at the level's read bandwidth,
  stores (the output, a quantized source's copy) at its write bandwidth.
- **Threads:** the team's, at most one per 16 columns of one row of one batch item
  (batch x M x N / 16, rounded up). The native tiles are at least 16 columns wide, M = 1
  calls take units of 32, and no plan splits K over threads, so a smaller call leaves
  the other threads idle: its compute, bandwidth and cache capacity are those of the
  threads it uses. A report's `threads` are those, `team` the team.
- **Instances:** `--instances N` bounds one of N copies of the call running at once over
  the same memory: its DRAM bandwidth is that of N times its threads, divided by N.
  Above the largest thread count the probe measured, DRAM keeps that point's bandwidth,
  and a note says so; probe the copies' CPUs together (their cpuset, `--threads` of all
  of them) to measure the share. L2 and L3 belong to each copy's cores and domains, and
  the copies are taken to run at the probe's all-core clock.
- **Compute path:** BF16 weights the BF16 dot product; s8 weights INT8 VNNI; s4 weights
  the INT4 tile's loop (`int4`); F16 weights FP16 FMA where the CPU has AVX-512-FP16,
  else the F16 kernels' loop (`f16_widen`); F32 weights FP32 FMA. Without the tile
  loops (a probe 1.0 file), s4 takes INT8 VNNI and F16 FP32 FMA.
- **SMT:** a team with two threads on the siblings of a core takes the cores, compute,
  L1 and L2 of half as many cores.
- **Bound:** attainable time = max(FLOPs / compute ceiling, read bytes / read bandwidth
  + stored bytes / write bandwidth of the level); the resource of the larger term
  bounds the call; efficiency = attainable / measured. The no-overlap sum of the two is
  reported as well.

Near the balance of the two terms (BF16 and INT8 calls of about 128 rows on a
32-thread team) the bound assumes an overlap the kernels do not reach, so
efficiencies there read low; decode calls with cold weights and prefill calls read
close to 100% when the kernels are good.

## The native BRGEMM cost model's inputs

    python3 -m op_perf_utility calibrate --ceilings C [--threads N] [--format table|csv|json|settings]

The planner's cost model (`native_cost_model.cpp`) prices a plan from hardware inputs it
otherwise assumes. `calibrate` derives them from the ceilings, at the team the tuning
runs with:

| Key | The cost model's use | From the ceilings |
|---|---|---|
| `cm_fma_pipes` | dot-product instructions a core issues per cycle (0: the model's, per path) | the path's instructions per cycle on one core |
| `cm_fma_lat` | least cycles of a K step, whatever the kernel's accumulator sets (0: the path's latency over the sets) | none: its built-in 6 is fitted to timed plans; the latency goes into `--format uarch` |
| `cm_l2_bytes` | bytes per cycle of a slab re-read from L2 | L2 bytes per cycle per thread, the team reading |
| `cm_l3_cycles` | cycles per KiB read from L3 | L3, one domain's threads reading |
| `cm_mem_cycles` | cycles per KiB a thread streams while its domain streams | DRAM, the team streaming |
| `cm_stream_cycles` | cycles per KiB one core streams alone | DRAM, one thread |

`cm_fma_pipes` is per path (BF16, INT8, the s4 tile's int4, FP16 or the F16 kernels'
f16_widen, FP32 calls), and only for a path the probe measures more than 20% away from
what the built-in plan prices with (`uarch.dot` where `uarch.priced`, else 2 a cycle;
`tune.py calibrate` compares with its campaign's schema instead); a probe before 1.2
prints no model, so every path gets it. `cm_fma_lat` is not derived: the measured
latency as the step floor made f16 calls of 16 rows slower. `--format settings` gives a
`ZENDNNL_MATMUL_NATIVE_CONFIG` of one clause per weight type, `--format uarch` a
`ZENDNNL_MATMUL_NATIVE_UARCH` of the measured model, each path's rate and latency and,
where the probe measured DRAM, the memory terms in the library's form (`mem_cycles`,
`cm_mem_cycles` times the square root of the L3 domain's cores; `stream_cycles`),
within the ranges the library takes: the model for every call, where the library's
table lacks the machine. The values are inputs
to try: the probe streams with every prefetcher helping and nothing else running, so
`tune.py calibrate` keeps them only after an A/B against the built-in inputs. It leaves
`cm_l2_bytes` out unless told: at the L2 streaming rate it made BF16 plans slower on
every machine it was tried on (a call's slab re-reads share the core with its FMAs).
The cost model refines the rules' plan only where an alternative is modeled
`cm_gain_pct` faster, so the inputs change few plans (mostly BF16 calls of about 16
rows), and it prices no f32 or per-group s8 plan.

## In the native BRGEMM tuner

`tune.py probe` measures the ceilings where a campaign's sessions run, `status` and
`export` then give each problem its bound and the built-in plan's and the best
plan's efficiency, `search --prioritize / --skip-at-pct / --stop-at-pct` spend
the budget by headroom, and `tune.py calibrate` A/Bs the cost model's inputs. See
[the tuner's README](../native_brgemm_tuning/README.md).

## Adding an op

An op gets a folder (`group_matmul/` for MoE, `attention/`) with the matmul folder's
interface: `work.py` (the FLOPs and compulsory bytes of one call), `bounds.py` (its
memory level, compute path and bound against `common/ceilings.py`) and
`benchdnn_ingest.py` (its benchdnn results as shapes and measured times), and a `--op`
choice in `cli.py`. A grouped matmul sums over the active experts: each expert's
weights once and its routed tokens, the gated gate and up projection, the down
projection.

## Tests

`python3 -m unittest discover -s tools/op_perf_utility/tests -t tools` (host only, no
library, no probe): the ceilings file and its lookups, the work and bound arithmetic
on hand-worked cases, benchdnn parsing and the cost-model inputs, all on the synthetic
machine. CI also builds the probe against its build and reads back a short run's file.

## Nothing measured is committed

A ceilings file describes one machine at one thread count and pinning. It is written by
the probe into the run's or the campaign's directory and stays there: keep it out of the
repository. The test fixture is a synthetic machine. A tuning file written by the tuner
records only percentages, bound kinds and the planner settings it exports
(`meta.op_perf`), never a measured clock, rate or bandwidth. CI reads its probe's file
back without printing it.

The code takes a machine's identity only from what it reads at run time (the library's
`native_hw()` and `native_plan_uarch()`, the OpenMP places), and docs describe machines
by public facts and give results as percentages. `zendnn-skills/op-perf-utility/SUBSKILL.md`
spells out these rules and lists what a change to the native kernels, the cost model,
the hardware detection or BenchDNN's matmul output updates here.
