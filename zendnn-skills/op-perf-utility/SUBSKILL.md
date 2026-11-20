---
name: op-perf-utility
description: >-
  Measure a machine's compute and memory ceilings with the op performance
  utility's probe, bound matmul calls against them (attainable time, bounding
  resource, efficiency: a roofline from measured ceilings), rate BenchDNN
  results, and use the ceilings in the native BRGEMM tuner. Also lists what a
  change to the native kernels, the cost model, the hardware detection or
  BenchDNN's matmul output must update in the utility, and the rules for
  machine data. Use for a roofline, a call's bound, efficiency or headroom, or
  machine ceilings, and for any change to tools/op_perf_utility or to the code
  it follows.
---

# Op performance utility

The utility (`tools/op_perf_utility`) rates a call against the machine it runs
on. The repository carries its formulas and code; every number comes from
running its probe on the machine. Paths are relative to the ZenDNN checkout.
`tools/op_perf_utility/README.md` and the op performance parts of
`tools/native_brgemm_tuning/README.md` are authoritative: use only the
commands and flags they document.

## Machine data

These rules cover code, tests, fixtures, docs, commit messages, PR text,
review comments and CI logs.

- A machine's numbers come only from running the probe on it. Its identity
  comes only from what software reads there at run time: the library's
  `native_hw()` and `native_plan_uarch()`, CPUID, `/proc/cpuinfo`, sysfs and
  the OpenMP places. Don't type a machine's measured or specified figures
  into the repository.
- The utility stands alone: it imports, reads and refers to no other
  performance tool, and copies no machine tables, figures or code from one.
- Keep ceilings files, probe output and campaign directories in the run
  directory, out of the repository, along with any number measured on a
  lab machine. A tuning file carries only percentages, bound kinds and the
  settings it exports (`meta.op_perf`).
- Tests run on the synthetic machine
  `tools/op_perf_utility/tests/fixtures/example_ceilings.json` (vendor
  `ExampleVendor`), or on values derived from it. A test that uses the
  tuner's schema fixture reads its fields at run time and never copies them.
- The tuner's schema fixture, `tools/native_brgemm_tuning/fixtures/schema_dump.json`,
  describes an example machine whatever machine its SchemaDump came from:
  `refresh_fixtures.sh` keeps it (`tune.py schema --extract <log> --machine-of`).
- Describe machines by public facts only, such as "the 8-core-L3-domain
  machine type", and give results as percentages and counts. Leave out
  codenames, part numbers and microarchitecture figures of unreleased parts,
  host and pod names, and internal paths.
- CI reads the probe's file back without printing it.
- Before pushing, search the change's added lines and messages:

  ```bash
  git diff origin/main... | rg '^\+' | rg -i 'GB/s|GHz|B/cycle|/proj/|/scratch/|pod-[0-9]'
  git log --format=%B origin/main.. | rg -i 'GB/s|GHz|B/cycle|per cycle'
  ```

  Add the names of any unreleased parts you know to the patterns. Review every
  match.

## Measure the ceilings

1. Build the probe against a ZenDNN build:
   `bash tools/op_perf_utility/probe/build.sh <worktree or build dir> <out-binary>`.
2. Run it the way the measured work runs: the same cpuset and
   `OMP_NUM_THREADS`, with `OMP_PROC_BIND=close OMP_PLACES=cores`:
   `<probe> --out <run dir>/ceilings.json`. A full run takes about 40 s;
   `--quick` takes a second, for a smoke test. It loads every CPU of the team,
   so run it where nothing else is being timed. Pass `--hugepages` only when
   the measured work advises huge pages.
3. From `tools/`, `python3 -m op_perf_utility show <file>` summarizes the file.

A file describes one machine at one thread count and pinning. Measure again
when any of them changes.

## Bound and rate calls

From `tools/`:

- One call:
  `python3 -m op_perf_utility bound --ceilings C --op matmul --shape m=..,n=..,k=..,wei=.. [--cold] [--instances N] [--measured-ms T]`.
- BenchDNN results:
  `python3 -m op_perf_utility report --ceilings C --op matmul --benchdnn <csv or printed output> [--cold] [--instances N] [--format table|csv|json]`.
- Pass `--cold` when the weights are cold, as in the tuner's replay runs with
  `--cold-mb`. BenchDNN's are warm.
- Pass `--instances N` when N copies of the work run at once over the same
  memory, such as several serving processes on one socket. For a measured
  share rather than an assumed one, probe their CPUs together.
- A small call uses fewer threads than the team (a report's `threads` against
  `team`): at most one per 16 columns of a row.

The bound is the larger of the FLOPs over the compute ceiling and the
compulsory bytes over the bandwidth of the level the working set fits.
Efficiency is the bound over the measured time. Near 100% of a DRAM bound, a
plan has little to gain. Well below a compute bound, look at the kernel, after
checking Known limits.

## In a native BRGEMM tuner campaign

After `tune.py init` (a replay or benchdnn campaign needs `--schema`, the
SchemaDump of the build on the pods' machine):

1. `tune.py probe --workdir D --probe-bin <probe>` measures the ceilings on
   the campaign's pods, with their team and pinning. `--local` runs it on this
   host, and `--ceilings F` installs a file measured by hand. It refuses a file
   from another machine or team unless `--force`. After `--dry-run`,
   `tune.py collect` takes each pod's probe once its job wrote `probe.rc`.
2. `tune.py status --workdir D` gives each problem its bound and efficiency.
   `tune.py export` writes them to `export_report.json` and `meta.op_perf`.
3. Optional, before any search: `tune.py calibrate --workdir D` derives the
   cost model's inputs and A/Bs them against the built-in ones. The export
   applies kept values to the campaign's team (its threads) only.
4. Optional: `tune.py search ... --skip-at-pct 95 --stop-at-pct 97` (both
   off by default). Use `--prioritize` when the shapes carry calls per M (a
   model trace's rows of each M range), `--prioritize relative` when their
   calls are flat.
5. `tune.py confirm --workdir D --alone` re-times each winner with its group's
   other problems at their built-in plans.

## Keep the utility in step with the code

A change in the left column updates the right column in the same PR, with
tests:

| The change | Update |
|---|---|
| A dot-product instruction or kernel kind of the native matmul kernels (kernel registry, `kernel_kind`, a new ISA path) | a probe path in `probe/probe.cpp` (kernel, ISA attribute, FLOPs per instruction), the path names in `common/ceilings.py`, `compute_path()` in `matmul/bounds.py`, the per-path pipes and latency in `matmul/calibrate.py` |
| The narrowest tile width (`nr` values), the GEMV path's column unit, or a schedule that splits K over threads | `UNIT_COLS` and `usable_threads()` in `matmul/bounds.py` |
| The inner loop or widest tile of the INT4 or F16 kernels (`native_int4_ukernel.cpp`, `native_f16_ukernel.cpp`), or how output tiles store | `tput_int4`, `tput_f16_widen` or `write_pass` in `probe/probe.cpp` |
| A data type, scale granularity, bias, post-op or epilogue of the matmul API or BenchDNN | the FLOPs and bytes in `matmul/work.py`, the readers in `matmul/benchdnn_ingest.py` |
| BenchDNN's matmul CSV header or printed result line | `matmul/benchdnn_ingest.py` (`LINE_FIELDS`, the CSV columns) and `tests/test_ingest.py` |
| A cost-model input (`cm_*`), its units, or the calls `native_cost_model.cpp` prices | `KEYS` and the units in `matmul/calibrate.py`; `PRICED` and `DEFAULT_KEYS` in `tools/native_brgemm_tuning/calibration.py` |
| The fields of `native_hw()` or `native_plan_uarch()` | the probe's `cpu` and `uarch` blocks, validation and matching in `common/ceilings.py`, `machine_mismatch()` in `tools/native_brgemm_tuning/op_perf.py` |
| The ceilings file's layout | the format version in the probe and `common/ceilings.py`, the README's table, the fixture |
| The tuner's sessions, backends or launcher | the probe jobs of `tune.py probe` (`backends/jobs.py`) |

Paths without a folder are under `tools/op_perf_utility/`. Then run, from
`tools/`:

```bash
python3 -m unittest discover -s op_perf_utility/tests -t .
python3 -m unittest discover -s native_brgemm_tuning/tests -t .
bash op_perf_utility/probe/build.sh <build dir> /tmp/op_perf_probe
OMP_NUM_THREADS=4 /tmp/op_perf_probe --out /tmp/op_perf_ceilings.json --threads 4 --quick
python3 -m op_perf_utility show /tmp/op_perf_ceilings.json > /dev/null
```

Check `probe.cpp` with `scripts/check_format.sh --check tools/op_perf_utility/probe/probe.cpp`
(clang-format 18.1.8, or `CLANG_FORMAT` pointing at it). Update both READMEs
and this file in the same PR.

## Known limits

Remove an item when it is fixed.

- The int4 and f16_widen ceilings time each kernel's widest tile. A narrower
  tile (a call of few rows) unpacks or widens more per instruction, and that
  lowers its efficiency rather than its bound.
- A quantized source's copy is counted, the quantizing arithmetic is not.
- Instances share only DRAM bandwidth. Copies sharing an L3 domain, and a lower
  clock with more cores busy, are not modeled.
- No backend times per-group s8 calls. A per-group s8 problem's scales are
  priced per group only when it gives its group. The cost model prices only
  bf16, f16, per-channel int8 and s4 calls.
- Only matmul is modeled. Grouped matmul (MoE) and attention follow the
  interface in the README's "Adding an op".

## Failure policy

If a command fails, stop and report the step, the exact command and the first
real error line. Wait for the user.
