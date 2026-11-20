# Native BRGEMM tuner

Offline tuner of the native BRGEMM matmul (`ZENDNNL_MATMUL_ALGO=10`). It times
`ZENDNNL_MATMUL_NATIVE_CONFIG` candidates against the built-in plan and writes
the `ZENDNNL_MATMUL_NATIVE_TUNING_FILE`: a versioned JSON file with a hardware
block, table entries and learned rules (format `zendnnl-native-brgemm-tuning`,
version 2). It tunes the BF16 calls, the INT8 calls with per-channel weight
scales (s8 weights, a bf16 source quantized per token in the call), the s4
calls (weights in groups along K, or one group over K, over a bf16 source
quantized per token) and the F16 calls (an f16 source and weight, on the F16
kernels, or the F16_PH kernels where the CPU has AVX512-FP16). It plans FP32
calls and per-group INT8 calls too, but does not time them.

Requirements: Python 3.8 or newer. The schema, candidates, sessions, grid and
random search, export and checks use the standard library only; the guided
search and the rule training need numpy; scikit-learn is optional (detected at
run time, gradient-boosted trees when present, numpy models otherwise).

## Layout

| File | Role |
|---|---|
| `schema.py` | Loads the planner's schema: config keys (type, range, default or auto value, dtype, INT8 schemes, bits), match keys, rule features and comparisons, kernel registry (BF16, FP32, an INT8 kernel family per scheme it serves, the INT4 group kernels and the INT4 kernels of one group over K, F16 and F16_PH, each with its tile widths and rows per call), hardware, uarch, the machine's value of every version 2 hardware key (`hw_match`), the fields of the `[MATMUL.NATIVE_BRGEMM]` line (`plan_line`) and the values of its `gemv` field (`plan_line_gemv`). Reads the JSON the `SchemaDump` gtest prints between `BEGIN NATIVE TUNING SCHEMA` and `END NATIVE TUNING SCHEMA` (raw or inside a gtest log), else `fixtures/schema_dump.json`, that gtest's output. `fixtures/schema_design.json`, written from the design, is the schema of an older build, without the rule keys, the blocks schedule and `hw_match`. |
| `problems.py` | Problems (the planner's problem, `native_problem_t`: the weight type bf16, s8, f32, s4 or f16, a weight of N not a multiple of 32 planned and matched at N rounded up to whole 32-column blocks, `plan_n`, an s8 problem's INT8 scheme, an s4 problem's group size and scale bytes, the epilogue class), shape lists (TSV), M classes, the rule features as the loader computes them. |
| `plan.py` | The built-in planner (`native_planner.cpp`) line for line: the rules' plan (the tile grid, the rules behind the switches `makespan`, `l1_pass`, `traffic_groups`, `bf16_alias`, `bf16_narrow_n` and `blocks`, the L3 rules only with an L3, BF16 tiles of 64 and 128 columns, the blocks schedule, the tiles of per-group INT8 and of s4 calls with their pass budget (12 rows per call at both widths for an s4 weight of one group over K), the F16_PH tiles of two blocks and K passes of whole blocks), then with `cost_model` the model's choice among the rules' variants (rows per call and row groups for BF16 and F16, the long K pass and short M blocks of long weight streams (where one core streams slower than its share of the link, the M blocks of a BF16 team that fills its L3 domains only over a K above `cm_block_k`), INT8 row groups and blocks of three calls, s4 M blocks (weighed at the group kernel's rows per call) and the K passes of one group); plan validation, single-override resolution, plan signature, the plan fields of the log line (and of a gated call's paired tiles). The built-in plan has each key at the schema's default; a rule whose switch or bound the schema lacks is one its build does not have. |
| `cost_model.py` | The planner's cost model (`native_cost_model.cpp`) line for line: the modeled cycles of a plan's busiest worker, with every floating-point expression in the library's order. |
| `candidates.py` | Override text (format, parse, clauses), static checks, the default search space, per-problem candidate spaces, de-duplication. |
| `results.py` | Results store (JSONL) and the statistics: per-session ratios, noise, significance. |
| `backends/` | `jobs.py` (pod sessions), `benchdnn.py`, `replay.py`, `synthetic.py`, `common.py` (API-log line). |
| `replay/` | The cold-weight replay driver (`replay.cpp`), its build script and a pod-side A/B runner. |
| `plan_dump/`, `refresh_fixtures.sh` | The library's plans for the cases of `tests/plan_cases.py` (`plan_dump.cpp` and its build script), and the refresh of both fixtures from a build (see Tests). |
| `search.py`, `surrogate.py` | Grid, random and model-guided search; surrogate models. |
| `export.py`, `rules.py` | Class winners, confirmation, the tuning file and its validation; the learned rules. |
| `tune.py` | Command line. |
| `tunings/` | The tuning files of the CI shapes, one per machine type (see Tuning files). |
| `tests/` | Host unit tests, the sync with the planner's schema among them (`test_sync.py`). |

## How a table is produced

1. **Schema.** Run the `SchemaDump` gtest (`NativeConfigSchema.SchemaDump`) of
   the build being tuned and extract its output: `tune.py schema --extract
   gtest.log -o schema.json`. Without `--schema`, the fixture is used.
   `tune.py schema --schema schema.json --compare fixtures/schema_dump.json`
   lists the differences. The heuristic reads `l1d_bytes` (the L1 passes),
   `l2_bytes`, `l3_bytes` (the L3 of one domain; 0 on a machine that reports
   no L3, where no L3 rule applies) and the L3 domain (`l3_domain_cores`,
   the physical cores of one L3, or `family_l3_cores` when that is 0): when
   the dump has no `uarch` block, they come from its `hw` block (the cache
   sizes, and the domain from `l3_domain_threads` and `smt`), or give them
   with `--uarch l3_domain_cores=8,...`. The dump's `hw_match` holds the machine's
   value of every version 2 hardware key; a dump from a build without it
   gives no CPU vendor, family or model (see step 8). The cost model also
   reads `fma_ports`. The built-in plan is the rules at their keys' defaults
   (the makespan grid, the BF16 and INT8 L1-sized K passes, column groups by
   memory traffic, BF16 rows per call for aliasing source rows, BF16 tiles of
   several slabs at narrow N, the blocks schedule where one L3 domain's share
   of the source fits its L3, the batches of an s4 tile's groups in L1 at
   decode, F16_PH tiles of two blocks from `f16_wide_min_m` rows), then the
   cost model's choice (`cost_model`, its margin `cm_gain_pct` and its inputs
   `cm_*`, the hardware ones derived from the L3 domain unless set).
   Each rule's switch defaults to 1 (on), and setting it to 0 leaves the rule
   out. A key whose auto value is a legal setting (`row_unit` 12,
   `cm_tall_pct` -1, `cm_mem_cycles` 0, `cm_fma_pipes` 0) set to it is the
   same as left unset, so a candidate never sets it so. The tuner's built-in
   plan has every key at the default of the schema it is given; a rule whose
   switch the schema lacks (an older build) is off.
   A key of the schema that `plan.py` does not read (a newer build's) is not
   searched, and a candidate that sets it is dropped: the tuner cannot vouch
   for its plans.
2. **Problems.** A shapes TSV (`model name N K wei calls m_lo m_hi [group]`,
   one row per weight; `wei` is bf16, s8, s4, f16 or f32, and an s4 row's
   `group` its group size along K, -1 for one group over K, `--s4-group`
   when the row names none) and M classes give one problem per shape and
   representative M (`--m-classes "2-16:8/16,17-64:32/64,..."`:
   `lo-hi:rep/rep`, `lo-` is unbounded; M = 1 runs on the GEMV kernels and
   has no plan to tune, see step 4). Rows of one (dtype, N, K, group)
   merge; `lm_head` rows are left out. An s4 problem's scales take the bytes
   of the backend's `--scale-dt`.
   N is the call's: the override filters, the table entries and the rule
   features use its padded width, as the library matches it, and the log
   line names that width as `padded_n=`.
3. **Candidates.** Per problem, each searched key takes a few values derived
   from the problem and the registry (tile widths that divide `pack_nr`, row
   counts the kernels cover, M blocks, K passes, column-group divisors,
   thread caps, INT8 row caps, widths and pass sizes). The default search
   space holds only keys whose bits column says `same` for the dtype;
   bit-changing keys (`bias_in_dequant`) need `--allow-bit-changing`, and an
   entry that uses them is written only with `export --allow-bit-changing`
   after a model-level accuracy check. A candidate is kept only if it
   passes the static checks and resolves to a legal plan (with
   `init --keep-threads`, also only if its plan's team is no smaller than
   the built-in plan's: a call on fewer threads changes the thread team the
   calls before and after it run on, which a session of one shape at a time
   does not see); candidates that
   resolve to the same work (same plan signature, or the built-in plan) are
   measured once. Each rule switch (`makespan`, `l1_pass`,
   `traffic_groups`, `blocks`, and for BF16 `bf16_alias` and
   `bf16_narrow_n`) at 0 (the rule off), and the constants of the rules
   (`min_gain_pct`, `idle_pct`, `decode_m`, `l1_pct`, `l1_pass_max_mb`,
   `min_pass_k`, which both dtypes read, `bf16_alias_mr`, and the bounds
   `bf16_narrow_n_min_m`, `bf16_blocks_min_m` and `int8_blocks_min_m` at 1,
   the rule from the first row) are searched too; BF16 problems also take
   tiles of 64 and 128 columns (`nr`), and the schedules take
   `schedule=blocks` and units of 2 and 4 column tiles (`unit_tiles`, a pin
   of that schedule). A constant of a rule that is off changes no plan and
   is merged away, so with the rules off the constants count together with a
   switch (`--keys` searches the rules alone, see Usage).
4. **Sessions.** A round hands its candidates to the backend. For the pod
   backends, `backends/jobs.py` writes one session directory per pod claim
   (up to `--parallel` per round) and starts `--launcher` on it, for example
   `'<tools>/run_one.sh --cmd {dir} -- bash {script}'`. With `--pods a,b,..`
   (for a launcher that claims a named pod) session i of a round runs on the
   i-th pod: the launcher gets `POD=<pod>` in its environment and `{pod}` in
   its text, and a round has at most one session per pod. Problems are grouped
   into processes; slot 0 of a group is the built-in plan and slot j holds
   the j-th candidate of each problem as one override clause per problem
   (`dtype=s8,m=8,n=4096,k=4096,threads=32:nr=16,...`, at most 64 clauses),
   so one process times many problems. Each slot first runs once with
   `ZENDNNL_API_LOG_LEVEL=4`: the log line's `source=` and `cfg=` show whether
   the candidate was applied, dropped (an illegal plan, the planner fell back)
   or replaced, and the logged plan is compared with the tuner's plan model
   (the line's plan fields: `nr`, `mb`, `kc` and the F16 call's `kernels`,
   `tiles`, `threads`, the schedule, `mr`, `row_unit`, `traversal`, the INT8
   rows, widths and K pass, an s4 call's `int8_scheme`, `gs` and
   `pass_budget`; a gated call over halves names the width its tiles paired
   them at, `pairing=`, and runs the paired plan). A call on the tiles logs
   `gemv=none`. Every M = 1 call runs on the GEMV kernels of its type
   instead, which take all of K for each block of columns and none of the
   plan's tiles, M blocks or K passes: its line names them (`gemv=bf16`,
   `f32`, `f16`, `f16_ph`, `s8`, `s8_group` or `s4`, which the plan model
   gives too) next to the plan of the tiles, and such a probe is recorded as
   `gemv` and kept out, so M = 1 has no plan to tune.
   Then the timed runs go A S1 .. Sk A per repeat (slots reversed on odd
   repeats), with the log off. Without `--launcher`, or with `--dry-run`,
   sessions are written and the command to run each is printed; run them,
   then `tune.py collect`.
5. **Results.** Every run becomes records in `results.jsonl`. A candidate is
   judged per session against that session's built-in runs: ratio =
   median(candidate) / median(built-in); noise = z * sqrt(rel_mad(built-in)^2
   + rel_mad(candidate)^2) (MAD-based, z = 2 by default). It counts as a gain
   when 1 - ratio exceeds max(min_gain, noise) in every session. Candidates
   whose probe was dropped, mismatched, unlogged, ran on the GEMV kernels,
   changed the output hash (replay `--check 1`) or ran another plan than the
   plan model's (an entry names the model's plan) are excluded. A probe
   whose run left no output (`failed`, as `collect --force` records a session
   that did not run) counts only until another probe of the candidate
   answers, so the next search round runs the candidate again and its
   results count.
6. **Search.** `grid` takes every one-key change first, then pairs, within
   the budget; `random` changes each key with a probability; `guided` starts
   with random candidates and then fits a surrogate of log(time / built-in
   time) on problem and resolved-plan features over all problems, proposing
   the lowest lower confidence bounds among a random pool and the neighbours
   of each problem's best candidates. A search resumes from the store.
7. **Confirmation.** `tune.py confirm` first measures the best candidates of
   each representative M at the class's other Ms (`--top`, a search round),
   since the search explores each problem on its own. It then picks each
   class winner: the lowest geometric-mean ratio over the class's Ms among
   candidates that gain beyond the noise at every M where they change the
   plan (an M where a candidate is the built-in plan is neutral) and that are
   legal at every M of the class. The winners run against the built-in plan
   again, with more repeats (`--confirm-repeats`). With a dry run, collect
   and rerun `confirm` after each written round.
8. **Export.** `tune.py export -o tuning.json` writes an entry per class
   whose confirmation agrees: `match` holds every problem field and the M
   range (from M 2: an M = 1 call runs on the GEMV kernels, so a class of
   M = 1 alone has no entry), `config` the full knob set the measurement
   fixed (the winner's keys, the built-in value of every other searched
   key, the rule switches among
   them (1 unless the winner sets one to 0), equivalent explicit values for
   `threads_cap` / `domain_threads`, and auto pins that reproduce the
   measured plan over the whole class, `unit_tiles` of a blocks plan among
   them), so learned rules only fill what an entry leaves open. `hardware`
   (version 2) matches CPU
   fields, each a value, a list or a `{"min", "max"}` range: by default the
   machine's `vendor`, `family`, `model`, `isa` (the features the CPU must all
   have), `l1d_kib`, `l2_kib`, `l3_kib` and `l3_cores` (the plan's L3 domain),
   but not its core count or SMT setting, so the file also applies to the same
   part with other core counts or SMT on. `--hw-keys` picks the keys (for
   example add `smt` or `cores`), and `--hw-set key=value` writes a value
   instead of the machine's: `model=0-15`, `l3_kib=32768|131072`, `cores=64-`,
   `vendor=A|B`, `isa=avx512bf16+avx512vnni`, so one file can serve several
   models. A schema without `hw_match` (an older build's dump) gives no
   vendor, family or model: dump it again, or leave those out with
   `--hw-keys`.
   The `rules` section is a decision tree over the rule features, trained to
   minimize time relative to the built-in plan, with the config of a leaf
   taken from the most frequent winners. It learns from and applies to the
   problems from M 2: the loader applies rules to every call, so the tree
   starts with a split `m <= 1` of no config, and each ordered rule with
   `["m", ">=", 2]`. `export_report.json` and the file's
   `meta.rules_report` report it on held-out shapes (K folds over dtype, N,
   K) next to the best measured configs, with null ratios when no held-out
   problem is measured (a campaign of one shape). The file is validated
   (format, types, ranges, limits) before it is written, as strict JSON
   within the loader's bounds (64 MiB, 128 levels of nesting, 2^21 values):
   the loader ignores a file it cannot parse or that exceeds one, entries
   included. Its `meta.plans` (which the loader ignores) records, per entry
   and problem of the campaign it matches, the plan the library should log
   under the file: `source=table`, the entry's config as the call applies it
   (`cfg=`), the plan's fields, and whether that is the built-in plan;
   `tune.py verify` checks them (see Validating a table).

## Backends

- **replay**: `replay/replay.cpp` makes the framework plugin's exact
  `matmul_direct` calls (a bf16 source over bf16 weights, or over s8 weights
  or s4 weights in groups along K (`--group`, -1 for one group) whose source
  it quantizes per token; an f16 source over f16 weights into an f16 dst),
  one weight type (and one s4 group size) a process, with `--layers` weight
  sets per shape
  visited in layer order so every weight is cold when it comes back, as in a
  model. A process whose shapes are small takes more layers, so that it
  cycles through at least `--cold-mb` of distinct weights (by default eight
  times the L3 of the domains the team spans) whichever problems share it: a
  confirmation run of a few small shapes measures them cold too. Below that,
  a weight may come back from the L3 or with its pages still mapped warm: a
  decode call's time grows with the distinct weights between its visits up
  to about a gigabyte on a 32-thread team. 32
  threads, and the model server's OpenMP setup in the job: `--preload` (the
  worker's tcmalloc and libiomp5), a `KMP_AFFINITY` proclist of the first
  `--threads` CPUs of the pod's cpuset (on a pod with more CPUs, the threads
  of its first L3 domains), `KMP_BLOCKTIME=5`, `KMP_TPAUSE=0`. Build it on the host against the
  ZenDNN build being tuned: `replay/build.sh <worktree or build dir> <out>`
  (it links the optional dependencies the build installed, so a build with
  the default dependencies, without oneDNN or with ParLooper works too).
- **benchdnn**: one matmul input line per problem with the kernel left to
  the library (`none`), so the job's `ZENDNNL_MATMUL_ALGO=10` selects the
  native BRGEMM. The lines take bf16 weights, or s8 weights with per-channel
  scales under a bf16 source quantized per token; `init` skips the problems
  of other weight types and schemes. One benchdnn process per line by
  default (`--whole-file` for one process per run); benchdnn's warm-weight
  timing, `OMP_PROC_BIND=close OMP_PLACES=cores`.
- **synthetic**: no measurement; a cost model of the resolved plan with
  noise, for tests and dry runs on the host.

The replay and benchdnn calls pack every weight in blocks of 32 columns, as
the library does, so those backends skip a problem of another `--pack-nr`
(the synthetic backend takes any).

## Usage

```
T=tools/native_brgemm_tuning/tune.py
tools/native_brgemm_tuning/replay/build.sh <zendnn worktree> <data>/replay_tune
python3 $T init --workdir <data>/camp1 --schema schema.json --shapes shapes.tsv --dtypes s8 \
    --backend replay --replay-bin <data>/replay_tune --preload <tcmalloc.so>:<libiomp5.so> \
    --launcher '<tools>/run_one.sh --cmd {dir} -- bash {script}' --parallel 2
# sessions on named pods: the launcher reads POD (or takes {pod})
python3 $T init --workdir <data>/camp2 ... --launcher '<tools>/run_one_named.sh --cmd {dir} -- bash {script}' \
    --pods <pod a>,<pod b> --parallel 2
python3 $T search  --workdir <data>/camp1 --strategy guided --budget 32 --batch 8
# the rule switches and constants alone, up to two keys at a time
python3 $T search  --workdir <data>/camp1 --strategy grid --max-keys 2 --budget 32 --batch 8 \
    --keys makespan,l1_pass,traffic_groups,blocks,min_gain_pct,idle_pct,decode_m,l1_pct,l1_pass_max_mb
python3 $T confirm --workdir <data>/camp1
python3 $T export  --workdir <data>/camp1 -o <data>/camp1/tuning.json
python3 $T check   <data>/camp1/tuning.json --workdir <data>/camp1
python3 $T status  --workdir <data>/camp1
# one file of two campaigns' tables (e.g. decode and prefill Ms timed with other backend settings)
python3 $T merge   -o <data>/tuning.json <data>/camp1/tuning.json <data>/camp2/tuning.json --schema schema.json
# a file for two models of one family, with their L3 sizes and domains
python3 $T export  --workdir <data>/camp1 -o <data>/camp1/family.json \
    --hw-set 'model=2|81' --hw-set l3_kib=32768-131072 --hw-set 'l3_cores=8|32'
```

`tune.py space --problem dtype=s8,m=8,n=4096,k=4096` shows one problem's
values and candidates. The session directory, its job script and outputs live
under the workdir (`sessions/<id>/`), which must be visible from the pods.

`tune.py merge` writes one file of the entries of several files: they must
have the same hardware block and no rules, and no two entries may match one
problem (the loader applies only the first); the plans the inputs name are
kept in `meta.plans` (two inputs may not name different plans for one
problem), so `tune.py verify` checks the merged file against each campaign;
the rest of the inputs' meta blocks are kept under `meta.merged` with their
file names, and the result is validated against the schema.

## Validating a table

- `tune.py verify tuning.json --workdir <campaign>` runs the campaign's
  problems in one pod session (the campaign's launcher) twice, with the
  built-in plan and with `ZENDNNL_MATMUL_NATIVE_TUNING_FILE=tuning.json`,
  with the plan lines and the output hashes logged, and checks that the
  library reads the file and gives the plans it names: every problem an entry
  names logs `source=table`, the entry's `cfg=` and the plan of `meta.plans`,
  every other problem keeps the built-in plan, and every output keeps the
  built-in plan's bits. `verify_report.json` lists each problem.
- `tune.py check tuning.json --workdir <campaign>` validates the file against
  the schema and shows, per problem, which layers the loader would apply, the
  merged config, and whether the resolved plan is legal; when the hardware
  block does not hold on the schema's machine, it names the first key that
  fails (and, for version 2, the machine's value).
- At op level, `replay/run_replay.sh` runs the built-in plan and the table in
  one pod session, e.g. items
  `base:<replay>:10,tuned:<replay>:10:ZENDNNL_MATMUL_NATIVE_TUNING_FILE=<file>`
  with `--check 1`: the times compare the two, the hashes show that the
  default bits are kept.
- With `ZENDNNL_API_LOG_LEVEL=4` the log line's `source=table` (or `rules`)
  and `cfg=` show the applied layers; a file that loads logs one line with
  its entry and rules counts.

## Tests

`python3 -m unittest discover -s tools/native_brgemm_tuning/tests -t tools`
(host only: no pod, no library). Tests that need numpy are skipped without it.
`test_sync.py` checks the tuner against the planner's schema: every key is
mirrored at its built-in value and has search values, the match keys and
their names, the rule features and comparisons, the tuning-file names, the
kernel kinds, the plan-line fields and `gemv` values and the uarch fields are
the tuner's. With `ZENDNNL_NATIVE_SCHEMA=<SchemaDump JSON or gtest log>` it
also checks that dump, which must agree with the fixture. The planner's side is
`NativeConfigSchema.TheTunerFollowsThePlanner` (the build's schema against
`fixtures/schema_dump.json`) and `NativeConfigSchema.PlanLineFields` (the
plan line's fields and `gemv` values against real lines, M = 1 calls of each
weight type among them). The schema shows no rule, cost-model term or
default of the planner, so the CI runs both with
`NativeConfigText.KeyDocsMatchTheKeyTable` (the key table the dump takes
enum defaults from, against the planner's defaults), then the plan
comparison below on the build (`plan_cases.py compare` of the build's plan
dump on the build's schema: the f16 kernel kind and the plans come from
one machine, so it holds on any), then these tests on the build's dump.
`test_cost_model.py` checks the cost model's mirror and choices against the
library's NativeCostModel cases. `test_tunings.py` checks the files of
`tunings/`: each is valid, and the plans `plan.py` gives its entries on its
machine (the hardware block's caches and L3 domain) are its `meta.plans`,
so a planner change that moves an entry's plan fails it.

`test_plan.py` checks `plan.py` against the library's plans:
`fixtures/plan_check.json` holds the SHA-256 of the library's plan lines for
the cases `tests/plan_cases.py` draws (random problems of every weight type,
machines and configs, with pins, the rule switches, constants, the cost
model's inputs and the blocks schedule, s4 group sizes and one group over K,
widths that are not whole 32-column blocks, per-group INT8 calls and machines
without an L3; a case without a config plans with the build's defaults, a
fraction sets every rule switch to 0 and another sets cost-model inputs; the
fixture records those counts and the cases of each weight type). Every field
of the plan is compared, its kernels and the problem's padded width.

When the planner, its keys or its defaults change, refresh both fixtures
from the new build, with the plans of a machine of the ISA of its
`SchemaDump` (the kernel registry follows the ISA: an f16 call takes the
F16_PH kernels only with AVX512-FP16). Take both from a machine with every
kernel kind, so that `NativeConfigSchema.TheTunerFollowsThePlanner` passes on
every machine (it accepts a kind the running CPU lacks):

    tools/native_brgemm_tuning/refresh_fixtures.sh <worktree> <SchemaDump gtest log> [build dir]
    # or with the plan dump made on that machine
    DUMP=<dump> tools/native_brgemm_tuning/refresh_fixtures.sh <worktree> <SchemaDump gtest log>

It builds `plan_dump/plan_dump.cpp` against the worktree's planner headers
(they are not installed) and the build's static archive
(`plan_dump/build.sh`), writes `fixtures/schema_dump.json` with `tune.py
schema --extract`, and prints `heuristic_plan` of every line of
`plan_cases.py cases` (the problem through `make_native_problem`, the
machine, and the config through `parse_native_config` and
`apply_native_config`) with the verdict of `validate_native_plan`, as
`plan_line()` writes them. Both fixtures are rewritten only when
`plan_cases.py compare` finds `plan.py` giving the library's plan in every
case; otherwise the differences are listed, field by field, and `plan.py` has
to follow the library first. Run on the build the fixtures came from, it
reproduces both fixtures byte for byte. A larger check, on one machine
description (`--machine l1d,l2,l3,family_l3,dom`):

    tools/native_brgemm_tuning/plan_dump/build.sh <worktree> /tmp/plan_dump [build dir]
    P=tools/native_brgemm_tuning/tests/plan_cases.py
    M=49152,1048576,33554432,8,8
    python3 $P cases --seed 11 --count 200000 --machine $M | /tmp/plan_dump > /tmp/dump.txt
    python3 $P compare /tmp/dump.txt --seed 11 --count 200000 --machine $M --schema schema.json

## Keeping the tuner in sync with the planner

The tuner mirrors the planner, so a change to the native planner
(`matmul_native/brgemm/planner/`), to a `ZENDNNL_MATMUL_NATIVE_CONFIG` key, a
match key, a rule feature or the tuning file's names, to the kernel registry,
or to the `[MATMUL.NATIVE_BRGEMM]` line updates this directory in the same
change:

1. Build, and run `NativeConfigSchema.*` (its `SchemaDump` prints the new
   schema). `TheTunerFollowsThePlanner` names each key, match key, feature,
   kernel, plan-line field or `gemv` value the tuner does not have yet; it
   does not see a change to a rule, a cost-model term or a default. For
   those, compare the build's plans with `plan.py` on the build's schema:
   `plan_dump/build.sh`, `plan_cases.py cases` through the plan dump, and
   `plan_cases.py compare <its output> --schema <that gtest log>` names
   each case the mirror plans otherwise, field by field. The CI runs both.
2. Make the tuner follow: a key is mirrored in `plan.py` (`BUILTIN_PARAMS`,
   and the rule that reads it) and gets search values in `candidates.py`; a
   rule or a cost-model input is mirrored in `plan.py` / `cost_model.py`; a
   problem field goes into `problems.py` (key, filters, features), the plan
   dump and the cases; a match key or name into `export.py` and
   `problems.py`; a kernel kind into `schema.py`; a plan-line field into
   `plan.LOG_LINE_FIELDS` (and `plan_log_fields` when it is a plan field);
   a `gemv` value, or a change to which calls run on the GEMV kernels, into
   `plan.GEMV_KINDS` and `plan.gemv_kind`.
3. Refresh both fixtures (above) and run the tests here; `test_sync.py` and
   `test_plan.py` fail until the tuner follows the planner, and
   `test_tunings.py` while a shipped entry's plan has moved (step 4).
4. Rerun the campaigns of each machine type and rewrite `tunings/` (below):
   a plan rule change can make an entry's pinned plan worse than the new
   built-in plan, or leave it unchanged. `tune.py verify` checks each file
   against the new library.

## Tuning files

`tunings/` holds one tuning file per machine type, written by this tool for
the CI models' dense linears and the expert and shared-expert shapes of the
CI MoE models, at M 2 to 4096 (M = 1 runs on the GEMV kernels), as
`ZENDNNL_MATMUL_NATIVE_TUNING_FILE` reads it. Each file's hardware block
matches its machine type by what the planner
reads (the ISA, the L1d, L2 and L3 sizes and the cores of an L3 domain:
`export --hw-keys isa,l1d_kib,l2_kib,l3_kib,l3_cores`), so the loader ignores
it on other machines. A file holds only entries whose gain over the built-in
plan was confirmed beyond the noise in separate sessions with cold weights;
the classes without one keep the built-in plan. Its `meta.plans` names the
plan of every problem an entry matches, which `tune.py verify` checks on the
library.

A file is made per machine type and team size with one campaign per weight
set and M range, then `tune.py merge`:

    python3 $T init --workdir <wd> --schema <SchemaDump log of the machine> \
        --shapes <the CI shapes TSV> --dtypes bf16,s8,s4,f16 --threads 32 \
        --m-classes 2-4:4,5-8:8,9-16:16,17-32:32,33-64:64,65-128:128 \
        --backend replay --replay-bin <replay> --preload <tcmalloc>:<libiomp5> \
        --layers 1 --iters 30 --warmup 2 --budget-ms 1500 --keys <keys> --values <key=a|b> ...
    python3 $T search  --workdir <wd> --strategy grid --max-keys 1 --budget 8 --batch 8
    python3 $T confirm --workdir <wd>; python3 $T confirm --workdir <wd> --fallback
    python3 $T export  --workdir <wd> -o <wd>/tuning.json --rules none \
        --hw-keys isa,l1d_kib,l2_kib,l3_kib,l3_cores
    python3 $T merge -o tunings/<machine type>.json <wd>/tuning.json ...
    python3 $T verify tunings/<machine type>.json --workdir <wd>

(prompt sizes: `--m-classes 129-512:512,513-2048:2048,2049-:4096`, fewer
iterations).

| File | Machine type | Teams | Entries |
|---|---|---|---|
| `l3_domain_8_cores_32mib.json` | L3 domains of 8 cores and 32 MiB, L2 1 MiB, L1d 48 KiB; AVX512-BF16 and VNNI, no AVX512-FP16 | 32 threads (four L3 domains) | 245: bf16 72, s8 44, s4 29, f16 100 |
| `l3_domain_32_cores_128mib.json` | L3 domains of 32 cores and 128 MiB, L2 1 MiB, L1d 48 KiB; AVX512-FP16 (F16_PH kernels) | 32 threads (one L3 domain), M 2-4096; 64 threads (two), M 2-128 | 677: 32 threads bf16 59, s8 176, s4 59, f16 111; 64 threads bf16 57, s8 149, s4 8, f16 58 |

Most entries take a K pass of 1024 or 2048 at M 65-128, 8-row INT8 calls (`int8_mr=8`) at M 16-128 on the
32-core domains, the 64-column F16_PH tiles at M 5-16, or column groups at M 128 and prompt sizes; decode
sizes up to M 4 keep the built-in plan almost everywhere. The hardware block cannot name AVX512-FP16: on a
machine of the same caches without it, an f16 entry of 64-column tiles is illegal for the F16 kernels and
dropped for that call, with a warning.
