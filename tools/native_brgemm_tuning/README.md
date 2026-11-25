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
| `plan.py` | The built-in planner (`native_planner.cpp`) line for line: the rules' plan (the tile grid, the rules behind the switches `makespan`, `l1_pass`, `traffic_groups`, `bf16_alias`, `bf16_narrow_n` and `blocks`, the L3 rules only with an L3, BF16 tiles of 64 and 128 columns and every type's when pinned (slabs of each kernel kind's widest call, the INT8 K pass at the tile's width), the blocks schedule, the tiles of per-group INT8 and of s4 calls with their pass budget (12 rows per call at both widths for an s4 weight of one group over K), the F16_PH tiles of two blocks and K passes of whole blocks), then with `cost_model` the model's choice among the rules' variants (rows per call and row groups for BF16 and F16, the long K pass and short M blocks of long weight streams (where one core streams slower than its share of the link, the M blocks of a BF16 team that fills its L3 domains only over a K above `cm_block_k`), INT8 row groups and blocks of three calls, s4 M blocks (weighed at the group kernel's rows per call) and the K passes of one group); plan validation, single-override resolution, plan signature (the K passes the calls of each INT8 tile take, in turn or one at a time, the tail tile's among them), the plan fields of the log line (a gated call's of the tiles it runs: the width they pair its halves at, `native_paired_width`, and their plan), and an M = 1 call's fields: the GEMV kernels or the tiles (`gemv`), whether it runs on the calling thread (`gemv_one_thread_pct`), and on the GEMV kernels its threads and blocks per kernel call over the units of its columns (`gemv_blocks`, the rules of `native_gemv_threads` and `native_gemv_call_blocks`). The built-in plan has each key at the schema's default; a rule whose switch or bound the schema lacks is one its build does not have, and a schema whose `l1_pct` has no auto value is a build whose single row block of more than `l1_pass_max_mb` rows keeps all of K. |
| `cost_model.py` | The planner's cost model (`native_cost_model.cpp`) line for line: the modeled cycles of a plan's busiest worker, with every floating-point expression in the library's order. |
| `candidates.py` | Override text (format, parse, clauses), static checks, the default search space, per-problem candidate spaces, de-duplication. |
| `results.py` | Results store (JSONL) and the statistics: per-session ratios, noise, significance. |
| `backends/` | `jobs.py` (pod sessions), `benchdnn.py`, `replay.py`, `synthetic.py`, `common.py` (API-log line). |
| `replay/` | The cold-weight replay driver (`replay.cpp`), its build script and a pod-side A/B runner. |
| `ab/` | The A/B gate of a library change: a corpus of calls, layouts and layout rule verdicts (`ab_corpus.cpp`, its build script) and `ab.py`, which compares two builds on it, their code size and their pack and unpack time (see Comparing two builds). |
| `plan_dump/`, `refresh_fixtures.sh` | The library's plans for the cases of `tests/plan_cases.py` (`plan_dump.cpp` and its build script), and the refresh of both fixtures from a build (see Tests). |
| `search.py`, `surrogate.py` | Grid, random and model-guided search; surrogate models. |
| `export.py`, `rules.py` | Class winners, confirmation, the tuning file and its validation; the learned rules. |
| `op_perf.py`, `calibration.py` | The machine's ceilings in a campaign (`tools/op_perf_utility`): the machine check, each problem's bound and efficiency, `meta.op_perf`; the cost model's inputs from the ceilings, their A/B and their rules (see Machine ceilings, headroom and calibration). |
| `tune.py` | Command line. |
| `tunings/` | The tuning files of the CI shapes, one per machine type, and those of single models in `models/` (see Tuning files). |
| `tests/` | Host unit tests, the sync with the planner's schema among them (`test_sync.py`). |

## How a table is produced

1. **Schema.** Run the `SchemaDump` gtest (`NativeConfigSchema.SchemaDump`) of
   the build being tuned and extract its output: `tune.py schema --extract
   gtest.log -o schema.json`. Without `--schema`, the fixture is used, which
   describes an example machine: only a synthetic campaign may do that. `init`
   refuses a replay or benchdnn campaign without it, and `export` refuses such a
   campaign made before unless `--hw-match none`.
   `tune.py schema --schema schema.json --compare fixtures/schema_dump.json`
   lists the differences. The heuristic reads `l1d_bytes` (the L1 passes),
   `l2_bytes`, `l3_bytes` (the L3 of one domain; 0 on a machine that reports
   no L3, where no L3 rule applies), the L3 domain (`l3_domain_cores`,
   the physical cores of one L3, or `family_l3_cores` when that is 0) and
   `physical_cores` (the L2 caches a team's threads can use; 0 counts one a
   thread): when the dump has no `uarch` block, they come from its `hw` block
   (the cache sizes, the domain from `l3_domain_threads` and `smt`, and
   `physical_cores`), or give them with `--uarch l3_domain_cores=8,...`. The
   dump's `hw_match` holds the machine's
   value of every version 2 hardware key; a dump from a build without it
   gives no CPU vendor, family or model (see step 8). The cost model also
   reads the library's machine model: `uarch.dot`, per compute path (each
   family's `path`: `fp32`, `bf16`, `int8`, `int4` the s4 tile's loop, `fp16`
   the F16_PH kernels', `f16_widen` the F16 kernels') the dot products a core
   issues a cycle (`ipc`) and their `latency`, and the memory terms
   `uarch.mem_cycles` and `uarch.stream_cycles` in the machine's cycles (0:
   not set, the planner's 2400 and 600). A kernel call of a step of
   rows x vectors instructions costs max(rows x vectors / ipc, `cm_fma_lat`)
   cycles a step, at `cm_fma_lat` 0 its path's latency over the call's
   accumulator sets; the ipc is the path's where `uarch.priced` (a model the
   environment sets; the library's class default, which every CPU takes
   without one, is not), else 2, and `cm_fma_pipes` pins it. A dump of a
   build without the model gives its `fma_ports` to every path with latency
   6, priced;
   `--uarch int8_ipc=4,int8_lat=4,mem_cycles=1500` sets paths and terms as
   `ZENDNNL_MATMUL_NATIVE_UARCH` does (`fma_ports=N` sets every path's ipc).
   The built-in plan is the rules at their keys' defaults
   (the makespan grid, the BF16 and INT8 L1-sized K passes, in a single row
   block of up to `decode_m` rows by the weight's fit in L3 (INT8 where an
   L3 domain holds at most 16 cores), at decode of a
   share of L1 by the weight against the L2 of the threads, column groups by
   memory traffic, BF16 rows per call for aliasing source rows, BF16 tiles of
   several slabs at narrow N, the blocks schedule where one L3 domain's share
   of the source fits its L3, the batches of an s4 tile's groups in L1 at
   decode: half of L1 where the weight fits the L2 of the plan's threads,
   one L2 a physical core, and the source rows of its row block fit one L2,
   else 125 percent of it, F16_PH tiles of two blocks from `f16_wide_min_m`
   rows), then the cost model's choice (`cost_model`, its margin
   `cm_gain_pct` and its inputs `cm_*`, the hardware ones derived from the L3
   domain unless set).
   Each rule's switch defaults to 1 (on), and setting it to 0 leaves the rule
   out. A key whose auto value is a legal setting (`row_unit` 12,
   `int4_l1_pct` -1, `l1_pct` -1, `int8_blocks_min_m` -1, `cm_tall_pct` -1,
   `cm_mem_cycles` 0, `cm_stream_cycles` -1, `cm_fma_pipes` 0) set to it is
   the same as left unset, so a candidate never sets it so. The tuner's
   built-in plan has every key at the default of the schema it is given; a
   rule whose switch the schema lacks (an older build) is off.
   A key of the schema that `plan.py` does not read (a newer build's) is not
   searched, and a candidate that sets it is dropped: the tuner cannot vouch
   for its plans.
2. **Problems.** A shapes TSV (`model name N K wei calls m_lo m_hi [group]`,
   one row per weight; `wei` is bf16, s8, s4, f16 or f32, and an s4 row's
   `group` its group size along K, -1 for one group over K, `--s4-group`
   when the row names none) and M classes give one problem per shape and
   representative M (`--m-classes "2-16:8/16,17-64:32/64,..."`:
   `lo-hi:rep/rep`, `lo-` is unbounded; M = 1 runs on the GEMV kernels,
   whose keys no other M reads: a class with M 1 among its representatives
   covers M 1 alone, `1`, and a problem at M 1 in a wider class, such as a
   shape whose M range is 1 alone or a `--problem` at M 1, is judged and
   exported in a class of M 1 alone). Rows of one (dtype, N, K, group) merge;
   `lm_head` rows are left out. An s4 problem's scales take the bytes
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
   the built-in plan's, an M = 1 call's the threads of its GEMV kernels or
   the calling thread alone: a call on fewer threads changes the thread team
   the calls before and after it run on, which a session of one shape at a
   time does not see); candidates that
   resolve to the same work (same plan signature, or the built-in plan) are
   measured once. Each rule switch (`makespan`, `l1_pass`,
   `traffic_groups`, `blocks`, and for BF16 `bf16_alias` and
   `bf16_narrow_n`) at 0 (the rule off), and the constants of the rules
   (`min_gain_pct`, `idle_pct`, `decode_m`, `l1_pct`, `l1_pass_max_mb`,
   `min_pass_k`, which both dtypes read, `bf16_alias_mr`, and the bounds
   `bf16_narrow_n_min_m`, `bf16_blocks_min_m` and `int8_blocks_min_m` at 1,
   the rule from the first row) are searched too; every type also takes
   tiles of 64 and 128 columns (`nr`) where the build's registry lists
   them (an older build: BF16 only), slabs of 32 columns, the widest call
   of every kernel kind but F16_PH, whose widest call is its tile of 64
   and whose 128 is two of those; the built-in plan takes them for BF16
   (`bf16_narrow_n`) and F16_PH 64 only, so only a pin reaches the others.
   The schedules take
   `schedule=blocks` and units of 64, 128 and 256 columns (`nb`, the N
   block, a pin of that schedule: the unit takes `nb` / `nr` column tiles,
   at least one, whatever `nr` the plan takes, and replaces `unit_tiles`,
   the unit in tiles, which is searched only when `--keys` names it or on
   the schema of a build without `nb`). An M = 1 problem, which runs on the
   GEMV kernels, takes their keys alone (every other M all the others;
   `--keys` adds others, such as `threads_cap`, whose cap on the plan's
   threads gives the GEMV kernels fewer): `gemv` at 0 (the tiles), `gemv_blocks` from 1 to 4 (an s8 or s4 weight's
   kernels take 2 at most, so 3 and 4 are its 2; a gated call whose halves
   pair as whole blocks reads a gate block with its up block, so 1 and 3 are
   its 2) and `gemv_one_thread_pct`
   from 0 (no call on the calling thread) to 1000 (layouts of up to 10 times
   L2 there); a value that gives the built-in schedule is merged away. A
   constant of a rule that is off changes no plan and is merged away, so
   with the rules off the constants count together with a switch (`--keys`
   searches the rules alone, see Usage).
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
   them at, `pairing=` (or `staged`), and runs the paired plan: the check
   expects that width to be the one the model pairs them at, and the paired
   plan's fields at it). A call on the tiles logs
   `gemv=none`. Every M = 1 call runs on the GEMV kernels of its type
   instead (unless `gemv` is 0), which take all of K for each block of
   columns and none of the plan's tiles, M blocks or K passes: its line names
   them (`gemv=bf16`, `f32`, `f16`, `f16_ph`, `s8`, `s8_group` or `s4`, which
   the plan model gives too) next to the plan of the tiles, and, when
   `gemv_blocks` or `gemv_one_thread_pct` is set, the threads that took its
   columns (at most its `team=`) and the most blocks one of their kernel
   calls read (`gemv_threads=`, `gemv_blocks=`; a thread's whole share where
   that is fewer than the call's blocks), which the check compares with the
   model's: a call that ran on fewer threads than the model's is a mismatch,
   and so is a line without them whose `team=` is below the model's threads.
   A probe of such a call whose candidate leaves the GEMV kernels' schedule
   (their threads and the blocks a kernel call reads) as the built-in
   plan's is recorded as `gemv` and kept out: the call took none of its
   keys. A key of the tiles changes that schedule only through the plan's
   threads, as `threads_cap` does.
   Then the timed runs go A S1 .. Sk A per repeat (slots reversed on odd
   repeats), with the log off. Without `--launcher`, or with `--dry-run`,
   sessions are written and the command to run each is printed; run them,
   then `tune.py collect`.
5. **Results.** Every run becomes records in `results.jsonl`. A candidate is
   judged per session against that session's built-in runs: ratio =
   median(candidate) / median(built-in); noise = z * sqrt(rel_mad(built-in)^2
   + rel_mad(candidate)^2) (MAD-based, z = 2 by default). It counts as a gain
   when 1 - ratio exceeds max(min_gain, noise) in every session. Candidates
   whose probe was dropped, mismatched, unlogged, ran on the GEMV kernels
   at the built-in plan's schedule, changed the output hash (replay `--check 1`) or
   ran another plan than the plan model's (an entry names the model's plan)
   are excluded. So are all
   candidates of a problem whose built-in run mismatched, for example a
   gated call paired at another width than the model's: the model's built-in
   plan, against which candidates are de-duplicated, is not the library's,
   and the search gives that problem no more candidates. A probe
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
   again, with more repeats (`--confirm-repeats`); with `--alone`, each
   winner in a slot of its own, the other problems of its process at their
   built-in plans (see the end of Machine ceilings, headroom and
   calibration). With a dry run, collect and rerun `confirm` after each
   written round.
8. **Export.** `tune.py export -o tuning.json` writes an entry per class
   whose confirmation agrees: `match` holds every problem field and the M
   range (from M 2: an M = 1 call runs on the GEMV kernels, so the entry of a
   class of M = 1 alone, at M 1, takes a winner that gives them another
   schedule than the built-in plan's, by their keys or a cap on the plan's
   threads, and a winner that leaves it gets none), `config` the full knob
   set the measurement
   fixed (the winner's keys, the built-in value of every other searched
   key, the rule switches among
   them (1 unless the winner sets one to 0), equivalent explicit values for
   `threads_cap` / `domain_threads`, and auto pins that reproduce the
   measured plan over the whole class, `nb` of a blocks plan among them,
   `nr` times its unit tiles, or `unit_tiles` where the search took it and
   not `nb` (`--keys`, or the schema of a build without `nb`); a winner that
   sets `nb` or `unit_tiles` keeps its unit as that key alone, since `nb`
   replaces `unit_tiles`), so learned rules only fill what an entry leaves
   open.
   `hardware` (version 2) matches CPU
   fields, each a value, a list or a `{"min", "max"}` range: by default the
   machine's `vendor`, `family`, `model`, `isa` (the features the CPU must all
   have: `avx512f`, `avx512bf16`, `avx512vnni`, `avx512fp16`, the last one
   telling the machines of the F16_PH kernels apart), `l1d_kib`, `l2_kib`, `l3_kib` and `l3_cores` (the plan's L3 domain),
   but not its core count or SMT setting, so the file also applies to the same
   part with other core counts or SMT on. `--hw-keys` picks the keys (for
   example add `smt` or `cores`), and `--hw-set key=value` writes a value
   instead of the machine's: `model=0-15`, `l3_kib=16384|32768`, `cores=64-`,
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
   included. Its `meta.plans` (which the loader ignores) records, per problem
   of the campaign an entry matches or the rules alone give a config, the
   plan the library should log under the file: the layers (`source=table`,
   `rules+table` or `rules`), their config as the call applies it (`cfg=`),
   the plan's fields, and whether that is the built-in plan;
   `tune.py verify` checks them (see Validating a table).

## Machine ceilings, headroom and calibration

The op performance utility (`tools/op_perf_utility`) measures the machine where
the sessions run and gives each problem its bound (see its README). All of it
is optional: a campaign without ceilings tunes as before.

- `tune.py probe --workdir D --probe-bin <probe>` (built with
  `tools/op_perf_utility/probe/build.sh`, kept in `campaign.json`) writes one
  probe job per pod of the campaign (`--pods` names others) and runs it through
  the launcher, with the team and pinning of the backend's timed runs (replay:
  one OpenMP place per CPU of its `KMP_AFFINITY` proclist; benchdnn:
  `OMP_PROC_BIND=close OMP_PLACES=cores`). `--local` runs it on this host (as
  for the synthetic backend), `--dry-run` writes the jobs for `tune.py
  collect`, and `--ceilings F` installs a file measured by hand. A file is
  installed as `<workdir>/ceilings.json` only when its CPU (vendor, family,
  model, caches, threads per L3 domain), the plan's L3 domain and its thread
  count are the campaign's (`--force` installs it anyway and records the
  differences); of several pods the first one's is installed, and the others'
  differences above 10% are noted. Each compute path whose measured
  instructions per cycle differ from the library's price (the schema's
  `uarch.dot`, or 2 where the model does not price) by more than 20% is
  noted too (`int8: model 2, measured 3.50`, kept as the campaign's
  `ceilings.model`): the probe measured the machine, the model is what the
  built-in plan assumes of it. A probe job writes the probe's exit status
  (`probe.rc`) when it ends: `collect` takes a probe once it has (`--force`
  before, as failed).
- With ceilings, `status` gives each problem its bounding resource and the
  built-in plan's and the best plan's efficiency (attainable / measured
  time, the weights cold for the replay backend with `--cold-mb`, warm for
  benchdnn), and their geometric means; `export_report.json` lists them per
  problem with the times, and the tuning file's `meta.op_perf` the
  percentages and bound kinds only. The best plan is the best candidate a
  confirmation held (`confirmed`), else the search's best (`unconfirmed`),
  or the built-in plan when a confirmation timed the search's best and it
  did not hold (`rejected`).
- `search --skip-at-pct P` gives no candidates to a problem whose built-in
  plan reaches P% of its bound, `--stop-at-pct P` no more once its best plan
  (the built-in one while no candidate beats it) does, and `--prioritize`
  splits budget x problems by headroom: the calls at the problem's M (the
  shapes' rows whose M range holds it, as a model trace gives them; all of the
  weight's calls when none does) x its best time x (1 - efficiency), or with
  `--prioritize relative` (1 - efficiency) alone, for shapes whose calls are
  flat. Each problem gets between a quarter and four times `--budget`, at
  most `--batch` a round (a
  session's slots follow the largest batch of a group). A problem not timed
  yet first gets `--scout` candidates (1), which time its built-in plan. All
  three are off by default.
- `tune.py calibrate --workdir D` derives the cost model's hardware inputs
  from the ceilings (`cm_fma_pipes` per weight type where the probe measures
  the path more than 20% away from what the campaign's schema prices it with,
  its model where it prices, else 2 a cycle; every path from a probe before
  1.2, which prints no model; no `cm_fma_lat`, a step floor fitted to timed
  plans rather than a latency;
  `cm_l3_cycles`, `cm_mem_cycles`, `cm_stream_cycles`, and `cm_l2_bytes` when
  `--keys` names it: at the probe's L2 streaming rate it made BF16 plans
  slower on every machine it was tried on) and times every problem whose plan
  they change against the built-in plan (phase `calibrate`, `--repeats`, by
  default the confirmation's). It keeps them when the geometric-mean ratio of
  those problems gains at least `--calib-min-gain` (0.01), none is slower
  beyond max(min_gain, noise) and each has a result. The cost model prices
  BF16, F16, per-channel INT8 and s4 calls: an f32 or per-group s8 plan takes
  none of the inputs, so f32 gets no values and a per-group s8 problem keeps
  its plan. Kept, they are the base of every candidate (the built-in plan
  stays the reference, and the base alone is each problem's first
  candidate), the search tries their halves, doubles and built-in values, and
  the export writes them in every entry and, from M 2, in the rules, for calls
  on the campaign's team (its threads) only, as they were measured on it: a
  tree of them by weight type (on `esz` and `bf16`, from the formats of the
  dump; a calibrated type that another type matches in both is refused), or
  under every leaf of the learned tree (ordered rules: ahead of every rule a
  variant of it for each calibrated type, so the other types and teams keep
  the rule, then one rule per weight type);
  `--no-calibration-rule` keeps them in the entries only (a file `merge`
  takes). `meta.op_perf.calibration` records the A/B, and the values when it
  kept them.
  Calibrate before the search: a campaign with search results is refused.
  The cost model refines the rules' plan only where an alternative is modeled
  `cm_gain_pct` faster, so the inputs change few plans.

What each weight type gets:

| Weights | Bound (compute path) | Ceiling the probe measures | Timed by replay / benchdnn | `calibrate` |
|---|---|---|---|---|
| bf16 | BF16 dot product | VDPBF16PS | yes / yes | yes |
| s8, per channel, source quantized in the call | INT8 VNNI | VPDPBUSD | yes / yes | yes |
| s8, per group | INT8 VNNI (scales per group when the problem gives its `group`) | VPDPBUSD | no / no | no |
| s8 over a u8 or s8 source | INT8 VNNI | VPDPBUSD | no / no | yes |
| s4 | the INT4 tile's loop (unpacking included) | the INT4 tile's loop | yes / no | yes |
| f16 | FP16 FMA with AVX512-FP16 (F16_PH), else the F16 kernels' loop (widening included) | VFMADD231PH where present, else the F16 kernels' loop | yes / no | yes |
| f32 | FP32 FMA | VFMADD231PS | no / no | no |

The bound counts bias, gated outputs, binary post-op tensors and a quantized
source's copy, prices stores at the write bandwidth, and uses the threads a
call can use (one per 16 columns of a row) on the cores they take; the
problems of the epilogue classes other than `store` (and replay's `gated`,
benchdnn's `bias`) are not timed. A type no backend times is bounded and
rated (`report` on benchdnn's output) but not tuned. A per-group s8 problem
may give its group (`group=`, with `scale_bytes=`, 4 by default) for the
bound's scale bytes; the planner reads no s8 group, so the plan, the rule
features and the match stay those of the problem without it.

On the 8-core-domain machine type of `tunings/` (32 threads, replay backend,
the CI shapes in bf16 and s8 at M 4, 16, 128 and 2048):

- the built-in plans reach about 95% of their DRAM bound at M 4 and 90% at
  M 16 (a replay of one shape alone streams at the probe's bandwidth), 85-90%
  of their compute bound at M 2048, and 55-60% at M 128, where the search's
  real gains were;
- `calibrate` kept nothing: the inputs changed six BF16 M = 16 plans and made
  them 2-7% slower, all of it from `cm_l2_bytes` (the probe's L2 streaming
  rate is not the rate of the kernels' slab re-reads);
- `--skip-at-pct 95` skipped 29 of 100 problems and lost no real gain, but
  saved little pod time: the skipped problems shared their processes with
  others; `--prioritize` (every shape's `calls` 1) gave the long M 2048 calls
  the budget and missed M 128 gains, in more rounds (calls per M from a trace,
  or `--prioritize relative`, are for that). On a second machine
  type, `--skip-at-pct 95 --stop-at-pct 97` reached the gain of a search
  without them with 30% fewer candidates. The three stay off by default; 95
  and 97 are the values to start from.

A session times the j-th candidates of a group's problems in one process, so
one problem's candidate can change another's time: on small MoE shapes at
M 4 and 16, gains of up to 2x vanished (and some were 15% slower) when the
candidate was timed with the other problems at their built-in plans, while
the gains at M 128 and 2048 held. `tune.py confirm --alone` times each
winner so: every run of a group times all its problems, as the built-in
runs do, with only the winner's clause set, which takes one process per
winner and repeat. On the second machine type it kept 19 of 32 search
winners (9 of the others were slower than the built-in plan), and 20 of 22
after a search with skip and stop.

## Backends

- **replay**: `replay/replay.cpp` makes the framework plugin's exact
  `matmul_direct` calls (a bf16 source over bf16 weights, or over s8 weights
  or s4 weights in groups along K (`--group`, -1 for one group) whose source
  it quantizes per token; an f16 source over f16 weights into an f16 dst;
  with `--gated silu_and_mul`, the gated calls of all but f16 weights, over a
  [gate | up] weight of N = 2I into an M x I dst, as the `gated` problems),
  one weight type (one s4 group size, one epilogue) a process, with
  `--layers` weight sets per shape
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
# the machine's ceilings first: bounds and efficiency, then the cost model's inputs and a search by headroom
bash tools/op_perf_utility/probe/build.sh <zendnn worktree> <data>/probe
python3 $T probe     --workdir <data>/camp3 --probe-bin <data>/probe
python3 $T calibrate --workdir <data>/camp3
python3 $T search    --workdir <data>/camp3 --strategy guided --budget 32 --batch 8 --prioritize --skip-at-pct 95
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
    --hw-set 'model=2|3' --hw-set l3_kib=16384-32768 --hw-set 'l3_cores=8|16'
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
  library reads the file and gives the plans it names: every problem the file
  names (an entry, or the rules alone) logs the layers, `cfg=` and plan of
  `meta.plans` (a gated call's `pairing=`, `pairing`, and its paired plan,
  `paired`, at the width its tiles pair at), every other problem keeps the
  built-in plan, and every output keeps the built-in plan's bits (the replay
  backend's `--check 1` hashes; the benchdnn backend prints none).
  `verify_report.json` lists each problem.
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

## Comparing two builds

A change to the native BRGEMM that should not change what it computes (a refactor of its layout,
packing, kernels or planner) is checked against the build it starts from on a fixed corpus:

    tools/native_brgemm_tuning/ab/build.sh <worktree of A> /tmp/ab_a <build dir of A>
    tools/native_brgemm_tuning/ab/build.sh <worktree of B> /tmp/ab_b <build dir of B>
    python3 tools/native_brgemm_tuning/ab/ab.py run /tmp/ab_a /tmp/ab_b --out /tmp/ab
    python3 tools/native_brgemm_tuning/ab/ab.py size <build dir of A> <build dir of B> --budget 0.1 --same-kernels

Each binary holds its build's library (the static archive), so a binary built once keeps
standing for its build. Both binaries build from the corpus of the checkout that runs `build.sh`,
also against a tree from before `native_unpack_weight` (whose unpacks are `native_unpack_bf16`
and `native_unpack_f16`). `run` runs both on the machine it runs on (the kernels follow its
instruction sets), under `ZENDNNL_MATMUL_ALGO=10` and `ZENDNNL_API_LOG_LEVEL=4` with the
same thread count, and compares, case by case:

- `calls`: about 2400 calls through `matmul_direct` on every route (bf16, f32, f16, s8 per
  channel and per group, s4 in groups and of one group over K) over M 1 to 13, 64 and 1024, N
  tails, long K, bias, every post-op the native epilogue applies and a chain of two, a bf16
  binary operand, gated activations, alpha and beta, dst types, an s8 or u8 dst requantized by
  a dst scale and zero point, transA, transB, a padded ldb and ldc, u8 (with a zero point), s8
  and quantized sources and their scales, a weight scale of one value, weight cache modes 0, 1
  and 2, a weight buffer that states a capacity larger than the weight (the in-place image of a
  layout larger than the weight) and thread counts. A few calls leave `lowoha_algo` none, so
  that `ZENDNNL_MATMUL_ALGO=10` alone selects the native BRGEMM: one it serves, one it hands
  back to the default route, one whose error it keeps. Each case runs twice (the second call
  reads the layout the first cached or wrote in place); the status of each call, the bytes of
  dst after each call (its rows, their padding up to ldc and 64 bytes after the last row), of
  the weight buffer after both (up to its capacity and 64 nonzero bytes after it) and of a
  quantized source's scales and zero point after each call (which a dynamic quantization
  writes), the `[MATMUL.NATIVE_BRGEMM]` lines and every other log line of the case with its
  module and level (times and pointers masked) must be equal.
- `layout`: the bytes `native_pack_weight` writes for every weight type over K and N tails,
  transB, padded ldb and thread counts (the tiles of a long K, the rows of a transposed weight,
  the element path of s4, the column sums of s8), and the bytes the unpack of a bf16 or f16
  layout writes back, each with the 64 bytes after the buffer.
- `rules`: the native BRGEMM's verdict (`native_layout_check`: served, or the status and reason
  of its refusal) on about 14000 calls over source, weight and dst types, static and dynamic
  quantization, weight scale groups, the K bounds of the int32 sums and ldb, each with
  `lowoha_algo` native_brgemm and with `lowoha_algo` none (`/algo10`, where a refusal may hand
  the call back to the default route: `taken=0`); `--filter` selects verdicts by these names.

It prints one line per mode and the first differences (`--show`), and exits with status 1 when
any differs, 2 when a mode compares nothing (an empty `--modes`, or a `--filter` that leaves
the A run of a mode without a case). Inputs come from generators seeded by the case names, so
`--filter` reruns a few cases with the same data. `size` compares the text of the static
archive (the gate, `--budget` percent), the text of its matmul_native objects and the
functions of the kernel objects (the archive's members, found in the build tree by name, size
and bytes: not the shared library's copies, nor the objects of sources removed or moved since),
and lists the kernel functions of one build only; `--same-kernels` fails when there are any. It
exits with status 2 when a build tree lacks an object of its archive or its kernel objects
define no function. `pack` times `native_pack_weight` on weights of model size and the unpack of
their bf16 and f16 layouts (a weight cache's rebuild of a weight from its in-place image), the
two binaries in turn (`--rounds`), for the time of a packing change: per case B / A, and the
geomean of the packs and of the unpacks; a pack that fails ends the run, and every round of both
binaries has to time the cases of the first round of A (exit status 2 otherwise, or when there
are none). The op-level time of the calls is
`replay/run_replay.sh` with the two builds' replay binaries. `layout` and `pack` exit with
status 2 on a CPU without the native BRGEMM's base instruction sets (`native_base_isa`).

## Tests

`python3 -m unittest discover -s tools/native_brgemm_tuning/tests -t tools`
(host only: no pod, no library). Tests that need numpy are skipped without it.
With `ZENDNNL_NATIVE_REPLAY=<replay binary>` `test_backends.py` also runs that binary on a
small shape (a float weight's call does not depend on `--src`), and with
`ZENDNNL_AB_CORPUS=<ab_corpus binary>` `test_ab.py` runs that binary on a few cases (the
verdicts a filter selects, weight scale groups that do not split K, the fields of a case).
`test_op_perf.py` covers the probe jobs, the machine check, the efficiencies,
the search by headroom and the calibration, on the op performance utility's
synthetic machine. `test_sync.py` checks the tuner against the planner's schema: every key is
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
dump on the build's schema, for the cases and for the N block's cases,
`--nb`: the f16 kernel kind and the plans come from one machine, so it
holds on any), then these tests on the build's dump.
`test_cost_model.py` checks the cost model's mirror and choices against the
library's NativeCostModel cases. `test_ab.py` checks the A/B gate's comparison
on corpus outputs (equal runs pass; a changed byte, line, layout or verdict and
a missing case are named), the size report on scratch build trees and the
prepack timing on stand-in binaries. `test_tunings.py` checks the files of
`tunings/` and `tunings/models/`: each is valid, and the plans `plan.py`
gives its entries on its machine (the hardware block's caches and L3 domain)
are its `meta.plans`, so a planner change that moves an entry's plan fails it.

`test_plan.py` checks `plan.py` against the library's plans:
`fixtures/plan_check.json` holds the SHA-256 of the library's plan lines for
the cases `tests/plan_cases.py` draws (random problems of every weight type,
machines and configs, with pins, tiles of 16 to 128 columns of every type
among them, the rule switches, constants, the cost model's inputs and the
blocks schedule, s4 group sizes and one group over K, widths that are not
whole 32-column blocks, per-group INT8 calls and machines without an L3; a
case without a config plans with the build's defaults, a fraction sets every
rule switch to 0 and another sets cost-model inputs; the fixture records
those counts, the cases of each weight type and the kernel kinds of the
machine its plans came from, which the test plans with). Every field of the
plan is compared, its kernels and the problem's padded width.

When the planner, its keys or its defaults change, refresh both fixtures
from the new build, with the plans of a machine of the ISA of its
`SchemaDump` (the kernel registry follows the ISA: an f16 call takes the
F16_PH kernels only with AVX512-FP16). Take both from a machine with every
kernel kind, so that `NativeConfigSchema.TheTunerFollowsThePlanner` passes on
every machine (it accepts a kind the running CPU lacks). The schema fixture
describes an example machine, not the one the dump was made on: vendor
`ExampleVendor`, L3 domains of 8 cores and 32 MiB, L2 1 MiB, L1d 48 KiB,
AVX512-FP16. The planner's test compares only the field names of its `hw`,
`uarch` and `hw_match` blocks, and the refresh keeps their values
(`tune.py schema --extract <log> --machine-of fixtures/schema_dump.json`). A
field the planner adds needs an example value in the fixture first:

    tools/native_brgemm_tuning/refresh_fixtures.sh <worktree> <SchemaDump gtest log> [build dir]
    # or with the plan dumps made on that machine, the nb dump under ZENDNNL_MATMUL_NATIVE_F16_ACCUM=f32
    DUMP=<dump> NB_DUMP=<nb dump> tools/native_brgemm_tuning/refresh_fixtures.sh <worktree> <SchemaDump gtest log>

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

The N block's cases (`--nb`: every case sets `nb`, half of them the blocks
schedule, a quarter `unit_tiles` too, and a share the keys of an M = 1 call,
most of them at M 1, whose GEMV fields the dump then adds) have their own
fixture, `fixtures/plan_check_nb.json`, which the refresh writes too. Each plan
fixture names the kernel kinds its plans came from, and the test plans with
those; each holds one f16 kernel family. `plan_check.json` holds the plans of
a machine with every kernel kind, its f16 calls on the F16_PH kernels (made
without AVX512-FP16, it would check no F16_PH plan). `plan_check_nb.json`
holds those of the F16 kernels, which serve f16 calls without AVX512-FP16
and, on any machine, under `ZENDNNL_MATMUL_NATIVE_F16_ACCUM=f32` (then the
`SchemaDump` lists no F16_PH); the refresh makes its dump under that setting
and leaves F16_PH out of its kinds:

    python3 $P cases | /tmp/plan_dump > /tmp/dump.txt
    python3 $P compare /tmp/dump.txt --schema <SchemaDump gtest log>
    python3 $P fixture /tmp/dump.txt --schema <SchemaDump gtest log> \
        > tools/native_brgemm_tuning/fixtures/plan_check.json
    # the nb dump and its SchemaDump gtest log under ZENDNNL_MATMUL_NATIVE_F16_ACCUM=f32
    python3 $P cases --nb | ZENDNNL_MATMUL_NATIVE_F16_ACCUM=f32 /tmp/plan_dump > /tmp/nb_dump.txt
    python3 $P compare /tmp/nb_dump.txt --nb --schema <F16 SchemaDump gtest log>
    python3 $P fixture /tmp/nb_dump.txt --nb --schema <F16 SchemaDump gtest log> \
        > tools/native_brgemm_tuning/fixtures/plan_check_nb.json

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
   each case the mirror plans otherwise, field by field (the same with
   `--nb` for the N block's cases). The CI runs both.
2. Make the tuner follow: a key is mirrored in `plan.py` (`BUILTIN_PARAMS`,
   and the rule that reads it) and gets search values in `candidates.py`; a
   rule or a cost-model input is mirrored in `plan.py` / `cost_model.py`; a
   problem field goes into `problems.py` (key, filters, features), the plan
   dump and the cases; a match key or name into `export.py` and
   `problems.py`; a kernel kind into `schema.py`; a plan-line field into
   `plan.LOG_LINE_FIELDS` (and `plan_log_fields` when it is a plan field);
   a `gemv` value, or a change to which calls run on the GEMV kernels, into
   `plan.GEMV_KINDS` and `plan.gemv_kind`; a change to an M = 1 call's
   fields or the rules of its threads and blocks per call into
   `plan.gemv_plan`.
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

Its entries start at M 5: decode sizes up to M 4 keep the built-in plan. A hardware block cannot name
AVX512-FP16: a file made on a machine with it also loads on one of the same caches without it, where an
f16 entry of 64-column tiles is illegal for the F16 kernels and dropped for that call, with a warning.

`tunings/models/<model>/` holds the files of single models, made the same way
from the model's own calls (a campaign per phase, the gated gate_up_proj
again in one of its own) and named by machine type as above. The library
never loads one by itself: a run names it, with the native BRGEMM selected:

    ZENDNNL_MATMUL_ALGO=10 \
    ZENDNNL_MATMUL_NATIVE_TUNING_FILE=tools/native_brgemm_tuning/tunings/models/<model>/l3_domain_8_cores_32mib.json \
        <the model's run>

The entries of both are for 32 threads, at decode M 5-8 and 17-32 and prefill
M 513-2048 and from 2049 (timed at 8, 32, 1024, 3077 and 4096, with cold
weights). Every other call keeps the built-in plan, the gated gate_up_proj
among them.

| Folder | Checkpoint | Shapes (N x K) | Entries |
|---|---|---|---|
| `llama-3.1-8b-instruct-w4a8-g128` | Llama-3.1-8B-Instruct in W4A8: int4 symmetric weights in groups of 128 along K, int8 activations quantized per token in the call; `lm_head` in bf16 | s4: qkv_proj 6144 x 4096, o_proj 4096 x 4096, gate_up_proj 28672 x 4096 (gated), down_proj 4096 x 14336; bf16: lm_head 128256 x 4096 (decode) | 4: o_proj at M 5-8 and 17-32, qkv_proj at M 17-32, down_proj from M 2049 |
| `qwen3.5-9b-w8a8` | `RedHatAI/Qwen3.5-9B-quantized.w8a8`: int8 weights with per-channel scales, int8 activations quantized per token in the call; the linear attention and `lm_head` in bf16 | s8: the full attention's qkv_proj 10240 x 4096 and o_proj 4096 x 4096, gate_up_proj 24576 x 4096 (gated), down_proj 4096 x 12288; bf16: the linear attention's in_proj_qkvz 12288 x 4096, in_proj_ba 64 x 4096 and out_proj 4096 x 4096, lm_head 248320 x 4096 (decode) | 4: in_proj_ba at M 5-8, 17-32, 513-2048 and from 2049 |
