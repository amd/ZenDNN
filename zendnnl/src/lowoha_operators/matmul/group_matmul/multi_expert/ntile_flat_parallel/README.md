# `multi_expert/ntile_flat_parallel/`

N-tile flat parallel, the MoE executor `ZENDNNL_GRP_MATMUL_ALGO=4` selects:
one parallel region per MoE layer call (W13 with its gated activation, W2 and
the top-k weighted reduce) over expert-major rows, for routed and grouped
calls alike. `ntile_flat_parallel.hpp` describes what a call runs and what
the executor keeps between calls. Its packs store a weight in blocks of 32
output columns (`nr32::kNr`), which the `nr32` names refer to.

## Layout

- `ntile_flat_parallel.hpp` declares the entry points (`execute`,
  `execute_grouped`, `flush`) and the test hooks. `ntile_flat_parallel.cpp`
  defines `flush` and the hooks; `api/routed.cpp` and `api/grouped.cpp`
  define `execute` and `execute_grouped`. `ntile_flat_parallel_config.hpp`
  and `.cpp` read `ZENDNNL_GRP_MATMUL_NTILE_FLAT_PARALLEL_CONFIG`: the
  settings a call takes by phase, format and API.
- `api/` holds the two APIs. `routed` keeps each routed layer's packs and
  plans and routes a call's tokens to expert-major rows. `grouped` checks a
  grouped call, keeps its call sites and runs it. `api_common` holds what
  both decide alike: the u8 sources, the plan budgets, the epilogues, the
  bias types and the decline and refusal logs.
- `engine/` is the parallel region both APIs run. `engine.hpp` runs a call's
  units and names no format. `schedule` cuts rows into runs and units and
  orders their queue, and `scratch` is the per-thread scratch. `source` says
  what each projection's kernels read (`source_spec`), holds the rows a call
  makes for them (`source_rows`) and makes, copies and views those rows; the
  fronts and the engine handle source rows only through it.
- `pack/` holds the packs. `nr32_pack` builds one weight's packs through its
  format's packer and reads them back. `store` keeps one pack per expert
  weight until a flush, `scales` reads the weight scale forms, and
  `call_sites` keeps the grouped call sites. The store's and the call sites'
  state is private to their files, which export operations on it.
- `tile/nr32_tile` plans a weight's tiles (which kernels run which rows and
  columns) and runs a tile's steps over its source rows (`tile_source`).
- `kernel/nr32_ukernel` is a kernel call (`ukernel_args`) and its selection
  through the format rows. `kernel/nr32_epilogue.hpp` is the epilogue every
  format's kernels share: bias loads, the gated activations and the step
  stores, typed by the output element of the row's `dst` (`out_elem`).
- `formats/` holds one directory per weight format: `<format>.hpp` its row,
  `<format>_pack.cpp` its packer and `<format>_kernels.cpp` its kernels.
  `format.hpp` says what a row holds, and `formats.hpp` lists the rows, finds
  a call's row and checks each row at compile time.

## Formats

A format is one row, a `format_desc`. A call finds its row by its key: the
weight type, the type of the source rows its kernels read, and its scales'
encoding (`format_for`). Formats may therefore share a weight type and
differ in their sources or scales: W4A8 and W4A16 are both s4 weights, and
MXFP4 and NVFP4 are both `f4_e2m1` weights with differently encoded scales.
Past that lookup everything reads the row: the packs (`weight_desc::fmt`,
the store's keys), the plans and tiles, the config's `dtype=` filter (a
row's `name`), and both APIs' checks. A row's index (`format_index`) is
found from its key, so keys and tables that hold it agree across modules.

A row says, as `format.hpp` groups its fields:

- how its packs store it: element bits, the words of a block row
  (`word_bytes`, `k_per_word`, `enc`), its scales' encoding (`scales`: none
  or real numbers today, with e8m0 and e4m3 as the MX and NV formats' next
  values), whether it always groups its scales, and whether its packs keep
  weight sums (`sums`);
- what a call over it passes and stores: `src`, `dst`, and whether a call
  may give s8 rows (`s8_rows`);
- its kernels' shape: the instructions they need (`isa`, a set), their row
  limits, and whether they take u8 sources;
- its functions: `pack`, which writes into a `pack_out`, `narrow`, `wide`
  and `built`.

`format_holds` checks that a row holds together, and `formats_hold` that the
rows' keys and names differ. Both run at compile time.

## Dependency rules

The layers below are listed lowest first. A file may include anything from
its own layer and the layers before it, and nothing from a later layer.

1. `ntile_flat_parallel.hpp`, `ntile_flat_parallel_config.hpp` and
   `kernel/nr32_ukernel.hpp` include nothing from this folder;
   `kernel/nr32_epilogue.hpp` includes only `kernel/nr32_ukernel.hpp`.
2. `formats/`: `format.hpp` includes `kernel/nr32_ukernel.hpp`; a format's
   files include only its own header, and its kernels also
   `kernel/nr32_epilogue.hpp`. `formats.hpp` includes every format's header.
3. `pack/nr32_pack`, `tile/`, `kernel/nr32_ukernel.cpp` and
   `ntile_flat_parallel_config.cpp` read rows through `formats/formats.hpp`.
4. `pack/store`, `pack/scales`, `pack/call_sites` and `engine/`.
5. `api/`.
6. `ntile_flat_parallel.cpp`.

Outside `formats/`, code names element types (s8 sources, bf16 and f16
outputs, scale and bias types) but no format: it reads a call's format from
its row. Comments elsewhere name today's formats only as examples.

## Adding a weight format

1. `formats/<format>/<format>.hpp`: in `nr32::<format>`, declare `pack`,
   `narrow`, `wide` (only with wide kernels) and `built` (only when a build
   may lack its kernels), and define the row `kFormat`; `format.hpp`
   documents every field.
2. `formats/<format>/<format>_pack.cpp`: `pack` writes one expert's blocks
   into `pack_out::blocks` in the row's words and, for a format with `sums`,
   adds each column's sums to `pack_out::sums`. The driver, `prepack` in
   `pack/nr32_pack`, allocates them, fills the packs' f32 weight scales and
   handles failures.
3. `formats/<format>/<format>_kernels.cpp`: the kernels for every epilogue
   and 1 to `max_mr` rows (1 to `max_mr_wide` for the wide split kernels),
   storing through `store_step` and `store_wide_step` in the row's output
   type (`<format>::kFormat.dst`), and `narrow` and `wide`, which pick them.
4. `formats/formats.hpp`: include the header and add `<format>::kFormat` to
   `kFormats`. The build fails until the row holds together and its key and
   name are its own.
5. `zendnnl/src/lowoha_operators/matmul/group_matmul/CMakeLists.txt`: the
   two sources, in `lowoha_group_matmul_sources`.
6. The APIs' admission, when the format's calls take another form: the
   routed gate (`ntile_flat_parallel_projection_refusal` in
   `routed_moe/routed_moe_dispatch.cpp`) takes a scaled format's sources as
   bf16 rows it quantizes, so a scaled format over 16-bit sources (W4A16)
   adds its case there. The grouped checks (`check_grouped` in
   `api/grouped.cpp`) admit a call by its row's source type.
7. Tests, in `gtests/group_matmul/multi_expert/ntile_flat_parallel/`:
   `test_format.cpp` runs its tests on every row. `test_nr32.cpp` checks
   packs and tiles against a reference over the types its tests list, so add
   the format to those lists, with its weights in `weight_set::build`, rows
   of a new source type in `make_rows`, and its reference in `dot_ref`.
   `test_source.cpp` covers the source rows.

## What else a format may need

Each of these is only for a format that needs it.

- **A scale encoding**, such as MXFP4's e8m0 scale per 32 elements or
  NVFP4's e4m3 scale per 16 with an f32 scale per tensor: a `scale_enc`
  value, and `scale_enc_of`, which maps a caller's scale type to it.
  `pack/scales` reads a scale form (`scale_form`, `pack_group_size`,
  `scale_at`); `add_scales` in `pack/store.cpp` widens per-tensor and f16
  scales to f32; `prepack`, `with_scales` and `fill_scales` in
  `pack/nr32_pack.cpp` take f32 and bf16 scales; and the routed gate's
  `ntile_flat_parallel_weight_groups` checks the form. The packs keep f32
  scales today. A format that keeps its own (e8m0 bytes, a per-tensor scale,
  or Q4_K's scales and minimums inside the weight) adds them to `pack_out`
  and `packed_expert`, which its packer writes and its kernels read.
- **A word encoding**, such as e2m1 nibbles: an `encode` value, its words
  rule in `format_holds`, and `unpack` in `pack/nr32_pack.cpp`, which the
  round-trip tests read packs back through.
- **Instructions**: an `isa_need` bit and its check in `isa_supported`
  (`kernel/nr32_ukernel.cpp`), and `built` when a compiler may build none of
  the kernels.
- **Another source**, such as bf16 rows against scaled weights (W4A16) or
  s8 rows with per-block sums: `runs_source` in `formats.hpp`;
  `source_spec`, `source_rows` and `nr32::tile_source`, and the functions
  that make and copy rows, in `engine/source`; the engine's W2 requantize;
  `run_tile` and `ukernel_args`; the fronts' scratch sizing (`run_call` in
  `api/routed.cpp`, `run_grouped` in `api/grouped.cpp`); and the admission
  above.
- **Another output**, such as FP32: `runs_output` in `formats.hpp`;
  `out_elem` and `store_out` in `kernel/nr32_epilogue.hpp`; Op1's rows in
  the scratch (`inter` and `stage`, 16-bit, in `engine/scratch.hpp`); the
  reduce (`reduce_f16` in `engine/engine.hpp`, bf16 or f16); W2's requantize
  of Op1's rows, which reads bf16; and the routed `moe_call`'s 16-bit `src`
  and `dst`.
- **A new data type**: `common/data_types.hpp`.
