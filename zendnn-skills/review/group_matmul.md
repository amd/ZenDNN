# Group matmul review checks

Checks for a change to group matmul and fused MoE, under
`zendnnl/src/lowoha_operators/matmul/group_matmul/`. Apply them after
[common](common/SUBSKILL.md) and the platform file.

## Functional correctness

- Empty groups, zero-M experts, sparse experts, leading inactive experts,
  single expert, many experts, and capacity limits are handled.
- Mixed per-expert M/N/K values do not get collapsed into fictitious shapes.
- Active expert count is not confused with padded slot count.
- Fused activation changes output width and pointer arithmetic consistently.

## Concurrency and OpenMP

- Reduced teams do not skip experts or activation work.

## Planner and integer arithmetic

- Per-expert constraints stay per-expert where inputs can differ.
- Inactive slots do not influence active-work planning.

## Tests and validation quality

- Heterogeneous and inactive-expert cases are included where the API permits
  them.
