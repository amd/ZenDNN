# Qwen BF16 decode specializations

This directory contains the Qwen-specific BF16 decode layer on top of the
existing group-matmul Algo 3 implementation.

## Ownership

- `bf16_decode_policy.*` owns exact shape checks and the Qwen3.5 W2 sidecar
  schedule.
- `bf16_decode_microkernel.*` owns the separately compiled one-panel,
  fixed-K, and FP32-continuation kernels selected by that policy.

The common scalar-to-group route, Algo 3 planner, `CallContext`, weight
unreorder/pack cache, and generic dispatch remain outside this directory.

## Invariants

- The policy runs only after normal Algo 3 preparation succeeds.
- Unknown shapes return `QwenBf16DecodeRoute::none` without changing
  established kernel selection or scheduling.
- Qwen shape selection and `max_mr` behavior match the pre-move
  implementation; this directory is organization-only.
- No environment variable is required to enable the Qwen decode path.
- The W2 sidecar consumes the existing packed-weight pointer; it does not
  introduce a second weight format or cache.

## Shape families

- Qwen3.5 gated: `M=9..32, K=4096, N=24576, silu_and_mul`
- Qwen3.5 W2: `M=9..32, K=12288, N=4096, act=none`
- Qwen3.5 plain: `M=9..32, K=4096,
  N in {4096, 10240, 12288, 248320}`
- Qwen3.6 plain: `M=1..32`,
  `(K=2048, N in {256, 12288, 248320})` or `(K=4096, N=2048)`
