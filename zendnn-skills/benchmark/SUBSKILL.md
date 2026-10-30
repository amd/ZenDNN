---
name: benchmark
description: >-
  Benchmark ZenDNN operators with BenchDNN. Use when asked to measure
  operator performance or run BenchDNN. gtests belong to the running-tests
  sub-skill.
---

# Benchmark

Run BenchDNN from the repository documentation, using only commands it documents.
Paths are relative to the ZenDNN checkout.

1. Follow `benchdnn/README.md` and, for the operator under test,
   `benchdnn/doc/<operator>.md`.
2. A BenchDNN build is described in `docs/zendnnl_build.md`. Use that when
   BenchDNN has not been built. Confirm with the user before replacing an
   existing build directory.
3. Where documents disagree, the build files under `cmake/` and the source
   are authoritative. Say which one you followed.
4. Don't invent flags or commands the documents don't contain.

## Failure policy

If a command fails, stop and report the step, the exact command, and the first
real error line. Wait for the user.
