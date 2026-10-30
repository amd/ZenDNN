---
name: running-tests
description: >-
  Run the ZenDNN gtest suite. Use when asked to run, filter, or debug
  GoogleTest. BenchDNN belongs to the benchmark sub-skill.
---

# Running tests

Run gtests from the repository documentation, using only commands it documents.
Paths are relative to the ZenDNN checkout.

1. Follow `gtests/Readme.md`.
2. A gtest build is described in `docs/zendnnl_build.md`. Use that when the
   suite has not been built. Confirm with the user before replacing an
   existing build directory.
3. Where documents disagree, the build files under `cmake/` and the source
   are authoritative. Say which one you followed.
4. Don't invent flags, filters, or commands the documents don't contain.

## Failure policy

If a command fails, stop and report the step, the exact command, and the first
real error line. Wait for the user.
