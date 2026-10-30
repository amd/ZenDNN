---
name: op-development-optimization
description: >-
  Develop or optimize a ZenDNN operator. Use when adding operator
  functionality or changing an operator for performance. Open this file for
  the shared procedure, the windows or linux sub-skill for the platform, and
  the operator file in this folder when one exists.
---

# Operator development and optimization

Shared procedure for developing or optimizing an operator. Platform-only
tricks live in [windows](windows/SUBSKILL.md) and [linux](linux/SUBSKILL.md).
The Windows file also holds the MSVC portability checklist, used on every
C++ or CMake edit. An operator file in this folder, named after the operator,
holds that operator's own steps. Open the platform sub-skill and the operator
file together with this file.

## Procedure

The shared steps are not written yet. Until they are:

1. Tell the user this procedure is not written yet.
2. Work from the repository documentation for that operator, using only
   commands it documents. Paths are relative to the ZenDNN checkout:
   - architecture and operators: `docs/zendnnl_architecture.md`, `docs/operator/`
   - builds: `docs/zendnnl_build.md`
3. Where documents disagree, the source under `zendnnl/src/` is authoritative.
   Say which one you followed.
4. Don't invent steps the documentation doesn't contain.

## Failure policy

If a command fails, stop and report the step, the exact command, and the first
real error line. Wait for the user.
