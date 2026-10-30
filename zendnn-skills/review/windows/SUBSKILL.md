---
name: review-windows
description: >-
  Windows-only checks for a ZenDNN review. Opened by the review orchestrator
  after the common checks when the change is for Windows.
---

# Windows

Platform-only review checks on Windows. The checks every review runs are in
[common](../common/SUBSKILL.md), and the
[orchestrator](../orchestrator/SUBSKILL.md) opens this file after them.

No Windows-only review checks are written here yet. Do not invent any. When
the change edits C++ or CMake, apply the MSVC checklist in
[op-development-optimization/windows](../../op-development-optimization/windows/SUBSKILL.md).
