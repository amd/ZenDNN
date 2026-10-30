---
name: review
description: >-
  Review a ZenDNN operator change in the working tree, or a ZenDNN pull
  request from its URL, on Linux or Windows. Hands every review to the
  orchestrator.
---

# Review

Open [orchestrator](orchestrator/SUBSKILL.md) and follow it. It opens the
common checks, the platform file, and the operator file when one exists, and
for a pull-request URL it runs the GitHub review.

## Layout

- [orchestrator](orchestrator/SUBSKILL.md): runs every review and holds the
  pull-request workflow.
- [common](common/SUBSKILL.md): checks every review runs.
- [linux](linux/SUBSKILL.md) and [windows](windows/SUBSKILL.md): platform-only
  checks.
- `<operator>.md`: one operator's own checks, named after its directory under
  `zendnnl/src/lowoha_operators/`. Today: [group_matmul](group_matmul.md).
