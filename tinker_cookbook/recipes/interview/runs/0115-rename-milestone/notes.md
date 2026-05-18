# 0115 rename-milestone

**Diff**: renamed tool `checkpoint` → `milestone` (all references).

**Result**: accuracy **0.874**, cadence `0:249, 1:1, 3:226, 4:13, 5:1, 6:8, 7:1, 12:1`. 49.8% 0-call.

**Status**: `variance`. Within noise of 0105 mean (0.8785). "Milestone" is at parity with "checkpoint".

**Naming summary**:
| Name | accuracy |
|------|----------|
| checkpoint | 0.8785 (n=4) |
| milestone | 0.874 (n=1) |
| scratchpad (0073) | 0.858 |
| note_to_self (0029, w/ different baseline) | 0.780 (catastrophic) |

"Checkpoint" and "milestone" are equivalent — both technical-progress-tracking framings. "Scratchpad" (working-area) and "note_to_self" (internal-monolog) both hurt — different connotations matter.

**Action**: revert to canonical "checkpoint" name for consistency.

**Best remains 0105 / "checkpoint" recipe at 4-sample mean 0.8785.**
