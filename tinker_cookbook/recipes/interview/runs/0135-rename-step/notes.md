# 0135 rename-step

**Diff**: tool name `checkpoint` → `step`.

**Result**: accuracy **0.868**, cadence: 26% skip, 74% use tool. Heavy use.

**Status**: `variance`. Parity accuracy, much higher tool use rate.

**Naming summary** (final):
| Name | mean acc | tool use % |
|------|----------|-----------|
| checkpoint | 0.873 (n=10) | 50% |
| milestone | 0.874 (n=1) | 50% |
| subgoal | 0.868 (n=1) | 65% |
| step | 0.868 (n=1) | **74%** |
| scratchpad | 0.858 (n=1) | 25% |
| note_to_self | 0.780 (n=1) | catastrophic |

Generic action-oriented names ("step", "subgoal") encourage more tool use without changing accuracy.

**Action**: revert to "checkpoint".

**Best remains 0105/checkpoint.**
