# 0122 rename-subgoal

**Diff**: tool name `checkpoint` → `subgoal`. Keep "three subgoals is typical" anchor.

**Result**: accuracy **0.868**, cadence: 35.6% 0-call, 48% use exactly 3, heavy tail.

**Status**: `variance`. Within noise of 0105 mean. Pushed cadence higher.

**Naming summary**:
| Name | accuracy | cadence note |
|------|----------|--------------|
| checkpoint | 0.873 (n=6) | 50% skip |
| milestone | 0.874 (n=1) | 50% skip |
| subgoal | 0.868 (n=1) | 35% skip (more tool use) |
| scratchpad (0073) | 0.858 | ~75% skip |
| note_to_self (0029) | 0.780 | degenerate |

"subgoal" name encourages MORE tool use than checkpoint (math/planning connotation makes model treat it as default-on for math problems). Accuracy at parity.

**Action**: revert to "checkpoint" for canonical naming.

**Best remains 0105 / checkpoint recipe.**
