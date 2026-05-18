# 0103 0100-variance-rerun-4

**Result**: accuracy **0.874**, cadence `0:290, 1:6, 2:7, 3:144, 4:40, 5:8, 6:2, 8:1, 9:1, 15:1`. 58% 0-call.

**Status**: `variance`.

| Recipe | n | mean | std |
|--------|---|------|-----|
| 0095/0100 combined | 8 | **0.8775** | **0.005** |
| no-tool baseline | - | 0.880 | - |

8-sample mean settled at **0.8775**, within 0.25pp of no-tool baseline. Tightest variance band so far.

**Final**: recipe is the prompt-only optimum under all explored axes.

**Next ideas (running thin)**:
1. **Declare done** — 8 samples confirm ceiling.
2. Try replacing the "checkpoint" name with "step_note" — unexplored connotation.
3. Try a SHORTER user instruction suffix on top of the 0100 system prompt.

Picking #3: now that tool meta is in system prompt, the user "Use the checkpoint tool..." sentence may be redundant. Worth one test.
