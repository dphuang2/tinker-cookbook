# 0111 0105-variance-rerun-4

**Result**: accuracy **0.880** (at no-tool baseline), cadence `0:252, 1:3, 3:220, 4:13, 5:2, 6:4, 7:1, 8:1, 9:2, 12:2`. 50.4% 0-call.

**Status**: `variance`.

| Recipe | n | samples | mean |
|--------|---|---------|------|
| 0100 (sys + 2-3) | 8 | various | 0.8775 |
| **0105 (sys + 3)** | **4** | **0.890, 0.878, 0.866, 0.880** | **0.8785** |

0105 4-sample mean **0.8785**, essentially identical to 0100. Tightly bimodal cadence (44% use exactly 3 calls, 50% skip).

**Final**: 0105-recipe is the locked best — sharply bimodal cadence at parity accuracy. Statistically at no-tool baseline 0.880.

**Recipe is fully saturated.** 12+ samples across two equivalent variants confirm ~0.878 ± 0.7pp ceiling.
