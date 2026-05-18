# 0110 0108-variance-rerun-3

**Result**: accuracy **0.858**, cadence `0:254, 1:7, 2:3, 3:217, 4:14, 6:4, 24:1`. 50.8% 0-call.

**Status**: `variance` → `discard` (no-sys-prompt variant is slightly worse).

| Recipe | n | samples | mean |
|--------|---|---------|------|
| 0100 (sys + 2-3) | 8 | various | 0.8775 |
| 0105 (sys + 3) | 3 | 0.890, 0.878, 0.866 | 0.878 |
| **0108 (nosys + 3)** | **3** | **0.888, 0.860, 0.858** | **0.869** |

The 0108 mean is ~0.9pp below 0100/0105. Removing the system prompt mildly hurts under the "three calls" anchor. Recipe robustness benefits from system prompt presence.

**Action**: revert to 0105 (system prompt + "three calls is typical"). Best 3-sample mean and cleaner bimodal cadence.

**Best**: 0105-recipe (3-sample mean 0.878, system prompt + single-number anchor).
