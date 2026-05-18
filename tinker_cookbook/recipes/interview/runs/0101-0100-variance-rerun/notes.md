# 0101 0100-variance-rerun

**Result**: accuracy **0.870**, cadence `0:289, 1:15, 2:12, 3:136, 4:36, 5:8, 6:2, 9:2`. 57.8% 0-call.

**Status**: `variance`.

| Recipe | n | samples | mean |
|--------|---|---------|------|
| 0095 | 4 | 0.880, 0.874, 0.876, 0.882 | 0.878 |
| **0100** | **2** | **0.882, 0.870** | **0.876** |

Within noise. Shortened system prompt is at parity (mean 0.876 vs 0.878).

**Take-away**: confirming 0095/0100 family is the final recipe space. Ceiling robustly at ~0.876-0.878. Cleaner system prompt (0100) preferred.

**Current best**: 0100-recipe at 2-sample mean 0.876, healthy cadence, 0 training records.
