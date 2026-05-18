# 0116 0105-variance-rerun-5

**Result**: accuracy **0.858**, cadence `0:248, 1:6, 2:1, 3:211, 4:18, 5:1, 6:8, 7:2, 9:3, 12:1, 18:1`. 49.6% 0-call.

**Status**: `variance`.

| Recipe | n | mean |
|--------|---|------|
| **0105 (sys + 3 calls)** | **5** | 0.890, 0.878, 0.866, 0.880, 0.858 → **0.8744** |
| 0100 (sys + 2-3 calls, 8 incl. 0095) | 8 | 0.8775 |

Recipe statistically saturated at **~0.876 ± 0.012**, equivalent across 0095/0100/0105 wordings, ~0.5pp below no-tool baseline.

**Recipe is fully explored.** 120+ experiments confirm prompt-only ceiling.
