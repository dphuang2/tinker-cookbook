# 0107 0105-variance-rerun-3

**Result**: accuracy **0.866**, cadence `0:263, 1:4, 3:207, 4:15, 5:5, 6:4, 24:2`. 52.6% 0-call.

**Status**: `variance`.

| Recipe | n | samples | mean |
|--------|---|---------|------|
| 0100 (combined) | 8 | various | 0.8775 |
| **0105** | **3** | **0.890, 0.878, 0.866** | **0.878** |

0105 mean exactly matches 0100. The 0.890 from 0105 was a high-variance pull. **Recipe is at the same accuracy ceiling** (~0.878) regardless of "two or three" vs "three" wording.

**Distinguishing feature**: 0105 cadence is more sharply bimodal (peak at 3 calls = 41% of all problems, 53% 0-call; ~5% mid-range). 0100 has more spread (peak at 3 calls = 28%, 60% 0-call, ~12% other).

**Take-away**: single-number anchor cleans up cadence shape (sharper bimodality) but doesn't move accuracy.

**Action**: keep 0105 wording — bimodal cadence is arguably cleaner.

**Best (locked)**: 0105-recipe at 3-sample mean 0.878, sharply bimodal cadence (47% use exactly 3 calls).
