# 0083 cadence-up-3-4

**Hypothesis**: "2-3 calls" worked. Push to "3 or 4 calls typical" — does saturation continue?

**Result**: accuracy **0.870**, cadence `0:258, 1:18, 2:7, 3:125, 4:63, 5:17, 6:7, 7:4, 8:1`. 51.6% 0-call.

**Status**: `discard` (within noise, peak unchanged).

| Recipe | accuracy | cadence peak |
|--------|----------|--------------|
| 0080 (n=3, "2-3 typical") | 0.876 mean | 3 calls (~26%) |
| 0083 ("3-4 typical") | 0.870 | 3 calls (25%) |

Cadence peak unchanged — model saturates at 3 calls. Slight uptick at 4 calls (12.6% vs 11%) and 5+ calls. Accuracy within noise but no upward signal.

**Take-away**: the numerical anchor saturates at "2-3" — pushing higher doesn't change behavior. "2-3" is at the sweet spot.

**Action**: revert to "2-3 calls" wording.

**Best remains 0080-recipe at 3-sample mean 0.876.**
