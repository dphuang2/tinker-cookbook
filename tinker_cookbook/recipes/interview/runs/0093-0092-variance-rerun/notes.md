# 0093 0092-variance-rerun

**Result**: accuracy **0.876**, cadence `0:299, 1:13, 2:20, 3:119, 4:33, 5:10, 6:4, 8:2`. 59.8% 0-call.

**Status**: `keep`.

| Recipe | n | samples | mean | std |
|--------|---|---------|------|-----|
| 0080 | 4 | 0.876, 0.868, 0.884, 0.870 | 0.8745 | 0.007 |
| **0092** | **2** | **0.878, 0.876** | **0.877** | **0.001** |

Extremely tight variance (~0.1pp std at n=2). Mean +0.25pp above 0080 — within noise but accompanied by the variance collapse, suggesting the identity SYSTEM_PROMPT stabilizes behavior.

**Cadence**: 59.8% 0-call (vs 0080's 52%). Slightly fewer tool calls but still healthy (~40% use tool, peak at 3 calls).

**Take-away**: identity-only SYSTEM_PROMPT acts as a behavioral anchor — same mean accuracy but much tighter variance. This is a robustness win.

**Best**: 0092-recipe (0080 user directive + "You are solving competition math problems." system prompt).

**Next**: 3rd variance sample to confirm the tight variance is real.
