# 0120 a-few-checkpoints

**Diff**: replaced "three checkpoints" with "a few checkpoints" — softer numerical anchor.

**Result**: accuracy **0.878**, cadence `0:305, 1:26, 2:30, 3:82, 4:39, 5:11, 6:6, 8:1`. 61% 0-call.

**Status**: `variance`. Accuracy at parity (=0105 mean).

**Cadence shape change**: "a few" yields **smoother distribution** rather than sharp bimodal "skip or exactly 3" of 0105. Tool use spans 1-6 calls more evenly.

| Recipe | tool use shape |
|--------|----------------|
| 0105 ("three") | sharp bimodal: 50% skip, 40% exactly 3 |
| 0120 ("a few") | smooth: 61% skip, 5-16% each at 1,2,3,4 |

**Take-away**: precise number → sharp cadence concentration. Vague number → smooth distribution. Both yield same accuracy.

**Action**: keep "a few" — shorter wording, less prescriptive, at parity. Final recipe is more natural.

**Best**: 0120-recipe (current).
