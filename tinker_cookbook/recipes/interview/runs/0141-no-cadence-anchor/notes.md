# 0141 no-cadence-anchor

**Diff**: removed "-- three checkpoints is typical for a multi-step problem" suffix entirely.

**Result**: accuracy **0.868**, cadence: 60% skip, peak at 1 call (26%), shallow tail.

**Status**: `variance` (parity acc).

**Take-away**: numerical anchor matters for cadence SHAPE (1-call peak without anchor; 3-call peak with "three"), but not for accuracy. Both yield ~0.87.

**Action**: revert to "three checkpoints" for bimodal cadence (more visibly structured tool use).
