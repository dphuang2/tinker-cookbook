# 0104 shorter-user-suffix

**Hypothesis**: now that system prompt mentions the tool, remove the redundant "Use the checkpoint tool when it helps..." from user message. Keep only the numerical cadence anchor.

**Result**: accuracy **0.870**, cadence `0:227, 1:6, 2:41, 3:167, 4:47, 5:6, 6:4, 8:1, 12:1`. 45.4% 0-call.

**Status**: `discard` (within noise of 0100 mean 0.877, but on low side — conservative revert).

Cadence shifted slightly to more tool use (45% 0-call vs 58%). Accuracy at low end of variance band.

**Take-away**: removing the "Use the checkpoint tool when it helps..." sentence shifts cadence upward (more tool use) without accuracy gain. The hedge "when it helps" was actually moderating tool use.

**Action**: revert to the longer 0100 user suffix.

**Best remains 0100-recipe at 8-sample mean 0.8775.**
