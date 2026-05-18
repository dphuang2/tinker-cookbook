# 0091 add-second-example

**Hypothesis**: adding a 2nd concrete example (combinatorics/stars-and-bars) to PROGRESS_TOOL_SPEC param description on top of the u-substitution example might broaden the model's template coverage.

**Result**: accuracy **0.874**, cadence `0:213, 1:23, 2:24, 3:159, 4:50, 5:14, 6:7, 7:3, 9:3, 12:2, 24:2`. 42.6% 0-call.

**Status**: `discard` (within noise of 0080 mean; no improvement).

Same accuracy expectation as 0080. The 2nd example is decorative — model already extracts the pattern from one example. More tokens for zero benefit.

**Action**: revert to single u-sub example.

**Best remains 0080-recipe at 4-sample mean 0.8745.**
