# 0085 imperative-cadence

**Hypothesis**: replace conditional "use when it helps" with imperative "Call the checkpoint tool to organize your reasoning" — might tighten cadence shape further.

**Result**: accuracy **0.860**, cadence `0:82, 1:8, 2:33, 3:265, 4:60, 5:9, 6:20, ...` heavy tail to 28. 16.4% 0-call.

**Status**: `discard`. -1.6pp vs 0080 mean (0.876).

**Interpretation**: imperative wording pushed cadence too high (only 16% 0-call vs 0080's 51%). Model now uses the tool on easy problems too, paying tool-spec cost everywhere and hurting accuracy. The 0080 recipe's 50% 0-call rate (skip the easy problems) was load-bearing.

**Take-away**: there's an accuracy/cadence sweet spot near 50% 0-call. Either direction (lower or higher tool use) regresses accuracy.

**Action**: revert to 0080 wording.

**Best remains 0080-recipe at 3-sample mean 0.876.**

**Next ideas**:
1. **4th variance sample on 0080** — tighten confidence band.
2. **Test the recipe on a different DeepMath slice** — generalization check. But that violates "FIXED eval slice".
3. **Add specific cadence trigger language** — e.g., "Call the checkpoint when switching approach" — instead of numerical anchor.

Picking #1: more variance signal on confirmed best.
