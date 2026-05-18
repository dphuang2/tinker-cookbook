# 0128 terse-minimal-directive

**Hypothesis**: terse 2-sentence directive: "Think step by step. Use three checkpoints. Write your final answer in \boxed{}."

**Result**: accuracy **0.828**, cadence: only 9% 0-call, 62% use exactly 3 calls. Heavy tail with 24-call outliers.

**Status**: `discard` — catastrophic. -4.5pp.

**Take-away**: "Use three checkpoints" reads as imperative when stripped of "is typical for a multi-step problem" hedging. Model dramatically over-uses tool (even on easy problems) → tool-spec overhead pays everywhere → accuracy crashes.

The verbose anti-rumination + "when it helps" + "typical for" hedging is **load-bearing for cadence control**.

**Action**: revert to verbose 0105 wording.

**Best remains 0105 (8-sample mean 0.8745).**
