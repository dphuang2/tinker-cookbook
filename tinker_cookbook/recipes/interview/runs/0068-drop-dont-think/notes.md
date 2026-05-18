# 0068 drop-dont-think-too-long

**Hypothesis**: the "Don't think for too long unnecessarily, especially when you have a reasonable degree of confidence" sentence was added when max_tokens=8192. With max_tokens=24576 unlocked (0062), the model may now under-use its thinking budget on hard problems. Dropping the sentence might recover accuracy on the harder tail.

**Diff**: removed second sentence of USER_INSTRUCTION_SUFFIX.

**Result**: accuracy **0.856**, cadence `0:352, 1:72, 2:45, 3:7, 4:8, 5:6, 6:2, 7:4, 8:4`. 70% 0-call.

| Run | accuracy | description |
|-----|----------|-------------|
| 0062 | **0.870** | full directive incl. "don't think too long" |
| 0068 | 0.856 | without "don't think too long" |

**Status**: `discard`. -1.4pp vs 0062. The "don't think too long" guidance actively helps.

**Interpretation**: counter to expectation, removing the cap let the model over-think (or wander) on problems where it had already converged, hurting accuracy. The directive functions as anti-rumination, not just a budget cap.

**Take-away**: every sentence in 0062's USER_INSTRUCTION_SUFFIX is load-bearing. Both directive components (anti-rumination + sparing-tool-use) contribute independently.

**Action**: revert USER_INSTRUCTION_SUFFIX to 0062's wording.

**Best remains 0062 at 0.870.**

**Pattern after 8 post-0062 experiments**: every variation around 0062 regresses. The optimum is sharp on multiple axes (CoT prefix, tool spec verbosity, tool suppressor, anti-rumination).

**Next ideas**:
1. **Variance re-run on 0062** — critical to validate single-sample 0.870. Std ~0.6-1.7pp.
2. **Add a verification sentence** — "Before writing your final answer, verify it briefly." Tests whether adding new content (rather than removing) helps.
3. **System prompt with reasoning rubric** — push rubric to SYSTEM_PROMPT instead of tool spec.

Picking #2: adds NEW content rather than tweaking existing wording — orthogonal experiment.
