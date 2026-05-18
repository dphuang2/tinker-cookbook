# 0069 add-verify-sentence

**Hypothesis**: adding NEW content ("Before writing your final answer, take a moment to verify it.") rather than removing might lift accuracy by encouraging a verification step. Orthogonal additive change.

**Diff**: inserted one verification sentence into USER_INSTRUCTION_SUFFIX.

**Result**: accuracy **0.860**, cadence `0:382, 1:75, 2:19, 3:12, 4:2, 5:3, 6:1, 7:1, 8:5`. 76.4% 0-call.

**Status**: `discard`. -1pp vs 0062, within single-sample noise but slightly worse. Cadence shape similar to 0062. The verify directive didn't help and likely added overhead.

**Take-away**: even adding sensible additional content regresses. This is the 5th consecutive prompt-engineering attempt around 0062 that fails to improve. Strong evidence 0062 is at the local optimum.

**Action**: revert USER_INSTRUCTION_SUFFIX.

**Best remains 0062 at 0.870.**

**Pattern after 9 post-0062 experiments (0063-0069)**: every single variation regresses. The optimum is sharp on every axis tested (CoT prefix, tool spec verbosity, tool suppressor wording, anti-rumination, max_tokens, verify directive).

**Decision**: time to validate 0062 with a variance re-run. With single-sample std ~0.6-1.7pp, the 0.870 figure needs corroboration. If 0062 reruns at 0.86-0.88, then we're at the ceiling; if it reruns at 0.85 or 0.89, we have new info.

**Action for 0070**: variance re-run of 0062's exact config. Reset eval and run again. To make it a committable distinct experiment, add a comment marker in sft_train.py.
