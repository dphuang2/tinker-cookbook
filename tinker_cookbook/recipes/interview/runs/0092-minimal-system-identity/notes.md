# 0092 minimal-system-identity

**Hypothesis**: 0072 tested SYSTEM_PROMPT with rubric (weakened cadence). Try pure identity-only: "You are solving competition math problems." No rubric, no role conflict with user-message tool directive.

**Result**: accuracy **0.878**, cadence `0:305, 1:11, 2:11, 3:109, 4:42, 5:13, 6:3, 7:1, 8:1, 12:1, 15:1, 21:1, 24:1`. 61% 0-call.

**Status**: `keep` (within noise of 0080 mean 0.8745; identity-only prompt is a clean addition).

| Run | accuracy | 0-call % |
|-----|----------|----------|
| 0080 (n=4) | 0.8745 mean | 52% |
| 0092 (n=1) | 0.878 | 61% |

Single-sample 0.878 is within 0.5pp of 0080 mean — could be variance. Cadence shifted slightly toward less tool use (61% 0-call vs 52%), but still healthy.

**Take-away**: identity-only SYSTEM_PROMPT is at parity with empty. Won't hurt; possibly marginally helps frame the task.

**Action**: keep as part of current recipe. Variance rerun next would confirm.

**Best**: 0080-recipe + identity SYSTEM_PROMPT (current).
