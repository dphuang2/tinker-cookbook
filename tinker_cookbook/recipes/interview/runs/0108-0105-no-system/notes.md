# 0108 0105-no-system

**Hypothesis**: 0105 ("three calls typical") + 0100 system prompt. Test without the system prompt — is system prompt actually contributing?

**Result**: accuracy **0.888**, cadence `0:238, 1:5, 2:2, 3:231, 4:14, 5:3, 6:5, 7:1, 8:1`. 47.6% 0-call.

**Status**: `keep` (single-sample 0.888 above no-tool baseline; needs corroboration).

| Run | accuracy | cadence shape |
|-----|----------|---------------|
| 0100 (n=8) | 0.8775 | peak ~28% at 3, 60% skip |
| 0105 (n=3) | 0.878 mean | peak 41% at 3, 53% skip |
| **0108 (n=1)** | **0.888** | peak **46% at 3**, **48% skip** |

Cadence is even more sharply bimodal. The recipe is converging on "either think alone or use exactly 3 checkpoints" pattern.

**Take-away**: removing system prompt doesn't hurt and may slightly help. The 0105 "three calls" wording is the dominant signal.

**Action**: keep, variance check.
