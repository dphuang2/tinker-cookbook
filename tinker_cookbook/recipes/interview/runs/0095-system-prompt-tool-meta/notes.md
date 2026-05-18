# 0095 system-prompt-tool-meta

**Hypothesis**: expand SYSTEM_PROMPT to introduce the tool: "You are solving competition math problems. You have access to a checkpoint tool for tracking your reasoning progress on hard multi-step problems." Pre-frames the tool before the user message.

**Result**: accuracy **0.880**, cadence `0:286, 1:9, 2:9, 3:151, 4:36, 5:4, 6:3, 7:1, 8:1`. 57.2% 0-call, peak at 3 calls (30.2%).

**Status**: `keep` (at no-tool baseline, +0.8pp over 0092 mean — needs corroboration).

| Run | accuracy | 0-call % |
|-----|----------|----------|
| 0092 (n=3) | 0.872 mean | 60% |
| 0095 (n=1) | 0.880 | 57.2% |
| no-tool baseline | 0.880 | 100% |

Single sample exactly at no-tool baseline. Whether real or variance, the recipe is now in the no-tool-baseline neighborhood with healthy tool use cadence.

**Take-away**: introducing the tool in SYSTEM_PROMPT pre-frames the model's task structure. Slightly tighter cadence shape (peak more concentrated at 3 calls).

**Action**: keep, variance rerun.
