# 0060 0059-rerun (variance confirmation)

**Result**: acc **0.832**. Confirms 0059's 0.846 was not a lucky sample.

**Two samples of 0059 config (max_tokens=12288)**:
| Run | acc |
|-----|-----|
| 0059 | 0.846 |
| 0060 | 0.832 |
| **mean** | **0.839** |

Compared to 0024 config (max_tokens=8192) 6-sample mean of 0.797:
**+4.2pp gain from raising max_tokens 8192 → 12288.** Solidly outside noise.

**Cadence**: same shape as 0024 (~80-81% 0-call, healthy non-degenerate).

**New best recipe = 0024's prompt + max_tokens=12288**. Mean ~0.84, 0 training records, 18-20% tool-call cadence.

This is the most important finding in the entire loop — the eval cap was the actual bottleneck, not the recipe. Many of our "noise" results would shift higher with more thinking budget.
