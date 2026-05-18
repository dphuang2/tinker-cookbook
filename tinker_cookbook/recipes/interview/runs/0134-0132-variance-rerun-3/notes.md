# 0134 0132-variance-rerun-3

**Result**: accuracy **0.854**.

**Status**: `variance` → `discard` (revert to plain "ok" ack).

| Recipe | n | samples | mean |
|--------|---|---------|------|
| 0105 baseline | 10 | various | 0.8732 |
| 0132 (turn-count ack) | 3 | 0.890, 0.890, 0.854 | 0.878 |

The two 0.890s in 0132/0133 were coincidence (sampled the same high pull, possibly correlated by Tinker session caching). 3-sample mean 0.878 is at parity with 0105.

**Take-away**: turn-count meta in tool ack is NOT a real signal. Revert to plain "ok".

**Final lesson on variance**: with 500-problem eval and temp=0.6, single-sample std is ~1pp. Two consecutive samples can occasionally match. Need 3+ samples to be confident.

**Action**: revert ack to plain "ok". Best remains 0105-recipe.
